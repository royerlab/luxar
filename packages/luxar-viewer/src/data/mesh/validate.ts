/**
 * Stage 2 of the mesh loader's admission gate: **post-decode value checks**.
 *
 * These are the checks that genuinely need the materialized arrays, so they run
 * after fetch + decode — but only ever on data Stage 1
 * (`preflight.ts`) already admitted, so the fetch they gate is bounded
 * before it happens.
 *
 * Two things are checked, and each closes a hole the other cannot see.
 *
 * ## Materialized lengths
 *
 * Stage 1 vets the *declared* `.zarray` shape. A store that declares correctly
 * but materializes short — a raw or mis-sized chunk, a non-compliant decoder —
 * would otherwise resurrect exactly the undersized-attribute `drawElements`
 * over-read that Stage 1's shape cross-checks exist to close.
 *
 * ## Face indices, checked on the SOURCE-TYPED values
 *
 * Every index must land in `[0, V)`, and the check must be **two-sided** and run
 * **before** the integer→u32 coercion, because each side of that cast hides its
 * own wrap-around:
 *
 * - an externally produced **signed** store's `-1` passes a one-sided `< V`
 *   pre-cast check and then wraps to `0xffffffff`;
 * - a **64-bit** store's `2^32 + 1` survives a check run only *after* the cast —
 *   it wraps to `1`, lands inside `[0, V)`, and silently rewrites topology
 *   instead of trapping.
 *
 * The two-sided source-value check rejects both. And because Stage 1 admits only
 * `V <= 2^27`, every index it passes is preserved bit-for-bit by the u32 cast —
 * so the values checked here are exactly the values the kernels receive.
 *
 * Getting this wrong is not a cosmetic bug. An out-of-range index **panics** the
 * Rust kernel — the crate is `panic = "abort"`, so the trap escapes as an
 * opaque, uncatchable `RuntimeError: unreachable` that takes down the whole WASM
 * module rather than one node — and silently corrupts the TypeScript backend,
 * whose out-of-bounds reads yield `undefined`.
 *
 * ## What is deliberately NOT checked
 *
 * Stage 2 does not finite-scan the float arrays (`vertices`, `normals`,
 * `colors`, `scalars`). A non-finite value can neither trap a kernel nor
 * over-read a buffer, and its blast radius is already per-node without a gate: a
 * `NaN`/`±Inf` coordinate on a hidden dimension hides the vertex (the #806
 * rule the cull kernels enforce), a non-finite *displayed* coordinate corrupts
 * at most that node's rasterization and bounding sphere (which the depth-sort
 * coordinator already refuses to sort by), non-finite colours are clamped by the
 * shared shader sanitizers, and a non-finite stored normal degrades only that
 * node's shading. No sibling loader finite-scans its decoded positions either;
 * mesh matches that policy rather than inventing a stricter one.
 *
 * @module data/mesh/validate
 */

import { LoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';

/**
 * Any typed array a zarr store can materialize a face-index array as.
 *
 * `BigInt64Array`/`BigUint64Array` are in the list because they are real: the
 * writer's own INDEX encoder emits `uint64` when the max index needs it, and an
 * external store may use `int64` freely. They read as `bigint`, not `number`,
 * which is why {@link validateFaceIndices} handles them on a separate path
 * rather than relying on numeric coercion.
 */
export type FaceIndexSource =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array;

/** Fail the node with a Stage-2 rejection (see `rejectMesh` in the preflight). */
function reject(path: string, message: string): never {
  throw new LoaderError('Validation', path, new Error(message));
}

/**
 * Assert a decoded array materialized to the length Stage 1 admitted.
 *
 * @param name - Array name, for the message.
 * @param actual - The materialized length.
 * @param expected - The length the declared shape implies.
 */
export function validateMaterializedLength(
  path: string,
  name: string,
  actual: number,
  expected: number
): void {
  if (actual !== expected) {
    reject(
      path,
      `${name} materialized ${actual.toLocaleString()} values but its declared shape ` +
        `implies ${expected.toLocaleString()}. A short array would make the indexed ` +
        'draw read past the buffer.'
    );
  }
}

/**
 * Range-check face indices on their source-typed values and widen to
 * `Uint32Array`.
 *
 * The widening and the check are one function on purpose: they are only sound
 * together. Splitting them invites a caller to coerce first and check after,
 * which is precisely the `2^32 + 1 → 1` hole described in the module docs.
 *
 * @param faces - The materialized source array, in whatever dtype the store used.
 * @param nVertices - Vertex count; valid indices are `[0, nVertices)`.
 * @param expectedLength - `3 × n_faces`, from the shape Stage 1 admitted.
 */
export function validateFaceIndices(
  path: string,
  faces: FaceIndexSource,
  nVertices: number,
  expectedLength: number
): Uint32Array {
  validateMaterializedLength(path, 'faces', faces.length, expectedLength);

  const out = new Uint32Array(expectedLength);

  if (faces instanceof BigInt64Array || faces instanceof BigUint64Array) {
    // Compared as BigInt throughout: converting a 64-bit index to `number`
    // first would round anything above 2^53 and could land a hostile value
    // back inside range before it is ever tested.
    const limit = BigInt(nVertices);
    for (let i = 0; i < expectedLength; i++) {
      const v = faces[i];
      if (v < 0n || v >= limit) {
        reject(
          path,
          `face index ${v.toString()} at position ${i} is outside [0, ${nVertices}). ` +
            'An out-of-range index traps the WASM cull kernel and silently corrupts ' +
            'the TypeScript one.'
        );
      }
      out[i] = Number(v);
    }
    return out;
  }

  for (let i = 0; i < expectedLength; i++) {
    const v = faces[i];
    // `!Number.isInteger` also catches NaN and ±Infinity, which no comparison
    // would: `NaN < 0` and `NaN >= nVertices` are both false, so a NaN index
    // would slip through a purely relational guard and coerce to 0.
    if (!Number.isInteger(v) || v < 0 || v >= nVertices) {
      reject(
        path,
        `face index ${String(v)} at position ${i} is not an integer in ` +
          `[0, ${nVertices}). An out-of-range index traps the WASM cull kernel and ` +
          'silently corrupts the TypeScript one.'
      );
    }
    out[i] = v;
  }
  return out;
}
