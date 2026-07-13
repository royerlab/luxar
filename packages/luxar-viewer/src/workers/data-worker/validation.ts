/**
 * Worker boundary validation helpers.
 *
 * Every WASM-bound entry point in the worker validates its typed-array
 * inputs before passing them across the JS→WASM boundary: the compiled
 * Rust code reads from caller-supplied buffers using lengths the JS
 * side promised, and an off-by-one on the JS side becomes a wild read
 * (or `memory.fill` panic) inside WASM. These helpers reject malformed
 * payloads at the JS boundary with a clear, namespaced error message
 * instead of letting WASM corrupt or crash.
 *
 * All five helpers + `MAX_WASM_DIMS` are pure functions with no
 * worker-local state, which is why they extract cleanly.
 */

import { MAX_SUPPORTED_DIMS } from '../../config/constants';

/**
 * Maximum number of dimensions the compiled WASM kernels support (fixed-size
 * `[f32; 16]` arrays). Inputs above this are NOT rejected — `pickBackend` in
 * `state.ts` routes them to the uncapped TypeScript reference instead, so >16D
 * datasets are supported (slower but correct). These validators therefore only
 * enforce `ndim >= 1` (integer) plus array-length consistency, which hold for
 * any backend. Aliased to `MAX_SUPPORTED_DIMS` so the worker stays
 * self-documenting while sharing the canonical constant.
 */
export const MAX_WASM_DIMS = MAX_SUPPORTED_DIMS;

/**
 * Validate the typed-array inputs that flow into WASM visibility /
 * projection calls. The compiled Rust code reads from `positions`,
 * `slicePosition`, `tolerance`, etc. assuming caller-supplied lengths
 * match `numItems × ndim` and `ndim`; if a malformed payload reaches
 * WASM, the read goes off the end of the buffer (undefined behavior in
 * WASM, may return garbage or trip a memory.fill panic). This helper
 * rejects bad inputs at the JS boundary with a clear error message
 * instead of letting the worker silently corrupt or crash.
 *
 * `radii` is optional: present for points, absent for lines/gsplats.
 */
export function validateNDArrays(
  fnName: string,
  positions: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number,
  numItems: number,
  positionsPerItem: number = ndim,
  radii?: Float32Array
): void {
  if (!Number.isInteger(ndim) || ndim < 1) {
    throw new Error(`${fnName}: ndim=${ndim} must be a positive integer`);
  }
  if (!Number.isInteger(numItems) || numItems < 0) {
    throw new Error(`${fnName}: numItems=${numItems} must be a non-negative integer`);
  }
  const expectedPositions = numItems * positionsPerItem;
  if (positions.length < expectedPositions) {
    throw new Error(
      `${fnName}: positions array too short (got ${positions.length}, expected ≥ ${expectedPositions})`
    );
  }
  if (slicePosition.length < ndim) {
    throw new Error(
      `${fnName}: slicePosition too short (got ${slicePosition.length}, expected ≥ ${ndim})`
    );
  }
  if (tolerance.length < ndim) {
    throw new Error(`${fnName}: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`);
  }
  if (radii && radii.length < numItems) {
    throw new Error(`${fnName}: radii too short (got ${radii.length}, expected ≥ ${numItems})`);
  }
}

/**
 * Validate projection inputs that share the WASM 3D-extraction
 * preconditions: positions ≥ numItems × ndim, displayDims length ≤ 3,
 * each displayDim < ndim, slicePosition ≥ ndim. Reused by every
 * `project*To3D` entry point before the first WASM call.
 */
export function validateProjectionInputs(
  fnName: string,
  // Only `.length` is read here; accept any numeric array so the
  // main-thread Points projection (positions may be Uint8/Uint16/Float16
  // before WASM coercion) can share this guard with the worker dispatchers.
  positions: ArrayLike<number>,
  displayDims: readonly number[] | Uint32Array,
  slicePosition: readonly number[] | Float32Array,
  ndim: number,
  numItems: number,
  positionsPerItem: number = ndim,
  // The 3D-extraction kernel (extract_3d_positions) does not read
  // slicePosition, so the main-thread Points path skips this check on its
  // extraction-only call (a pure-3D view legitimately carries a short/empty
  // slicePosition) and validates length separately in its effective-radius
  // branch, where the kernel actually consumes it. Worker dispatchers keep
  // the default (slicePosition always required).
  requireSlicePosition: boolean = true
): void {
  if (!Number.isInteger(ndim) || ndim < 1) {
    throw new Error(`${fnName}: ndim=${ndim} must be a positive integer`);
  }
  if (!Number.isInteger(numItems) || numItems < 0) {
    throw new Error(`${fnName}: numItems=${numItems} must be a non-negative integer`);
  }
  const expectedPositions = numItems * positionsPerItem;
  if (positions.length < expectedPositions) {
    throw new Error(
      `${fnName}: positions array too short (got ${positions.length}, expected ≥ ${expectedPositions})`
    );
  }
  if (displayDims.length === 0 || displayDims.length > 3) {
    throw new Error(`${fnName}: displayDims must have 1–3 entries (got ${displayDims.length})`);
  }
  for (let i = 0; i < displayDims.length; i++) {
    const d = displayDims[i];
    if (!Number.isInteger(d) || d < 0 || d >= ndim) {
      throw new Error(`${fnName}: displayDims[${i}]=${d} out of range [0, ${ndim - 1}]`);
    }
  }
  if (requireSlicePosition && slicePosition.length < ndim) {
    throw new Error(
      `${fnName}: slicePosition too short (got ${slicePosition.length}, expected ≥ ${ndim})`
    );
  }
}

/**
 * Validate decode-entry-point inputs. The decoders share a common
 * preconditions: `data.length` and (for LUT modes) `lut` length must
 * be ≥ implied minimums; numeric scalars (bounds, maxLog, k) must be
 * finite. Rejecting at the worker boundary keeps WASM from reading
 * past the end of caller-supplied buffers.
 */
export function validateDecodeArgs(
  fnName: string,
  data: ArrayLike<number>,
  opts: {
    /** Expected element count, or undefined to skip the strict check. */
    minLength?: number;
    /** A finite-number bound to validate (e.g. min/max/maxLog). */
    finiteScalar?: { name: string; value: number };
    /** A pair of bounds `[min, max]` requiring max > min. */
    boundsPair?: { name: string; bounds: readonly [number, number] };
    /** A non-empty LUT array. */
    lut?: { name: string; values: ArrayLike<number>; minLength?: number };
    /** A positive integer count. */
    positiveInt?: { name: string; value: number };
  } = {}
): void {
  if (opts.minLength !== undefined && data.length < opts.minLength) {
    throw new Error(
      `${fnName}: data array too short (got ${data.length}, expected ≥ ${opts.minLength})`
    );
  }
  if (opts.finiteScalar) {
    const { name, value } = opts.finiteScalar;
    if (!Number.isFinite(value)) {
      throw new Error(`${fnName}: ${name}=${value} must be a finite number`);
    }
  }
  if (opts.boundsPair) {
    const { name, bounds } = opts.boundsPair;
    if (!Number.isFinite(bounds[0]) || !Number.isFinite(bounds[1])) {
      throw new Error(`${fnName}: ${name}=[${bounds[0]}, ${bounds[1]}] must be finite`);
    }
    if (bounds[1] <= bounds[0]) {
      throw new Error(
        `${fnName}: ${name} max (${bounds[1]}) must be greater than min (${bounds[0]})`
      );
    }
  }
  if (opts.lut) {
    const { name, values, minLength } = opts.lut;
    if (values.length === 0) {
      throw new Error(`${fnName}: ${name} must be non-empty`);
    }
    if (minLength !== undefined && values.length < minLength) {
      throw new Error(
        `${fnName}: ${name} too short (got ${values.length}, expected ≥ ${minLength})`
      );
    }
  }
  if (opts.positiveInt) {
    const { name, value } = opts.positiveInt;
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${fnName}: ${name}=${value} must be a positive integer`);
    }
  }
}

/**
 * Validate that every vertex index referenced by `segments` falls inside
 * the position buffer, and that any optional per-vertex attributes cover
 * up through the maximum referenced vertex.
 *
 * Lines uniquely have an indirection (segments index into vertex data)
 * that {@link validateProjectionInputs} can't catch: positions might
 * have ≥ ndim entries but a malformed segments[i] could still reach
 * past the end. WASM clip / interpolate primitives index by segments[i]
 * with no bounds check, so this validation runs before every entry
 * point that touches segments. Cost is O(numSegments) — small relative
 * to the WASM call that follows.
 */
export function validateLineSegmentReferences(
  fnName: string,
  segments: Uint32Array,
  numSegments: number,
  positions: ArrayLike<number> | Float32Array | Uint16Array,
  ndim: number,
  opts: {
    widths?: ArrayLike<number>;
    colors?: ArrayLike<number>;
    sharpness?: ArrayLike<number>;
    scalars?: ArrayLike<number>;
  } = {}
): void {
  if (!Number.isInteger(numSegments) || numSegments < 0) {
    throw new Error(`${fnName}: numSegments=${numSegments} must be a non-negative integer`);
  }
  if (segments.length < numSegments * 2) {
    throw new Error(
      `${fnName}: segments too short (got ${segments.length}, expected ≥ ${numSegments * 2})`
    );
  }
  if (numSegments === 0) return;

  let maxVertex = 0;
  for (let i = 0; i < numSegments * 2; i++) {
    const v = segments[i];
    if (v > maxVertex) maxVertex = v;
  }

  const minVertices = maxVertex + 1;
  if (positions.length < minVertices * ndim) {
    throw new Error(
      `${fnName}: positions too short for max segment vertex ${maxVertex} ` +
        `(got ${positions.length}, expected ≥ ${minVertices * ndim})`
    );
  }
  if (opts.widths && opts.widths.length < minVertices) {
    throw new Error(
      `${fnName}: widths too short for max segment vertex ${maxVertex} ` +
        `(got ${opts.widths.length}, expected ≥ ${minVertices})`
    );
  }
  // Per-vertex colors (RGB triplet) — 3 entries per referenced vertex.
  if (opts.colors && opts.colors.length < minVertices * 3) {
    throw new Error(
      `${fnName}: colors too short for max segment vertex ${maxVertex} ` +
        `(got ${opts.colors.length}, expected ≥ ${minVertices * 3})`
    );
  }
  if (opts.sharpness && opts.sharpness.length < minVertices) {
    throw new Error(
      `${fnName}: sharpness too short for max segment vertex ${maxVertex} ` +
        `(got ${opts.sharpness.length}, expected ≥ ${minVertices})`
    );
  }
  // Per-vertex scalar (1 entry per referenced vertex). Pre-fix, the WASM
  // `interpolate_scalars_batch` would read past the end on a short input,
  // panicking inside WASM (or returning garbage in the TS fallback).
  // [workers OOS] — three-geometry symmetry: Points + GSplats validate
  // their own scalars; Lines must too.
  if (opts.scalars && opts.scalars.length < minVertices) {
    throw new Error(
      `${fnName}: scalars too short for max segment vertex ${maxVertex} ` +
        `(got ${opts.scalars.length}, expected ≥ ${minVertices})`
    );
  }
}
