/**
 * Stage-2 post-decode value checks.
 *
 * The centre of gravity here is the face-index range check. It must be
 * TWO-SIDED and run on the SOURCE-TYPED values, before the integer→u32
 * coercion, because each side of that cast hides its own wrap-around. Both
 * hazards get their own test with the exact value that triggers them, and both
 * are asserted to be rejected rather than merely "handled" — an out-of-range
 * index traps the WASM kernel (`panic = "abort"`, so it takes down the whole
 * module) and silently corrupts the TypeScript one.
 */

import { describe, it, expect } from 'vitest';
import { validateFaceIndices, validateMaterializedLength } from '../../../../data/mesh/validate';
import type { FaceIndexSource } from '../../../../data/mesh/validate';
import { LoaderError } from '../../../../data/scene-loader/nodes/load-leaf-error-dispatch';

const PATH = '/surface';

function expectReject(fn: () => unknown, pattern: RegExp): void {
  let thrown: unknown;
  try {
    fn();
  } catch (e) {
    thrown = e;
  }
  expect(thrown).toBeInstanceOf(LoaderError);
  expect((thrown as LoaderError).kind).toBe('Validation');
  expect((thrown as LoaderError).path).toBe(PATH);
  expect((thrown as Error).message).toMatch(pattern);
}

describe('validateMaterializedLength', () => {
  it('accepts an exact match', () => {
    expect(() => validateMaterializedLength(PATH, 'vertices', 12, 12)).not.toThrow();
  });

  it.each([
    ['short', 11],
    ['long', 13],
  ])('rejects a %s array', (_label, actual) => {
    // Stage 1 vets only the DECLARED shape. A store that declares correctly but
    // materializes short resurrects exactly the undersized-attribute
    // drawElements over-read that Stage 1's shape checks exist to close.
    expectReject(
      () => validateMaterializedLength(PATH, 'vertices', actual, 12),
      /vertices materialized .* but its declared shape implies/
    );
  });
});

describe('validateFaceIndices — the happy path, in every dtype a store may use', () => {
  const cases: Array<[string, FaceIndexSource]> = [
    ['Uint8Array', new Uint8Array([0, 1, 2, 1, 2, 3])],
    ['Int8Array', new Int8Array([0, 1, 2, 1, 2, 3])],
    ['Uint16Array', new Uint16Array([0, 1, 2, 1, 2, 3])],
    ['Int16Array', new Int16Array([0, 1, 2, 1, 2, 3])],
    ['Uint32Array', new Uint32Array([0, 1, 2, 1, 2, 3])],
    ['Int32Array', new Int32Array([0, 1, 2, 1, 2, 3])],
    ['BigUint64Array', new BigUint64Array([0n, 1n, 2n, 1n, 2n, 3n])],
    ['BigInt64Array', new BigInt64Array([0n, 1n, 2n, 1n, 2n, 3n])],
  ];

  it.each(cases)('widens a %s to Uint32Array, value-preserving', (_label, faces) => {
    const out = validateFaceIndices(PATH, faces, 4, 6);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([0, 1, 2, 1, 2, 3]);
  });

  it('accepts index V-1 and rejects V — the range is half-open', () => {
    expect(Array.from(validateFaceIndices(PATH, new Uint32Array([3, 3, 3]), 4, 3))).toEqual([
      3, 3, 3,
    ]);
    expectReject(
      () => validateFaceIndices(PATH, new Uint32Array([0, 1, 4]), 4, 3),
      /face index 4 at position 2 is not an integer in \[0, 4\)/
    );
  });
});

describe('validateFaceIndices — the two wrap-around hazards', () => {
  it('rejects a SIGNED store’s -1, which a one-sided check would pass', () => {
    // -1 satisfies `< V`. Coerced to u32 it becomes 0xffffffff, which then traps
    // the kernel. This is why the check is two-sided.
    const signed = new Int32Array([0, 1, -1]);
    expectReject(
      () => validateFaceIndices(PATH, signed, 4, 3),
      /face index -1 at position 2 is not an integer in \[0, 4\)/
    );
    // Demonstrate the hazard is real rather than theoretical: the coercion the
    // check runs BEFORE would indeed land far out of range.
    expect(new Uint32Array([signed[2]])[0]).toBe(0xffffffff);
  });

  it('rejects a 64-bit store’s 2^32 + 1, which a post-cast check would pass', () => {
    // 2^32 + 1 wraps to 1 under a u32 cast — inside [0, V) — so a check run
    // after the coercion sees a valid index and the topology is silently
    // rewritten. Only a pre-cast source-value check catches it.
    const wide = new BigUint64Array([0n, 1n, BigInt(2 ** 32) + 1n]);
    expectReject(
      () => validateFaceIndices(PATH, wide, 4, 3),
      /face index 4294967297 at position 2 is outside \[0, 4\)/
    );
    expect(Number(BigInt.asUintN(32, wide[2]))).toBe(1);
  });

  it('range-checks a signed 64-bit source at the top of its range', () => {
    // Covers the BigInt64Array branch with a value far above any legal index.
    //
    // This deliberately does NOT claim to prove "BigInt comparison is required". That
    // claim was here and it is not constructible: `nVertices <= 2^27`, u64 -> double is
    // exact below 2^53 and non-decreasing above, so `Number(v) >= 2^27` for every
    // `v >= 2^27` and an out-of-range value stays out of range whichever way it is
    // compared. Verified by mutation — replacing the BigInt comparison with a
    // post-`Number()` one leaves this whole file green, which is exactly what a
    // test asserting an unachievable premise looks like.
    //
    // The narrowing that IS lossy is the u32 store, and the test above pins that with
    // 2^32 + 1 wrapping to 1.
    const huge = new BigInt64Array([0n, 1n, 2n ** 53n + 1n]);
    expectReject(() => validateFaceIndices(PATH, huge, 4, 3), /is outside \[0, 4\)/);
  });

  it('rejects a negative 64-bit index too', () => {
    expectReject(() => validateFaceIndices(PATH, new BigInt64Array([0n, 1n, -1n]), 4, 3), /-1/);
  });
});

describe('validateFaceIndices — non-integral values', () => {
  it('rejects NaN, which no relational guard would catch', () => {
    // `NaN < 0` and `NaN >= V` are BOTH false, so a purely relational check
    // passes NaN through, and it coerces to 0 — a silently rewired triangle.
    expectReject(
      () => validateFaceIndices(PATH, new Float64Array([0, 1, NaN]), 4, 3),
      /face index NaN at position 2/
    );
  });

  it('rejects +Infinity and -Infinity', () => {
    for (const v of [Infinity, -Infinity]) {
      expectReject(
        () => validateFaceIndices(PATH, new Float64Array([0, 1, v]), 4, 3),
        /face index/
      );
    }
  });

  it('rejects a fractional index', () => {
    // Stage 1 rejects a float faces DTYPE, but a value check that trusted that
    // would be relying on another gate; this keeps the function sound alone.
    expectReject(
      () => validateFaceIndices(PATH, new Float32Array([0, 1, 2.5]), 4, 3),
      /face index 2\.5 at position 2/
    );
  });
});

describe('validateFaceIndices — length', () => {
  it('rejects a short faces array before checking any value', () => {
    expectReject(
      () => validateFaceIndices(PATH, new Uint32Array([0, 1]), 4, 3),
      /faces materialized 2 values but its declared shape implies 3/
    );
  });

  it('checks only the first expectedLength values of a longer source', () => {
    // A longer array is itself a length rejection, so this documents that the
    // length gate runs first rather than the loop silently truncating.
    expectReject(
      () => validateFaceIndices(PATH, new Uint32Array([0, 1, 2, 99]), 4, 3),
      /faces materialized 4 values/
    );
  });

  it('reports the FIRST offending position, so the message points somewhere useful', () => {
    expectReject(
      () => validateFaceIndices(PATH, new Uint32Array([0, 9, 9]), 4, 3),
      /at position 1\b/
    );
  });
});
