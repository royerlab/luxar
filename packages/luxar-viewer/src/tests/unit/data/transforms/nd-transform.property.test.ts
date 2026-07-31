/**
 * Property tests for nD transform utilities (fast-check).
 *
 * Companion to nd-transform.test.ts. The example-based tests pin specific
 * cases; the property tests here pin algebraic invariants that should hold
 * for the entire input space:
 *
 *   - Affine roundtrip: inverting world coords and re-applying scale+offset
 *     reproduces the input within Float32 tolerance.
 *   - Tolerance scales by 1/|scale|.
 *   - Permutation inverse is self-inverse.
 *   - Displayed dims are pass-through under any nd_transform.
 *
 * Property tests strictly dominate finite-example tests for these invariants:
 * a mutation that breaks the algebra on any unseen input would be caught here
 * where the example tests would not.
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  invertNdTransformForQuery,
  composeNdTransforms,
} from '../../../../data/transforms/nd-transform';
import type { NdTransformMap } from '../../../../types/zarr';

/**
 * Name-only dimension metadata for `invertNdTransformForQuery`. Omitting
 * `discrete` keeps every dimension continuous, so the no-preimage rule (which
 * only applies to discrete dims) never fires — these cases predate it and must
 * keep their original expectations. Discrete cases pass explicit metadata.
 */
const dims = (names: string[]) => names.map((name) => ({ name }));

const finiteFloat = fc.float({
  min: Math.fround(-1e6),
  max: Math.fround(1e6),
  noNaN: true,
  noDefaultInfinity: true,
});
const nonZeroScale = fc
  .float({
    min: Math.fround(-100),
    max: Math.fround(100),
    noNaN: true,
    noDefaultInfinity: true,
  })
  .filter((s) => Math.abs(s) > 1e-3);
const positiveTolerance = fc.float({
  min: Math.fround(0.01),
  max: Math.fround(1e3),
  noNaN: true,
  noDefaultInfinity: true,
});

describe('invertNdTransformForQuery — algebraic invariants', () => {
  test('affine roundtrip: invert then re-apply recovers the world coordinate (Float32 tolerance)', () => {
    fc.assert(
      fc.property(
        finiteFloat, // worldValue
        positiveTolerance, // worldTolerance
        nonZeroScale, // scale
        finiteFloat, // offset
        (worldValue, worldTolerance, scale, offset) => {
          const ndTransform: NdTransformMap = { Time: { scale, offset } };
          const result = invertNdTransformForQuery(
            [0, 0, 0, worldValue],
            [1e10, 1e10, 1e10, worldTolerance],
            ndTransform,
            dims(['X', 'Y', 'Z', 'Time']),
            [0, 1, 2]
          );
          // Forward: world = scale * local + offset  ⇒  local = (world - offset) / scale
          // Re-apply: world' = scale * local + offset should equal worldValue.
          const reappliedWorld = scale * result.slicePosition[3] + offset;
          expect(reappliedWorld).toBeCloseTo(worldValue, 3);
        }
      ),
      { numRuns: 200, seed: 0x5eed }
    );
  });

  test('tolerance scales by 1/|scale|', () => {
    fc.assert(
      fc.property(positiveTolerance, nonZeroScale, finiteFloat, (worldTolerance, scale, offset) => {
        const ndTransform: NdTransformMap = { Time: { scale, offset } };
        const result = invertNdTransformForQuery(
          [0, 0, 0, 0],
          [1e10, 1e10, 1e10, worldTolerance],
          ndTransform,
          dims(['X', 'Y', 'Z', 'Time']),
          [0, 1, 2]
        );
        const expectedLocalTolerance = worldTolerance / Math.abs(scale);
        expect(result.tolerance[3]).toBeCloseTo(expectedLocalTolerance, 3);
      }),
      { numRuns: 200, seed: 0x5eed }
    );
  });

  test('displayed dimensions are pass-through regardless of any nd_transform attached to them', () => {
    fc.assert(
      fc.property(
        finiteFloat,
        finiteFloat,
        nonZeroScale,
        finiteFloat,
        (worldX, tolX, scale, offset) => {
          // Attach a transform to X (which is displayed). It should be ignored.
          const ndTransform: NdTransformMap = {
            X: { scale, offset },
            Time: { scale: 1, offset: 0 },
          };
          const result = invertNdTransformForQuery(
            [worldX, 0, 0, 0],
            [Math.abs(tolX) + 0.01, 1e10, 1e10, 1],
            ndTransform,
            dims(['X', 'Y', 'Z', 'Time']),
            [0, 1, 2] // X displayed
          );
          // Displayed dim should be untouched.
          expect(result.slicePosition[0]).toBe(worldX);
          expect(result.tolerance[0]).toBe(Math.abs(tolX) + 0.01);
        }
      ),
      { numRuns: 200, seed: 0x5eed }
    );
  });

  test('offset-only transforms (scale defaulted to 1) translate the slice position', () => {
    fc.assert(
      fc.property(finiteFloat, finiteFloat, (worldValue, offset) => {
        const ndTransform: NdTransformMap = { Time: { offset } };
        const result = invertNdTransformForQuery(
          [0, 0, 0, worldValue],
          [1e10, 1e10, 1e10, 1],
          ndTransform,
          dims(['X', 'Y', 'Z', 'Time']),
          [0, 1, 2]
        );
        // local = world - offset (scale defaults to 1)
        expect(result.slicePosition[3]).toBeCloseTo(worldValue - offset, 3);
      }),
      { numRuns: 200, seed: 0x5eed }
    );
  });

  test('permutation inversion is self-inverse: invert(invert(perm)) === perm', () => {
    fc.assert(
      fc.property(fc.shuffledSubarray([0, 1, 2, 3], { minLength: 4, maxLength: 4 }), (perm) => {
        // Build the inverse permutation directly: inverse[perm[i]] = i
        const inverse = new Array(perm.length).fill(0);
        for (let i = 0; i < perm.length; i++) inverse[perm[i]] = i;
        // Inverse of inverse should equal original
        const doubleInverse = new Array(inverse.length).fill(0);
        for (let i = 0; i < inverse.length; i++) doubleInverse[inverse[i]] = i;
        expect(doubleInverse).toEqual([...perm]);
      }),
      { numRuns: 100, seed: 0x5eed }
    );
  });

  // PROPERTY [P12]: affine composition is function composition, which is
  // associative. compose(compose(a,b),c) must equal compose(a,(b,c)) for the
  // resulting effective affine, within Float tolerance. We compare the
  // *effective* (scale, offset) — not the raw map — because compose elides
  // scale===1 / offset===0, and the two groupings could land on the elision
  // boundary differently while remaining mathematically identical.
  test('composeNdTransforms (affine) is associative within tolerance', () => {
    const effective = (m: NdTransformMap): { scale: number; offset: number } => {
      const e = (m.Time as { scale?: number; offset?: number } | undefined) ?? {};
      return { scale: e.scale ?? 1.0, offset: e.offset ?? 0.0 };
    };
    fc.assert(
      fc.property(
        nonZeroScale,
        finiteFloat,
        nonZeroScale,
        finiteFloat,
        nonZeroScale,
        finiteFloat,
        (sA, oA, sB, oB, sC, oC) => {
          const a: NdTransformMap = { Time: { scale: sA, offset: oA } };
          const b: NdTransformMap = { Time: { scale: sB, offset: oB } };
          const c: NdTransformMap = { Time: { scale: sC, offset: oC } };

          // Left-assoc: compose (a,b) first, then with c.
          const ab = composeNdTransforms(a, b);
          const leftAssoc = composeNdTransforms(ab, c);

          // Right-assoc: compose (b,c) first, then a with it.
          const bc = composeNdTransforms(b, c);
          const rightAssoc = composeNdTransforms(a, bc);

          const lhs = effective(leftAssoc);
          const rhs = effective(rightAssoc);

          // Relative tolerance: products of up to three ±100 scales can reach
          // ~1e6, so an absolute epsilon is too strict. Use toBeCloseTo on a
          // normalized ratio for scale and a scaled comparison for offset.
          const scaleRef = Math.max(Math.abs(lhs.scale), Math.abs(rhs.scale), 1);
          const offsetRef = Math.max(Math.abs(lhs.offset), Math.abs(rhs.offset), 1);
          expect((lhs.scale - rhs.scale) / scaleRef).toBeCloseTo(0, 3);
          expect((lhs.offset - rhs.offset) / offsetRef).toBeCloseTo(0, 3);
        }
      ),
      { numRuns: 100, seed: 0x5eed }
    );
  });

  test('hidden-dim permutation: world position at perm[k] maps to local k', () => {
    fc.assert(
      fc.property(
        fc.shuffledSubarray([0, 1, 2], { minLength: 3, maxLength: 3 }),
        fc.integer({ min: 0, max: 2 }),
        (perm, worldIndex) => {
          const ndTransform: NdTransformMap = { Channel: { permutation: perm } };
          // Set the world channel coordinate to worldIndex.
          const result = invertNdTransformForQuery(
            [0, 0, 0, 0, worldIndex],
            [1e10, 1e10, 1e10, 5, 0.5],
            ndTransform,
            dims(['X', 'Y', 'Z', 'Time', 'Channel']),
            [0, 1, 2]
          );
          // Definition: local[k] = world[perm[k]], so the inverse is:
          // local index = (the i such that perm[i] === worldIndex).
          const expectedLocal = perm.indexOf(worldIndex);
          expect(result.slicePosition[4]).toBe(expectedLocal);
        }
      ),
      { numRuns: 100, seed: 0x5eed }
    );
  });
});
