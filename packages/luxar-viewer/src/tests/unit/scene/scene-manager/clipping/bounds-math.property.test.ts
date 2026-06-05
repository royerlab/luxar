/**
 * Property tests for bounds-math (fast-check).
 *
 * Companion to bounds-math.test.ts. bounds-math is pure numeric geometry, so
 * the example tests there are best backed by algebraic invariants holding over
 * arbitrary boxes/points (audit findings clipping G2 / G8):
 *
 *   mergeBoundingBoxes (union):
 *     - commutative:  merge([A, B]) == merge([B, A])
 *     - associative:  merge([A, B, C]) == merge([merge([A, B]), C])
 *     - idempotent:   merge([A, A]) == A
 *     - containment:  every input box lies inside the union
 *   calculateBoundingBoxFromPositions:
 *     - every input point lies inside the computed box
 *   expandBoundingBox:
 *     - monotone:     larger margin ⇒ box contains the smaller-margin box
 *     - additive:     expanding by a then by b == expanding by (a + b)
 *   boundingBoxToSphere:
 *     - containment:  all 8 corners lie within the sphere radius
 *   getBoundingBoxDiagonal:
 *     - is the largest pairwise corner distance (min↔max corner)
 *
 * The example tests pin specific cases; this file exercises arbitrary inputs.
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import {
  BoundingBox,
  mergeBoundingBoxes,
  calculateBoundingBoxFromPositions,
  expandBoundingBox,
  boundingBoxToSphere,
  getBoundingBoxDiagonal,
  isPointInBoundingBox,
} from '../../../../../scene/scene-manager/clipping/bounds-math';

// Finite, well-scaled coordinate (avoids NaN/Inf and float-blowup noise).
const coord = fc.double({ min: -1e4, max: 1e4, noNaN: true, noDefaultInfinity: true });

// A normalized box arbitrary: min <= max on every axis (the helpers' invariant).
const boxArb: fc.Arbitrary<BoundingBox> = fc
  .record({
    ax: coord,
    bx: coord,
    ay: coord,
    by: coord,
    az: coord,
    bz: coord,
  })
  .map(({ ax, bx, ay, by, az, bz }) => ({
    min: { x: Math.min(ax, bx), y: Math.min(ay, by), z: Math.min(az, bz) },
    max: { x: Math.max(ax, bx), y: Math.max(ay, by), z: Math.max(az, bz) },
  }));

const boxesEqual = (a: BoundingBox, b: BoundingBox, eps = 1e-6): boolean =>
  Math.abs(a.min.x - b.min.x) <= eps &&
  Math.abs(a.min.y - b.min.y) <= eps &&
  Math.abs(a.min.z - b.min.z) <= eps &&
  Math.abs(a.max.x - b.max.x) <= eps &&
  Math.abs(a.max.y - b.max.y) <= eps &&
  Math.abs(a.max.z - b.max.z) <= eps;

const corners = (b: BoundingBox) =>
  [b.min.x, b.max.x].flatMap((x) =>
    [b.min.y, b.max.y].flatMap((y) => [b.min.z, b.max.z].map((z) => ({ x, y, z })))
  );

describe('bounds-math properties', () => {
  describe('mergeBoundingBoxes (union)', () => {
    test('is commutative', () => {
      fc.assert(
        fc.property(boxArb, boxArb, (a, b) => {
          expect(boxesEqual(mergeBoundingBoxes([a, b]), mergeBoundingBoxes([b, a]))).toBe(true);
        })
      );
    });

    test('is associative', () => {
      fc.assert(
        fc.property(boxArb, boxArb, boxArb, (a, b, c) => {
          const left = mergeBoundingBoxes([mergeBoundingBoxes([a, b]), c]);
          const flat = mergeBoundingBoxes([a, b, c]);
          expect(boxesEqual(left, flat)).toBe(true);
        })
      );
    });

    test('is idempotent', () => {
      fc.assert(
        fc.property(boxArb, (a) => {
          expect(boxesEqual(mergeBoundingBoxes([a, a]), a)).toBe(true);
        })
      );
    });

    test('contains every input box (union ⊇ inputs)', () => {
      fc.assert(
        fc.property(fc.array(boxArb, { minLength: 1, maxLength: 6 }), (boxes) => {
          const union = mergeBoundingBoxes(boxes);
          for (const b of boxes) {
            // Every corner of every input box is inside the union.
            for (const c of corners(b)) {
              expect(isPointInBoundingBox(c, union)).toBe(true);
            }
          }
        })
      );
    });
  });

  describe('calculateBoundingBoxFromPositions', () => {
    test('contains every input point', () => {
      const tripletArb = fc.array(fc.tuple(coord, coord, coord), { minLength: 1, maxLength: 50 });
      fc.assert(
        fc.property(tripletArb, (pts) => {
          const flat = pts.flat();
          const box = calculateBoundingBoxFromPositions(flat);
          for (const [x, y, z] of pts) {
            expect(isPointInBoundingBox({ x, y, z }, box)).toBe(true);
          }
        })
      );
    });
  });

  describe('expandBoundingBox', () => {
    test('is monotone: a larger positive margin contains the smaller-margin box', () => {
      const marginArb = fc.double({ min: 0, max: 1e3, noNaN: true, noDefaultInfinity: true });
      fc.assert(
        fc.property(boxArb, marginArb, marginArb, (box, m1, m2) => {
          const small = expandBoundingBox(box, Math.min(m1, m2));
          const large = expandBoundingBox(box, Math.max(m1, m2));
          for (const c of corners(small)) {
            expect(isPointInBoundingBox(c, large)).toBe(true);
          }
        })
      );
    });

    test('is additive: expand(a) then expand(b) == expand(a + b)', () => {
      const marginArb = fc.double({ min: -1e2, max: 1e3, noNaN: true, noDefaultInfinity: true });
      fc.assert(
        fc.property(boxArb, marginArb, marginArb, (box, a, b) => {
          const sequential = expandBoundingBox(expandBoundingBox(box, a), b);
          const combined = expandBoundingBox(box, a + b);
          expect(boxesEqual(sequential, combined, 1e-4)).toBe(true);
        })
      );
    });
  });

  describe('boundingBoxToSphere', () => {
    test('circumscribes the box: all 8 corners are within the radius', () => {
      fc.assert(
        fc.property(boxArb, (box) => {
          const s = boundingBoxToSphere(box);
          for (const c of corners(box)) {
            const d = Math.hypot(c.x - s.center.x, c.y - s.center.y, c.z - s.center.z);
            // Allow a relative float slack proportional to the radius.
            expect(d).toBeLessThanOrEqual(s.radius + s.radius * 1e-9 + 1e-9);
          }
        })
      );
    });
  });

  describe('getBoundingBoxDiagonal', () => {
    test('equals the min↔max corner distance (the largest pairwise distance)', () => {
      fc.assert(
        fc.property(boxArb, (box) => {
          const diag = getBoundingBoxDiagonal(box);
          const cs = corners(box);
          let maxPair = 0;
          for (let i = 0; i < cs.length; i++) {
            for (let j = i + 1; j < cs.length; j++) {
              maxPair = Math.max(
                maxPair,
                Math.hypot(cs[i].x - cs[j].x, cs[i].y - cs[j].y, cs[i].z - cs[j].z)
              );
            }
          }
          expect(diag).toBeCloseTo(maxPair, 6);
        })
      );
    });
  });
});
