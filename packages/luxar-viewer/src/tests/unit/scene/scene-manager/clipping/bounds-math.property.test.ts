/**
 * Property tests for bounds-math (fast-check).
 *
 * Companion to bounds-math.test.ts. bounds-math is pure numeric geometry, so
 * the example tests there are best backed by algebraic invariants holding over
 * arbitrary boxes/points (audit findings clipping G2 / G8):
 *
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
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  getBoundingBoxDiagonal,
  minNearForRadius,
  SPHERE_SAFETY_EXPANSION,
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

const corners = (b: BoundingBox) =>
  [b.min.x, b.max.x].flatMap((x) =>
    [b.min.y, b.max.y].flatMap((y) => [b.min.z, b.max.z].map((z) => ({ x, y, z })))
  );

describe('bounds-math properties', () => {
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

  describe('calculateClippingPlanesFromSphere (scale-aware near floor)', () => {
    // Camera along +z from a sphere at the origin — the planes depend only
    // on (dist, radius), so one axis covers the full input space.
    const radiusArb = fc.double({ min: 1e-6, max: 1e4, noNaN: true, noDefaultInfinity: true });
    const distArb = fc.double({ min: 0, max: 1e6, noNaN: true, noDefaultInfinity: true });
    const planesAt = (radius: number, dist: number) =>
      calculateClippingPlanesFromSphere(
        { center: { x: 0, y: 0, z: 0 }, radius },
        { x: 0, y: 0, z: dist }
      );

    test('always yields a valid frustum (0 < near < far) for any non-degenerate sphere', () => {
      fc.assert(
        fc.property(radiusArb, distArb, (radius, dist) => {
          const { near, far } = planesAt(radius, dist);
          expect(near).toBeGreaterThan(0);
          expect(far).toBeGreaterThan(near);
        })
      );
    });

    test('near never drops below the scale-aware floor', () => {
      fc.assert(
        fc.property(radiusArb, distArb, (radius, dist) => {
          const { near } = planesAt(radius, dist);
          expect(near).toBeGreaterThanOrEqual(minNearForRadius(radius * SPHERE_SAFETY_EXPANSION));
        })
      );
    });

    test('near and far are monotone non-decreasing in camera distance', () => {
      fc.assert(
        fc.property(radiusArb, distArb, distArb, (radius, d1, d2) => {
          const lo = Math.min(d1, d2);
          const hi = Math.max(d1, d2);
          const a = planesAt(radius, lo);
          const b = planesAt(radius, hi);
          expect(b.near).toBeGreaterThanOrEqual(a.near);
          expect(b.far).toBeGreaterThanOrEqual(a.far);
        })
      );
    });

    test('near is continuous across the sphere-surface boundary (dist == R)', () => {
      // Crossing from inside (near = floor) to just outside
      // (near = max(floor, dist - R)) must not jump: at dist = R + ε the
      // outside branch gives max(floor, ε) which converges to the floor
      // as ε → 0. A discontinuity here would visibly pop the near plane
      // while orbiting through the sphere surface.
      fc.assert(
        fc.property(
          radiusArb,
          fc.double({ min: 1e-12, max: 1e-3, noNaN: true, noDefaultInfinity: true }),
          (radius, epsFraction) => {
            const R = radius * SPHERE_SAFETY_EXPANSION;
            const eps = R * epsFraction;
            const inside = planesAt(radius, R - eps);
            const outside = planesAt(radius, R + eps);
            // Jump bounded by the crossing step (2ε) plus float64 rounding
            // at the working magnitude (a few ulps of R — the 1e-12-style
            // absolute slack is smaller than ulp(2R) for R ≳ 1e4).
            const slack = 2 * eps + 8 * R * Number.EPSILON;
            expect(Math.abs(outside.near - inside.near)).toBeLessThanOrEqual(slack);
            expect(Math.abs(outside.far - inside.far)).toBeLessThanOrEqual(slack);
          }
        )
      );
    });
  });
});
