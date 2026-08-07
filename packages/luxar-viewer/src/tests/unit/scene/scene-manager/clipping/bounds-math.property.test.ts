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
  MAX_NEAR_FAR_RATIO,
  minNearForRadius,
  nearPlaneFloor,
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

    // The invariant the whole depth-precision fix rests on. Depth
    // quantization on a 24-bit buffer is proportional to (far-near)/(near*far),
    // so an unbounded ratio is what produced the reported z-fighting once the
    // camera moved inside the bounding sphere. Asserting it as a PROPERTY (not
    // just at the reported pose) is what makes it a guarantee: no camera
    // distance, at any scene scale, can push the ratio past the bound.
    test('near/far ratio never exceeds MAX_NEAR_FAR_RATIO', () => {
      fc.assert(
        fc.property(radiusArb, distArb, (radius, dist) => {
          const { near, far } = planesAt(radius, dist);
          // Float slack only — the bound is an equality at the floor.
          expect(far / near).toBeLessThanOrEqual(MAX_NEAR_FAR_RATIO * (1 + 1e-9));
        })
      );
    });

    // Upper-bound half of the MAX_NEAR_FAR_RATIO derivation, stated as the
    // thing that actually matters: whenever the FLOOR is what sets `near`,
    // everything the near plane clips was already being discarded by the
    // point / line / gsplat vertex shaders, so raising the floor cannot hide
    // geometry those three types would have drawn. That is what makes the
    // ratio bound lossless rather than a quality tradeoff.
    //
    // Those shaders multiply by `perspectiveNearFade` (see
    // `materials/_shared/glsl-lib.ts`) and reject the vertex when it drops
    // below 0.01. The fade is monotone in view depth, so asserting
    // fade(near) <= 0.01 covers every depth the frustum clips.
    //
    // NOTE the margin here is genuinely thin, and asymmetric on purpose:
    // `far` carries SPHERE_SAFETY_EXPANSION but `nearCull` does not, so with
    // the camera right at the sphere surface the floor reaches ~1.05x
    // nearCull -- still inside the fade band's reject region (which extends
    // to ~1.058x nearCull), but only just. Raising MAX_NEAR_FAR_RATIO's
    // reciprocal any further would start clipping visible geometry, which is
    // the upper bound on the constant.
    const NEAR_FADE_REJECT = 0.01;
    const smoothstep = (e0: number, e1: number, x: number) => {
      const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
      return t * t * (3 - 2 * t);
    };

    // Generates the regime this property is ABOUT: every camera distance at
    // which the FLOOR is what sets `near`, which is NOT merely "inside the
    // sphere". The floor keeps binding until the surface distance overtakes it:
    //
    //   dist - R > (dist + R) / C   ⟺   dist > R · (C + 1) / (C - 1)
    //
    // i.e. out to dist ≈ 1.002 · R for C = 1000. That last 0.2% matters,
    // because it is where the floor sits HIGHEST relative to `nearCull` and so
    // where the losslessness margin is thinnest — the true worst case is there,
    // not on the sphere surface (fade 0.00755 vs 0.00725). An earlier version
    // generated `dist <= R` and therefore never visited the tightest point.
    //
    // Also deliberately NOT `fc.pre(near === floor)` over the wide `distArb`:
    // that spelling was measured to hold in 0.06% of cases (distArb reaches 1e6
    // while radiusArb stops at 1e4, so the camera is nearly always far OUTSIDE
    // the sphere), leaving the property's body to run ~0 times in fast-check's
    // default 100 runs — green and proving nothing. Parameterizing the
    // arbitrary keeps every generated case on-topic.
    const floorFracArb = fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true });

    test('when the floor sets near, the clipped band is already shader-rejected', () => {
      fc.assert(
        fc.property(radiusArb, floorFracArb, (radius, floorFrac) => {
          const R = radius * SPHERE_SAFETY_EXPANSION;
          // Crossover derived from the constant, not hardcoded, so tuning
          // MAX_NEAR_FAR_RATIO keeps this covering the right region.
          const crossover =
            (R * (MAX_NEAR_FAR_RATIO + 1)) / (MAX_NEAR_FAR_RATIO - 1) - Number.EPSILON * R;
          const dist = crossover * floorFrac;
          const { near, far } = planesAt(radius, dist);
          // Guard against this property silently going vacuous again: the
          // whole point is the regime where the floor binds.
          expect(near).toBe(nearPlaneFloor(R, far));
          // diagonal = 2 * radius (boundingBoxToSphere inverse), so
          // nearCull = 1e-3 * 2 * radius. See scene-bounds-cache.ts.
          const nearCull = 2 * radius * 0.001;
          expect(smoothstep(nearCull, 2 * nearCull, near)).toBeLessThanOrEqual(NEAR_FADE_REJECT);
        })
      );
    });

    // The tightest point specifically, as a fixed case rather than trusting the
    // arbitrary to sample it: at the crossover the floor/nearCull ratio peaks,
    // which is what pins MAX_NEAR_FAR_RATIO's lower bound at 993.
    test('the margin is thinnest at the floor/surface crossover, and holds there', () => {
      const radius = 50;
      const R = radius * SPHERE_SAFETY_EXPANSION;
      const nearCull = 2 * radius * 0.001;
      const crossover = (R * (MAX_NEAR_FAR_RATIO + 1)) / (MAX_NEAR_FAR_RATIO - 1);
      const atCrossover = planesAt(radius, crossover * (1 - 1e-12));
      const atSurface = planesAt(radius, R);

      const fadeAt = (n: number) => smoothstep(nearCull, 2 * nearCull, n);
      // Strictly worse than the sphere surface — the case the docs used to name.
      expect(fadeAt(atCrossover.near)).toBeGreaterThan(fadeAt(atSurface.near));
      // ...and still inside the reject band.
      expect(fadeAt(atCrossover.near)).toBeLessThanOrEqual(NEAR_FADE_REJECT);
      // The required constant at that point, which is what 1000 must clear.
      const requiredC = atCrossover.far / (1.0582 * nearCull);
      expect(requiredC).toBeCloseTo(993, 0);
      expect(MAX_NEAR_FAR_RATIO).toBeGreaterThanOrEqual(requiredC);
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
