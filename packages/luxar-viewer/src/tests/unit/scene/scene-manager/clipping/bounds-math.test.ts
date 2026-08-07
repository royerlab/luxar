/**
 * Tests for scene manager utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  getBoundingBoxCenter,
  getBoundingBoxSize,
  getBoundingBoxMaxDimension,
  calculateCameraDistance,
  validateFOV,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  projectBoundsToDisplayDims,
  SPHERE_SAFETY_EXPANSION,
  MIN_NEAR_PLANE,
  MAX_NEAR_FAR_RATIO,
  minNearForRadius,
  nearPlaneFloor,
  getBoundingBoxDiagonal,
  transformBoundingBox,
  BoundingBox,
  CameraConfig,
} from '../../../../../scene/scene-manager/clipping/bounds-math';

describe('bounds-math', () => {
  describe('getBoundingBoxCenter', () => {
    it('should calculate center correctly', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -2, z: -3 },
        max: { x: 1, y: 2, z: 3 },
      };

      const center = getBoundingBoxCenter(box);

      expect(center).toEqual({ x: 0, y: 0, z: 0 });
    });

    it('should handle offset box', () => {
      const box: BoundingBox = {
        min: { x: 10, y: 20, z: 30 },
        max: { x: 20, y: 30, z: 40 },
      };

      const center = getBoundingBoxCenter(box);

      expect(center).toEqual({ x: 15, y: 25, z: 35 });
    });
  });

  describe('getBoundingBoxSize', () => {
    it('should calculate size correctly', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -2, z: -3 },
        max: { x: 1, y: 2, z: 3 },
      };

      const size = getBoundingBoxSize(box);

      expect(size).toEqual({ x: 2, y: 4, z: 6 });
    });

    it('should handle zero-size box', () => {
      const box: BoundingBox = {
        min: { x: 5, y: 5, z: 5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const size = getBoundingBoxSize(box);

      expect(size).toEqual({ x: 0, y: 0, z: 0 });
    });
  });

  describe('getBoundingBoxMaxDimension', () => {
    it('should return maximum dimension (y is largest)', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 10, y: 20, z: 15 },
      };

      expect(getBoundingBoxMaxDimension(box)).toBe(20);
    });

    // W1: original test only exercised the case where the *last* compared
    // axis (y) is the max. Cover x-dominant and z-dominant so a mutation that
    // returns a fixed axis (e.g. always size.z) is killed.
    it('returns the x extent when x is the largest dimension', () => {
      const box: BoundingBox = {
        min: { x: 0, y: -100, z: 0 },
        max: { x: 200, y: 50, z: 1 },
      };
      // sizes: x=200, y=150, z=1 → max is x.
      expect(getBoundingBoxMaxDimension(box)).toBe(200);
    });

    it('returns the z extent when z is the largest dimension', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: -30 },
        max: { x: 5, y: 7, z: 70 },
      };
      // sizes: x=5, y=7, z=100 → max is z.
      expect(getBoundingBoxMaxDimension(box)).toBe(100);
    });

    it('handles negative coordinates symmetrically (extent is a positive width)', () => {
      const box: BoundingBox = {
        min: { x: -50, y: -3, z: -2 },
        max: { x: -10, y: -1, z: -1 },
      };
      // sizes: x=40, y=2, z=1 → max is x=40 (all-negative coords, positive extent).
      expect(getBoundingBoxMaxDimension(box)).toBe(40);
    });
  });

  describe('calculateCameraDistance', () => {
    // W2/M2: reference implementation of the documented formula so tests can
    // assert the exact value instead of a loose band. Mirrors bounds-math.ts:
    //   vertical   = maxDim / fitRatio / (2*tan(halfFov))
    //   horizontal = vertical / aspect
    //   distance   = max(vertical, horizontal) * 1.2  (20% margin)
    const expectedCameraDistance = (
      maxDim: number,
      fovDeg: number,
      aspect: number,
      fitRatio: number
    ): number => {
      const halfFov = (fovDeg * Math.PI) / 180 / 2;
      const vertical = maxDim / fitRatio / (2 * Math.tan(halfFov));
      const horizontal = vertical / aspect;
      return Math.max(vertical, horizontal) * 1.2;
    };

    it('should calculate distance for perspective camera', () => {
      const box: BoundingBox = {
        min: { x: -5, y: -5, z: -5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const camera: CameraConfig = {
        fov: 60,
        aspect: 16 / 9,
        near: 0.1,
        far: 1000,
      };

      const distance = calculateCameraDistance(box, camera);

      // Should be approximately 10 / tan(30°) * 1.1 * fitRatio factor
      expect(distance).toBeGreaterThan(10);
      expect(distance).toBeLessThan(50);
    });

    // W2: pin the exact formula value (explicit fitRatio removes the config
    // dependency). Kills constant-offset / wrong-margin / missing-factor
    // mutants that a loose [10,50] band would survive.
    it('matches the documented FOV/aspect/fitRatio formula exactly', () => {
      const box: BoundingBox = { min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } };
      const camera: CameraConfig = { fov: 60, aspect: 16 / 9, near: 0.1, far: 1000 };
      const distance = calculateCameraDistance(box, camera, 0.75);
      // maxDim = 10, fov = 60 (aspect > 1 → vertical fit dominates).
      expect(distance).toBeCloseTo(expectedCameraDistance(10, 60, 16 / 9, 0.75), 6);
    });

    it('should handle different aspect ratios', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const wideCamera: CameraConfig = {
        fov: 60,
        aspect: 21 / 9,
        near: 0.1,
        far: 1000,
      };

      const tallCamera: CameraConfig = {
        fov: 60,
        aspect: 9 / 16,
        near: 0.1,
        far: 1000,
      };

      const wideDistance = calculateCameraDistance(box, wideCamera);
      const tallDistance = calculateCameraDistance(box, tallCamera);

      // Tall camera needs more distance to fit same object
      expect(tallDistance).toBeGreaterThan(wideDistance);

      // M1: pin the *magnitude* of the difference, not just the sign. For a
      // wide aspect (>1) the vertical fit dominates, so wideDistance is
      // independent of aspect. For the tall aspect (9/16) the horizontal fit
      // dominates: distance = vertical / aspect. Hence the ratio equals
      // exactly 1/aspect_tall = 16/9 ≈ 1.778. A mutant that drops the aspect
      // term from horizontalFit would make the ratio 1 and survive a sign-only
      // assertion.
      expect(tallDistance / wideDistance).toBeCloseTo(16 / 9, 6);
    });

    it('should scale distance proportionally with scene size', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 0.1, far: 1000 };

      const smallBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1, y: 1, z: 1 },
      };
      const largeBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 500, y: 500, z: 500 },
      };

      const smallDist = calculateCameraDistance(smallBox, camera);
      const largeDist = calculateCameraDistance(largeBox, camera);

      // Distance scales linearly with scene size (maxDim ratio is exactly 500).
      // M2: the original tolerance of 0 digits (±5) let a constant-offset
      // mutant (`maxDim/fitRatio + 1`) survive — it shifts the ratio only to
      // ~499.8. Tighten to 3 digits (±5e-4): pure linear scaling gives 500.0,
      // any additive constant breaks it.
      expect(largeDist / smallDist).toBeCloseTo(500, 3);

      // Belt-and-suspenders: a mid-size box must fall exactly on the same
      // proportionality line through the origin (no offset).
      const midBox: BoundingBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 50, y: 50, z: 50 } };
      const midDist = calculateCameraDistance(midBox, camera);
      expect(midDist / smallDist).toBeCloseTo(50, 3);
    });

    it('should return 0 for zero-size bounding box', () => {
      const camera: CameraConfig = { fov: 60, aspect: 1, near: 0.1, far: 1000 };
      const zeroBox: BoundingBox = {
        min: { x: 5, y: 5, z: 5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const distance = calculateCameraDistance(zeroBox, camera);
      expect(distance).toBe(0);
    });

    it('should handle very small scenes (nanometer scale)', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 0.001, far: 100 };
      const nanoBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0.001, y: 0.001, z: 0.001 },
      };

      const distance = calculateCameraDistance(nanoBox, camera);
      expect(distance).toBeGreaterThan(0);
      expect(distance).toBeLessThan(0.01);
      expect(Number.isFinite(distance)).toBe(true);
    });

    it('should handle very large scenes', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 1, far: 100000 };
      const hugeBox: BoundingBox = {
        min: { x: -5000, y: -5000, z: -5000 },
        max: { x: 5000, y: 5000, z: 5000 },
      };

      const distance = calculateCameraDistance(hugeBox, camera);
      expect(distance).toBeGreaterThan(5000);
      expect(Number.isFinite(distance)).toBe(true);
    });
  });

  describe('validateFOV', () => {
    it('should accept valid FOV values', () => {
      expect(validateFOV(60)).toBe(60);
      expect(validateFOV(45)).toBe(45);
      expect(validateFOV(90)).toBe(90);
    });

    it('should clamp to minimum', () => {
      expect(validateFOV(5)).toBe(10);
      expect(validateFOV(-10)).toBe(10);
    });

    it('should clamp to maximum', () => {
      expect(validateFOV(150)).toBe(120);
      expect(validateFOV(200)).toBe(120);
    });

    it('should use custom limits', () => {
      expect(validateFOV(25, 20, 80)).toBe(25);
      expect(validateFOV(15, 20, 80)).toBe(20);
      expect(validateFOV(90, 20, 80)).toBe(80);
    });
  });

  describe('boundingBoxToSphere', () => {
    it('should compute circumscribed sphere from bounding box', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const sphere = boundingBoxToSphere(box);

      expect(sphere.center).toEqual({ x: 0, y: 0, z: 0 });
      // half-diagonal = sqrt(20^2 + 20^2 + 20^2) / 2 = sqrt(1200) / 2 ≈ 17.32
      expect(sphere.radius).toBeCloseTo(17.32, 1);
    });

    it('should handle offset box', () => {
      const box: BoundingBox = {
        min: { x: 10, y: 20, z: 30 },
        max: { x: 20, y: 30, z: 40 },
      };

      const sphere = boundingBoxToSphere(box);

      expect(sphere.center).toEqual({ x: 15, y: 25, z: 35 });
      // half-diagonal of 10x10x10 cube = sqrt(300)/2 ≈ 8.66
      expect(sphere.radius).toBeCloseTo(8.66, 1);
    });

    // H2: the circumscribed-sphere radius is EXACTLY half the box diagonal —
    // the SPHERE_SAFETY_EXPANSION factor is applied later (in
    // calculateClippingPlanesFromSphere), NOT here. Pin the precise value so
    // a mutant that folds the expansion in early, or changes the 0.5 factor,
    // is killed.
    it('radius equals exactly half the box diagonal (no safety expansion applied here)', () => {
      const box: BoundingBox = { min: { x: 0, y: 0, z: 0 }, max: { x: 2, y: 4, z: 4 } };
      const sphere = boundingBoxToSphere(box);
      // diagonal = sqrt(4 + 16 + 16) = 6, half = 3 (exact).
      expect(sphere.radius).toBeCloseTo(3, 10);
      expect(sphere.radius).toBeCloseTo(getBoundingBoxDiagonal(box) / 2, 12);
    });

    // G5: containment invariant — the sphere must contain all 8 box corners.
    it('contains all 8 box corners within the radius', () => {
      const box: BoundingBox = { min: { x: -3, y: 1, z: -7 }, max: { x: 11, y: 5, z: 2 } };
      const sphere = boundingBoxToSphere(box);
      const corners = [box.min.x, box.max.x].flatMap((x) =>
        [box.min.y, box.max.y].flatMap((y) => [box.min.z, box.max.z].map((z) => ({ x, y, z })))
      );
      expect(corners).toHaveLength(8);
      for (const c of corners) {
        const d = Math.hypot(c.x - sphere.center.x, c.y - sphere.center.y, c.z - sphere.center.z);
        // Each corner sits on or inside the circumscribed sphere (allow tiny
        // float slack above the radius).
        expect(d).toBeLessThanOrEqual(sphere.radius + 1e-9);
      }
    });
  });

  describe('minNearForRadius', () => {
    // Literal-value pins: the plane tests below assert against
    // minNearForRadius(R) itself (wiring), which cannot catch a broken
    // formula. These pin the formula's actual values.
    it('reproduces the historical 1e-4 floor at a typical diagonal-100 scene', () => {
      // diagonal 100 → radius 50 → expanded radius 52.5 → 52.5 * 2e-6.
      expect(minNearForRadius(52.5)).toBeCloseTo(1.05e-4, 9);
    });

    it('scales down proportionally for tiny scenes', () => {
      // diagonal 0.01 → expanded radius ~0.0053 → ~1.05e-8 (well below
      // the old absolute 1e-4, above the 1e-9 last-resort floor).
      expect(minNearForRadius(0.00525)).toBeCloseTo(1.05e-8, 12);
    });

    it('falls back to the absolute floor for a degenerate zero radius', () => {
      expect(minNearForRadius(0)).toBe(MIN_NEAR_PLANE);
      expect(minNearForRadius(0)).toBe(1e-9);
    });
  });

  describe('nearPlaneFloor', () => {
    it('bounds the near/far ratio to MAX_NEAR_FAR_RATIO', () => {
      // The user-reported pose: diagonal-100 scene (expanded R = 52.5) seen
      // from 8.5 units off centre → far = 61. The old floor gave
      // near = 1.05e-4 (ratio 5.8e5:1, ~4e-2 world units of depth
      // quantization at 24 bits); the bound gives 0.061.
      expect(nearPlaneFloor(52.5, 61)).toBeCloseTo(0.061, 9);
      expect(61 / nearPlaneFloor(52.5, 61)).toBeCloseTo(MAX_NEAR_FAR_RATIO, 6);
    });

    // The two constraints that pin MAX_NEAR_FAR_RATIO, as executable arithmetic
    // rather than prose. Both are LOWER bounds on C (a bigger C means a smaller
    // floor), so C wants to be the smallest value clearing both — which is why
    // the margin below is only 0.8% and why that thinness is intentional.
    it('satisfies both constraints that pin the constant, and shows how tightly', () => {
      const NEAR_CULL_FACTOR = 0.001; // scene-bounds-cache.ts
      const MIN_DISTANCE_FACTOR = 0.001; // config.controls.scaleMultipliers
      const FADE_REJECT_HEADROOM = 1.0582; // smoothstep(nc, 2nc, x) < 0.01 below this * nc
      // Per unit R: diagonal = 2R / SPHERE_SAFETY_EXPANSION.
      const perR = (factor: number) => (factor * 2) / SPHERE_SAFETY_EXPANSION;

      // (1) Don't clip the orbit target at maximum zoom-in (far ~ 1.05 R there).
      const cMinTarget = SPHERE_SAFETY_EXPANSION / perR(MIN_DISTANCE_FACTOR);
      // (2) Stay inside the shader reject band. The worst case is the crossover
      //     just OUTSIDE the sphere — dist = R(C+1)/(C-1) ~ 1.002 R, the last
      //     distance where the floor still beats the surface term — not the
      //     surface itself, where the floor is slightly lower.
      const crossoverOverR = (MAX_NEAR_FAR_RATIO + 1) / (MAX_NEAR_FAR_RATIO - 1);
      const farAtCrossoverOverR = crossoverOverR + 1;
      const cMinLossless = farAtCrossoverOverR / (FADE_REJECT_HEADROOM * perR(NEAR_CULL_FACTOR));

      expect(cMinTarget).toBeCloseTo(551, 0);
      expect(cMinLossless).toBeCloseTo(993, 0);
      // Strictly tighter than the surface case the derivation used to name.
      const cMinAtSurface = 2 / (FADE_REJECT_HEADROOM * perR(NEAR_CULL_FACTOR));
      expect(cMinLossless).toBeGreaterThan(cMinAtSurface);
      // The lossless constraint binds, and the constant clears it — barely.
      expect(cMinLossless).toBeGreaterThan(cMinTarget);
      expect(MAX_NEAR_FAR_RATIO).toBeGreaterThanOrEqual(cMinLossless);
      expect(MAX_NEAR_FAR_RATIO / cMinLossless - 1).toBeLessThan(0.01); // ~0.7% margin
    });

    it('dominates the minNearForRadius backstop for every non-degenerate sphere', () => {
      // far >= R always holds (far = dist + R, dist >= 0), so far/1000 is
      // always >= 1e-3*R, which is 500x the 2e-6*R backstop. The backstop
      // therefore only ever surfaces at R -> 0. Pinning this keeps the
      // Math.max in nearPlaneFloor honest: if someone raises
      // MIN_NEAR_RADIUS_FACTOR back into the dominant position, the
      // Z-precision bound silently stops applying.
      for (const R of [1e-3, 1, 52.5, 1e4]) {
        for (const distFactor of [0, 0.5, 1, 10, 1000]) {
          const far = R * distFactor + R;
          expect(nearPlaneFloor(R, far)).toBe(far / MAX_NEAR_FAR_RATIO);
        }
      }
    });

    // Ortho opts OUT of the ratio bound. Its depth is linear in eye space, so
    // resolution is (far - near) / 2^24 regardless of near — the bound buys it
    // nothing — while `perspectiveNearFade` returns 1.0 under ortho, meaning
    // ALL FOUR geometry types render right up to `near` there. Measured: at the
    // deepest legal orbit distance on a diagonal-100 scene, applying the bound
    // under ortho clips 52.7% of the eye-to-target depth for a 0.1% change in
    // depth resolution.
    it('omits the ratio bound for an orthographic projection', () => {
      const R = 52.5;
      const far = 61;
      expect(nearPlaneFloor(R, far, false)).toBe(minNearForRadius(R));
      expect(nearPlaneFloor(R, far, false)).toBeCloseTo(1.05e-4, 9);
      // ...and still applies it for perspective, from the same inputs.
      expect(nearPlaneFloor(R, far, true)).toBeCloseTo(0.061, 9);
      // Default is perspective — the common case must not need the argument.
      expect(nearPlaneFloor(R, far)).toBe(nearPlaneFloor(R, far, true));
    });

    it('keeps the absolute floor under ortho for a degenerate zero radius', () => {
      expect(nearPlaneFloor(0, 0, false)).toBe(MIN_NEAR_PLANE);
      expect(nearPlaneFloor(0, 100, false)).toBe(MIN_NEAR_PLANE);
    });

    it('falls back to the absolute floor when far collapses to zero', () => {
      // Degenerate single-point scene with the camera on the point: R = 0
      // and far = 0. near must stay > 0 so `near >= far` still trips the
      // callers' degenerate-frustum guard instead of writing NaNs.
      expect(nearPlaneFloor(0, 0)).toBe(MIN_NEAR_PLANE);
    });

    it('stays below the nearCull threshold that already hides points/lines/gsplats', () => {
      // `perspectiveNearFade` (materials/_shared/glsl-lib.ts) returns 0 at
      // -viewZ <= nearCull = 1e-3 * diagonal, and the point/line/gsplat
      // vertex shaders reject the vertex there. A floor at or below nearCull
      // is therefore provably lossless for those three geometry types --
      // this is the upper-bound half of the MAX_NEAR_FAR_RATIO derivation.
      for (const diagonal of [0.01, 1, 100, 1e5]) {
        const R = 0.5 * diagonal * SPHERE_SAFETY_EXPANSION;
        const nearCull = diagonal * 0.001;
        // Worst case for the floor is the camera at the bounds centre,
        // where far is smallest relative to nothing else raising near.
        expect(nearPlaneFloor(R, R)).toBeLessThanOrEqual(nearCull);
      }
    });
  });

  describe('calculateClippingPlanesFromSphere', () => {
    it('should calculate clipping planes when outside sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;
      // Camera at z=50, facing center
      const cameraPos = { x: 0, y: 0, z: 50 };
      const planes = calculateClippingPlanesFromSphere(sphere, cameraPos);

      // dist = 50 exactly; near/far are 50∓R. M3: tightened from 1 to 6 digits
      // so a mutant that drops SPHERE_SAFETY_EXPANSION from far (R → radius)
      // is killed — the difference (0.05*radius ≈ 0.87) far exceeds tolerance.
      expect(planes.near).toBeCloseTo(50 - R, 6);
      expect(planes.far).toBeCloseTo(50 + R, 6);
    });

    // W3: camera exactly on the (expanded) sphere surface, dist == R. The
    // `dist < R` branch is false, so near = max(nearPlaneFloor(R, far), dist - R)
    // = max(nearPlaneFloor(R, 2R), 0) = the floor, and far = 2R.
    it('clamps near to the depth-precision floor when camera sits on the sphere surface (dist == R)', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };
      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;
      // Place the camera exactly R away from the center along +z.
      const planes = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: R });
      expect(planes.near).toBe(nearPlaneFloor(R, 2 * R));
      expect(planes.far).toBeCloseTo(2 * R, 6);
      // Literal pin: the floor is far/MAX_NEAR_FAR_RATIO = 2R/1000, NOT the
      // dominated minNearForRadius backstop (2e-6*R). Asserting against
      // nearPlaneFloor alone cannot catch a broken formula inside it.
      expect(planes.near).toBeCloseTo((2 * R) / 1000, 12);
      expect(planes.near).toBeGreaterThan(minNearForRadius(R) * 100);
    });

    // W3: camera very far from the sphere (dist >> R) — near must track
    // (dist - R) and stay well above the floor; far tracks (dist + R).
    it('returns near = dist - R and far = dist + R when camera is far outside the sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };
      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;
      const dist = 100000;
      const planes = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: dist });
      expect(planes.near).toBeCloseTo(dist - R, 4);
      expect(planes.far).toBeCloseTo(dist + R, 4);
      expect(planes.near).toBeGreaterThan(MIN_NEAR_PLANE);
    });

    it('should use the depth-precision floor as near plane when inside sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;
      // Camera inside the sphere at (0,0,8)
      const cameraPos = { x: 0, y: 0, z: 8 };
      const planes = calculateClippingPlanesFromSphere(sphere, cameraPos);

      expect(planes.near).toBe(nearPlaneFloor(R, 8 + R));
      expect(planes.far).toBeCloseTo(8 + R, 1);
      // The regression this floor exists for: near/far must stay within
      // MAX_NEAR_FAR_RATIO so 24-bit depth stays usable. Before the ratio
      // bound this arm returned 2e-6*R ≈ 3.6e-5 — a ratio of ~7e5:1.
      expect(planes.far / planes.near).toBeCloseTo(MAX_NEAR_FAR_RATIO, 6);
    });

    it('should enforce a positive near plane for tiny scenes', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0.01, y: 0.01, z: 0.01 },
      };

      const sphere = boundingBoxToSphere(box);
      const cameraPos = { x: 0.005, y: 0.005, z: 0.02 };
      const planes = calculateClippingPlanesFromSphere(sphere, cameraPos);

      expect(planes.near).toBeGreaterThanOrEqual(MIN_NEAR_PLANE);
      expect(planes.near).toBeGreaterThan(0);
    });

    // Regression for scale-invariance of the near floor: with the 1000x
    // zoom-in headroom, a tiny scene (diagonal ~0.01) lets the camera orbit at
    // distances far below the OLD absolute floor (0.0001). The floor must
    // scale with the scene so target-adjacent geometry is never behind
    // the near plane at the deepest legal zoom. The far-derived
    // `nearPlaneFloor` inherits this for free — `far` is itself scene-scaled —
    // which is why replacing the radius-proportional floor with the ratio
    // bound does not reintroduce the #573 tiny-scene clipping bug.
    it('keeps near below the deepest zoom-in distance for tiny scenes (scale-invariant floor)', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0.01, y: 0.01, z: 0.01 },
      };
      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;
      // Deepest legal orbit distance ~ framed distance / ZOOM_IN_FACTOR;
      // framed distance is on the order of the diagonal (~2R), so use
      // 2R / 1000 as the representative deepest zoom.
      const deepestZoom = (2 * R) / 1000;
      const cameraPos = {
        x: sphere.center.x,
        y: sphere.center.y,
        z: sphere.center.z + deepestZoom,
      };
      const planes = calculateClippingPlanesFromSphere(sphere, cameraPos);

      // Inside the sphere → near is the depth-precision floor, which must sit
      // strictly below the camera-to-target distance (the old absolute
      // 0.0001 floor failed this: 0.0001 > ~1.8e-5).
      expect(planes.near).toBe(nearPlaneFloor(R, planes.far));
      expect(planes.near).toBeLessThan(deepestZoom);
      // The margin is the reason MAX_NEAR_FAR_RATIO cannot go much below
      // 1000: at the deepest legal zoom far ≈ R, so the floor is ≈ R/1000
      // while the target sits at 2R/1000 — a factor of 2. At C = 500 the
      // near plane would swallow the thing you zoomed in on.
      expect(deepestZoom / planes.near).toBeGreaterThan(1.9);
    });

    it('should provide smooth near-plane transition approaching the sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const sphere = boundingBoxToSphere(box);
      // Camera moving from z=50 toward center — near should decrease smoothly
      const far = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 50 });
      const mid = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 30 });
      const close = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 22 });

      expect(far.near).toBeGreaterThan(mid.near);
      expect(mid.near).toBeGreaterThan(close.near);
      expect(close.near).toBeGreaterThan(0);
    });

    // M4: three sample points only prove ordering, which a piecewise/jumpy
    // function could also satisfy. Sample densely and assert the near plane is
    // strictly monotonic in camera distance AND continuous (each step changes
    // by ~the step in distance, i.e. d(near)/d(dist) ≈ 1 in the outside-sphere
    // regime) — no discontinuities.
    it('near plane decreases continuously (no jumps) as the camera approaches', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };
      const sphere = boundingBoxToSphere(box);
      const R = sphere.radius * SPHERE_SAFETY_EXPANSION;

      // Sample from z=60 down to just outside the sphere in 1-unit steps.
      const zs: number[] = [];
      for (let z = 60; z > Math.ceil(R) + 1; z -= 1) zs.push(z);
      const nears = zs.map(
        (z) => calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z }).near
      );

      for (let i = 1; i < nears.length; i++) {
        // Strictly decreasing as distance shrinks.
        expect(nears[i]).toBeLessThan(nears[i - 1]);
        // Outside the sphere near = dist - R, so a 1-unit move changes near by
        // exactly 1 unit. A jump/discontinuity would break this.
        expect(nears[i - 1] - nears[i]).toBeCloseTo(1, 6);
      }
    });
  });

  describe('getBoundingBoxDiagonal', () => {
    it('should calculate diagonal length', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 3, y: 4, z: 0 },
      };

      // 3-4-5 triangle
      expect(getBoundingBoxDiagonal(box)).toBe(5);
    });

    it('should handle 3D diagonal', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 2, y: 2, z: 2 },
      };

      // sqrt(2^2 + 2^2 + 2^2) = sqrt(12)
      expect(getBoundingBoxDiagonal(box)).toBeCloseTo(3.464, 3);
    });
  });

  describe('transformBoundingBox', () => {
    it('should handle identity transform', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Identity matrix (column-major)
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      const transformed = transformBoundingBox(box, identity);

      expect(transformed.min).toEqual(box.min);
      expect(transformed.max).toEqual(box.max);
    });

    it('should handle translation', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Translation matrix (translate by 5, 10, 15)
      const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1];

      const transformed = transformBoundingBox(box, translation);

      expect(transformed.min).toEqual({ x: 5, y: 10, z: 15 });
      expect(transformed.max).toEqual({ x: 6, y: 11, z: 16 });
    });

    it('should handle scaling', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Scale matrix (scale by 2, 3, 4)
      const scale = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];

      const transformed = transformBoundingBox(box, scale);

      expect(transformed.min).toEqual({ x: -2, y: -3, z: -4 });
      expect(transformed.max).toEqual({ x: 2, y: 3, z: 4 });
    });

    // scene.md C3: missing boundary cases for the perspective-division
    // path. `transformBoundingBox` divides each corner by
    // w = matrix[3]*x + matrix[7]*y + matrix[11]*z + matrix[15], so any
    // matrix with non-zero perspective entries or matrix[15] != 1 changes
    // the result. The tests above only cover identity / translation / scale
    // matrices where w is always 1 — the divide path is unexercised.
    it('handles matrix[15] != 1 (uniform projective scaling: result scales by 1/matrix[15])', () => {
      const box = { min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } };
      // Identity rotation + non-unit homogeneous w; should scale by 1/2.
      const projective = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2];
      const result = transformBoundingBox(box, projective);
      expect(result.min).toEqual({ x: -0.5, y: -0.5, z: -0.5 });
      expect(result.max).toEqual({ x: 0.5, y: 0.5, z: 0.5 });
    });

    // Bug fix: a perspective-projection matrix can produce w == 0 for points
    // on the camera plane, which silently yielded Infinity/NaN coordinates
    // before the epsilon guard was added. The result must always be finite.
    it('returns a finite bounding box for a perspective projection that crosses the camera plane', () => {
      // Box that straddles the camera plane (z spans -1..1).
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };
      // THREE.js-style perspective matrix (column-major, flat). The key
      // property: matrix[11] = -1 (so w = -z), which makes corners with z=0
      // produce w=0 → division by zero. We use a representative perspective
      // matrix (fov ~90°, aspect 1, near 0.1, far 100):
      //   m[0]=1, m[5]=1, m[10]≈-1.002, m[11]=-1, m[14]≈-0.2, m[15]=0.
      // Any corner with z=0 produces w = -1*0 + 0 = 0.
      // We pick a box with z=0 on one face by translating.
      const boxOnPlane: BoundingBox = {
        min: { x: -1, y: -1, z: 0 },
        max: { x: 1, y: 1, z: 2 },
      };
      const perspective = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2002, 0];
      const result = transformBoundingBox(boxOnPlane, perspective);
      // All six floats must be finite — no NaN, no ±Infinity.
      for (const v of [
        result.min.x,
        result.min.y,
        result.min.z,
        result.max.x,
        result.max.y,
        result.max.z,
      ]) {
        expect(Number.isFinite(v)).toBe(true);
      }
      // And the original-box case (no z=0 corner) — make sure we didn't
      // regress finite behaviour for sane perspective inputs.
      const safeResult = transformBoundingBox(box, perspective);
      for (const v of [
        safeResult.min.x,
        safeResult.min.y,
        safeResult.min.z,
        safeResult.max.x,
        safeResult.max.y,
        safeResult.max.z,
      ]) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });

    // Covers the "every corner degenerate" fallback: when the w-row is all
    // zero (matrix[3]=matrix[7]=matrix[11]=matrix[15]=0), every corner has
    // |w| < W_EPSILON and is skipped, so the function returns a copy of the
    // input box rather than (Infinity, -Infinity).
    it('falls back to a copy of the input box when ALL corners are degenerate (w≈0)', () => {
      const box: BoundingBox = { min: { x: -1, y: -2, z: -3 }, max: { x: 4, y: 5, z: 6 } };
      // w-row (indices 3,7,11,15) all zero → w = 0 for every corner.
      const zeroW = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
      const result = transformBoundingBox(box, zeroW);
      expect(result.min).toEqual(box.min);
      expect(result.max).toEqual(box.max);
      // It must be a copy, not the same object reference (source spreads).
      expect(result.min).not.toBe(box.min);
    });
  });

  describe('projectBoundsToDisplayDims', () => {
    it('maps the first three displayDims to X / Y / Z when all three are present', () => {
      const min = [0, 10, 20, 30];
      const max = [1, 11, 21, 31];
      const box = projectBoundsToDisplayDims(min, max, [0, 1, 2]);
      expect(box.min).toEqual({ x: 0, y: 10, z: 20 });
      expect(box.max).toEqual({ x: 1, y: 11, z: 21 });
    });

    it('reorders bounds when displayDims pick non-leading dimensions', () => {
      const min = [0, 10, 20, 30];
      const max = [1, 11, 21, 31];
      const box = projectBoundsToDisplayDims(min, max, [3, 1, 0]);
      expect(box.min).toEqual({ x: 30, y: 10, z: 0 });
      expect(box.max).toEqual({ x: 31, y: 11, z: 1 });
    });

    it('leaves Z at 0 when fewer than 3 displayDims are supplied', () => {
      const box = projectBoundsToDisplayDims([0, 10], [1, 11], [0, 1]);
      expect(box.min).toEqual({ x: 0, y: 10, z: 0 });
      expect(box.max).toEqual({ x: 1, y: 11, z: 0 });
    });

    it('leaves an axis at 0 when its displayDim is out of range (defensive)', () => {
      // displayDims[1] = 99 → out of range, so y stays 0.
      const box = projectBoundsToDisplayDims([0, 10], [1, 11], [0, 99, 1]);
      expect(box.min).toEqual({ x: 0, y: 0, z: 10 });
      expect(box.max).toEqual({ x: 1, y: 0, z: 11 });
    });

    it('returns the all-zero default for empty displayDims', () => {
      const box = projectBoundsToDisplayDims([0, 10, 20], [1, 11, 21], []);
      expect(box.min).toEqual({ x: 0, y: 0, z: 0 });
      expect(box.max).toEqual({ x: 0, y: 0, z: 0 });
    });

    it('caps at 3 axes even when more displayDims are supplied', () => {
      const min = [0, 10, 20, 30, 40];
      const max = [1, 11, 21, 31, 41];
      const box = projectBoundsToDisplayDims(min, max, [0, 1, 2, 3, 4]);
      // Only first 3 used; 4th and 5th are ignored.
      expect(box.min).toEqual({ x: 0, y: 10, z: 20 });
      expect(box.max).toEqual({ x: 1, y: 11, z: 21 });
    });

    // W5: negative and very large finite bounds map through unchanged.
    it('preserves negative and large-magnitude bounds verbatim', () => {
      const min = [-1e6, -5, 1e6];
      const max = [0, 1, 2e6];
      const box = projectBoundsToDisplayDims(min, max, [0, 1, 2]);
      expect(box.min).toEqual({ x: -1e6, y: -5, z: 1e6 });
      expect(box.max).toEqual({ x: 0, y: 1, z: 2e6 });
    });

    // W5: repeated indices map the same source dimension onto multiple axes.
    it('maps a repeated displayDim onto every axis that references it', () => {
      const min = [7, 100];
      const max = [9, 200];
      const box = projectBoundsToDisplayDims(min, max, [0, 0, 1]);
      expect(box.min).toEqual({ x: 7, y: 7, z: 100 });
      expect(box.max).toEqual({ x: 9, y: 9, z: 200 });
    });
  });

  // G3: the box helpers assume min <= max. Passing an inverted box is not
  // guarded; the size/diagonal helpers return signed/garbage values. Document
  // the invariant so callers know inverted input is their responsibility.
  describe('inverted box (min > max) — documented unsafe behaviour', () => {
    const inverted: BoundingBox = { min: { x: 10, y: 10, z: 10 }, max: { x: 0, y: 0, z: 0 } };

    it('getBoundingBoxSize returns negative extents for an inverted box', () => {
      expect(getBoundingBoxSize(inverted)).toEqual({ x: -10, y: -10, z: -10 });
    });

    it('getBoundingBoxMaxDimension returns a negative value for an inverted box', () => {
      expect(getBoundingBoxMaxDimension(inverted)).toBe(-10);
    });

    it('getBoundingBoxDiagonal still returns a positive magnitude (squares cancel the sign)', () => {
      // sqrt((-10)^2 * 3) = sqrt(300) ≈ 17.32 — diagonal is sign-agnostic.
      expect(getBoundingBoxDiagonal(inverted)).toBeCloseTo(Math.sqrt(300), 6);
    });
  });
});
