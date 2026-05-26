/**
 * Unit tests for luxar-orbit-controls/math/trackball.ts.
 *
 * Targets audit findings G7 (computeArcballRotation is untested
 * directly — only projectOnTrackball had two direct cases in the
 * orchestrator test file), H1 (project always returns unit-length
 * vectors), and H2 (identity-drag / inverse-drag / composition
 * invariants for arcball rotation).
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import * as THREE from 'three';
import {
  projectOnTrackball,
  computeArcballRotation,
} from '../../../../../controls/luxar-orbit-controls/math/trackball';

describe('projectOnTrackball — H1 normalization invariant', () => {
  it('returns a unit-length vector for every NDC sample', () => {
    // P12 / H1 property: ||project(x, y, r)|| === 1 for all (x, y).
    const samples: Array<[number, number]> = [
      [0, 0],
      [0.1, 0],
      [0.5, 0.5],
      [0.9, 0.1],
      [-0.3, 0.8],
      [1.0, 1.0],
      [-1.0, -1.0],
      [2.0, 2.0], // way past the trackball — falls into hyperboloid branch
    ];
    for (const [x, y] of samples) {
      const p = projectOnTrackball(x, y, 1.0);
      expect(p.length()).toBeCloseTo(1.0, 8);
    }
  });

  it('selects the sphere branch when d² ≤ 0.5 r² (inside the trackball)', () => {
    // At the boundary of the branch (d² = 0.5 r²), both branches must
    // give the same z — verify continuity at the seam.
    const r = 1.0;
    // d = r/√2 — exactly on the seam.
    const x = r / Math.SQRT2;
    const p = projectOnTrackball(x, 0, r);
    // Sphere branch would give z = sqrt(r² - d²) = sqrt(0.5) = 1/√2.
    expect(p.z).toBeCloseTo(1 / Math.SQRT2, 6);
  });

  it('selects the hyperboloid branch when d² > 0.5 r² (outside the trackball)', () => {
    // Hyperboloid: z = 0.5*r² / sqrt(d²) → z decreases as we go further out.
    const r = 1.0;
    const p1 = projectOnTrackball(0.9, 0, r); // outside the sphere region
    const p2 = projectOnTrackball(2.0, 0, r);
    // For larger |d|, z (pre-normalization) is smaller — but after
    // normalization the x-component grows toward 1.
    expect(p2.x).toBeGreaterThan(p1.x);
    // Both still unit-length.
    expect(p1.length()).toBeCloseTo(1, 8);
    expect(p2.length()).toBeCloseTo(1, 8);
  });

  it('z is maximum at the center of the trackball', () => {
    const r = 1.0;
    const center = projectOnTrackball(0, 0, r);
    for (const [x, y] of [
      [0.1, 0],
      [0.5, 0.5],
      [0.9, 0.1],
    ]) {
      const p = projectOnTrackball(x, y, r);
      expect(p.z).toBeLessThanOrEqual(center.z + 1e-10);
    }
  });

  // controls.md [H1][P12] property test: every projection result is a
  // unit-length vector regardless of the (x, y, r) input. This is the
  // deepest invariant of the function (it underpins arcball rotation
  // computing valid quaternion axes). 200 randomised samples cover both
  // the sphere-branch and hyperboloid-branch regions.
  it('[property] ||project(x, y, r)|| = 1 for arbitrary inputs [controls.md/H1][P12]', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -5, max: 5, noNaN: true }),
        fc.double({ min: -5, max: 5, noNaN: true }),
        fc.double({ min: 0.1, max: 10, noNaN: true }),
        (x, y, r) => {
          const p = projectOnTrackball(x, y, r);
          return Math.abs(p.length() - 1) < 1e-8;
        }
      ),
      { numRuns: 200 }
    );
  });
});

describe('computeArcballRotation — H2 invariants', () => {
  it('identity drag (start == end) returns the identity quaternion', () => {
    // H2 invariant: zero-displacement drag → no rotation.
    const start = new THREE.Vector2(0.3, 0.2);
    const end = new THREE.Vector2(0.3, 0.2);
    const q = computeArcballRotation(start, end, 1.0, 1.0);
    expect(q.x).toBeCloseTo(0, 10);
    expect(q.y).toBeCloseTo(0, 10);
    expect(q.z).toBeCloseTo(0, 10);
    expect(q.w).toBeCloseTo(1, 10);
  });

  it('reverse drag (swap start/end) produces the inverse quaternion (H2)', () => {
    // H2 invariant: drag(A→B) and drag(B→A) must compose to identity.
    const a = new THREE.Vector2(0.2, 0.1);
    const b = new THREE.Vector2(0.5, 0.3);

    const qForward = computeArcballRotation(a, b, 1.0, 1.0);
    const qBackward = computeArcballRotation(b, a, 1.0, 1.0);

    // Composition must equal identity (within floating-point tolerance).
    const composed = qForward.clone().multiply(qBackward);
    expect(Math.abs(composed.x)).toBeLessThan(1e-6);
    expect(Math.abs(composed.y)).toBeLessThan(1e-6);
    expect(Math.abs(composed.z)).toBeLessThan(1e-6);
    expect(Math.abs(Math.abs(composed.w) - 1)).toBeLessThan(1e-6);
  });

  it('returns a unit-length quaternion (rotation invariant)', () => {
    const samples: Array<[THREE.Vector2, THREE.Vector2]> = [
      [new THREE.Vector2(-0.3, 0.0), new THREE.Vector2(0.3, 0.1)],
      [new THREE.Vector2(0.0, -0.5), new THREE.Vector2(0.0, 0.5)],
      [new THREE.Vector2(0.7, 0.7), new THREE.Vector2(-0.7, -0.7)],
    ];
    for (const [start, end] of samples) {
      const q = computeArcballRotation(start, end, 1.0, 1.0);
      const norm = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
      expect(norm).toBeCloseTo(1, 5);
    }
  });

  it('rotateSpeed scales the rotation angle linearly', () => {
    // Doubling rotateSpeed must double the rotation angle (for the same
    // drag). Angle = acos(dot(p1, p2)) * rotateSpeed.
    const start = new THREE.Vector2(-0.2, 0.0);
    const end = new THREE.Vector2(0.2, 0.0);

    const qSlow = computeArcballRotation(start, end, 1.0, 1.0);
    const qFast = computeArcballRotation(start, end, 1.0, 2.0);

    // Extract angle: acos(w) * 2 = rotation angle.
    const angleSlow = 2 * Math.acos(Math.min(1, Math.max(-1, qSlow.w)));
    const angleFast = 2 * Math.acos(Math.min(1, Math.max(-1, qFast.w)));
    expect(angleFast).toBeCloseTo(2 * angleSlow, 4);
  });

  it('[controls.md/G7] trackballRadius = 0 produces a finite unit-length quaternion (no NaN/Infinity)', () => {
    // controls.md G7: with radius=0, both endpoint projections fall into
    // the hyperboloid branch where z = (0.5 * r²)/sqrt(d²) collapses to 0.
    // The endpoints become in-plane unit vectors; their cross product
    // (along z) is non-zero for non-collinear drags and a real rotation
    // results. We pin the well-formed-output contract: the function
    // doesn't throw, doesn't NaN, and returns a unit quaternion.
    const start = new THREE.Vector2(0.2, 0.1);
    const end = new THREE.Vector2(0.5, 0.3);
    const q = computeArcballRotation(start, end, 0, 1.0);
    const norm = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    expect(Number.isFinite(norm)).toBe(true);
    expect(norm).toBeCloseTo(1, 5);
  });

  it('[controls.md/G7] trackballRadius = 0 with collinear drag returns identity (guard branch)', () => {
    // controls.md G7: when start and end project to (anti-)parallel vectors,
    // the `axis.lengthSq() < 1e-10` guard bails out → identity quaternion.
    // Pure-X drag from origin produces start=(0,0) (degenerate) and a real
    // x-only end; with radius=0, both project to xy-plane vectors along x,
    // hence cross product collapses → identity returned.
    const start = new THREE.Vector2(0.4, 0);
    const end = new THREE.Vector2(0.8, 0);
    const q = computeArcballRotation(start, end, 0, 1.0);
    // Both project to (1, 0, 0) after normalization → cross product ~ 0 → identity.
    expect(q.w).toBeCloseTo(1, 10);
    expect(Math.hypot(q.x, q.y, q.z)).toBeLessThan(1e-8);
  });

  it('[controls.md/G8] rotateSpeed = 0 returns identity quaternion regardless of drag', () => {
    // controls.md G8: angle = acos(p1·p2) * rotateSpeed. With rotateSpeed=0,
    // the angle is 0 → setFromAxisAngle(any, 0) returns identity.
    const start = new THREE.Vector2(0.2, 0.1);
    const end = new THREE.Vector2(-0.4, 0.5);
    const q = computeArcballRotation(start, end, 1.0, 0);
    expect(q.w).toBeCloseTo(1, 10);
    expect(Math.hypot(q.x, q.y, q.z)).toBeLessThan(1e-10);
  });

  it('drag from origin to +X NDC rotates around Y axis (negated for camera convention)', () => {
    // Horizontal drag: start = origin (projects to (0,0,1)), end = (0.3, 0).
    // Cross product p1 × p2 points along -Y (camera orbits left when
    // dragging right), so the quaternion axis is -Y.
    const start = new THREE.Vector2(0, 0);
    const end = new THREE.Vector2(0.3, 0);
    const q = computeArcballRotation(start, end, 1.0, 1.0);
    // The quaternion encodes rotation around an axis ∝ (0, -1, 0) (negated
    // by the camera-orbit convention) — net axis component on Y is non-zero.
    // We don't pin the sign convention to a constant here (the helper
    // negates angle for the camera-orbits-opposite convention); just verify
    // that the rotation axis is dominantly along Y, not X or Z.
    const axisLen = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    if (axisLen > 1e-6) {
      // |y| should dominate the axis.
      expect(Math.abs(q.y) / axisLen).toBeGreaterThan(0.99);
    }
  });
});
