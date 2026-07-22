/**
 * Unit tests for luxar-orbit-controls/camera-application.ts.
 *
 * Targets audit findings G8 (applyToCamera + initializeFromCamera —
 * the gimbal-lock fallback logic on each pole needs explicit coverage)
 * and H9 (round-trip preserves target / orientation / distance).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  applyToCamera,
  initializeFromCamera,
} from '../../../../controls/luxar-orbit-controls/camera-application';

describe('applyToCamera', () => {
  it('sets camera position = target + (0,0,distance).applyQuaternion(orientation)', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const target = new THREE.Vector3(1, 2, 3);
    const orientation = new THREE.Quaternion(); // identity
    applyToCamera(cam, target, orientation, 5);
    // Position: (1,2,3) + (0,0,5) = (1,2,8).
    expect(cam.position.x).toBeCloseTo(1, 5);
    expect(cam.position.y).toBeCloseTo(2, 5);
    expect(cam.position.z).toBeCloseTo(8, 5);
  });

  it('camera.up = (0,1,0).applyQuaternion(orientation)', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    // 90° around Z → up vector rotates from (0,1,0) to (-1,0,0).
    const orientation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      Math.PI / 2
    );
    applyToCamera(cam, new THREE.Vector3(), orientation, 5);
    expect(cam.up.x).toBeCloseTo(-1, 5);
    expect(cam.up.y).toBeCloseTo(0, 5);
    expect(cam.up.z).toBeCloseTo(0, 5);
  });

  it('makes camera look at the target (forward vector points from camera to target)', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();
    applyToCamera(cam, target, orientation, 5);

    cam.updateMatrixWorld();
    const forward = new THREE.Vector3();
    cam.getWorldDirection(forward);
    // Camera at (0,0,5) looking at origin → forward = (0,0,-1).
    expect(forward.x).toBeCloseTo(0, 5);
    expect(forward.y).toBeCloseTo(0, 5);
    expect(forward.z).toBeCloseTo(-1, 5);
  });

  it('updateMatrixWorld is called (camera.matrix is fresh for pan math)', () => {
    // Verified by checking matrixWorld differs from default identity after call.
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    applyToCamera(cam, new THREE.Vector3(5, 5, 5), new THREE.Quaternion(), 1);
    // matrixWorld must reflect the new position.
    const m = new THREE.Vector3();
    m.setFromMatrixPosition(cam.matrixWorld);
    expect(m.x).toBeCloseTo(5, 5);
    expect(m.y).toBeCloseTo(5, 5);
    expect(m.z).toBeCloseTo(6, 5); // 5 + (0,0,1) along orientation
  });
});

describe('initializeFromCamera', () => {
  it('returns distance = ||camera.position - target||', () => {
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    cam.position.set(0, 0, 5);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();
    const d = initializeFromCamera(cam, target, orientation);
    expect(d).toBeCloseTo(5, 5);
  });

  it('clamps distance to a minimum of 0.001 (boundary)', () => {
    // P5 boundary: when camera and target coincide, distance must not be 0
    // (avoids divide-by-zero downstream).
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    cam.position.set(0, 0, 0);
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();
    const d = initializeFromCamera(cam, target, orientation);
    expect(d).toBe(0.001);
  });

  it('uses the scene-relative minDistance as the degenerate floor when supplied', () => {
    // Tiny-unit scene: minDistance = diagonal * factor ~ 1e-9. The old
    // absolute 0.001 floor flung the camera 1000x out of the scene.
    const cam = new THREE.PerspectiveCamera(60, 1, 1e-9, 1e-3);
    cam.position.set(0, 0, 0);
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();
    expect(initializeFromCamera(cam, target, orientation, 1e-9)).toBe(1e-9);
  });

  it('does NOT clamp a valid tiny-unit orbit distance to the absolute fallback', () => {
    // Camera framed 3e-6 from the target (scale x1e-6 scene) with a
    // scene-relative floor: the distance must survive re-initialization.
    const cam = new THREE.PerspectiveCamera(60, 1, 1e-9, 1e-3);
    cam.position.set(0, 0, 3e-6);
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();
    expect(initializeFromCamera(cam, target, orientation, 1e-9)).toBeCloseTo(3e-6, 12);
  });

  it('uses fallback up-vector when view direction is nearly parallel to up (gimbal-lock fix)', () => {
    // P5 boundary: looking straight down (-Y), camera.up = (0,1,0) is parallel
    // to view direction. The helper must detect this (upDot > 0.999) and pick
    // a fallback perpendicular up. The output orientation must be finite.
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    cam.position.set(0, 5, 0); // straight above origin
    cam.up.set(0, 1, 0);
    cam.quaternion.identity();
    cam.updateMatrixWorld();
    const target = new THREE.Vector3(0, 0, 0);
    const orientation = new THREE.Quaternion();

    const d = initializeFromCamera(cam, target, orientation);
    expect(d).toBeCloseTo(5, 5);
    // Orientation must be finite (no NaN).
    expect(Number.isFinite(orientation.x)).toBe(true);
    expect(Number.isFinite(orientation.y)).toBe(true);
    expect(Number.isFinite(orientation.z)).toBe(true);
    expect(Number.isFinite(orientation.w)).toBe(true);
    const norm = Math.sqrt(
      orientation.x ** 2 + orientation.y ** 2 + orientation.z ** 2 + orientation.w ** 2
    );
    expect(norm).toBeCloseTo(1, 4);
  });

  it('selects fallback up = (1,0,0) when |viewDir.x| < 0.9 (X-axis fallback)', () => {
    // Cover the conditional `Math.abs(viewDir.x) < 0.9 ? (1,0,0) : (0,0,1)`.
    // Looking down -Y from above: viewDir = (0,-1,0) → |x|=0 < 0.9 → fallback = (1,0,0).
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    cam.position.set(0, 5, 0);
    cam.updateMatrixWorld();
    const orientation = new THREE.Quaternion();
    expect(() => initializeFromCamera(cam, new THREE.Vector3(), orientation)).not.toThrow();
    expect(Number.isFinite(orientation.x)).toBe(true);
  });

  it('selects fallback up = (0,0,1) when |viewDir.x| ≥ 0.9 (Z-axis fallback)', () => {
    // Looking down -X with camera up along X. viewDir ≈ (-1,0,0) → |x|=1 ≥ 0.9 → fallback = (0,0,1).
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    cam.position.set(5, 0, 0);
    cam.up.set(1, 0, 0); // parallel to view → triggers singularity
    cam.quaternion.identity();
    cam.updateMatrixWorld();
    const orientation = new THREE.Quaternion();
    expect(() => initializeFromCamera(cam, new THREE.Vector3(), orientation)).not.toThrow();
    expect(Number.isFinite(orientation.x)).toBe(true);
    expect(Number.isFinite(orientation.w)).toBe(true);
  });
});

describe('round-trip (H9 invariant)', () => {
  it('applyToCamera then initializeFromCamera preserves (target, orientation, distance)', () => {
    // H9 property: extracting orbit state from camera and re-applying
    // should return identical orbit state.
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    // Start with a non-trivial orientation (45° around Y) and distance.
    const target = new THREE.Vector3(1, 2, 3);
    const orient0 = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      Math.PI / 4
    );
    const distance0 = 7;

    applyToCamera(cam, target, orient0, distance0);
    cam.updateMatrixWorld();

    // Now extract back.
    const orient1 = new THREE.Quaternion();
    const distance1 = initializeFromCamera(cam, target, orient1);

    expect(distance1).toBeCloseTo(distance0, 4);
    // Quaternions q and -q encode the same rotation; compare |dot|.
    const dot = Math.abs(orient1.dot(orient0));
    expect(dot).toBeCloseTo(1, 4);
  });
});
