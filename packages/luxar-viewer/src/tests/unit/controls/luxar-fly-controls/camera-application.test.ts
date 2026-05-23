/**
 * Unit tests for luxar-fly-controls/camera-application.ts.
 *
 * Targets audit finding G17 (lookAtSmooth, initializeFromCamera,
 * updateOrientation — `smoothness=1` and intermediate slerp invariants
 * untested) and the P8 symmetry with orbit-controls/camera-application
 * (fly camera-application is the simpler quaternion-copy path).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  initializeFromCamera,
  updateOrientation,
  lookAtSmooth,
} from '../../../../controls/luxar-fly-controls/camera-application';

describe('initializeFromCamera (fly)', () => {
  it('copies camera.quaternion into orientation verbatim', () => {
    // Symmetric to orbit-controls camera-application.test.ts but simpler
    // (no gimbal-lock branching; fly just copies the quaternion).
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 4);
    const orientation = new THREE.Quaternion();
    initializeFromCamera(camera, orientation);
    expect(orientation.x).toBeCloseTo(camera.quaternion.x, 8);
    expect(orientation.y).toBeCloseTo(camera.quaternion.y, 8);
    expect(orientation.z).toBeCloseTo(camera.quaternion.z, 8);
    expect(orientation.w).toBeCloseTo(camera.quaternion.w, 8);
  });

  it('handles identity-quaternion camera (boundary)', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const orientation = new THREE.Quaternion(0.1, 0.2, 0.3, 0.4);
    initializeFromCamera(camera, orientation);
    expect(orientation.equals(new THREE.Quaternion())).toBe(true);
  });
});

describe('updateOrientation (fly)', () => {
  it('copies orientation into camera.quaternion', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const orientation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI / 3
    );
    updateOrientation(camera, orientation);
    expect(camera.quaternion.x).toBeCloseTo(orientation.x, 8);
    expect(camera.quaternion.y).toBeCloseTo(orientation.y, 8);
    expect(camera.quaternion.z).toBeCloseTo(orientation.z, 8);
    expect(camera.quaternion.w).toBeCloseTo(orientation.w, 8);
  });

  it('keeps camera.up in sync = (0,1,0).applyQuaternion(orientation)', () => {
    // Required so non-screenSpacePanning pan + state export work
    // correctly when switching out of fly mode.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    // 90° around Z: (0,1,0) → (-1,0,0).
    const orientation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      Math.PI / 2
    );
    updateOrientation(camera, orientation);
    expect(camera.up.x).toBeCloseTo(-1, 5);
    expect(camera.up.y).toBeCloseTo(0, 5);
    expect(camera.up.z).toBeCloseTo(0, 5);
  });
});

describe('lookAtSmooth (fly)', () => {
  it('smoothness=0 snaps to look-at orientation exactly (G17 boundary)', () => {
    // P5 boundary: smoothness=0 → slerp(start, target, 1) = target.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const orientation = camera.quaternion.clone();

    const target = new THREE.Vector3(10, 0, 5); // straight right
    lookAtSmooth(camera, orientation, target, 0);

    // Camera forward should point exactly at target.
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const expected = target.clone().sub(camera.position).normalize();
    expect(forward.x).toBeCloseTo(expected.x, 5);
    expect(forward.y).toBeCloseTo(expected.y, 5);
    expect(forward.z).toBeCloseTo(expected.z, 5);
  });

  it('smoothness=1 keeps orientation unchanged (boundary)', () => {
    // P5 boundary: smoothness=1 → slerp(start, target, 0) = start.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const startQuat = camera.quaternion.clone();
    const orientation = camera.quaternion.clone();

    lookAtSmooth(camera, orientation, new THREE.Vector3(10, 0, 0), 1);

    // Quaternion unchanged.
    const dot = Math.abs(orientation.dot(startQuat));
    expect(dot).toBeCloseTo(1, 5);
  });

  it('intermediate smoothness produces a slerp partway to target (G17)', () => {
    // smoothness=0.5 → slerp(start, target, 0.5).
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const orientation = camera.quaternion.clone();

    const targetPos = new THREE.Vector3(10, 0, 5);
    // Compute the expected lookAt quaternion exactly as the helper does.
    const lookMat = new THREE.Matrix4().lookAt(
      camera.position,
      targetPos,
      new THREE.Vector3(0, 1, 0)
    );
    const targetQuat = new THREE.Quaternion().setFromRotationMatrix(lookMat);

    lookAtSmooth(camera, orientation, targetPos, 0.5);

    const expectedHalf = camera.quaternion.clone(); // saved after the operation
    // Re-compute the expected slerp from the pristine startQuat.
    const startQuat = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    startQuat.position.set(0, 0, 5);
    startQuat.lookAt(0, 0, 0);
    startQuat.updateMatrixWorld();
    const expected = startQuat.quaternion.clone().slerp(targetQuat, 0.5);
    const dot = Math.abs(expectedHalf.dot(expected));
    expect(dot).toBeCloseTo(1, 5);
  });

  it('camera.up tracks orientation after smooth lookAt (P8 symmetry contract)', () => {
    // Same contract as updateOrientation — fly side keeps camera.up in sync.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const orientation = camera.quaternion.clone();

    lookAtSmooth(camera, orientation, new THREE.Vector3(1, 0, 0), 0);

    const expectedUp = new THREE.Vector3(0, 1, 0).applyQuaternion(orientation);
    expect(camera.up.x).toBeCloseTo(expectedUp.x, 5);
    expect(camera.up.y).toBeCloseTo(expectedUp.y, 5);
    expect(camera.up.z).toBeCloseTo(expectedUp.z, 5);
  });
});
