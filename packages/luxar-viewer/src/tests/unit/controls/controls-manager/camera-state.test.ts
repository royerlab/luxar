/**
 * Unit tests for controls-manager/camera-state.ts.
 *
 * Targets audit finding G3 (saveCameraState / restoreCameraState only
 * smoke-tested via the orchestrator). These helpers own the cross-mode
 * target persistence, so direct testing matters.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  saveCameraState,
  restoreCameraState,
  type CameraStateCtx,
} from '../../../../controls/controls-manager/camera-state';
import { LuxarOrbitControls } from '../../../../controls/luxar-orbit-controls';
import { LuxarFlyControls } from '../../../../controls/luxar-fly-controls';

function makeCtx(overrides: Partial<CameraStateCtx>): CameraStateCtx {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  return {
    camera,
    currentControls: null,
    sceneScale: 0,
    savedCameraPosition: new THREE.Vector3(),
    savedCameraRotation: new THREE.Euler(),
    savedCameraUp: new THREE.Vector3(0, 1, 0),
    savedTarget: new THREE.Vector3(),
    ...overrides,
  };
}

describe('saveCameraState', () => {
  it('copies camera position, rotation, and up into the saved accumulators', () => {
    const ctx = makeCtx({});
    ctx.camera.position.set(7, 8, 9);
    ctx.camera.rotation.set(0.1, 0.2, 0.3);
    ctx.camera.up.set(0, 0, 1); // non-default

    saveCameraState(ctx);

    expect(ctx.savedCameraPosition.x).toBeCloseTo(7, 5);
    expect(ctx.savedCameraPosition.y).toBeCloseTo(8, 5);
    expect(ctx.savedCameraPosition.z).toBeCloseTo(9, 5);
    expect(ctx.savedCameraRotation.x).toBeCloseTo(0.1, 5);
    expect(ctx.savedCameraRotation.y).toBeCloseTo(0.2, 5);
    expect(ctx.savedCameraRotation.z).toBeCloseTo(0.3, 5);
    expect(ctx.savedCameraUp.x).toBeCloseTo(0, 5);
    expect(ctx.savedCameraUp.y).toBeCloseTo(0, 5);
    expect(ctx.savedCameraUp.z).toBeCloseTo(1, 5);
  });

  it('uses orbit controls target when current controls is orbit', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    const domElement = document.createElement('div');
    const orbitControls = new LuxarOrbitControls(camera, domElement);
    orbitControls.target.set(1, 2, 3);

    const ctx = makeCtx({ camera, currentControls: orbitControls });
    saveCameraState(ctx);

    expect(ctx.savedTarget.x).toBeCloseTo(1, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(2, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(3, 5);

    orbitControls.dispose();
  });

  it('derives target from camera direction + sceneScale when controls is not orbit (fly fallback)', () => {
    // currentControls=null mimics first-call or fly mode for the save path.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0); // forward = -Z
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 20 });
    saveCameraState(ctx);

    // Expected: position + forward * sceneScale = (0,0,5) + (0,0,-1)*20 = (0,0,-15).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(-15, 5);
  });

  it('uses the 10-unit fallback when sceneScale = 0 (boundary)', () => {
    // P5 boundary: sceneScale === 0 → fallback distance is hardcoded to 10.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 0 });
    saveCameraState(ctx);

    expect(ctx.savedTarget.z).toBeCloseTo(5 - 10, 5);
  });
});

describe('restoreCameraState', () => {
  it('copies savedTarget into orbit controls and reinitializes orientation', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const domElement = document.createElement('div');
    const orbitControls = new LuxarOrbitControls(camera, domElement);

    const ctx = makeCtx({ camera, currentControls: orbitControls });
    ctx.savedTarget.set(2, 3, 4);

    restoreCameraState(ctx);

    expect(orbitControls.target.x).toBeCloseTo(2, 5);
    expect(orbitControls.target.y).toBeCloseTo(3, 5);
    expect(orbitControls.target.z).toBeCloseTo(4, 5);

    orbitControls.dispose();
  });

  it('is a no-op when currentControls is null (saved fields untouched)', () => {
    // controls.md [G3][P10] strengthening: was `.not.toThrow()` only.
    // Pin the no-op contract: the saved* slots on the ctx are untouched,
    // and the camera state (position/rotation) is not modified either.
    const ctx = makeCtx({});
    ctx.savedTarget.set(2, 3, 4);
    const savedTargetBefore = ctx.savedTarget.clone();
    const cameraPosBefore = ctx.camera.position.clone();
    const cameraQuatBefore = ctx.camera.quaternion.clone();
    restoreCameraState(ctx);
    // savedTarget remains exactly what we set.
    expect(ctx.savedTarget.x).toBe(savedTargetBefore.x);
    expect(ctx.savedTarget.y).toBe(savedTargetBefore.y);
    expect(ctx.savedTarget.z).toBe(savedTargetBefore.z);
    // Camera is untouched (the fallback path bails before mutating).
    expect(ctx.camera.position.equals(cameraPosBefore)).toBe(true);
    expect(ctx.camera.quaternion.equals(cameraQuatBefore)).toBe(true);
  });
});

describe('saveCameraState — fly counterpart [controls.md G25]', () => {
  // controls.md G25[P5][P8]: prior coverage tested `currentControls=null`
  // (which exercises the fallback derivation branch). The `LuxarFlyControls`
  // branch traverses the same code path — `instanceof LuxarOrbitControls`
  // is false — but the test wiring (with a real fly instance) was missing
  // for symmetry. Pin it explicitly so a future refactor that introduced
  // a dedicated fly branch (or broke the fallback assumption) would surface.
  it('[G25] with LuxarFlyControls instance, target is derived from camera direction (same as null)', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0); // forward = -Z
    camera.updateMatrixWorld();
    const domElement = document.createElement('div');
    const fly = new LuxarFlyControls(camera, domElement);

    const ctx = makeCtx({ camera, currentControls: fly, sceneScale: 25 });
    saveCameraState(ctx);

    // Expected: position + forward * sceneScale = (0,0,5) + (0,0,-1)*25 = (0,0,-20).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(-20, 5);

    fly.dispose();
  });

  it('[G25] with LuxarFlyControls, sceneScale === 0 falls back to default scale of 10 (||-fallback)', () => {
    // The source uses `ctx.sceneScale || 10` for the fly fallback.
    // Pin this so a `??` mutation (which would let 0 through) would fail.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const domElement = document.createElement('div');
    const fly = new LuxarFlyControls(camera, domElement);

    const ctx = makeCtx({ camera, currentControls: fly, sceneScale: 0 });
    saveCameraState(ctx);

    // Expected: position + forward * 10 = (0,0,5) + (0,0,-1)*10 = (0,0,-5).
    expect(ctx.savedTarget.z).toBeCloseTo(-5, 5);
    fly.dispose();
  });
});
