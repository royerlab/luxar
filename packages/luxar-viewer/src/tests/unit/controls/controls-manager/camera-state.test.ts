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
  const ctx: CameraStateCtx = {
    camera,
    currentControls: null,
    sceneScale: 0,
    minPivotDepth: 0,
    savedCameraPosition: new THREE.Vector3(),
    savedCameraRotation: new THREE.Euler(),
    savedCameraUp: new THREE.Vector3(0, 1, 0),
    savedTarget: new THREE.Vector3(),
    ...overrides,
  };
  // Default mirrors ControlsManager.minPivotDepth() with no auto-frame limits.
  if (overrides.minPivotDepth === undefined) {
    ctx.minPivotDepth = (ctx.sceneScale || 10) * 1e-3;
  }
  return ctx;
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

  it('reuses the previous pivot depth: no movement returns the old target exactly (fly)', () => {
    // #774: with no fly movement the fly derivation projects the PREVIOUS
    // pivot onto the current view ray, so the saved target is unchanged.
    // currentControls=null mimics first-call or fly mode for the save path.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0); // forward = -Z, looking straight at the old pivot
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 20 });
    ctx.savedTarget.set(0, 0, 0); // previous pivot at the origin
    saveCameraState(ctx);

    // Projected depth = 5 → (0,0,5) + (0,0,-1)*5 = (0,0,0), the old target.
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(0, 5);
  });

  it('falls back to sceneScale depth when the camera looks away from the old pivot (dot <= 0)', () => {
    // The old pivot is behind the camera along the view ray, so its depth is
    // negative → the derivation falls back to walking sceneScale forward.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 10); // forward = +Z, away from the origin pivot
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 20 });
    ctx.savedTarget.set(0, 0, 0); // pivot now behind the camera
    saveCameraState(ctx);

    // Fallback: position + forward * sceneScale = (0,0,5) + (0,0,1)*20 = (0,0,25).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(25, 5);
  });

  it('uses the 10-unit fallback when sceneScale = 0 and looking away (boundary)', () => {
    // P5 boundary: sceneScale === 0 → fallback distance is hardcoded to 10.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 10); // forward = +Z, away from origin → fallback path
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 0 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    // position + forward * 10 = (0,0,5) + (0,0,1)*10 = (0,0,15).
    expect(ctx.savedTarget.z).toBeCloseTo(15, 5);
  });

  it('cone rule: reject band 0 < d < 0.5·dist (~76° off) still falls back to sceneScale', () => {
    // A small POSITIVE depth (old pivot ahead but well outside the ~60° cone)
    // must be REJECTED — reusing it would collapse the pivot toward the camera.
    // This kills a mutant that relaxes `d > 0.5*dist` to `d > 0`.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -1); // forward = -Z
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 10 });
    // toOld = (4,0,-1); dist = sqrt(17) ≈ 4.123; d = toOld·(0,0,-1) = 1 (~76°).
    ctx.savedTarget.set(4, 0, -1);
    saveCameraState(ctx);

    // Fallback (NOT the reused depth 1, which would give (0,0,-1)):
    // pos + forward * sceneScale = (0,0,0) + (0,0,-1)*10 = (0,0,-10).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(-10, 5);
  });

  it('cone rule: camera ~90° off the old pivot falls back to sceneScale depth', () => {
    // The old pivot is perpendicular to the view ray (d == 0, inside neither
    // the >60° reuse cone nor "behind"). The cone rule rejects it — a small
    // positive depth must NOT collapse the pivot onto the camera.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0); // forward = -Z
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 30 });
    ctx.savedTarget.set(5, 0, 5); // to the side, same depth → d = 0 (perpendicular)
    saveCameraState(ctx);

    // Fallback: position + forward * sceneScale = (0,0,5) + (0,0,-1)*30 = (0,0,-25).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(-25, 5);
  });

  it('scale-free: tiny sub-micro-unit scene, no movement reuses the exact old pivot', () => {
    // Absolute-epsilon rules break here (every legit depth is < 1e-6). The
    // scale-free cone test still reuses the old pivot exactly. With the OLD
    // `d > 1e-6` rule this depth (5e-7) would be rejected and the pivot pushed
    // to a sceneScale-ahead point — this test discriminates that regression.
    const camera = new THREE.PerspectiveCamera(60, 1, 1e-9, 1);
    camera.position.set(0, 0, 5e-7);
    camera.lookAt(0, 0, 0); // still facing the old pivot at the origin
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 1e-6 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    // Reused depth = 5e-7 → target is the origin (the old pivot), exactly.
    expect(ctx.savedTarget.x).toBeCloseTo(0, 9);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 9);
    expect(ctx.savedTarget.z).toBeCloseTo(0, 9);
  });

  it('no movement with 0 < depth < sceneScale·1e-3 reuses the old pivot exactly', () => {
    // Auto-frame distance limits can legitimately put the orbit distance
    // below the scale-derived sceneScale·1e-3 floor (fit distance /
    // ZOOM_IN_FACTOR shrinks with wide FOVs). The floor is minPivotDepth —
    // the actual legal minimum — so such a pivot must round-trip exactly.
    // A hardcoded sceneScale·1e-3 floor (= 1 here) would push it to depth 1.
    const camera = new THREE.PerspectiveCamera(60, 1, 1e-4, 1e4);
    camera.position.set(0, 0, 0.5); // 0.5 units from the pivot; scale = 1000
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 1000, minPivotDepth: 0.05 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    expect(ctx.savedTarget.x).toBeCloseTo(0, 6);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 6);
    expect(ctx.savedTarget.z).toBeCloseTo(0, 6);
  });

  it('floors the reused depth at minPivotDepth when flown right up to the pivot', () => {
    // Camera flown to 1e-4 in front of the old pivot (still aligned): the
    // reused depth must not collapse below the orbit system's legal minimum,
    // or the next ortho swap frames a ~0-height frustum.
    const camera = new THREE.PerspectiveCamera(60, 1, 1e-4, 1e4);
    camera.position.set(0, 0, 1e-4);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();

    const ctx = makeCtx({ camera, currentControls: null, sceneScale: 1000, minPivotDepth: 1 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    // Depth floored to 1 → target = (0,0,1e-4) + (0,0,-1)·1 ≈ (0,0,-1).
    expect(ctx.savedTarget.z).toBeCloseTo(1e-4 - 1, 6);
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
  it('[G25] with LuxarFlyControls instance, target is derived like the null branch', () => {
    // Looking away from the old pivot exercises the sceneScale fallback via a
    // real fly instance (the LuxarOrbitControls instanceof check is false).
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 10); // forward = +Z, away from origin
    camera.updateMatrixWorld();
    const domElement = document.createElement('div');
    const fly = new LuxarFlyControls(camera, domElement);

    const ctx = makeCtx({ camera, currentControls: fly, sceneScale: 25 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    // Fallback: position + forward * sceneScale = (0,0,5) + (0,0,1)*25 = (0,0,30).
    expect(ctx.savedTarget.x).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.y).toBeCloseTo(0, 5);
    expect(ctx.savedTarget.z).toBeCloseTo(30, 5);

    fly.dispose();
  });

  it('[G25] with LuxarFlyControls, sceneScale === 0 falls back to default scale of 10 (||-fallback)', () => {
    // The source uses `ctx.sceneScale || 10` for the fly fallback.
    // Pin this so a `??` mutation (which would let 0 through) would fail.
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 10); // forward = +Z, away from origin → fallback path
    camera.updateMatrixWorld();
    const domElement = document.createElement('div');
    const fly = new LuxarFlyControls(camera, domElement);

    const ctx = makeCtx({ camera, currentControls: fly, sceneScale: 0 });
    ctx.savedTarget.set(0, 0, 0);
    saveCameraState(ctx);

    // Expected: position + forward * 10 = (0,0,5) + (0,0,1)*10 = (0,0,15).
    expect(ctx.savedTarget.z).toBeCloseTo(15, 5);
    fly.dispose();
  });
});
