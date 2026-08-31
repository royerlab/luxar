// @vitest-environment jsdom
/**
 * Unit tests for camera-mode helpers used by SceneManager.
 *
 * Verifies the perspective ↔ orthographic swap policy and the
 * setControlType integration that drives it. Event dispatch is
 * deliberately NOT tested here — it stays at the SceneManager
 * call site (covered by scene-manager-events.test.ts).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  type CameraModeCtx,
  setControlType,
  swapToOrthographic,
  swapToPerspective,
} from '../../../../../scene/scene-manager/camera/camera-mode';
import type { ControlsManager } from '../../../../../controls/controls-manager';
import type { PostProcessingManager } from '../../../../../rendering/post-processing/post-processing-manager';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';

function makeRenderer(width = 800, height = 600): Renderer {
  const canvas = document.createElement('canvas');
  Object.defineProperty(canvas, 'clientWidth', { value: width, configurable: true });
  Object.defineProperty(canvas, 'clientHeight', { value: height, configurable: true });
  return { domElement: canvas } as unknown as Renderer;
}

function makeControls(focusTarget = new THREE.Vector3(0, 0, 0)) {
  const setCamera = vi.fn();
  const setControlTypeMock = vi.fn();
  const getFocusTarget = vi.fn(() => focusTarget.clone());
  const returnAutoDollyToBaseline = vi.fn();
  return {
    controls: {
      setCamera,
      setControlType: setControlTypeMock,
      getFocusTarget,
      returnAutoDollyToBaseline,
    } as unknown as ControlsManager,
    setCamera,
    setControlTypeMock,
    returnAutoDollyToBaseline,
  };
}

function makePostProcessing() {
  const setCamera = vi.fn();
  return {
    pp: { setCamera } as unknown as PostProcessingManager,
    setCamera,
  };
}

function makeCtx(
  initialCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
  controlsOverride?: ReturnType<typeof makeControls>['controls']
) {
  let currentCamera: THREE.Camera = initialCamera;
  const made = makeControls();
  const controls = controlsOverride ?? made.controls;
  const { setCamera: controlsSetCamera, setControlTypeMock } = made;
  const { pp, setCamera: ppSetCamera } = makePostProcessing();
  const updateMaterialsForCurrentCamera = vi.fn();
  const setLastOrthoZoom = vi.fn();
  // Backing store so the FOV round-trips through the ctx exactly like the host.
  let lastPerspectiveFov = 47;
  const setLastPerspectiveFov = vi.fn((fov: number) => {
    lastPerspectiveFov = fov;
  });

  const ctx: CameraModeCtx = {
    getCamera: () => currentCamera as CameraModeCtx['getCamera'] extends () => infer R ? R : never,
    setCamera: (cam) => {
      currentCamera = cam;
    },
    controls,
    renderer: makeRenderer(),
    postProcessing: pp,
    updateMaterialsForCurrentCamera,
    setLastOrthoZoom,
    getLastPerspectiveFov: () => lastPerspectiveFov,
    setLastPerspectiveFov,
  };

  return {
    ctx,
    getCurrentCamera: () => currentCamera,
    controlsSetCamera,
    setControlTypeMock,
    ppSetCamera,
    updateMaterialsForCurrentCamera,
    setLastOrthoZoom,
    setLastPerspectiveFov,
    getLastPerspectiveFov: () => lastPerspectiveFov,
  };
}

describe('swapToOrthographic', () => {
  it('floors the frustum distance at the controls SCENE-RELATIVE minDistance (tiny-unit scenes)', () => {
    // Camera sits 1e-7 from the target (a 1e-6-unit scene). The old
    // absolute 0.001 floor inflated the ortho frustum 10,000x on the
    // perspective->ortho swap; the floor must come from the controls'
    // scene-derived minDistance.
    const persp = new THREE.PerspectiveCamera(60, 16 / 9, 1e-9, 1);
    persp.position.set(0, 0, 1e-7);
    const harness = makeCtx(persp);
    (harness.ctx.controls as unknown as { getControls: () => unknown }).getControls = () => ({
      minDistance: 1e-8,
    });

    swapToOrthographic(harness.ctx);

    const ortho = harness.getCurrentCamera() as THREE.OrthographicCamera;
    const frustumHeight = ortho.top - ortho.bottom;
    // 2 * 1e-7 * tan(30 deg) — NOT 2 * 0.001 * tan(30 deg) ≈ 1.15e-3.
    expect(frustumHeight).toBeLessThan(1e-5);
    expect(frustumHeight).toBeGreaterThan(0);
  });

  it('replaces perspective camera with an orthographic instance', () => {
    const persp = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    swapToOrthographic(harness.ctx);

    expect(harness.getCurrentCamera()).toBeInstanceOf(THREE.OrthographicCamera);
    expect(harness.ppSetCamera).toHaveBeenCalledWith(harness.getCurrentCamera());
    expect(harness.setLastOrthoZoom).toHaveBeenCalledTimes(1);
  });

  it('is a no-op when current camera is already orthographic', () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    const harness = makeCtx(ortho);

    swapToOrthographic(harness.ctx);

    expect(harness.getCurrentCamera()).toBe(ortho);
    expect(harness.ppSetCamera).not.toHaveBeenCalled();
  });

  it('matches frustum at current target distance (visible area preserved)', () => {
    const persp = new THREE.PerspectiveCamera(60, 2, 0.1, 1000);
    persp.position.set(0, 0, 100);
    const harness = makeCtx(persp);

    swapToOrthographic(harness.ctx);

    const ortho = harness.getCurrentCamera() as THREE.OrthographicCamera;
    const fovRad = (60 * Math.PI) / 180;
    const expectedHeight = 2 * 100 * Math.tan(fovRad / 2);
    const expectedHalfWidth = (expectedHeight * 2) / 2; // aspect = 2

    expect(ortho.top).toBeCloseTo(expectedHeight / 2, 3);
    expect(ortho.bottom).toBeCloseTo(-expectedHeight / 2, 3);
    expect(ortho.right).toBeCloseTo(expectedHalfWidth, 3);
    expect(ortho.left).toBeCloseTo(-expectedHalfWidth, 3);
  });

  // #774: the swap to ortho is POSE-PRESERVING — it copies the perspective
  // camera's exact position, orientation and up so cycling control modes with
  // no interaction never shifts or re-frames the view. Use a tilted camera
  // with a non-default up so a mutant that reset to a front view is caught.
  it('preserves the perspective pose verbatim (position, quaternion, up)', () => {
    const focus = new THREE.Vector3(3, 4, 0);
    const persp = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    // OFF-AXIS position: x/y differ from focus so the assertion is NOT
    // vacuously equal to the old `focus + distance*+Z` front-view reset.
    persp.position.set(10, -6, 25);
    persp.up.set(1, 0, 0); // deliberately non-default up
    persp.lookAt(focus);
    persp.updateMatrixWorld();
    const quatBefore = persp.quaternion.clone();
    const { controls } = makeControls(focus);
    const harness = makeCtx(persp, controls);

    swapToOrthographic(harness.ctx);

    const ortho = harness.getCurrentCamera() as THREE.OrthographicCamera;
    // Position copied verbatim (NOT reset to focus + distance·+Z).
    expect(ortho.position.x).toBeCloseTo(10, 6);
    expect(ortho.position.y).toBeCloseTo(-6, 6);
    expect(ortho.position.z).toBeCloseTo(25, 6);
    // Orientation + up copied verbatim (NOT reset to a front view / +Y up).
    expect(ortho.quaternion.x).toBeCloseTo(quatBefore.x, 6);
    expect(ortho.quaternion.y).toBeCloseTo(quatBefore.y, 6);
    expect(ortho.quaternion.z).toBeCloseTo(quatBefore.z, 6);
    expect(ortho.quaternion.w).toBeCloseTo(quatBefore.w, 6);
    expect(ortho.up.toArray()).toEqual([1, 0, 0]);
  });

  // #774: the perspective FOV is stashed at swap-in so the inverse swap can
  // restore it exactly instead of falling back to the config default.
  it('stashes the perspective FOV for the inverse swap', () => {
    const persp = new THREE.PerspectiveCamera(73, 1, 0.1, 1000);
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    swapToOrthographic(harness.ctx);

    expect(harness.setLastPerspectiveFov).toHaveBeenCalledWith(73);
    expect(harness.getLastPerspectiveFov()).toBe(73);
  });
});

describe('swapToPerspective', () => {
  it('replaces orthographic camera with a perspective instance', () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    ortho.position.set(0, 0, 50);
    const harness = makeCtx(ortho);

    swapToPerspective(harness.ctx);

    expect(harness.getCurrentCamera()).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(harness.ppSetCamera).toHaveBeenCalledWith(harness.getCurrentCamera());
  });

  it('is a no-op when current camera is already perspective', () => {
    const persp = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    const harness = makeCtx(persp);

    swapToPerspective(harness.ctx);

    expect(harness.getCurrentCamera()).toBe(persp);
    expect(harness.ppSetCamera).not.toHaveBeenCalled();
  });

  it('preserves quaternion, up and near/far from the ortho camera', () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    ortho.position.set(1, 2, 3);
    ortho.up.set(0, 0, 1);
    ortho.quaternion.set(0.1, 0.2, 0.3, 0.9).normalize();
    ortho.updateMatrixWorld();
    const harness = makeCtx(ortho);

    swapToPerspective(harness.ctx);

    const persp = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    expect(persp.up.toArray()).toEqual([0, 0, 1]);
    expect(persp.quaternion.x).toBeCloseTo(ortho.quaternion.x, 6);
    expect(persp.near).toBe(0.1);
    expect(persp.far).toBe(1000);
  });

  it('restores the stashed perspective FOV instead of the config default', () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    ortho.position.set(0, 0, 50);
    const harness = makeCtx(ortho);
    harness.setLastPerspectiveFov(73); // stashed at the earlier persp→ortho swap

    swapToPerspective(harness.ctx);

    const persp = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    expect(persp.fov).toBe(73);
  });

  it('dollies to a degenerate-safe position when the ortho frustum is zero (no NaN)', () => {
    // top == bottom → h == 0 → newDist == 0 → fall back to the ortho position.
    const ortho = new THREE.OrthographicCamera(-10, 10, 0, 0, 0.1, 1000);
    ortho.position.set(1, 2, 3);
    ortho.updateMatrixWorld();
    const harness = makeCtx(ortho);

    swapToPerspective(harness.ctx);

    const persp = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    expect(Number.isFinite(persp.position.x)).toBe(true);
    expect(persp.position.toArray()).toEqual([1, 2, 3]);
  });
});

describe('perspective ↔ orthographic round trip (#774)', () => {
  it('persp → ortho → persp restores an OFF-AXIS pose and NON-default fov (zoom == 1)', () => {
    // A full swap cycle with no interaction must land the camera back at its
    // exact starting pose. Deliberately use a NON-default FOV (33) and an
    // OFF-AXIS tilted pose with a non-(0,1,0) up: the old front-view reset and
    // the old default-FOV restore each fail this independently.
    const focus = new THREE.Vector3(2, -1, 4);
    const persp = new THREE.PerspectiveCamera(33, 1, 0.1, 1000);
    persp.position.set(30, 20, 15);
    persp.up.set(0.2, 0.9, 0.1).normalize();
    persp.lookAt(focus);
    persp.updateMatrixWorld();
    const posBefore = persp.position.clone();
    const quatBefore = persp.quaternion.clone();
    const upBefore = persp.up.clone();

    const { controls } = makeControls(focus);
    const harness = makeCtx(persp, controls);

    swapToOrthographic(harness.ctx); // stashes fov = 33, ortho zoom stays 1
    swapToPerspective(harness.ctx); // restores fov, dollies back

    const persp2 = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    expect(persp2).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(persp2.position.x).toBeCloseTo(posBefore.x, 4);
    expect(persp2.position.y).toBeCloseTo(posBefore.y, 4);
    expect(persp2.position.z).toBeCloseTo(posBefore.z, 4);
    expect(persp2.quaternion.x).toBeCloseTo(quatBefore.x, 5);
    expect(persp2.quaternion.y).toBeCloseTo(quatBefore.y, 5);
    expect(persp2.quaternion.z).toBeCloseTo(quatBefore.z, 5);
    expect(persp2.quaternion.w).toBeCloseTo(quatBefore.w, 5);
    expect(persp2.up.x).toBeCloseTo(upBefore.x, 6);
    expect(persp2.up.y).toBeCloseTo(upBefore.y, 6);
    expect(persp2.up.z).toBeCloseTo(upBefore.z, 6);
    expect(persp2.fov).toBe(33);
  });

  it('dollies by the ortho zoom on swap-back (zoom == 2 → distance halved)', () => {
    // The headline new math: newDist = h/(2·tan(fov/2)) with h=(top-bottom)/zoom.
    // At zoom 2 the effective frustum halves, so the perspective camera must
    // dolly to HALF the original pivot distance, positioned at pivot - viewDir·newDist.
    const focus = new THREE.Vector3(0, 0, 0);
    const persp = new THREE.PerspectiveCamera(47, 1, 0.1, 1000);
    persp.position.set(0, 0, 60); // distance 60 to the pivot
    persp.lookAt(focus);
    persp.updateMatrixWorld();

    const { controls } = makeControls(focus);
    const harness = makeCtx(persp, controls);

    swapToOrthographic(harness.ctx); // ortho at (0,0,60), zoom 1
    const ortho = harness.getCurrentCamera() as THREE.OrthographicCamera;
    ortho.zoom = 2; // user zoomed in 2x in ortho
    ortho.updateProjectionMatrix();

    swapToPerspective(harness.ctx);

    const persp2 = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    const newDist = persp2.position.distanceTo(focus);
    // Original distance 60 → halved to 30 (position - viewDir·30 = (0,0,30)).
    expect(newDist).toBeCloseTo(30, 3);
    expect(persp2.position.x).toBeCloseTo(0, 5);
    expect(persp2.position.y).toBeCloseTo(0, 5);
    expect(persp2.position.z).toBeCloseTo(30, 3);
  });
});

describe('setControlType', () => {
  beforeEach(() => {
    // ensure clean state per test
  });

  it('returns cameraChanged=true when switching to ortho from perspective', () => {
    const persp = new THREE.PerspectiveCamera();
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);
    const before = harness.getCurrentCamera();

    const { cameraChanged } = setControlType('ortho', harness.ctx);

    expect(cameraChanged).toBe(true);
    expect(harness.getCurrentCamera()).toBeInstanceOf(THREE.OrthographicCamera);
    // W1: cameraChanged=true ⟺ the camera object reference actually changed.
    expect(harness.getCurrentCamera()).not.toBe(before);
  });

  it('returns cameraChanged=false when switching control type without projection change', () => {
    const persp = new THREE.PerspectiveCamera();
    const harness = makeCtx(persp);
    const before = harness.getCurrentCamera();

    const { cameraChanged } = setControlType('fly', harness.ctx);

    expect(cameraChanged).toBe(false);
    expect(harness.getCurrentCamera()).toBe(persp);
    // W1: cameraChanged=false ⟺ the same camera reference is retained.
    expect(harness.getCurrentCamera()).toBe(before);
  });

  it('swaps ortho→perspective (cameraChanged=true) when leaving ortho mode', () => {
    // Covers the swapToPerspective branch of setControlType (the reciprocal of
    // the ortho swap above). Starting from an orthographic camera and asking
    // for a non-ortho control type must replace it with a perspective camera.
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    ortho.position.set(0, 0, 50);
    const harness = makeCtx(ortho);
    const before = harness.getCurrentCamera();

    const { cameraChanged } = setControlType('orbit', harness.ctx);

    expect(cameraChanged).toBe(true);
    expect(harness.getCurrentCamera()).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(harness.getCurrentCamera()).not.toBe(before);
    expect(harness.setControlTypeMock).toHaveBeenCalledWith('orbit');
  });

  it('forwards type + new camera ref to controls', () => {
    const persp = new THREE.PerspectiveCamera();
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    setControlType('ortho', harness.ctx);

    expect(harness.setControlTypeMock).toHaveBeenCalledWith('ortho');
    expect(harness.controlsSetCamera).toHaveBeenCalledWith(harness.getCurrentCamera());
  });

  it('returns the dolly to baseline before deriving the replacement projection', () => {
    const persp = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    persp.position.set(0, 0, 5 / 1.15);
    const made = makeControls();
    made.returnAutoDollyToBaseline.mockImplementation(() => persp.position.set(0, 0, 5));
    const harness = makeCtx(persp, made.controls);

    setControlType('ortho', harness.ctx);

    const ortho = harness.getCurrentCamera() as THREE.OrthographicCamera;
    expect(made.returnAutoDollyToBaseline).toHaveBeenCalledTimes(1);
    expect(ortho.position.z).toBeCloseTo(5, 6);
    expect(ortho.top - ortho.bottom).toBeCloseTo(2 * 5 * Math.tan(Math.PI / 6), 6);
  });

  it('triggers material update after the swap', () => {
    const persp = new THREE.PerspectiveCamera();
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    setControlType('ortho', harness.ctx);

    expect(harness.updateMaterialsForCurrentCamera).toHaveBeenCalledTimes(1);
  });
});
