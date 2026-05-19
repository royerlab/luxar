/**
 * Unit tests for camera-mode helpers extracted from SceneManager
 * in Step 8 of the scene-folder layout overhaul.
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
  return {
    controls: {
      setCamera,
      setControlType: setControlTypeMock,
      getFocusTarget,
    } as unknown as ControlsManager,
    setCamera,
    setControlTypeMock,
  };
}

function makePostProcessing() {
  const setCamera = vi.fn();
  return {
    pp: { setCamera } as unknown as PostProcessingManager,
    setCamera,
  };
}

function makeCtx(initialCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera) {
  let currentCamera: THREE.Camera = initialCamera;
  const { controls, setCamera: controlsSetCamera, setControlTypeMock } = makeControls();
  const { pp, setCamera: ppSetCamera } = makePostProcessing();
  const updateMaterialsForCurrentCamera = vi.fn();
  const setLastOrthoZoom = vi.fn();

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
  };

  return {
    ctx,
    getCurrentCamera: () => currentCamera,
    controlsSetCamera,
    setControlTypeMock,
    ppSetCamera,
    updateMaterialsForCurrentCamera,
    setLastOrthoZoom,
  };
}

describe('swapToOrthographic', () => {
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

  it('preserves position, quaternion and up vector from the ortho camera', () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    ortho.position.set(1, 2, 3);
    ortho.up.set(0, 0, 1);
    ortho.quaternion.set(0.1, 0.2, 0.3, 0.9).normalize();
    const harness = makeCtx(ortho);

    swapToPerspective(harness.ctx);

    const persp = harness.getCurrentCamera() as THREE.PerspectiveCamera;
    expect(persp.position.toArray()).toEqual([1, 2, 3]);
    expect(persp.up.toArray()).toEqual([0, 0, 1]);
    expect(persp.quaternion.x).toBeCloseTo(ortho.quaternion.x, 6);
    expect(persp.near).toBe(0.1);
    expect(persp.far).toBe(1000);
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

    const { cameraChanged } = setControlType('ortho', harness.ctx);

    expect(cameraChanged).toBe(true);
    expect(harness.getCurrentCamera()).toBeInstanceOf(THREE.OrthographicCamera);
  });

  it('returns cameraChanged=false when switching control type without projection change', () => {
    const persp = new THREE.PerspectiveCamera();
    const harness = makeCtx(persp);

    const { cameraChanged } = setControlType('fly', harness.ctx);

    expect(cameraChanged).toBe(false);
    expect(harness.getCurrentCamera()).toBe(persp);
  });

  it('forwards type + new camera ref to controls', () => {
    const persp = new THREE.PerspectiveCamera();
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    setControlType('ortho', harness.ctx);

    expect(harness.setControlTypeMock).toHaveBeenCalledWith('ortho');
    expect(harness.controlsSetCamera).toHaveBeenCalledWith(harness.getCurrentCamera());
  });

  it('triggers material update after the swap', () => {
    const persp = new THREE.PerspectiveCamera();
    persp.position.set(0, 0, 50);
    const harness = makeCtx(persp);

    setControlType('ortho', harness.ctx);

    expect(harness.updateMaterialsForCurrentCamera).toHaveBeenCalledTimes(1);
  });
});
