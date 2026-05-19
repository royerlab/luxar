/**
 * Unit tests for the camera-materials helpers extracted from
 * SceneManager in Step 7 of the scene-folder layout overhaul.
 *
 * Mocks the global materialManager (which is the singleton bridge
 * between scene-manager and the rendering layer) and verifies:
 *  - perspective path sends FOV radians + orthographic=false;
 *  - orthographic path sends frustum height + orthographic=true;
 *  - adjustFOV clamps to config min/max and re-runs material update;
 *  - adjustFOV is a no-op on orthographic cameras.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

vi.mock('../../../../../rendering/material-manager', () => ({
  materialManager: {
    updateCameraParams: vi.fn(),
  },
}));

import {
  type CameraMaterialsCtx,
  updateMaterialsForCurrentCamera,
  adjustFOV,
} from '../../../../../scene/scene-manager/camera/camera-materials';
import { SceneBoundsCache } from '../../../../../scene/scene-manager/clipping/scene-bounds-cache';
import { materialManager } from '../../../../../rendering/material-manager';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';
import { config } from '../../../../../config';

function makeRenderer(): Renderer {
  return {
    getDrawingBufferSize: vi.fn((target: THREE.Vector2) => {
      target.set(800, 600);
      return target;
    }),
  } as unknown as Renderer;
}

function makeCtx(opts: {
  camera: THREE.Camera;
  bufferSize?: THREE.Vector2;
}): CameraMaterialsCtx {
  return {
    renderer: makeRenderer(),
    camera: opts.camera as CameraMaterialsCtx['camera'],
    scene: new THREE.Scene(),
    boundsCache: new SceneBoundsCache(),
    bufferSize: opts.bufferSize ?? new THREE.Vector2(),
  };
}

describe('updateMaterialsForCurrentCamera', () => {
  beforeEach(() => {
    vi.mocked(materialManager.updateCameraParams).mockClear();
  });

  it('sends FOV (radians) + orthographic=false for perspective camera', () => {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    const ctx = makeCtx({ camera });

    updateMaterialsForCurrentCamera(ctx);

    expect(materialManager.updateCameraParams).toHaveBeenCalledTimes(1);
    const [proj, buf, isOrtho] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    // FOV converted to radians (~1.047 for 60deg).
    expect(proj).toBeCloseTo((60 * Math.PI) / 180, 5);
    expect(buf).toBeInstanceOf(THREE.Vector2);
    expect(isOrtho).toBe(false);
  });

  it('sends frustum height + orthographic=true for orthographic camera', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    const ctx = makeCtx({ camera });

    updateMaterialsForCurrentCamera(ctx);

    expect(materialManager.updateCameraParams).toHaveBeenCalledTimes(1);
    const [proj, buf, isOrtho] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    // Frustum height = (top - bottom) / zoom = 10 / 1 = 10.
    expect(proj).toBe(10);
    expect(buf).toBeInstanceOf(THREE.Vector2);
    expect(isOrtho).toBe(true);
  });

  it('uses the supplied pre-allocated Vector2 (no allocation)', () => {
    const camera = new THREE.PerspectiveCamera();
    const bufferSize = new THREE.Vector2(0, 0);
    const ctx = makeCtx({ camera, bufferSize });

    updateMaterialsForCurrentCamera(ctx);

    const [, buf] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    expect(buf).toBe(bufferSize); // Same reference, not a fresh allocation.
    expect(bufferSize.x).toBe(800);
    expect(bufferSize.y).toBe(600);
  });

  it('forwards bounds-cache near-cull margin (uses default 0.1 for empty cache)', () => {
    const camera = new THREE.PerspectiveCamera();
    const ctx = makeCtx({ camera });

    updateMaterialsForCurrentCamera(ctx);

    const [, , , nearCull] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    expect(nearCull).toBe(0.1);
  });
});

describe('adjustFOV', () => {
  beforeEach(() => {
    vi.mocked(materialManager.updateCameraParams).mockClear();
  });

  it('mutates perspective camera fov and refreshes materials', () => {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    const ctx = makeCtx({ camera });

    const fovBefore = camera.fov;
    adjustFOV(ctx, 10); // positive deltaY → increase fov
    expect(camera.fov).not.toBe(fovBefore);
    expect(updateSpy).toHaveBeenCalledTimes(1);
    expect(materialManager.updateCameraParams).toHaveBeenCalledTimes(1);
  });

  it('clamps fov to config.camera.fovMax', () => {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    const ctx = makeCtx({ camera });

    // Huge deltaY → should clamp to config.camera.fovMax.
    adjustFOV(ctx, 1_000_000);
    expect(camera.fov).toBeLessThanOrEqual(config.camera.fovMax);
  });

  it('clamps fov to config.camera.fovMin', () => {
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
    const ctx = makeCtx({ camera });

    adjustFOV(ctx, -1_000_000);
    expect(camera.fov).toBeGreaterThanOrEqual(config.camera.fovMin);
  });

  it('is a no-op on orthographic cameras', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    const ctx = makeCtx({ camera });

    adjustFOV(ctx, 100);

    expect(updateSpy).not.toHaveBeenCalled();
    expect(materialManager.updateCameraParams).not.toHaveBeenCalled();
  });
});
