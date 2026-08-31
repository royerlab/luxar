/**
 * Unit tests for the camera-materials helpers used by SceneManager.
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
    getPixelRatio: vi.fn(() => 2),
  } as unknown as Renderer;
}

function makeCtx(opts: { camera: THREE.Camera; bufferSize?: THREE.Vector2 }): CameraMaterialsCtx {
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

  it('mutates the supplied bufferSize in place on each call (MED-47 contract)', () => {
    // Regression for MED-47: the bufferSize is a borrowed, shared reference
    // owned by SceneManager. Consumers that retain it across calls observe
    // it mutate in place when the renderer's drawing-buffer size changes.
    // This test pins the contract: callers MUST copy() if they want to
    // keep the value, and the helper itself overwrites the prior contents.
    const camera = new THREE.PerspectiveCamera();
    const bufferSize = new THREE.Vector2(99, 99);
    const ctx = makeCtx({ camera, bufferSize });

    // First call — getDrawingBufferSize stub writes (800, 600).
    updateMaterialsForCurrentCamera(ctx);
    expect(bufferSize.x).toBe(800);
    expect(bufferSize.y).toBe(600);

    // Simulate a renderer resize and a second call. The same Vector2
    // instance is mutated in place — a retainer of the reference would
    // see its "saved" value silently overwritten.
    (ctx.renderer.getDrawingBufferSize as ReturnType<typeof vi.fn>).mockImplementation(
      (target: THREE.Vector2) => {
        target.set(1024, 768);
        return target;
      }
    );
    updateMaterialsForCurrentCamera(ctx);
    expect(bufferSize.x).toBe(1024);
    expect(bufferSize.y).toBe(768);
  });

  it('forwards bounds-cache near-cull margin (uses default 0.1 for empty cache)', () => {
    const camera = new THREE.PerspectiveCamera();
    const ctx = makeCtx({ camera });

    updateMaterialsForCurrentCamera(ctx);

    const [, , , nearCull] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    expect(nearCull).toBe(0.1);
  });

  it('forwards the active renderer pixel ratio', () => {
    const camera = new THREE.PerspectiveCamera();
    const ctx = makeCtx({ camera });

    updateMaterialsForCurrentCamera(ctx);

    const [, , , , pixelRatio] = vi.mocked(materialManager.updateCameraParams).mock.calls[0];
    expect(pixelRatio).toBe(2);
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
    // Returns true so callers (window wheel handler) know the FOV was
    // applied and can sync FOV-coupled UI (the "Custom" preset stamp).
    expect(adjustFOV(ctx, 10)).toBe(true); // positive deltaY → increase fov
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

  it('is a no-op on orthographic cameras (returns false)', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    const ctx = makeCtx({ camera });

    // false tells callers nothing changed — the window wheel handler
    // uses this to skip the "Custom" preset stamp in ortho mode.
    expect(adjustFOV(ctx, 100)).toBe(false);

    expect(updateSpy).not.toHaveBeenCalled();
    expect(materialManager.updateCameraParams).not.toHaveBeenCalled();
  });
});
