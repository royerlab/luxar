/**
 * Smoke test for the four SceneManager events that downstream
 * consumers depend on. Helper files live in
 * `scene/scene-manager/{camera,clipping,render-pipeline,viewport}/`,
 * but the orchestrator class itself stays at `scene/scene-manager.ts`.
 *
 * These public event names and payloads are contractual; this test fails
 * the build if any of the four names is dropped or renamed accidentally.
 *
 * The second describe block exercises the WebGPU `device.lost` observer
 * in `setupContextLossHandling()` — the only place that dispatches
 * `webgpu-device-lost` in production. Covered here because it's a
 * companion to the event-surface smoke test, and because the heavy
 * mocks in `scene-manager.test.ts` are unnecessary for this single
 * private-method behaviour.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { SceneManager } from '../../../scene/scene-manager';
import { config } from '../../../config';

describe('SceneManager event surface', () => {
  it('extends THREE.EventDispatcher with the public scene event names', () => {
    // Construct without init() — the constructor doesn't touch
    // WebGL / DOM, so we can verify dispatcher wiring in jsdom.
    const sm = new SceneManager();
    expect(sm).toBeInstanceOf(THREE.EventDispatcher);

    const handlers = {
      change: vi.fn(),
      'camera-changed': vi.fn(),
      'projection-changed': vi.fn(),
      'webgl-context-restored': vi.fn(),
      'webgpu-device-lost': vi.fn(),
    } as const;

    for (const [type, fn] of Object.entries(handlers)) {
      sm.addEventListener(type as keyof typeof handlers, fn);
    }

    // Dispatch each event with a minimal payload; the test only checks
    // that the dispatch reaches the registered listener.
    sm.dispatchEvent({ type: 'change' });
    sm.dispatchEvent({ type: 'camera-changed' });
    sm.dispatchEvent({ type: 'projection-changed' });
    sm.dispatchEvent({ type: 'webgl-context-restored' });
    sm.dispatchEvent({ type: 'webgpu-device-lost', reason: 'test', message: 'synthetic' });

    expect(handlers.change).toHaveBeenCalledTimes(1);
    expect(handlers['camera-changed']).toHaveBeenCalledTimes(1);
    expect(handlers['projection-changed']).toHaveBeenCalledTimes(1);
    expect(handlers['webgl-context-restored']).toHaveBeenCalledTimes(1);
    expect(handlers['webgpu-device-lost']).toHaveBeenCalledTimes(1);
  });
});

/**
 * Issue #1916 — a field-of-view edit reprojects the SAME camera, so it moves
 * every element on screen while producing NO controls `change` event: the
 * rendering-controls panel calls `updateFOV` directly, and the Ctrl/Cmd-wheel
 * path is explicitly declined by the orbit controls
 * (`luxar-orbit-controls/input/pointer.ts`). `projection-changed` is the only
 * notification projection-dependent caches get, and the GPU pick buffer is one
 * — without it, hover picking answered at the pre-FOV projection until some
 * unrelated camera move happened to dirty it.
 */
describe('SceneManager.updateFOV — projection-changed', () => {
  /**
   * Bare SceneManager with just the state `adjustFOV` →
   * `updateMaterialsForCurrentCamera` reads. `init()` is not run: it would
   * need a real WebGL context. `boundsCache` / `_bufferSize` are field
   * initializers so they already exist; `scene` and `renderer` are not.
   */
  function makeFovSceneManager(fov: number): SceneManager {
    const sm = new SceneManager();
    const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 100);
    const inject = sm as unknown as { camera: unknown; scene: unknown; renderer: unknown };
    inject.camera = camera;
    inject.scene = new THREE.Scene();
    inject.renderer = { getDrawingBufferSize: (v: THREE.Vector2) => v.set(800, 600) };
    return sm;
  }

  it('dispatches when the live FOV actually moves', () => {
    const sm = makeFovSceneManager(50);
    const listener = vi.fn();
    sm.addEventListener('projection-changed', listener);

    expect(sm.updateFOV(10)).toBe(true);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(sm.currentFov).not.toBe(50);
  });

  it('does NOT dispatch when the request is clamped to the FOV already in effect', () => {
    // Already at fovMax (170); pushing further out clamps to the same value.
    // `adjustFOV` still returns true, so gating on its return value alone would
    // fire on every wheel tick at the bound and fade the tooltip for nothing.
    const sm = makeFovSceneManager(config.camera.fovMax);
    const listener = vi.fn();
    sm.addEventListener('projection-changed', listener);

    expect(sm.updateFOV(100)).toBe(true);

    expect(listener).not.toHaveBeenCalled();
    expect(sm.currentFov).toBe(config.camera.fovMax);
  });

  it('does NOT dispatch in orthographic mode, where only the perspective stash moves', () => {
    const sm = new SceneManager();
    const inject = sm as unknown as { camera: unknown; scene: unknown; renderer: unknown };
    inject.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    inject.scene = new THREE.Scene();
    inject.renderer = { getDrawingBufferSize: (v: THREE.Vector2) => v.set(800, 600) };
    const listener = vi.fn();
    sm.addEventListener('projection-changed', listener);

    // Returns true (the stash was updated for a later ortho→perspective swap)
    // but the LIVE projection is untouched, so nothing may be invalidated.
    expect(sm.updateFOV(10)).toBe(true);

    expect(listener).not.toHaveBeenCalled();
  });
});

describe('SceneManager setupContextLossHandling — WebGPU device.lost observer', () => {
  it('dispatches webgpu-device-lost with reason+message when device.lost resolves', async () => {
    // Build a SceneManager with the WebGPU branch wired:
    //   - capabilities.apiSurface === 'webgpu' (selects WebGPU branch)
    //   - renderer.backend.device.lost is a Promise that resolves to
    //     the device-loss info struct.
    // The test doesn't run init() — instead it injects the minimal
    // state setupContextLossHandling() reads, then invokes it via
    // bracket access (the method is private).
    const sm = new SceneManager();
    const lostInfo = { reason: 'destroyed' as const, message: 'synthetic test loss' };
    const fakeRenderer = {
      backend: {
        device: {
          lost: Promise.resolve(lostInfo),
        },
      },
    };
    const fakeCaps = { apiSurface: 'webgpu' as const };
    // Inject private fields via bracket access — keeps the test
    // focused on the device.lost handler without needing the full
    // init pipeline (renderer creation, post-processing, controls).
    (sm as unknown as { renderer: unknown }).renderer = fakeRenderer;
    (sm as unknown as { capabilities: unknown }).capabilities = fakeCaps;

    const listener = vi.fn();
    sm.addEventListener('webgpu-device-lost', listener);

    (sm as unknown as { setupContextLossHandling(): void }).setupContextLossHandling();

    // The handler is attached via Promise.then(); flush microtasks.
    await Promise.resolve();
    await Promise.resolve();

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'webgpu-device-lost',
        reason: 'destroyed',
        message: 'synthetic test loss',
      })
    );
  });

  it('forwards undefined reason/message when device.lost info struct omits them', async () => {
    const sm = new SceneManager();
    const fakeRenderer = {
      backend: { device: { lost: Promise.resolve({}) } },
    };
    (sm as unknown as { renderer: unknown }).renderer = fakeRenderer;
    (sm as unknown as { capabilities: unknown }).capabilities = { apiSurface: 'webgpu' };

    const listener = vi.fn();
    sm.addEventListener('webgpu-device-lost', listener);

    (sm as unknown as { setupContextLossHandling(): void }).setupContextLossHandling();
    await Promise.resolve();
    await Promise.resolve();

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0]).toMatchObject({
      type: 'webgpu-device-lost',
      reason: undefined,
      message: undefined,
    });
  });

  it('is a no-op when renderer.backend.device is absent (build without device-loss reporting)', async () => {
    const sm = new SceneManager();
    // Backend present but no device — mirrors a Three.js build that
    // doesn't expose the GPUDevice on the renderer backend.
    (sm as unknown as { renderer: unknown }).renderer = { backend: {} };
    (sm as unknown as { capabilities: unknown }).capabilities = { apiSurface: 'webgpu' };

    const listener = vi.fn();
    sm.addEventListener('webgpu-device-lost', listener);

    expect(() => {
      (sm as unknown as { setupContextLossHandling(): void }).setupContextLossHandling();
    }).not.toThrow();

    await Promise.resolve();
    expect(listener).not.toHaveBeenCalled();
  });
});
