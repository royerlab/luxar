/**
 * Smoke test for the four SceneManager events that downstream
 * consumers depend on. Added in step 13 of the god-object refactor to
 * pin the public event surface against the folder-index move; the
 * SceneManager class implementation now lives in
 * `scene/scene-manager/index.ts` rather than `scene/scene-manager.ts`.
 *
 * The plan's non-goals list (item 2) explicitly preserves these event
 * names and payloads — this test fails the build if any of the four
 * names is dropped or renamed accidentally.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { SceneManager } from '../../../scene/scene-manager';

describe('SceneManager event surface', () => {
  it('extends THREE.EventDispatcher with the four refactor-non-goal event names', () => {
    // Construct without init() — the constructor doesn't touch
    // WebGL / DOM, so we can verify dispatcher wiring in jsdom.
    const sm = new SceneManager();
    expect(sm).toBeInstanceOf(THREE.EventDispatcher);

    const handlers = {
      change: vi.fn(),
      'camera-changed': vi.fn(),
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
    sm.dispatchEvent({ type: 'webgl-context-restored' });
    sm.dispatchEvent({ type: 'webgpu-device-lost', reason: 'test', message: 'synthetic' });

    expect(handlers.change).toHaveBeenCalledTimes(1);
    expect(handlers['camera-changed']).toHaveBeenCalledTimes(1);
    expect(handlers['webgl-context-restored']).toHaveBeenCalledTimes(1);
    expect(handlers['webgpu-device-lost']).toHaveBeenCalledTimes(1);
  });
});
