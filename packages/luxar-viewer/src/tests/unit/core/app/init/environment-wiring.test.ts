// @vitest-environment jsdom
/**
 * The scene environment's live wiring: WHEN a capture is asked for. A stale mark on a
 * geometry commit, a slice change and an appearance change; a per-frame tick; the
 * runtime attached once; the `?bake-env` one-shot armed only when requested and fired
 * only after the loader has stayed settled for the grace window.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

vi.mock('../../../../../data/scene-loader-manager', () => ({
  getSceneLoader: vi.fn(() => ({})),
}));
vi.mock('../../../../../rendering/environment/bake', () => ({
  bakeEnvironment: vi.fn(async () => ({
    header: { resolution: 8, probe: { spec: 'auto', position: [0, 0, 0] } },
    faces: [],
    bytes: new Uint8Array([1, 2, 3]),
  })),
  containerToBase64: vi.fn(() => 'AQID'),
}));
vi.mock('../../../../../ui/recording-panel/screenshot-exporter', () => ({
  downloadBlob: vi.fn(),
}));

import {
  BAKE_SETTLED_FRAMES,
  wireSceneEnvironment,
} from '../../../../../core/app/init/environment-wiring';
import { bakeEnvironment } from '../../../../../rendering/environment/bake';
import { downloadBlob } from '../../../../../ui/recording-panel/screenshot-exporter';
import { eventBus } from '../../../../../utils/cross-layer/event-bus';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';
import type { SceneManager } from '../../../../../scene/scene-manager';
import type { AnimationController } from '../../../../../scene/animation/animation-controller';

function makeHarness(options: { bakeEnvironment?: { probe?: string; resolution?: number } } = {}) {
  const callbacks = new Map<string, () => void>();
  const animationController = {
    addPerFrameCallback: vi.fn((id: string, cb: () => void) => {
      callbacks.set(id, cb);
    }),
    removePerFrameCallback: vi.fn((id: string) => callbacks.delete(id)),
  } as unknown as AnimationController;
  const scene = new THREE.Scene();
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  root.userData.sceneContentHash = 'abc123';
  scene.add(root);
  const environment = {
    tick: vi.fn(),
    markStale: vi.fn(),
    setBaked: vi.fn(),
    activeKind: () => 'scene',
    captureCount: 0,
    hasBaked: () => false,
  };
  const sceneManager = {
    scene,
    renderer: {},
    capabilities: { apiSurface: 'webgl2' },
    environment,
    attachEnvironmentRuntime: vi.fn(),
    getSceneViewerConfig: () => undefined,
  } as unknown as SceneManager;
  const events = new EventGroup();
  const settled = { value: true };
  wireSceneEnvironment({
    sceneManager,
    animationController,
    events,
    options: { canvas: {} as HTMLCanvasElement, ...options },
    isSettled: () => settled.value,
  });
  return { callbacks, animationController, environment, sceneManager, events, settled };
}

describe('wireSceneEnvironment', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
  });

  it('attaches the runtime, ticks per frame, and marks stale on commit / slice / appearance', () => {
    const h = makeHarness();
    expect(h.sceneManager.attachEnvironmentRuntime).toHaveBeenCalledTimes(1);
    expect(h.callbacks.has('environment-capture')).toBe(true);
    expect(h.callbacks.has('environment-bake')).toBe(false);

    h.callbacks.get('environment-capture')!();
    expect(h.environment.tick).toHaveBeenCalledTimes(1);

    eventBus.emit('geometry-committed', {});
    expect(h.environment.markStale).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new CustomEvent('luxar-layers-changed'));
    expect(h.environment.markStale).toHaveBeenCalledTimes(2);
    // The dims manager notifies its listeners on a value change.
    (sceneDimsManager as unknown as { notifyListeners: () => void }).notifyListeners();
    expect(h.environment.markStale).toHaveBeenCalledTimes(3);

    // Disposing the event group unhooks everything.
    h.events.dispose();
    eventBus.emit('geometry-committed', {});
    window.dispatchEvent(new CustomEvent('luxar-layers-changed'));
    (sceneDimsManager as unknown as { notifyListeners: () => void }).notifyListeners();
    expect(h.environment.markStale).toHaveBeenCalledTimes(3);
    expect(h.callbacks.size).toBe(0);
  });

  it('under ?bake-env, bakes once after the loader stays settled for the grace window', async () => {
    const h = makeHarness({ bakeEnvironment: { probe: 'node:shell', resolution: 32 } });
    const bake = h.callbacks.get('environment-bake')!;
    expect(bake).toBeDefined();

    // An unsettled frame resets the count.
    for (let i = 0; i < BAKE_SETTLED_FRAMES - 1; i++) bake();
    h.settled.value = false;
    bake();
    h.settled.value = true;
    for (let i = 0; i < BAKE_SETTLED_FRAMES - 1; i++) bake();
    expect(bakeEnvironment).not.toHaveBeenCalled();
    bake();
    expect(bakeEnvironment).toHaveBeenCalledTimes(1);
    // Armed once: the callback removed itself, and a previously baked map is cleared
    // so the bake captures the SCENE.
    expect(h.callbacks.has('environment-bake')).toBe(false);
    expect(h.environment.setBaked).toHaveBeenCalledWith(null);
    const request = vi.mocked(bakeEnvironment).mock.calls[0][0];
    expect(request.probe).toEqual({ node: 'shell' });
    expect(request.resolution).toBe(32);
    expect(request.sceneContentHash).toBe('abc123');

    await vi.waitFor(() => expect(downloadBlob).toHaveBeenCalledTimes(1));
    expect(window.__luxarDebug?.environment?.lastBake).toMatchObject({
      base64: 'AQID',
      byteLength: 3,
    });
  });

  it('a malformed ?probe= records the error instead of baking', async () => {
    const h = makeHarness({ bakeEnvironment: { probe: 'centre' } });
    const bake = h.callbacks.get('environment-bake')!;
    for (let i = 0; i < BAKE_SETTLED_FRAMES; i++) bake();
    await vi.waitFor(() =>
      expect(window.__luxarDebug?.environment?.bakeError).toContain("malformed probe 'centre'")
    );
    expect(bakeEnvironment).not.toHaveBeenCalled();
  });
});
