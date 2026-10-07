// @vitest-environment jsdom
/**
 * The scene environment's live wiring: WHEN a capture is asked for. A stale mark on a
 * geometry commit, a slice change and an appearance change; a per-frame tick; the
 * runtime attached once; the `?bakeEnv` one-shot armed only when requested and fired
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
  wireEnvironmentToLayers,
  wireSceneEnvironment,
} from '../../../../../core/app/init/environment-wiring';
import { LayerStateManager } from '../../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../../data/data-loader-types';
import { bakeEnvironment } from '../../../../../rendering/environment/bake';
import { downloadBlob } from '../../../../../ui/recording-panel/screenshot-exporter';
import { eventBus } from '../../../../../utils/cross-layer/event-bus';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';
import { config } from '../../../../../config';
import * as workerPool from '../../../../../workers/worker-pool';
import type { SceneManager } from '../../../../../scene/scene-manager';
import type { AnimationController } from '../../../../../scene/animation/animation-controller';

function makeHarness(options: { bakeEnvironment?: { probe?: string; resolution?: number } } = {}) {
  const callbacks = new Map<string, () => boolean>();
  const animationController = {
    addPerFrameCallback: vi.fn((id: string, cb: () => boolean) => {
      callbacks.set(id, cb);
    }),
    removePerFrameCallback: vi.fn((id: string) => callbacks.delete(id)),
    requestRender: vi.fn(),
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
    postProcessing: {
      suspendFrameRendersDuring: vi.fn(async (capture: () => Promise<unknown>) => capture()),
    },
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
    config.dataLoading.performance.useWebWorkers = true;
    vi.stubGlobal('Worker', class {});
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
  });

  it('captures and bakes after settlement when Web Workers are disabled', async () => {
    config.dataLoading.performance.useWebWorkers = false;
    const h = makeHarness({ bakeEnvironment: { resolution: 8 } });
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(captureReady()).toBe(true);

    h.settled.value = false;
    eventBus.emit('geometry-committed', {});
    expect(h.callbacks.has('environment-capture-ready')).toBe(true);
    expect(h.callbacks.get('environment-capture-ready')!()).toBe(false);
    h.settled.value = true;
    h.environment.tick.mockReturnValueOnce(true);
    expect(h.callbacks.get('environment-capture-ready')!()).toBe(true);
    expect(h.callbacks.has('environment-capture-ready')).toBe(false);

    const bake = h.callbacks.get('environment-bake')!;
    for (let i = 0; i < BAKE_SETTLED_FRAMES; i++) bake();
    expect(bakeEnvironment).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(window.__luxarDebug?.environment?.lastBake).toBeDefined());
    h.events.dispose();
  });

  it('allows capture when the Worker API is unavailable', () => {
    vi.stubGlobal('Worker', undefined);
    const h = makeHarness();
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(captureReady()).toBe(true);
    h.events.dispose();
  });

  it('keeps waiting if workers are disabled during an in-flight initialization', () => {
    const pending = vi
      .spyOn(workerPool, 'isDataWorkerPoolInitializationPending')
      .mockReturnValue(true);
    config.dataLoading.performance.useWebWorkers = false;
    const h = makeHarness();
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(captureReady()).toBe(false);
    pending.mockReturnValue(false);
    expect(captureReady()).toBe(true);
    h.events.dispose();
    pending.mockRestore();
  });

  it('captures and bakes when workers are enabled but no pool has started', async () => {
    const h = makeHarness({ bakeEnvironment: { resolution: 8 } });
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(workerPool.isDataWorkerPoolInitializationPending()).toBe(false);
    expect(captureReady()).toBe(true);

    h.settled.value = false;
    eventBus.emit('geometry-committed', {});
    expect(h.callbacks.has('environment-capture-ready')).toBe(true);
    h.settled.value = true;
    h.environment.tick.mockReturnValueOnce(true);
    expect(h.callbacks.get('environment-capture-ready')!()).toBe(true);
    expect(h.callbacks.has('environment-capture-ready')).toBe(false);

    const bake = h.callbacks.get('environment-bake')!;
    for (let i = 0; i < BAKE_SETTLED_FRAMES; i++) bake();
    expect(bakeEnvironment).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(window.__luxarDebug?.environment?.lastBake).toBeDefined());
    h.events.dispose();
  });

  it('captures and bakes after worker initialization stops pending', async () => {
    const pending = vi
      .spyOn(workerPool, 'isDataWorkerPoolInitializationPending')
      .mockReturnValue(true);
    const h = makeHarness({ bakeEnvironment: { resolution: 8 } });
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(captureReady()).toBe(false);
    pending.mockReturnValue(false);
    expect(captureReady()).toBe(true);

    h.settled.value = false;
    eventBus.emit('geometry-committed', {});
    expect(h.callbacks.has('environment-capture-ready')).toBe(true);
    h.settled.value = true;
    expect(h.callbacks.get('environment-capture-ready')!()).toBe(false);
    expect(h.callbacks.has('environment-capture-ready')).toBe(false);
    const bake = h.callbacks.get('environment-bake')!;
    for (let i = 0; i < BAKE_SETTLED_FRAMES; i++) bake();
    expect(bakeEnvironment).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(window.__luxarDebug?.environment?.lastBake).toBeDefined());
    h.events.dispose();
    pending.mockRestore();
  });

  it('attaches the runtime at init, gates capture, then ticks and marks stale', () => {
    const pending = vi
      .spyOn(workerPool, 'isDataWorkerPoolInitializationPending')
      .mockReturnValue(true);
    const h = makeHarness();
    expect(h.sceneManager.attachEnvironmentRuntime).toHaveBeenCalledTimes(1);
    const captureReady = vi.mocked(h.sceneManager.attachEnvironmentRuntime).mock.calls[0][1];
    expect(captureReady()).toBe(false);
    expect(h.callbacks.has('environment-capture')).toBe(true);
    expect(h.callbacks.has('environment-bake')).toBe(false);
    expect(h.callbacks.has('environment-runtime-attach')).toBe(false);
    expect(h.callbacks.has('environment-capture-ready')).toBe(false);

    // The callback reports whether a capture landed (render-on-change).
    h.environment.tick.mockReturnValueOnce(false).mockReturnValueOnce(true);
    expect(h.callbacks.get('environment-capture')!()).toBe(false);
    expect(h.environment.tick).toHaveBeenCalledTimes(1);
    expect(h.callbacks.get('environment-capture')!()).toBe(true);

    eventBus.emit('geometry-committed', {});
    expect(h.environment.markStale).toHaveBeenCalledTimes(1);
    expect(h.callbacks.has('environment-capture-ready')).toBe(true);
    const ready = h.callbacks.get('environment-capture-ready')!;
    expect(ready()).toBe(false);
    h.settled.value = false;
    pending.mockReturnValue(false);
    expect(captureReady()).toBe(true);
    expect(ready()).toBe(false);
    h.settled.value = true;
    expect(ready()).toBe(false);
    expect(h.callbacks.has('environment-capture-ready')).toBe(false);
    // The dims manager notifies its listeners on a value change.
    (
      sceneDimsManager as unknown as { notifyListeners: (changed: boolean) => void }
    ).notifyListeners(true);
    expect(h.environment.markStale).toHaveBeenCalledTimes(2);

    // Disposing the event group unhooks everything.
    h.events.dispose();
    eventBus.emit('geometry-committed', {});
    (
      sceneDimsManager as unknown as { notifyListeners: (changed: boolean) => void }
    ).notifyListeners(true);
    expect(h.environment.markStale).toHaveBeenCalledTimes(2);
    expect(h.callbacks.size).toBe(0);
    pending.mockRestore();
  });

  it('marks stale on every Layers-panel appearance edit, but not on a selection or gain change', () => {
    // `luxar-layers-changed` fires once per dataset load, so it never carried an
    // edit: a slider drag on a layer the cube map reflects left the map stale.
    const markStale = vi.fn();
    const sceneManager = { environment: { markStale } } as unknown as SceneManager;
    const layerState = new LayerStateManager();
    const leaf = (path: string): SceneNode =>
      ({ path, type: 'points', hasSpatialIndex: false, attrs: { layer: true } }) as SceneNode;
    const graph = {
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        leaf('/a'),
        leaf('/b'),
        { path: '/hum', type: 'sound', hasSpatialIndex: false, attrs: { layer: true, gain: 1 } },
      ],
    } as SceneNode;
    layerState.initFromSceneGraph(graph);
    const events = new EventGroup();
    wireEnvironmentToLayers(sceneManager, layerState, events);

    layerState.select('/a', 'single');
    layerState.select('/b', 'add');
    // A sound row's gain reaches the audio graph, not anything the cube map sees.
    layerState.setSoundGain('/hum', 0.5);
    expect(layerState.getLayer('/hum')?.sound?.gain).toBe(0.5);
    expect(markStale).not.toHaveBeenCalled();

    layerState.setGamma('/a', 2);
    layerState.applyToSelected((l) => (l.opacity = 0.5));
    layerState.setVisible('/b', false);
    expect(markStale).toHaveBeenCalledTimes(3);
    // "Reset all layers" re-derives the whole state.
    layerState.initFromSceneGraph(graph);
    expect(markStale).toHaveBeenCalledTimes(4);

    events.dispose();
    layerState.setGamma('/a', 3);
    expect(markStale).toHaveBeenCalledTimes(4);
  });

  it('marks stale for a changed slice but not a forced same-slice refresh', () => {
    const h = makeHarness();
    h.sceneManager.scene.userData.sceneDimensions = {
      dimensions: [{ name: 'time', range: [0, 2], step: 1, discrete: true, display: false }],
    };
    sceneDimsManager.initFromScene(h.sceneManager.scene);

    sceneDimsManager.setDimensionValue(0, 0, { force: true });
    expect(h.environment.markStale).not.toHaveBeenCalled();

    sceneDimsManager.setDimensionValue(0, 1);
    expect(h.environment.markStale).toHaveBeenCalledTimes(1);
    sceneDimsManager.setDimensionValue(0, 1, { force: true });
    expect(h.environment.markStale).toHaveBeenCalledTimes(1);

    eventBus.emit('geometry-committed', {});
    expect(h.environment.markStale).toHaveBeenCalledTimes(2);
    h.events.dispose();
    sceneDimsManager.reset();
  });

  it('under ?bakeEnv, bakes once after the loader stays settled for the grace window', async () => {
    const pending = vi
      .spyOn(workerPool, 'isDataWorkerPoolInitializationPending')
      .mockReturnValue(true);
    const h = makeHarness({ bakeEnvironment: { probe: 'node:shell', resolution: 32 } });
    const bake = h.callbacks.get('environment-bake')!;
    expect(bake).toBeDefined();

    for (let i = 0; i < BAKE_SETTLED_FRAMES + 1; i++) bake();
    expect(bakeEnvironment).not.toHaveBeenCalled();
    pending.mockReturnValue(false);

    // An unsettled frame resets the count.
    for (let i = 0; i < BAKE_SETTLED_FRAMES - 1; i++) bake();
    h.settled.value = false;
    bake();
    h.settled.value = true;
    for (let i = 0; i < BAKE_SETTLED_FRAMES - 1; i++) bake();
    expect(bakeEnvironment).not.toHaveBeenCalled();
    bake();
    expect(bakeEnvironment).toHaveBeenCalledTimes(1);
    expect(h.sceneManager.postProcessing.suspendFrameRendersDuring).toHaveBeenCalledTimes(1);
    // Armed once: the callback removed itself, and a previously baked map is cleared
    // so the bake captures the SCENE.
    expect(h.callbacks.has('environment-bake')).toBe(false);
    expect(h.environment.setBaked).toHaveBeenCalledWith(null);
    const request = vi.mocked(bakeEnvironment).mock.calls[0][0];
    expect(request.probe).toEqual({ node: 'shell' });
    expect(request.resolution).toBe(32);
    expect(request.sceneContentHash).toBe('abc123');

    await vi.waitFor(() => expect(downloadBlob).toHaveBeenCalledTimes(1));
    // The bake swapped the scene's environment: the loop is asked to redraw.
    await vi.waitFor(() =>
      expect(h.animationController.requestRender).toHaveBeenCalledWith('environmentBake')
    );
    expect(window.__luxarDebug?.environment?.lastBake).toMatchObject({
      base64: 'AQID',
      byteLength: 3,
    });
    pending.mockRestore();
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
