/**
 * Unit tests for core/app/dataset/load-dataset.ts.
 *
 * The helper composes a precise sequence of orchestrator callbacks +
 * direct collaborator calls. The sequence is part of observable
 * behaviour (e.g. setSceneId before loadSceneData ensures persisted
 * settings reach materials at construction time, layers before
 * colormap legend, overlays before picking, animation last).
 */

import * as THREE from 'three';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { loadDataset, type LoadDatasetPorts } from '../../../../../core/app/dataset/load-dataset';
import { captureViewerState } from '../../../../../config/zarr-bridge/viewer-state-capture';
import { config } from '../../../../../config';
import { syncCameraFovState } from '../../../../../ui/rendering-controls/sync-current-state';
import { deferred, type Deferred } from '../../../../helpers/deferred';
import { defineLifecycleContract } from '../../../_shared/lifecycle-contract';

vi.mock('../../../../../data/scene-loader-manager', () => ({
  getSceneLoader: vi.fn(),
}));
vi.mock('../../../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({ getCurrentTheme: () => ({ id: 'dark', name: 'Dark Theme' }) }),
  },
}));
import { getSceneLoader } from '../../../../../data/scene-loader-manager';

interface Trace {
  order: string[];
  recordedViewerConfig: unknown;
}

function makePorts(trace: Trace, overrides: Partial<LoadDatasetPorts> = {}): LoadDatasetPorts {
  const viewerConfig = { theme: 'dark' };
  const inputHandler = {
    clearDimensionUI: vi.fn(() => trace.order.push('clearDimensionUI')),
    initDimensionSliders: vi.fn(() => trace.order.push('initDimensionSliders')),
  };
  const renderingControls = {
    settings: { ...config.renderingControls.defaults },
    setSceneId: vi.fn(() => trace.order.push('setSceneId')),
    setZarrViewerConfig: vi.fn(() => trace.order.push('setZarrViewerConfig')),
    hasStoredSettings: vi.fn().mockReturnValue(false),
    getSettingsSnapshot: vi.fn(() => ({ ...config.renderingControls.defaults })),
    adoptSceneContentHash: vi.fn().mockReturnValue(false),
    resetToDefaults: vi.fn(() => trace.order.push('resetToDefaults')),
    applyZarrDefaults: vi.fn(() => trace.order.push('applyZarrDefaults')),
    syncCameraFovState: vi.fn(() => trace.order.push('syncCameraFovState')),
    updateSceneScale: vi.fn(() => trace.order.push('updateSceneScale')),
  };
  const sceneManager = {
    loadSceneData: vi.fn().mockImplementation(async () => trace.order.push('loadSceneData')),
    getSceneViewerConfig: vi.fn().mockReturnValue(viewerConfig),
    getSceneContentHash: vi.fn().mockReturnValue('hash-b'),
    updateClippingPlanes: vi.fn(() => trace.order.push('updateClippingPlanes')),
    warmBlendModePrograms: vi.fn(async () => {
      trace.order.push('warmBlendModePrograms');
    }),
    scene: { children: [] as THREE.Object3D[] },
  };
  const animationController = {
    startAnimation: vi.fn(() => trace.order.push('startAnimation')),
  };

  return {
    inputHandler: inputHandler as never,
    renderingControls: renderingControls as never,
    sceneManager: sceneManager as never,
    animationController: animationController as never,
    layersPanel: undefined,
    loaderConfig: undefined,
    openCacheStats: false,
    disposeOverlays: vi.fn(() => trace.order.push('disposeOverlays')),
    disposePicking: vi.fn(() => trace.order.push('disposePicking')),
    initScaleBar: vi.fn(() => trace.order.push('initScaleBar')),
    initColormapLegend: vi.fn(() => trace.order.push('initColormapLegend')),
    initOverlays: vi.fn(async () => {
      trace.order.push('initOverlays');
    }),
    initPicking: vi.fn(async () => {
      trace.order.push('initPicking');
    }),
    applyViewerConfigState: vi.fn((config) => {
      trace.order.push('applyViewerConfigState');
      trace.recordedViewerConfig = config;
    }),
    openCacheStatsView: vi.fn(() => trace.order.push('openCacheStatsView')),
    isStale: vi.fn(() => false),
    ...overrides,
  };
}

describe('loadDataset', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(null);
  });

  it('always calls clearDimensionUI and disposeOverlays before loadSceneData', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const clearIdx = trace.order.indexOf('clearDimensionUI');
    const disposeIdx = trace.order.indexOf('disposeOverlays');
    const loadIdx = trace.order.indexOf('loadSceneData');
    expect(clearIdx).toBeGreaterThanOrEqual(0);
    expect(disposeIdx).toBeGreaterThanOrEqual(0);
    expect(clearIdx).toBeLessThan(loadIdx);
    expect(disposeIdx).toBeLessThan(loadIdx);
  });

  it('resets the previous scene environment before loading new data', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    const resetForDataset = vi.fn(() => trace.order.push('resetForDataset'));
    ports.sceneManager.environment = {
      resetForDataset,
      configure: vi.fn(),
      setBaked: vi.fn(),
    } as never;
    ports.sceneManager.getSceneBakedEnvironment = vi.fn().mockReturnValue(null);
    await loadDataset('scene.zarr', ports);

    expect(resetForDataset).toHaveBeenCalledOnce();
    expect(trace.order.indexOf('resetForDataset')).toBeLessThan(
      trace.order.indexOf('loadSceneData')
    );
  });

  it('disposePicking runs BEFORE loadSceneData (stale-session-vs-disposed-geometry guard)', async () => {
    // The previous picking session's nodeMap references geometries that
    // loadSceneData's clearSceneContent() disposes. Disposing the session
    // up-front (alongside disposeOverlays) means a mid-load failure can
    // never leave a stale session firing picks against disposed
    // geometries; the end-of-load initPicking is the (re)creation point.
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const disposeIdx = trace.order.indexOf('disposePicking');
    const loadIdx = trace.order.indexOf('loadSceneData');
    expect(disposeIdx).toBeGreaterThanOrEqual(0);
    expect(loadIdx).toBeGreaterThanOrEqual(0);
    expect(disposeIdx).toBeLessThan(loadIdx);
  });

  it('a failing loadSceneData still leaves the old picking session disposed', async () => {
    // The error-recovery contract: even when the load throws mid-way,
    // disposePicking already ran (so no stale picks) and initPicking
    // never ran (no half-built new session).
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.sceneManager.loadSceneData as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('network down')
    );

    await expect(loadDataset('scene.zarr', ports)).rejects.toThrow('network down');

    expect(ports.disposePicking).toHaveBeenCalledOnce();
    expect(ports.initPicking).not.toHaveBeenCalled();
  });

  it('setSceneId must run BEFORE loadSceneData (so persisted settings reach materials)', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const setIdIdx = trace.order.indexOf('setSceneId');
    const loadIdx = trace.order.indexOf('loadSceneData');
    // Guard: indexOf returns -1 when absent; `-1 < N` would vacuously pass.
    expect(setIdIdx).toBeGreaterThanOrEqual(0);
    expect(loadIdx).toBeGreaterThanOrEqual(0);
    expect(setIdIdx).toBeLessThan(loadIdx);
  });

  it('renderingControls.setSceneId receives the dataset src', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    await loadDataset('http://example.com/scene.zarr', ports);

    expect(ports.renderingControls.setSceneId).toHaveBeenCalledExactlyOnceWith(
      'http://example.com/scene.zarr'
    );
  });

  it('allows viewer-config FOV framing only when hasStoredSettings=false', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    await loadDataset('scene.zarr', ports);

    expect(ports.sceneManager.loadSceneData).toHaveBeenCalledExactlyOnceWith(
      'scene.zarr',
      undefined,
      { applyViewerConfigFov: true, beforeFrame: expect.any(Function) }
    );
    expect(trace.order).toContain('applyZarrDefaults');
  });

  it('preserves stored FOV during framing and skips zarr defaults', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    await loadDataset('scene.zarr', ports);

    expect(ports.sceneManager.loadSceneData).toHaveBeenCalledExactlyOnceWith(
      'scene.zarr',
      undefined,
      { applyViewerConfigFov: false, beforeFrame: expect.any(Function) }
    );
    expect(ports.renderingControls.applyZarrDefaults).not.toHaveBeenCalled();
  });

  it('syncs an authored-position FOV into exported state when stored settings exist', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    ports.renderingControls.settings.fov = 47;
    ports.renderingControls.settings.fovPreset = '50mm Normal';

    const camera = new THREE.PerspectiveCamera(47, 1, 0.1, 1000);
    const focusTarget = new THREE.Vector3();
    Object.assign(ports.sceneManager, {
      camera,
      controls: { getFocusTarget: () => focusTarget },
    });
    Object.defineProperty(ports.sceneManager, 'currentFov', {
      configurable: true,
      get: () => camera.fov,
    });
    (ports.sceneManager.loadSceneData as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      trace.order.push('loadSceneData');
      camera.fov = 63;
    });
    (ports.renderingControls.syncCameraFovState as ReturnType<typeof vi.fn>).mockImplementation(
      () => {
        trace.order.push('syncCameraFovState');
        syncCameraFovState(ports.renderingControls.settings, ports.sceneManager);
      }
    );

    await loadDataset('scene.zarr', ports);

    const exported = captureViewerState(ports.sceneManager, ports.renderingControls, {
      getDims: () => null,
    } as never);
    expect(ports.renderingControls.syncCameraFovState).toHaveBeenCalledOnce();
    expect(exported.camera?.fov).toBe(camera.fov);
    expect(exported.camera?.fov_preset).toBe('35mm');
  });

  it('layers panel hydration runs only when layersPanel present AND sceneLoader has sceneGraph', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const initFromScene = vi.fn();
    const ports = makePorts(trace, {
      layersPanel: {
        initFromScene,
        setFailedLoadsProvider: vi.fn(),
        setCameraFramer: vi.fn(),
      } as never,
    });
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    ports.sceneManager.scene.children = [root];
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
      getFailedLoadsProvider: vi.fn(() => ({ getFailedPaths: () => [], retryAll: vi.fn() })),
      setLeafMaterializedListener: vi.fn(),
    });

    await loadDataset('scene.zarr', ports);

    expect(initFromScene).toHaveBeenCalledOnce();
  });

  it('hands the layers panel the failed-loads provider, AFTER initFromScene', async () => {
    // #1055: the panel must receive the SAME provider getFailedLoadsProvider()
    // returns, and only after initFromScene (whose clear() resets any prior
    // provider first — injecting before would be wiped).
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const initFromScene = vi.fn();
    const setFailedLoadsProvider = vi.fn();
    const ports = makePorts(trace, {
      layersPanel: { initFromScene, setFailedLoadsProvider, setCameraFramer: vi.fn() } as never,
    });
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    ports.sceneManager.scene.children = [root];
    const provider = { getFailedPaths: () => [], retryAll: vi.fn() };
    const getFailedLoadsProvider = vi.fn(() => provider);
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
      getFailedLoadsProvider,
      setLeafMaterializedListener: vi.fn(),
    });

    await loadDataset('scene.zarr', ports);

    // The panel got exactly the object the loader handed out…
    expect(setFailedLoadsProvider).toHaveBeenCalledExactlyOnceWith(provider);
    // …and strictly after initFromScene (load-ordering contract).
    expect(setFailedLoadsProvider.mock.invocationCallOrder[0]).toBeGreaterThan(
      initFromScene.mock.invocationCallOrder[0]
    );
  });

  it('routes leaves the loader materialises LATER to the layers panel, AFTER initFromScene', async () => {
    // A partition part activated by the LOD registry (or any leaf built after
    // load) must pick up the panel's live layer state before it is drawn. The
    // loader reports it through this listener; the panel checks the graph.
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const initFromScene = vi.fn();
    const applyLayerStateToNewLeaf = vi.fn();
    const ports = makePorts(trace, {
      layersPanel: {
        initFromScene,
        setFailedLoadsProvider: vi.fn(),
        setCameraFramer: vi.fn(),
        applyLayerStateToNewLeaf,
      } as never,
    });
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    ports.sceneManager.scene.children = [root];
    const setLeafMaterializedListener = vi.fn();
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
      getFailedLoadsProvider: vi.fn(() => ({ getFailedPaths: () => [], retryAll: vi.fn() })),
      setLeafMaterializedListener,
    });

    await loadDataset('scene.zarr', ports);

    expect(setLeafMaterializedListener).toHaveBeenCalledOnce();
    expect(setLeafMaterializedListener.mock.invocationCallOrder[0]).toBeGreaterThan(
      initFromScene.mock.invocationCallOrder[0]
    );
    const listener = setLeafMaterializedListener.mock.calls[0][0] as (
      graph: unknown,
      path: string,
      object: THREE.Object3D
    ) => void;
    const graph = { kind: 'graph' };
    const leaf = new THREE.Mesh();
    listener(graph, '/nuclei/part_0', leaf);
    expect(applyLayerStateToNewLeaf).toHaveBeenCalledExactlyOnceWith(graph, '/nuclei/part_0', leaf);
  });

  it('layers panel hydration is skipped when layersPanel is undefined', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
    });

    await loadDataset('scene.zarr', ports);
    // initColormapLegend is also gated on layersPanel — should NOT run.
    expect(ports.initColormapLegend).not.toHaveBeenCalled();
  });

  it('initColormapLegend runs ONLY when layersPanel present (and BEFORE initOverlays)', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace, {
      layersPanel: { initFromScene: vi.fn() } as never,
    });
    await loadDataset('scene.zarr', ports);

    const legendIdx = trace.order.indexOf('initColormapLegend');
    const overlaysIdx = trace.order.indexOf('initOverlays');
    expect(legendIdx).toBeGreaterThanOrEqual(0);
    expect(overlaysIdx).toBeGreaterThanOrEqual(0);
    expect(legendIdx).toBeLessThan(overlaysIdx);
  });

  it('initOverlays runs BEFORE initPicking', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const overlaysIdx = trace.order.indexOf('initOverlays');
    const pickingIdx = trace.order.indexOf('initPicking');
    // Guard: indexOf returns -1 when absent; `-1 < N` would vacuously pass.
    expect(overlaysIdx).toBeGreaterThanOrEqual(0);
    expect(pickingIdx).toBeGreaterThanOrEqual(0);
    expect(overlaysIdx).toBeLessThan(pickingIdx);
  });

  it('applyViewerConfigState is called AFTER initPicking', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const pickingIdx = trace.order.indexOf('initPicking');
    const applyIdx = trace.order.indexOf('applyViewerConfigState');
    // Guard: indexOf returns -1 when absent; `-1 < N` would vacuously pass.
    expect(pickingIdx).toBeGreaterThanOrEqual(0);
    expect(applyIdx).toBeGreaterThanOrEqual(0);
    expect(pickingIdx).toBeLessThan(applyIdx);
    expect(trace.recordedViewerConfig).toEqual({ theme: 'dark' });
  });

  it('openCacheStatsView fires only when openCacheStats flag is true', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace, { openCacheStats: true });
    await loadDataset('scene.zarr', ports);

    expect(ports.openCacheStatsView).toHaveBeenCalledOnce();
  });

  it('openCacheStatsView is NOT called when openCacheStats is false', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    await loadDataset('scene.zarr', ports);

    expect(ports.openCacheStatsView).not.toHaveBeenCalled();
  });

  it('warms blend programs only at the end, after animation starts', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    const startIdx = trace.order.indexOf('startAnimation');
    const warmIdx = trace.order.indexOf('warmBlendModePrograms');
    expect(startIdx).toBeGreaterThanOrEqual(0);
    expect(warmIdx).toBeGreaterThan(startIdx);
    expect(trace.order[trace.order.length - 1]).toBe('warmBlendModePrograms');
  });

  it('does not resolve dataset readiness until blend warming completes', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    let signalWarmStarted!: () => void;
    let finishWarm!: () => void;
    const warmStarted = new Promise<void>((resolve) => {
      signalWarmStarted = resolve;
    });
    const warmFinished = new Promise<void>((resolve) => {
      finishWarm = resolve;
    });
    const ports = makePorts(trace);
    (ports.sceneManager.warmBlendModePrograms as ReturnType<typeof vi.fn>).mockImplementation(
      () => {
        trace.order.push('warmBlendModePrograms');
        signalWarmStarted();
        return warmFinished;
      }
    );

    let resolved = false;
    const loading = loadDataset('scene.zarr', ports).then(() => {
      resolved = true;
    });
    await warmStarted;
    expect(resolved).toBe(false);

    finishWarm();
    await loading;
    expect(resolved).toBe(true);
  });

  it.each([
    ['loadSceneData', 'initDimensionSliders'],
    ['initOverlays', 'initPicking'],
    ['initPicking', 'applyViewerConfigState'],
  ])('a load gone stale during %s stops before %s', async (staleAfter, firstSkipped) => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace, {
      isStale: () => trace.order.includes(staleAfter),
    });

    await loadDataset('scene.zarr', ports);

    expect(trace.order).toContain(staleAfter);
    expect(trace.order).not.toContain(firstSkipped);
    expect(trace.order).not.toContain('startAnimation');
    expect(trace.order).not.toContain('warmBlendModePrograms');
  });

  it('loaderConfig is forwarded to sceneManager.loadSceneData', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const loaderConfig = { prefetch: true } as never;
    const ports = makePorts(trace, { loaderConfig });
    await loadDataset('scene.zarr', ports);

    expect(ports.sceneManager.loadSceneData).toHaveBeenCalledExactlyOnceWith(
      'scene.zarr',
      loaderConfig,
      { applyViewerConfigFov: true, beforeFrame: expect.any(Function) }
    );
  });
});

describe('loadDataset — lifecycle', () => {
  beforeEach(() => {
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(null);
  });

  defineLifecycleContract('loadDataset', {
    create: () => {
      // LuxarApp.loadDataset's session discipline: starting a load disposes the
      // outgoing session, and disposing the app disposes the current one.
      const staleSessions = new Set<number>();
      let sessions = 0;
      let disposed = false;
      const steps: Array<Deferred<void>> = [];
      const step = (): Promise<void> => {
        const d = deferred();
        steps.push(d);
        return d.promise;
      };
      /** The tail a load commits once its data has loaded. */
      const committed: string[] = [];
      return {
        start: (n) => {
          for (let s = 0; s < sessions; s++) staleSessions.add(s);
          const session = sessions++;
          const trace: Trace = { order: [], recordedViewerConfig: undefined };
          const ports = makePorts(trace, {
            initOverlays: step,
            initPicking: step,
            applyViewerConfigState: () => committed.push(`viewerConfigState:${n}`),
            isStale: () => disposed || staleSessions.has(session),
          });
          Object.assign(ports.sceneManager, { loadSceneData: step });
          Object.assign(ports.renderingControls, {
            setZarrViewerConfig: () => committed.push(`zarrViewerConfig:${n}`),
          });
          Object.assign(ports.animationController, {
            startAnimation: () => committed.push(`startAnimation:${n}`),
          });
          return loadDataset(`scene-${n}.zarr`, ports);
        },
        pending: () => steps.length,
        settle: (index, outcome) =>
          outcome === 'ok' ? steps[index].resolve() : steps[index].reject(new Error('404')),
        observe: () => [...committed],
        dispose: () => {
          disposed = true;
        },
      };
    },
    cases: {
      abort: {
        na: 'loadDataset takes no signal: a newer load or an app dispose makes its session stale, and it stops at its next await (the isStale port)',
      },
      dispose: true,
      failure: true,
      retry: true,
      supersede: true,
      doubleDispose: true,
    },
    usableAfterDispose: {
      na: 'disposing the app ends its sessions; the next load belongs to a re-initialised app and opens a new session (LuxarApp.loadDataset)',
    },
  });
});

describe('loadDataset with saved settings from another build', () => {
  it('restores authored manual clipping planes after metadata framing', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (ports.renderingControls.adoptSceneContentHash as ReturnType<typeof vi.fn>).mockReturnValue(
      true
    );
    (ports.renderingControls.getSettingsSnapshot as ReturnType<typeof vi.fn>).mockReturnValue({
      dynamicClippingEnabled: false,
      near: 0.5,
      far: 500,
    });
    (ports.sceneManager.loadSceneData as ReturnType<typeof vi.fn>).mockImplementation(
      (_src: string, _config: unknown, options: { beforeFrame?: (root: THREE.Group) => void }) => {
        const root = new THREE.Group();
        root.userData.contentHash = 'hash-b';
        options.beforeFrame?.(root);
        trace.order.push('frameFromRootMetadata');
      }
    );
    await loadDataset('scene.zarr', ports);

    expect(ports.sceneManager.updateClippingPlanes).toHaveBeenCalledExactlyOnceWith(0.5, 500);
    expect(trace.order.indexOf('updateClippingPlanes')).toBeGreaterThan(
      trace.order.indexOf('frameFromRootMetadata')
    );
  });

  it('keeps an edit made while scene nodes load instead of applying late defaults', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    let edited = false;
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockImplementation(
      () => edited
    );
    (ports.sceneManager.loadSceneData as ReturnType<typeof vi.fn>).mockImplementation(
      (_src: string, _config: unknown, options: { beforeFrame?: (root: THREE.Group) => void }) => {
        const root = new THREE.Group();
        root.userData.contentHash = 'hash-b';
        options.beforeFrame?.(root);
        edited = true;
      }
    );
    await loadDataset('scene.zarr', ports);

    expect(ports.renderingControls.applyZarrDefaults).not.toHaveBeenCalled();
  });

  it('resets to the new build defaults instead of keeping the stale edits', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (ports.renderingControls.adoptSceneContentHash as ReturnType<typeof vi.fn>).mockReturnValue(
      true
    );
    const frame = vi.fn();
    (ports.sceneManager.loadSceneData as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _src: string,
        _config: unknown,
        options: {
          beforeFrame?: (root: THREE.Group) => void;
          applyViewerConfigFov: boolean;
        }
      ) => {
        const root = new THREE.Group();
        root.userData.contentHash = 'hash-b';
        root.userData.viewerConfig = { theme: 'dark' };
        options.beforeFrame?.(root);
        frame(options.applyViewerConfigFov, trace.order.includes('resetToDefaults'));
      }
    );
    await loadDataset('scene.zarr', ports);

    expect(frame).toHaveBeenCalledExactlyOnceWith(true, true);
    expect(ports.renderingControls.adoptSceneContentHash).toHaveBeenCalledWith('hash-b');
    expect(trace.order).not.toContain('syncCameraFovState');
  });

  it('keeps edits saved on this same build', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    await loadDataset('scene.zarr', ports);

    expect(trace.order).not.toContain('resetToDefaults');
    expect(trace.order).not.toContain('applyZarrDefaults');
    expect(trace.order).toContain('syncCameraFovState');
  });
});
