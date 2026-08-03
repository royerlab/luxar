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

vi.mock('../../../../../data/scene-loader-manager', () => ({
  getSceneLoader: vi.fn(),
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
    setSceneId: vi.fn(() => trace.order.push('setSceneId')),
    setZarrViewerConfig: vi.fn(() => trace.order.push('setZarrViewerConfig')),
    hasStoredSettings: vi.fn().mockReturnValue(false),
    applyZarrDefaults: vi.fn(() => trace.order.push('applyZarrDefaults')),
    updateSceneScale: vi.fn(() => trace.order.push('updateSceneScale')),
  };
  const sceneManager = {
    loadSceneData: vi.fn().mockImplementation(async () => trace.order.push('loadSceneData')),
    getSceneViewerConfig: vi.fn().mockReturnValue(viewerConfig),
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

  it('applyZarrDefaults runs only when hasStoredSettings=false AND viewerConfig present', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    expect(trace.order).toContain('applyZarrDefaults');
  });

  it('applyZarrDefaults is SKIPPED when hasStoredSettings=true', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const ports = makePorts(trace);
    (ports.renderingControls.hasStoredSettings as ReturnType<typeof vi.fn>).mockReturnValue(true);
    await loadDataset('scene.zarr', ports);

    expect(ports.renderingControls.applyZarrDefaults).not.toHaveBeenCalled();
  });

  it('layers panel hydration runs only when layersPanel present AND sceneLoader has sceneGraph', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const initFromScene = vi.fn();
    const ports = makePorts(trace, {
      layersPanel: { initFromScene, setFailedLoadsProvider: vi.fn() } as never,
    });
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    ports.sceneManager.scene.children = [root];
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
      getFailedLoadsProvider: vi.fn(() => ({ getFailedPaths: () => [], retryAll: vi.fn() })),
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
      layersPanel: { initFromScene, setFailedLoadsProvider } as never,
    });
    const root = new THREE.Group();
    root.name = 'LuxarScene';
    ports.sceneManager.scene.children = [root];
    const provider = { getFailedPaths: () => [], retryAll: vi.fn() };
    const getFailedLoadsProvider = vi.fn(() => provider);
    (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      sceneGraph: { kind: 'graph' },
      getFailedLoadsProvider,
    });

    await loadDataset('scene.zarr', ports);

    // The panel got exactly the object the loader handed out…
    expect(setFailedLoadsProvider).toHaveBeenCalledExactlyOnceWith(provider);
    // …and strictly after initFromScene (load-ordering contract).
    expect(setFailedLoadsProvider.mock.invocationCallOrder[0]).toBeGreaterThan(
      initFromScene.mock.invocationCallOrder[0]
    );
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

  it('animationController.startAnimation is the LAST step', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    await loadDataset('scene.zarr', makePorts(trace));

    expect(trace.order[trace.order.length - 1]).toBe('startAnimation');
  });

  it('loaderConfig is forwarded to sceneManager.loadSceneData', async () => {
    const trace: Trace = { order: [], recordedViewerConfig: undefined };
    const loaderConfig = { prefetch: true } as never;
    const ports = makePorts(trace, { loaderConfig });
    await loadDataset('scene.zarr', ports);

    expect(ports.sceneManager.loadSceneData).toHaveBeenCalledExactlyOnceWith(
      'scene.zarr',
      loaderConfig
    );
  });
});
