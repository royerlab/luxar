/**
 * Tests for LuxarLayer — Luxar rendered inside a host-owned Three.js pipeline.
 *
 * Runs in the `node` environment with `window`/`document` stubbed, rather than
 * under jsdom: the layer owns no DOM (that is the whole point of it), and the
 * only thing it asks of the browser is that those globals exist. Avoiding jsdom
 * also keeps this file runnable below the repo's Node 22.16 test floor.
 *
 * The layer is almost entirely wiring, so these tests target the contract a
 * host depends on rather than re-testing the subsystems it wires:
 *   - ordering constraints that are easy to regress and silent when broken
 *     (capabilities before any node; dims resolved after attach, before the
 *     first slice),
 *   - the nD update coalescing, which exists because a scrubbing host outruns
 *     the loader by ~30x,
 *   - teardown actually reaching the process singletons (a leak here strands
 *     the whole data-worker pool on every host remount),
 *   - the no-op guarantees a host relies on when calling `update()` every
 *     frame from before load until after dispose.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import type { DimensionMetadata, SimpleDims } from '../../../types/dims';

// The layer's only browser requirement: these must exist.
vi.stubGlobal('window', globalThis.window ?? {});
vi.stubGlobal('document', globalThis.document ?? {});

const loadSceneMock = vi.fn();
const updateSceneForDimensionsMock = vi.fn();
const prefetchSceneForDimensionsMock = vi.fn();

vi.mock('../../../data/zarr-loader', () => ({
  loadScene: (...args: unknown[]) => loadSceneMock(...args),
  updateSceneForDimensions: (...args: unknown[]) => updateSceneForDimensionsMock(...args),
  prefetchSceneForDimensions: (...args: unknown[]) => prefetchSceneForDimensionsMock(...args),
}));

const setLODGroupRegistryFactory = vi.fn();
const setRequestRender = vi.fn();
const setKTX2TextureDecoder = vi.fn();
const disposeInstance = vi.fn();
const getProfiler = vi.fn();
function makeSceneLoaderStub() {
  let archiveFault: Error | null = null;
  const archiveFaultListeners = new Set<(error: Error) => void>();
  return {
    lodGroupRegistry: { evaluatePerFrame: vi.fn(() => false) },
    nodeFactory: { rebuildAfterContextRestore: vi.fn() },
    isUpdateInProgress: vi.fn(() => false),
    updateView: vi.fn(),
    get archiveFault() {
      return archiveFault;
    },
    onArchiveFault: vi.fn(
      (listener: (error: Error) => void, options: { replayCurrent?: boolean } = {}) => {
        archiveFaultListeners.add(listener);
        if (options.replayCurrent && archiveFault) listener(archiveFault);
        return () => archiveFaultListeners.delete(listener);
      }
    ),
    emitArchiveFault(error: Error) {
      archiveFault = error;
      for (const listener of [...archiveFaultListeners]) listener(error);
    },
    resetArchiveFault() {
      archiveFault = null;
      archiveFaultListeners.clear();
    },
  };
}

const sceneLoaderStub = makeSceneLoaderStub();
let currentSceneLoaderStub = sceneLoaderStub;

const destroyAllAsync = vi.fn(async () => {});
const destroyLoaderAsync = vi.fn(async (_id: string) => {});

vi.mock('../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: () => ({
      setLODGroupRegistryFactory,
      setRequestRender,
      setKTX2TextureDecoder,
      getProfiler,
      destroyAllAsync: () => destroyAllAsync(),
      destroyLoaderAsync: (id: string) => destroyLoaderAsync(id),
    }),
    disposeInstance: () => disposeInstance(),
  },
  getSceneLoader: () => currentSceneLoaderStub,
}));

vi.mock('../../../scene/lod-group-registry', () => ({
  LODGroupRegistry: class {
    constructor(public deps: unknown) {}
  },
}));

const dimsState: SimpleDims = {
  ndim: 4,
  displayed: [0, 1, 2],
  currentStep: [0, 0, 0, 0],
  metadata: [],
};
const initFromScene = vi.fn();
const setDimensionValueMock = vi.fn();
const resetDims = vi.fn();
const getDimsMock = vi.fn((): SimpleDims | null => dimsState);
const getDimensionMetadataMock = vi.fn((): DimensionMetadata[] => dimsState.metadata ?? []);
const getDimensionRangesMock = vi.fn((): Array<[number, number]> => []);

vi.mock('../../../scene/scene-dims-manager', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../scene/scene-dims-manager')>();
  return {
    ...actual,
    sceneDimsManager: {
      initFromScene: (...a: unknown[]) => initFromScene(...a),
      getDims: () => getDimsMock(),
      getDimensionNames: () => ['x', 'y', 'z', 'time'],
      getDimensionMetadata: () => getDimensionMetadataMock(),
      getDimensionRanges: () => getDimensionRangesMock(),
      setDimensionValue: (...a: unknown[]) => setDimensionValueMock(...a),
      reset: () => resetDims(),
    },
  };
});

const setCaps = vi.fn();
const updateCameraParams = vi.fn();
const disposeMaterials = vi.fn();
const rebuildMaterials = vi.fn();
vi.mock('../../../rendering/material-manager', () => ({
  materialManager: {
    setCaps: (...a: unknown[]) => setCaps(...a),
    updateCameraParams: (...a: unknown[]) => updateCameraParams(...a),
    register: vi.fn(),
    dispose: () => disposeMaterials(),
    rebuildAfterContextRestore: () => rebuildMaterials(),
  },
}));

vi.mock('../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: () => ({ backend: 'webgl', apiSurface: 'webgl2' }),
  isWebGLRenderer: (renderer: { isWebGLRenderer?: boolean }) => renderer.isWebGLRenderer === true,
}));
const reduceGpuByteBudgetForContextLoss = vi.fn();
vi.mock('../../../rendering/gpu-byte-budget', () => ({
  getGpuByteBudget: () => 1024,
  reduceGpuByteBudgetForContextLoss: () => reduceGpuByteBudgetForContextLoss(),
}));

const configureBlendModeProgramWarmup = vi.fn();
const warmSceneBlendModePrograms = vi.fn(async (_root: THREE.Object3D) => {});
const clearBlendModeProgramWarmup = vi.fn();
vi.mock('../../../rendering/webgl-blend-warmup', () => ({
  configureBlendModeProgramWarmup: (...a: unknown[]) => configureBlendModeProgramWarmup(...a),
  warmSceneBlendModePrograms: (root: THREE.Object3D) => warmSceneBlendModePrograms(root),
  clearBlendModeProgramWarmup: () => clearBlendModeProgramWarmup(),
}));

const configureDepthSort = vi.fn();
const setDepthSortEnabled = vi.fn();
const evaluateDepthSortPerFrame = vi.fn();
const warmUpDepthSortWorker = vi.fn();
const releaseDepthSortNode = vi.fn();
const disposeDepthSort = vi.fn();
vi.mock('../../../rendering/depth-sort-coordinator', () => ({
  configureDepthSort: (...a: unknown[]) => configureDepthSort(...a),
  setDepthSortEnabled: (...a: unknown[]) => setDepthSortEnabled(...a),
  evaluateDepthSortPerFrame: () => evaluateDepthSortPerFrame(),
  warmUpDepthSortWorker: () => warmUpDepthSortWorker(),
  releaseDepthSortNode: (mesh: THREE.Mesh) => releaseDepthSortNode(mesh),
  disposeDepthSort: () => disposeDepthSort(),
}));

const disposeWorkerPool = vi.fn();
vi.mock('../../../workers/worker-pool', () => ({ disposeWorkerPool: () => disposeWorkerPool() }));

const applyModuleOverrides = vi.fn();
vi.mock('../../../core/app/init/module-overrides', () => ({
  applyModuleOverrides: (...a: unknown[]) => applyModuleOverrides(...a),
}));

import { LuxarLayer, type LuxarLayerOptions } from '../../../core/layer/luxar-layer';

function makeOptions(overrides: Partial<LuxarLayerOptions> = {}): LuxarLayerOptions {
  const renderer = {
    isWebGLRenderer: true,
    getDrawingBufferSize: (v: THREE.Vector2) => v.set(800, 600),
  } as unknown as LuxarLayerOptions['renderer'];
  return {
    renderer,
    getCamera: () => new THREE.PerspectiveCamera(50, 1, 0.1, 100),
    getViewportSize: () => ({ width: 800, height: 600 }),
    scene: new THREE.Scene(),
    ...overrides,
  };
}

describe('LuxarLayer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sceneLoaderStub.resetArchiveFault();
    currentSceneLoaderStub = sceneLoaderStub;
    loadSceneMock.mockImplementation(async () => new THREE.Group());
    updateSceneForDimensionsMock.mockResolvedValue(undefined);
    getDimsMock.mockReturnValue(dimsState);
    getDimensionMetadataMock.mockReturnValue(dimsState.metadata ?? []);
    getDimensionRangesMock.mockReturnValue([]);
  });

  describe('construction', () => {
    it('pushes renderer capabilities into materials before any node can be built', () => {
      new LuxarLayer(makeOptions());
      expect(setCaps).toHaveBeenCalledTimes(1);
      // The GLSL/TSL dispatch branches on caps, so a node built before this
      // would silently get the wrong backend.
      expect(setCaps.mock.invocationCallOrder[0]).toBeLessThan(
        setLODGroupRegistryFactory.mock.invocationCallOrder[0]
      );
    });

    it('seeds camera params from the host camera', () => {
      new LuxarLayer(makeOptions());
      expect(updateCameraParams).toHaveBeenCalledWith(
        expect.any(Number),
        expect.any(THREE.Vector2),
        false
      );
    });

    it('forwards wasm/worker path overrides', () => {
      new LuxarLayer(makeOptions({ wasmPath: '/w.js', workerPath: '/dw.js' }));
      expect(applyModuleOverrides).toHaveBeenCalledWith({
        wasmPath: '/w.js',
        workerPath: '/dw.js',
      });
    });

    it('warms the depth-sort worker while the page is still idle', () => {
      new LuxarLayer(makeOptions());
      expect(warmUpDepthSortWorker).toHaveBeenCalled();
    });

    it('installs the renderer-owned KTX2 decoder on the loader manager', () => {
      new LuxarLayer(makeOptions());
      expect(setKTX2TextureDecoder).toHaveBeenCalledWith(expect.any(Function));
    });

    it('configures blend-program warm-up against the host WebGL pipeline', async () => {
      const options = makeOptions();
      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/scene.zarr');
      expect(configureBlendModeProgramWarmup).toHaveBeenCalledWith({
        enabled: true,
        renderer: options.renderer,
        camera: expect.any(THREE.Camera),
        targetScene: options.scene,
      });
    });

    it('disables WebGL blend-program warm-up for a WebGPU renderer', async () => {
      const options = makeOptions({
        renderer: {
          getDrawingBufferSize: (v: THREE.Vector2) => v.set(800, 600),
        } as unknown as LuxarLayerOptions['renderer'],
      });
      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/scene.zarr');

      expect(configureBlendModeProgramWarmup).toHaveBeenCalledWith({
        enabled: false,
        renderer: null,
        camera: expect.any(THREE.Camera),
        targetScene: options.scene,
      });
    });

    it('skips depth-sort wiring when disabled', () => {
      new LuxarLayer(makeOptions({ depthSort: false }));
      expect(configureDepthSort).not.toHaveBeenCalled();
      expect(warmUpDepthSortWorker).not.toHaveBeenCalled();
      // Not wiring is not the same as disabling: the module-level flag gates
      // registration, the per-frame renderOrder pass, and worker spawn, so
      // returning early without clearing it leaves the option doing nothing.
      expect(setDepthSortEnabled).toHaveBeenCalledWith(false);
    });

    it('enables the depth-sort module flag by default', () => {
      new LuxarLayer(makeOptions());
      expect(setDepthSortEnabled).toHaveBeenCalledWith(true);
    });

    it('supplies every LOD-registry dep the eviction path needs', () => {
      // A dep that is merely ABSENT degrades silently: `lod-eviction` bails on
      // `!getResidentBytes`, so passing the budget without the measurement
      // makes the budget decorative and nothing is ever evicted. Snapshotting
      // the key set is the cheap way to catch a dep going missing.
      new LuxarLayer(makeOptions());
      const factory = setLODGroupRegistryFactory.mock.calls[0][0] as (o: unknown) => {
        deps: Record<string, unknown>;
      };
      const { deps } = factory({ currentViewVersion: 1, gpuBufferPool: undefined });

      expect(Object.keys(deps).sort()).toEqual(
        [
          'getCamera',
          'getCrossFadeEnabled',
          'getDisplayDims',
          'getEnergyCompEnabled',
          'getForceFinestLOD',
          'getResidentByteBudget',
          'getResidentBytes',
          'getViewVersion',
          'getViewportSize',
          'registerMaterial',
          'requestRender',
        ].sort()
      );
    });

    it('reports no resident bytes rather than throwing when the pool is absent', () => {
      new LuxarLayer(makeOptions());
      const factory = setLODGroupRegistryFactory.mock.calls[0][0] as (o: unknown) => {
        deps: { getResidentBytes: () => number };
      };
      const { deps } = factory({ currentViewVersion: 1, gpuBufferPool: undefined });
      expect(deps.getResidentBytes()).toBe(0);
    });

    it('reports no display dims before the scene resolves them', () => {
      // NOT [0, 1, 2]: that default projects a 2D scene onto a phantom Z. An
      // empty list trips the registry's own `length < 2` early return instead.
      getDimsMock.mockReturnValueOnce(null);
      new LuxarLayer(makeOptions());
      const factory = setLODGroupRegistryFactory.mock.calls[0][0] as (o: unknown) => {
        deps: { getDisplayDims: () => number[] };
      };
      const { deps } = factory({ currentViewVersion: 1 });
      expect(deps.getDisplayDims()).toEqual([]);
    });

    it('threads the LOD flags through to the registry', () => {
      new LuxarLayer(makeOptions({ lodFade: false, lodEnergyComp: false, lodFinest: true }));
      const factory = setLODGroupRegistryFactory.mock.calls[0][0] as (o: unknown) => {
        deps: {
          getCrossFadeEnabled: () => boolean;
          getEnergyCompEnabled: () => boolean;
          getForceFinestLOD: () => boolean;
        };
      };
      const { deps } = factory({ currentViewVersion: 1 });
      expect(deps.getCrossFadeEnabled()).toBe(false);
      expect(deps.getEnergyCompEnabled()).toBe(false);
      expect(deps.getForceFinestLOD()).toBe(true);
    });
  });

  describe('load', () => {
    it('delivers terminal dataset faults and replays them to late subscribers', async () => {
      const layer = new LuxarLayer(makeOptions());
      const earlyListener = vi.fn();
      layer.onDatasetFault(earlyListener);
      await layer.load('http://example.test/scene.zarr');

      const fault = new Error('archive unavailable');
      sceneLoaderStub.emitArchiveFault(fault);

      expect(earlyListener).toHaveBeenCalledOnce();
      expect(earlyListener).toHaveBeenCalledWith(fault);
      expect(layer.getDatasetFault()).toBe(fault);

      const lateListener = vi.fn();
      layer.onDatasetFault(lateListener);
      expect(lateListener).toHaveBeenCalledOnce();
      expect(lateListener).toHaveBeenCalledWith(fault);
    });

    it('replays a fault latched during load and replaces the loader subscription on switch', async () => {
      const firstLoader = makeSceneLoaderStub();
      const secondLoader = makeSceneLoaderStub();
      const firstFault = new Error('first archive unavailable');
      const secondFault = new Error('second archive unavailable');
      currentSceneLoaderStub = firstLoader;
      loadSceneMock.mockImplementationOnce(async () => {
        firstLoader.emitArchiveFault(firstFault);
        return new THREE.Group();
      });
      const listener = vi.fn();
      const layer = new LuxarLayer(makeOptions());
      layer.onDatasetFault(listener);

      await layer.load('http://example.test/a.zarr');
      expect(listener).toHaveBeenCalledWith(firstFault);
      expect(layer.getDatasetFault()).toBe(firstFault);

      loadSceneMock.mockImplementationOnce(async () => {
        currentSceneLoaderStub = secondLoader;
        return new THREE.Group();
      });
      await layer.load('http://example.test/b.zarr');
      firstLoader.emitArchiveFault(new Error('stale archive unavailable'));
      secondLoader.emitArchiveFault(secondFault);

      expect(listener).toHaveBeenCalledTimes(2);
      expect(listener).toHaveBeenLastCalledWith(secondFault);
      expect(layer.getDatasetFault()).toBe(secondFault);
    });

    it('isolates a throwing dataset fault listener during replay', async () => {
      const fault = new Error('archive unavailable');
      loadSceneMock.mockImplementationOnce(async () => {
        sceneLoaderStub.emitArchiveFault(fault);
        return new THREE.Group();
      });
      const layer = new LuxarLayer(makeOptions());
      layer.onDatasetFault(() => {
        throw new Error('listener boom');
      });
      const healthyListener = vi.fn();
      layer.onDatasetFault(healthyListener);

      await expect(layer.load('http://example.test/scene.zarr')).resolves.toBeInstanceOf(
        THREE.Group
      );
      expect(healthyListener).toHaveBeenCalledWith(fault);
    });

    it('attaches the root, then resolves dims, then requests the first slice', async () => {
      const layer = new LuxarLayer(makeOptions());
      const root = await layer.load('http://example.test/scene.zarr');

      expect(layer.root).toBe(root);
      expect(root.parent).not.toBeNull();
      // Dims are read off the scene, so attach must precede initFromScene, and
      // the first slice query reads the displayed-dims set it produces.
      expect(initFromScene.mock.invocationCallOrder[0]).toBeGreaterThan(0);
      expect(updateSceneForDimensionsMock.mock.invocationCallOrder[0]).toBeGreaterThan(
        initFromScene.mock.invocationCallOrder[0]
      );
    });

    it('detaches the previous root on a second load', async () => {
      // `loadScene` disposes the previous SceneLoader, so a stale group left
      // attached keeps the host drawing over disposed backing stores.
      const options = makeOptions();
      const first = new THREE.Group();
      const nested = new THREE.Group();
      const oldMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      const disposeGeometry = vi.spyOn(oldMesh.geometry, 'dispose');
      const disposeMaterial = vi.spyOn(oldMesh.material, 'dispose');
      nested.add(oldMesh);
      first.add(nested);
      const second = new THREE.Group();
      const liveMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      second.add(liveMesh);
      loadSceneMock.mockImplementationOnce(async () => first);
      loadSceneMock.mockImplementationOnce(async () => second);

      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/a.zarr');
      await layer.load('http://example.test/b.zarr');

      expect(options.scene.children).toContain(second);
      expect(options.scene.children).not.toContain(first);
      expect(layer.root).toBe(second);
      expect(releaseDepthSortNode).toHaveBeenCalledWith(oldMesh);
      expect(releaseDepthSortNode).not.toHaveBeenCalledWith(liveMesh);
      expect(disposeGeometry).toHaveBeenCalledTimes(1);
      expect(disposeMaterial).not.toHaveBeenCalled();
    });

    it('detaches the previous root when a dataset switch fails', async () => {
      // The failed switch has already disposed the previous SceneLoader, so
      // retaining its root would expose geometry backed by dead resources.
      const options = makeOptions();
      const first = new THREE.Group();
      const oldMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      const disposeGeometry = vi.spyOn(oldMesh.geometry, 'dispose');
      first.add(oldMesh);
      loadSceneMock.mockImplementationOnce(async () => first);
      loadSceneMock.mockRejectedValueOnce(new Error('dataset unavailable'));

      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/a.zarr');
      await expect(layer.load('http://example.test/b.zarr')).rejects.toThrow('dataset unavailable');

      expect(options.scene.children).not.toContain(first);
      expect(layer.root).toBeNull();
      expect(layer.getBounds()).toBeNull();
      expect(releaseDepthSortNode).toHaveBeenCalledWith(oldMesh);
      expect(disposeGeometry).toHaveBeenCalledTimes(1);
    });

    it('detaches the new root when its first slice fails', async () => {
      const options = makeOptions();
      const first = new THREE.Group();
      const second = new THREE.Group();
      const newMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      const disposeGeometry = vi.spyOn(newMesh.geometry, 'dispose');
      second.add(newMesh);
      loadSceneMock.mockResolvedValueOnce(first).mockResolvedValueOnce(second);
      updateSceneForDimensionsMock
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error('slice unavailable'));

      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/a.zarr');
      await expect(layer.load('http://example.test/b.zarr')).rejects.toThrow('slice unavailable');

      expect(options.scene.children).not.toContain(first);
      expect(options.scene.children).not.toContain(second);
      expect(layer.root).toBeNull();
      expect(layer.getBounds()).toBeNull();
      expect(releaseDepthSortNode).toHaveBeenCalledWith(newMesh);
      expect(disposeGeometry).toHaveBeenCalledTimes(1);
      expect(warmSceneBlendModePrograms).not.toHaveBeenCalledWith(second);
    });

    it('refuses a concurrent load rather than racing two loaders', async () => {
      // The second load's createLoaderAsync disposes the first's loader
      // mid-flight, and whichever resolves LAST wins the root slot — so the
      // loser can leave a group backed by a disposed loader in the host scene.
      let release!: (g: THREE.Group) => void;
      loadSceneMock.mockImplementationOnce(() => new Promise<THREE.Group>((r) => (release = r)));
      const layer = new LuxarLayer(makeOptions());

      const first = layer.load('http://example.test/a.zarr');
      await expect(layer.load('http://example.test/b.zarr')).rejects.toThrow(/already in progress/);

      release(new THREE.Group());
      await first;
      // The guard clears, so a sequential switch still works.
      await expect(layer.load('http://example.test/b.zarr')).resolves.toBeInstanceOf(THREE.Group);
    });

    it('forwards loaderConfig to the loader', async () => {
      const loaderConfig = { noCache: true } as LuxarLayerOptions['loaderConfig'];
      const layer = new LuxarLayer(makeOptions({ loaderConfig }));
      await layer.load('http://example.test/scene.zarr');
      expect(loadSceneMock).toHaveBeenCalledWith(
        'http://example.test/scene.zarr',
        loaderConfig,
        expect.any(String)
      );
    });

    it('applies the configured renderOrder through nested and lazy groups', async () => {
      const root = new THREE.Group();
      const sceneGroup = new THREE.Group();
      const lodGroup = new THREE.Group();
      const mesh = new THREE.Mesh();
      mesh.renderOrder = 7;
      lodGroup.add(mesh);
      sceneGroup.add(lodGroup);
      root.add(sceneGroup);
      loadSceneMock.mockResolvedValueOnce(root);
      const layer = new LuxarLayer(makeOptions({ renderOrder: 42 }));
      await layer.load('http://example.test/scene.zarr');

      expect(root.renderOrder).toBe(42);
      expect(sceneGroup.renderOrder).toBe(42);
      expect(lodGroup.renderOrder).toBe(42);
      expect(mesh.renderOrder).toBe(7);

      const lazyPartition = new THREE.Group();
      lodGroup.add(lazyPartition);
      const commit = setRequestRender.mock.calls[0][0] as () => void;
      commit();
      layer.update();
      expect(lazyPartition.renderOrder).toBe(42);
    });

    it('warms reachable blend programs after the first slice commits', async () => {
      const root = new THREE.Group();
      loadSceneMock.mockResolvedValueOnce(root);
      const layer = new LuxarLayer(makeOptions());

      await layer.load('http://example.test/scene.zarr');

      expect(warmSceneBlendModePrograms).toHaveBeenCalledWith(root);
      expect(updateSceneForDimensionsMock.mock.invocationCallOrder[0]).toBeLessThan(
        warmSceneBlendModePrograms.mock.invocationCallOrder[0]
      );
    });

    it('does not arm blend warm-up when disposal starts during the first slice', async () => {
      const root = new THREE.Group();
      loadSceneMock.mockResolvedValueOnce(root);
      let releaseSlice!: () => void;
      updateSceneForDimensionsMock.mockImplementationOnce(
        () => new Promise<void>((resolve) => (releaseSlice = resolve))
      );
      const layer = new LuxarLayer(makeOptions());

      const loading = layer.load('http://example.test/scene.zarr');
      while (!releaseSlice) await Promise.resolve();
      const disposing = layer.dispose();
      releaseSlice();
      await Promise.all([loading, disposing]);

      expect(configureBlendModeProgramWarmup).toHaveBeenCalledTimes(1);
      expect(configureBlendModeProgramWarmup).toHaveBeenCalledWith({
        enabled: false,
        renderer: null,
        camera: null,
        targetScene: null,
      });
      expect(warmSceneBlendModePrograms).not.toHaveBeenCalled();
    });

    it('waits for and cleans up a load before tearing down globals', async () => {
      const options = makeOptions();
      let release: (g: THREE.Group) => void = () => {};
      let finishLoaderCleanup: () => void = () => {};
      loadSceneMock.mockImplementation(() => new Promise((r) => (release = r as typeof release)));
      destroyLoaderAsync.mockImplementationOnce(
        () => new Promise<void>((resolve) => (finishLoaderCleanup = resolve))
      );

      const layer = new LuxarLayer(options);
      const pending = layer.load('http://example.test/scene.zarr');
      const disposing = layer.dispose();
      await Promise.resolve();
      expect(destroyAllAsync).not.toHaveBeenCalled();
      expect(disposeMaterials).not.toHaveBeenCalled();

      const abandonedRoot = new THREE.Group();
      const abandonedMesh = new THREE.Mesh(
        new THREE.BufferGeometry(),
        new THREE.MeshBasicMaterial()
      );
      const disposeGeometry = vi.spyOn(abandonedMesh.geometry, 'dispose');
      abandonedRoot.add(abandonedMesh);
      release(abandonedRoot);
      await Promise.resolve();
      await Promise.resolve();
      expect(destroyLoaderAsync).toHaveBeenCalledWith('default');
      expect(disposeMaterials).not.toHaveBeenCalled();

      finishLoaderCleanup();
      await Promise.all([pending, disposing]);

      expect(options.scene.children).toHaveLength(0);
      expect(releaseDepthSortNode).toHaveBeenCalledWith(abandonedMesh);
      expect(disposeGeometry).toHaveBeenCalledTimes(1);
      expect(destroyLoaderAsync.mock.invocationCallOrder[0]).toBeLessThan(
        disposeMaterials.mock.invocationCallOrder[0]
      );
    });
  });

  describe('update', () => {
    it('is a no-op before load', () => {
      new LuxarLayer(makeOptions()).update();
      expect(evaluateDepthSortPerFrame).not.toHaveBeenCalled();
    });

    it('sorts before evaluating LOD once loaded', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      layer.update();

      // A LOD swap can invalidate the cross-node order sorting just assigned,
      // so sorting has to run first.
      expect(evaluateDepthSortPerFrame.mock.invocationCallOrder[0]).toBeLessThan(
        (sceneLoaderStub.lodGroupRegistry.evaluatePerFrame as ReturnType<typeof vi.fn>).mock
          .invocationCallOrder[0]
      );
    });

    it('only re-stamps render order after a geometry commit', async () => {
      const root = new THREE.Group();
      const traverse = vi.spyOn(root, 'traverse');
      loadSceneMock.mockResolvedValueOnce(root);
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      traverse.mockClear();

      layer.update();
      expect(traverse).not.toHaveBeenCalled();

      const lazyGroup = new THREE.Group();
      root.add(lazyGroup);
      const commit = setRequestRender.mock.calls[0][0] as () => void;
      commit();
      layer.update();
      expect(traverse).toHaveBeenCalledTimes(1);
      expect(lazyGroup.renderOrder).toBe(10);

      traverse.mockClear();
      layer.update();
      expect(traverse).not.toHaveBeenCalled();
    });

    it('is a no-op after dispose', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();
      vi.clearAllMocks();

      layer.update();
      expect(evaluateDepthSortPerFrame).not.toHaveBeenCalled();
    });
  });

  describe('nD navigation', () => {
    it('finds a dimension by name, case-insensitively', () => {
      const layer = new LuxarLayer(makeOptions());
      expect(layer.findDimension('TIME')).toBe(3);
      expect(layer.findDimension('nope')).toBeNull();
    });

    it('coalesces updates issued while one is in flight', async () => {
      // Every call to the loader parks a resolver here, so the test can
      // release passes one at a time regardless of when the drain loop issues
      // the next one.
      const passes: Array<() => void> = [];
      updateSceneForDimensionsMock.mockImplementation(
        () => new Promise<void>((resolve) => passes.push(resolve))
      );
      const releaseNextPass = async () => {
        // Wait for the pass to have been issued, then let it commit.
        while (passes.length === 0) await Promise.resolve();
        passes.shift()!();
        await Promise.resolve();
      };

      const layer = new LuxarLayer(makeOptions());
      const loaded = layer.load('http://example.test/scene.zarr');
      await releaseNextPass();
      await loaded;
      updateSceneForDimensionsMock.mockClear();

      // Three scrub steps, the latter two landing while the first is still in
      // flight.
      const first = layer.setDimensionValue(3, 1);
      const second = layer.setDimensionValue(3, 2);
      const third = layer.setDimensionValue(3, 3);
      expect(updateSceneForDimensionsMock).toHaveBeenCalledTimes(1);

      await releaseNextPass();
      await first;

      await releaseNextPass();
      await Promise.all([second, third]);

      // One pass for the in-flight value, one for the coalesced latest —
      // NOT one per call.
      expect(updateSceneForDimensionsMock).toHaveBeenCalledTimes(2);
      // Every requested value still reached the dims manager, so the final
      // position is correct even though intermediate slices were skipped.
      expect(setDimensionValueMock).toHaveBeenCalledWith(3, 3);
    });

    it('prefetches a predicted value without moving the view', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      updateSceneForDimensionsMock.mockClear();

      layer.prefetchDimensionValue(3, 7, 12);

      expect(prefetchSceneForDimensionsMock).toHaveBeenCalledWith(
        expect.objectContaining({ currentStep: [0, 0, 0, 7] }),
        expect.anything(),
        'default',
        { budgetMs: 12 }
      );
      expect(updateSceneForDimensionsMock).not.toHaveBeenCalled();
    });

    it('clamps and snaps prefetches to the same discrete value as foreground updates', async () => {
      const metadata: DimensionMetadata[] = [
        { name: 'x', unit: '', scale: 1 },
        { name: 'y', unit: '', scale: 1 },
        { name: 'z', unit: '', scale: 1 },
        {
          name: 'time',
          unit: 'frame',
          scale: 1,
          discrete: true,
          step: 5,
          range: [1, 12],
        },
      ];
      getDimsMock.mockReturnValue({ ...dimsState, metadata });
      getDimensionRangesMock.mockReturnValue([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 12],
      ]);
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.prefetchDimensionValue(3, 16);

      expect(prefetchSceneForDimensionsMock).toHaveBeenCalledWith(
        expect.objectContaining({ currentStep: [0, 0, 0, 11] }),
        expect.anything(),
        'default',
        { budgetMs: 16 }
      );
    });

    it('does not strand a later update when dimension metadata is absent', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      getDimsMock.mockReturnValue(null);

      await layer.setDimensionValue(3, 1);
      await expect(layer.setDimensionValue(3, 2)).resolves.toBeUndefined();
    });

    it('settles callers and the drain when a slice update rejects', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      updateSceneForDimensionsMock.mockRejectedValueOnce(new Error('slice fetch failed'));

      const update = layer.setDimensionValue(3, 1);

      await expect(update).resolves.toBeUndefined();
      await expect(layer.awaitDimensionUpdate()).resolves.toBeUndefined();
      await expect(layer.setDimensionValue(3, 2)).resolves.toBeUndefined();
    });

    it('deep-clones metadata and ranges returned to the host', async () => {
      const categories = ['a', 'b'];
      const range: [number, number] = [1, 12];
      const metadata: DimensionMetadata[] = [
        { name: 'time', unit: 'frame', scale: 1, range, categories },
      ];
      const ranges: Array<[number, number]> = [[1, 12]];
      getDimensionMetadataMock.mockReturnValue(metadata);
      getDimensionRangesMock.mockReturnValue(ranges);
      const layer = new LuxarLayer(makeOptions());

      const exposed = layer.getDimensions()!;
      exposed.metadata[0].range![0] = 99;
      exposed.metadata[0].categories![0] = 'changed';
      exposed.ranges[0][0] = 99;

      expect(range).toEqual([1, 12]);
      expect(categories).toEqual(['a', 'b']);
      expect(ranges).toEqual([[1, 12]]);
    });
  });

  describe('context restoration', () => {
    it('backs off the GPU budget and clears stale warm-up programs on context loss', () => {
      const requestRender = vi.fn();
      const layer = new LuxarLayer(makeOptions({ requestRender }));

      layer.handleContextLost();

      expect(clearBlendModeProgramWarmup).toHaveBeenCalledTimes(1);
      expect(reduceGpuByteBudgetForContextLoss).toHaveBeenCalledTimes(1);
      expect(requestRender).not.toHaveBeenCalled();
    });

    it('rebuilds layer-owned materials, geometry, registrations, and camera uniforms', async () => {
      const requestRender = vi.fn();
      const attribute = new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3);
      const mesh = new THREE.Mesh(new THREE.BufferGeometry().setAttribute('position', attribute));
      const root = new THREE.Group();
      root.add(mesh);
      loadSceneMock.mockResolvedValueOnce(root);
      const layer = new LuxarLayer(makeOptions({ requestRender }));
      await layer.load('http://example.test/scene.zarr');
      updateCameraParams.mockClear();
      configureBlendModeProgramWarmup.mockClear();
      warmSceneBlendModePrograms.mockClear();

      const versionBefore = attribute.version;
      layer.handleContextRestored();

      expect(rebuildMaterials).toHaveBeenCalledTimes(1);
      expect(attribute.version).toBeGreaterThan(versionBefore);
      expect(sceneLoaderStub.nodeFactory.rebuildAfterContextRestore).toHaveBeenCalledWith(root);
      expect(rebuildMaterials.mock.invocationCallOrder[0]).toBeLessThan(
        sceneLoaderStub.nodeFactory.rebuildAfterContextRestore.mock.invocationCallOrder[0]
      );
      expect(updateCameraParams).toHaveBeenCalledTimes(1);
      expect(configureBlendModeProgramWarmup).toHaveBeenCalledTimes(1);
      expect(warmSceneBlendModePrograms).toHaveBeenCalledWith(root);
      expect(requestRender).toHaveBeenCalled();
    });
  });

  describe('alignTo', () => {
    it('places the root without disturbing the host scene', async () => {
      const layer = new LuxarLayer(makeOptions());
      const root = await layer.load('http://example.test/scene.zarr');

      const m = new THREE.Matrix4().makeTranslation(1, 2, 3);
      layer.alignTo(m);

      expect(root.matrixAutoUpdate).toBe(false);
      expect(root.matrix.elements).toEqual(m.elements);
    });

    it('applies a placement declared before the scene finished loading', async () => {
      // A host derives its matrix from its own metadata, which resolves on a
      // schedule unrelated to the scene fetch. Dropping the earlier of the two
      // would leave the layer in the wrong place with nothing thrown.
      const layer = new LuxarLayer(makeOptions());
      const m = new THREE.Matrix4().makeTranslation(1, 2, 3);

      layer.alignTo(m);
      const root = await layer.load('http://example.test/scene.zarr');

      expect(root.matrixAutoUpdate).toBe(false);
      expect(root.matrix.elements).toEqual(m.elements);
    });

    it('does not alias the caller’s matrix', async () => {
      const layer = new LuxarLayer(makeOptions());
      const m = new THREE.Matrix4().makeTranslation(1, 2, 3);

      layer.alignTo(m);
      m.makeTranslation(9, 9, 9); // caller reuses its scratch matrix

      const root = await layer.load('http://example.test/scene.zarr');
      expect(root.matrix.elements).toEqual(new THREE.Matrix4().makeTranslation(1, 2, 3).elements);
    });
  });

  describe('resize', () => {
    it('reports an ortho frustum height and the ortho flag', () => {
      // A real branch: the ortho path feeds a frustum height rather than a FOV,
      // and materials size points from a different formula depending on it.
      const camera = new THREE.OrthographicCamera(-2, 2, 1.5, -1.5, 0.1, 100);
      const layer = new LuxarLayer(makeOptions({ getCamera: () => camera }));
      updateCameraParams.mockClear();

      layer.resize();

      expect(updateCameraParams).toHaveBeenCalledWith(expect.any(Number), expect.anything(), true);
      expect(updateCameraParams.mock.calls[0][0]).toBeCloseTo(3); // top - bottom
    });

    it('is a no-op after dispose', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.dispose();
      updateCameraParams.mockClear();
      layer.resize();
      expect(updateCameraParams).not.toHaveBeenCalled();
    });
  });

  describe('visibility and bounds', () => {
    it('hides via the root flag rather than detaching', async () => {
      // Detaching would drop caches and in-flight streams; keeping the root
      // attached lets lazy LOD loading resume after re-showing.
      const options = makeOptions();
      const layer = new LuxarLayer(options);
      const root = await layer.load('http://example.test/scene.zarr');

      layer.setVisible(false);
      expect(root.visible).toBe(false);
      expect(layer.isVisible()).toBe(false);
      expect(options.scene.children).toContain(root);

      layer.setVisible(true);
      expect(layer.isVisible()).toBe(true);
    });

    it('reports not-visible and null bounds before load', () => {
      const layer = new LuxarLayer(makeOptions());
      expect(layer.isVisible()).toBe(false);
      expect(layer.getBounds()).toBeNull();
    });

    it('preserves a hidden state across initial load and dataset switches', async () => {
      const first = new THREE.Group();
      const second = new THREE.Group();
      loadSceneMock.mockResolvedValueOnce(first).mockResolvedValueOnce(second);
      const layer = new LuxarLayer(makeOptions());

      layer.setVisible(false);
      await layer.load('http://example.test/a.zarr');
      await layer.load('http://example.test/b.zarr');

      expect(first.visible).toBe(false);
      expect(second.visible).toBe(false);
      expect(layer.isVisible()).toBe(false);
    });

    it('measures world bounds from the loaded root', async () => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2));
      loadSceneMock.mockImplementation(async () => {
        const g = new THREE.Group();
        g.add(mesh);
        return g;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      const box = layer.getBounds();
      expect(box).not.toBeNull();
      expect(box!.min.x).toBeCloseTo(-1);
      expect(box!.max.x).toBeCloseTo(1);
    });
  });

  describe('requestRender wiring', () => {
    it('reaches the loader manager and the registry', () => {
      const requestRender = vi.fn();
      new LuxarLayer(makeOptions({ requestRender }));

      const loaderCommit = setRequestRender.mock.calls[0][0] as () => void;
      loaderCommit();
      expect(requestRender).toHaveBeenCalledTimes(1);

      // A commit that repaints goes through the registry's own callback, so it
      // has to be threaded too — not just handed to the manager.
      const factory = setLODGroupRegistryFactory.mock.calls[0][0] as (o: unknown) => {
        deps: { requestRender: () => void };
      };
      factory({ currentViewVersion: 1 }).deps.requestRender();
      expect(requestRender).toHaveBeenCalledTimes(2);
    });

    it('still tracks geometry commits when the host renders continuously', () => {
      new LuxarLayer(makeOptions());
      expect(setRequestRender).toHaveBeenCalledWith(expect.any(Function));
    });
  });

  describe('awaitDimensionUpdate', () => {
    it('resolves immediately when nothing is in flight', async () => {
      const layer = new LuxarLayer(makeOptions());
      await expect(layer.awaitDimensionUpdate()).resolves.toBeUndefined();
    });

    it('waits for an in-flight slice update to settle', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr'); // resolves via the default mock

      // Only now make the next slice update block, so the load itself is not
      // the thing being awaited.
      let release!: () => void;
      updateSceneForDimensionsMock.mockImplementation(
        () => new Promise<void>((r) => (release = r))
      );
      void layer.setDimensionValue(3, 5);

      let settled = false;
      const waiter = layer.awaitDimensionUpdate().then(() => {
        settled = true;
      });
      await Promise.resolve();
      expect(settled).toBe(false);

      release();
      await waiter;
      expect(settled).toBe(true);
    });
  });

  describe('setExposure', () => {
    /** A stand-in for a Luxar geometry material: `uOpacity` plus the updaters. */
    function splatMesh(authored: number): THREE.Mesh {
      const mat = Object.assign(new THREE.Material(), {
        uniforms: { uOpacity: { value: authored } },
        updateOpacity(v: number) {
          this.uniforms.uOpacity.value = v;
        },
        getOpacity() {
          return this.uniforms.uOpacity.value;
        },
        updateIntensity() {},
        updateOffset() {},
        updateGamma() {},
      });
      return new THREE.Mesh(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
    }
    const opacityOf = (m: THREE.Mesh) =>
      (m.material as unknown as { uniforms: { uOpacity: { value: number } } }).uniforms.uOpacity
        .value;

    it('scales the authored opacity', async () => {
      const mesh = splatMesh(0.2);
      loadSceneMock.mockImplementation(async () => {
        const g = new THREE.Group();
        g.add(mesh);
        return g;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(2);
      expect(opacityOf(mesh)).toBeCloseTo(0.4);
      expect(layer.getExposure()).toBe(2);
    });

    it('applies exposure to an opaque mesh through its opacity uniform', async () => {
      const mesh = splatMesh(1);
      const material = mesh.material as unknown as {
        getBlendingMode: () => string;
        updateOpacity: (value: number) => void;
      };
      material.getBlendingMode = () => 'opaque';
      const updateOpacity = vi.spyOn(material, 'updateOpacity');
      loadSceneMock.mockImplementation(async () => {
        const root = new THREE.Group();
        root.add(mesh);
        return root;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(0.4);

      expect(updateOpacity).toHaveBeenCalledOnce();
      expect(updateOpacity).toHaveBeenCalledWith(0.4);
      expect(opacityOf(mesh)).toBeCloseTo(0.4);
    });

    it('composes against the authored value rather than compounding', async () => {
      // A slider emits a stream of values; scaling the LIVE opacity each time
      // would make the result depend on the drag path, not the final position.
      const mesh = splatMesh(0.2);
      loadSceneMock.mockImplementation(async () => {
        const g = new THREE.Group();
        g.add(mesh);
        return g;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(2);
      layer.setExposure(3);
      layer.setExposure(0.5);
      expect(opacityOf(mesh)).toBeCloseTo(0.1);

      layer.setExposure(1);
      expect(opacityOf(mesh)).toBeCloseTo(0.2); // back to authored
    });

    it('applies the exposure to geometry that streams in later', async () => {
      // LOD swaps and chunk commits mint materials after the slider moved; those
      // would otherwise render at the authored exposure next to adjusted ones.
      const present = splatMesh(0.2);
      const root = new THREE.Group();
      root.add(present);
      loadSceneMock.mockImplementation(async () => root);
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(0.5);
      const late = splatMesh(0.2);
      root.add(late);
      expect(opacityOf(late)).toBeCloseTo(0.2); // not yet seen

      layer.update();
      expect(opacityOf(late)).toBeCloseTo(0.1);
      expect(opacityOf(present)).toBeCloseTo(0.1);
    });

    it('takes the authored base from the fade snapshot, not a mid-fade uniform', async () => {
      // During a cross-fade `uOpacity` is `_lodFadeBase * product`. A material
      // first seen mid-fade would cache that fraction as its authored value —
      // permanently — and every later exposure would compound the dimming.
      // Likely rather than exotic: update() re-asserts exposure on nodes the
      // moment they commit, which is exactly when they begin fading in.
      const mesh = splatMesh(0.05); // live uniform: mid-fade
      mesh.userData._lodFadeBase = 0.2; // what it was authored with
      loadSceneMock.mockImplementation(async () => {
        const g = new THREE.Group();
        g.add(mesh);
        return g;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(2);
      expect(mesh.userData._lodFadeBase).toBeCloseTo(0.4); // 2 x authored 0.2
    });

    it('rebases an in-flight LOD fade instead of writing the uniform', async () => {
      // lod-fade.ts recomputes `_lodFadeBase x product` every frame, so a direct
      // write here is clobbered on the next fade frame and the edit lost.
      const mesh = splatMesh(0.2);
      mesh.userData._lodFadeBase = 0.2;
      loadSceneMock.mockImplementation(async () => {
        const g = new THREE.Group();
        g.add(mesh);
        return g;
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      layer.setExposure(0.5);
      expect(mesh.userData._lodFadeBase).toBeCloseTo(0.1);
      expect(opacityOf(mesh)).toBeCloseTo(0.2); // uniform left to the fade
    });

    it('is a no-op before load', () => {
      const layer = new LuxarLayer(makeOptions());
      expect(() => layer.setExposure(0.5)).not.toThrow();
      expect(layer.getExposure()).toBe(0.5);
    });
  });

  describe('dispose', () => {
    it('clears the current fault and unsubscribes host listeners', async () => {
      const listener = vi.fn();
      const layer = new LuxarLayer(makeOptions());
      layer.onDatasetFault(listener);
      await layer.load('http://example.test/scene.zarr');
      const fault = new Error('archive unavailable');
      sceneLoaderStub.emitArchiveFault(fault);
      expect(listener).toHaveBeenCalledOnce();

      await layer.dispose();
      sceneLoaderStub.emitArchiveFault(new Error('post-dispose archive unavailable'));

      expect(layer.getDatasetFault()).toBeNull();
      expect(listener).toHaveBeenCalledOnce();
    });

    it('detaches the root and tears down the process singletons', async () => {
      const options = makeOptions();
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      const disposeGeometry = vi.spyOn(mesh.geometry, 'dispose');
      loadSceneMock.mockImplementationOnce(async () => {
        const root = new THREE.Group();
        root.add(mesh);
        return root;
      });
      const layer = new LuxarLayer(options);
      await layer.load('http://example.test/scene.zarr');

      await layer.dispose();

      expect(options.scene.children).toHaveLength(0);
      expect(layer.root).toBeNull();
      // A host that remounts would otherwise strand the whole worker pool.
      expect(resetDims).toHaveBeenCalled();
      expect(disposeInstance).toHaveBeenCalled();
      expect(disposeMaterials).toHaveBeenCalled();
      expect(disposeWorkerPool).toHaveBeenCalled();
      expect(disposeDepthSort).toHaveBeenCalled();
      expect(releaseDepthSortNode).toHaveBeenCalledWith(mesh);
      expect(disposeGeometry).toHaveBeenCalledTimes(1);
    });

    it('releases the host pipeline from blend warm-up on teardown', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      await layer.dispose();

      expect(configureBlendModeProgramWarmup).toHaveBeenLastCalledWith({
        enabled: false,
        renderer: null,
        camera: null,
        targetScene: null,
      });
    });

    it('tears down the loader before the pools that serve it', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();

      expect(disposeInstance.mock.invocationCallOrder[0]).toBeLessThan(
        disposeWorkerPool.mock.invocationCallOrder[0]
      );
    });

    it('awaits the loader drain before dropping the singleton', async () => {
      // `disposeInstance()` alone fires disposes without awaiting them, so a
      // host awaiting dispose() would get a promise guaranteeing nothing and a
      // following load() could race the drain.
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();

      expect(destroyAllAsync).toHaveBeenCalled();
      expect(destroyAllAsync.mock.invocationCallOrder[0]).toBeLessThan(
        disposeInstance.mock.invocationCallOrder[0]
      );
    });

    it('awaits an in-flight dimension update before destroying its loader', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      let release!: () => void;
      updateSceneForDimensionsMock.mockImplementationOnce(
        () => new Promise<void>((resolve) => (release = resolve))
      );
      const update = layer.setDimensionValue(3, 1);

      const disposing = layer.dispose();
      await Promise.resolve();
      expect(destroyAllAsync).not.toHaveBeenCalled();

      release();
      await Promise.all([update, disposing]);
      expect(destroyAllAsync).toHaveBeenCalledTimes(1);
    });

    it('is idempotent', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();
      await layer.dispose();
      expect(disposeWorkerPool).toHaveBeenCalledTimes(1);
      expect(
        configureBlendModeProgramWarmup.mock.calls.filter(([config]) => !config.enabled)
      ).toHaveLength(1);
    });

    it('keeps tearing down after a step throws', async () => {
      disposeInstance.mockImplementationOnce(() => {
        throw new Error('boom');
      });
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');

      await expect(layer.dispose()).resolves.toBeUndefined();
      expect(disposeWorkerPool).toHaveBeenCalled();
    });

    it('rejects further loads', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.dispose();
      await expect(layer.load('http://example.test/scene.zarr')).rejects.toThrow('disposed');
    });
  });
});
