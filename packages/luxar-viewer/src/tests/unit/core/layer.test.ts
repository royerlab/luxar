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
const disposeInstance = vi.fn();
const getProfiler = vi.fn();
const sceneLoaderStub = {
  lodGroupRegistry: { evaluatePerFrame: vi.fn(() => false) },
  isUpdateInProgress: vi.fn(() => false),
  updateView: vi.fn(),
};

vi.mock('../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: () => ({ setLODGroupRegistryFactory, setRequestRender, getProfiler }),
    disposeInstance: () => disposeInstance(),
  },
  getSceneLoader: () => sceneLoaderStub,
}));

vi.mock('../../../scene/lod-group-registry', () => ({
  LODGroupRegistry: class {
    constructor(public deps: unknown) {}
  },
}));

const dimsState = {
  ndim: 4,
  displayed: [0, 1, 2],
  currentStep: [0, 0, 0, 0],
  metadata: [],
};
const initFromScene = vi.fn();
const setDimensionValueMock = vi.fn();
const resetDims = vi.fn();

vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    initFromScene: (...a: unknown[]) => initFromScene(...a),
    getDims: () => dimsState,
    getDimensionNames: () => ['x', 'y', 'z', 'time'],
    getDimensionMetadata: () => [],
    getDimensionRanges: () => [],
    setDimensionValue: (...a: unknown[]) => setDimensionValueMock(...a),
    reset: () => resetDims(),
  },
}));

const setCaps = vi.fn();
const updateCameraParams = vi.fn();
const disposeMaterials = vi.fn();
vi.mock('../../../rendering/material-manager', () => ({
  materialManager: {
    setCaps: (...a: unknown[]) => setCaps(...a),
    updateCameraParams: (...a: unknown[]) => updateCameraParams(...a),
    register: vi.fn(),
    dispose: () => disposeMaterials(),
  },
}));

vi.mock('../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: () => ({ backend: 'webgl' }),
}));
vi.mock('../../../rendering/gpu-byte-budget', () => ({ getGpuByteBudget: () => 1024 }));

const configureDepthSort = vi.fn();
const evaluateDepthSortPerFrame = vi.fn();
const warmUpDepthSortWorker = vi.fn();
const disposeDepthSort = vi.fn();
vi.mock('../../../rendering/depth-sort-coordinator', () => ({
  configureDepthSort: (...a: unknown[]) => configureDepthSort(...a),
  evaluateDepthSortPerFrame: () => evaluateDepthSortPerFrame(),
  warmUpDepthSortWorker: () => warmUpDepthSortWorker(),
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
    loadSceneMock.mockImplementation(async () => new THREE.Group());
    updateSceneForDimensionsMock.mockResolvedValue(undefined);
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

    it('skips depth-sort wiring when disabled', () => {
      new LuxarLayer(makeOptions({ depthSort: false }));
      expect(configureDepthSort).not.toHaveBeenCalled();
      expect(warmUpDepthSortWorker).not.toHaveBeenCalled();
    });
  });

  describe('load', () => {
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

    it('applies the configured renderOrder to the root', async () => {
      const layer = new LuxarLayer(makeOptions({ renderOrder: 42 }));
      const root = await layer.load('http://example.test/scene.zarr');
      expect(root.renderOrder).toBe(42);
    });

    it('does not attach when disposed mid-load', async () => {
      const options = makeOptions();
      let release: (g: THREE.Group) => void = () => {};
      loadSceneMock.mockImplementation(() => new Promise((r) => (release = r as typeof release)));

      const layer = new LuxarLayer(options);
      const pending = layer.load('http://example.test/scene.zarr');
      await layer.dispose();
      release(new THREE.Group());
      await pending;

      expect(options.scene.children).toHaveLength(0);
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

  describe('setExposure', () => {
    /** A stand-in for a Luxar geometry material: `uOpacity` plus the updaters. */
    function splatMesh(authored: number): THREE.Mesh {
      const mat = Object.assign(new THREE.Material(), {
        uniforms: { uOpacity: { value: authored } },
        updateOpacity(v: number) {
          this.uniforms.uOpacity.value = v;
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
    it('detaches the root and tears down the process singletons', async () => {
      const options = makeOptions();
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
    });

    it('tears down the loader before the pools that serve it', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();

      expect(disposeInstance.mock.invocationCallOrder[0]).toBeLessThan(
        disposeWorkerPool.mock.invocationCallOrder[0]
      );
    });

    it('is idempotent', async () => {
      const layer = new LuxarLayer(makeOptions());
      await layer.load('http://example.test/scene.zarr');
      await layer.dispose();
      await layer.dispose();
      expect(disposeWorkerPool).toHaveBeenCalledTimes(1);
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
