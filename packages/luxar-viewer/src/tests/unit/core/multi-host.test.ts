/**
 * Several Luxar hosts on one page: a LuxarApp plus LuxarLayers, or layers alone.
 *
 * Runs the REAL `SceneLoaderManager`, `zarr-loader`, `SceneDimsManager` and
 * `LuxarLayer`; only the `SceneLoader` itself (network + GPU) is a fake, so what
 * is under test is the identity plumbing: which manager a host's loader lives
 * in, which dimension state it resolves, and what its teardown reaches.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';

vi.stubGlobal('window', globalThis.window ?? {});
vi.stubGlobal('document', globalThis.document ?? {});

interface FakeLoader {
  id: string;
  src: string | null;
  dispose: ReturnType<typeof vi.fn>;
}
const loaders: FakeLoader[] = [];

function sceneWithDims(names: string[]): THREE.Group {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  root.userData.sceneDimensions = {
    dimensions: names.map((name, i) => ({
      name,
      unit: '',
      range: [0, 10],
      step: 1,
      display: i < 3,
    })),
  };
  return root;
}

vi.mock('../../../data/scene-loader', () => ({
  SceneLoader: class {
    readonly id: string;
    src: string | null = null;
    readonly dispose = vi.fn(async () => {});
    readonly lodGroupRegistry = null;
    readonly nodeFactory = { rebuildAfterContextRestore: vi.fn() };
    readonly archiveFault = null;
    constructor(_config: unknown, id: string) {
      this.id = id;
      loaders.push(this as unknown as FakeLoader);
    }
    setRequestRender(): void {}
    setAutoRetryableFailureCallback(): void {}
    setRefinementDensityProvider(): void {}
    setDepthSortCoordinator(): void {}
    setMaterialManager(): void {}
    async loadScene(src: string): Promise<THREE.Group> {
      this.src = src;
      return sceneWithDims(src.includes('layer-b') ? ['u', 'v', 'w'] : ['x', 'y', 'z', 't']);
    }
    hasFailures(): boolean {
      return false;
    }
    isAtViewState(): boolean {
      return true;
    }
    async updateView(): Promise<void> {}
    onArchiveFault(): () => void {
      return () => {};
    }
    isUpdateInProgress(): boolean {
      return false;
    }
    isLoadPassInProgress(): boolean {
      return false;
    }
  },
}));
vi.mock('../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: () => ({
    backend: 'webgl',
    apiSurface: 'webgl2',
    maxTextureSize: 4096,
  }),
  isWebGLRenderer: () => false,
}));
vi.mock('../../../rendering/element-texture-row-upload', () => ({
  installElementTextureRowUploads: () => {},
}));
vi.mock('../../../rendering/ktx2-texture-decoder', () => ({
  createKTX2TextureDecoder: () => ({ dispose: () => {} }),
}));
vi.mock('../../../rendering/environment/scene-environment', () => ({
  createSceneEnvironment: () => ({
    ensure: () => {},
    isReady: () => false,
    rebuild: () => {},
    dispose: () => {},
  }),
}));

import { LuxarLayer, type LuxarLayerOptions } from '../../../core/layer/luxar-layer';
import { loadScene } from '../../../data/zarr-loader';
import { SceneLoaderManager } from '../../../data/scene-loader-manager';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { log } from '../../../utils/log';

function layerOptions(scene = new THREE.Scene()): LuxarLayerOptions {
  return {
    renderer: {
      getDrawingBufferSize: (v: THREE.Vector2) => v.set(800, 600),
      getPixelRatio: () => 1,
    } as unknown as LuxarLayerOptions['renderer'],
    getCamera: () => new THREE.PerspectiveCamera(50, 1, 0.1, 100),
    getViewportSize: () => ({ width: 800, height: 600 }),
    scene,
    depthSort: false,
  };
}

/** What a LuxarApp does on load: the app's loader, then the app's dimensions. */
async function loadAppScene(): Promise<THREE.Group> {
  const root = await loadScene(SceneLoaderManager.getInstance(), 'http://app.test/app.zarr');
  sceneDimsManager.initFromScene(root as unknown as THREE.Scene);
  return root;
}

describe('several Luxar hosts on one page', () => {
  beforeEach(() => {
    loaders.length = 0;
    SceneLoaderManager.disposeInstance();
    sceneDimsManager.reset();
  });

  it("a layer's load leaves the app's loader and the app's dimensions alone", async () => {
    await loadAppScene();
    const appLoader = loaders[0];

    const layer = new LuxarLayer(layerOptions());
    await layer.load('http://layer-b.test/layer.zarr');

    expect(appLoader.dispose).not.toHaveBeenCalled();
    expect(SceneLoaderManager.getInstance().getDefaultLoader()).toBe(appLoader);
    expect(sceneDimsManager.getDimensionNames()).toEqual(['x', 'y', 'z', 't']);
    // …and the layer resolved ITS OWN dimension set.
    expect(layer.getDimensionNames()).toEqual(['u', 'v', 'w']);
    await layer.dispose();
  });

  it("two layers each keep their own loader, and one layer's dispose spares the other", async () => {
    const a = new LuxarLayer(layerOptions());
    const b = new LuxarLayer(layerOptions());
    await a.load('http://layer-a.test/a.zarr');
    const aLoader = loaders[0];
    await b.load('http://layer-b.test/b.zarr');
    const bLoader = loaders[1];

    expect(aLoader.dispose).not.toHaveBeenCalled();
    expect(a.getDimensionNames()).toEqual(['x', 'y', 'z', 't']);
    expect(b.getDimensionNames()).toEqual(['u', 'v', 'w']);

    await a.dispose();
    expect(aLoader.dispose).toHaveBeenCalled();
    expect(bLoader.dispose).not.toHaveBeenCalled();
    expect(b.getDimensionNames()).toEqual(['u', 'v', 'w']);
    await b.dispose();
  });

  it("a layer's dispose leaves the app's loader and dimensions alone", async () => {
    await loadAppScene();
    const appLoader = loaders[0];
    const layer = new LuxarLayer(layerOptions());
    await layer.load('http://layer-b.test/layer.zarr');

    await layer.dispose();

    expect(appLoader.dispose).not.toHaveBeenCalled();
    expect(SceneLoaderManager.getInstance().getDefaultLoader()).toBe(appLoader);
    expect(sceneDimsManager.getDimensionNames()).toEqual(['x', 'y', 'z', 't']);
  });

  it('warns when a second layer shares a THREE.Scene, naming the renderOrder contract', () => {
    // Each layer depth-sorts and render-orders only its own nodes, so two layers
    // in one scene do not interleave by depth: the contract is whole-layer
    // order by the `renderOrder` option, and an equal order is undefined.
    const warning = vi.spyOn(log, 'warning');
    const scene = new THREE.Scene();
    const a = new LuxarLayer(layerOptions(scene));
    expect(warning).not.toHaveBeenCalledWith(expect.anything(), expect.stringContaining('share'));

    const b = new LuxarLayer(layerOptions(scene));
    const shared = warning.mock.calls.find(([, message]) =>
      String(message).includes('share one THREE.Scene')
    );
    expect(shared).toBeDefined();
    expect(String(shared![1])).toContain('renderOrder');

    // A layer in its OWN scene is the supported, silent case.
    warning.mockClear();
    const c = new LuxarLayer(layerOptions(new THREE.Scene()));
    expect(warning.mock.calls.some(([, m]) => String(m).includes('share one THREE.Scene'))).toBe(
      false
    );
    void a.dispose();
    void b.dispose();
    void c.dispose();
    warning.mockRestore();
  });
});
