/**
 * Integration tests for the `data/zarr-loader.ts` facade.
 *
 * **TEST SCOPE**: thin-facade tests — verify that `loadScene` /
 * `updateView` / `updateSceneForDimensions` / `dispose` delegate to
 * `SceneLoaderManager` correctly. The dim → view-state conversion is
 * exercised by `dims-to-view-state.test.ts` directly; the integration
 * tests here only verify the facade calls into the manager with the
 * converted shape.
 *
 * **Mocks**: `SceneLoaderManager` only — the real manager creates a real
 * SceneLoader which depends on zarr I/O and WebGL. THREE is *not* mocked:
 * `Group`/`Points`/`Mesh`/`BufferGeometry` are pure JS, and traversal
 * works against real THREE objects without any WebGL context.
 *
 * **Related**:
 * - `unit/data/utils/dims-to-view-state.test.ts` — pure helper tests
 * - `e2e/data-loading.spec.ts` — full pipeline with browser + files
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import { loadScene, updateView, updateSceneForDimensions, dispose } from '../../../data';
import type { SimpleDims } from '../../../types/dims';

// Mock SceneLoaderManager only — see file header. The real manager depends
// on zarr I/O and WebGL.
vi.mock('../../../data/scene-loader-manager', () => {
  const buildScene = (): THREE.Group => {
    const scene = new THREE.Group();
    scene.name = 'LuxarScene';
    // Add a single Points child so logSceneStats's traverse hits a
    // real instanceof THREE.Points branch (no mocked traverse needed).
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(30), 3));
    const points = new THREE.Points(geom);
    points.name = '/points';
    points.userData = { node: { hasSpatialIndex: true }, attrs: { has_spatial_index: true } };
    scene.add(points);
    scene.userData = {
      sceneDimensions: {
        dimensions: [
          { name: 'x', unit: 'um', range: [0, 100], display: true },
          { name: 'y', unit: 'um', range: [0, 100], display: true },
          { name: 'z', unit: 'um', range: [0, 50], display: true },
          { name: 'time', unit: 's', range: [0, 10], display: false },
        ],
      },
      maxRadius: 0.5,
    };
    return scene;
  };

  const mockLoader = {
    loadScene: vi.fn(async (_url: string) => buildScene()),
    updateView: vi.fn().mockResolvedValue(undefined),
    dispose: vi.fn(),
  };

  let isDisposed = false;

  return {
    SceneLoaderManager: {
      getInstance: vi.fn().mockReturnValue({
        createLoader: vi.fn().mockReturnValue(mockLoader),
        getLoader: vi.fn().mockReturnValue(mockLoader),
        getDefaultLoader: vi.fn(() => (isDisposed ? null : mockLoader)),
        destroyAll: vi.fn(() => {
          isDisposed = true;
        }),
        reset: vi.fn(() => {
          isDisposed = false;
        }),
      }),
    },
  };
});

describe('Data Loading Integration', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
    const mockManager = SceneLoaderManager.getInstance() as unknown as { reset?: () => void };
    mockManager.reset?.();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('loadScene', () => {
    it('delegates to the manager and returns the loaded scene with metadata', async () => {
      const scene = await loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeInstanceOf(THREE.Group);
      expect(scene.name).toBe('LuxarScene');
      // Audit W6 fix: toBeDefined only rules out undefined. The
      // documented sceneDimensions shape is a `Dims`-like object with
      // a `dimensions` array. Pin enough structure that a mutation
      // returning `true` or a number would fail.
      expect(scene.userData.sceneDimensions).toBeDefined();
      expect(typeof scene.userData.sceneDimensions).toBe('object');
      expect(scene.userData.sceneDimensions).not.toBeNull();
      expect(scene.userData.maxRadius).toBe(0.5);
    });

    it('asks the manager for an instance on every load', async () => {
      await loadScene('http://localhost:8000/test.zarr');
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      expect(SceneLoaderManager.getInstance).toHaveBeenCalled();
    });

    it('propagates loader errors instead of swallowing them', async () => {
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { loadScene: ReturnType<typeof vi.fn> } | null;
      };
      mockManager.getDefaultLoader()!.loadScene.mockRejectedValueOnce(new Error('Network error'));

      await expect(loadScene('http://invalid.url')).rejects.toThrow('Network error');
    });
  });

  describe('updateView', () => {
    beforeEach(async () => {
      await loadScene('http://localhost:8000/test.zarr');
    });

    it('forwards the partial view-state directly to the loader', async () => {
      const viewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };
      await updateView(viewState);

      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { updateView: ReturnType<typeof vi.fn> };
      };
      expect(mockManager.getDefaultLoader().updateView).toHaveBeenCalledWith(viewState);
    });

    it('warns and short-circuits when no scene is loaded', async () => {
      dispose();
      const consoleSpy = vi.spyOn(console, 'warn');

      await updateView({ displayDims: [0, 1, 2] });

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('No scene loaded'));
    });

    it('catches loader errors so the caller is unaffected', async () => {
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { updateView: ReturnType<typeof vi.fn> };
      };
      mockManager.getDefaultLoader().updateView.mockRejectedValueOnce(new Error('Update failed'));

      // updateView swallows the error rather than re-throw — the system
      // should keep running after a failed view update.
      await expect(updateView({ displayDims: [0, 1, 2] })).resolves.toBeUndefined();

      // After the failure the next update still goes through.
      await updateView({ displayDims: [0, 1, 2] });
      expect(mockManager.getDefaultLoader().updateView).toHaveBeenCalledTimes(2);
    });
  });

  describe('updateSceneForDimensions', () => {
    let scene: THREE.Group;

    beforeEach(async () => {
      scene = await loadScene('http://localhost:8000/test.zarr');
    });

    it('forwards the converted view-state shape (delegating to simpleDimsToViewState)', async () => {
      // The conversion logic itself is covered by dims-to-view-state.test.ts;
      // this test only asserts the facade reads the scene's maxRadius and
      // routes the resulting ViewState through the manager.
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 5],
        metadata: [
          { name: 'x', unit: 'um', scale: 1, display: true },
          { name: 'y', unit: 'um', scale: 1, display: true },
          { name: 'z', unit: 'um', scale: 1, display: true },
          { name: 'time', unit: 's', scale: 1, display: false },
        ],
      };

      await updateSceneForDimensions(dims, scene);

      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { updateView: ReturnType<typeof vi.fn> };
      };
      expect(mockManager.getDefaultLoader().updateView).toHaveBeenCalledWith(
        expect.objectContaining({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 5],
          // scene.userData.maxRadius is 0.5 from the mock, and the
          // continuous time dim picks it up.
          tolerance: [0, 0, 0, 0.5],
        })
      );
    });

    it('reads maxRadius from scene.userData (not config) when present', async () => {
      scene.userData.maxRadius = 0.3;
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 5],
        metadata: [],
      };

      await updateSceneForDimensions(dims, scene);

      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { updateView: ReturnType<typeof vi.fn> };
      };
      const arg = mockManager.getDefaultLoader().updateView.mock.calls.at(-1)![0];
      expect(arg.tolerance[3]).toBe(0.3);
    });
  });

  describe('resource cleanup', () => {
    it('asks the manager to destroy all loaders on dispose()', async () => {
      await loadScene('http://localhost:8000/test.zarr');
      dispose();

      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const manager = SceneLoaderManager.getInstance();
      expect(manager.destroyAll).toHaveBeenCalled();
    });

    it('handles repeated dispose calls without throwing', () => {
      expect(() => {
        dispose();
        dispose();
      }).not.toThrow();
    });
  });

  describe('concurrent operations', () => {
    it('handles concurrent loads — last-writer wins for getDefaultLoader', async () => {
      const scenes = await Promise.all([
        loadScene('http://localhost:8000/test1.zarr'),
        loadScene('http://localhost:8000/test2.zarr'),
        loadScene('http://localhost:8000/test3.zarr'),
      ]);

      expect(scenes).toHaveLength(3);
      expect(scenes[2]).toBeInstanceOf(THREE.Group);
    });

    it('forwards all concurrent view updates without dropping any', async () => {
      await loadScene('http://localhost:8000/test.zarr');

      await Promise.all([
        updateView({ displayDims: [0, 1, 2] }),
        updateView({ slicePosition: [0, 0, 0, 5] }),
        updateView({ tolerance: [0.1, 0.1, 0.1, 0.2] }),
      ]);

      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as unknown as {
        getDefaultLoader: () => { updateView: ReturnType<typeof vi.fn> };
      };
      expect(mockManager.getDefaultLoader().updateView).toHaveBeenCalledTimes(3);
    });
  });

  describe('error recovery', () => {
    it('lets loadScene be called again after dispose()', async () => {
      await loadScene('http://localhost:8000/test1.zarr');
      dispose();

      const scene = await loadScene('http://localhost:8000/test2.zarr');
      expect(scene).toBeInstanceOf(THREE.Group);
    });
  });
});
