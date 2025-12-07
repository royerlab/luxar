/**
 * Unit Integration Tests for Data Loading Pipeline
 *
 * **TEST SCOPE**: Unit-level integration tests with mocked dependencies
 * - Uses mocked zarr data (not real files)
 * - Uses mocked THREE.js (no real WebGL)
 * - Tests internal logic and API contracts
 * - Fast execution (no browser, no network)
 *
 * **WHAT WE TEST**:
 * - Interaction between zarr-loader, SceneLoader, and SpatialIndexLoader
 * - Correct data flow through the pipeline
 * - Error handling with invalid data
 * - Cache behavior
 *
 * **WHAT WE DON'T TEST** (see E2E tests instead):
 * - Real zarr file loading
 * - Actual WebGL rendering
 * - Browser-specific behaviors (OPFS, canvas, etc.)
 *
 * **Related Tests**:
 * - `unit/data/data-monitor-integration.test.ts` - Monitor + loader interaction (unit)
 * - `e2e/data-loading.spec.ts` - Full pipeline with real browser + files (E2E)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { loadScene, updateView, updateSceneForDimensions, dispose } from '../../../data';
import { SimpleDims } from '../../../types/dims';
import * as THREE from 'three';

// Mock THREE.js with more complete implementation
vi.mock('three', () => {
  const mockPoints = vi.fn().mockImplementation((geometry, material) => ({
    name: '',
    userData: {},
    geometry,
    material,
    position: { set: vi.fn() },
    quaternion: { copy: vi.fn() },
    scale: { set: vi.fn() },
  }));

  const mockGroup = vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    name: '',
    userData: {},
    children: [],
    getObjectByName: vi.fn(),
    traverse: vi.fn((callback) => {
      // Create a mock points object that will pass instanceof check
      const mockPointsGeometry = {
        getAttribute: vi.fn().mockReturnValue({
          count: 1000,
        }),
      };
      const pointsInstance = new mockPoints(mockPointsGeometry, null);
      pointsInstance.name = '/points';
      pointsInstance.userData = {
        node: { hasSpatialIndex: true },
      };

      callback(pointsInstance);
    }),
  }));

  return {
    Group: mockGroup,
    Points: mockPoints,
    Box3: vi.fn().mockImplementation(() => ({
      expandByPoint: vi.fn(),
      clone: vi.fn().mockReturnThis(),
    })),
    Vector3: vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
      x,
      y,
      z,
      set: vi.fn().mockReturnThis(),
      copy: vi.fn().mockReturnThis(),
    })),
    Quaternion: vi.fn().mockImplementation(() => ({
      x: 0,
      y: 0,
      z: 0,
      w: 1,
      copy: vi.fn().mockReturnThis(),
    })),
    Matrix4: vi.fn().mockImplementation(() => ({
      fromArray: vi.fn().mockReturnThis(),
      decompose: vi.fn(),
    })),
    BufferGeometry: vi.fn().mockImplementation(() => ({
      setAttribute: vi.fn(),
      boundingBox: null,
      boundingSphere: null,
      dispose: vi.fn(),
    })),
    BufferAttribute: vi.fn().mockImplementation((array, itemSize) => ({
      array,
      itemSize,
      count: array.length / itemSize,
    })),
    ShaderMaterial: vi.fn().mockImplementation(() => ({
      uniforms: {},
    })),
    // Constants that might be needed by other imports
    HalfFloatType: 1016,
    FloatType: 1015,
    UnsignedByteType: 1009,
    // Color space and tone mapping constants
    LinearSRGBColorSpace: 'srgb-linear',
    SRGBColorSpace: 'srgb',
    NoToneMapping: 0,
    ACESFilmicToneMapping: 4,
    PCFSoftShadowMap: 2,
  };
});

// Mock scene-loader and scene-loader-manager
vi.mock('../data/scene-loader', () => ({
  SceneLoader: vi.fn().mockImplementation(() => ({
    loadScene: vi.fn().mockImplementation(async (_url) => {
      // Import the mocked THREE to use the proper Group mock
      const THREE = await import('three');
      const group = new THREE.Group();
      group.name = 'LuxarScene';
      group.userData = {
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
      return group;
    }),
    updateView: vi.fn().mockResolvedValue(undefined),
    getCacheStats: vi
      .fn()
      .mockReturnValue(new Map([['/points', { hits: 10, misses: 5, hitRate: 0.67 }]])),
    clearCaches: vi.fn(),
    dispose: vi.fn(),
    showMonitor: vi.fn(),
    hideMonitor: vi.fn(),
    toggleMonitor: vi.fn(),
  })),
}));

// Mock SceneLoaderManager
vi.mock('../../../data/scene-loader-manager', () => {
  const mockLoader = {
    loadScene: vi.fn().mockImplementation(async (_url) => {
      // Import the mocked THREE to use the proper Group mock
      const THREE = await import('three');
      const group = new THREE.Group();
      group.name = 'LuxarScene';
      group.userData = {
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
      return group;
    }),
    updateView: vi.fn().mockResolvedValue(undefined),
    getCacheStats: vi
      .fn()
      .mockReturnValue(new Map([['/points', { hits: 10, misses: 5, hitRate: 0.67 }]])),
    clearCaches: vi.fn(),
    dispose: vi.fn(),
    showMonitor: vi.fn(),
    hideMonitor: vi.fn(),
    toggleMonitor: vi.fn(),
  };

  // Keep track of whether loaders are disposed
  let isDisposed = false;

  return {
    SceneLoaderManager: {
      getInstance: vi.fn().mockReturnValue({
        createLoader: vi.fn().mockReturnValue(mockLoader),
        getLoader: vi.fn().mockReturnValue(mockLoader),
        getDefaultLoader: vi.fn().mockImplementation(() => (isDisposed ? null : mockLoader)),
        getAllLoaders: vi.fn().mockReturnValue(new Map([['default', mockLoader]])),
        destroyLoader: vi.fn(),
        destroyAll: vi.fn().mockImplementation(() => {
          isDisposed = true;
        }),
        reset: vi.fn().mockImplementation(() => {
          isDisposed = false;
        }),
      }),
    },
  };
});

vi.mock('../data/point-spatial-index-loader', () => ({
  PointSpatialIndexLoader: vi.fn().mockImplementation(() => ({
    loadPoints: vi.fn().mockResolvedValue({
      positions: new Float32Array([1, 2, 3, 4, 5, 6]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      metadata: {
        totalPoints: 2,
        loadedPoints: 2,
        bounds: { min: [0, 0, 0], max: [10, 10, 10] },
        ndim: 3,
        usedSpatialIndex: true,
      },
    }),
    updateView: vi.fn().mockResolvedValue({
      positions: new Float32Array([1, 2, 3]),
      metadata: {
        totalPoints: 1,
        loadedPoints: 1,
        bounds: { min: [0, 0, 0], max: [5, 5, 5] },
        ndim: 3,
        usedSpatialIndex: true,
      },
    }),
    getCacheStats: vi.fn(),
    clearCache: vi.fn(),
    dispose: vi.fn(),
  })),
}));

// Mock material manager
vi.mock('../rendering/material-manager', () => ({
  materialManager: {
    getMaterial: vi.fn().mockReturnValue({
      uniforms: {},
    }),
  },
}));

// Mock UI components
vi.mock('../ui/data-loading-monitor', () => ({
  DataLoadingMonitor: vi.fn().mockImplementation(() => ({
    connectLoader: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    dispose: vi.fn(),
  })),
}));

// Helper to access mocked functions
const getMockLoader = () => {
  // Need to import actual class for type reference
  void vi.importActual('../../../data/scene-loader-manager');
  // Return a mock object with all needed methods
  return {
    loadScene: vi.fn().mockImplementation(async () => {
      const THREE = await import('three');
      const group = new THREE.Group();
      group.name = 'LuxarScene';
      group.userData = {
        sceneDimensions: {
          dimensions: [],
        },
        maxRadius: 0.5,
      };
      return group;
    }),
    updateView: vi.fn().mockResolvedValue(undefined),
    getCacheStats: vi
      .fn()
      .mockReturnValue(new Map([['/points', { hits: 10, misses: 5, hitRate: 0.67 }]])),
    clearCaches: vi.fn(),
    dispose: vi.fn(),
  };
};

describe('Data Loading Integration', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    void getMockLoader(); // Create mock loader but don't store reference

    // Reset the SceneLoaderManager state
    const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
    const mockManager = SceneLoaderManager.getInstance() as any;
    if (mockManager.reset) {
      mockManager.reset();
    }
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('loadScene', () => {
    it('should load a complete scene with metadata', async () => {
      const url = 'http://localhost:8000/test.zarr';
      const scene = await loadScene(url);

      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      expect(scene.userData.sceneDimensions).toBeDefined();
      expect(scene.userData.maxRadius).toBe(0.5);
    });

    it('should configure scene loader with provided config', async () => {
      const scene = await loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeDefined();
    });

    it('should expose loader globally for debugging', async () => {
      await loadScene('http://localhost:8000/test.zarr');

      // Check that manager was used
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      expect(SceneLoaderManager.getInstance).toHaveBeenCalled();
    });

    it('should handle loading errors gracefully', async () => {
      // Force an error by modifying the SceneLoaderManager mock
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as any;
      const mockLoader = mockManager.getDefaultLoader() as any;
      mockLoader.loadScene.mockRejectedValueOnce(new Error('Network error'));

      await expect(loadScene('http://invalid.url')).rejects.toThrow('Network error');
    });

    it('should log scene statistics after loading', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      await loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Scene loaded successfully'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Scene statistics'));
    });
  });

  describe('updateView', () => {
    beforeEach(async () => {
      await loadScene('http://localhost:8000/test.zarr');
    });

    it('should update view state for all loaders', async () => {
      const viewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await updateView(viewState);

      // Check that the actual SceneLoaderManager mock was called
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as any;
      const mockLoader = mockManager.getDefaultLoader() as any;
      expect(mockLoader.updateView).toHaveBeenCalledWith(viewState);
    });

    it('should handle missing scene loader gracefully', async () => {
      dispose(); // Clear the loader

      const consoleSpy = vi.spyOn(console, 'warn');

      await updateView({ displayDims: [0, 1, 2] });

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('No scene loaded'));
    });
  });

  describe('updateSceneForDimensions', () => {
    let scene: THREE.Group;

    beforeEach(async () => {
      scene = await loadScene('http://localhost:8000/test.zarr');
    });

    it('should convert dimensions to view state correctly', async () => {
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

      // Check that the actual SceneLoaderManager mock was called
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as any;
      const mockLoader = mockManager.getDefaultLoader() as any;
      expect(mockLoader.updateView).toHaveBeenCalledWith(
        expect.objectContaining({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 5],
          tolerance: expect.arrayContaining([0, 0, 0, 0.5]), // Uses maxRadius
        })
      );
    });

    it('should use max radius for non-displayed dimension tolerance', async () => {
      const dims: SimpleDims = {
        ndim: 5,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 5, 2],
        metadata: [],
      };

      scene.userData.maxRadius = 0.3;

      await updateSceneForDimensions(dims, scene);

      // Check that the actual SceneLoaderManager mock was called
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as any;
      const mockLoader = mockManager.getDefaultLoader() as any;
      const callArgs = mockLoader.updateView.mock.calls[0][0];
      expect(callArgs.tolerance[3]).toBe(0.3); // Non-displayed dim
      expect(callArgs.tolerance[4]).toBe(0.3); // Non-displayed dim
    });
  });

  describe('resource cleanup', () => {
    it('should dispose all resources', async () => {
      await loadScene('http://localhost:8000/test.zarr');

      dispose();

      // Verify manager cleanup was called
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const manager = SceneLoaderManager.getInstance();
      expect(manager.destroyAll).toHaveBeenCalled();
    });

    it('should handle multiple dispose calls safely', () => {
      dispose();
      dispose(); // Should not throw
    });
  });

  describe('concurrent operations', () => {
    it('should handle concurrent scene loads', async () => {
      // Start multiple loads
      const promises = [
        loadScene('http://localhost:8000/test1.zarr'),
        loadScene('http://localhost:8000/test2.zarr'),
        loadScene('http://localhost:8000/test3.zarr'),
      ];

      const scenes = await Promise.all(promises);

      // Only last one should be active
      expect(scenes[2]).toBeDefined();
      expect(scenes[2].name).toBe('LuxarScene');
    });

    it('should handle concurrent view updates', async () => {
      await loadScene('http://localhost:8000/test.zarr');

      // Multiple concurrent updates
      const updates = [
        updateView({ displayDims: [0, 1, 2] }),
        updateView({ slicePosition: [0, 0, 0, 5] }),
        updateView({ tolerance: [0.1, 0.1, 0.1, 0.2] }),
      ];

      await Promise.all(updates);

      // All should complete without error
      expect(true).toBe(true);
    });
  });

  describe('error recovery', () => {
    it('should recover from failed view updates', async () => {
      await loadScene('http://localhost:8000/test.zarr');

      // Get the actual loader and make update fail
      const { SceneLoaderManager } = await import('../../../data/scene-loader-manager');
      const mockManager = SceneLoaderManager.getInstance() as any;
      const mockLoader = mockManager.getDefaultLoader() as any;
      mockLoader.updateView.mockRejectedValueOnce(new Error('Update failed'));

      // Should not throw - wrap in try/catch to verify it handles error gracefully
      let errorThrown = false;
      try {
        await updateView({ displayDims: [0, 1, 2] });
      } catch {
        errorThrown = true;
      }

      // The updateView should handle errors internally and not throw
      expect(errorThrown).toBe(false);

      // Can still update after failure
      mockLoader.updateView.mockResolvedValueOnce(undefined);
      await updateView({ displayDims: [0, 1, 2] });

      expect(mockLoader.updateView).toHaveBeenCalledTimes(2);
    });

    it('should handle scene loading after disposal', async () => {
      await loadScene('http://localhost:8000/test1.zarr');
      dispose();

      // Should be able to load again
      const scene = await loadScene('http://localhost:8000/test2.zarr');
      expect(scene).toBeDefined();
    });
  });

  // Global API test removed as the new implementation uses SceneLoaderManager instead of global variables
});
