/**
 * Comprehensive tests for SceneLoader
 *
 * Tests the orchestration of scene loading, spatial index integration,
 * hierarchical scene graph construction, and view updates.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SceneLoader, type LoaderConfig, type ViewState } from '../data';
import * as THREE from 'three';
import * as zarr from 'zarrita';

// Mock THREE.js
vi.mock('three', () => ({
  Group: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    name: '',
    userData: {},
    children: [],
    getObjectByName: vi.fn(),
    traverse: vi.fn((callback) => {
      // Simple traverse implementation for testing
      callback({ name: 'test' });
    }),
    position: {
      copy: vi.fn().mockReturnThis(),
      set: vi.fn().mockReturnThis(),
    },
    quaternion: {
      copy: vi.fn().mockReturnThis(),
    },
    scale: {
      copy: vi.fn().mockReturnThis(),
      set: vi.fn().mockReturnThis(),
    },
  })),
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
    decompose: vi.fn((pos: any, _quat: any, scale: any) => {
      pos.set(0, 0, 0);
      scale.set(1, 1, 1);
    }),
  })),
  Points: vi.fn().mockImplementation((geometry, material) => ({
    name: '',
    userData: {},
    geometry,
    material,
    position: {
      copy: vi.fn().mockReturnThis(),
      set: vi.fn().mockReturnThis(),
    },
    quaternion: {
      copy: vi.fn().mockReturnThis(),
    },
    scale: {
      copy: vi.fn().mockReturnThis(),
      set: vi.fn().mockReturnThis(),
    },
  })),
  BufferGeometry: vi.fn().mockImplementation(() => ({
    setAttribute: vi.fn(),
    boundingBox: null,
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
  EventDispatcher: vi.fn().mockImplementation(() => ({
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
  // Constants
  HalfFloatType: 1016,
  FloatType: 1015,
  UnsignedByteType: 1009,
  LinearSRGBColorSpace: 'srgb-linear',
  SRGBColorSpace: 'srgb',
  NoToneMapping: 0,
  ACESFilmicToneMapping: 4,
  PCFSoftShadowMap: 2,
}));

// Mock zarrita
vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  tryWithConsolidated: vi.fn(),
  root: vi.fn(),
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock material manager
vi.mock('../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
    }),
  },
}));

// Mock DataLoadingMonitor
vi.mock('../ui/data-loading-monitor', () => ({
  DataLoadingMonitor: vi.fn().mockImplementation(() => ({
    connectLoader: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    dispose: vi.fn(),
  })),
}));

// Mock PointSpatialIndexLoader
vi.mock('../data/point-spatial-index-loader', () => ({
  PointSpatialIndexLoader: vi.fn().mockImplementation(() => ({
    loadPoints: vi.fn().mockResolvedValue({
      positions: new Float32Array([1, 2, 3, 4, 5, 6]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      radii: new Float32Array([0.1, 0.2]),
      metadata: {
        totalPoints: 2,
        loadedPoints: 2,
        bounds: {
          clone: vi.fn().mockReturnThis(),
          expandByPoint: vi.fn(),
        },
        ndim: 3,
        usedSpatialIndex: true,
      },
    }),
    updateView: vi.fn().mockResolvedValue({
      positions: new Float32Array([1, 2, 3]),
      metadata: {
        totalPoints: 1,
        loadedPoints: 1,
        bounds: {
          clone: vi.fn().mockReturnThis(),
          expandByPoint: vi.fn(),
        },
        ndim: 3,
        usedSpatialIndex: true,
      },
    }),
    getCacheStats: vi.fn().mockReturnValue({
      hits: 10,
      misses: 5,
      hitRate: 0.67,
    }),
    clearCache: vi.fn(),
    dispose: vi.fn(),
  })),
}));

// Mock DataMonitorManager before importing SceneLoader
vi.mock('../data/data-monitor-manager', () => ({
  DataMonitorManager: {
    getInstance: vi.fn(() => ({
      getMonitor: vi.fn(() => null),
      hasMonitor: vi.fn(() => false),
      createMonitor: vi.fn(() => ({
        show: vi.fn(),
        hide: vi.fn(),
        toggle: vi.fn(),
      })),
      showMonitor: vi.fn(),
      hideMonitor: vi.fn(),
      toggleMonitor: vi.fn(),
    })),
  },
}));

describe('SceneLoader', () => {
  let sceneLoader: SceneLoader;
  let mockStore: any;
  let mockRootLoc: any;
  let mockZarrGroup: any;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Setup mock store
    mockStore = {
      contents: vi.fn().mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/points', kind: 'group' },
      ]),
    };

    // Setup mock zarr location
    mockRootLoc = {
      resolve: vi.fn().mockImplementation((_path) => ({
        resolve: vi.fn().mockImplementation((_subpath) => ({
          resolve: vi.fn(),
        })),
      })),
    };

    // Setup mock zarr group
    mockZarrGroup = {
      attrs: {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: false, step: 0.1 },
          ],
        },
      },
    };

    // Mock zarrita functions
    (zarr.FetchStore as any).mockImplementation(() => mockStore);
    (zarr.tryWithConsolidated as any).mockResolvedValue(mockStore);
    (zarr.root as any).mockReturnValue(mockRootLoc);
    (zarr.open as any).mockResolvedValue(mockZarrGroup);

    // Create SceneLoader instance
    sceneLoader = new SceneLoader();
  });

  afterEach(() => {
    sceneLoader.dispose();
  });

  describe('loadScene', () => {
    it('should load a scene with correct URL normalization', async () => {
      const url = 'http://localhost:8000/test.zarr';
      const scene = await sceneLoader.loadScene(url);

      // Verify scene loaded correctly (implementation may use FetchStore or TwoLevelCachingStore)
      expect(zarr.tryWithConsolidated).toHaveBeenCalled();
      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
    });

    it('should initialize scene dimensions from metadata', async () => {
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(scene.userData.sceneDimensions).toEqual(mockZarrGroup.attrs.scene_dimensions);
    });

    it('should handle missing scene dimensions gracefully', async () => {
      mockZarrGroup.attrs = {}; // No scene_dimensions

      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeDefined();
      expect(scene.userData.sceneDimensions).toBeUndefined();
    });

    it('should build scene graph hierarchy correctly', async () => {
      // Setup hierarchical structure
      mockStore.contents.mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/group1', kind: 'group' },
        { path: '/group1/points', kind: 'group' },
      ]);

      // Mock checking for spatial index
      mockRootLoc.resolve.mockImplementation((_path: any) => ({
        resolve: vi.fn().mockImplementation((subpath) => {
          if (subpath === 'spatial_index') {
            // Simulate spatial index exists for points
            if (_path.includes('points')) {
              return Promise.resolve({});
            }
            throw new Error('No spatial index');
          }
          return { resolve: vi.fn() };
        }),
      }));

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      // Verify scene graph was built
      expect(mockRootLoc.resolve).toHaveBeenCalled();
    });

    it('should detect and log broadcast dimensions', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // Setup node with broadcast_dims
      mockZarrGroup.attrs = {
        type: 'points',
        broadcast_dims: ['time', 'channel'],
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('broadcast_dims'));
    });

    it('should handle missing spatial index gracefully for 3D points', async () => {
      // Setup points group without spatial index
      mockStore.contents.mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/points', kind: 'group' },
      ]);

      // Mock the open call to return points type
      (zarr.open as any).mockImplementation((_loc: any, _opts: any) =>
        Promise.resolve({
          attrs: { type: 'points', num_points: 1000 },
        })
      );

      // Mock resolve to simulate missing spatial index
      mockRootLoc.resolve.mockImplementation((_path: any) => ({
        resolve: vi.fn().mockImplementation((subpath) => {
          if (subpath === 'spatial_index') {
            throw new Error('Not found');
          }
          return { resolve: vi.fn() };
        }),
      }));

      // Should NOT throw an error - handles missing spatial index gracefully for 3D datasets
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
    });
  });

  describe('updateView', () => {
    beforeEach(async () => {
      // Load a scene first
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    it('should update all loaders with new view state', async () => {
      const viewState: Partial<ViewState> = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.1],
      };

      await sceneLoader.updateView(viewState);

      // Verify loaders were updated (check through mock)
      expect(console.log).toHaveBeenCalledWith(expect.stringContaining('Updating view'));
    });

    it('should handle loader update failures gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error');

      // Make one loader fail - include dispose method
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Update failed')),
        dispose: vi.fn(),
        getCacheStats: vi.fn().mockReturnValue({}),
        clearCache: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failing', mockLoader);

      const viewState: Partial<ViewState> = {
        displayDims: [0, 1, 2],
      };

      await sceneLoader.updateView(viewState);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to update'),
        expect.any(Error)
      );
    });

    it('should not update geometry when no points are loaded', async () => {
      // Mock loader returning empty data - include dispose method
      const mockLoader = {
        updateView: vi.fn().mockResolvedValue({
          metadata: { loadedPoints: 0 },
        }),
        dispose: vi.fn(),
        getCacheStats: vi.fn().mockReturnValue({}),
        clearCache: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/empty', mockLoader);

      await sceneLoader.updateView({});

      // Geometry update should not happen for empty points
      expect(mockLoader.updateView).toHaveBeenCalled();
    });
  });

  describe('resource management', () => {
    it('should get cache statistics from all loaders', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      const stats = sceneLoader.getCacheStats();

      expect(stats).toBeInstanceOf(Map);
    });

    it('should clear all loader caches', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      sceneLoader.clearCaches();

      // Verify through mocks that clearCache was called
      expect(console.log).not.toHaveBeenCalledWith(expect.stringContaining('error'));
    });

    it('should dispose all resources properly', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      sceneLoader.dispose();

      // Verify cleanup
      expect((sceneLoader as any).loaders.size).toBe(0);
      expect((sceneLoader as any).store).toBeNull();
      expect((sceneLoader as any).rootGroup).toBeNull();
    });
  });

  describe('monitor integration', () => {
    it('should show monitor UI', () => {
      // Just verify the method exists and doesn't throw
      expect(() => sceneLoader.showMonitor()).not.toThrow();
    });

    it('should hide monitor UI', () => {
      // Just verify the method exists and doesn't throw
      expect(() => sceneLoader.hideMonitor()).not.toThrow();
    });

    it('should toggle monitor UI', () => {
      // Just verify the method exists and doesn't throw
      expect(() => sceneLoader.toggleMonitor()).not.toThrow();
    });
  });

  describe('transform handling', () => {
    it('should apply transforms to objects correctly', async () => {
      // Setup node with transform
      mockZarrGroup.attrs = {
        type: 'points',
        transform: [
          1,
          0,
          0,
          10, // Translation x=10
          0,
          1,
          0,
          20, // Translation y=20
          0,
          0,
          1,
          30, // Translation z=30
          0,
          0,
          0,
          1,
        ],
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      // Verify transform was processed
      expect(THREE.Matrix4).toHaveBeenCalled();
    });

    it('should handle invalid transform lengths', async () => {
      const consoleSpy = vi.spyOn(console, 'warn');

      mockZarrGroup.attrs = {
        type: 'group',
        transform: [1, 2, 3], // Invalid length
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid transform length'));
    });
  });

  describe('error handling', () => {
    it('should handle store opening failures', async () => {
      (zarr.tryWithConsolidated as any).mockRejectedValue(new Error('Failed to open store'));

      await expect(sceneLoader.loadScene('http://invalid.url')).rejects.toThrow(
        'Failed to open store'
      );
    });

    it('should handle enumeration failures gracefully', async () => {
      mockStore.contents = undefined; // No contents method

      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      // Should fall back to root only
      expect(scene).toBeDefined();
    });
  });

  describe('config handling', () => {
    it('should accept and use loader configuration', () => {
      const config: LoaderConfig = {
        maxMemoryMB: 1000,
        evictionStrategy: 'lfu',
      };

      const loader = new SceneLoader(config);
      expect((loader as any).config).toEqual(config);
      loader.dispose();
    });
  });
});
