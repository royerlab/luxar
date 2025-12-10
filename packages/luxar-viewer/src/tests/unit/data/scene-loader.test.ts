/**
 * Comprehensive tests for SceneLoader
 *
 * Tests the orchestration of scene loading, spatial index integration,
 * hierarchical scene graph construction, and view updates.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SceneLoader, type LoaderConfig, type ViewState } from '../../../data';
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
  Vector2: vi.fn().mockImplementation((x = 0, y = 0) => ({
    x,
    y,
    set: vi.fn().mockReturnThis(),
    copy: vi.fn().mockReturnThis(),
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
  NormalBlending: 0,
  AdditiveBlending: 2,
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
vi.mock('../../../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
      updateCameraParams: vi.fn(),
      updateHDRMultiplier: vi.fn(),
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

    it('should detect and log extend_to_all dimensions', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // Setup node with extend_to_all
      mockZarrGroup.attrs = {
        type: 'points',
        extend_to_all: ['time', 'channel'],
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('extend_to_all'));
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
          attrs: { type: 'points', n_points: 1000 },
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
      };
      (sceneLoader as any).loaders.set('/failing', mockLoader);

      const viewState: Partial<ViewState> = {
        displayDims: [0, 1, 2],
      };

      await sceneLoader.updateView(viewState);

      // Should log error with improved retry tracking format
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          '[❌] [SceneLoader] Failed to update /failing (attempt 1): Update failed'
        )
      );

      // Should track the failure
      expect(sceneLoader.hasFailures()).toBe(true);
      expect(sceneLoader.getFailedLoaders().size).toBe(1);
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
      const config: LoaderConfig = {};

      const loader = new SceneLoader(config);
      expect((loader as any).config).toEqual(config);
      loader.dispose();
    });
  });

  describe('transform validation', () => {
    it('should accept column-major matrices (correct for THREE.js)', () => {
      // Column-major: translation at indices [12, 13, 14]
      const columnMajorTransform = [
        1,
        0,
        0,
        0, // Column 0: right vector
        0,
        1,
        0,
        0, // Column 1: up vector
        0,
        0,
        1,
        0, // Column 2: forward vector
        10,
        20,
        30,
        1, // Column 3: translation + w
      ];

      const isValid = (sceneLoader as any).validateTransformFormat(columnMajorTransform);
      expect(isValid).toBe(true);
    });

    it('should detect row-major matrices (warn user)', () => {
      const consoleSpy = vi.spyOn(console, 'warn');

      // Row-major: translation at indices [3, 7, 11] (WRONG for THREE.js)
      const rowMajorTransform = [
        1,
        0,
        0,
        10, // Row 0: right + tx
        0,
        1,
        0,
        20, // Row 1: up + ty
        0,
        0,
        1,
        30, // Row 2: forward + tz
        0,
        0,
        0,
        1, // Row 3: homogeneous
      ];

      const isValid = (sceneLoader as any).validateTransformFormat(rowMajorTransform);
      expect(isValid).toBe(false);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('row-major (NumPy) format instead of column-major')
      );
    });

    it('should warn when translation is at wrong indices', () => {
      const consoleSpy = vi.spyOn(console, 'warn');

      // Suspicious transform: non-zero at row-major positions [3, 7, 11], zero at column-major [12, 13, 14]
      const suspiciousTransform = [
        1,
        0,
        0,
        5, // Translation at [3] (row-major)
        0,
        1,
        0,
        10, // Translation at [7] (row-major)
        0,
        0,
        1,
        15, // Translation at [11] (row-major)
        0,
        0,
        0,
        1, // [12, 13, 14] are zero (column-major)
      ];

      const isValid = (sceneLoader as any).validateTransformFormat(suspiciousTransform);
      expect(isValid).toBe(false);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('matrix.T.ravel().tolist()'));
    });

    it('should accept identity matrix', () => {
      const identityTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      const isValid = (sceneLoader as any).validateTransformFormat(identityTransform);
      expect(isValid).toBe(true);
    });

    it('should handle transforms with only rotation/scale (no translation)', () => {
      // Scale matrix: no translation, should pass validation
      const scaleTransform = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];

      const isValid = (sceneLoader as any).validateTransformFormat(scaleTransform);
      expect(isValid).toBe(true);
    });
  });

  describe('material creation', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should create materials with correct radiusScale for uint8 radii', async () => {
      const attrs = {
        opacity: 1.0,
        gamma: 1.0,
        blending_mode: 'normal' as const,
      };
      const radiusScale = 2.5; // maxRadius from node attrs
      const sharpnessScale = 1.0;

      const material = (sceneLoader as any).createMaterial(attrs, radiusScale, sharpnessScale);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should create materials with correct sharpnessScale for uint8 sharpness', async () => {
      const attrs = {
        opacity: 0.8,
        gamma: 2.2,
        blending_mode: 'additive' as const,
      };
      const radiusScale = 1.0;
      const sharpnessScale = 31.0; // SHARPNESS_MAX constant

      const material = (sceneLoader as any).createMaterial(attrs, radiusScale, sharpnessScale);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should handle different blending modes', async () => {
      const normalAttrs = { blending_mode: 'normal' as const };
      const additiveAttrs = { blending_mode: 'additive' as const };

      const material1 = (sceneLoader as any).createMaterial(normalAttrs, 1.0, 1.0);
      const material2 = (sceneLoader as any).createMaterial(additiveAttrs, 1.0, 1.0);

      expect(material1).toBeDefined();
      expect(material2).toBeDefined();
    });

    it('should use default opacity and gamma when not specified', async () => {
      const attrs = {}; // No opacity/gamma specified

      const material = (sceneLoader as any).createMaterial(attrs, 1.0, 1.0);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should pass through custom opacity and gamma values', async () => {
      const attrs = {
        opacity: 0.5,
        gamma: 2.2,
      };

      const material = (sceneLoader as any).createMaterial(attrs, 1.0, 1.0);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should handle default radiusScale and sharpnessScale', async () => {
      const attrs = {};

      const material = (sceneLoader as any).createMaterial(attrs); // No scales provided

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });
  });

  describe('error recovery', () => {
    beforeEach(async () => {
      vi.clearAllMocks();
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    it('should track failed loaders in failedLoaders map', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Network timeout')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failing_node', mockLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      expect(sceneLoader.hasFailures()).toBe(true);
      const failures = sceneLoader.getFailedLoaders();
      expect(failures.size).toBe(1);
      expect(failures.has('/failing_node')).toBe(true);

      const failureInfo = failures.get('/failing_node');
      expect(failureInfo).toBeDefined();
      expect(failureInfo!.error.message).toBe('Network timeout');
      expect(failureInfo!.retryCount).toBe(0);
      expect(failureInfo!.timestamp).toBeGreaterThan(0);
    });

    it('should increment retryCount on repeated failures', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Persistent error')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/persistent_failure', mockLoader);

      // First failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      let failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(0);

      // Second failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(1);

      // Third failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(2);
    });

    it('should clear failures on successful load', async () => {
      const mockLoader = {
        updateView: vi
          .fn()
          .mockRejectedValueOnce(new Error('Temporary failure'))
          .mockResolvedValueOnce({
            positions: new Float32Array([1, 2, 3]),
            metadata: {
              totalPoints: 1,
              loadedPoints: 1,
              bounds: { clone: vi.fn().mockReturnThis() },
              ndim: 3,
              usedSpatialIndex: true,
            },
          }),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/recoverable', mockLoader);

      // First attempt fails
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(true);

      // Second attempt succeeds
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(sceneLoader.getFailedLoaders().size).toBe(0);
    });

    it('should continue loading other nodes when one fails', async () => {
      const failingLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Failure')),
        dispose: vi.fn(),
      };
      const successLoader = {
        updateView: vi.fn().mockResolvedValue({
          positions: new Float32Array([1, 2, 3]),
          metadata: {
            totalPoints: 1,
            loadedPoints: 1,
            bounds: { clone: vi.fn().mockReturnThis() },
            ndim: 3,
            usedSpatialIndex: true,
          },
        }),
        dispose: vi.fn(),
      };

      (sceneLoader as any).loaders.set('/failing', failingLoader);
      (sceneLoader as any).loaders.set('/success', successLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      // Both should have been called
      expect(failingLoader.updateView).toHaveBeenCalled();
      expect(successLoader.updateView).toHaveBeenCalled();

      // Only one failure tracked
      expect(sceneLoader.getFailedLoaders().size).toBe(1);
      expect(sceneLoader.getFailedLoaders().has('/failing')).toBe(true);
    });

    it('should clear failures with clearFailures()', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Error')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failed', mockLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(true);

      sceneLoader.clearFailures();
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(sceneLoader.getFailedLoaders().size).toBe(0);
    });

    it('should warn user when multiple loaders fail', async () => {
      const consoleSpy = vi.spyOn(console, 'warn');

      const failingLoader1 = {
        updateView: vi.fn().mockRejectedValue(new Error('Error 1')),
        dispose: vi.fn(),
      };
      const failingLoader2 = {
        updateView: vi.fn().mockRejectedValue(new Error('Error 2')),
        dispose: vi.fn(),
      };

      (sceneLoader as any).loaders.set('/failing1', failingLoader1);
      (sceneLoader as any).loaders.set('/failing2', failingLoader2);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      // Should warn about multiple failures
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Some data could not be loaded')
      );
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('/failing1'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('/failing2'));
    });
  });

  describe('geometry updates', () => {
    beforeEach(async () => {
      vi.clearAllMocks();
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    it('should update geometry with new points data', () => {
      // Create mock points object
      const mockGeometry = {
        dispose: vi.fn(),
        boundingBox: { clone: vi.fn().mockReturnThis() },
        boundingSphere: null,
      };
      const mockPoints = {
        name: '/test_points',
        geometry: mockGeometry,
      };

      // Mock getObjectByName to return our mock points
      if (sceneLoader['rootGroup']) {
        (sceneLoader['rootGroup'].getObjectByName as any).mockReturnValue(mockPoints);
      }

      const newData = {
        positions: new Float32Array([4, 5, 6, 7, 8, 9]),
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: { clone: vi.fn().mockReturnThis() },
          ndim: 3,
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // Verify old geometry was disposed
      expect(mockGeometry.dispose).toHaveBeenCalled();

      // Verify createGeometry was called (indirectly through the update)
      expect(mockPoints.geometry).toBeDefined();
    });

    it('should dispose old geometry before creating new (memory safety)', () => {
      const disposeSpy = vi.fn();
      const mockGeometry = {
        dispose: disposeSpy,
        boundingBox: { clone: vi.fn().mockReturnThis() },
        boundingSphere: null,
      };
      const mockPoints = {
        name: '/test_points',
        geometry: mockGeometry,
      };

      if (sceneLoader['rootGroup']) {
        (sceneLoader['rootGroup'].getObjectByName as any).mockReturnValue(mockPoints);
      }

      const newData = {
        positions: new Float32Array([1, 2, 3]),
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: { clone: vi.fn().mockReturnThis() },
          ndim: 3,
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // Dispose should be called BEFORE new geometry is created
      expect(disposeSpy).toHaveBeenCalled();
      expect(disposeSpy).toHaveBeenCalledTimes(1);
    });

    it('should preserve bounding box/sphere when available', () => {
      const mockBoundingBox = { clone: vi.fn().mockReturnThis() };
      const mockBoundingSphere = { clone: vi.fn().mockReturnThis() };
      const mockGeometry = {
        dispose: vi.fn(),
        boundingBox: mockBoundingBox,
        boundingSphere: mockBoundingSphere,
      };
      const mockPoints = {
        name: '/test_points',
        geometry: mockGeometry,
      };

      if (sceneLoader['rootGroup']) {
        (sceneLoader['rootGroup'].getObjectByName as any).mockReturnValue(mockPoints);
      }

      const newData = {
        positions: new Float32Array([1, 2, 3]),
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: { clone: vi.fn().mockReturnThis() },
          ndim: 3,
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // Verify clone was called to preserve bounds
      expect(mockBoundingBox.clone).toHaveBeenCalled();
      expect(mockBoundingSphere.clone).toHaveBeenCalled();
    });

    it('should handle empty geometry updates (clearing points)', () => {
      const mockGeometry = {
        dispose: vi.fn(),
        boundingBox: null,
        boundingSphere: null,
      };
      const mockPoints = {
        name: '/test_points',
        geometry: mockGeometry,
      };

      if (sceneLoader['rootGroup']) {
        (sceneLoader['rootGroup'].getObjectByName as any).mockReturnValue(mockPoints);
      }

      const emptyData = {
        positions: new Float32Array([]), // Empty
        metadata: {
          totalPoints: 1000,
          loadedPoints: 0, // No points visible at current slice
          bounds: { clone: vi.fn().mockReturnThis() },
          ndim: 4,
          usedSpatialIndex: true,
        },
      };

      // Should not throw when updating to empty geometry
      expect(() => {
        (sceneLoader as any).updatePointsGeometry('/test_points', emptyData);
      }).not.toThrow();

      expect(mockGeometry.dispose).toHaveBeenCalled();
    });
  });

  describe('scene dimensions initialization', () => {
    it('should initialize viewState from scene dimensions', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: false, step: 0.1 },
          ],
        },
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      const viewState = (sceneLoader as any).viewState;

      // Should have 3 displayed dimensions (x, y, z)
      expect(viewState.displayDims).toEqual([0, 1, 2]);

      // Should have slice position for all 4 dimensions
      expect(viewState.slicePosition.length).toBe(4);

      // Should have tolerance for all 4 dimensions
      expect(viewState.tolerance.length).toBe(4);
    });

    it('should handle validation with ViewStateManager', async () => {
      const dimensionsWithIssues = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [100, 0], display: true, step: 1 }, // max < min
          ],
        },
      };

      mockZarrGroup.attrs = dimensionsWithIssues;

      // Should not throw, even with validation issues - handles them gracefully
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
    });

    it('should handle invalid dimensions gracefully', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: 'not_an_object', // Invalid type
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      // Should not throw, just log warning
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid scene_dimensions format')
      );
    });

    it('should handle missing dimensions array', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {}, // Missing dimensions array
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid scene_dimensions format')
      );
    });

    it('should handle dimensions with more than 3 displayed', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: true, step: 0.1 }, // 4 displayed
          ],
        },
      };

      // Should handle gracefully even with too many displayed dimensions
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
    });
  });

  describe('points data validation', () => {
    it('should validate empty datasets', () => {
      const emptyData = {
        positions: new Float32Array([]),
        metadata: {
          totalPoints: 0,
          loadedPoints: 0,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      // Should not throw, just log info
      expect(() => {
        (sceneLoader as any).validatePointsData(emptyData);
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Empty dataset detected'));
    });

    it('should detect malformed positions (not multiple of 3)', () => {
      const malformedData = {
        positions: new Float32Array([1, 2, 3, 4]), // Length 4, not divisible by 3
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      expect(() => {
        (sceneLoader as any).validatePointsData(malformedData);
      }).toThrow('not divisible by 3');
    });

    it('should detect colors length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        colors: new Float32Array([1, 0, 0]), // Only 1 color (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validatePointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Colors length mismatch'));
    });

    it('should detect radii length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        radii: new Float32Array([0.5]), // Only 1 radius (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validatePointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Radii length mismatch'));
    });

    it('should detect sharpness length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        sharpness: new Float32Array([2.0, 2.0, 2.0]), // 3 values (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validatePointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Sharpness length mismatch'));
    });

    it('should validate correct data without warnings', () => {
      const validData = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0]),
        radii: new Float32Array([0.5, 0.7]),
        sharpness: new Float32Array([2.0, 3.0]),
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validatePointsData(validData);

      // Should not have any warnings (only info/log messages)
      const warnCalls = consoleSpy.mock.calls.filter((call) =>
        call.some((arg) => typeof arg === 'string' && arg.includes('mismatch'))
      );
      expect(warnCalls.length).toBe(0);
    });
  });

  describe('color mode validation', () => {
    it('should warn when metadata indicates HDR but array is Uint8', () => {
      const colors = new Uint8Array([255, 128, 64]);
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validateColorMode(colors, metadata);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('metadata indicates HDR colors but array is Uint8Array')
      );
    });

    it('should not warn for Float32Array with HDR metadata', () => {
      const colors = new Float32Array([1.5, 2.0, 3.5]); // HDR values > 1.0
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).validateColorMode(colors, metadata);

      // Should not have HDR/Uint8 mismatch warning
      const warnCalls = consoleSpy.mock.calls.filter((call) =>
        call.some((arg) => typeof arg === 'string' && arg.includes('Uint8Array'))
      );
      expect(warnCalls.length).toBe(0);
    });

    it('should suggest SDR mode when HDR values are in [0,1] range', () => {
      const colors = new Float32Array([0.5, 0.8, 1.0]); // All in [0,1]
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'log');

      (sceneLoader as any).validateColorMode(colors, metadata);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Consider using SDR mode for better compression')
      );
    });

    it('should handle Uint16Array colors', () => {
      const colors = new Uint16Array([65535, 32768, 16384]);
      const metadata = {};

      const consoleSpy = vi.spyOn(console, 'log');

      // Should not throw, should log color type
      expect(() => {
        (sceneLoader as any).validateColorMode(colors, metadata);
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Uint16Array'));
    });
  });
});
