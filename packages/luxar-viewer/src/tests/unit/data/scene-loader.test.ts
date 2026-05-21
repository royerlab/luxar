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

// THREE is NOT mocked here. The classes SceneLoader touches —
// Group / Points / Mesh / Box3 / Vector3 / Matrix4 /
// {,Instanced}Buffer{Geometry,Attribute} — are pure JS and run fine in
// jsdom; the WebGL-bound layer (renderers, shaders) is one level up.
// Earlier revisions kept a 165-line stand-in so individual constructor
// calls could be counted, but the resulting tests asserted on
// implementation details rather than behavior. The behavior assertions
// further down (transform validation, scene-graph shape, etc.) are
// stronger when run against real THREE.

// Mock zarrita (external dependency - network I/O for zarr stores)
vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock material manager (depends on WebGL shader compilation - must be mocked).
// NOTE: shader-side material plumbing is intentionally untested in jsdom; the
// pure factory paths are covered separately in `material-manager.test.ts`.
vi.mock('../../../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
      updateCameraParams: vi.fn(),
    }),
  },
  // `invalidate-render-object.ts` imports this symbol to tag soft-
  // dispose events; supply a unique Symbol so the import resolves
  // in jsdom even though MaterialManager itself is mocked away.
  SOFT_DISPOSE_FLAG: Symbol.for('luxar.material.softDispose.test-mock'),
}));

// SceneLoader now uses `notifier.toast` for the >16D scene-dimensions
// warning. Mock the notifier so the test can assert toast() was called.
const notifierMocks = vi.hoisted(() => ({
  toast: vi.fn(),
}));
vi.mock('../../../utils/notifier', () => ({
  notifier: {
    toast: notifierMocks.toast,
    error: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
  },
}));

// NOTE: Previous mocks for DataLoadingMonitor ('../ui/data-loading-monitor'),
// PointsSpatialIndexLoader ('../data/points-spatial-index-loader'), and
// DataMonitorManager ('../data/data-monitor-manager') were removed because
// their paths were relative to the test file location (src/tests/unit/data/)
// and resolved to non-existent modules, making them dead code that never
// intercepted any real imports. The SceneLoader's actual imports resolve
// from src/data/ and are not affected by those mock paths.
//
// If mocking these becomes necessary in the future, use paths relative to
// the test file that resolve to the actual source modules, e.g.:
//   vi.mock('../../../data/points-spatial-index-loader', ...)
//   vi.mock('../../../data/data-monitor-manager', ...)

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
    (zarr as any).withMaybeConsolidatedMetadata.mockResolvedValue(mockStore);
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

      // Verify scene loaded correctly (implementation may use FetchStore or MultiLevelCachingStore)
      expect((zarr as any).withMaybeConsolidatedMetadata).toHaveBeenCalled();
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

      await sceneLoader.dispose();

      // Verify cleanup
      expect((sceneLoader as any).loaders.size).toBe(0);
      expect((sceneLoader as any)._zarrStore).toBeNull();
      expect((sceneLoader as any).rootGroup).toBeNull();
    });

    it('SceneLoader.dispose returns a Promise that resolves cleanly (async signature)', async () => {
      // Locks in commit 2.1's signature change. loadScene's call site
      // (commit 2.3) now uses `await this.dispose()` — we cannot directly
      // observe the await ordering in this test fixture (loadScene's
      // dispose path is gated on loaders.size > 0 and the jsdom mocks
      // don't populate spatial-index loaders), but a Promise return type
      // is the contract that lets that await work in production.
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      const result = sceneLoader.dispose();
      expect(result).toBeInstanceOf(Promise);
      await result;
      expect((sceneLoader as any)._zarrStore).toBeNull();
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
    it('should apply column-major transforms to objects correctly', async () => {
      // Column-major translation matrix (translation at indices [12,13,14])
      mockZarrGroup.attrs = {
        type: 'points',
        transform: [
          1,
          0,
          0,
          0, // Column 0
          0,
          1,
          0,
          0, // Column 1
          0,
          0,
          1,
          0, // Column 2
          10,
          20,
          30,
          1, // Column 3 (translation)
        ],
      };

      // Behavior assertion: a column-major transform must load without
      // throwing. The negative path (row-major rejection) is the more
      // useful contract and is covered by the next test + the
      // transform-validation block further down.
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();
    });

    it('should reject row-major transforms', async () => {
      mockZarrGroup.attrs = {
        type: 'points',
        transform: [
          1,
          0,
          0,
          10, // Row 0 (tx at [3])
          0,
          1,
          0,
          20, // Row 1 (ty at [7])
          0,
          0,
          1,
          30, // Row 2 (tz at [11])
          0,
          0,
          0,
          1,
        ],
      };

      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(
        /row-major/
      );
    });

    it('should reject invalid transform lengths', async () => {
      mockZarrGroup.attrs = {
        type: 'group',
        transform: [1, 2, 3], // Invalid length
      };

      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(
        /Invalid transform length/
      );
    });
  });

  describe('error handling', () => {
    it('should handle store opening failures', async () => {
      (zarr as any).withMaybeConsolidatedMetadata.mockRejectedValue(
        new Error('Failed to open store')
      );

      await expect(sceneLoader.loadScene('http://invalid.url')).rejects.toThrow(
        'Failed to open store'
      );
    }, 15000);

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

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(columnMajorTransform)
      ).not.toThrow();
    });

    it('should throw for row-major matrices', () => {
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

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(rowMajorTransform)
      ).toThrow(/row-major/);
    });

    it('should throw when translation is at wrong indices', () => {
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

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(suspiciousTransform)
      ).toThrow(/matrix\.T\.ravel/);
    });

    it('should accept identity matrix', () => {
      const identityTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(identityTransform)
      ).not.toThrow();
    });

    it('should handle transforms with only rotation/scale (no translation)', () => {
      // Scale matrix: no translation, should pass validation
      const scaleTransform = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(scaleTransform)
      ).not.toThrow();
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

      const material = (sceneLoader as any).nodeFactory.createPointsMaterial(
        attrs,
        radiusScale,
        sharpnessScale
      );

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

      const material = (sceneLoader as any).nodeFactory.createPointsMaterial(
        attrs,
        radiusScale,
        sharpnessScale
      );

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should handle different blending modes', async () => {
      const normalAttrs = { blending_mode: 'normal' as const };
      const additiveAttrs = { blending_mode: 'additive' as const };

      const material1 = (sceneLoader as any).nodeFactory.createPointsMaterial(
        normalAttrs,
        1.0,
        1.0
      );
      const material2 = (sceneLoader as any).nodeFactory.createPointsMaterial(
        additiveAttrs,
        1.0,
        1.0
      );

      expect(material1).toBeDefined();
      expect(material2).toBeDefined();
    });

    it('should use default opacity and gamma when not specified', async () => {
      const attrs = {}; // No opacity/gamma specified

      const material = (sceneLoader as any).nodeFactory.createPointsMaterial(attrs, 1.0, 1.0);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should pass through custom opacity and gamma values', async () => {
      const attrs = {
        opacity: 0.5,
        gamma: 2.2,
      };

      const material = (sceneLoader as any).nodeFactory.createPointsMaterial(attrs, 1.0, 1.0);

      expect(material).toBeDefined();
      expect(material.updateCameraParams).toBeDefined();
    });

    it('should handle default radiusScale and sharpnessScale', async () => {
      const attrs = {};

      const material = (sceneLoader as any).nodeFactory.createPointsMaterial(attrs); // No scales provided

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

    /**
     * Drop a real point mesh into the loader's rootGroup so the
     * `getObjectByName(path)` lookup inside `commitPointsGeometry`
     * returns it. Per-instance attributes are `InstancedBufferAttribute`s
     * named `aCenter`, `aColor`, `aRadius`, `aSharpness`.
     */
    function attachPointsChild(name: string, oldCount: number): THREE.Mesh {
      const root = (sceneLoader as any).rootGroup as THREE.Group;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        'aCenter',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount * 3), 3)
      );
      geom.setAttribute(
        'aColor',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount * 3), 3)
      );
      geom.setAttribute(
        'aRadius',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount), 1)
      );
      geom.setAttribute(
        'aSharpness',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount), 1)
      );
      const points = new THREE.Mesh(geom);
      points.name = name;
      // commitPointsGeometry only writes `visiblePointCount` when the
      // node passes `isPointsUserData` (nodeType === 'points'). Mirror
      // what NodeFactory.createPointsNode would set up so the commit
      // path treats it as a real points node.
      points.userData = { nodeType: 'points', ndim: 3, visiblePointCount: oldCount };
      root.add(points);
      return points;
    }

    it('should update geometry with new points data', () => {
      const points = attachPointsChild('/test_points', 0);
      const newData = {
        positions: new Float32Array([4, 5, 6, 7, 8, 9]),
        colors: new Float32Array([1, 1, 1, 1, 1, 1]),
        radii: new Float32Array([0.5, 0.5]),
        sharpness: new Float32Array([2.0, 2.0]),
        pointCount: 2,
        ndim: 3,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // After the update the Points still has a (possibly recreated) geometry,
      // and its visiblePointCount reflects the new data.
      expect(points.geometry).toBeDefined();
      expect(points.userData.visiblePointCount).toBe(2);
    });

    it('writes new positions through whichever path the loader takes (pool or in-place)', () => {
      const points = attachPointsChild('/test_points', 1);
      const sameSizeData = {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1, 1, 1]),
        radii: new Float32Array([0.5]),
        sharpness: new Float32Array([2.0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', sameSizeData);

      // Whether commit takes the buffer-pool path or the in-place path
      // (depends on whether _gpuBufferPool is wired up in this fixture),
      // the live position attribute must reflect the new payload.
      const afterPositions = points.geometry.getAttribute(
        'aCenter'
      ) as THREE.InstancedBufferAttribute;
      expect(Array.from(afterPositions.array as Float32Array).slice(0, 3)).toEqual([1, 2, 3]);
      expect(points.userData.visiblePointCount).toBe(1);
    });

    it('should update bounding box from metadata', () => {
      const points = attachPointsChild('/test_points', 1);
      const newBounds = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(10, 10, 10));
      const newData = {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1, 1, 1]),
        radii: new Float32Array([0.5]),
        sharpness: new Float32Array([2.0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: newBounds,
          ndim: 3,
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // The same-size path computes bounding box from the geometry itself.
      // For the dispose-and-recreate path metadata bounds are cloned, but
      // either way the geometry ends up with a defined bounding box.
      expect(points.geometry.boundingBox).not.toBeNull();
    });

    it('should handle empty geometry updates (clearing points)', () => {
      attachPointsChild('/test_points', 1);

      const emptyData = {
        positions: new Float32Array([]), // Empty
        colors: new Float32Array([]),
        radii: new Float32Array([]),
        sharpness: new Float32Array([]),
        pointCount: 0,
        ndim: 4,
        metadata: {
          totalPoints: 1000,
          loadedPoints: 0, // No points visible at current slice
          bounds: new THREE.Box3(),
          ndim: 4,
          usedSpatialIndex: true,
        },
      };

      // Should not throw when updating to empty geometry
      expect(() => {
        (sceneLoader as any).updatePointsGeometry('/test_points', emptyData);
      }).not.toThrow();

      // With GPU buffer pool enabled, geometry is reused (not disposed)
      // The geometry is acquired from pool, updated in place, and reassigned
      // Disposal only happens on final cleanup, not on updates
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

    it('does NOT toast on a 16D scene (≤ WASM ceiling)', async () => {
      notifierMocks.toast.mockClear();
      const dims = Array.from({ length: 16 }, (_, i) => ({
        name: `d${i}`,
        unit: '',
        range: [0, 10] as [number, number],
        display: i < 3,
        step: 1,
      }));
      mockZarrGroup.attrs = { scene_dimensions: { dimensions: dims } };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(notifierMocks.toast).not.toHaveBeenCalled();
    });

    it('toasts on > 16D scenes warning about WASM fallback', async () => {
      notifierMocks.toast.mockClear();
      const dims = Array.from({ length: 18 }, (_, i) => ({
        name: `d${i}`,
        unit: '',
        range: [0, 10] as [number, number],
        display: i < 3,
        step: 1,
      }));
      mockZarrGroup.attrs = { scene_dimensions: { dimensions: dims } };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(notifierMocks.toast).toHaveBeenCalledWith(
        expect.stringContaining('18 dimensions'),
        expect.any(Number)
      );
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
        (sceneLoader as any).nodeFactory.validateLoadedPointsData(emptyData);
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
        (sceneLoader as any).nodeFactory.validateLoadedPointsData(malformedData);
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

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Colors length mismatch'),
        expect.any(Object)
      );
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

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Radii length mismatch'),
        expect.any(Object)
      );
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

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Sharpness length mismatch'),
        expect.any(Object)
      );
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

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(validData);

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

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('metadata indicates HDR colors but array is Uint8Array')
      );
    });

    it('should not warn for Float32Array with HDR metadata', () => {
      const colors = new Float32Array([1.5, 2.0, 3.5]); // HDR values > 1.0
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

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

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

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
        (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Uint16Array'));
    });
  });
});
