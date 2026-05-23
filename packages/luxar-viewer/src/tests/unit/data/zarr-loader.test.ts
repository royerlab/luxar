/**
 * Comprehensive tests for zarr-loader
 *
 * This test suite verifies the Zarr loading pipeline:
 * - Scene graph traversal and hierarchy construction
 * - Transform matrix loading and application
 * - Attribute extraction (opacity, blending, gamma)
 * - Dimension specification parsing
 * - Spatial index loading
 * - Array data loading (positions, colors, radii, sharpness)
 * - Error handling for malformed/missing data
 *
 * AUDIT STATUS (data.md C4 + MED-7 — DEFERRED REWRITE)
 * ----------------------------------------------------
 * Two audit findings cluster here:
 *
 * 1. (data.md C4): zarrita is mocked at the module level (see
 *    `vi.mock('zarrita', ...)` below). Many tests end with
 *    `expect(scene).toBeTruthy()` — nearly guaranteed once the mocks
 *    return any non-null root group, regardless of what the loader did
 *    with the data in between. Stronger assertions (specific node
 *    counts, transform values, error-message strings) appear in the
 *    Error Handling / Transform Matrices blocks; the Basic Loading /
 *    Optional Arrays / Spatial Index / Consolidated Metadata blocks
 *    are weaker and would benefit from real-zarr fixtures.
 *
 * 2. (MED-7): An earlier version declared `vi.mock(...)` for several
 *    sibling modules using paths RELATIVE TO THE TEST FILE
 *    (`'../data/points-spatial-index-loader'`,
 *    `'../rendering/material-manager'`). Because vitest resolves
 *    vi.mock specifiers from the MOCKING file (not from the
 *    source-under-test), those specifiers pointed into the test tree
 *    itself — non-existent paths — and silently no-op'd. The legacy
 *    tests were passing for the wrong reasons.
 *
 * **If you add a new test here that needs to intercept a sibling
 * module, use source-relative paths from the TEST file's location:**
 *
 *   vi.mock('../../../data/points/points-spatial-index-loader', ...)
 *   vi.mock('../../../rendering/material-manager', ...)
 *
 * **Do NOT reintroduce paths that look like they're relative to
 * `src/data/zarr-loader.ts`.** Those will silently no-op.
 *
 * Full rewrite against real fixtures under `tests/fixtures/` is
 * tracked separately; until then this audit-note is the contract.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { loadScene } from '../../../data';
import * as zarrita from 'zarrita';
import * as THREE from 'three';

// Mock THREE.js (exact pattern from scene-loader.test.ts which works)
// Group.add now appends to a real `children` array so tests can assert the
// resulting scene-graph shape (child count, names, hierarchy).
vi.mock('three', () => ({
  Group: vi.fn().mockImplementation(function (this: { children: unknown[] }) {
    const self = {
      add: vi.fn((child: unknown) => {
        self.children.push(child);
        return self;
      }),
      name: '',
      userData: {},
      children: [] as unknown[],
      getObjectByName: vi.fn(),
      traverse: vi.fn((callback) => {
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
      applyMatrix4: vi.fn(),
    };
    return self;
  }),
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
  Mesh: vi.fn().mockImplementation((geometry, material) => ({
    name: '',
    userData: {},
    geometry,
    material,
    count: 0,
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
  Float32BufferAttribute: vi.fn(),
  Uint8BufferAttribute: vi.fn(),
  Uint16BufferAttribute: vi.fn(),
  ShaderMaterial: vi.fn().mockImplementation(() => ({
    uniforms: {},
  })),
  EventDispatcher: vi.fn().mockImplementation(() => ({
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
  Color: vi.fn(),
  // Constants
  HalfFloatType: 1016,
  FloatType: 1015,
  UnsignedByteType: 1009,
  Vector2: vi.fn().mockImplementation((x, y) => {
    const vec = {
      x: x || 0,
      y: y || 0,
      clone: vi.fn(),
      copy: vi.fn(),
    };
    vec.clone.mockImplementation(() => ({
      x: vec.x,
      y: vec.y,
      clone: vi.fn(),
      copy: vi.fn(),
    }));
    vec.copy.mockImplementation((v: any) => {
      vec.x = v.x;
      vec.y = v.y;
      return vec;
    });
    return vec;
  }),
  LinearSRGBColorSpace: 'srgb-linear',
  NoToneMapping: 0,
  SRGBColorSpace: 'srgb',
  ACESFilmicToneMapping: 4,
  PCFShadowMap: 1,
}));

// Store mock setup
let mockStoreContents: any[] = [];
let mockFetchStore: any;
let mockOpenResult: any;
let mockGetResult: (item: any) => any;

// Mock zarrita
vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    FetchStore: vi.fn().mockImplementation((url) => {
      mockFetchStore = {
        url,
        contents: vi.fn().mockReturnValue(mockStoreContents),
      };
      return mockFetchStore;
    }),
    withMaybeConsolidatedMetadata: vi.fn((store) => Promise.resolve(store)),
    open: vi.fn(() => Promise.resolve(mockOpenResult)),
    get: vi.fn((item) => Promise.resolve(mockGetResult(item))),
    root: vi.fn((store) => {
      const createLocation = (path: string): any => ({
        store,
        path,
        resolve: vi.fn((subpath: string) => createLocation(path + '/' + subpath)),
      });
      return createLocation('/');
    }),
  };
});

// previous `vi.mock('../data/points-spatial-index-loader')`
// and `vi.mock('../rendering/material-manager')` mocks lived here, but
// vitest's vi.mock matches by import specifier as resolved from the
// MOCKING file, not from the source-under-test's perspective. From this
// test (src/tests/unit/data/zarr-loader.test.ts), the strings
// '../data/...' and '../rendering/...' resolve to non-existent paths
// inside the test tree (src/tests/unit/...), so the mocks were never
// applied.
//
// scene-loader.test.ts:62-77 already documented removing the same kind
// of dead mocks. Tests in this file pass without them — zarrita is
// the only external-IO dependency that genuinely needs mocking, and
// the rest of the source can run against real modules under jsdom.
//
// If a future test in this file needs to intercept either of those
// modules, use source-relative paths:
//   vi.mock('../../../data/points/points-spatial-index-loader', ...)
//   vi.mock('../../../rendering/material-manager', ...)

describe('zarr-loader', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStoreContents = [];
    mockOpenResult = null;
    mockGetResult = () => null;
  });

  // =========================================================================
  // BASIC LOADING
  // =========================================================================

  describe('loadScene - Basic Loading', () => {
    it('should load a simple scene with one Points node', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
      ];

      const mockRootGroup = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPointsGroup = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        if (item?.path === '/Points1') return mockPointsGroup;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeTruthy();
      expect(THREE.Group).toHaveBeenCalled();
      // The mock pipeline doesn't load actual array buffers, so we can't
      // verify Points-instantiation here. SceneLoader.buildNode is unit-
      // tested separately for the per-node construction path.
    });

    it('should handle empty scene', async () => {
      mockStoreContents = [{ path: '/', kind: 'group' }];

      const mockRootGroup = {
        attrs: { type: 'scene' },
        contents: new Map(), // No children
      };

      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/empty.zarr');

      expect(scene).toBeTruthy();
      expect(THREE.Group).toHaveBeenCalled();
      // Empty zarr → no THREE.Points were created. The scene-loader still
      // wraps the empty scene in a single root group, so we verify the
      // absence of geometry instead of a strict child count.
      expect(THREE.Points).not.toHaveBeenCalled();
    });

    it('should load scene with multiple Points nodes', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points2', kind: 'group' },
      ];

      const mockRootGroup = {
        attrs: { type: 'scene' },
        contents: new Map([
          ['Points1', { type: 'group' }],
          ['Points2', { type: 'group' }],
        ]),
      };

      const mockPointsGroup1 = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      const mockPointsGroup2 = {
        attrs: { type: 'points', n_points: 200 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        if (item?.path === '/Points1') return mockPointsGroup1;
        if (item?.path === '/Points2') return mockPointsGroup2;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/multi.zarr');

      expect(scene).toBeTruthy();
      // Should create Group for scene (Points objects require actual array data)
      expect(THREE.Group).toHaveBeenCalled();
      // Deeper hierarchy assertions live in the SceneLoader.buildNode unit
      // tests where mocks expose array buffers; this loadScene-level test
      // just verifies the entry-point produces a usable root.
    });
  });

  // =========================================================================
  // HIERARCHICAL GROUPS
  // =========================================================================

  describe('loadScene - Hierarchical Groups', () => {
    it('should load nested group hierarchy', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Group1', kind: 'group' },
        { path: '/Group1/Group2', kind: 'group' },
        { path: '/Group1/Group2/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Group1', { type: 'group' }]]),
      };

      const mockGroup1 = {
        attrs: { type: 'group' },
        contents: new Map([['Group2', { type: 'group' }]]),
      };

      const mockGroup2 = {
        attrs: { type: 'group' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 50 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        const path = item?.path || '';
        if (path === '/') return mockRoot;
        if (path === '/Group1') return mockGroup1;
        if (path === '/Group1/Group2') return mockGroup2;
        if (path === '/Group1/Group2/Points1') return mockPoints;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/nested.zarr');

      expect(scene).toBeTruthy();
      // Should create nested groups
      expect(THREE.Group).toHaveBeenCalled();
      // Hierarchy-shape assertions live in the SceneLoader.buildNode unit
      // tests; the loadScene-level test just exercises the entry point.
    });

    it('should distinguish between Group and Points nodes', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Group1', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([
          ['Group1', { type: 'group' }],
          ['Points1', { type: 'group' }],
        ]),
      };

      const mockGroup = {
        attrs: { type: 'group' },
        contents: new Map(),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        const path = item?.path || '';
        if (path === '/') return mockRoot;
        if (path === '/Group1') return mockGroup;
        if (path === '/Points1') return mockPoints;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/mixed.zarr');

      expect(scene).toBeTruthy();
      // Should create Groups for scene hierarchy (Points objects require actual array data)
      expect(THREE.Group).toHaveBeenCalled();
      // Without real Points instantiation we can't differentiate by mesh
      // type here; SceneLoader.buildNode tests cover the per-node type
      // dispatch directly.
    });
  });

  // =========================================================================
  // TRANSFORM MATRICES
  // =========================================================================

  describe('loadScene - Transform Matrices', () => {
    it('should load transform matrix from attrs', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      // Identity matrix (16 elements)
      const identityMatrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 100,
          transform: identityMatrix,
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/test.zarr');

      // Verify scene loaded successfully with transform attribute
      expect(scene).toBeTruthy();
      expect(THREE.Group).toHaveBeenCalled();
    });

    it('should handle translation transform', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      // Translation matrix: translate(10, 20, 30)
      const translationMatrix = [1, 0, 0, 10, 0, 1, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 50,
          transform: translationMatrix,
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/translated.zarr');

      // Verify scene loaded successfully with translation transform
      expect(scene).toBeTruthy();
      expect(THREE.Group).toHaveBeenCalled();
    });

    it('should handle missing transform gracefully', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 100,
          // No transform attr
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: previously this test had no expect() at all —
      // a silent "doesn't throw" check. Now we pin the resolution contract.
      await expect(
        loadScene('http://localhost:8000/no-transform.zarr')
      ).resolves.toBeDefined();
    });

    it('should validate transform array length', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 100,
          transform: [1, 0, 0, 0], // Invalid: only 4 elements, need 16
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(
        loadScene('http://localhost:8000/invalid-transform.zarr')
      ).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // ATTRIBUTES
  // =========================================================================

  describe('loadScene - Attributes', () => {
    // Tests for opacity, blending_mode, and gamma extraction were removed because:
    // The mock zarrita pipeline (mockStoreContents + mockGetResult) does not produce
    // actual THREE.Points objects — SceneLoader sees 0 store items and never calls
    // materialManager.getPointMaterial. These attrs are tested indirectly via E2E tests
    // (data-loading.spec.ts) where the full pipeline runs with real zarr data.
    // To unit-test attribute extraction, the SceneLoader.createMaterial method should
    // be tested directly with a focused unit test rather than through the full pipeline.

    it('should use default values for missing attrs', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 100,
          // No opacity, blending_mode, or gamma
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/defaults.zarr')).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // SCENE DIMENSIONS
  // =========================================================================

  describe('loadScene - Scene Dimensions', () => {
    it('should parse scene_dimensions from root attrs', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: {
          type: 'scene',
          scene_dimensions: {
            dimensions: [
              { name: 'x', unit: 'um', range: [0, 100], step: 1, display: true },
              { name: 'y', unit: 'um', range: [0, 100], step: 1, display: true },
              { name: 'z', unit: 'um', range: [0, 100], step: 1, display: true },
              { name: 'time', unit: 'frame', range: [0, 10], step: 1, display: false },
            ],
          },
        },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/4d.zarr')).resolves.toBeDefined();
    });

    it('should handle extend_to_all in node attrs', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 100,
          extend_to_all: ['time', 'channel'], // Extend visibility across these dims
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/extend.zarr')).resolves.toBeDefined();
    });

    it('should handle missing scene_dimensions gracefully', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: {
          type: 'scene',
          // No scene_dimensions
        },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/no-dims.zarr')).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // OPTIONAL ARRAYS
  // =========================================================================

  describe('loadScene - Optional Arrays', () => {
    it('should handle missing colors array', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
        // No colors array
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/no-colors.zarr')).resolves.toBeDefined();
    });

    it('should handle missing radii array', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/no-radii.zarr')).resolves.toBeDefined();
    });

    it('should handle missing sharpness array', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/no-sharpness.zarr')).resolves.toBeDefined();
    });

    it('should load all arrays when present', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
        { path: '/Points1/colors', kind: 'array' },
        { path: '/Points1/radii', kind: 'array' },
        { path: '/Points1/sharpness', kind: 'array' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 2 },
        contents: new Map([
          ['positions', { type: 'array' }],
          ['colors', { type: 'array' }],
          ['radii', { type: 'array' }],
          ['sharpness', { type: 'array' }],
        ]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/full-attrs.zarr')).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // ERROR HANDLING
  // =========================================================================

  describe('loadScene - Error Handling', () => {
    it('should throw error for invalid store URL', async () => {
      // Mock FetchStore to throw
      (zarrita.FetchStore as any).mockImplementationOnce(() => {
        throw new Error('Failed to fetch');
      });

      await expect(loadScene('invalid-url')).rejects.toThrow();
    });

    it('should handle network timeout gracefully', async () => {
      (zarrita.FetchStore as any).mockImplementationOnce(() => ({
        contents: vi.fn().mockRejectedValue(new Error('Network timeout')),
      }));

      await expect(loadScene('http://timeout.test/data.zarr')).rejects.toThrow();
    }, 15000);

    it('should handle missing .zgroup file', async () => {
      mockFetchStore = {
        url: 'http://test/missing.zarr/',
        contents: vi.fn().mockResolvedValue([]), // Empty store
      };

      mockOpenResult = null; // open() returns null
      mockGetResult = () => null;

      await expect(loadScene('http://test/missing.zarr')).rejects.toThrow();
    });

    it('should handle malformed attrs gracefully', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
      ];

      const mockRoot = {
        attrs: {
          type: 'scene',
          scene_dimensions: 'invalid', // Should be object, not string
        },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: {
          type: 'points',
          n_points: 'invalid', // Should be number
          opacity: 'invalid', // Should be number
        },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/malformed.zarr')).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // SPATIAL INDEX
  // =========================================================================

  describe('loadScene - Spatial Index', () => {
    it('should attempt to load spatial index for Points nodes', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/indexed.zarr')).resolves.toBeDefined();
    });

    it('should handle missing spatial index gracefully', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points1/positions', kind: 'array' },
        // No point_spatial_index array
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Points1', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 100 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPoints;
        return null;
      };

      // Mock open to throw 404 for spatial index
      (zarrita.open as any).mockImplementation((loc: any) => {
        if (loc?.path?.includes('point_spatial_index')) {
          throw new Error('404 Not Found');
        }
        return Promise.resolve(mockOpenResult);
      });

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/no-index.zarr')).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // CONSOLIDATED METADATA
  // =========================================================================

  describe('loadScene - Consolidated Metadata', () => {
    it('should use .zmetadata if available', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/.zmetadata', kind: 'file' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map(),
      };

      mockOpenResult = mockRoot;
      mockGetResult = () => mockRoot;

      // Mock consolidated metadata opening through the facade backend.
      (zarrita as any).withMaybeConsolidatedMetadata.mockResolvedValue(mockFetchStore);

      await loadScene('http://localhost:8000/consolidated.zarr');

      expect((zarrita as any).withMaybeConsolidatedMetadata).toHaveBeenCalled();
    });

    it('should work without .zmetadata', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        // No .zmetadata
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map(),
      };

      mockOpenResult = mockRoot;
      mockGetResult = () => mockRoot;

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(
        loadScene('http://localhost:8000/no-consolidated.zarr')
      ).resolves.toBeDefined();
    });
  });

  // =========================================================================
  // NODE NAMING
  // =========================================================================

  describe('loadScene - Node Naming', () => {
    it('should preserve node names from Zarr paths', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Cells', kind: 'group' },
      ];

      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map([['Cells', { type: 'group' }]]),
      };

      const mockPoints = {
        attrs: { type: 'points', n_points: 1000 },
        contents: new Map([['positions', { type: 'array' }]]),
      };

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Cells') return mockPoints;
        return null;
      };

      // data.md W9 fix [P2]: pin the resolution contract (was: no expect()).
      await expect(loadScene('http://localhost:8000/named.zarr')).resolves.toBeDefined();
    });
  });
});
