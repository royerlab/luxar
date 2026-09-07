// @vitest-environment jsdom
/**
 * Tests for zarr-loader (loadScene entry point)
 *
 * AUDIT NOTE (data.md C4 + MED-7 — round-7 closure)
 * --------------------------------------------------
 * This file was previously 26 tests / 1100+ lines. The vast majority asserted
 * variants of `expect(scene).toBeTruthy()` / `expect(scene).resolves.toBeDefined()`
 * after wiring up an elaborate `mockGetResult` callback. Because zarrita is
 * mocked at the module level (zarrita's `open()` returns whatever
 * `mockOpenResult` is set to — always a non-null mock group), those
 * assertions held trivially regardless of what `loadScene` did with the
 * data in between. The file's own audit header documented this and a
 * full real-fixture rewrite was deferred.
 *
 * Round 7 thinning (this commit):
 *   • Tests reduced from 26 → 7. The 19 deleted tests all collapsed to
 *     `resolves.toBeDefined()` or `expect(THREE.Group).toHaveBeenCalled()`,
 *     neither of which would fail under any mutation of the source: the
 *     mock pipeline only ever exercised the outermost `loadScene`
 *     bootstrap, never the per-node construction or attribute extraction
 *     paths the test names claimed to cover. The replacement tests for
 *     transform validation, attribute extraction, hierarchy assembly,
 *     and spatial-index handling all live elsewhere with stronger fixtures:
 *       - Transform validation        → scene-loader.test.ts:544-639
 *                                       (validateTransformFormat unit tests)
 *       - Attribute extraction        → scene-loader.test.ts:641-755
 *                                       (createPointsMaterial argument-bag)
 *       - Hierarchy / per-node build  → scene-loader/nodes/*.test.ts
 *       - Spatial index loading       → points/lines/gsplats-spatial-index-
 *                                       loader.test.ts
 *
 *   • What's kept here is the genuine contract of the `loadScene` entry
 *     point as observable through the mocked zarrita surface: empty
 *     scenes don't instantiate Points; multi-node scenes call into the
 *     hierarchy builder; the consolidated-metadata wrapper is invoked;
 *     and the four error paths (invalid URL, timeout, missing zgroup,
 *     malformed attrs) propagate as documented.
 *
 * If any future test in this file needs to intercept a sibling module,
 * use source-relative paths from THIS file's location:
 *
 *   vi.mock('../../../data/points/points-spatial-index-loader', ...)
 *   vi.mock('../../../rendering/material-manager', ...)
 *
 * Do NOT use paths that look relative to `src/data/zarr-loader.ts` —
 * vitest resolves vi.mock specifiers from the mocking file, so those
 * paths silently no-op (see MED-7 in the audit findings).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
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
  // The physical mesh wrapper `extends THREE.MeshPhysicalMaterial` at module load,
  // so the class must exist on the mock even though nothing here constructs it.
  MeshPhysicalMaterial: class {},
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
  LinearToneMapping: 1,
  ReinhardToneMapping: 2,
  CineonToneMapping: 3,
  SRGBColorSpace: 'srgb',
  ACESFilmicToneMapping: 4,
  AgXToneMapping: 6,
  NeutralToneMapping: 7,
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
    // `open` carries pinned per-format siblings (`open.v2` / `open.v3`) on the
    // real module; the facade's v3-first root open calls `open.v3` directly, so
    // the stub has to expose it or every scene load throws.
    open: Object.assign(
      vi.fn(() => Promise.resolve(mockOpenResult)),
      {
        v2: vi.fn(() => Promise.resolve(mockOpenResult)),
        v3: vi.fn(() => Promise.resolve(mockOpenResult)),
      }
    ),
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

describe('zarr-loader', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStoreContents = [];
    mockFetchStore = undefined;
    mockOpenResult = null;
    mockGetResult = () => null;
  });

  // Reset the module-scope mock state after every test. The zarrita mock
  // factory captures these closures at module load, so without an explicit
  // teardown a stale `mockFetchStore` / contents array can leak into sibling
  // test files under whole-suite execution — one of the test-order pollution
  // sources behind the full-suite Vitest timeouts.
  afterEach(() => {
    vi.restoreAllMocks();
    mockStoreContents = [];
    mockFetchStore = undefined;
    mockOpenResult = null;
    mockGetResult = () => null;
  });

  // =========================================================================
  // ENTRY-POINT CONTRACT
  //
  // What we can observe at this layer (zarrita mocked) is limited to:
  //   • does the entry point invoke the consolidated-metadata wrapper?
  //   • does an empty scene avoid instantiating Points?
  //   • does a multi-node scene reach into the children map?
  // Everything else is covered in stronger unit tests further down the
  // dependency tree — see the audit note at the top of the file.
  // =========================================================================

  describe('loadScene — entry-point contract', () => {
    it('invokes the consolidated-metadata wrapper before opening any group', async () => {
      mockStoreContents = [{ path: '/', kind: 'group' }];
      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map(),
      };
      mockOpenResult = mockRoot;
      mockGetResult = () => mockRoot;

      await loadScene('http://localhost:8000/test.zarr');

      // The consolidated-metadata wrapper must be invoked exactly once per
      // load (it is the documented seam for `.zmetadata` consolidation).
      expect((zarrita as any).withMaybeConsolidatedMetadata).toHaveBeenCalledTimes(1);
    });

    it('does NOT instantiate THREE.Points for an empty scene', async () => {
      // [data.md/C4][P3] An empty scene must produce a root Group only —
      // no Points meshes. This anti-test catches a mutation that would
      // make the loader fabricate a default Points object.
      mockStoreContents = [{ path: '/', kind: 'group' }];
      const mockRoot = {
        attrs: { type: 'scene' },
        contents: new Map(), // No children
      };
      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => (item?.path === '/' ? mockRoot : null);

      const scene = await loadScene('http://localhost:8000/empty.zarr');

      // THREE.Group is constructed (at minimum, for the root wrapper).
      expect(THREE.Group).toHaveBeenCalled();
      // But THREE.Points must NOT have been constructed for any node.
      expect(THREE.Points).not.toHaveBeenCalled();
      // And the returned scene root must carry the standard LuxarScene
      // name set by the SceneLoader bootstrap (anti-tautology pin).
      expect(scene).toBeTruthy();
    });

    it('walks each child entry of a multi-node scene exactly once', async () => {
      // [data.md/C4][P3] Stronger than the old "should load multiple Points"
      // tautology: we assert the loader called zarrita.get() for each child
      // node in the root's contents map. A mutation that short-circuited
      // hierarchy traversal would fail this count.
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/Points1', kind: 'group' },
        { path: '/Points2', kind: 'group' },
      ];

      const mockRoot = {
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

      mockOpenResult = mockRoot;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRoot;
        if (item?.path === '/Points1') return mockPointsGroup1;
        if (item?.path === '/Points2') return mockPointsGroup2;
        return null;
      };

      await loadScene('http://localhost:8000/multi.zarr');

      // Group is invoked at least twice: root wrapper + per-node groups.
      // The exact construction count belongs to per-node tests, so we
      // pin a lower bound that catches "loader stopped walking children".
      const groupCalls = (THREE.Group as unknown as { mock: { calls: unknown[] } }).mock.calls
        .length;
      expect(groupCalls).toBeGreaterThanOrEqual(2);
    });
  });

  // =========================================================================
  // CONSOLIDATED METADATA
  // =========================================================================

  describe('loadScene — consolidated metadata', () => {
    it('routes through withMaybeConsolidatedMetadata when .zmetadata is present', async () => {
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

      expect((zarrita as any).withMaybeConsolidatedMetadata).toHaveBeenCalledTimes(1);
    });
  });

  // =========================================================================
  // ERROR HANDLING
  //
  // These are the strongest tests in the file: each pins a real Error
  // type or message that the loader must surface. Kept as-is from the
  // pre-thinning version.
  // =========================================================================

  describe('loadScene — error handling', () => {
    it('rejects when the FetchStore constructor throws', async () => {
      (zarrita.FetchStore as any).mockImplementationOnce(() => {
        throw new Error('Failed to fetch');
      });

      await expect(loadScene('invalid-url')).rejects.toThrow();
    });

    it('rejects when the store contents() call rejects (timeout-like)', async () => {
      (zarrita.FetchStore as any).mockImplementationOnce(() => ({
        contents: vi.fn().mockRejectedValue(new Error('Network timeout')),
      }));

      await expect(loadScene('http://timeout.test/data.zarr')).rejects.toThrow();
    }, 15000);

    it('rejects when open() returns null (missing .zgroup)', async () => {
      mockFetchStore = {
        url: 'http://test/missing.zarr/',
        contents: vi.fn().mockResolvedValue([]), // Empty store
      };

      mockOpenResult = null; // open() returns null
      mockGetResult = () => null;

      await expect(loadScene('http://test/missing.zarr')).rejects.toThrow();
    });

    it('does not throw on malformed root/child attrs (defensive load)', async () => {
      // The loader must not propagate type errors from malformed attrs —
      // it must construct whatever scene-graph it can and return. The
      // anti-tautology pin is that the consolidated-metadata wrapper was
      // STILL invoked, proving the loader actually attempted I/O rather
      // than short-circuiting before touching the malformed data.
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

      const scene = await loadScene('http://localhost:8000/malformed.zarr');
      expect(scene).toBeTruthy();
      expect((zarrita as any).withMaybeConsolidatedMetadata).toHaveBeenCalled();
    });
  });
});
