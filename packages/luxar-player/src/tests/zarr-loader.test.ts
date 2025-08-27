import { describe, it, expect, vi, beforeEach } from 'vitest';
import { loadScene } from '../data';
import * as zarrita from 'zarrita';
import * as THREE from 'three';

// Mock Three.js
vi.mock('three', () => ({
  Group: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    name: '',
    userData: {},
  })),
  BufferGeometry: vi.fn().mockImplementation(() => ({
    setAttribute: vi.fn(),
    computeBoundingSphere: vi.fn(),
  })),
  BufferAttribute: vi.fn(),
  Float32BufferAttribute: vi.fn(),
  Uint8BufferAttribute: vi.fn(),
  Points: vi.fn().mockImplementation(() => ({
    name: '',
    userData: {},
  })),
  Color: vi.fn(),
  Matrix4: vi.fn().mockImplementation(() => ({
    fromArray: vi.fn().mockReturnThis(),
  })),
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
  HalfFloatType: 1016,
  LinearSRGBColorSpace: 'srgb-linear',
  NoToneMapping: 0,
  SRGBColorSpace: 'srgb',
  ACESFilmicToneMapping: 4,
  PCFSoftShadowMap: 2,
}));

// Store mock setup
let mockStoreContents: any;
let mockFetchStore: any;

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
    tryWithConsolidated: vi.fn((store) => Promise.resolve(store)),
    open: vi.fn(() => Promise.resolve(mockOpenResult)),
    get: vi.fn((item) => Promise.resolve(mockGetResult(item))),
    root: vi.fn((store) => ({
      store,
      path: '/',
      resolve: vi.fn((path) => ({ store, path: '/' + path })),
    })),
  };
});

// Mock material manager - updated to use getPointMaterial
vi.mock('../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {
        hdrMultiplier: { value: 1.0 },
        opacity: { value: 1.0 },
        gamma: { value: 1.0 },
        baseAlpha: { value: 0.01 },
        fov: { value: 1.047 },
        resolution: { value: { x: 1, y: 1 } },
      },
      userData: {},
      updateCameraParams: vi.fn(),
      updateHDRMultiplier: vi.fn(),
      updateOpacity: vi.fn(),
      updateGamma: vi.fn(),
    }),
    updateCameraParams: vi.fn(),
    updateHDRMultiplier: vi.fn(),
  },
  BlendingMode: {},
}));

// Mock implementations
let mockOpenResult: any;
let mockGetResult: any;

describe('zarr_loader', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock data
    mockStoreContents = [];
    mockOpenResult = null;
    mockGetResult = () => null;
  });

  describe('loadScene', () => {
    it('should load a scene from zarr store', async () => {
      // Setup mock data
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/PointCloud1', kind: 'group' },
        { path: '/PointCloud1/positions', kind: 'array' },
        { path: '/PointCloud1/colors', kind: 'array' },
        { path: '/PointCloud1/radii', kind: 'array' },
        { path: '/PointCloud1/sharpness', kind: 'array' },
      ];

      const mockRootGroup = {
        attrs: {
          scene_type: 'points',
          version: '1.0',
        },
        contents: new Map([['PointCloud1', { type: 'group' }]]),
      };

      const mockNodeGroup = {
        attrs: {
          node_type: 'points',
          n_points: 1000,
        },
        contents: new Map([
          ['positions', { type: 'array' }],
          ['colors', { type: 'array' }],
          ['radii', { type: 'array' }],
          ['sharpness', { type: 'array' }],
        ]),
      };

      const mockArrays = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        colors: new Uint8Array([255, 0, 0, 0, 255, 0]), // 2 colors
        radii: new Float32Array([0.1, 0.2]),
        sharpness: new Float32Array([2.0, 3.0]),
      };

      // Setup mock responses
      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        if (item?.path === '/PointCloud1') return mockNodeGroup;
        // Return array data based on the node path
        const pathMatch = item?.store?.url?.match(/\/PointCloud1\/(\w+)\/?$/);
        if (pathMatch) {
          return mockArrays[pathMatch[1] as keyof typeof mockArrays];
        }
        return null;
      };

      const scene = await loadScene('http://localhost:8000/test.zarr');

      expect(THREE.Group).toHaveBeenCalled();
      expect(scene).toBeDefined();
      expect(zarrita.FetchStore).toHaveBeenCalledWith('http://localhost:8000/test.zarr/');
    });

    it('should handle missing optional arrays gracefully', async () => {
      mockStoreContents = [
        { path: '/', kind: 'group' },
        { path: '/TestNode', kind: 'group' },
        { path: '/TestNode/positions', kind: 'array' },
      ];

      const mockRootGroup = {
        attrs: { scene_type: 'points' },
        contents: new Map([['TestNode', { type: 'group' }]]),
      };

      const mockNodeGroup = {
        attrs: { node_type: 'points', n_points: 100 },
        contents: new Map([
          ['positions', { type: 'array' }], // Only positions, no optional arrays
        ]),
      };

      const mockPositions = new Float32Array([1, 2, 3]); // 1 point

      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        if (item?.path === '/TestNode') return mockNodeGroup;
        if (item?.path?.includes('TestNode')) return mockNodeGroup;
        const pathMatch = item?.store?.url?.match(/\/TestNode\/positions\/?$/);
        if (pathMatch) return mockPositions;
        // Handle array data access
        if (item?.data) return item;
        if (item === mockPositions) return { data: mockPositions };
        return null;
      };

      // Update open mock to handle different paths
      (zarrita.open as any).mockImplementation((loc: any) => {
        if (loc?.path === '/TestNode') return Promise.resolve(mockNodeGroup);
        if (loc?.path?.includes('positions'))
          return Promise.resolve({
            data: mockPositions,
            shape: [1, 3],
            dtype: '<f4',
          });
        return Promise.resolve(mockOpenResult);
      });

      const scene = await loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeDefined();
      // Should not throw errors even without colors/radii/sharpness
      expect(THREE.Group).toHaveBeenCalled();
    });

    it('should throw error for invalid store URL', async () => {
      // Mock FetchStore to throw when contents() is called
      (zarrita.FetchStore as any).mockImplementationOnce(() => ({
        contents: vi.fn().mockRejectedValue(new Error('Failed to fetch')),
      }));

      await expect(loadScene('invalid-url')).rejects.toThrow('Cannot read properties of null');
    });

    it('should handle empty store', async () => {
      mockStoreContents = [{ path: '/', kind: 'group' }];

      const mockRootGroup = {
        attrs: { scene_type: 'points' },
        contents: new Map(), // No nodes
      };

      mockOpenResult = mockRootGroup;
      mockGetResult = (item: any) => {
        if (item?.path === '/') return mockRootGroup;
        return null;
      };

      const scene = await loadScene('http://localhost:8000/empty.zarr');

      expect(scene).toBeDefined();
      expect(THREE.Group).toHaveBeenCalled();
      // Scene should be empty but valid
    });
  });
});
