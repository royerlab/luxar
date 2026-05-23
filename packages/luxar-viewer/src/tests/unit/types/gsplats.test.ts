import { describe, it, expect } from 'vitest';
import {
  isGSplatsMetadata,
  isGSplatsUserData,
  choleskyPackedSize,
  CHOLESKY_SIZES,
} from '../../../types/gsplats';

describe('GSplats Types', () => {
  describe('isGSplatsMetadata', () => {
    it('should return true for valid gsplats metadata', () => {
      const validMetadata = {
        type: 'gsplats',
        n_splats: 1000,
        ndim: 3,
        has_colors: true,
        chunk_size: 2000,
        amplitude_range: { min: 0.0, max: 10.0 },
        center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
        ordering: 'hilbert',
      };

      expect(isGSplatsMetadata(validMetadata)).toBe(true);
    });

    it('should return false for points metadata', () => {
      const pointsMetadata = {
        type: 'points',
        n_points: 1000,
        ndim: 3,
      };

      expect(isGSplatsMetadata(pointsMetadata)).toBe(false);
    });

    it('should return false for lines metadata', () => {
      const linesMetadata = {
        type: 'lines',
        n_vertices: 100,
        n_segments: 50,
      };

      expect(isGSplatsMetadata(linesMetadata)).toBe(false);
    });

    it('should return false for missing type', () => {
      const invalidMetadata = {
        n_splats: 1000,
        ndim: 3,
      };

      expect(isGSplatsMetadata(invalidMetadata)).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isGSplatsMetadata(null)).toBe(false);
      expect(isGSplatsMetadata(undefined)).toBe(false);
    });

    it('should return false for non-object', () => {
      expect(isGSplatsMetadata('gsplats')).toBe(false);
      expect(isGSplatsMetadata(123)).toBe(false);
      expect(isGSplatsMetadata([])).toBe(false);
    });
  });

  describe('isGSplatsUserData', () => {
    it('should return true for valid gsplats userData', () => {
      const validUserData = {
        nodeType: 'gsplats',
        loader: {}, // Actual loader would be a GSplatsDataLoader instance
        attrs: {
          type: 'gsplats',
          n_splats: 1000,
          ndim: 3,
          has_colors: true,
          chunk_size: 2000,
          amplitude_range: { min: 0.0, max: 10.0 },
          center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
          ordering: 'morton',
        },
      };

      expect(isGSplatsUserData(validUserData)).toBe(true);
    });

    it('should return false for points userData', () => {
      const pointsUserData = {
        nodeType: 'points',
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(pointsUserData)).toBe(false);
    });

    it('should return false for lines userData', () => {
      const linesUserData = {
        nodeType: 'lines',
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(linesUserData)).toBe(false);
    });

    it('should return false for missing nodeType', () => {
      const invalidUserData = {
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(invalidUserData)).toBe(false);
    });

    it('should return false for wrong nodeType', () => {
      const invalidUserData = {
        nodeType: 'group',
        children: [],
      };

      expect(isGSplatsUserData(invalidUserData)).toBe(false);
    });
  });

  describe('choleskyPackedSize', () => {
    it('should return correct size for 2D', () => {
      expect(choleskyPackedSize(2)).toBe(3);
    });

    it('should return correct size for 3D', () => {
      expect(choleskyPackedSize(3)).toBe(6);
    });

    it('should return correct size for 4D', () => {
      expect(choleskyPackedSize(4)).toBe(10);
    });

    it('should return correct size for 5D', () => {
      expect(choleskyPackedSize(5)).toBe(15);
    });

    it('should match CHOLESKY_SIZES constant', () => {
      expect(choleskyPackedSize(2)).toBe(CHOLESKY_SIZES['2D']);
      expect(choleskyPackedSize(3)).toBe(CHOLESKY_SIZES['3D']);
      expect(choleskyPackedSize(4)).toBe(CHOLESKY_SIZES['4D']);
    });
  });

  describe('CHOLESKY_SIZES constant', () => {
    it('should have correct values', () => {
      expect(CHOLESKY_SIZES['2D']).toBe(3);
      expect(CHOLESKY_SIZES['3D']).toBe(6);
      expect(CHOLESKY_SIZES['4D']).toBe(10);
    });

    it('MED-44: covers 5D through 16D (the WASM dimension ceiling)', () => {
      // Regression: the constant previously stopped at 4D even though
      // LoadedGSplatsData and choleskyPackedSize support arbitrary ndim.
      // A 5D+ caller indexing into the constant would have hit undefined
      // and produced silently wrong stride math. The constant is now
      // populated up to 16D (the WASM 16-dim limit per CLAUDE.md), and
      // every entry must match `choleskyPackedSize(n) = n*(n+1)/2`.
      for (let n = 2; n <= 16; n++) {
        const key = `${n}D` as keyof typeof CHOLESKY_SIZES;
        const value = (CHOLESKY_SIZES as Record<string, number>)[key];
        expect(value).toBeDefined();
        expect(value).toBe(choleskyPackedSize(n));
        expect(value).toBe((n * (n + 1)) / 2);
      }
      // Spot-check a few specific values to lock in the table.
      expect(CHOLESKY_SIZES['5D']).toBe(15);
      expect(CHOLESKY_SIZES['8D']).toBe(36);
      expect(CHOLESKY_SIZES['16D']).toBe(136);
    });
  });
});

describe('GSplats Type Definitions', () => {
  describe('GSplatsMetadata interface', () => {
    it('should allow all required fields', () => {
      // This is a compile-time test - if it compiles, the types are correct
      const metadata = {
        type: 'gsplats' as const,
        n_splats: 10000,
        ndim: 3,
        has_colors: true,
        chunk_size: 2000,
        amplitude_range: { min: 0.1, max: 5.0 },
        center_bounds: { min: [0, 0, 0], max: [512, 512, 100] },
        ordering: 'hilbert' as const,
      };

      expect(metadata.type).toBe('gsplats');
      expect(metadata.n_splats).toBe(10000);
      expect(metadata.ndim).toBe(3);
      expect(metadata.has_colors).toBe(true);
      expect(metadata.chunk_size).toBe(2000);
      expect(metadata.amplitude_range.min).toBe(0.1);
      expect(metadata.amplitude_range.max).toBe(5.0);
      expect(metadata.ordering).toBe('hilbert');
    });

    it('should allow optional ordering metadata', () => {
      const metadata = {
        type: 'gsplats' as const,
        n_splats: 5000,
        ndim: 4,
        has_colors: false,
        chunk_size: 1000,
        amplitude_range: { min: 0.0, max: 1.0 },
        center_bounds: { min: [0, 0, 0, 0], max: [100, 100, 100, 10] },
        ordering: 'morton' as const,
        ordering_min: [0, 0, 0, 0],
        ordering_max: [100, 100, 100, 10],
        ordering_bits_per_dim: 16,
      };

      expect(metadata.ordering_min).toEqual([0, 0, 0, 0]);
      expect(metadata.ordering_max).toEqual([100, 100, 100, 10]);
      expect(metadata.ordering_bits_per_dim).toBe(16);
    });

    it('should allow optional rendering attributes', () => {
      const metadata = {
        type: 'gsplats' as const,
        n_splats: 1000,
        ndim: 3,
        has_colors: true,
        chunk_size: 500,
        amplitude_range: { min: 0.0, max: 1.0 },
        center_bounds: { min: [0, 0, 0], max: [10, 10, 10] },
        ordering: 'none' as const,
        transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 5, 5, 1],
        opacity: 0.8,
        gamma: 1.2,
        blending_mode: 'additive' as const,
        extend_to_all: ['Time'],
      };

      expect(metadata.transform?.length).toBe(16);
      expect(metadata.opacity).toBe(0.8);
      expect(metadata.gamma).toBe(1.2);
      expect(metadata.blending_mode).toBe('additive');
      expect(metadata.extend_to_all).toEqual(['Time']);
    });
  });

  describe('LoadedGSplatsData interface', () => {
    it('should represent raw loaded gsplats data', () => {
      const data = {
        positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]), // 3 splats * 3D
        amplitudes: new Float32Array([1.0, 0.5, 0.8]),
        choleskyFactors: new Float32Array([
          1,
          0,
          1,
          0,
          0,
          1, // Splat 0: identity covariance
          2,
          0,
          2,
          0,
          0,
          2, // Splat 1: scaled
          1,
          0.5,
          1,
          0,
          0.5,
          1, // Splat 2: anisotropic
        ]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), // RGB per splat
        splatCount: 3,
        ndim: 3,
      };

      expect(data.splatCount).toBe(3);
      expect(data.ndim).toBe(3);
      expect(data.positions.length).toBe(9); // 3 splats * 3 dims
      expect(data.choleskyFactors.length).toBe(18); // 3 splats * 6 elements
    });

    it('should allow null colors', () => {
      const data = {
        positions: new Float32Array([0, 0, 0]),
        amplitudes: new Float32Array([1.0]),
        choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
        colors: null,
        splatCount: 1,
        ndim: 3,
      };

      expect(data.colors).toBeNull();
    });
  });

  describe('ProcessedGSplatsData interface', () => {
    it('should represent GPU-ready gsplats data', () => {
      const data = {
        centers3D: new Float32Array([0, 0, 0, 1, 1, 1]),
        amplitudes: new Float32Array([0.8, 0.4]), // Attenuated
        choleskyFactors3D: new Float32Array([1, 0, 1, 0, 0, 1, 2, 0, 2, 0, 0, 2]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0]),
        splatCount: 2,
      };

      expect(data.splatCount).toBe(2);
      expect(data.centers3D.length).toBe(6); // 2 splats * 3 dims
      expect(data.choleskyFactors3D.length).toBe(12); // 2 splats * 6 elements
      expect(data.amplitudes[0]).toBeLessThan(1.0); // Attenuated
    });
  });
});
