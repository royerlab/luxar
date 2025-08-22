/**
 * Tests for Zarr loader utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  normalizeZarrPath,
  extractDimensionMetadata,
  inheritRenderingAttributes,
  validatePointCloudData,
  calculateInitialSlicePosition,
  isPointCloudGroup,
  calculateBoundingBox,
  processTransformAttribute,
  estimatePointCloudMemory,
  validateRenderingAttributes,
  determineLoadingStrategy,
  getArrayConstructor,
} from '../data/zarr-loader-utils';
import { DimensionsBuilder } from './builders/test-data-builders';

describe('zarr-loader-utils', () => {
  describe('normalizeZarrPath', () => {
    it('should handle absolute URLs', () => {
      const url = 'https://example.com/data.zarr';
      expect(normalizeZarrPath(url)).toBe('https://example.com/data.zarr/');

      const withSlash = 'https://example.com/data.zarr/';
      expect(normalizeZarrPath(withSlash)).toBe(withSlash);
    });

    it('should handle relative paths', () => {
      const path = 'data/test.zarr';
      const normalized = normalizeZarrPath(path, 'http://localhost:3000');
      expect(normalized).toBe('http://localhost:3000/data/test.zarr/');
    });

    it('should handle paths starting with slash', () => {
      const path = '/data/test.zarr';
      const normalized = normalizeZarrPath(path, 'http://localhost:3000');
      expect(normalized).toBe('http://localhost:3000/data/test.zarr/');
    });
  });

  describe('extractDimensionMetadata', () => {
    it('should extract metadata from new format', () => {
      const attrs = {
        dimensions: {
          metadata: {
            0: { name: 'x', unit: 'μm', range: [0, 100] },
            1: { name: 'y', unit: 'μm', range: [0, 100] },
          },
        },
      };

      const metadata = extractDimensionMetadata(attrs);
      expect(metadata).toEqual(attrs.dimensions.metadata);
    });

    it('should convert legacy format', () => {
      const attrs = {
        dimensions: {
          names: ['x', 'y', 'z'],
          units: ['μm', 'μm', 'μm'],
          ranges: [
            [0, 100],
            [0, 100],
            [0, 50],
          ],
          displayed: [0, 1, 2],
        },
      };

      const metadata = extractDimensionMetadata(attrs);
      expect(metadata).not.toBeNull();
      expect(metadata![0].name).toBe('x');
      expect(metadata![0].unit).toBe('μm');
      expect(metadata![0].range).toEqual([0, 100]);
      expect(metadata![0].display).toBe(true);
    });

    it('should return null for missing dimensions', () => {
      expect(extractDimensionMetadata({})).toBeNull();
      expect(extractDimensionMetadata(null)).toBeNull();
    });
  });

  describe('inheritRenderingAttributes', () => {
    it('should inherit missing attributes from parent', () => {
      const attrs = { opacity: 0.5 };
      const parentAttrs = { opacity: 1.0, gamma: 1.5, blending_mode: 'additive' };

      const result = inheritRenderingAttributes(attrs, parentAttrs);

      expect(result.opacity).toBe(0.5); // Keep own value
      expect(result.gamma).toBe(1.5); // Inherit from parent
      expect(result.blending_mode).toBe('additive'); // Inherit from parent
    });

    it('should handle no parent attributes', () => {
      const attrs = { opacity: 0.8 };
      const result = inheritRenderingAttributes(attrs);

      expect(result).toEqual(attrs);
    });

    it('should preserve all child attributes when present', () => {
      const attrs = {
        opacity: 0.5,
        gamma: 1.0,
        blending_mode: 'normal',
      };
      const parentAttrs = {
        opacity: 1.0,
        gamma: 2.0,
        blending_mode: 'additive',
      };

      const result = inheritRenderingAttributes(attrs, parentAttrs);
      expect(result).toEqual(attrs);
    });
  });

  describe('validatePointCloudData', () => {
    it('should validate correct data', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
      const result = validatePointCloudData(positions, 2, 3);

      expect(result).toBe(positions);
    });

    it('should throw on point count mismatch', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]);

      expect(() => validatePointCloudData(positions, 3, 3)).toThrow('Position data mismatch');
    });

    it('should throw on invalid values', () => {
      const positions = new Float32Array([1, 2, NaN, 4, 5, 6]);

      expect(() => validatePointCloudData(positions, 2, 3)).toThrow('Invalid position value');
    });

    it('should throw on infinity values', () => {
      const positions = new Float32Array([1, 2, Infinity, 4, 5, 6]);

      expect(() => validatePointCloudData(positions, 2, 3)).toThrow('Invalid position value');
    });
  });

  describe('calculateInitialSlicePosition', () => {
    it('should set non-displayed dimensions to range minimum for discrete', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withDisplayed(0, 1, 2)
        .withDimension(3, 'time', 's', [5, 15], { discrete: true })
        .build();

      const position = calculateInitialSlicePosition(dims);

      expect(position[3]).toBe(5); // Minimum for discrete
    });

    it('should set non-displayed dimensions to center for continuous', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withDisplayed(0, 1, 2)
        .withDimension(3, 'time', 's', [0, 10], { discrete: false })
        .build();

      const position = calculateInitialSlicePosition(dims);

      expect(position[3]).toBe(5); // Center for continuous
    });

    it('should handle all dimensions displayed', () => {
      const dims = new DimensionsBuilder().withNDimensions(3).withDisplayed(0, 1, 2).build();

      const position = calculateInitialSlicePosition(dims);

      expect(position).toEqual([0, 0, 0]);
    });
  });

  describe('isPointCloudGroup', () => {
    it('should identify by type attribute', () => {
      expect(isPointCloudGroup({ type: 'points' }, 'group')).toBe(true);
      expect(isPointCloudGroup({ type: 'pointcloud' }, 'group')).toBe(true);
      expect(isPointCloudGroup({ type: 'mesh' }, 'group')).toBe(false);
    });

    it('should identify by arrays attribute', () => {
      expect(isPointCloudGroup({ arrays: ['positions', 'colors'] }, 'group')).toBe(true);
      expect(isPointCloudGroup({ arrays: ['vertices'] }, 'group')).toBe(false);
    });

    it('should identify by group name', () => {
      expect(isPointCloudGroup({}, 'points_001')).toBe(true);
      expect(isPointCloudGroup({}, 'pointcloud_data')).toBe(true);
      expect(isPointCloudGroup({}, 'particles')).toBe(true);
      expect(isPointCloudGroup({}, 'mesh_data')).toBe(false);
    });
  });

  describe('calculateBoundingBox', () => {
    it('should calculate correct bounds', () => {
      const positions = new Float32Array([0, 0, 0, 1, 2, 3, -1, -2, -3, 5, 5, 5]);

      const bounds = calculateBoundingBox(positions, 3);

      expect(bounds.min).toEqual([-1, -2, -3]);
      expect(bounds.max).toEqual([5, 5, 5]);
    });

    it('should handle empty positions', () => {
      const positions = new Float32Array([]);
      const bounds = calculateBoundingBox(positions, 3);

      expect(bounds.min).toEqual([0, 0, 0]);
      expect(bounds.max).toEqual([0, 0, 0]);
    });

    it('should handle nD data', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

      const bounds = calculateBoundingBox(positions, 4);

      expect(bounds.min).toEqual([1, 2, 3, 4]);
      expect(bounds.max).toEqual([5, 6, 7, 8]);
    });
  });

  describe('processTransformAttribute', () => {
    it('should handle flat array', () => {
      const flat = Array(16)
        .fill(0)
        .map((_, i) => i);
      const result = processTransformAttribute(flat);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result!.length).toBe(16);
      expect(Array.from(result!)).toEqual(flat);
    });

    it('should handle 4x4 matrix', () => {
      const matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ];

      const result = processTransformAttribute(matrix);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result![0]).toBe(1); // [0][0]
      expect(result![5]).toBe(1); // [1][1]
      expect(result![10]).toBe(1); // [2][2]
      expect(result![15]).toBe(1); // [3][3]
    });

    it('should handle object with matrix property', () => {
      const obj = {
        matrix: Array(16)
          .fill(0)
          .map((_, i) => i),
      };

      const result = processTransformAttribute(obj);
      expect(result).toBeInstanceOf(Float32Array);
    });

    it('should return null for invalid input', () => {
      expect(processTransformAttribute(null)).toBeNull();
      expect(processTransformAttribute([])).toBeNull();
      expect(processTransformAttribute([1, 2, 3])).toBeNull();
    });
  });

  describe('estimatePointCloudMemory', () => {
    it('should calculate memory for basic points', () => {
      const memory = estimatePointCloudMemory(1000, 3);
      expect(memory).toBeCloseTo(0.0114, 3); // 1000 * 3 * 4 / (1024*1024)
    });

    it('should include optional attributes', () => {
      const memory = estimatePointCloudMemory(1000, 3, true, true, true);
      // positions: 1000 * 3 * 4 = 12000
      // colors: 1000 * 3 * 4 = 12000
      // radii: 1000 * 4 = 4000
      // sharpness: 1000 * 4 = 4000
      // total: 32000 bytes = 0.0305 MB
      expect(memory).toBeCloseTo(0.0305, 3);
    });
  });

  describe('validateRenderingAttributes', () => {
    it('should apply defaults', () => {
      const result = validateRenderingAttributes({});

      expect(result.opacity).toBe(1);
      expect(result.gamma).toBe(1);
      expect(result.blending_mode).toBe('normal');
      expect(result.point_size).toBe(0.1);
    });

    it('should clamp values to valid ranges', () => {
      const result = validateRenderingAttributes({
        opacity: 2,
        gamma: -1,
        point_size: -5,
      });

      expect(result.opacity).toBe(1); // Clamped to max
      expect(result.gamma).toBe(0.1); // Clamped to min
      expect(result.point_size).toBe(0.001); // Clamped to min
    });
  });

  describe('determineLoadingStrategy', () => {
    it('should return full for small datasets', () => {
      const strategy = determineLoadingStrategy(1000, 1000);
      expect(strategy).toBe('full');
    });

    it('should return chunked for medium datasets', () => {
      const strategy = determineLoadingStrategy(1000000, 100);
      expect(strategy).toBe('chunked');
    });

    it('should return lazy for large datasets', () => {
      const strategy = determineLoadingStrategy(10000000, 100);
      expect(strategy).toBe('lazy');
    });
  });

  describe('getArrayConstructor', () => {
    it('should return Float32Array for float types', () => {
      expect(getArrayConstructor('<f4')).toBe(Float32Array);
      expect(getArrayConstructor('float32')).toBe(Float32Array);
      expect(getArrayConstructor('<f8')).toBe(Float32Array); // Downcast
    });

    it('should return Uint8Array for uint8 types', () => {
      expect(getArrayConstructor('|u1')).toBe(Uint8Array);
      expect(getArrayConstructor('uint8')).toBe(Uint8Array);
    });

    it('should default to Float32Array', () => {
      expect(getArrayConstructor('unknown')).toBe(Float32Array);
    });
  });
});
