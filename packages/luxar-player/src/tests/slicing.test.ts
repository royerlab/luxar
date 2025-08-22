/**
 * Comprehensive tests for nD slicing algorithms
 *
 * These tests verify the core slicing logic that enables navigation
 * through high-dimensional datasets. The slicing module is critical
 * for the entire nD visualization pipeline.
 */

import { describe, it, expect } from 'vitest';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColors,
  sliceColorsFloat32,
  computeEffectiveRadii,
  sliceScalarAttribute,
  getDimsLabel,
} from '../utils/slicing';
import { SimpleDims } from '../types/dims';

describe('slicing', () => {
  describe('slicePoints', () => {
    it('should return all points when all dimensions are displayed', () => {
      // 3D points, all dimensions displayed
      const positions = new Float32Array([
        0,
        0,
        0, // Point 0
        1,
        1,
        1, // Point 1
        2,
        2,
        2, // Point 2
      ]);

      const dims: SimpleDims = {
        ndim: 3,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0],
      };

      const result = slicePoints(positions, dims, 3);

      expect(result).toHaveLength(3);
      expect(Array.from(result)).toEqual([0, 1, 2]);
    });

    it('should filter points based on non-displayed continuous dimensions', () => {
      // 4D points: x, y, z, time
      // Display only x, y, z; slice at time=0.5
      const positions = new Float32Array([
        0,
        0,
        0,
        0.0, // Point 0 at time=0.0
        1,
        1,
        1,
        0.4, // Point 1 at time=0.4 (within tolerance)
        2,
        2,
        2,
        0.6, // Point 2 at time=0.6 (within tolerance)
        3,
        3,
        3,
        1.5, // Point 3 at time=1.5 (outside tolerance)
      ]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2], // Display x, y, z
        currentStep: [0, 0, 0, 0.5], // Slice at time=0.5
      };

      const result = slicePoints(positions, dims, 4, undefined, 0.2);

      // Points 1 and 2 are within tolerance (0.2) of time=0.5
      expect(Array.from(result)).toEqual([1, 2]);
    });

    it('should use per-point radii when provided', () => {
      // 4D points with varying radii
      const positions = new Float32Array([
        0,
        0,
        0,
        0.0, // Point 0 at time=0.0
        1,
        1,
        1,
        0.3, // Point 1 at time=0.3
        2,
        2,
        2,
        0.8, // Point 2 at time=0.8
      ]);

      const radii = new Float32Array([
        0.1, // Small radius for point 0
        0.3, // Medium radius for point 1
        0.5, // Large radius for point 2
      ]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0.5], // Slice at time=0.5
      };

      const result = slicePoints(positions, dims, 3, radii);

      // Point 0: |0.0 - 0.5| = 0.5 > 0.1 (excluded)
      // Point 1: |0.3 - 0.5| = 0.2 < 0.3 (included)
      // Point 2: |0.8 - 0.5| = 0.3 < 0.5 (included)
      expect(Array.from(result)).toEqual([1, 2]);
    });

    it('should handle discrete dimensions with exact matching', () => {
      // 4D points with discrete time dimension
      const positions = new Float32Array([
        0,
        0,
        0,
        0, // Point 0 at time frame 0
        1,
        1,
        1,
        1, // Point 1 at time frame 1
        2,
        2,
        2,
        1, // Point 2 at time frame 1
        3,
        3,
        3,
        2, // Point 3 at time frame 2
      ]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 1], // Select time frame 1
        metadata: {
          3: { discrete: true }, // Time is discrete
        },
      };

      const result = slicePoints(positions, dims, 4);

      // Only points at exactly time=1 are included
      expect(Array.from(result)).toEqual([1, 2]);
    });

    it('should handle multiple non-displayed dimensions', () => {
      // 5D points: x, y, z, time, channel
      const positions = new Float32Array([
        0,
        0,
        0,
        0.5,
        1, // Point 0: time=0.5, channel=1
        1,
        1,
        1,
        0.5,
        2, // Point 1: time=0.5, channel=2
        2,
        2,
        2,
        1.5,
        1, // Point 2: time=1.5, channel=1
        3,
        3,
        3,
        0.5,
        1, // Point 3: time=0.5, channel=1 (matches both)
      ]);

      const dims: SimpleDims = {
        ndim: 5,
        displayed: [0, 1, 2], // Display only x, y, z
        currentStep: [0, 0, 0, 0.5, 1], // Slice at time=0.5, channel=1
        metadata: {
          4: { discrete: true }, // Channel is discrete
        },
      };

      const result = slicePoints(positions, dims, 4, undefined, 0.1);

      // Point 0: time matches (within tolerance), channel matches exactly
      // Point 1: time matches but channel doesn't (excluded)
      // Point 2: channel matches but time doesn't (excluded)
      // Point 3: both match (included)
      expect(Array.from(result)).toEqual([0, 3]);
    });

    it('should handle edge case with no points passing filter', () => {
      const positions = new Float32Array([
        0,
        0,
        0,
        10, // All points far from slice
        1,
        1,
        1,
        20,
      ]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0], // Slice at time=0
      };

      const result = slicePoints(positions, dims, 2, undefined, 0.1);

      expect(result).toHaveLength(0);
    });
  });

  describe('extractDisplayDimensions', () => {
    it('should extract 3D positions from nD data', () => {
      // 5D points
      const positions = new Float32Array([
        1,
        2,
        3,
        4,
        5, // Point 0
        6,
        7,
        8,
        9,
        10, // Point 1
      ]);

      const indices = new Uint32Array([0, 1]);

      const dims: SimpleDims = {
        ndim: 5,
        displayed: [0, 2, 4], // Display dimensions 0, 2, 4
        currentStep: [0, 0, 0, 0, 0],
      };

      const result = extractDisplayDimensions(positions, indices, dims);

      expect(result).toHaveLength(6); // 2 points * 3 coords
      expect(Array.from(result)).toEqual([
        1,
        3,
        5, // Point 0: dims 0, 2, 4
        6,
        8,
        10, // Point 1: dims 0, 2, 4
      ]);
    });

    it('should pad with zeros when fewer than 3 dimensions displayed', () => {
      // 4D points, display only 2D
      const positions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

      const indices = new Uint32Array([0, 1]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [1, 3], // Display only 2 dimensions
        currentStep: [0, 0, 0, 0],
      };

      const result = extractDisplayDimensions(positions, indices, dims);

      expect(Array.from(result)).toEqual([
        2,
        4,
        0, // Point 0: dims 1, 3, padded
        6,
        8,
        0, // Point 1: dims 1, 3, padded
      ]);
    });

    it('should handle single dimension display', () => {
      const positions = new Float32Array([
        1,
        2,
        3, // Point 0
        4,
        5,
        6, // Point 1
      ]);
      const indices = new Uint32Array([0, 1]);

      const dims: SimpleDims = {
        ndim: 3,
        displayed: [1], // Display only y
        currentStep: [0, 0, 0],
      };

      const result = extractDisplayDimensions(positions, indices, dims);

      expect(Array.from(result)).toEqual([
        2,
        0,
        0, // Point 0: y=2, padded
        5,
        0,
        0, // Point 1: y=5, padded
      ]);
    });

    it('should handle reordering based on indices', () => {
      const positions = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);

      const indices = new Uint32Array([3, 1]); // Reverse order

      const dims: SimpleDims = {
        ndim: 3,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0],
      };

      const result = extractDisplayDimensions(positions, indices, dims);

      expect(Array.from(result)).toEqual([
        3,
        3,
        3, // Point 3 first
        1,
        1,
        1, // Point 1 second
      ]);
    });
  });

  describe('sliceColors', () => {
    it('should extract colors for visible points', () => {
      const colors = new Uint8Array([
        255,
        0,
        0, // Red
        0,
        255,
        0, // Green
        0,
        0,
        255, // Blue
        255,
        255,
        0, // Yellow
      ]);

      const indices = new Uint32Array([0, 2, 3]);

      const result = sliceColors(colors, indices);

      expect(result).toHaveLength(9); // 3 points * 3 channels
      expect(Array.from(result!)).toEqual([
        255,
        0,
        0, // Red (point 0)
        0,
        0,
        255, // Blue (point 2)
        255,
        255,
        0, // Yellow (point 3)
      ]);
    });

    it('should handle null colors', () => {
      const indices = new Uint32Array([0, 1]);
      const result = sliceColors(null, indices);
      expect(result).toBeNull();
    });

    it('should handle empty indices', () => {
      const colors = new Uint8Array([255, 0, 0]);
      const indices = new Uint32Array([]);
      const result = sliceColors(colors, indices);
      expect(result).toHaveLength(0);
    });
  });

  describe('sliceColorsFloat32', () => {
    it('should extract HDR colors for visible points', () => {
      const colors = new Float32Array([
        1.0,
        0.0,
        0.0, // Normal red
        0.0,
        2.5,
        0.0, // HDR green (>1.0)
        0.0,
        0.0,
        0.5, // Dark blue
        10.0,
        10.0,
        10.0, // HDR white
      ]);

      const indices = new Uint32Array([1, 3]);

      const result = sliceColorsFloat32(colors, indices);

      expect(result).toHaveLength(6); // 2 points * 3 channels
      expect(Array.from(result!)).toEqual([
        0.0,
        2.5,
        0.0, // HDR green
        10.0,
        10.0,
        10.0, // HDR white
      ]);
    });

    it('should preserve HDR values above 1.0', () => {
      const colors = new Float32Array([
        5.0,
        10.0,
        15.0, // Very bright HDR color
      ]);

      const indices = new Uint32Array([0]);
      const result = sliceColorsFloat32(colors, indices);

      expect(result![0]).toBe(5.0);
      expect(result![1]).toBe(10.0);
      expect(result![2]).toBe(15.0);
    });

    it('should handle null colors', () => {
      const indices = new Uint32Array([0, 1]);
      const result = sliceColorsFloat32(null, indices);
      expect(result).toBeNull();
    });
  });

  describe('computeEffectiveRadii', () => {
    it('should compute correct effective radii for sliced hyperspheres', () => {
      // 4D points with radii
      const positions = new Float32Array([
        0,
        0,
        0,
        0, // Point at origin
        1,
        1,
        1,
        0.3, // Point offset in time
        2,
        2,
        2,
        0.4, // Point offset in time
      ]);

      const indices = new Uint32Array([0, 1, 2]);
      const originalRadii = new Float32Array([1.0, 1.0, 0.5]);

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2], // Display x, y, z
        currentStep: [0, 0, 0, 0], // Slice at time=0
      };

      const result = computeEffectiveRadii(positions, indices, dims, originalRadii);

      // Point 0: at slice plane, full radius = 1.0
      expect(result[0]).toBeCloseTo(1.0);

      // Point 1: distance 0.3 from slice, effective radius = √(1² - 0.3²) ≈ 0.954
      expect(result[1]).toBeCloseTo(Math.sqrt(1 - 0.3 * 0.3));

      // Point 2: distance 0.4 from slice, effective radius = √(0.5² - 0.4²) = 0.3
      expect(result[2]).toBeCloseTo(0.3);
    });

    it('should handle points at hypersphere boundary', () => {
      const positions = new Float32Array([
        0,
        0,
        0,
        1.0, // Point at distance 1.0 from slice
      ]);

      const indices = new Uint32Array([0]);
      const originalRadii = new Float32Array([1.0]); // Radius equals distance

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0],
      };

      const result = computeEffectiveRadii(positions, indices, dims, originalRadii);

      // At boundary: effective radius should be 0
      expect(result[0]).toBeCloseTo(0);
    });

    it('should handle multiple non-displayed dimensions', () => {
      // 5D point offset in two non-displayed dimensions
      const positions = new Float32Array([
        0,
        0,
        0,
        0.3,
        0.4, // Point offset in dims 3 and 4
      ]);

      const indices = new Uint32Array([0]);
      const originalRadii = new Float32Array([1.0]);

      const dims: SimpleDims = {
        ndim: 5,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0, 0],
      };

      const result = computeEffectiveRadii(positions, indices, dims, originalRadii);

      // Distance = √(0.3² + 0.4²) = 0.5
      // Effective radius = √(1² - 0.5²) = √0.75 ≈ 0.866
      expect(result[0]).toBeCloseTo(Math.sqrt(0.75));
    });
  });

  describe('sliceScalarAttribute', () => {
    it('should extract scalar attributes for visible points', () => {
      const attribute = new Float32Array([0.1, 0.5, 0.9, 0.3]);
      const indices = new Uint32Array([1, 3]);

      const result = sliceScalarAttribute(attribute, indices);

      expect(result).toHaveLength(2);
      expect(result![0]).toBeCloseTo(0.5);
      expect(result![1]).toBeCloseTo(0.3);
    });

    it('should handle null attributes', () => {
      const indices = new Uint32Array([0, 1]);
      const result = sliceScalarAttribute(null, indices);
      expect(result).toBeNull();
    });

    it('should preserve attribute order based on indices', () => {
      const attribute = new Float32Array([1, 2, 3, 4, 5]);
      const indices = new Uint32Array([4, 2, 0]); // Reverse selection

      const result = sliceScalarAttribute(attribute, indices);

      expect(Array.from(result!)).toEqual([5, 3, 1]);
    });
  });

  describe('getDimsLabel', () => {
    it('should generate label for non-displayed dimensions', () => {
      const dims: SimpleDims = {
        ndim: 5,
        displayed: [0, 1, 2], // x, y, z displayed
        currentStep: [0, 0, 0, 5.25, 2.0],
        metadata: {
          3: { name: 'Time', unit: 's' },
          4: { name: 'Channel', unit: '' },
        },
      };

      const label = getDimsLabel(dims);

      expect(label).toBe('Time=5.25s, Channel=2.00');
    });

    it('should use default names when metadata is missing', () => {
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1],
        currentStep: [0, 0, 3.14, 2.71],
      };

      const label = getDimsLabel(dims);

      expect(label).toBe('D2=3.14, D3=2.71');
    });

    it('should handle all dimensions displayed', () => {
      const dims: SimpleDims = {
        ndim: 3,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0],
      };

      const label = getDimsLabel(dims);

      expect(label).toBe(''); // No non-displayed dimensions
    });

    it('should handle single non-displayed dimension', () => {
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 42.5],
        metadata: {
          3: { name: 'Depth', unit: 'μm' },
        },
      };

      const label = getDimsLabel(dims);

      expect(label).toBe('Depth=42.50μm');
    });

    it('should format numbers to 2 decimal places', () => {
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 3.14159],
      };

      const label = getDimsLabel(dims);

      expect(label).toBe('D3=3.14');
    });
  });

  describe('edge cases and performance', () => {
    it('should handle large datasets efficiently', () => {
      // Create a large dataset
      const numPoints = 10000;
      const positions = new Float32Array(numPoints * 4);

      // Half points at time=0, half at time=1
      for (let i = 0; i < numPoints; i++) {
        positions[i * 4] = Math.random();
        positions[i * 4 + 1] = Math.random();
        positions[i * 4 + 2] = Math.random();
        positions[i * 4 + 3] = i < numPoints / 2 ? 0 : 1;
      }

      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0],
        metadata: {
          3: { discrete: true },
        },
      };

      const startTime = performance.now();
      const result = slicePoints(positions, dims, numPoints);
      const elapsed = performance.now() - startTime;

      expect(result).toHaveLength(numPoints / 2);
      expect(elapsed).toBeLessThan(50); // Should be fast
    });

    it('should handle degenerate cases gracefully', () => {
      // Empty positions
      const emptyPositions = new Float32Array(0);
      const dims: SimpleDims = {
        ndim: 3,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0],
      };

      const result = slicePoints(emptyPositions, dims, 0);
      expect(result).toHaveLength(0);

      // No displayed dimensions
      const positions = new Float32Array([1, 2, 3]);
      const noDims: SimpleDims = {
        ndim: 3,
        displayed: [],
        currentStep: [0, 0, 0],
      };

      const extracted = extractDisplayDimensions(positions, new Uint32Array([0]), noDims);
      expect(Array.from(extracted)).toEqual([0, 0, 0]);
    });
  });
});
