/**
 * Tests for effective radius calculation for nD hypersphere slicing.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateEffectiveRadii,
  calculateSpatialQueryTolerance,
  shouldApplyEffectiveRadius,
  type EffectiveRadiusConfig,
} from '../data/effective-radius-calculator';
import { ViewState } from '../data/data-loader-types';

describe('effective-radius-calculator', () => {
  describe('calculateEffectiveRadii', () => {
    it('should return original radii when point is exactly on slice plane', () => {
      // Setup: 4D point at origin with radius 1.0
      const positions = new Float32Array([0, 0, 0, 0]);
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2], // First 3 dims displayed
        slicePosition: [0, 0, 0, 0], // Slice at origin
        tolerance: [0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true], // All spatial
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Point is exactly on slice plane, so effective radius = original radius
      expect(result[0]).toBeCloseTo(1.0);
    });

    it('should reduce radius based on distance in non-displayed spatial dimensions only', () => {
      // Setup: 5D point with mixed spatial/non-spatial dimensions
      const positions = new Float32Array([0, 0, 0, 0.6, 0]); // 0.6 units away in dim 3
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2], // First 3 dims displayed
        slicePosition: [0, 0, 0, 0, 0], // Slice at origin
        tolerance: [0.1, 0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true, false], // Dim 4 is non-spatial (e.g., time)
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 5);

      // R_effective = √(1² - 0.6²) = √(1 - 0.36) = √0.64 = 0.8
      expect(result[0]).toBeCloseTo(0.8);
    });

    it('should ignore non-spatial dimensions when calculating distance', () => {
      // Setup: 5D point with distance in both spatial and non-spatial dims
      const positions = new Float32Array([0, 0, 0, 0.6, 10.0]); // Far away in non-spatial dim 4
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0, 0], // Slice at origin
        tolerance: [0.1, 0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true, false], // Dim 4 is non-spatial
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 5);

      // Should only consider distance in dim 3 (0.6), not dim 4 (10.0)
      expect(result[0]).toBeCloseTo(0.8);
    });

    it('should return zero for points at hypersphere boundary', () => {
      // Setup: Point at exact radius distance
      const positions = new Float32Array([0, 0, 0, 1.0]); // Exactly 1.0 away
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // R_effective = √(1² - 1²) = 0
      expect(result[0]).toBeCloseTo(0);
    });

    it('should handle multiple points with different radii', () => {
      // Setup: 3 points at different distances
      const positions = new Float32Array([
        0,
        0,
        0,
        0.0, // Point 1: on slice
        0,
        0,
        0,
        0.3, // Point 2: 0.3 away
        0,
        0,
        0,
        0.5, // Point 3: 0.5 away
      ]);
      const radii = new Float32Array([1.0, 0.5, 2.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 2.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Point 1: √(1² - 0²) = 1.0
      expect(result[0]).toBeCloseTo(1.0);
      // Point 2: √(0.5² - 0.3²) = √(0.25 - 0.09) = √0.16 = 0.4
      expect(result[1]).toBeCloseTo(0.4);
      // Point 3: √(2² - 0.5²) = √(4 - 0.25) = √3.75 ≈ 1.936
      expect(result[2]).toBeCloseTo(1.936, 2);
    });

    it('should handle complex mixed spatial and non-spatial dimensions', () => {
      // Setup: 6D data with spatial xyz, non-spatial time, spatial depth, non-spatial channel
      const positions = new Float32Array([
        0,
        0,
        0,
        5.0,
        0.3,
        2.0, // time=5, depth=0.3, channel=2
      ]);
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2], // xyz displayed
        slicePosition: [0, 0, 0, 5.0, 0, 1.0], // Exact time match, different depth and channel
        tolerance: [0.1, 0.1, 0.1, 0.001, 0.1, 0.001],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, false, true, false], // time and channel non-spatial
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 6);

      // Only depth (dim 4) contributes: distance = 0.3 - 0 = 0.3
      // R_effective = √(1² - 0.3²) = √0.91 ≈ 0.954
      expect(result[0]).toBeCloseTo(0.954, 2);
    });
  });

  describe('calculateSpatialQueryTolerance', () => {
    it('should use maxRadius for non-displayed spatial dimensions', () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0, 0],
        tolerance: [], // No explicit tolerance
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true, false],
        maxRadius: 0.5,
      };

      const result = calculateSpatialQueryTolerance(viewState, config, 5);

      expect(result[0]).toBe(0); // Displayed
      expect(result[1]).toBe(0); // Displayed
      expect(result[2]).toBe(0); // Displayed
      expect(result[3]).toBe(0.5); // Non-displayed spatial → maxRadius
      expect(result[4]).toBe(0); // Non-spatial → exact match (0 tolerance)
    });

    it('should use zero tolerance for non-spatial dimensions (exact match)', () => {
      const viewState: ViewState = {
        displayDims: [0, 1],
        slicePosition: [0, 0, 0, 0],
        tolerance: [],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, false, false], // Last two are non-spatial
        maxRadius: 1.0,
      };

      const result = calculateSpatialQueryTolerance(viewState, config, 4);

      expect(result[0]).toBe(0); // Displayed
      expect(result[1]).toBe(0); // Displayed
      expect(result[2]).toBe(0); // Non-spatial (e.g., time) → exact match
      expect(result[3]).toBe(0); // Non-spatial (e.g., category) → exact match
    });

    it('should always use maxRadius for spatial dimensions, ignoring tolerance array', () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.1, 0.2, 0.3, 0.4], // Explicit tolerances (should be ignored for spatial dims)
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateSpatialQueryTolerance(viewState, config, 4);

      expect(result[0]).toBe(0); // Displayed
      expect(result[1]).toBe(0); // Displayed
      expect(result[2]).toBe(0); // Displayed
      expect(result[3]).toBe(1.0); // Always use maxRadius for spatial dims, not tolerance
    });
  });

  describe('shouldApplyEffectiveRadius', () => {
    it('should return true when non-displayed spatial dimensions exist', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true, false],
        maxRadius: 1.0,
      };
      const displayDims = [0, 1, 2]; // Dim 3 is spatial but not displayed

      const result = shouldApplyEffectiveRadius(config, displayDims, true);

      expect(result).toBe(true);
    });

    it('should return false when all spatial dimensions are displayed', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, false, false],
        maxRadius: 1.0,
      };
      const displayDims = [0, 1, 2]; // All spatial dims are displayed

      const result = shouldApplyEffectiveRadius(config, displayDims, true);

      expect(result).toBe(false);
    });

    it('should return false when no radii data is available', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 1.0,
      };
      const displayDims = [0, 1, 2];

      const result = shouldApplyEffectiveRadius(config, displayDims, false);

      expect(result).toBe(false);
    });

    it('should return false when config is null', () => {
      const result = shouldApplyEffectiveRadius(null, [0, 1, 2], true);

      expect(result).toBe(false);
    });

    it('should handle edge case with all dimensions displayed', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true],
        maxRadius: 1.0,
      };
      const displayDims = [0, 1, 2]; // All dims displayed

      const result = shouldApplyEffectiveRadius(config, displayDims, true);

      expect(result).toBe(false);
    });

    it('should handle mixed spatial configuration correctly', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, false, true, true, false, true],
        maxRadius: 1.0,
      };
      const displayDims = [0, 2]; // Display spatial dims 0 and 2

      const result = shouldApplyEffectiveRadius(config, displayDims, true);

      // Dims 3 and 5 are spatial but not displayed
      expect(result).toBe(true);
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle empty point arrays', () => {
      const positions = new Float32Array(0);
      const radii = new Float32Array(0);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      expect(result.length).toBe(0);
    });

    it('should handle points beyond hypersphere boundary gracefully', () => {
      // Point at distance > radius
      const positions = new Float32Array([0, 0, 0, 2.0]); // 2.0 away but radius is 1.0
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Should clamp to 0 when distance > radius (negative under sqrt)
      expect(result[0]).toBe(0);
    });

    it('should handle all dimensions being displayed', () => {
      const positions = new Float32Array([1, 2, 3]);
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2], // All 3 dims displayed
        slicePosition: [0, 0, 0],
        tolerance: [0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 3);

      // No non-displayed dimensions, so radius unchanged
      expect(result[0]).toBe(1.0);
    });
  });
});
