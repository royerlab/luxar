/**
 * Tests for effective radius calculation for nD hypersphere slicing.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateEffectiveRadii,
  calculateSpatialQueryTolerance,
  shouldApplyEffectiveRadius,
  type EffectiveRadiusConfig,
} from '../../../data/points/effective-radius-calculator';
import { ViewState } from '../../../data/data-loader-types';

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
      expect(result[0]).toBeCloseTo(1.0, 5);
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
      expect(result[0]).toBeCloseTo(0.8, 5);
    });

    it('should ignore non-spatial dimensions when calculating distance', () => {
      // Setup: 5D point with distance in spatial dim 3 and matching discrete dim 4
      // Discrete dim 4 matches slice position, so point is visible
      const positions = new Float32Array([0, 0, 0, 0.6, 0]); // Matches slice in discrete dim 4
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0, 0], // Slice at origin
        tolerance: [0.1, 0.1, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, true, false], // Dim 4 is non-spatial (discrete)
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 5);

      // Should only consider distance in dim 3 (0.6), not dim 4 (discrete, exact match)
      // R_eff = sqrt(1^2 - 0.6^2) = sqrt(1 - 0.36) = sqrt(0.64) = 0.8
      expect(result[0]).toBeCloseTo(0.8, 5);
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
      // [ndim.md/C4][P2] strict equality — boundary is exactly zero,
      // a tight check kills mutants that return tiny non-zero values.
      expect(result[0]).toBe(0);
    });

    it('MED-12: exact-boundary point (D === R) returns exactly 0 via sqrt path, not fallback', () => {
      // Regression: previously `effectiveRadiusSquared > 0` sent the
      // boundary case (D == R, value == 0) into the fallback branch. With
      // `>= 0`, the exact-zero argument goes through Math.sqrt(0) === 0,
      // which is mathematically the correct path and preserves the
      // "boundary is just barely visible" semantic (returns 0 not via
      // a discontinuous clamp from the negative side).
      const positions = new Float32Array([0, 0, 0, 1.0]);
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

      // Must be exactly 0 (not just close to 0) — sqrt(0) === 0.
      expect(result[0]).toBe(0);
      // And not NaN (would happen if the clamp were `> 0` AND tiny float
      // drift made R² - D² fractionally negative without the fallback).
      expect(Number.isNaN(result[0])).toBe(false);
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
      expect(result[0]).toBeCloseTo(1.0, 5);
      // Point 2: √(0.5² - 0.3²) = √(0.25 - 0.09) = √0.16 = 0.4
      expect(result[1]).toBeCloseTo(0.4, 5);
      // Point 3: √(2² - 0.5²) = √(4 - 0.25) = √3.75
      // [ndim.md/C3][P5] use Float32-appropriate precision (5 digits, ~1e-5).
      expect(result[2]).toBeCloseTo(Math.sqrt(3.75), 5);
    });

    it('should handle complex mixed spatial and non-spatial dimensions', () => {
      // Setup: 6D data with spatial xyz, non-spatial time, spatial depth, non-spatial channel
      // Discrete dimensions (time=dim3, channel=dim5) must match slice position exactly
      const positions = new Float32Array([
        0,
        0,
        0,
        5.0,
        0.3,
        1.0, // time=5 (matches), depth=0.3 (spatial offset), channel=1 (matches)
      ]);
      const radii = new Float32Array([1.0]);
      const viewState: ViewState = {
        displayDims: [0, 1, 2], // xyz displayed
        slicePosition: [0, 0, 0, 5.0, 0, 1.0], // time=5, depth=0, channel=1
        tolerance: [0.1, 0.1, 0.1, 0.001, 0.1, 0.001],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, false, true, false], // time and channel non-spatial
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 6);

      // Discrete dims (time, channel) match exactly
      // Only depth (dim 4) contributes spatial distance: 0.3 - 0 = 0.3
      // R_effective = √(1² - 0.3²) = √0.91
      // [ndim.md/C3][P5] use Float32-appropriate precision against the
      // exact expected value (not a rounded literal).
      expect(result[0]).toBeCloseTo(Math.sqrt(0.91), 5);
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

      // Displayed dimensions use infinite tolerance (1e10) to show all points
      expect(result[0]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[1]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[2]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[3]).toBe(0.5); // Non-displayed spatial → maxRadius
      expect(result[4]).toBe(0.5); // Non-spatial → 0.5 tolerance for chunk query (precise filtering done in calculateEffectiveRadii)
    });

    it('should use 0.5 tolerance for non-spatial dimensions (for chunk queries)', () => {
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

      // Displayed dimensions use infinite tolerance (1e10) to show all points
      expect(result[0]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[1]).toBe(1e10); // Displayed → infinite tolerance
      // Non-spatial (discrete) dimensions use 0.5 tolerance for chunk queries
      // This matches the discreteTolerance in calculateEffectiveRadii and handles
      // float precision issues in chunk bounds. The precise filtering is done
      // in calculateEffectiveRadii with discreteTolerance = 0.5
      expect(result[2]).toBe(0.5); // Non-spatial → 0.5 tolerance for chunk query
      expect(result[3]).toBe(0.5); // Non-spatial → 0.5 tolerance for chunk query
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

      // Displayed dimensions use infinite tolerance (1e10) to show all points
      expect(result[0]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[1]).toBe(1e10); // Displayed → infinite tolerance
      expect(result[2]).toBe(1e10); // Displayed → infinite tolerance
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

    it('should return true when there are non-displayed discrete dimensions', () => {
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [true, true, true, false, false],
        maxRadius: 1.0,
      };
      const displayDims = [0, 1, 2]; // All spatial dims are displayed, but dims 3,4 are discrete

      const result = shouldApplyEffectiveRadius(config, displayDims, true);

      // Should return true because dims 3 and 4 are non-displayed (discrete) dims that need filtering
      expect(result).toBe(true);
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

  describe('discrete dimension filtering', () => {
    it('should set radius to zero for points not matching discrete dimension value', () => {
      // Simulating quantum orbitals: dim 0 is orbital index (discrete), dims 1,2,3 are spatial xyz
      // spatialExtendDims = [false, true, true, true] - orbital index is NOT spatial
      // displayDims = [1, 2, 3] - showing x, y, z
      // This means we're slicing on orbital index (dim 0)
      const positions = new Float32Array([
        0,
        0,
        0,
        0, // Point 1: orbital 0, at origin
        1,
        0,
        0,
        0, // Point 2: orbital 1, at origin
        2,
        0,
        0,
        0, // Point 3: orbital 2, at origin
        0,
        1,
        0,
        0, // Point 4: orbital 0, offset in x
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2, 3], // xyz displayed
        slicePosition: [0, 0, 0, 0], // Looking at orbital 0
        tolerance: [0, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true], // Orbital index is discrete
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Points 1 and 4 match orbital 0 → full radius
      expect(result[0]).toBe(1.0);
      expect(result[3]).toBe(1.0);
      // Points 2 and 3 don't match orbital 0 → filtered out (radius = 0)
      expect(result[1]).toBe(0);
      expect(result[2]).toBe(0);
    });

    it('should apply tolerance to discrete dimension matching', () => {
      // Test that discrete matching uses the 0.5 tolerance for floating point comparison
      const positions = new Float32Array([
        0.4,
        0,
        0,
        0, // Orbital ~0, within tolerance
        0.6,
        0,
        0,
        0, // Orbital ~1, outside tolerance
        1.4,
        0,
        0,
        0, // Orbital ~1, within tolerance of 1
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2, 3],
        slicePosition: [0, 0, 0, 0], // Looking at orbital 0
        tolerance: [0, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // 0.4 is within 0.5 of 0 → visible
      expect(result[0]).toBe(1.0);
      // 0.6 is outside 0.5 of 0 → filtered out
      expect(result[1]).toBe(0);
      // 1.4 is outside 0.5 of 0 → filtered out
      expect(result[2]).toBe(0);
    });

    it('should filter out points from wrong discrete value even when all spatial dims displayed', () => {
      // Key bug case: all spatial dimensions displayed, but discrete dimension needs filtering
      // This is the quantum orbitals scenario where shouldApplyEffectiveRadius must return true
      const positions = new Float32Array([
        0,
        1,
        2,
        3, // Orbital 0 at (1,2,3)
        5,
        1,
        2,
        3, // Orbital 5 at (1,2,3)
      ]);
      const radii = new Float32Array([1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2, 3], // All spatial dims displayed
        slicePosition: [0, 0, 0, 0], // Looking at orbital 0
        tolerance: [0, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true], // Dim 0 is discrete (orbital index)
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Orbital 0 visible, orbital 5 filtered out
      expect(result[0]).toBe(1.0);
      expect(result[1]).toBe(0);
    });

    it('should require ALL discrete dimensions to match (not just one)', () => {
      // 5D data: dim0=time(discrete), dim1,2,3=xyz(spatial), dim4=channel(discrete)
      // Point must match BOTH time AND channel to be visible
      const positions = new Float32Array([
        0,
        0,
        0,
        0,
        0, // time=0, channel=0 (both match)
        0,
        0,
        0,
        0,
        1, // time=0, channel=1 (channel mismatch)
        1,
        0,
        0,
        0,
        0, // time=1, channel=0 (time mismatch)
        1,
        0,
        0,
        0,
        1, // time=1, channel=1 (both mismatch)
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2, 3], // xyz displayed
        slicePosition: [0, 0, 0, 0, 0], // time=0, channel=0
        tolerance: [0, 0.1, 0.1, 0.1, 0],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true, false], // time and channel are discrete
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 5);

      // Only point 0 matches both discrete dimensions
      expect(result[0]).toBe(1.0); // time=0, channel=0 → visible
      expect(result[1]).toBe(0); // time=0, channel=1 → filtered
      expect(result[2]).toBe(0); // time=1, channel=0 → filtered
      expect(result[3]).toBe(0); // time=1, channel=1 → filtered
    });

    it('should combine discrete filtering with spatial distance calculation', () => {
      // 5D: dim0=orbital(discrete), dim1,2=xy(displayed), dim3=z(spatial, not displayed), dim4=time(discrete)
      // Points must match orbital AND time, then z-distance affects radius
      const positions = new Float32Array([
        0,
        0,
        0,
        0.0,
        0, // orbital=0, z=0.0, time=0 (full match)
        0,
        0,
        0,
        0.6,
        0, // orbital=0, z=0.6, time=0 (spatial offset)
        1,
        0,
        0,
        0.0,
        0, // orbital=1, z=0.0, time=0 (orbital mismatch)
        0,
        0,
        0,
        0.0,
        1, // orbital=0, z=0.0, time=1 (time mismatch)
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2], // xy displayed
        slicePosition: [0, 0, 0, 0, 0], // orbital=0, z=0, time=0
        tolerance: [0, 0.1, 0.1, 0.1, 0],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true, false], // orbital,time=discrete; xyz=spatial
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 5);

      // Point 0: matches discrete, z=0 → full radius
      expect(result[0]).toBe(1.0);
      // Point 1: matches discrete, z=0.6 → reduced radius = sqrt(1 - 0.36) = 0.8
      expect(result[1]).toBeCloseTo(0.8, 5);
      // Point 2: orbital mismatch → filtered
      expect(result[2]).toBe(0);
      // Point 3: time mismatch → filtered
      expect(result[3]).toBe(0);
    });

    it('should handle discrete dimension at non-zero slice position', () => {
      // Verify filtering works when slicing at orbital 2, not orbital 0
      const positions = new Float32Array([
        0,
        0,
        0,
        0, // Orbital 0
        1,
        0,
        0,
        0, // Orbital 1
        2,
        0,
        0,
        0, // Orbital 2 (target)
        3,
        0,
        0,
        0, // Orbital 3
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const viewState: ViewState = {
        displayDims: [1, 2, 3],
        slicePosition: [2, 0, 0, 0], // Looking at orbital 2
        tolerance: [0, 0.1, 0.1, 0.1],
      };
      const config: EffectiveRadiusConfig = {
        spatialExtendDims: [false, true, true, true],
        maxRadius: 1.0,
      };

      const result = calculateEffectiveRadii(positions, radii, viewState, config, 4);

      // Only orbital 2 should be visible
      expect(result[0]).toBe(0); // orbital 0
      expect(result[1]).toBe(0); // orbital 1
      expect(result[2]).toBe(1.0); // orbital 2 → visible
      expect(result[3]).toBe(0); // orbital 3
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
