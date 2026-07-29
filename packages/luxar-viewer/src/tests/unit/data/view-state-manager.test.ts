/**
 * Tests for ViewStateManager - Critical ViewState initialization and validation logic
 */

import { describe, it, expect } from 'vitest';
import {
  ViewStateManager,
  type SceneDimensions,
  type DimensionMetadata,
} from '../../../data/view-state-manager';

describe('ViewStateManager', () => {
  describe('initializeFromDimensions', () => {
    it('should initialize ViewState for 3D dataset', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'um', scale: 1.0, range: [-10, 10], display: true },
          { name: 'y', unit: 'um', scale: 1.0, range: [-10, 10], display: true },
          { name: 'z', unit: 'um', scale: 1.0, range: [-10, 10], display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.displayDims).toEqual([0, 1, 2]);
      expect(viewState.slicePosition).toEqual([0, 0, 0]); // Displayed dims start at 0
      expect(viewState.tolerance).toEqual([0, 0, 0]); // Displayed dims have 0 tolerance
      expect(viewState.dimensions?.length).toBe(3);
    });

    it('should initialize ViewState for 5D dataset with 3 displayed', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'time', unit: 's', scale: 0.1, range: [0, 100], display: false, discrete: true },
          { name: 'channel', unit: '', scale: 1.0, range: [0, 4], display: false, discrete: true },
          { name: 'x', unit: 'um', scale: 1.0, range: [-50, 50], display: true },
          { name: 'y', unit: 'um', scale: 1.0, range: [-50, 50], display: true },
          { name: 'z', unit: 'um', scale: 1.0, range: [-20, 20], display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.displayDims).toEqual([2, 3, 4]); // x, y, z
      expect(viewState.slicePosition[0]).toBe(0); // time: starts at minimum of [0, 100]
      expect(viewState.slicePosition[1]).toBe(0); // channel: starts at minimum of [0, 4]
      expect(viewState.slicePosition[2]).toBe(0); // x: displayed, starts at 0
      expect(viewState.tolerance[0]).toBe(0); // time: discrete, exact match
      expect(viewState.tolerance[1]).toBe(0); // channel: discrete, exact match
      expect(viewState.tolerance[2]).toBe(0); // x: displayed, no tolerance
    });

    it('should use floor() for discrete dimension initial position', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'frames', unit: '', scale: 1.0, range: [0, 9], display: false, discrete: true }, // Starts at minimum = 0
          { name: 'x', unit: 'px', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition[0]).toBe(0); // Discrete dims start at range minimum
    });

    it('should handle continuous non-displayed dimensions', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'wavelength', unit: 'nm', scale: 1.0, range: [400, 700], display: false }, // Continuous
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition[0]).toBe(550); // Continuous: starts at center of [400, 700]
      expect(viewState.tolerance[0]).toBeGreaterThan(0); // Should have default tolerance
    });

    it('should handle dimensions without ranges', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'um', scale: 1.0, display: true }, // No range
          { name: 'y', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition).toEqual([0, 0]); // Default to 0
    });

    it('should limit displayed dimensions to 3', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'w', unit: 'um', scale: 1.0, display: true },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
          { name: 'y', unit: 'um', scale: 1.0, display: true },
          { name: 'z', unit: 'um', scale: 1.0, display: true }, // 4th one, should be ignored
          { name: 't', unit: 's', scale: 1.0, display: true }, // 5th one, should be ignored
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.displayDims).toHaveLength(3); // Only first 3
      expect(viewState.displayDims).toEqual([0, 1, 2]);
    });
  });

  describe('validateDimensions', () => {
    it('should validate correct dimensions', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1.0, range: [-10, 10], display: true, step: 0.1 },
        { name: 'y', unit: 'um', scale: 1.0, range: [-10, 10], display: true, step: 0.1 },
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should warn about duplicate dimension names', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1.0, display: true },
        { name: 'x', unit: 'nm', scale: 1.0, display: false }, // Duplicate!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.isValid).toBe(true); // Warnings don't make it invalid
      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.warnings[0]).toContain('Duplicate dimension names');
    });

    it('should warn about too many displayed dimensions', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'w', unit: 'um', scale: 1.0, display: true },
        { name: 'x', unit: 'um', scale: 1.0, display: true },
        { name: 'y', unit: 'um', scale: 1.0, display: true },
        { name: 'z', unit: 'um', scale: 1.0, display: true }, // 4th displayed
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('4 displayed dimensions'))).toBe(true);
    });

    it('should warn about no displayed dimensions', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1.0, display: false },
        { name: 'y', unit: 'um', scale: 1.0, display: false },
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('no displayed dimensions'))).toBe(true);
    });

    it('should warn about invalid range format', () => {
      const dimensions: any[] = [
        { name: 'x', unit: 'um', scale: 1.0, range: [10, 5], display: true }, // Min > Max!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('invalid range'))).toBe(true);
    });

    it('should NOT warn about zero-width range for a single-category dimension', () => {
      // Regression (#755): a single category auto-ranges to [0, 0], which is legal.
      const dimensions: any[] = [
        {
          name: 'channel',
          unit: '',
          scale: 1.0,
          categories: ['DAPI'],
          range: [0, 0],
          discrete: true,
        },
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('range') || w.includes('Min should'))).toBe(
        false
      );
    });

    it('should NOT warn about zero-width range for a single-index discrete dimension', () => {
      // Regression (#878): a single timepoint/channel auto-ranges to [0, 0], legal.
      const dimensions: any[] = [
        {
          name: 't',
          unit: 's',
          scale: 1.0,
          range: [0, 0],
          discrete: true,
          display: false,
        },
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('range') || w.includes('Min should'))).toBe(
        false
      );
    });

    it('should still warn about zero-width range for a non-categorical dimension', () => {
      const dimensions: any[] = [
        { name: 'x', unit: 'um', scale: 1.0, range: [5, 5], display: true }, // Min == Max!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('invalid range'))).toBe(true);
    });

    it('should still warn about zero-width range for a continuous dimension', () => {
      // The #878 discrete relaxation must not leak to continuous (non-discrete) dims.
      const dimensions: any[] = [
        { name: 'x', unit: 'um', scale: 1.0, range: [0, 0], discrete: false, display: true },
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('invalid range'))).toBe(true);
    });

    it('should warn about an inverted categorical range', () => {
      const dimensions: any[] = [
        { name: 'c', unit: '', scale: 1.0, categories: ['a', 'b'], range: [2, 0] }, // Inverted!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('inverted range'))).toBe(true);
    });

    it('should warn about negative or zero step', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1.0, step: 0, display: true }, // Zero step!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('invalid step'))).toBe(true);
    });

    it('should warn about discrete dimensions without range', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'channel', unit: '', scale: 1.0, discrete: true, display: false }, // No range!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('should have a defined range'))).toBe(true);
    });

    it('should warn about missing dimension names', () => {
      const dimensions: any[] = [
        { unit: 'um', scale: 1.0, display: true }, // No name!
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.some((w) => w.includes('missing name'))).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single dimension', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [{ name: 'x', unit: 'um', scale: 1.0, display: true }],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.displayDims).toEqual([0]);
      expect(viewState.dimensions?.length).toBe(1);
    });

    it('should handle dimensions with very large ranges', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'time', unit: 's', scale: 1.0, range: [0, 1000000], display: false },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition[0]).toBe(500000); // Continuous: starts at center
    });

    it('should handle negative ranges correctly', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'offset', unit: 'um', scale: 1.0, range: [-100, -50], display: false },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition[0]).toBe(-75); // Continuous: starts at center of [-100, -50]
    });

    it('should handle fractional discrete dimension ranges', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          // Weird but valid: discrete with non-integer range
          {
            name: 'weird',
            unit: '',
            scale: 1.0,
            range: [0.5, 5.5],
            display: false,
            discrete: true,
          },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.slicePosition[0]).toBe(0); // floor(0.5) = 0 (starts at minimum, floored)
    });
  });

  describe('Integration with Existing Code', () => {
    it('should return DimensionMetadata compatible with SimpleDims', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'um', scale: 1.0, display: true },
          { name: 'y', unit: 'um', scale: 1.0, display: true },
          { name: 'z', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      // Check that metadata matches expected structure
      expect(viewState.dimensions).toBeDefined();
      expect(viewState.dimensions).toHaveLength(3);
      expect(viewState.dimensions?.[0].name).toBe('x');
      expect(viewState.dimensions?.[0].unit).toBe('um');
      expect(viewState.dimensions?.[0].scale).toBe(1.0);
    });

    it('should handle all optional DimensionMetadata fields', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          {
            name: 'time',
            unit: 's',
            scale: 0.01,
            range: [0, 1000],
            display: false,
            discrete: true,
            step: 10,
          },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      const timeMeta = viewState.dimensions?.[0];
      expect(timeMeta?.discrete).toBe(true);
      expect(timeMeta?.step).toBe(10);
      expect(timeMeta?.range).toEqual([0, 1000]);
    });

    it('should preserve spatial, cyclic, categories, and description fields', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          {
            name: 'channel',
            unit: '',
            scale: 1.0,
            range: [0, 3],
            display: false,
            discrete: true,
            spatial: false,
            cyclic: false,
            categories: ['DAPI', 'GFP', 'RFP', 'Merge'],
            description: 'Fluorescence channel',
          },
          {
            name: 'angle',
            unit: 'rad',
            scale: 1.0,
            range: [0, 6.28],
            display: false,
            cyclic: true,
            spatial: true,
            description: 'Rotation angle',
          },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      const channelMeta = viewState.dimensions?.[0];
      expect(channelMeta?.spatial).toBe(false);
      expect(channelMeta?.cyclic).toBe(false);
      expect(channelMeta?.categories).toEqual(['DAPI', 'GFP', 'RFP', 'Merge']);
      expect(channelMeta?.description).toBe('Fluorescence channel');

      const angleMeta = viewState.dimensions?.[1];
      expect(angleMeta?.spatial).toBe(true);
      expect(angleMeta?.cyclic).toBe(true);
      expect(angleMeta?.description).toBe('Rotation angle');
    });
  });

  describe('Validation Logic', () => {
    it('should accumulate multiple warnings', () => {
      const dimensions: any[] = [
        { name: 'x', unit: 'um', scale: 1.0, range: [10, 5], display: true, step: -1 }, // 2 warnings
        { unit: 'nm', scale: 1.0, display: false }, // 1 warning (no name)
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.warnings.length).toBeGreaterThanOrEqual(3);
    });

    it('should not have errors for valid edge cases', () => {
      const dimensions: DimensionMetadata[] = [
        { name: 'x', unit: '', scale: 1.0 }, // Empty unit - valid
        { name: 'y', unit: 'um', scale: 0.001 }, // Very small scale - valid
        { name: 'z', unit: 'km', scale: 1000.0 }, // Large scale - valid
      ];

      const result = ViewStateManager.validateDimensions(dimensions);

      expect(result.errors).toHaveLength(0);
      expect(result.isValid).toBe(true);
    });
  });

  describe('Tolerance Calculation', () => {
    it('should set 0 tolerance for displayed dimensions', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'um', scale: 1.0, display: true },
          { name: 'y', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.tolerance).toEqual([0, 0]);
    });

    it('should set 0 tolerance for discrete dimensions', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'frame', unit: '', scale: 1.0, display: false, discrete: true },
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.tolerance[0]).toBe(0); // Discrete, exact match
      expect(viewState.tolerance[1]).toBe(0); // Displayed
    });

    it('should use default tolerance for continuous non-displayed dimensions', () => {
      const sceneDims: SceneDimensions = {
        dimensions: [
          { name: 'wavelength', unit: 'nm', scale: 1.0, display: false }, // Continuous, non-displayed
          { name: 'x', unit: 'um', scale: 1.0, display: true },
        ],
      };

      const viewState = ViewStateManager.initializeFromDimensions(sceneDims);

      expect(viewState.tolerance[0]).toBeGreaterThan(0); // Should have default tolerance
      expect(viewState.tolerance[1]).toBe(0); // Displayed
    });
  });
});
