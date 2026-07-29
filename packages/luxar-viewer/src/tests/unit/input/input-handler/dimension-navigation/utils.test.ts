/**
 * Tests for nD dimension navigation utility functions
 *
 * These utilities handle navigation through non-displayed dimensions
 * in multi-dimensional point clouds (4D, 5D, etc.)
 */

import { describe, it, expect } from 'vitest';
import {
  getNextDimensionIndex,
  getNonDisplayedDimensions,
  mapKeyToDimension,
} from '../../../../../input/input-handler/dimension-navigation/selection';
import {
  calculateStepSize,
  calculateNextPosition,
  NavigationConfig,
} from '../../../../../input/input-handler/dimension-navigation/step-math';
import {
  formatDimensionValue,
  generateNavigationHelp,
} from '../../../../../input/input-handler/dimension-navigation/format';
import { DimensionsBuilder } from '../../../../builders/test-data-builders';

describe('nD Navigation Utilities', () => {
  describe('getNextDimensionIndex', () => {
    it('should cycle through non-displayed dimensions', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(5)
        .withDisplayed(0, 1, 2) // x, y, z displayed
        .build();

      // Non-displayed are indices 3 and 4
      expect(getNextDimensionIndex(-1, 1, dims)).toBe(3); // Start at first
      expect(getNextDimensionIndex(3, 1, dims)).toBe(4); // Next from 3
      expect(getNextDimensionIndex(4, 1, dims)).toBe(3); // Wrap to start
      expect(getNextDimensionIndex(3, -1, dims)).toBe(4); // Previous wraps to end
    });

    it('should return -1 when all dimensions are displayed', () => {
      const dims = new DimensionsBuilder().withNDimensions(3).withDisplayed(0, 1, 2).build();

      expect(getNextDimensionIndex(0, 1, dims)).toBe(-1);
    });

    it('should handle starting from invalid index', () => {
      const dims = new DimensionsBuilder().withNDimensions(4).withDisplayed(0, 1).build();

      // Non-displayed are 2 and 3
      expect(getNextDimensionIndex(-1, 1, dims)).toBe(2); // Invalid to first
      expect(getNextDimensionIndex(-1, -1, dims)).toBe(3); // Invalid to last
    });
  });

  describe('getNonDisplayedDimensions', () => {
    it('should return non-displayed dimension indices', () => {
      const dims = new DimensionsBuilder().withNDimensions(5).withDisplayed(0, 2, 4).build();

      expect(getNonDisplayedDimensions(dims)).toEqual([1, 3]);
    });

    it('should return empty array when all displayed', () => {
      const dims = new DimensionsBuilder().withNDimensions(3).withDisplayed(0, 1, 2).build();

      expect(getNonDisplayedDimensions(dims)).toEqual([]);
    });

    it('should handle no displayed dimensions', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withDisplayed() // None displayed
        .build();

      expect(getNonDisplayedDimensions(dims)).toEqual([0, 1, 2, 3]);
    });
  });

  describe('calculateStepSize', () => {
    it('should use metadata step if available', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'px', [0, 100], { step: 5 })
        .build();

      expect(calculateStepSize(0, dims)).toBe(5);
    });

    it('should calculate step from range if no step defined', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'px', [0, 100])
        .build();

      // 1% of range (100 - 0)
      expect(calculateStepSize(0, dims)).toBe(1);
    });

    it('should apply fine control with shift', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'px', [0, 100], { step: 10 })
        .build();

      const step = calculateStepSize(0, dims, { shift: true });
      expect(step).toBe(1); // 10 / 10 (fineStepDivisor)
    });

    it('should apply coarse control with ctrl', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'px', [0, 100], { step: 1 })
        .build();

      const step = calculateStepSize(0, dims, { ctrl: true });
      expect(step).toBe(10); // 1 * 10 (coarseStepMultiplier)
    });

    it('should floor at one grid cell for discrete dimensions', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'frame', '', [0, 10], { step: 0.1, discrete: true })
        .build();

      const step = calculateStepSize(0, dims, { shift: true });
      expect(step).toBe(0.1); // One grid cell (meta.step) is the floor for discrete dims
    });

    it('floors a classic step-1 discrete dim at one whole cell under fine control', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'frame', '', [0, 10], { step: 1, discrete: true })
        .build();

      const step = calculateStepSize(0, dims, { shift: true });
      expect(step).toBe(1); // step-1 discrete: max(1, round(0.1)) = 1, unchanged from historical behavior
    });

    it('steps a fractional-step discrete dim by ten grid cells under coarse control', () => {
      const step = 2.0 / 50; // 0.04
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'w', '', [0, 2], { step, discrete: true })
        .build();

      const result = calculateStepSize(0, dims, { ctrl: true });
      expect(result).toBe(10 * step); // 0.04 ×10 coarse = 0.4, quantized to 10 cells on the 0.04 grid
    });

    it('should use custom navigation config', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'px', [0, 100], { step: 10 })
        .build();

      const config: NavigationConfig = {
        stepSizeMultiplier: 2.0,
        fineStepDivisor: 5,
        coarseStepMultiplier: 20,
        wrapAround: false,
      };

      expect(calculateStepSize(0, dims, {}, config)).toBe(20); // 10 * 2.0
      expect(calculateStepSize(0, dims, { shift: true }, config)).toBe(4); // (10 / 5) * 2.0
      expect(calculateStepSize(0, dims, { ctrl: true }, config)).toBe(400); // (10 * 20) * 2.0
    });
  });

  describe('calculateNextPosition', () => {
    it('should move forward by step size', () => {
      const next = calculateNextPosition(5, 1, 2, [0, 10]);
      expect(next).toBe(7);
    });

    it('should move backward by step size', () => {
      const next = calculateNextPosition(5, -1, 2, [0, 10]);
      expect(next).toBe(3);
    });

    it('should clamp to range boundaries', () => {
      expect(calculateNextPosition(9, 1, 2, [0, 10])).toBe(10); // Clamp to max
      expect(calculateNextPosition(1, -1, 2, [0, 10])).toBe(0); // Clamp to min
    });

    it('should round discrete values', () => {
      const next = calculateNextPosition(5.2, 1, 1.3, [0, 10], true);
      expect(next).toBe(7); // 5.2 + 1.3 = 6.5, rounded to 7
    });

    it('should wrap around when enabled', () => {
      expect(calculateNextPosition(9, 1, 2, [0, 10], false, true)).toBe(1); // Wrap to start
      expect(calculateNextPosition(1, -1, 2, [0, 10], false, true)).toBe(9); // Wrap to end
    });

    it('should handle zero-range dimension without NaN', () => {
      // Degenerate range where min === max: only one valid position
      expect(calculateNextPosition(5, 1, 1, [5, 5], false, true)).toBe(5); // Wrap-around
      expect(calculateNextPosition(5, -1, 1, [5, 5], false, false)).toBe(5); // Clamp
    });
  });

  describe('mapKeyToDimension', () => {
    it('should map number keys to navigable dimension positions', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(5)
        .withDisplayed(0, 1) // Only x, y displayed
        .build();

      // Non-displayed: [2, 3, 4]. Key 1→first navigable (dim 2), etc.
      expect(mapKeyToDimension('1', dims)).toBe(2); // First navigable
      expect(mapKeyToDimension('2', dims)).toBe(3); // Second navigable
      expect(mapKeyToDimension('3', dims)).toBe(4); // Third navigable
      expect(mapKeyToDimension('4', dims)).toBe(-1); // Only 3 navigable dims
      expect(mapKeyToDimension('5', dims)).toBe(-1);
    });

    it('should return -1 for invalid keys', () => {
      const dims = new DimensionsBuilder().withNDimensions(5).build();

      expect(mapKeyToDimension('0', dims)).toBe(-1);
      expect(mapKeyToDimension('a', dims)).toBe(-1);
      expect(mapKeyToDimension('10', dims)).toBe(-1);
    });

    it('should return -1 when no navigable dimensions exist', () => {
      const dims = new DimensionsBuilder().withNDimensions(3).withDisplayed(0, 1, 2).build();

      expect(mapKeyToDimension('1', dims)).toBe(-1); // All displayed
      expect(mapKeyToDimension('4', dims)).toBe(-1); // Beyond ndim
    });
  });

  describe('formatDimensionValue', () => {
    it('should format discrete values as integers', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'frame', '', [0, 100], { discrete: true })
        .build();

      expect(formatDimensionValue(5.7, 0, dims)).toBe('6');
    });

    it('should format continuous values with appropriate decimals', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', '', [0, 1], { step: 0.01 })
        .build();

      expect(formatDimensionValue(0.123, 0, dims)).toBe('0.12');
    });

    it('should add units when available', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(2)
        .withDimension(0, 'x', 'μm', [0, 100])
        .build();

      expect(formatDimensionValue(50.5, 0, dims)).toBe('50.50μm');
    });

    it('should use default formatting when no metadata', () => {
      const dims = new DimensionsBuilder().withNDimensions(2).build();

      expect(formatDimensionValue(3.14159, 0, dims)).toBe('3.14');
    });
  });

  describe('generateNavigationHelp', () => {
    it('should show current selection and available dimensions', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(5)
        .withDisplayed(0, 1, 2)
        .withDimension(3, 'time', 's', [0, 10])
        .withDimension(4, 'channel', '', [0, 3])
        .build();

      dims.currentStep[3] = 5;
      dims.currentStep[4] = 1;

      const help = generateNavigationHelp(3, dims);

      expect(help).toContain('Selected: time = 5.00s');
      expect(help).toContain('Non-displayed dimensions:');
      expect(help.some((line) => line.includes('[1] time'))).toBe(true);
      expect(help.some((line) => line.includes('[2] channel'))).toBe(true);
    });

    it('should indicate when all dimensions are displayed', () => {
      const dims = new DimensionsBuilder().withNDimensions(3).withDisplayed(0, 1, 2).build();

      const help = generateNavigationHelp(-1, dims);

      expect(help).toContain('All dimensions are displayed (3D view)');
    });

    it('should show navigation instructions', () => {
      const dims = new DimensionsBuilder().withNDimensions(4).withDisplayed(0, 1, 2).build();

      const help = generateNavigationHelp(-1, dims);

      expect(help).toContain('Navigation:');
      expect(help.some((line) => line.includes('[1-9] Select dimension'))).toBe(true);
      expect(help.some((line) => line.includes('[ ]') && line.includes('Navigate'))).toBe(true);
    });
  });
});
