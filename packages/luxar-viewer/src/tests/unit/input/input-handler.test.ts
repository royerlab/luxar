/**
 * Unit tests for InputHandler utilities and helper functions
 *
 * The InputHandler class has many complex dependencies (THREE.js, DOM, multiple managers).
 * Full integration testing is done via E2E tests. These unit tests focus on:
 * - Pure utility functions that can be tested in isolation
 * - Basic construction and type verification
 *
 * For full keyboard interaction testing, see:
 * - src/tests/e2e/controls-interaction.spec.ts
 */

import { describe, it, expect } from 'vitest';
import type { SimpleDims } from '../../../types/dims';
import {
  getNonDisplayedDimensions,
  calculateStepSize,
  calculateNextPosition,
  mapKeyToDimension,
  isNavigationKey,
  calculateFovChange,
  shouldBlockShortcut,
  getNextDimensionIndex,
  formatDimensionValue,
} from '../../../input/input-handler-utils';

// Helper to create SimpleDims test objects
function createDims(ndim: number, displayed: number[], metadata?: any[]): SimpleDims {
  return {
    ndim,
    displayed,
    currentStep: new Array(ndim).fill(0),
    metadata: metadata || new Array(ndim).fill(null).map((_, i) => ({ name: `dim${i}` })),
  };
}

describe('InputHandler Utilities', () => {
  describe('getNonDisplayedDimensions', () => {
    it('should return empty array for 3D datasets (all displayed)', () => {
      const dims = createDims(3, [0, 1, 2]);
      const result = getNonDisplayedDimensions(dims);
      expect(result).toEqual([]);
    });

    it('should return non-displayed dimensions for 4D dataset', () => {
      const dims = createDims(4, [0, 1, 2]);
      const result = getNonDisplayedDimensions(dims);
      expect(result).toEqual([3]);
    });

    it('should return non-displayed dimensions for 5D dataset', () => {
      const dims = createDims(5, [0, 1, 2]);
      const result = getNonDisplayedDimensions(dims);
      expect(result).toEqual([3, 4]);
    });

    it('should handle custom display dims', () => {
      const dims = createDims(5, [1, 2, 3]);
      const result = getNonDisplayedDimensions(dims);
      expect(result).toEqual([0, 4]);
    });

    it('should handle no displayed dims', () => {
      const dims = createDims(3, []);
      const result = getNonDisplayedDimensions(dims);
      expect(result).toEqual([0, 1, 2]);
    });
  });

  describe('calculateStepSize', () => {
    it('should use step from metadata', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 0.5 }]
      );
      const result = calculateStepSize(3, dims);
      expect(result).toBe(0.5);
    });

    it('should calculate 1% of range when no step provided', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', range: [0, 100] }]
      );
      const result = calculateStepSize(3, dims);
      expect(result).toBe(1); // 1% of 100
    });

    it('should apply shift modifier (fine control)', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 10 }]
      );
      const result = calculateStepSize(3, dims, { shift: true });
      expect(result).toBe(1); // 10 / 10
    });

    it('should apply ctrl modifier (coarse control)', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 1 }]
      );
      const result = calculateStepSize(3, dims, { ctrl: true });
      expect(result).toBe(10); // 1 * 10
    });

    it('should return at least 1 for discrete dimensions', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'frame', step: 0.1, discrete: true }]
      );
      const result = calculateStepSize(3, dims, { shift: true });
      expect(result).toBe(1); // Minimum 1 for discrete
    });

    it('should default to 1.0 when no metadata', () => {
      const dims = createDims(4, [0, 1, 2]);
      dims.metadata = undefined;
      const result = calculateStepSize(3, dims);
      expect(result).toBe(1);
    });
  });

  describe('calculateNextPosition', () => {
    it('should move forward by step size', () => {
      const result = calculateNextPosition(5, 1, 1, [0, 10]);
      expect(result).toBe(6);
    });

    it('should move backward by step size', () => {
      const result = calculateNextPosition(5, -1, 1, [0, 10]);
      expect(result).toBe(4);
    });

    it('should clamp to maximum', () => {
      const result = calculateNextPosition(9, 1, 2, [0, 10]);
      expect(result).toBe(10);
    });

    it('should clamp to minimum', () => {
      const result = calculateNextPosition(1, -1, 2, [0, 10]);
      expect(result).toBe(0);
    });

    it('should round for discrete dimensions', () => {
      const result = calculateNextPosition(5.3, 1, 1.7, [0, 10], true);
      expect(result).toBe(7); // 5.3 + 1.7 = 7 (rounded)
    });

    it('should wrap around when enabled', () => {
      const result = calculateNextPosition(9, 1, 3, [0, 10], false, true);
      // 9 + 3 = 12, wraps to 0 + (12 - 10) % 10 = 2
      expect(result).toBe(2);
    });
  });

  describe('mapKeyToDimension', () => {
    it('should map "1" to first non-displayed dimension', () => {
      const dims = createDims(5, [1, 2, 3]); // Non-displayed: 0, 4
      expect(mapKeyToDimension('1', dims)).toBe(0); // First navigable = dim 0
    });

    it('should map "1" to first navigable dim even when dim 0 is displayed', () => {
      const dims = createDims(5, [0, 1, 2]); // Non-displayed: 3, 4
      expect(mapKeyToDimension('1', dims)).toBe(3); // First navigable = dim 3
      expect(mapKeyToDimension('2', dims)).toBe(4); // Second navigable = dim 4
    });

    it('should return -1 when key exceeds navigable count', () => {
      const dims = createDims(10, [0, 1, 2]); // Non-displayed: 3,4,5,6,7,8,9 (7 navigable)
      expect(mapKeyToDimension('7', dims)).toBe(9); // 7th navigable = dim 9
      expect(mapKeyToDimension('8', dims)).toBe(-1); // Only 7 navigable dims
    });

    it('should return -1 for "0"', () => {
      const dims = createDims(5, [0, 1, 2]);
      expect(mapKeyToDimension('0', dims)).toBe(-1);
    });

    it('should return -1 for non-numeric keys', () => {
      const dims = createDims(5, [0, 1, 2]);
      expect(mapKeyToDimension('a', dims)).toBe(-1);
      expect(mapKeyToDimension('[', dims)).toBe(-1);
      expect(mapKeyToDimension(' ', dims)).toBe(-1);
    });

    it('should return -1 when all dimensions are displayed', () => {
      const dims = createDims(3, [0, 1, 2]);
      expect(mapKeyToDimension('1', dims)).toBe(-1); // No navigable dims
      expect(mapKeyToDimension('5', dims)).toBe(-1);
    });

    it('should return -1 for empty string', () => {
      const dims = createDims(5, [0, 1, 2]);
      expect(mapKeyToDimension('', dims)).toBe(-1);
    });
  });

  describe('getNextDimensionIndex', () => {
    it('should cycle to next non-displayed dimension', () => {
      const dims = createDims(5, [0, 1, 2]); // Non-displayed: 3, 4
      const result = getNextDimensionIndex(3, 1, dims);
      expect(result).toBe(4);
    });

    it('should wrap around at end', () => {
      const dims = createDims(5, [0, 1, 2]); // Non-displayed: 3, 4
      const result = getNextDimensionIndex(4, 1, dims);
      expect(result).toBe(3); // Wraps to first
    });

    it('should cycle backward', () => {
      const dims = createDims(5, [0, 1, 2]); // Non-displayed: 3, 4
      const result = getNextDimensionIndex(4, -1, dims);
      expect(result).toBe(3);
    });

    it('should return -1 when no non-displayed dimensions', () => {
      const dims = createDims(3, [0, 1, 2]);
      const result = getNextDimensionIndex(0, 1, dims);
      expect(result).toBe(-1);
    });
  });

  // NOTE: isNavigationKey and calculateFovChange are also tested in
  // src/tests/unit/controls/input-validation.test.ts with additional edge cases
  // (e.g., textarea blocking, custom sensitivity). Both suites test the same
  // pure functions from input-handler-utils.ts — keep them in sync.
  describe('isNavigationKey', () => {
    it('should return true for [ key', () => {
      const event = new KeyboardEvent('keydown', { key: '[' });
      expect(isNavigationKey(event)).toBe(true);
    });

    it('should return true for ] key', () => {
      const event = new KeyboardEvent('keydown', { key: ']' });
      expect(isNavigationKey(event)).toBe(true);
    });

    it('should return true for number keys 1-9', () => {
      for (let i = 1; i <= 9; i++) {
        const event = new KeyboardEvent('keydown', { key: String(i) });
        expect(isNavigationKey(event)).toBe(true);
      }
    });

    it('should return false for other keys', () => {
      const event = new KeyboardEvent('keydown', { key: 'a' });
      expect(isNavigationKey(event)).toBe(false);
    });

    it('should return false when target is input element', () => {
      const input = document.createElement('input');
      const event = new KeyboardEvent('keydown', { key: '[' });
      Object.defineProperty(event, 'target', { value: input });
      expect(isNavigationKey(event)).toBe(false);
    });
  });

  describe('calculateFovChange', () => {
    it('should increase FOV with positive delta', () => {
      const result = calculateFovChange(60, 10, 0.1);
      expect(result).toBe(61);
    });

    it('should decrease FOV with negative delta', () => {
      const result = calculateFovChange(60, -10, 0.1);
      expect(result).toBe(59);
    });

    it('should clamp to minimum FOV (10)', () => {
      const result = calculateFovChange(15, -100, 0.1);
      expect(result).toBe(10);
    });

    it('should clamp to maximum FOV (170)', () => {
      const result = calculateFovChange(160, 200, 0.1);
      expect(result).toBe(170); // config.camera.fovMax = 170
    });
  });

  describe('shouldBlockShortcut', () => {
    it('should block when modal is active', () => {
      const event = new KeyboardEvent('keydown', { key: 'p' });
      expect(shouldBlockShortcut(event, ['settings-modal'])).toBe(true);
    });

    it('should block when typing in input', () => {
      const input = document.createElement('input');
      const event = new KeyboardEvent('keydown', { key: 'p' });
      Object.defineProperty(event, 'target', { value: input });
      expect(shouldBlockShortcut(event, [])).toBe(true);
    });

    it('should not block in normal context', () => {
      const event = new KeyboardEvent('keydown', { key: 'p' });
      expect(shouldBlockShortcut(event, [])).toBe(false);
    });
  });

  describe('formatDimensionValue', () => {
    it('should format discrete values as integers', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'frame', discrete: true }]
      );
      dims.currentStep[3] = 42.7;
      const result = formatDimensionValue(42.7, 3, dims);
      expect(result).toBe('43');
    });

    it('should add unit suffix when present', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', unit: 's' }]
      );
      const result = formatDimensionValue(5.0, 3, dims);
      expect(result).toBe('5.00s');
    });

    it('should use adaptive precision based on step', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 0.01 }]
      );
      const result = formatDimensionValue(5.234, 3, dims);
      expect(result).toBe('5.23');
    });
  });
});

describe('InputHandler Type Definitions', () => {
  it(
    'should export InputHandler class',
    async () => {
      // Dynamic import to avoid triggering complex dependencies.
      // The import pulls in the full dependency graph (THREE.js, scene managers,
      // UI components), which normally takes ~600ms but can exceed 5s under load.
      const module = await import('../../../input/input-handler');
      expect(module.InputHandler).toBeDefined();
      expect(typeof module.InputHandler).toBe('function');
    },
    { timeout: 15_000 }
  );
});
