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
import fc from 'fast-check';
import type { SimpleDims } from '../../../types/dims';
import {
  getNonDisplayedDimensions,
  mapKeyToDimension,
  getNextDimensionIndex,
} from '../../../input/input-handler/dimension-navigation/selection';
import {
  calculateStepSize,
  calculateNextPosition,
} from '../../../input/input-handler/dimension-navigation/step-math';
import {
  formatDimensionValue,
  generateNavigationHelp,
} from '../../../input/input-handler/dimension-navigation/format';
import {
  isNavigationKey,
  shouldBlockShortcut,
} from '../../../input/input-handler/keyboard-validation';
import { calculateFovChange } from '../../../input/input-handler/window-events/fov-wheel-math';

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

    it('should floor at one grid cell for discrete dimensions', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'frame', step: 0.1, discrete: true }]
      );
      const result = calculateStepSize(3, dims, { shift: true });
      expect(result).toBe(0.1); // One grid cell (meta.step) is the floor for discrete dims
    });

    it('floors a classic step-1 discrete dim at one whole cell under fine control', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'frame', step: 1, discrete: true }]
      );
      const result = calculateStepSize(3, dims, { shift: true });
      expect(result).toBe(1); // step-1 discrete: max(1, round(0.1)) = 1, unchanged from historical behavior
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

    describe('negative-wrap formula [input.md G1]', () => {
      // input.md G1[P5]: the negative-wrap branch (step-math.ts L122) uses
      // `range[1] - ((range[0] - newPos) % rangeSize)` which is non-obvious.
      // Prior tests only covered the positive-wrap direction; a modular-
      // arithmetic mutant (`%` → `-`, or swapped operands) on the negative
      // branch would survive without these tests.
      it('[G1] step=15 from currentPos=0 with direction=-1 in [0,100] wraps to 85', () => {
        // newPos = 0 + (-1) * 15 = -15. range[0] - newPos = 0 - (-15) = 15.
        // 15 % 100 = 15. range[1] - 15 = 100 - 15 = 85.
        const result = calculateNextPosition(0, -1, 15, [0, 100], false, true);
        expect(result).toBe(85);
      });

      it('[G1] negative wrap past multiple range cycles: step=205 from currentPos=0 in [0,100] → 95', () => {
        // newPos = -205. range[0] - newPos = 205. 205 % 100 = 5.
        // range[1] - 5 = 95.
        const result = calculateNextPosition(0, -1, 205, [0, 100], false, true);
        expect(result).toBe(95);
      });

      it('[G1] negative wrap exactly one cycle: step=100 from currentPos=0 in [0,100] → 100', () => {
        // newPos = -100. range[0] - newPos = 100. 100 % 100 = 0.
        // range[1] - 0 = 100.
        const result = calculateNextPosition(0, -1, 100, [0, 100], false, true);
        expect(result).toBe(100);
      });

      it('[G1] negative range start: step=5 backward from -1 in [-10, 0] → -6 (no wrap, in range)', () => {
        const result = calculateNextPosition(-1, -1, 5, [-10, 0], false, true);
        expect(result).toBe(-6);
      });

      it('[G1] negative range start with wrap: step=15 backward from -1 in [-10, 0] → -6 (wraps)', () => {
        // newPos = -16, range = [-10, 0], rangeSize=10.
        // range[0] - newPos = -10 - (-16) = 6. 6 % 10 = 6. range[1] - 6 = -6.
        const result = calculateNextPosition(-1, -1, 15, [-10, 0], false, true);
        expect(result).toBe(-6);
      });
    });

    describe('discrete + wrapAround combination [input.md G2]', () => {
      // input.md G2[P5]: Math.round applies BEFORE clamp/wrap. The interaction
      // of `discrete && wrapAround` was untested. Cyclic frame indices that
      // round-then-wrap is a real use case (animated time dimensions).
      it('[G2] discrete + wrap: fractional input rounds first, then wraps', () => {
        // newPos = 9 + 3.4 = 12.4 → round → 12 → wrap.
        // `[0, 10]` discrete with step 1 is ELEVEN positions (0..10), so the
        // period is 11, not 10: 0 + (12 % 11) = 1.
        const result = calculateNextPosition(9, 1, 3.4, [0, 10], true, true);
        expect(result).toBe(1);
      });

      it('[G2] discrete + wrap backward: rounds then wraps correctly', () => {
        // newPos = 0 + (-1) * 2.4 = -2.4 → round → -2 → wrap.
        // Period 11 (eleven inclusive positions): 0 + mod(-2, 11) = 9.
        const result = calculateNextPosition(0, -1, 2.4, [0, 10], true, true);
        expect(result).toBe(9);
      });

      it('[G2] discrete + wrap: rounding pushes value JUST past the range, wraps to start', () => {
        // newPos = 9.7 + 0.5 = 10.2 → round → 10 (still in range, no wrap).
        // Test that the rounding happens before clamp.
        const result = calculateNextPosition(9.7, 1, 0.5, [0, 10], true, true);
        expect(result).toBe(10);
      });
    });

    describe('inclusive discrete period: every category stays reachable', () => {
      // A discrete `[min, max]` is an INCLUSIVE set of positions, so its cyclic
      // period is `(max - min) + step` — NOT the continuous `max - min`. Using
      // the continuous period skipped one position at each wrap, which made the
      // first and last CATEGORIES unreachable by keyboard. `[0, 7]` with step 1
      // is exactly what `Dimension.__post_init__` derives for an 8-category
      // dim, so this is the shape every categorical dataset ships.
      const CATEGORICAL: [number, number] = [0, 7];

      it('wraps backward from the first category to the LAST, not last-minus-one', () => {
        expect(calculateNextPosition(0, -1, 1, CATEGORICAL, true, true, 1)).toBe(7);
      });

      it('wraps forward from the last category to the FIRST, not first-plus-one', () => {
        expect(calculateNextPosition(7, 1, 1, CATEGORICAL, true, true, 1)).toBe(0);
      });

      it('a full cycle of forward steps visits all 8 categories exactly once', () => {
        const seen: number[] = [];
        let pos = 0;
        for (let i = 0; i < 8; i++) {
          seen.push(pos);
          pos = calculateNextPosition(pos, 1, 1, CATEGORICAL, true, true, 1);
        }
        expect(seen).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        expect(pos).toBe(0); // and closes the loop
      });

      it('overshoot of several cycles still lands on a valid position', () => {
        // 8 positions, period 8: +19 from 0 is 19 % 8 = 3.
        expect(calculateNextPosition(0, 1, 19, CATEGORICAL, true, true, 1)).toBe(3);
        // -19 backward: mod(-19, 8) = 5.
        expect(calculateNextPosition(0, -1, 19, CATEGORICAL, true, true, 1)).toBe(5);
      });

      it('honours a fractional declared step', () => {
        // range [0, 1] with step 0.25 = five positions; period 1.25.
        expect(calculateNextPosition(0, -1, 0.25, [0, 1], true, true, 0.25)).toBeCloseTo(1, 10);
        expect(calculateNextPosition(1, 1, 0.25, [0, 1], true, true, 0.25)).toBeCloseTo(0, 10);
      });

      it('a single-position discrete dim stays put instead of producing NaN', () => {
        expect(calculateNextPosition(5, 1, 1, [5, 5], true, true, 1)).toBe(5);
        expect(calculateNextPosition(5, -1, 1, [5, 5], true, true, 1)).toBe(5);
      });

      it('CONTINUOUS cyclic dims keep the max-min period (max ≡ min)', () => {
        // Regression guard: an angle in [0, 360] must still wrap with period
        // 360, not 361 — the discrete +step must not leak into this path.
        expect(calculateNextPosition(0, -1, 5, [0, 360], false, true)).toBe(355);
        expect(calculateNextPosition(360, 1, 5, [0, 360], false, true)).toBe(5);
      });

      describe('range width that is NOT a whole number of steps', () => {
        // `[0, 1]` with step 0.3 has its last on-grid position at 0.9, so the
        // period is 4 * 0.3 = 1.2, not (1 - 0) + 0.3 = 1.3. Deriving the period
        // from `(max - min) + step` left an overshoot of 1.2 unwrapped and
        // returned a position ABOVE max, breaking the documented "guaranteed to
        // be within range".
        it('never returns a position outside the declared range', () => {
          for (const [from, dir] of [
            [0.9, 1],
            [0, -1],
            [0.6, 1],
            [0.3, -1],
          ] as Array<[number, 1 | -1]>) {
            const r = calculateNextPosition(from, dir, 0.3, [0, 1], true, true, 0.3);
            expect(r).toBeGreaterThanOrEqual(0);
            expect(r).toBeLessThanOrEqual(1);
          }
        });

        it('wraps forward off the last on-grid position back to the first', () => {
          expect(calculateNextPosition(0.9, 1, 0.3, [0, 1], true, true, 0.3)).toBeCloseTo(0, 10);
        });

        it('wraps backward off the first position to the last ON-GRID one (0.9, not 1)', () => {
          expect(calculateNextPosition(0, -1, 0.3, [0, 1], true, true, 0.3)).toBeCloseTo(0.9, 10);
        });

        it('float error in count derivation does not drop a position', () => {
          // 0.9 / 0.3 === 2.9999999999999996; without the epsilon this range
          // would be treated as 3 positions and 0.9 would become unreachable.
          expect(calculateNextPosition(0.6, 1, 0.3, [0, 0.9], true, true, 0.3)).toBeCloseTo(
            0.9,
            10
          );
          expect(calculateNextPosition(0.9, 1, 0.3, [0, 0.9], true, true, 0.3)).toBeCloseTo(0, 10);
        });
      });
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

    // input.md [H4][P12] fast-check property test: cycling forward then
    // backward (from any currently-non-displayed start dim) returns to
    // the starting dim. The function operates on the non-displayed
    // dimension subset; forward(currentDim) then backward(result) must
    // be a round-trip identity, exercising the wrap-around at both ends.
    it('[property] forward then backward returns to start for any non-displayed dim', () => {
      fc.assert(
        fc.property(
          // ndim in [4, 12]; displayed = the first 3 dims (so non-displayed
          // are [3, ndim)). We pick a startDim from the non-displayed range.
          fc
            .integer({ min: 4, max: 12 })
            .chain((ndim) => fc.tuple(fc.constant(ndim), fc.integer({ min: 3, max: ndim - 1 }))),
          ([ndim, startDim]) => {
            const dims = createDims(ndim, [0, 1, 2]);
            const next = getNextDimensionIndex(startDim, 1, dims);
            const back = getNextDimensionIndex(next, -1, dims);
            return back === startDim;
          }
        ),
        { numRuns: 200 }
      );
    });

    // input.md [H4][P12] follow-up: backward then forward is also identity.
    // Together with the prior property this pins the full inverse-relation
    // contract — any sign-flip mutation in the wrap arithmetic would surface.
    it('[property] backward then forward returns to start for any non-displayed dim', () => {
      fc.assert(
        fc.property(
          fc
            .integer({ min: 4, max: 12 })
            .chain((ndim) => fc.tuple(fc.constant(ndim), fc.integer({ min: 3, max: ndim - 1 }))),
          ([ndim, startDim]) => {
            const dims = createDims(ndim, [0, 1, 2]);
            const prev = getNextDimensionIndex(startDim, -1, dims);
            const fwd = getNextDimensionIndex(prev, 1, dims);
            return fwd === startDim;
          }
        ),
        { numRuns: 200 }
      );
    });
  });

  // NOTE: isNavigationKey and calculateFovChange are also tested in
  // src/tests/unit/controls/input-validation.test.ts with additional edge cases
  // (e.g., textarea blocking, custom sensitivity). Both suites exercise the same
  // pure functions (keyboard-validation.ts + window-events/fov-wheel-math.ts) —
  // keep them in sync.
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

    // input.md C3 fix: previous tests skipped the clamp boundary itself.
    // Adding the in-range/out-of-range boundary pair so any change to the
    // clamp predicate (e.g. `>` ↔ `>=`) is caught.
    it('returns currentFov when already at the clamp upper boundary (170) with positive delta', () => {
      expect(calculateFovChange(170, 50, 0.1)).toBe(170);
    });

    it('returns currentFov when already at the clamp lower boundary (10) with negative delta', () => {
      expect(calculateFovChange(10, -50, 0.1)).toBe(10);
    });

    it('returns 169 when one step below the max (boundary pin for the upper clamp)', () => {
      // currentFov + delta*sensitivity = 168 + 10*0.1 = 169 (in range, no clamp).
      expect(calculateFovChange(168, 10, 0.1)).toBe(169);
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

    it('should block when typing in textarea', () => {
      const textarea = document.createElement('textarea');
      const event = new KeyboardEvent('keydown', { key: 'p' });
      Object.defineProperty(event, 'target', { value: textarea });
      expect(shouldBlockShortcut(event, [])).toBe(true);
    });

    it('should not block in normal context', () => {
      const event = new KeyboardEvent('keydown', { key: 'p' });
      expect(shouldBlockShortcut(event, [])).toBe(false);
    });

    it('should not block browser shortcuts (Cmd/Meta + s/o/p)', () => {
      // metaKey alone defers to the browser
      const meta = new KeyboardEvent('keydown', { key: 's', metaKey: true });
      expect(shouldBlockShortcut(meta, [])).toBe(false);

      // Ctrl + s/o/p also defer
      for (const k of ['s', 'o', 'p']) {
        const ev = new KeyboardEvent('keydown', { key: k, ctrlKey: true });
        expect(shouldBlockShortcut(ev, [])).toBe(false);
      }
    });

    it('should not block ctrl-combos that are not s/o/p', () => {
      const ev = new KeyboardEvent('keydown', { key: 'a', ctrlKey: true });
      expect(shouldBlockShortcut(ev, [])).toBe(false);
    });
  });

  describe('generateNavigationHelp', () => {
    it('reports the all-displayed sentinel when no hidden dimensions exist', () => {
      const dims = createDims(3, [0, 1, 2]);
      expect(generateNavigationHelp(0, dims)).toEqual(['All dimensions are displayed (3D view)']);
    });

    it('reports the selected dimension and its formatted current value', () => {
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'X' }, { name: 'Y' }, { name: 'Z' }, { name: 'time', step: 1, discrete: true }]
      );
      dims.currentStep[3] = 7;

      const help = generateNavigationHelp(3, dims);
      // Selected line is always first when there are hidden dims.
      expect(help[0]).toBe('Selected: time = 7');
      // Lists the available non-displayed dimensions section.
      expect(help.some((line) => line === 'Non-displayed dimensions:')).toBe(true);
      // Includes the [1] hotkey marker for the first non-displayed dim.
      expect(help.some((line) => line.includes('[1] time'))).toBe(true);
      // Marks the currently-selected dim with the arrow indicator.
      expect(help.some((line) => line.includes('←'))).toBe(true);
    });

    it('falls back to "No dimension selected" for an out-of-range index', () => {
      const dims = createDims(4, [0, 1, 2]);
      const help = generateNavigationHelp(-1, dims);
      expect(help[0]).toBe('No dimension selected');
    });

    it('uses fallback names ("Dim N") when metadata is missing', () => {
      const dims: SimpleDims = {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0],
        metadata: undefined,
      } as SimpleDims;
      const help = generateNavigationHelp(3, dims);
      // Fallbacks kick in for both the selected line and the list.
      expect(help.some((line) => line.includes('Dimension 3'))).toBe(true);
      expect(help.some((line) => line.includes('Dim 3'))).toBe(true);
    });

    it('emits the navigation instructions block at the end', () => {
      const dims = createDims(4, [0, 1, 2]);
      const help = generateNavigationHelp(3, dims);
      expect(help).toContain('Navigation:');
      expect(help).toContain('  [1-9] Select dimension');
      expect(help).toContain('  [ ]   Navigate selected dimension');
    });

    it('uses blanks instead of [N] markers past the 9th non-displayed dim', () => {
      // 14 dimensions, 3 displayed → 11 hidden, navIdx 9 and 10 use blank keys.
      const dims = createDims(14, [0, 1, 2]);
      const help = generateNavigationHelp(0, dims);
      // The 10th hidden dim line (navIdx=9) should not contain "[10]".
      const tenthLine = help.find((line) => line.includes(' dim12:') || line.includes(' Dim 12'));
      expect(tenthLine).toBeDefined();
      expect(tenthLine).not.toMatch(/\[10\]/);
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

    describe('degenerate step values [input.md G25]', () => {
      // input.md G25[P5]: `Math.floor(Math.log10(meta.step))` produces
      // -Infinity when step === 0. The `Math.max(0, ...)` clamp ensures
      // negative-Infinity becomes Infinity, which Number.toFixed REJECTS
      // with RangeError. Pin the actual behaviour so a future guard
      // (e.g. `step > 0 ? ... : 2`) surfaces as an intentional change.
      it('[G25] step === 0 falls back to default 2 decimals (truthy-guard at format.ts L35)', () => {
        // Documents the actual behaviour: `meta?.step ? ... : 2` treats
        // step=0 as falsy → uses default 2. A future hardening that
        // expanded the guard to `step > 0 ? ... : 2` would be equivalent
        // here. A regression that changed the truthy check to
        // `meta?.step !== undefined` would expose the -Infinity bug
        // (decimals=Infinity → toFixed throws RangeError).
        const dims = createDims(
          4,
          [0, 1, 2],
          [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 0 }]
        );
        const result = formatDimensionValue(5.234, 3, dims);
        expect(result).toBe('5.23');
      });

      it('[G25] step undefined falls back to 2 decimal places (default branch)', () => {
        // The `meta?.step` falsy check uses `?` not ternary truth-test, so
        // undefined falls through to the `2` default. Pin that.
        const dims = createDims(
          4,
          [0, 1, 2],
          [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time' }]
        );
        const result = formatDimensionValue(5.234567, 3, dims);
        expect(result).toBe('5.23');
      });

      it('[G25] step === 1 → 0 decimal places (Math.log10(1)=0 → -Math.floor(0)=0)', () => {
        // Boundary: step=1 produces zero decimals (integer formatting).
        const dims = createDims(
          4,
          [0, 1, 2],
          [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 1 }]
        );
        const result = formatDimensionValue(5.7, 3, dims);
        expect(result).toBe('6');
      });

      it('[G25] step === 10 → 0 decimal places (Math.log10(10)=1 → max(0, -1)=0)', () => {
        // Symmetric upper-boundary test: when step is greater than 1, the
        // `Math.max(0, ...)` clamp ensures decimals never goes negative.
        const dims = createDims(
          4,
          [0, 1, 2],
          [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'time', step: 10 }]
        );
        const result = formatDimensionValue(123.456, 3, dims);
        expect(result).toBe('123');
      });
    });

    describe('discrete .5 boundary [input.md G26]', () => {
      // input.md G26[P5]: `Math.round(value)` for discrete dimensions —
      // .5 rounding behaviour was not pinned. JS uses "round half away
      // from -Infinity" (NOT banker's rounding): Math.round(0.5)=1,
      // Math.round(-0.5)=0, Math.round(1.5)=2, Math.round(2.5)=3.
      const dims = createDims(
        4,
        [0, 1, 2],
        [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 'frame', discrete: true }]
      );

      it.each([
        { input: 0.5, expected: '1' }, // Math.round(0.5) = 1
        { input: 1.5, expected: '2' }, // Math.round(1.5) = 2
        { input: 2.5, expected: '3' }, // Math.round(2.5) = 3
        { input: -0.5, expected: '0' }, // Math.round(-0.5) = 0 (NOT -1)
        { input: -1.5, expected: '-1' }, // Math.round(-1.5) = -1 (NOT -2)
      ])(
        '[G26] formats $input → "$expected" (JS-style round-half-up, not banker)',
        ({ input, expected }) => {
          expect(formatDimensionValue(input, 3, dims)).toBe(expected);
        }
      );
    });
  });
});

describe('InputHandler Type Definitions', () => {
  it('should export InputHandler class', { timeout: 15_000 }, async () => {
    // Dynamic import to avoid triggering complex dependencies.
    // The import pulls in the full dependency graph (THREE.js, scene managers,
    // UI components), which normally takes ~600ms but can exceed 5s under load.
    const module = await import('../../../input/input-handler');
    expect(module.InputHandler).toBeDefined();
    expect(typeof module.InputHandler).toBe('function');
    // input.md G12 fix: pin the actual class shape — the prototype
    // must expose `init` and `dispose` (the lifecycle contract). A
    // `typeof === 'function'` check alone would still pass for any
    // exported function, including a stub.
    expect(typeof module.InputHandler.prototype.init).toBe('function');
    expect(typeof module.InputHandler.prototype.dispose).toBe('function');
  });
});
