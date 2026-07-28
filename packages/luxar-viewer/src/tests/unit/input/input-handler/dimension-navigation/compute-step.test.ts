/**
 * Unit tests for the dimension-navigation pure helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  computeDimensionStep,
  getSelectedDimensionIndex,
  resolveSelectedDimension,
} from '../../../../../input/input-handler/dimension-navigation/compute-step';
import type { SimpleDims, DimensionMetadata } from '../../../../../types/dims';

function makeDims(overrides: Partial<SimpleDims> = {}): SimpleDims {
  return {
    ndim: 5,
    // 5D dataset, displayed = X, Y, Z (dims 0, 1, 2). Time and Channel
    // (dims 3, 4) are the navigable ones.
    currentStep: [0, 0, 0, 0, 0],
    displayed: [0, 1, 2],
    metadata: undefined,
    ...overrides,
  };
}

function makeMetadata(overrides: Partial<DimensionMetadata> = {}): DimensionMetadata {
  return {
    name: 'X',
    unit: '',
    scale: 1,
    ...overrides,
  };
}

describe('computeDimensionStep', () => {
  it('returns null when dims is null', () => {
    expect(computeDimensionStep(1, 0, null, [])).toBeNull();
  });

  it('returns null when dimensionRanges is null', () => {
    expect(computeDimensionStep(1, 0, makeDims(), null)).toBeNull();
  });

  it('returns null when no navigable (non-displayed) dimensions exist', () => {
    // 3D dataset where all 3 dims are displayed
    const dims = makeDims({ ndim: 3, currentStep: [0, 0, 0], displayed: [0, 1, 2] });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
    ];
    expect(computeDimensionStep(1, 0, dims, ranges)).toBeNull();
  });

  it('navigates the first navigable dim when selectedDimension=0', () => {
    const dims = makeDims({ currentStep: [0, 0, 0, 5, 5] });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100], // Time
      [0, 100], // Channel
    ];
    const result = computeDimensionStep(1, 0, dims, ranges);
    expect(result).not.toBeNull();
    expect(result!.targetDim).toBe(3); // first navigable = Time
    expect(result!.changed).toBe(true);
    expect(result!.newValue).toBeGreaterThan(5); // direction +1, moved up
  });

  it('navigates the second navigable dim when selectedDimension=1', () => {
    const dims = makeDims({ currentStep: [0, 0, 0, 5, 5] });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 1, dims, ranges);
    expect(result!.targetDim).toBe(4); // second navigable = Channel
  });

  it('clamps selectedDimension to navigable range', () => {
    // selectedDimension = 99, but only 2 navigable dims → should target the last (index 1 → dim 4)
    const dims = makeDims({ currentStep: [0, 0, 0, 5, 5] });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 99, dims, ranges);
    expect(result!.targetDim).toBe(4);
  });

  it('reports changed=false when at the maximum and direction is +1 (clamped to range)', () => {
    const dims = makeDims({
      currentStep: [0, 0, 0, 100, 0], // Time pinned at max
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'Time', step: 1 }),
        makeMetadata({ name: 'Channel', step: 1 }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 0, dims, ranges);
    expect(result!.targetDim).toBe(3);
    expect(result!.newValue).toBe(100);
    expect(result!.changed).toBe(false);
  });

  it('respects discrete metadata flag (rounds to integers)', () => {
    const dims = makeDims({
      currentStep: [0, 0, 0, 0.4, 0],
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'Time', discrete: true }),
        makeMetadata({ name: 'Channel' }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 0, dims, ranges);
    expect(Number.isInteger(result!.newValue)).toBe(true);
  });

  it('respects cyclic metadata flag (wraps around)', () => {
    const dims = makeDims({
      currentStep: [0, 0, 0, 100, 0], // pinned at max
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'Time', cyclic: true, step: 5 }),
        makeMetadata({ name: 'Channel' }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 0, dims, ranges);
    // Cyclic → wraps from 100+5 back into [0, 100], NOT clamped to 100
    expect(result!.newValue).not.toBe(100);
  });

  // input.md G9 fix: direction=-1 with cyclic wrap (the symmetric case
  // of the forward wrap test above).
  it('cyclic + direction=-1 wraps from the lower bound back to the upper end', () => {
    const dims = makeDims({
      currentStep: [0, 0, 0, 0, 0], // pinned at min
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'Time', cyclic: true, step: 5 }),
        makeMetadata({ name: 'Channel' }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    const result = computeDimensionStep(-1, 0, dims, ranges);
    // Cyclic + going back from 0 by step=5: wraps to max - (0 - (-5)) % 100 = 95
    expect(result!.newValue).toBe(95);
    expect(result!.changed).toBe(true);
  });

  // input.md G10 fix: pin the rounding direction for discrete dims.
  // calculateNextPosition uses Math.round AFTER applying step+direction:
  // current=0.4, direction=+1, step=1 → 1.4 → Math.round → 1.
  it('pins rounding direction for discrete dims (Math.round on the post-step value)', () => {
    const dims = makeDims({
      currentStep: [0, 0, 0, 0.4, 0],
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'Time', discrete: true, step: 1 }),
        makeMetadata({ name: 'Channel' }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [0, 100],
      [0, 100],
    ];
    // 0.4 + 1 = 1.4 → round → 1
    const fwd = computeDimensionStep(1, 0, dims, ranges);
    expect(fwd!.newValue).toBe(1);
    // 0.4 - 1 = -0.6 → round → -1 → clamped to 0 (min)
    const back = computeDimensionStep(-1, 0, dims, ranges);
    expect(back!.newValue).toBe(0);
  });

  // Discrete dims with a FRACTIONAL declared step (e.g. a 4th spatial axis
  // sampled every 0.04 units) must step cell-by-cell on the k*step grid,
  // not collapse to whole units. Historically the step was forced to
  // max(1, round(step)) and the position rounded to integers, so [/]
  // could only reach 3 of the 50 planes of such a dim.
  it('steps a fractional-step discrete dim by exactly one grid cell', () => {
    const step = 2.0 / 50; // 0.04
    const dims = makeDims({
      currentStep: [0, 0, 0, -1.0, 0],
      metadata: [
        makeMetadata({ name: 'X', display: true }),
        makeMetadata({ name: 'Y', display: true }),
        makeMetadata({ name: 'Z', display: true }),
        makeMetadata({ name: 'w', discrete: true, step }),
        makeMetadata({ name: 'Channel' }),
      ],
    });
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [-1.0, 0.96],
      [0, 100],
    ];
    const result = computeDimensionStep(1, 0, dims, ranges);
    // Bit-identical to the manager's own snap: round(v/step)*step
    expect(result!.newValue).toBe(-24 * step);
    expect(result!.changed).toBe(true);
  });

  it('traverses every k*step stop of a fractional-step discrete dim', () => {
    const step = 2.0 / 50;
    const metadata = [
      makeMetadata({ name: 'X', display: true }),
      makeMetadata({ name: 'Y', display: true }),
      makeMetadata({ name: 'Z', display: true }),
      makeMetadata({ name: 'w', discrete: true, step }),
      makeMetadata({ name: 'Channel' }),
    ];
    const ranges: ReadonlyArray<readonly [number, number]> = [
      [0, 10],
      [0, 10],
      [0, 10],
      [-1.0, 0.96],
      [0, 100],
    ];
    let pos = -1.0;
    for (let k = -25; k < 24; k++) {
      const dims = makeDims({ currentStep: [0, 0, 0, pos, 0], metadata });
      const result = computeDimensionStep(1, 0, dims, ranges);
      expect(result!.changed).toBe(true);
      expect(result!.newValue).toBe((k + 1) * step);
      pos = result!.newValue;
    }
    // 49 presses from the first stop land exactly on the last stop
    expect(pos).toBe(24 * step);
  });
});

describe('getSelectedDimensionIndex', () => {
  it('returns -1 when selectedDimension is negative', () => {
    expect(getSelectedDimensionIndex(-1, makeDims())).toBe(-1);
  });

  it('returns -1 when dims is null', () => {
    expect(getSelectedDimensionIndex(0, null)).toBe(-1);
  });

  it('returns -1 when selectedDimension is past the navigable range', () => {
    // 5D dataset, 2 navigable (3, 4). Select index 5.
    expect(getSelectedDimensionIndex(5, makeDims())).toBe(-1);
  });

  it('returns the actual dim index for a valid 0-based selection', () => {
    expect(getSelectedDimensionIndex(0, makeDims())).toBe(3); // first navigable = Time
    expect(getSelectedDimensionIndex(1, makeDims())).toBe(4); // second navigable = Channel
  });

  it('returns -1 for any selection in a 3D-all-displayed dataset', () => {
    const dims = makeDims({ ndim: 3, currentStep: [0, 0, 0], displayed: [0, 1, 2] });
    expect(getSelectedDimensionIndex(0, dims)).toBe(-1);
    expect(getSelectedDimensionIndex(1, dims)).toBe(-1);
  });
});

describe('resolveSelectedDimension', () => {
  it('returns null + count=0 when dims is null', () => {
    const result = resolveSelectedDimension(0, null);
    expect(result).toEqual({ selectedDimension: null, navigableCount: 0 });
  });

  it('returns selectedDimension=index when key resolves to a valid navigable dim', () => {
    const dims = makeDims(); // 5D, dims 3,4 are navigable
    expect(resolveSelectedDimension(0, dims)).toEqual({ selectedDimension: 0 });
    expect(resolveSelectedDimension(1, dims)).toEqual({ selectedDimension: 1 });
  });

  it('returns null + the navigable count when key falls past the available dims', () => {
    const dims = makeDims(); // 2 navigable dims (3 and 4)
    const result = resolveSelectedDimension(2, dims); // key '3' → no third navigable
    expect(result).toEqual({ selectedDimension: null, navigableCount: 2 });
  });

  it('handles 3D-all-displayed (zero navigable dims)', () => {
    const dims = makeDims({ ndim: 3, currentStep: [0, 0, 0], displayed: [0, 1, 2] });
    const result = resolveSelectedDimension(0, dims);
    expect(result).toEqual({ selectedDimension: null, navigableCount: 0 });
  });
});
