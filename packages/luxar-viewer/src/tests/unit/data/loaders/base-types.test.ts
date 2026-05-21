/**
 * Unit tests for base-types view-state helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  hasDimensionMetadata,
  getDisplayDimCount,
  isHiddenDimension,
  type BaseViewState,
} from '../../../../data/loaders';
import type { DimensionMetadata } from '../../../../types/dims';

const sampleDims: DimensionMetadata[] = [
  { name: 't', unit: 'frame', scale: 1 },
  { name: 'z', unit: 'um', scale: 1 },
  { name: 'y', unit: 'um', scale: 1 },
  { name: 'x', unit: 'um', scale: 1 },
];

function viewState(overrides: Partial<BaseViewState> = {}): BaseViewState {
  return {
    displayDims: [1, 2, 3],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0.5, 0, 0, 0],
    ...overrides,
  };
}

describe('hasDimensionMetadata', () => {
  it('returns false when dimensions is undefined', () => {
    expect(hasDimensionMetadata(viewState())).toBe(false);
  });

  it('returns false when dimensions is an empty array', () => {
    expect(hasDimensionMetadata(viewState({ dimensions: [] }))).toBe(false);
  });

  it('returns true when dimensions has at least one entry', () => {
    expect(hasDimensionMetadata(viewState({ dimensions: sampleDims }))).toBe(true);
  });

  it('narrows the type so dimensions is non-optional', () => {
    const state = viewState({ dimensions: sampleDims });
    if (hasDimensionMetadata(state)) {
      // Compile-time + runtime check that .dimensions is now defined
      expect(state.dimensions.length).toBe(4);
    } else {
      throw new Error('expected narrowing to succeed');
    }
  });
});

describe('getDisplayDimCount', () => {
  it('returns 3 for a typical 3-axis display', () => {
    expect(getDisplayDimCount(viewState({ displayDims: [0, 1, 2] }))).toBe(3);
  });

  it('returns 2 for a 2D display', () => {
    expect(getDisplayDimCount(viewState({ displayDims: [0, 1] }))).toBe(2);
  });

  it('caps at 3 even when more dims are listed', () => {
    expect(getDisplayDimCount(viewState({ displayDims: [0, 1, 2, 3, 4] }))).toBe(3);
  });

  it('returns 0 for an empty displayDims array', () => {
    expect(getDisplayDimCount(viewState({ displayDims: [] }))).toBe(0);
  });
});

describe('isHiddenDimension', () => {
  it('returns false for a dimension that IS displayed', () => {
    const state = viewState({ displayDims: [1, 2, 3] });
    expect(isHiddenDimension(state, 1)).toBe(false);
    expect(isHiddenDimension(state, 2)).toBe(false);
    expect(isHiddenDimension(state, 3)).toBe(false);
  });

  it('returns true for a dimension that is NOT displayed', () => {
    const state = viewState({ displayDims: [1, 2, 3] });
    expect(isHiddenDimension(state, 0)).toBe(true);
    expect(isHiddenDimension(state, 4)).toBe(true);
  });
});
