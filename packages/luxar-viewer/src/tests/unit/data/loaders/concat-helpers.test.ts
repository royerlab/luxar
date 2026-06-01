/**
 * Unit tests for the shared progressive-loader concatenation helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  concatRequiredField,
  concatOptionalField,
} from '../../../../data/loaders/progressive/concat-helpers';

type Part = {
  count: number;
  values?: Float32Array | Uint8Array;
};

describe('concatRequiredField', () => {
  it('concatenates a per-row field at the running offset (perItem=1)', () => {
    const parts: Part[] = [
      { count: 2, values: new Float32Array([1, 2]) },
      { count: 3, values: new Float32Array([3, 4, 5]) },
    ];
    const out = concatRequiredField(parts, (p) => p.values!, (p) => p.count);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5]);
    expect(out).toBeInstanceOf(Float32Array);
  });

  it('respects perItem (e.g. 3-vector positions)', () => {
    const parts: Part[] = [
      { count: 1, values: new Float32Array([0, 0, 0]) },
      { count: 2, values: new Float32Array([1, 1, 1, 2, 2, 2]) },
    ];
    const out = concatRequiredField(parts, (p) => p.values!, (p) => p.count, 3);
    expect(out.length).toBe(9);
    expect(Array.from(out.slice(3))).toEqual([1, 1, 1, 2, 2, 2]);
  });

  it('preserves the source dtype (Uint8Array)', () => {
    const parts: Part[] = [
      { count: 1, values: new Uint8Array([10]) },
      { count: 1, values: new Uint8Array([20]) },
    ];
    const out = concatRequiredField(parts, (p) => p.values!, (p) => p.count);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out)).toEqual([10, 20]);
  });
});

describe('concatOptionalField', () => {
  it('returns the concatenation when every part has the field', () => {
    const parts: Part[] = [
      { count: 1, values: new Float32Array([1]) },
      { count: 1, values: new Float32Array([2]) },
    ];
    const out = concatOptionalField(parts, (p) => p.values, (p) => p.count);
    expect(out && Array.from(out)).toEqual([1, 2]);
  });

  it('returns undefined when any part lacks the field (all-or-nothing)', () => {
    const parts: Part[] = [
      { count: 1, values: new Float32Array([1]) },
      { count: 1 }, // missing
    ];
    const out = concatOptionalField(parts, (p) => p.values, (p) => p.count);
    expect(out).toBeUndefined();
  });
});
