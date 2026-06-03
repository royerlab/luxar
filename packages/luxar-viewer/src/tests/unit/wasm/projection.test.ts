/**
 * Unit tests for the TypeScript fallback of the WASM projection routines.
 *
 * These run when the WASM module isn't available (or loaded asynchronously
 * the call site couldn't wait for); the fallbacks must behave identically
 * to the WASM implementations. Pure math on typed arrays — no mocks.
 */

import { describe, it, expect } from 'vitest';
import {
  calculate_bounds_3d,
  compact_by_mask,
  count_visible,
  extract_3d_positions,
  radii_to_visibility_mask,
} from '../../../wasm/typescript/projection';

describe('extract_3d_positions', () => {
  it('selects the displayed dimensions from each nD position', () => {
    // 3 points in 4D; show dims [0, 1, 2].
    const positionsNd = new Float32Array([1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 99]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const output = new Float32Array(9);
    extract_3d_positions(positionsNd, displayDims, 4, 3, output);
    expect(Array.from(output)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('reorders dimensions when displayDims is non-canonical', () => {
    const positionsNd = new Float32Array([1, 2, 3, 4]);
    const displayDims = new Uint32Array([3, 0, 2]);
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 4, 1, output);
    expect(Array.from(output)).toEqual([4, 1, 3]);
  });

  it('zero-fills the unused output axes when fewer than 3 displayDims', () => {
    const positionsNd = new Float32Array([10, 20, 30]);
    const displayDims = new Uint32Array([0, 1]); // 2D output
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 3, 1, output);
    expect(output[0]).toBe(10);
    expect(output[1]).toBe(20);
    expect(output[2]).toBe(0);
  });

  it('caps at 3 displayDims even when more are supplied', () => {
    // ndim=5, displayDims has 5 entries — only first 3 should be used.
    const positionsNd = new Float32Array([1, 2, 3, 4, 5]);
    const displayDims = new Uint32Array([0, 1, 2, 3, 4]);
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 5, 1, output);
    expect(Array.from(output)).toEqual([1, 2, 3]);
  });

  it('handles numPoints=0 by leaving output untouched (no iteration)', () => {
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(new Float32Array(0), new Uint32Array([0, 1, 2]), 3, 0, output);
    expect(Array.from(output)).toEqual([42, 42, 42]);
  });
});

describe('calculate_bounds_3d', () => {
  it('returns 0-bounds for empty input', () => {
    const output = new Float32Array(6);
    const n = calculate_bounds_3d(new Float32Array(0), 0, output);
    expect(n).toBe(0);
    expect(Array.from(output)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('initializes with the first point and produces a degenerate box for one point', () => {
    const positions = new Float32Array([1, 2, 3]);
    const output = new Float32Array(6);
    const n = calculate_bounds_3d(positions, 1, output);
    expect(n).toBe(1);
    expect(Array.from(output)).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it('expands min/max across multiple points', () => {
    const positions = new Float32Array([1, 0, 0, -2, 5, 1, 3, -1, 4]);
    const output = new Float32Array(6);
    calculate_bounds_3d(positions, 3, output);
    // min: -2, -1, 0   max: 3, 5, 4
    expect(Array.from(output)).toEqual([-2, -1, 0, 3, 5, 4]);
  });

  it('returns numPoints (not visible count) so callers know the input size', () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const output = new Float32Array(6);
    const n = calculate_bounds_3d(positions, 2, output);
    expect(n).toBe(2);
  });
});

describe('compact_by_mask', () => {
  it('keeps only entries with mask[i] !== 0', () => {
    const input = new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3]);
    const mask = new Uint8Array([1, 0, 1]);
    const output = new Float32Array(6);
    const visible = compact_by_mask(input, mask, 3, 3, output);
    expect(visible).toBe(2);
    // First match (index 0): 1,1,1
    // Second match (index 2): 3,3,3
    expect(Array.from(output.slice(0, 6))).toEqual([1, 1, 1, 3, 3, 3]);
  });

  it('returns 0 and writes nothing when no entries are visible', () => {
    const input = new Float32Array([1, 2, 3]);
    const mask = new Uint8Array([0, 0, 0]);
    const output = new Float32Array(3).fill(99);
    const visible = compact_by_mask(input, mask, 3, 1, output);
    expect(visible).toBe(0);
    expect(Array.from(output)).toEqual([99, 99, 99]);
  });

  it('handles stride=1 (scalar) compaction', () => {
    const input = new Float32Array([10, 20, 30, 40]);
    const mask = new Uint8Array([1, 0, 0, 1]);
    const output = new Float32Array(4);
    const visible = compact_by_mask(input, mask, 4, 1, output);
    expect(visible).toBe(2);
    expect(Array.from(output.slice(0, 2))).toEqual([10, 40]);
  });

  it('treats any non-zero mask value as visible (not just 1)', () => {
    const input = new Float32Array([1, 2, 3]);
    const mask = new Uint8Array([7, 0, 255]);
    const output = new Float32Array(2);
    const visible = compact_by_mask(input, mask, 3, 1, output);
    expect(visible).toBe(2);
    expect(Array.from(output)).toEqual([1, 3]);
  });
});

describe('count_visible', () => {
  it('returns 0 for an all-hidden mask', () => {
    expect(count_visible(new Uint8Array([0, 0, 0, 0]), 4)).toBe(0);
  });

  it('returns the count for an all-visible mask', () => {
    expect(count_visible(new Uint8Array([1, 1, 1, 1]), 4)).toBe(4);
  });

  it('counts only the first `count` entries even when the mask is longer', () => {
    const mask = new Uint8Array([1, 1, 1, 0, 1]);
    expect(count_visible(mask, 3)).toBe(3); // first 3 only
    expect(count_visible(mask, 5)).toBe(4); // 4 of 5
  });

  it('treats any non-zero value as visible', () => {
    expect(count_visible(new Uint8Array([2, 0, 3, 255]), 4)).toBe(3);
  });
});

describe('radii_to_visibility_mask', () => {
  it('returns 0 visible when every radius is at or below the threshold', () => {
    const radii = new Float32Array([0, 1, 1, 0.5]);
    const out = new Uint8Array(4);
    const visible = radii_to_visibility_mask(radii, 1.0, 4, out);
    expect(visible).toBe(0);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });

  it('marks radii > threshold as visible (strict inequality)', () => {
    const radii = new Float32Array([0, 0.5, 1.0, 1.5, 2.0]);
    const out = new Uint8Array(5);
    const visible = radii_to_visibility_mask(radii, 1.0, 5, out);
    expect(visible).toBe(2);
    expect(Array.from(out)).toEqual([0, 0, 0, 1, 1]);
  });

  it('marks all visible when every radius exceeds the threshold', () => {
    const radii = new Float32Array([2, 3, 4]);
    const out = new Uint8Array(3);
    const visible = radii_to_visibility_mask(radii, 1.0, 3, out);
    expect(visible).toBe(3);
    expect(Array.from(out)).toEqual([1, 1, 1]);
  });

  it('handles count=0 by returning 0 without writing', () => {
    const out = new Uint8Array(3).fill(99);
    const visible = radii_to_visibility_mask(new Float32Array(0), 0.5, 0, out);
    expect(visible).toBe(0);
    expect(Array.from(out)).toEqual([99, 99, 99]);
  });

  // [wasm.md/G13][P5] threshold<0 boundary: source uses strict `radii[i] >
  // threshold`. A negative threshold means every non-negative radius is
  // visible, including zero-radius points. Pins the strict inequality so
  // a mutant `>` -> `>=` would not silently invert the zero case.
  it('marks zero radii as NOT visible against a negative threshold (strict >)', () => {
    const radii = new Float32Array([0, 0, 0]);
    const out = new Uint8Array(3);
    const visible = radii_to_visibility_mask(radii, -0.5, 3, out);
    // 0 > -0.5 → true, so all three are visible.
    expect(visible).toBe(3);
    expect(Array.from(out)).toEqual([1, 1, 1]);
  });

  // [wasm.md/G13][P5] NaN propagation: `NaN > threshold` is always false
  // in IEEE-754, so NaN radii must be marked hidden regardless of the
  // threshold value. Pins the silent-NaN contract — a mutant that
  // pre-converted NaN to 0 would lose this guard.
  it('marks NaN radii as hidden (NaN > x is always false in IEEE-754)', () => {
    const radii = new Float32Array([NaN, 1.0, NaN]);
    const out = new Uint8Array(3);
    const visible = radii_to_visibility_mask(radii, 0.5, 3, out);
    expect(visible).toBe(1); // only the middle 1.0 passes
    expect(Array.from(out)).toEqual([0, 1, 0]);
  });

  // [wasm.md/G13][P5] NaN threshold: comparison against NaN is always
  // false, so every radius (including infinity) must be marked hidden.
  // A mutant that special-cased a NaN threshold would survive without
  // this pin.
  it('threshold=NaN marks every radius hidden (x > NaN is always false)', () => {
    const radii = new Float32Array([0, 1, 100, Number.POSITIVE_INFINITY]);
    const out = new Uint8Array(4);
    const visible = radii_to_visibility_mask(radii, NaN, 4, out);
    expect(visible).toBe(0);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });

  // [wasm.md/G13][P5] Infinity threshold: comparison must drop every
  // finite radius. Only +Infinity (if present) passes — and even then
  // `Infinity > Infinity` is false (strict inequality), so all-finite
  // and +Infinity radii are uniformly hidden.
  it('threshold=+Infinity hides every finite radius (and +Infinity itself: strict >)', () => {
    const radii = new Float32Array([0, 1, 1e30, Number.POSITIVE_INFINITY]);
    const out = new Uint8Array(4);
    const visible = radii_to_visibility_mask(radii, Number.POSITIVE_INFINITY, 4, out);
    expect(visible).toBe(0);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });
});

// [wasm.md/G8][P5] extract_3d_positions displayDims.length=0 and =1 cases.
// The source caps numDisplayDims at min(displayDims.length, 3) and
// zero-fills the rest; the boundary cases are unverified above.
describe('extract_3d_positions — displayDims boundary lengths', () => {
  it('displayDims.length=0 zero-fills all 3 output slots', () => {
    const positionsNd = new Float32Array([10, 20, 30]);
    const displayDims = new Uint32Array([]); // no displayed dims
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(positionsNd, displayDims, 3, 1, output);
    expect(Array.from(output)).toEqual([0, 0, 0]);
  });

  it('displayDims.length=1 puts that dim at output[0] and zero-fills rest', () => {
    // ndim=4, single displayed dim selecting index 2 → output [d2, 0, 0].
    const positionsNd = new Float32Array([100, 200, 300, 400]);
    const displayDims = new Uint32Array([2]);
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(positionsNd, displayDims, 4, 1, output);
    expect(Array.from(output)).toEqual([300, 0, 0]);
  });
});

// [wasm.md/G7][P5] calculate_bounds_3d with NaN/Infinity inputs. Pins
// the IEEE-754 propagation contract: Math.min/Math.max with NaN yield
// NaN, with +/-Infinity yield the infinite value. A mutant that pre-
// filtered NaN/Inf would silently corrupt bounding boxes.
describe('calculate_bounds_3d — NaN / Infinity propagation', () => {
  it('propagates a NaN coordinate into the corresponding bounds slot', () => {
    const positions = new Float32Array([1, 2, 3, NaN, 5, 6]);
    const output = new Float32Array(6);
    calculate_bounds_3d(positions, 2, output);
    // min_x sees NaN at the second point; Math.min(1, NaN) → NaN.
    expect(Number.isNaN(output[0])).toBe(true);
    expect(Number.isNaN(output[3])).toBe(true);
    // y/z dims are unaffected.
    expect(output[1]).toBe(2);
    expect(output[4]).toBe(5);
  });

  it('lets +Infinity be the max-x and -Infinity be the min-x', () => {
    const positions = new Float32Array([0, 0, 0, Infinity, 10, 10, -Infinity, -10, -10]);
    const output = new Float32Array(6);
    calculate_bounds_3d(positions, 3, output);
    expect(output[0]).toBe(-Infinity); // min_x
    expect(output[3]).toBe(Infinity); // max_x
    expect(output[1]).toBe(-10); // min_y
    expect(output[4]).toBe(10); // max_y
  });
});
