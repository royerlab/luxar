/**
 * Tests for `src/wasm/typescript/projection.ts` (nD → 3D projection,
 * bounds, mask-based compaction, visibility-mask helpers).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D7). Combines the five sibling
 * `projection: *` describes (extract_3d_positions, calculate_bounds_3d,
 * compact_by_mask, count_visible, radii_to_visibility_mask) since all
 * five test the same source module.
 */

import { describe, it, expect } from 'vitest';
import {
  extract_3d_positions,
  calculate_bounds_3d,
  compact_by_mask,
  count_visible,
  radii_to_visibility_mask,
} from '../../../../wasm/typescript';

describe('projection: extract_3d_positions', () => {
  it('should extract 3D positions from 5D data', () => {
    // 5D data: 2 points
    const positionsNd = new Float32Array([
      1.0,
      2.0,
      3.0,
      4.0,
      5.0, // point 0
      6.0,
      7.0,
      8.0,
      9.0,
      10.0, // point 1
    ]);
    const displayDims = new Uint32Array([0, 2, 4]); // X=dim0, Y=dim2, Z=dim4
    const output = new Float32Array(6);

    extract_3d_positions(positionsNd, displayDims, 5, 2, output);

    expect(output[0]).toBe(1.0); // point0.x = dim0
    expect(output[1]).toBe(3.0); // point0.y = dim2
    expect(output[2]).toBe(5.0); // point0.z = dim4
    expect(output[3]).toBe(6.0); // point1.x = dim0
    expect(output[4]).toBe(8.0); // point1.y = dim2
    expect(output[5]).toBe(10.0); // point1.z = dim4
  });

  it('should handle 2D display (z=0)', () => {
    // 4D data displayed as 2D
    const positionsNd = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const displayDims = new Uint32Array([0, 1]); // Only X and Y
    const output = new Float32Array(3);

    extract_3d_positions(positionsNd, displayDims, 4, 1, output);

    expect(output[0]).toBe(1.0); // x
    expect(output[1]).toBe(2.0); // y
    expect(output[2]).toBe(0.0); // z = 0 (default)
  });

  it('should handle reordered dimensions', () => {
    const positionsNd = new Float32Array([10.0, 20.0, 30.0]);
    const displayDims = new Uint32Array([2, 0, 1]); // Z->X, X->Y, Y->Z
    const output = new Float32Array(3);

    extract_3d_positions(positionsNd, displayDims, 3, 1, output);

    expect(output[0]).toBe(30.0); // x = dim2
    expect(output[1]).toBe(10.0); // y = dim0
    expect(output[2]).toBe(20.0); // z = dim1
  });
});

describe('projection: calculate_bounds_3d', () => {
  it('should calculate correct bounds', () => {
    const positions = new Float32Array([
      -1.0,
      2.0,
      3.0, // point 0
      4.0,
      -5.0,
      6.0, // point 1
      7.0,
      8.0,
      -9.0, // point 2
    ]);
    const output = new Float32Array(6);

    const count = calculate_bounds_3d(positions, 3, output);

    expect(count).toBe(3);
    expect(output[0]).toBe(-1.0); // min_x
    expect(output[1]).toBe(-5.0); // min_y
    expect(output[2]).toBe(-9.0); // min_z
    expect(output[3]).toBe(7.0); // max_x
    expect(output[4]).toBe(8.0); // max_y
    expect(output[5]).toBe(6.0); // max_z
  });

  it('should handle empty input', () => {
    const output = new Float32Array(6).fill(999);

    const count = calculate_bounds_3d(new Float32Array(0), 0, output);

    expect(count).toBe(0);
    for (let i = 0; i < 6; i++) {
      expect(output[i]).toBe(0);
    }
  });

  it('should handle single point', () => {
    const positions = new Float32Array([1.5, 2.5, 3.5]);
    const output = new Float32Array(6);

    const count = calculate_bounds_3d(positions, 1, output);

    expect(count).toBe(1);
    expect(output[0]).toBe(1.5); // min_x = max_x
    expect(output[1]).toBe(2.5); // min_y = max_y
    expect(output[2]).toBe(3.5); // min_z = max_z
    expect(output[3]).toBe(1.5);
    expect(output[4]).toBe(2.5);
    expect(output[5]).toBe(3.5);
  });
});

describe('projection: compact_by_mask', () => {
  it('should compact arrays by visibility mask', () => {
    const input = new Float32Array([
      1.0,
      2.0,
      3.0, // visible
      4.0,
      5.0,
      6.0, // hidden
      7.0,
      8.0,
      9.0, // visible
    ]);
    const mask = new Uint8Array([1, 0, 1]);
    const output = new Float32Array(6);

    const visible = compact_by_mask(input, mask, 3, 3, output);

    expect(visible).toBe(2);
    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(2.0);
    expect(output[2]).toBe(3.0);
    expect(output[3]).toBe(7.0);
    expect(output[4]).toBe(8.0);
    expect(output[5]).toBe(9.0);
  });

  it('should handle scalar stride', () => {
    const input = new Float32Array([10, 20, 30, 40, 50]);
    const mask = new Uint8Array([1, 0, 1, 0, 1]);
    const output = new Float32Array(3);

    const visible = compact_by_mask(input, mask, 5, 1, output);

    expect(visible).toBe(3);
    expect(output[0]).toBe(10);
    expect(output[1]).toBe(30);
    expect(output[2]).toBe(50);
  });

  it('should handle all hidden', () => {
    const input = new Float32Array([1, 2, 3]);
    const mask = new Uint8Array([0, 0, 0]);
    const output = new Float32Array(3);

    const visible = compact_by_mask(input, mask, 3, 1, output);

    expect(visible).toBe(0);
  });
});

describe('projection: count_visible', () => {
  it('should count non-zero mask values', () => {
    const mask = new Uint8Array([1, 0, 1, 0, 1, 1]);
    expect(count_visible(mask, 6)).toBe(4);
  });

  it('should handle all visible', () => {
    const mask = new Uint8Array([1, 1, 1]);
    expect(count_visible(mask, 3)).toBe(3);
  });

  it('should handle all hidden', () => {
    const mask = new Uint8Array([0, 0, 0, 0]);
    expect(count_visible(mask, 4)).toBe(0);
  });

  it('should handle empty', () => {
    expect(count_visible(new Uint8Array(0), 0)).toBe(0);
  });
});

describe('projection: radii_to_visibility_mask', () => {
  it('should create mask from radii threshold', () => {
    const radii = new Float32Array([0.5, 0.0001, 0.2, 0.0]);
    const output = new Uint8Array(4);

    const visible = radii_to_visibility_mask(radii, 0.0001, 4, output);

    expect(visible).toBe(2); // 0.5 and 0.2 > 0.0001
    expect(output[0]).toBe(1);
    expect(output[1]).toBe(0); // exactly at threshold = not visible
    expect(output[2]).toBe(1);
    expect(output[3]).toBe(0);
  });

  it('should handle zero threshold', () => {
    const radii = new Float32Array([0.001, 0.0, 0.5]);
    const output = new Uint8Array(3);

    const visible = radii_to_visibility_mask(radii, 0, 3, output);

    expect(visible).toBe(2);
    expect(output[0]).toBe(1);
    expect(output[1]).toBe(0); // 0 is not > 0
    expect(output[2]).toBe(1);
  });

  it('should handle large threshold (all hidden)', () => {
    const radii = new Float32Array([1.0, 2.0, 3.0]);
    const output = new Uint8Array(3);

    const visible = radii_to_visibility_mask(radii, 100, 3, output);

    expect(visible).toBe(0);
    expect(output[0]).toBe(0);
    expect(output[1]).toBe(0);
    expect(output[2]).toBe(0);
  });
});
