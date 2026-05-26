/**
 * Tests for `src/wasm/typescript/lines.ts` (segment endpoint visibility).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D2): the 1714-line mega-file
 * is being split into per-source-module test files mirroring
 * `src/wasm/typescript/`.
 */

import { describe, it, expect } from 'vitest';
import { compute_nd_visibility_lines } from '../../../../wasm/typescript';

describe('lines: compute_nd_visibility_lines', () => {
  it('should filter segments based on endpoint visibility', () => {
    const ndim = 5;

    const vertices = new Float32Array([
      0.0,
      0.0,
      0.0,
      0.0,
      0.0, // V0: visible
      1.0,
      1.0,
      1.0,
      0.5,
      0.0, // V1: visible
      2.0,
      2.0,
      2.0,
      10.0,
      0.0, // V2: hidden
      3.0,
      3.0,
      3.0,
      0.0,
      10.0, // V3: hidden
    ]);

    const segments = new Uint32Array([0, 1, 1, 2, 2, 3]);
    const widths = new Float32Array([1.0, 1.0, 1.0, 1.0]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
    const tolerance = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0]);

    const output = new Uint8Array(3);
    const count = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      slicePos,
      tolerance,
      ndim,
      3,
      output
    );

    expect(output[0]).toBe(1); // 0-1: both visible
    expect(output[1]).toBe(1); // 1-2: one visible
    expect(output[2]).toBe(0); // 2-3: both hidden
    expect(count).toBe(2);
  });
});
