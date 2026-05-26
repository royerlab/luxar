/**
 * Tests for `src/wasm/typescript/spatial.ts` (chunk bounding-box queries).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1): the 1714-line mega-file is being
 * split into per-source-module test files mirroring `src/wasm/typescript/`.
 */

import { describe, it, expect } from 'vitest';
import { query_chunks_for_view } from '../../../../wasm/typescript';

describe('spatial: query_chunks_for_view', () => {
  it('should find intersecting chunks in 3D', () => {
    const chunkBounds = new Float32Array([
      // Chunk 0: [0,0,0] to [1,1,1]
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      // Chunk 1: [1,1,1] to [2,2,2]
      1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
      // Chunk 2: [5,5,5] to [6,6,6] (far away)
      5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
    ]);

    const slicePos = new Float32Array([0.5, 0.5, 0.5]);
    const tolerance = new Float32Array([0.6, 0.6, 0.6]);

    const output = new Uint32Array(3);
    const count = query_chunks_for_view(chunkBounds, slicePos, tolerance, 3, 3, output);

    expect(count).toBe(2);
    expect(output[0]).toBe(0);
    expect(output[1]).toBe(1);
  });

  it('should handle 5D data with selective tolerance', () => {
    const chunkBounds = new Float32Array([
      // Chunk 0: all dims [0, 1]
      0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
      // Chunk 1: all dims [2, 3]
      2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
    ]);

    const slicePos = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5]);
    const tolerance = new Float32Array([100, 100, 100, 0.6, 0.6]);

    const output = new Uint32Array(2);
    const count = query_chunks_for_view(chunkBounds, slicePos, tolerance, 5, 2, output);

    expect(count).toBe(1);
    expect(output[0]).toBe(0);
  });

  it('should handle empty input', () => {
    const output = new Uint32Array(0);
    const count = query_chunks_for_view(
      new Float32Array(0),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      0,
      output
    );
    expect(count).toBe(0);
  });

  it('should find all chunks with infinite tolerance', () => {
    const chunkBounds = new Float32Array([
      0.0, 1.0, 0.0, 1.0, 100.0, 200.0, 0.0, 1.0, -50.0, -40.0, 0.0, 1.0,
    ]);

    const output = new Uint32Array(3);
    const count = query_chunks_for_view(
      chunkBounds,
      new Float32Array([0, 0]),
      new Float32Array([Infinity, Infinity]),
      2,
      3,
      output
    );

    expect(count).toBe(3);
  });
});
