/**
 * Unit tests for the lines-specific vertex-index helpers in
 * `data/lines/chunk-index-loader`.
 *
 * Focus:
 *   - `sortedUniqueVertexIndices` — sort + single-pass dedupe over the raw
 *     per-segment `Uint32Array`, including the 2^24 regression guard for
 *     issue #1049 (the old `Set`-based implementation threw `RangeError`).
 *   - `computeVertexRangesFromIndices` — accepts a `Uint32Array` and yields
 *     the same ranges as the equivalent `number[]`.
 */

import { describe, it, expect } from 'vitest';
import {
  sortedUniqueVertexIndices,
  computeVertexRangesFromIndices,
  remapSegmentIndices,
} from '../../../../data/lines/chunk-index-loader';

/**
 * Independent reference remap, identical in behaviour to the old Map-based
 * code the helper replaces: assigns a local index to every vertex in the
 * concatenation of the ascending ranges, then looks each segment index up.
 * Used to cross-check {@link remapSegmentIndices} on random valid inputs.
 */
function referenceRemap(
  segmentData: Uint32Array,
  ranges: readonly { start: number; end: number }[]
): { segments: Uint32Array; count: number } {
  const map = new Map<number, number>();
  let localIdx = 0;
  for (const range of ranges) {
    for (let i = range.start; i < range.end; i++) {
      map.set(i, localIdx++);
    }
  }
  const segments = new Uint32Array(segmentData.length);
  for (let i = 0; i < segmentData.length; i++) {
    const local = map.get(segmentData[i]);
    if (local === undefined) {
      throw new Error(`Vertex index ${segmentData[i]} not found in loaded data`);
    }
    segments[i] = local;
  }
  return { segments, count: map.size };
}

describe('sortedUniqueVertexIndices', () => {
  it('returns an empty array for empty input', () => {
    const result = sortedUniqueVertexIndices(new Uint32Array(0));
    expect(result).toBeInstanceOf(Uint32Array);
    expect(result.length).toBe(0);
  });

  it('sorts and de-duplicates unsorted input with duplicates', () => {
    // Multi-digit values so this test alone pins numeric (not lexicographic)
    // ordering: a string sort would give [0, 100, 2, 33, 5].
    const input = new Uint32Array([100, 2, 33, 2, 100, 5, 33, 0]);
    const result = sortedUniqueVertexIndices(input);
    const reference = Array.from(new Set(Array.from(input))).sort((a, b) => a - b);
    expect(Array.from(result)).toEqual(reference);
    expect(Array.from(result)).toEqual([0, 2, 5, 33, 100]);
  });

  it('does not mutate the caller-supplied segmentData', () => {
    const input = new Uint32Array([5, 1, 3, 1, 5, 2, 3, 0]);
    const snapshot = Array.from(input);
    sortedUniqueVertexIndices(input);
    expect(Array.from(input)).toEqual(snapshot);
  });

  it('handles a single value', () => {
    const result = sortedUniqueVertexIndices(new Uint32Array([42]));
    expect(Array.from(result)).toEqual([42]);
  });

  it('collapses all-duplicate input to a single value', () => {
    const result = sortedUniqueVertexIndices(new Uint32Array([7, 7, 7, 7, 7]));
    expect(Array.from(result)).toEqual([7]);
  });

  // Regression guard for issue #1049: a lines node referencing more than
  // 2^24 unique vertex indices must load. The old `Set<number>`-based
  // implementation threw `RangeError: Set maximum size exceeded` here
  // because V8 caps a `Set` at exactly 2^24 (16,777,216) entries.
  // NOTE: allocates a ~67MB Uint32Array (2^24 + 1 entries); a plain
  // for-loop fill keeps allocations minimal.
  it('handles more than 2^24 unique indices without throwing', () => {
    const n = 16_777_217; // 2^24 + 1
    const input = new Uint32Array(n);
    for (let i = 0; i < n; i++) input[i] = i; // already unique & sorted
    let result: Uint32Array | undefined;
    expect(() => {
      result = sortedUniqueVertexIndices(input);
    }).not.toThrow();
    expect(result!.length).toBe(n);
    expect(result![0]).toBe(0);
    expect(result![n - 1]).toBe(16_777_216);
  });
});

describe('computeVertexRangesFromIndices', () => {
  it('accepts a Uint32Array and matches the equivalent number[]', () => {
    const values = [0, 1, 2, 5, 6, 10];
    const fromArray = computeVertexRangesFromIndices(values);
    const fromTyped = computeVertexRangesFromIndices(Uint32Array.from(values));
    expect(fromTyped).toEqual(fromArray);
    expect(fromTyped).toEqual([
      { start: 0, end: 3 },
      { start: 5, end: 7 },
      { start: 10, end: 11 },
    ]);
  });

  it('composes with sortedUniqueVertexIndices end-to-end', () => {
    const segmentData = new Uint32Array([2, 0, 1, 2, 5, 6, 0]);
    const ranges = computeVertexRangesFromIndices(sortedUniqueVertexIndices(segmentData));
    expect(ranges).toEqual([
      { start: 0, end: 3 },
      { start: 5, end: 7 },
    ]);
  });
});

describe('remapSegmentIndices', () => {
  it('remaps a single range to a 0-based local space', () => {
    // One range [100, 105): global 100 → local 0, 104 → local 4.
    const ranges = [{ start: 100, end: 105 }];
    const segmentData = new Uint32Array([100, 104, 102, 100]);
    const out = new Uint32Array(segmentData.length);
    const count = remapSegmentIndices(segmentData, ranges, out);
    expect(count).toBe(5);
    expect(Array.from(out)).toEqual([0, 4, 2, 0]);
  });

  it('assigns local indices as the offset within the concatenated ranges', () => {
    // Ranges concatenate ascending: [10,13) → locals 0,1,2; [20,22) → 3,4;
    // [30,31) → 5. This is exactly the old Map-based localIdx++ ordering.
    const ranges = [
      { start: 10, end: 13 },
      { start: 20, end: 22 },
      { start: 30, end: 31 },
    ];
    const segmentData = new Uint32Array([30, 10, 21, 12, 20, 11]);
    const out = new Uint32Array(segmentData.length);
    const count = remapSegmentIndices(segmentData, ranges, out);
    expect(count).toBe(6);
    expect(Array.from(out)).toEqual([5, 0, 4, 2, 3, 1]);
    // Cross-check against the independent Map-based reference.
    const ref = referenceRemap(segmentData, ranges);
    expect(Array.from(out)).toEqual(Array.from(ref.segments));
    expect(count).toBe(ref.count);
  });

  it('handles repeated indices and arbitrary order', () => {
    const ranges = [
      { start: 0, end: 2 },
      { start: 5, end: 8 },
    ];
    const segmentData = new Uint32Array([7, 7, 0, 5, 1, 5, 6, 0]);
    const out = new Uint32Array(segmentData.length);
    const count = remapSegmentIndices(segmentData, ranges, out);
    const ref = referenceRemap(segmentData, ranges);
    expect(Array.from(out)).toEqual(Array.from(ref.segments));
    expect(count).toBe(ref.count);
    expect(count).toBe(5);
  });

  it('matches the Map-based reference on random valid inputs', () => {
    // Deterministic LCG so failures reproduce.
    let seed = 0x1234abcd;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let trial = 0; trial < 25; trial++) {
      // Build a few disjoint, ascending ranges with gaps between them.
      const nRanges = 1 + Math.floor(rand() * 5);
      const ranges: { start: number; end: number }[] = [];
      let cursor = Math.floor(rand() * 50);
      for (let r = 0; r < nRanges; r++) {
        const width = 1 + Math.floor(rand() * 8);
        ranges.push({ start: cursor, end: cursor + width });
        cursor += width + 1 + Math.floor(rand() * 10); // gap keeps ranges disjoint
      }
      // Collect all valid global indices, then sample segments from them.
      const valid: number[] = [];
      for (const range of ranges) {
        for (let i = range.start; i < range.end; i++) valid.push(i);
      }
      const nSeg = 1 + Math.floor(rand() * 40);
      const segmentData = new Uint32Array(nSeg);
      for (let i = 0; i < nSeg; i++) {
        segmentData[i] = valid[Math.floor(rand() * valid.length)];
      }
      const out = new Uint32Array(nSeg);
      const count = remapSegmentIndices(segmentData, ranges, out);
      const ref = referenceRemap(segmentData, ranges);
      expect(Array.from(out)).toEqual(Array.from(ref.segments));
      expect(count).toBe(ref.count);
    }
  });

  it('throws "not found in loaded data" for an index outside the ranges', () => {
    const ranges = [
      { start: 0, end: 3 },
      { start: 10, end: 12 },
    ];
    // 5 falls in the gap between the two ranges.
    const segmentData = new Uint32Array([0, 5, 11]);
    const out = new Uint32Array(segmentData.length);
    expect(() => remapSegmentIndices(segmentData, ranges, out)).toThrow(
      'Vertex index 5 not found in loaded data'
    );
  });

  it('throws for an index above every range', () => {
    const ranges = [{ start: 0, end: 4 }];
    const segmentData = new Uint32Array([0, 99]);
    const out = new Uint32Array(segmentData.length);
    expect(() => remapSegmentIndices(segmentData, ranges, out)).toThrow(
      'Vertex index 99 not found in loaded data'
    );
  });

  // Regression guard for issue #1049 at the REMAP stage: ranges spanning more
  // than 2^24 vertices. The old global → local `Map` set one entry per vertex,
  // so it threw `RangeError: Map maximum size exceeded` at the 2^24 + 1-th
  // entry. The prefix-offset table + binary search has no per-vertex structure,
  // so it must return 16_777_217 and remap correctly without throwing.
  // Only a few thousand SEGMENTS are needed — the point is that the RANGES
  // exceed 2^24, not that we materialize millions of segments.
  it('remaps ranges covering more than 2^24 vertices without throwing', () => {
    const ranges = [{ start: 0, end: 16_777_217 }]; // 2^24 + 1 vertices
    // Sample a few thousand global indices, including some past 2^24.
    const sampled = [
      0,
      1,
      100,
      16_776_000,
      16_777_215,
      16_777_216, // the 2^24-th index (0-based) — past the old Map cap
    ];
    const nSeg = 4000;
    const segmentData = new Uint32Array(nSeg);
    for (let i = 0; i < nSeg; i++) {
      segmentData[i] = sampled[i % sampled.length];
    }
    const out = new Uint32Array(nSeg);
    let count: number | undefined;
    expect(() => {
      count = remapSegmentIndices(segmentData, ranges, out);
    }).not.toThrow();
    expect(count).toBe(16_777_217);
    // With a single range starting at 0, local index === global index.
    for (let i = 0; i < nSeg; i++) {
      expect(out[i]).toBe(sampled[i % sampled.length]);
    }
  });
});
