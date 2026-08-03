/**
 * Range Extraction Tests for Encoded Arrays
 *
 * These tests verify the full encoded-array range extraction path:
 *   1. Load encoded zarr array
 *   2. Decode array (ArrayDecoder)
 *   3. Extract specific point ranges from decoded data
 *   4. Verify extracted values are correct
 *
 * This exercises the points loader path that extracts ranges from decoded
 * arrays, including LUT arrays where `elementsPerPoint` must match the
 * decoded attribute width.
 *
 * Test datasets from: packages/luxar-viewer/tests/fixtures/generate_test_data.py
 */

import { describe, it, expect } from 'vitest';
import { ArrayDecoder, ArrayRefRegistry } from '../../../../data/array-decoder/decoder';
import type { ArrayMetadata } from '../../../../data/array-decoder/decoder';
import type { PointRange } from '../../../../data/data-loader-types';
import * as zarr from '../../../../data/zarr';
import { FileSystemStore } from '@zarrita/storage';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURES_DIR = path.resolve(__dirname, '../../../../../tests/fixtures');

/**
 * Helper to load array with metadata from zarr
 */
async function loadArrayWithAttrs(
  datasetName: string,
  arrayPath: string
): Promise<{
  array: zarr.Array<zarr.DataType, zarr.Readable>;
  attrs: ArrayMetadata;
  rootLoc: zarr.Location<zarr.Readable>;
}> {
  const storePath = path.join(FIXTURES_DIR, datasetName);
  const rawStore = new FileSystemStore(storePath);
  const store = await zarr.openStore(rawStore);
  const rootLoc = zarr.root(store);
  const arrayLoc = rootLoc.resolve(arrayPath);
  const array = await zarr.open(arrayLoc, { kind: 'array' });
  const attrs = array.attrs as unknown as ArrayMetadata;
  return { array, attrs, rootLoc };
}

/**
 * Extract ranges from decoded array - mirrors the logic in points-spatial-index-loader.ts
 *
 * CRITICAL: This is the exact logic that had the LUT bug.
 * The bug was using elementsPerPoint instead of actualElementsPerPoint.
 *
 * @param decoded - Full decoded Float32Array
 * @param ranges - Point ranges to extract
 * @param actualElementsPerPoint - Elements per point in DECODED array (e.g., 3 for RGB)
 */
function extractRangesFromDecoded(
  decoded: Float32Array,
  ranges: PointRange[],
  actualElementsPerPoint: number
): Float32Array {
  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  const output = new Float32Array(totalPoints * actualElementsPerPoint);

  let destOffset = 0;
  for (const range of ranges) {
    const rangeSize = (range.end - range.start) * actualElementsPerPoint;
    const srcOffset = range.start * actualElementsPerPoint;
    output.set(decoded.subarray(srcOffset, srcOffset + rangeSize), destOffset);
    destOffset += rangeSize;
  }

  return output;
}

/**
 * Get actualElementsPerPoint - mirrors logic in points-spatial-index-loader.ts lines 632-635
 */
function getActualElementsPerPoint(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata
): number {
  const shape = array.shape;
  const elementsPerPoint = shape.length === 2 ? shape[1] : 1;
  const isEncoded = ArrayDecoder.isEncoded(attrs);

  // CRITICAL: For LUT-encoded arrays, original_shape[1] has the DECODED elements per point
  // For LUT: stored shape is [n, 1] (indices), original_shape is [n, 3] (RGB)
  return isEncoded && attrs.encoding?.original_shape?.[1]
    ? attrs.encoding.original_shape[1]
    : elementsPerPoint;
}

describe('Encoded Array Range Extraction', () => {
  describe('LUT Encoding - Critical Bug Fix', () => {
    it('should correctly extract ranges from LUT-encoded colors', async () => {
      // Load LUT-encoded colors (1000 points, 10 unique colors)
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      // Verify this is LUT encoding
      expect(attrs.encoding?.name).toBe('lut_uint8');
      expect(attrs.encoding?.lut).toBeDefined();

      // Get the correct elementsPerPoint
      const storedShape = array.shape;
      const storedElementsPerPoint = storedShape.length === 2 ? storedShape[1] : 1;
      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);

      // CRITICAL: For LUT, stored shape is [n, 1] but decoded is [n, 3]
      expect(storedElementsPerPoint).toBe(1); // Indices are 1 per point
      expect(actualElementsPerPoint).toBe(3); // Decoded RGB is 3 per point

      // Decode full array
      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      // Verify decoded size
      expect(decoded.length).toBe(1000 * 3); // 1000 points × 3 RGB = 3000

      // Test range extraction with multiple ranges
      const ranges: PointRange[] = [
        { start: 0, end: 10 }, // First 10 points
        { start: 500, end: 510 }, // Middle 10 points
        { start: 990, end: 1000 }, // Last 10 points
      ];

      // Extract ranges using CORRECT actualElementsPerPoint
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      // Verify extracted size: 30 points × 3 RGB = 90 elements
      expect(extracted.length).toBe(30 * 3);

      // Verify extracted values match decoded values
      // Range 0-10: points 0-9
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          expect(extracted[i * 3 + j]).toBeCloseTo(decoded[i * 3 + j], 5);
        }
      }

      // Range 500-510: points 500-509
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          const extractedIdx = (10 + i) * 3 + j; // Offset by first range
          const decodedIdx = (500 + i) * 3 + j;
          expect(extracted[extractedIdx]).toBeCloseTo(decoded[decodedIdx], 5);
        }
      }

      // Range 990-1000: points 990-999
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          const extractedIdx = (20 + i) * 3 + j; // Offset by first two ranges
          const decodedIdx = (990 + i) * 3 + j;
          expect(extracted[extractedIdx]).toBeCloseTo(decoded[decodedIdx], 5);
        }
      }
    });

    it('extractRangesFromDecoded helper: WRONG elementsPerPoint produces under-sized output (helper-arithmetic pin)', async () => {
      // data.md C1 fix: previous name "should fail with WRONG elementsPerPoint
      // (demonstrates the bug)" misleadingly implies this tests the
      // production loader. It does not. `extractRangesFromDecoded` is a
      // LOCAL helper defined in this test file (see top), not the
      // production code path. The real production loader's getActualElementsPerPoint
      // path is exercised by the surrounding "should correctly extract
      // ranges from …" tests. This test only verifies the helper's
      // arithmetic — useful as documentation of the off-by-three bug
      // shape but does NOT guard the production fix.
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      const ranges: PointRange[] = [{ start: 0, end: 10 }];

      const wrongElementsPerPoint = 1;
      const buggyExtracted = extractRangesFromDecoded(decoded, ranges, wrongElementsPerPoint);

      // Helper arithmetic: 10 points × 1 = 10 (would have been 30 with correct epp).
      expect(buggyExtracted.length).toBe(10);
    });
  });

  describe('Broadcasting Encoding', () => {
    it('should correctly extract ranges from broadcasted colors', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_broadcasting.luxar.zarr',
        'points/colors'
      );

      expect(attrs.encoding?.name).toBe('broadcasted');

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);

      // Decode with expected element count
      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      expect(decoded.length).toBe(1000 * 3);

      // Extract ranges
      const ranges: PointRange[] = [
        { start: 0, end: 100 },
        { start: 500, end: 600 },
      ];

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(200 * 3);

      // All colors should be the same (uniform)
      const firstColor = [extracted[0], extracted[1], extracted[2]];
      for (let i = 0; i < 200; i++) {
        expect(extracted[i * 3]).toBeCloseTo(firstColor[0], 5);
        expect(extracted[i * 3 + 1]).toBeCloseTo(firstColor[1], 5);
        expect(extracted[i * 3 + 2]).toBeCloseTo(firstColor[2], 5);
      }
    });

    it('should correctly extract ranges from broadcasted radii', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_broadcasting.luxar.zarr',
        'points/radii'
      );

      expect(attrs.encoding?.name).toBe('broadcasted');

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(1); // Radii are scalar

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      expect(decoded.length).toBe(1000);

      const ranges: PointRange[] = [{ start: 100, end: 200 }];

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(100);

      // All radii should be the same (uniform)
      const firstRadius = extracted[0];
      for (let i = 0; i < 100; i++) {
        expect(extracted[i]).toBeCloseTo(firstRadius, 5);
      }
    });
  });

  describe('Quantization Encoding (rgb_uint8)', () => {
    it('should correctly extract ranges from quantized colors', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_quantization.luxar.zarr',
        'points/colors'
      );

      // Verify this is quantized encoding
      expect(attrs.encoding?.name).toBe('rgb_uint8');

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(3);

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      expect(decoded.length).toBe(1000 * 3);

      // Extract specific ranges
      const ranges: PointRange[] = [
        { start: 50, end: 100 },
        { start: 750, end: 800 },
      ];

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(100 * 3);

      // Verify values are in valid range [0, 1] after dequantization
      for (let i = 0; i < extracted.length; i++) {
        expect(extracted[i]).toBeGreaterThanOrEqual(0);
        expect(extracted[i]).toBeLessThanOrEqual(1);
      }

      // Verify extracted matches decoded at correct offsets
      for (let i = 0; i < 50; i++) {
        for (let j = 0; j < 3; j++) {
          const extractedIdx = i * 3 + j;
          const decodedIdx = (50 + i) * 3 + j; // First range starts at point 50
          expect(extracted[extractedIdx]).toBeCloseTo(decoded[decodedIdx], 5);
        }
      }
    });

    it('should correctly extract ranges from radii (may be float32 or quantized)', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_quantization.luxar.zarr',
        'points/radii'
      );

      // Radii may use float32 or bounded scalar quantization depending on encoder settings
      // The important thing is that range extraction works correctly
      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(1);

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      expect(decoded.length).toBe(1000);

      const ranges: PointRange[] = [{ start: 0, end: 500 }];

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(500);

      // Radii should be in expected range [0.1, 2.0] from generate_test_data.py
      for (let i = 0; i < extracted.length; i++) {
        expect(extracted[i]).toBeGreaterThanOrEqual(0.09); // Slight tolerance
        expect(extracted[i]).toBeLessThanOrEqual(2.01);
      }
    });
  });

  describe('Mixed Encoding Modes', () => {
    it('should correctly extract from multiple encoding modes in same scene', async () => {
      // Test uniform (broadcasting)
      const { array: uniformArray, attrs: uniformAttrs } = await loadArrayWithAttrs(
        'test_mixed.luxar.zarr',
        'uniform/colors'
      );
      expect(uniformAttrs.encoding?.name).toBe('broadcasted');

      // Test LUT
      const { array: lutArray, attrs: lutAttrs } = await loadArrayWithAttrs(
        'test_mixed.luxar.zarr',
        'lut/colors'
      );
      expect(lutAttrs.encoding?.name).toBe('lut_uint8');

      // Decode both
      const decoder = new ArrayDecoder(new ArrayRefRegistry());

      const uniformDecoded = await decoder.decode(uniformArray, uniformAttrs, 500);
      const lutDecoded = await decoder.decode(lutArray, lutAttrs, 500);

      expect(uniformDecoded.length).toBe(500 * 3);
      expect(lutDecoded.length).toBe(500 * 3);

      // Extract same ranges from both
      const ranges: PointRange[] = [{ start: 100, end: 200 }];

      const uniformActual = getActualElementsPerPoint(uniformArray, uniformAttrs);
      const lutActual = getActualElementsPerPoint(lutArray, lutAttrs);

      const uniformExtracted = extractRangesFromDecoded(uniformDecoded, ranges, uniformActual);
      const lutExtracted = extractRangesFromDecoded(lutDecoded, ranges, lutActual);

      expect(uniformExtracted.length).toBe(100 * 3);
      expect(lutExtracted.length).toBe(100 * 3);
    });
  });

  describe('Direct (Non-Encoded) Arrays - Positions', () => {
    it('extracts ranges from decoded per-channel positions with data verification', async () => {
      const { array, attrs } = await loadArrayWithAttrs('test_lut.luxar.zarr', 'points/positions');

      // Positions are per-channel encoded (uint16 fixed-point) → decode to float32.
      expect(ArrayDecoder.isEncoded(attrs)).toBe(true);

      const shape = array.shape;
      const elementsPerPoint = shape.length === 2 ? shape[1] : 1;
      expect(elementsPerPoint).toBe(3);

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(3);

      // Decode the full positions to float32, then extract ranges from the decoded
      // buffer (self-consistent: extraction must reproduce the decoded values).
      const fullPositions = await new ArrayDecoder(new ArrayRefRegistry()).decode(array, attrs);
      expect(fullPositions.length).toBe(1000 * 3);

      // Independent ground truth (not decoded-vs-decoded): hand-compute the
      // per-channel inverse x = lo[c] + level/levels·(hi[c]-lo[c]) from the raw
      // stored levels + encoding metadata, so a decode bug (wrong column index,
      // dropped -min offset, raw levels returned) fails here rather than
      // cancelling out of the extraction comparison below.
      const enc = attrs.encoding as unknown as {
        name: string;
        bits: number;
        col_lo: number[];
        col_hi: number[];
      };
      expect(enc.name).toBe('linear_perchannel_u16');
      const rawChunk = await zarr.get(array);
      const rawLevels = rawChunk.data as Uint16Array;
      const levels = (1 << enc.bits) - 1;
      for (const i of [0, 1, 499, 500, 998, 999]) {
        for (let c = 0; c < 3; c++) {
          const rng = Math.max(enc.col_hi[c] - enc.col_lo[c], 1e-30);
          const expected = enc.col_lo[c] + (Number(rawLevels[i * 3 + c]) / levels) * rng;
          expect(fullPositions[i * 3 + c]).toBeCloseTo(expected, 5);
        }
      }

      // Test range extraction on raw float32 positions
      const ranges: PointRange[] = [
        { start: 0, end: 10 },
        { start: 500, end: 510 },
        { start: 990, end: 1000 },
      ];

      const extracted = extractRangesFromDecoded(fullPositions, ranges, actualElementsPerPoint);

      // Verify extracted size: 30 points × 3 XYZ = 90 elements
      expect(extracted.length).toBe(30 * 3);

      // Verify extracted values match original positions
      // Range 0-10
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          expect(extracted[i * 3 + j]).toBeCloseTo(fullPositions[i * 3 + j], 5);
        }
      }

      // Range 500-510
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          const extractedIdx = (10 + i) * 3 + j;
          const originalIdx = (500 + i) * 3 + j;
          expect(extracted[extractedIdx]).toBeCloseTo(fullPositions[originalIdx], 5);
        }
      }

      // Range 990-1000
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 3; j++) {
          const extractedIdx = (20 + i) * 3 + j;
          const originalIdx = (990 + i) * 3 + j;
          expect(extracted[extractedIdx]).toBeCloseTo(fullPositions[originalIdx], 5);
        }
      }
    });

    it('rejects per-channel scales whose length disagrees with the array width', async () => {
      // Regression: the full-array decode used to derive the column count from
      // col_lo.length itself, making makePerChannelDequant's length guard a
      // tautology — a corrupt file with 2 scales on an (N, 3) array silently
      // decoded with col = i % 2, misaligning every axis. The column count now
      // comes from the array's own last dimension (like Python's
      // `_perchannel_scales`), so the mismatch fails loud.
      const { array, attrs } = await loadArrayWithAttrs('test_lut.luxar.zarr', 'points/positions');
      const enc = attrs.encoding as unknown as { col_lo: number[]; col_hi: number[] };
      const corrupt = {
        ...attrs,
        encoding: {
          ...attrs.encoding,
          col_lo: enc.col_lo.slice(0, 2),
          col_hi: enc.col_hi.slice(0, 2),
        },
      } as unknown as ArrayMetadata;
      await expect(new ArrayDecoder(new ArrayRefRegistry()).decode(array, corrupt)).rejects.toThrow(
        /col_lo\/col_hi must each have 3 entries/
      );
    });

    it('extracts ranges from decoded 4D per-channel positions correctly', async () => {
      const { array, attrs } = await loadArrayWithAttrs('test_4d.luxar.zarr', 'points/positions');

      // 4D positions are per-channel encoded (uint16 fixed-point) → decode to float32.
      expect(ArrayDecoder.isEncoded(attrs)).toBe(true);

      const shape = array.shape;
      const elementsPerPoint = shape.length === 2 ? shape[1] : 1;
      expect(elementsPerPoint).toBe(4); // time, x, y, z

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(4);

      const totalPoints = shape[0];
      const fullPositions = await new ArrayDecoder(new ArrayRefRegistry()).decode(array, attrs);

      // Verify we got totalPoints × 4 elements
      expect(fullPositions.length).toBe(totalPoints * 4);

      // Test range extraction on first 100 points
      const ranges: PointRange[] = [{ start: 0, end: 100 }];
      const extracted = extractRangesFromDecoded(fullPositions, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(100 * 4); // 100 points × 4D

      // Verify extraction matches
      for (let i = 0; i < 100; i++) {
        for (let j = 0; j < 4; j++) {
          expect(extracted[i * 4 + j]).toBeCloseTo(fullPositions[i * 4 + j], 5);
        }
      }
    });
  });

  describe('Array Reference (Deduplication) Encoding', () => {
    it('should correctly extract ranges from array_ref encoded colors', async () => {
      // The test_array_refs.luxar.zarr has two point clouds with shared colors
      // Second one should have array_ref encoding pointing to first

      // First, check points1 colors
      const {
        array: colors1,
        attrs: attrs1,
        rootLoc,
      } = await loadArrayWithAttrs('test_array_refs.luxar.zarr', 'points1/colors');

      // Second, check points2 colors (may have array_ref)
      const { array: colors2, attrs: attrs2 } = await loadArrayWithAttrs(
        'test_array_refs.luxar.zarr',
        'points2/colors'
      );

      // Decode both
      const decoder = new ArrayDecoder(new ArrayRefRegistry());

      // Decode points1 colors first (this populates the registry)
      const decoded1 = await decoder.decode(colors1, attrs1, 500, rootLoc);
      expect(decoded1.length).toBe(500 * 3);

      // Decode points2 colors (may use array_ref to reuse data)
      const decoded2 = await decoder.decode(colors2, attrs2, 500, rootLoc);
      expect(decoded2.length).toBe(500 * 3);

      // Both should have valid RGB data in [0, 1] range
      for (let i = 0; i < decoded1.length; i++) {
        expect(decoded1[i]).toBeGreaterThanOrEqual(0);
        expect(decoded1[i]).toBeLessThanOrEqual(1);
      }

      for (let i = 0; i < decoded2.length; i++) {
        expect(decoded2[i]).toBeGreaterThanOrEqual(0);
        expect(decoded2[i]).toBeLessThanOrEqual(1);
      }

      // Test range extraction on both
      const ranges: PointRange[] = [
        { start: 0, end: 50 },
        { start: 200, end: 250 },
      ];

      const actual1 = getActualElementsPerPoint(colors1, attrs1);
      const actual2 = getActualElementsPerPoint(colors2, attrs2);

      const extracted1 = extractRangesFromDecoded(decoded1, ranges, actual1);
      const extracted2 = extractRangesFromDecoded(decoded2, ranges, actual2);

      expect(extracted1.length).toBe(100 * 3);
      expect(extracted2.length).toBe(100 * 3);

      // The two source colors arrays are constructed to be IDENTICAL in
      // values (the array_ref fixture defines them that way), so array_ref
      // working correctly means decoded2 === decoded1 byte-for-byte, and
      // therefore extracted2 === extracted1. The previous comment
      // ("not necessarily same values - depends on if arrays were truly
      // identical") punted the load-bearing assertion — data.md C6 fix.
      expect(Array.from(decoded2)).toEqual(Array.from(decoded1));
      expect(Array.from(extracted2)).toEqual(Array.from(extracted1));
    });
  });

  describe('Sharpness Encoding', () => {
    it('should correctly extract ranges from quantized sharpness', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_sharpness_range.luxar.zarr',
        'sharpness_test/sharpnesses'
      );

      // Sharpness uses bounded_scalar_uint8 encoding
      expect(attrs.encoding?.name).toBe('bounded_scalar_uint8');

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(1); // Sharpness is scalar

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 32, rootLoc);

      expect(decoded.length).toBe(32);

      // Sharpness is a normalized [0, 1] knob (fixture = linspace(0, 1, 32)).
      for (let i = 0; i < decoded.length; i++) {
        expect(decoded[i]).toBeGreaterThanOrEqual(-0.01); // small quantization tolerance
        expect(decoded[i]).toBeLessThanOrEqual(1.01);
      }

      // Test range extraction
      const ranges: PointRange[] = [
        { start: 0, end: 10 },
        { start: 20, end: 32 },
      ];

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(22); // 10 + 12 points

      // Verify extracted matches decoded
      for (let i = 0; i < 10; i++) {
        expect(extracted[i]).toBeCloseTo(decoded[i], 4);
      }
      for (let i = 0; i < 12; i++) {
        expect(extracted[10 + i]).toBeCloseTo(decoded[20 + i], 4);
      }
    });

    it('should verify the full sharpness range [0, 1] is preserved after decoding', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_sharpness_range.luxar.zarr',
        'sharpness_test/sharpnesses'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 32, rootLoc);

      // Sharpness is a normalized [0, 1] knob; fixture = linspace(0, 1, 32).
      // Spatial ordering may reorder points, so check sorted values.
      const sorted = Array.from(decoded).sort((a, b) => a - b);

      // Smallest value should be close to 0.0
      expect(sorted[0]).toBeGreaterThanOrEqual(-0.01);
      expect(sorted[0]).toBeLessThanOrEqual(0.05);

      // Largest value should be close to 1.0
      expect(sorted[31]).toBeGreaterThanOrEqual(0.95);
      expect(sorted[31]).toBeLessThanOrEqual(1.01);

      // Middle value should be near 0.5
      expect(sorted[16]).toBeGreaterThanOrEqual(0.45);
      expect(sorted[16]).toBeLessThanOrEqual(0.6);
    });
  });

  /**
   * CRITICAL TEST: Scalar LUT on 4D Positions (Quantum Orbitals Bug)
   *
   * This test catches the exact bug found in quantum orbitals demo:
   * - 4D positions stored as scalar LUT (one index per coordinate)
   * - original_shape must be [n_points, 4] to get actualElementsPerPoint = 4
   * - Partial range extraction must use correct offsets
   *
   * Without this test, a bug where actualElementsPerPoint = 1 (from indices shape)
   * instead of 4 (from original_shape) would go undetected.
   */
  describe('Scalar LUT on 4D Positions (Quantum Orbitals Bug)', () => {
    it('should correctly identify scalar LUT mode and original_shape for 4D positions', async () => {
      const { array, attrs } = await loadArrayWithAttrs(
        'test_4d_scalar_lut.luxar.zarr',
        'points/positions'
      );

      // Verify encoding metadata
      expect(ArrayDecoder.isEncoded(attrs)).toBe(true);
      expect(attrs.encoding?.name).toBe('lut_uint8');
      expect(attrs.encoding?.lut_mode).toBe('scalar');

      // CRITICAL: original_shape must be [200, 4] for 4D positions
      expect(attrs.encoding?.original_shape).toEqual([200, 4]);

      // Indices array shape should also be [200, 4] in scalar mode
      expect(array.shape).toEqual([200, 4]);

      // actualElementsPerPoint should be 4 (from original_shape[1])
      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      expect(actualElementsPerPoint).toBe(4);
    });

    it('should correctly decode full 4D positions array with scalar LUT', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_4d_scalar_lut.luxar.zarr',
        'points/positions'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 200 * 4, rootLoc);

      // Decoded should have 200 points × 4 dims = 800 elements
      expect(decoded.length).toBe(800);

      // First 10 points have predictable values from fixture generator:
      // Point i: [i, i*2, i*0.5, i%5]

      // Point 0: [0, 0, 0, 0]
      expect(decoded[0 * 4 + 0]).toBeCloseTo(0.0, 4); // dim 0
      expect(decoded[0 * 4 + 1]).toBeCloseTo(0.0, 4); // dim 1
      expect(decoded[0 * 4 + 2]).toBeCloseTo(0.0, 4); // dim 2
      expect(decoded[0 * 4 + 3]).toBeCloseTo(0.0, 4); // dim 3

      // Point 5: [5, 10, 2.5, 0]
      expect(decoded[5 * 4 + 0]).toBeCloseTo(5.0, 4);
      expect(decoded[5 * 4 + 1]).toBeCloseTo(10.0, 4);
      expect(decoded[5 * 4 + 2]).toBeCloseTo(2.5, 4);
      expect(decoded[5 * 4 + 3]).toBeCloseTo(0.0, 4);

      // Point 9: [9, 18, 4.5, 4]
      expect(decoded[9 * 4 + 0]).toBeCloseTo(9.0, 4);
      expect(decoded[9 * 4 + 1]).toBeCloseTo(18.0, 4);
      expect(decoded[9 * 4 + 2]).toBeCloseTo(4.5, 4);
      expect(decoded[9 * 4 + 3]).toBeCloseTo(4.0, 4);
    });

    it('should correctly extract partial range from scalar LUT 4D positions', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_4d_scalar_lut.luxar.zarr',
        'points/positions'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 200 * 4, rootLoc);

      // Extract points 5-10 (5 points)
      const ranges: PointRange[] = [{ start: 5, end: 10 }];
      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);

      // CRITICAL: This tests the bug fix!
      // If actualElementsPerPoint was incorrectly 1 (from indices shape),
      // we'd get 5 elements instead of 20.
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      // Should get 5 points × 4 dims = 20 elements
      expect(extracted.length).toBe(20);

      // Verify first point in range (point 5): [5, 10, 2.5, 0]
      expect(extracted[0]).toBeCloseTo(5.0, 4);
      expect(extracted[1]).toBeCloseTo(10.0, 4);
      expect(extracted[2]).toBeCloseTo(2.5, 4);
      expect(extracted[3]).toBeCloseTo(0.0, 4);

      // Verify last point in range (point 9): [9, 18, 4.5, 4]
      expect(extracted[16]).toBeCloseTo(9.0, 4);
      expect(extracted[17]).toBeCloseTo(18.0, 4);
      expect(extracted[18]).toBeCloseTo(4.5, 4);
      expect(extracted[19]).toBeCloseTo(4.0, 4);
    });

    it('should correctly extract multiple non-contiguous ranges from scalar LUT 4D positions', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_4d_scalar_lut.luxar.zarr',
        'points/positions'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 200 * 4, rootLoc);

      // Extract points 0-3 and 7-10 (non-contiguous)
      const ranges: PointRange[] = [
        { start: 0, end: 3 }, // 3 points
        { start: 7, end: 10 }, // 3 points
      ];
      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);

      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      // Should get 6 points × 4 dims = 24 elements
      expect(extracted.length).toBe(24);

      // First range: points 0-2
      // Point 0: [0, 0, 0, 0]
      expect(extracted[0]).toBeCloseTo(0.0, 4);
      expect(extracted[1]).toBeCloseTo(0.0, 4);
      expect(extracted[2]).toBeCloseTo(0.0, 4);
      expect(extracted[3]).toBeCloseTo(0.0, 4);

      // Point 2: [2, 4, 1, 2]
      expect(extracted[8]).toBeCloseTo(2.0, 4);
      expect(extracted[9]).toBeCloseTo(4.0, 4);
      expect(extracted[10]).toBeCloseTo(1.0, 4);
      expect(extracted[11]).toBeCloseTo(2.0, 4);

      // Second range starts at extracted[12]: point 7
      // Point 7: [7, 14, 3.5, 2]
      expect(extracted[12]).toBeCloseTo(7.0, 4);
      expect(extracted[13]).toBeCloseTo(14.0, 4);
      expect(extracted[14]).toBeCloseTo(3.5, 4);
      expect(extracted[15]).toBeCloseTo(2.0, 4);
    });
  });

  describe('Edge Cases', () => {
    it('should handle single-point ranges', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      const ranges: PointRange[] = [{ start: 500, end: 501 }]; // Single point

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(3); // 1 point × 3 RGB

      expect(extracted[0]).toBeCloseTo(decoded[500 * 3], 5);
      expect(extracted[1]).toBeCloseTo(decoded[500 * 3 + 1], 5);
      expect(extracted[2]).toBeCloseTo(decoded[500 * 3 + 2], 5);
    });

    it('should handle empty ranges', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      const ranges: PointRange[] = []; // No ranges

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(0);
    });

    it('should handle full array as single range', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      const ranges: PointRange[] = [{ start: 0, end: 1000 }]; // Full array

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(decoded.length);

      // Should be identical to decoded
      for (let i = 0; i < extracted.length; i++) {
        expect(extracted[i]).toBeCloseTo(decoded[i], 5);
      }
    });

    it('should handle many small ranges', async () => {
      const { array, attrs, rootLoc } = await loadArrayWithAttrs(
        'test_lut.luxar.zarr',
        'points/colors'
      );

      const decoder = new ArrayDecoder(new ArrayRefRegistry());
      const decoded = await decoder.decode(array, attrs, 1000, rootLoc);

      // 100 ranges of 5 points each
      const ranges: PointRange[] = [];
      for (let i = 0; i < 100; i++) {
        ranges.push({ start: i * 10, end: i * 10 + 5 });
      }

      const actualElementsPerPoint = getActualElementsPerPoint(array, attrs);
      const extracted = extractRangesFromDecoded(decoded, ranges, actualElementsPerPoint);

      expect(extracted.length).toBe(500 * 3); // 100 ranges × 5 points × 3 RGB
    });
  });
});
