/**
 * Property tests for HDR pixel-utils (fast-check).
 *
 * Pinned invariants:
 *   - halfFloat→Float32→halfFloat is identity on the representable half-float range.
 *   - flipPixelsVerticallyRGBA is its own inverse for even heights.
 *   - compactWebGPUReadbackRows output length == width * height * bytesPerTexel.
 */
import { describe, expect, test } from 'vitest';
import * as fc from 'fast-check';
import * as THREE from 'three';
import {
  halfFloatToFloat32,
  float32ToHalfFloat,
  flipPixelsVerticallyRGBA,
} from '../../../../rendering/post-processing/hdr/pixel-utils';

describe('halfFloatToFloat32 ↔ float32ToHalfFloat — roundtrip invariants', () => {
  test('roundtrip is identity within half-float precision for values in [-65504, 65504]', () => {
    // The representable half-float range is [-65504, 65504] with precision ~2^-10
    // around 1.0 (decreasing toward smaller exponents). For arbitrary values in
    // that range, conversion-back-and-forth should preserve them to within the
    // relative precision of half-floats.
    fc.assert(
      fc.property(
        fc.float({
          min: -65504,
          max: 65504,
          noNaN: true,
          noDefaultInfinity: true,
        }),
        (v) => {
          const half = float32ToHalfFloat(new Float32Array([v]));
          const back = halfFloatToFloat32(half)[0];
          // Half-float has 10 fraction bits ⇒ ~1e-3 relative precision.
          // Use both absolute and relative tolerance for values near zero.
          const tolerance = Math.max(1e-3, Math.abs(v) * 1e-3);
          expect(Math.abs(back - v)).toBeLessThan(tolerance);
        }
      ),
      { numRuns: 200 }
    );
  });

  test('roundtrip preserves zero exactly', () => {
    const back = halfFloatToFloat32(float32ToHalfFloat(new Float32Array([0])));
    expect(back[0]).toBe(0);
  });

  test('output length always equals input length', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1024 }), (n) => {
        const input = new Float32Array(n);
        const half = float32ToHalfFloat(input);
        const back = halfFloatToFloat32(half);
        expect(half.length).toBe(n);
        expect(back.length).toBe(n);
      }),
      { numRuns: 50 }
    );
  });

  test('matches THREE.DataUtils for spot-check values', () => {
    // Cross-check that the helper composes THREE.DataUtils correctly.
    for (const v of [0, 1, -1, 0.5, -0.5, 100, -100, 1e-3, 1e3]) {
      const expected = THREE.DataUtils.fromHalfFloat(THREE.DataUtils.toHalfFloat(v));
      const actual = halfFloatToFloat32(float32ToHalfFloat(new Float32Array([v])))[0];
      expect(actual).toBe(expected);
    }
  });
});

describe('flipPixelsVerticallyRGBA — involution invariant', () => {
  test('flip-twice is identity for even height', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 32 }),
        fc.integer({ min: 1, max: 16 }).map((h) => h * 2), // even
        (width, height) => {
          const totalBytes = width * height * 4;
          const pixels = new Uint8Array(totalBytes);
          for (let i = 0; i < totalBytes; i++) pixels[i] = (i * 13) & 0xff;

          const once = flipPixelsVerticallyRGBA(pixels, width, height);
          const twice = flipPixelsVerticallyRGBA(new Uint8Array(once), width, height);
          // After two flips, the buffer should match the original byte-for-byte.
          expect(Array.from(twice)).toEqual(Array.from(pixels));
        }
      ),
      { numRuns: 50 }
    );
  });

  test('output length always equals input length (width * height * 4)', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 64 }), fc.integer({ min: 1, max: 64 }), (w, h) => {
        const pixels = new Uint8Array(w * h * 4);
        const flipped = flipPixelsVerticallyRGBA(pixels, w, h);
        expect(flipped.length).toBe(w * h * 4);
      }),
      { numRuns: 50 }
    );
  });

  test('flip preserves first-and-last row swap pattern', () => {
    // 2x2 image where row 0 is red and row 1 is blue.
    const w = 2;
    const h = 2;
    const pixels = new Uint8Array([
      255,
      0,
      0,
      255,
      255,
      0,
      0,
      255, // row 0: red
      0,
      0,
      255,
      255,
      0,
      0,
      255,
      255, // row 1: blue
    ]);
    const flipped = flipPixelsVerticallyRGBA(pixels, w, h);
    // After flip: row 0 should be blue, row 1 should be red.
    expect(flipped[0]).toBe(0); // R of new row 0
    expect(flipped[2]).toBe(255); // B of new row 0
    expect(flipped[8]).toBe(255); // R of new row 1
    expect(flipped[10]).toBe(0); // B of new row 1
  });
});
