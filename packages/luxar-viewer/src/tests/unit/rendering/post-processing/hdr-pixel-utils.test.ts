/**
 * Unit tests for the post-processing HDR pixel utilities.
 *
 * Pure conversions over typed arrays — no GL context needed; we feed
 * known values and check round-trips and orientation.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  halfFloatToFloat32,
  float32ToHalfFloat,
  flipPixelsVerticallyRGBA,
  compactWebGPUReadbackRows,
} from '../../../../rendering/post-processing/hdr-pixel-utils';

describe('halfFloatToFloat32', () => {
  it('returns an empty Float32Array for empty input', () => {
    const out = halfFloatToFloat32(new Uint16Array(0));
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(0);
  });

  it('preserves the array length', () => {
    expect(halfFloatToFloat32(new Uint16Array(5)).length).toBe(5);
  });

  it('decodes half-float bit patterns to expected float values', () => {
    // Encode a few canonical floats via THREE, then decode and compare.
    const source = new Float32Array([0, 1, -1, 0.5, -0.25, 65504]);
    const encoded = new Uint16Array(source.length);
    for (let i = 0; i < source.length; i++) {
      encoded[i] = THREE.DataUtils.toHalfFloat(source[i]);
    }
    const decoded = halfFloatToFloat32(encoded);
    for (let i = 0; i < source.length; i++) {
      // Half-float has limited precision; allow small rounding error.
      expect(decoded[i]).toBeCloseTo(source[i], 1);
    }
  });
});

describe('float32ToHalfFloat', () => {
  it('returns an empty Uint16Array for empty input', () => {
    const out = float32ToHalfFloat(new Float32Array(0));
    expect(out).toBeInstanceOf(Uint16Array);
    expect(out.length).toBe(0);
  });

  it('preserves the array length', () => {
    expect(float32ToHalfFloat(new Float32Array(7)).length).toBe(7);
  });

  it('round-trips losslessly for half-float-representable values', () => {
    const source = new Float32Array([0, 1, -1, 0.5, -0.25, 16, -2048]);
    const encoded = float32ToHalfFloat(source);
    const decoded = halfFloatToFloat32(encoded);
    for (let i = 0; i < source.length; i++) {
      // Each chosen value is exactly representable in half-float.
      expect(decoded[i]).toBe(source[i]);
    }
  });
});

describe('flipPixelsVerticallyRGBA', () => {
  it('returns a Uint8ClampedArray of the same total size', () => {
    const pixels = new Uint8Array(2 * 3 * 4); // 2x3 RGBA
    const out = flipPixelsVerticallyRGBA(pixels, 2, 3);
    expect(out).toBeInstanceOf(Uint8ClampedArray);
    expect(out.length).toBe(2 * 3 * 4);
  });

  it('flips a 1×2 image (just two rows swap)', () => {
    // 1x2 RGBA: rows = [10,20,30,40] (y=0, bottom) and [50,60,70,80] (y=1, top).
    // After flip, y=0 should hold the top row and y=1 should hold the bottom row.
    const pixels = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const out = flipPixelsVerticallyRGBA(pixels, 1, 2);
    expect(Array.from(out)).toEqual([50, 60, 70, 80, 10, 20, 30, 40]);
  });

  it('flips a 2×3 image — rows reorder, RGBA byte order within each row preserved', () => {
    // 2 wide × 3 tall, 4 components per pixel. Rows are 8 bytes each.
    // y=0 row: [1..8], y=1 row: [9..16], y=2 row: [17..24].
    const pixels = new Uint8Array([
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8, // y=0
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16, // y=1
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24, // y=2
    ]);
    const out = flipPixelsVerticallyRGBA(pixels, 2, 3);
    expect(Array.from(out)).toEqual([
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24, // y=0 ← was y=2
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16, // y=1 unchanged
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8, // y=2 ← was y=0
    ]);
  });

  it('returns an all-zero buffer for zero-area inputs without throwing', () => {
    expect(flipPixelsVerticallyRGBA(new Uint8Array(0), 0, 0).length).toBe(0);
    expect(flipPixelsVerticallyRGBA(new Uint8Array(4), 1, 0).length).toBe(0);
  });
});

describe('compactWebGPUReadbackRows', () => {
  // WebGPU pads `bytesPerRow` to a multiple of 256. The helper drops
  // that padding so callers see a compact row layout regardless of the
  // backend. Widths whose `width * bytesPerTexel` is already a multiple
  // of 256 should take the no-op fast path (return the input array
  // unchanged).

  it('returns the input unchanged when width × bytesPerTexel is already 256-aligned (RGBA8 width=64)', () => {
    // 64 × 4 = 256 → no padding required.
    const raw = new Uint8Array(64 * 1 * 4);
    raw[0] = 7;
    raw[raw.length - 1] = 9;
    const out = compactWebGPUReadbackRows(raw, 64, 1, 4);
    expect(out).toBe(raw);
  });

  it('compacts a 5×5 RGBA32F readback (80 B/row → 256 B padded)', () => {
    // Mirrors picking-system.ts: PICK_SIZE=5, bytesPerTexel=16.
    const elementsPerRowReal = 5 * 4; // 20 floats
    const elementsPerRowPadded = 256 / 4; // 64 floats
    const raw = new Float32Array(elementsPerRowPadded * 5);
    // Fill the compact region with row-index codes so we can verify
    // ordering after compaction. Padding columns get a sentinel.
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] =
          col < elementsPerRowReal ? row * 100 + col : -999;
      }
    }
    const out = compactWebGPUReadbackRows(raw, 5, 5, 16);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out).not.toBe(raw);
    expect(out.length).toBe(elementsPerRowReal * 5);
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        expect(out[row * elementsPerRowReal + col]).toBe(row * 100 + col);
      }
    }
    // The sentinel padding values must NOT leak into the compact output.
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).not.toBe(-999);
    }
  });

  it('compacts a 853×2 RGBA8 readback (3412 B/row → 3584 B padded)', () => {
    // 853 × 4 = 3412; ceil(3412 / 256) * 256 = 3584; 172 bytes of pad.
    const width = 853;
    const height = 2;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 3584;
    const raw = new Uint8Array(elementsPerRowPadded * height);
    // Encode (row << 24 | col) into the compact region; padding gets 0xFF.
    raw.fill(0xff);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        raw[row * elementsPerRowPadded + col] = (row * 13 + col) & 0xff;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 4);
    expect(out.length).toBe(elementsPerRowReal * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        expect(out[row * elementsPerRowReal + col]).toBe((row * 13 + col) & 0xff);
      }
    }
  });

  it('compacts a 853×2 RGBA16F readback (6824 B/row → 6912 B padded)', () => {
    // 853 × 8 = 6824; ceil(6824 / 256) * 256 = 6912; 88 bytes of pad
    // = 44 Uint16 entries.
    const width = 853;
    const height = 2;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 6912 / 2; // Uint16
    const raw = new Uint16Array(elementsPerRowPadded * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] = col < elementsPerRowReal ? row * 1000 + col : 0;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 8);
    expect(out).toBeInstanceOf(Uint16Array);
    expect(out.length).toBe(elementsPerRowReal * height);
    expect(out[0]).toBe(0);
    expect(out[elementsPerRowReal - 1]).toBe(elementsPerRowReal - 1);
    expect(out[elementsPerRowReal]).toBe(1000); // first byte of row 1
  });

  it('compacts a 853×3 RGBA32F readback (13648 B/row → 13824 B padded)', () => {
    // 853 × 16 = 13648; ceil(13648 / 256) * 256 = 13824; 176 bytes of pad
    // = 44 Float32 entries.
    const width = 853;
    const height = 3;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 13824 / 4;
    const raw = new Float32Array(elementsPerRowPadded * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] =
          col < elementsPerRowReal ? row + col * 0.001 : Number.NaN;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 16);
    expect(out.length).toBe(elementsPerRowReal * height);
    // No NaN sentinels should have leaked from the padding band.
    for (let i = 0; i < out.length; i++) {
      expect(Number.isNaN(out[i])).toBe(false);
    }
    // Spot-check a few entries.
    expect(out[0]).toBeCloseTo(0, 4);
    expect(out[elementsPerRowReal - 1]).toBeCloseTo((elementsPerRowReal - 1) * 0.001, 4);
    expect(out[elementsPerRowReal]).toBeCloseTo(1, 4); // first entry of row 1
  });

  it('returns input unchanged when length already matches compact (WebGL2 caller path)', () => {
    // The WebGL2 backend returns a compact array even for unaligned
    // widths. The helper must detect this and short-circuit.
    const width = 5;
    const height = 5;
    const raw = new Float32Array(width * height * 4); // compact RGBA32F
    const out = compactWebGPUReadbackRows(raw, width, height, 16);
    expect(out).toBe(raw);
  });
});
