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
      1, 2, 3, 4, 5, 6, 7, 8, // y=0
      9, 10, 11, 12, 13, 14, 15, 16, // y=1
      17, 18, 19, 20, 21, 22, 23, 24, // y=2
    ]);
    const out = flipPixelsVerticallyRGBA(pixels, 2, 3);
    expect(Array.from(out)).toEqual([
      17, 18, 19, 20, 21, 22, 23, 24, // y=0 ← was y=2
      9, 10, 11, 12, 13, 14, 15, 16, // y=1 unchanged
      1, 2, 3, 4, 5, 6, 7, 8, // y=2 ← was y=0
    ]);
  });

  it('returns an all-zero buffer for zero-area inputs without throwing', () => {
    expect(flipPixelsVerticallyRGBA(new Uint8Array(0), 0, 0).length).toBe(0);
    expect(flipPixelsVerticallyRGBA(new Uint8Array(4), 1, 0).length).toBe(0);
  });
});
