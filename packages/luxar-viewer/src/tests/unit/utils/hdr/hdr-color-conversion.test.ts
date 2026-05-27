/**
 * Unit tests for HDR color conversion utilities.
 *
 * Tests the RGB → BT.2020 → PQ → YCbCr → I420P10 pipeline
 * used for 10-bit HDR video encoding.
 */

import { describe, it, expect } from 'vitest';
import { rgbaFloatToI420P10 } from '../../../../utils/hdr/hdr-color-conversion';

describe('rgbaFloatToI420P10', () => {
  // utils.md O3 / Phase E43: P9 rename — pins the I420P10 packed
  // buffer size: Y plane (w·h) + U plane (w/2·h/2) + V plane (w/2·h/2).
  it('emits a Uint16Array sized w·h + 2·(w/2·h/2) (Y + U + V planes packed)', () => {
    const width = 4;
    const height = 4;
    const rgba = new Float32Array(width * height * 4);
    const result = rgbaFloatToI420P10(rgba, width, height);

    // I420P10: Y(w*h) + U(w/2*h/2) + V(w/2*h/2) = 4*4 + 2*2 + 2*2 = 24
    const ySize = width * height;
    const uvSize = (width >> 1) * (height >> 1);
    expect(result.length).toBe(ySize + 2 * uvSize);
  });

  // utils.md O3 / Phase E43: P9 rename — pins the Uint16Array dtype
  // (10-bit values packed into 16-bit lanes).
  it('returns a Uint16Array (10-bit Y/U/V values in 16-bit lanes)', () => {
    const rgba = new Float32Array(4 * 4 * 4);
    const result = rgbaFloatToI420P10(rgba, 4, 4);
    expect(result).toBeInstanceOf(Uint16Array);
  });

  // utils.md O3 / Phase E43: P9 rename — pins the BT.709 limited-range
  // mapping for pure-black input: Y=64 (limited-range min), Cb/Cr=512
  // (achromatic center, 10-bit).
  it('all-black RGBA emits limited-range Y=64 and achromatic Cb/Cr=512', () => {
    // All zeros = black
    const rgba = new Float32Array(4 * 4 * 4); // All zeros
    const result = rgbaFloatToI420P10(rgba, 4, 4);

    // Y for black should be at limited-range min (~64)
    expect(result[0]).toBe(64);

    // Cb/Cr for black should be at achromatic center (~512)
    const ySize = 4 * 4;
    expect(result[ySize]).toBe(512); // U
    expect(result[ySize + 2 * 2]).toBe(512); // V
  });

  // utils.md O3 / Phase E46: P9 rename — Y luma scales monotonically
  // with input intensity (brighter RGB → larger Y after PQ + matrix).
  it('Y luma increases monotonically with input intensity (bright RGB > dark RGB)', () => {
    const width = 2;
    const height = 2;

    // Dark pixels
    const darkRgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      darkRgba[i * 4] = 0.1;
      darkRgba[i * 4 + 1] = 0.1;
      darkRgba[i * 4 + 2] = 0.1;
      darkRgba[i * 4 + 3] = 1.0;
    }

    // Bright pixels
    const brightRgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      brightRgba[i * 4] = 1.0;
      brightRgba[i * 4 + 1] = 1.0;
      brightRgba[i * 4 + 2] = 1.0;
      brightRgba[i * 4 + 3] = 1.0;
    }

    const darkResult = rgbaFloatToI420P10(darkRgba, width, height);
    const brightResult = rgbaFloatToI420P10(brightRgba, width, height);

    // Bright should have higher Y than dark
    expect(brightResult[0]).toBeGreaterThan(darkResult[0]);
  });

  // utils.md O3 / Phase E46: P9 rename — pins BT.2020 10-bit
  // limited-range Y bounds (64 = black, 940 = peak white).
  it('extreme HDR brightness clamps Y into limited-range [64, 940]', () => {
    const width = 2;
    const height = 2;

    // Extreme bright
    const rgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = 10.0; // Very bright (HDR)
      rgba[i * 4 + 1] = 10.0;
      rgba[i * 4 + 2] = 10.0;
      rgba[i * 4 + 3] = 1.0;
    }

    const result = rgbaFloatToI420P10(rgba, width, height);

    // All Y values should be within limited range
    for (let i = 0; i < width * height; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(64);
      expect(result[i]).toBeLessThanOrEqual(940);
    }
  });

  // utils.md O3 / Phase E46: P9 rename — pins BT.2020 10-bit
  // limited-range Cb/Cr bounds (64 = chroma min, 960 = chroma max).
  it('pure-red RGBA clamps Cb/Cr into limited-range [64, 960]', () => {
    const width = 2;
    const height = 2;

    // Pure red (will have non-zero chroma)
    const rgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = 1.0;
      rgba[i * 4 + 1] = 0.0;
      rgba[i * 4 + 2] = 0.0;
      rgba[i * 4 + 3] = 1.0;
    }

    const result = rgbaFloatToI420P10(rgba, width, height);
    const ySize = width * height;
    const uvSize = (width >> 1) * (height >> 1);

    for (let i = 0; i < uvSize; i++) {
      expect(result[ySize + i]).toBeGreaterThanOrEqual(64);
      expect(result[ySize + i]).toBeLessThanOrEqual(960);
      expect(result[ySize + uvSize + i]).toBeGreaterThanOrEqual(64);
      expect(result[ySize + uvSize + i]).toBeLessThanOrEqual(960);
    }
  });

  // utils.md O3 / Phase E46: P9 rename — pure-gray RGBA (R=G=B) emits
  // achromatic Cb=Cr=512 (10-bit chroma center).
  it('neutral gray RGBA (R=G=B=0.5) emits achromatic Cb=Cr=512', () => {
    const width = 2;
    const height = 2;
    const rgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = 0.5;
      rgba[i * 4 + 1] = 0.5;
      rgba[i * 4 + 2] = 0.5;
      rgba[i * 4 + 3] = 1.0;
    }

    const result = rgbaFloatToI420P10(rgba, width, height);
    const ySize = width * height;

    // For gray, Cb and Cr should be at achromatic center (512)
    expect(result[ySize]).toBe(512); // U
    expect(result[ySize + 1]).toBe(512); // V
  });
});
