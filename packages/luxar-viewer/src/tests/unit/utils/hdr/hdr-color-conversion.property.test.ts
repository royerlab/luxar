/**
 * Property regression tests for sRGB→BT.2020 matrix energy preservation.
 *
 * Per MED-43: `SRGB_TO_BT2020` row sums describe how each BT.2020 primary
 * is mixed from sRGB primaries. For an achromatic (pure-white) sRGB input,
 * the matrix must map to a near-achromatic BT.2020 output — otherwise white
 * acquires a chroma tint after the conversion, polluting HDR output.
 *
 * These tests don't change source; they pin the existing matrix coefficients
 * to a measurable property (row sums ~ 1.0) so any future edit that drifts
 * the matrix is flagged immediately.
 */

import { describe, it, expect } from 'vitest';
import { rgbaFloatToI420P10 } from '../../../../utils/hdr/hdr-color-conversion';

// The matrix is module-private; we exercise it end-to-end via the public
// pipeline and assert the achromatic property on the output.
describe('SRGB_TO_BT2020 — energy preservation (property tests)', () => {
  it('maps pure-white linear sRGB to a near-achromatic BT.2020 frame', () => {
    // 2x2 image, fully white (linear sRGB R=G=B=1.0)
    const width = 2;
    const height = 2;
    const rgba = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4 + 0] = 1.0; // R
      rgba[i * 4 + 1] = 1.0; // G
      rgba[i * 4 + 2] = 1.0; // B
      rgba[i * 4 + 3] = 1.0; // A
    }

    const out = rgbaFloatToI420P10(rgba, width, height);

    const ySize = width * height;
    const uvWidth = width >> 1;
    const uvHeight = height >> 1;
    const uvSize = uvWidth * uvHeight;
    const uOffset = ySize;
    const vOffset = ySize + uvSize;

    // For achromatic input, Cb and Cr must be ≈ 512 (the limited-range
    // achromatic center). Any matrix-row drift (or non-symmetric channel
    // weighting) would shift them away.
    for (let i = 0; i < uvSize; i++) {
      expect(out[uOffset + i]).toBeGreaterThanOrEqual(509);
      expect(out[uOffset + i]).toBeLessThanOrEqual(515);
      expect(out[vOffset + i]).toBeGreaterThanOrEqual(509);
      expect(out[vOffset + i]).toBeLessThanOrEqual(515);
    }

    // Y for white should be well above the achromatic midpoint (i.e. the
    // luma encoder actually sees luminance, not zero).
    for (let i = 0; i < ySize; i++) {
      expect(out[i]).toBeGreaterThan(500);
    }
  });

  it('maps pure-black linear sRGB to limited-range black with achromatic chroma', () => {
    // Black: R=G=B=0 → BT.2020 = (0,0,0) → PQ(0) = 0 → Y limited-range = 64,
    // Cb/Cr centered at 512. Tests that the matrix preserves zero exactly
    // (row sums × 0 = 0) and that no chroma leaks in from rounding.
    const rgba = new Float32Array(2 * 2 * 4); // all zeros (alpha included)
    const out = rgbaFloatToI420P10(rgba, 2, 2);

    const ySize = 2 * 2;
    expect(out[0]).toBe(64);
    expect(out[ySize]).toBe(512); // U
    expect(out[ySize + 1]).toBe(512); // V (1x1 chroma plane @ 2x2 image)
  });
});
