/**
 * HDR color conversion utilities for 10-bit video encoding.
 *
 * Converts linear sRGB float data (from WebGL readPixels) to BT.2020 PQ I420P10
 * format suitable for WebCodecs VideoFrame with HDR metadata.
 */

import { clamp } from './clamp';

// sRGB linear → BT.2020 linear color space matrix (3x3)
// Source: ITU-R BT.2087 conversion from BT.709 to BT.2020
const SRGB_TO_BT2020 = [0.6274, 0.3293, 0.0433, 0.0691, 0.9195, 0.0114, 0.0164, 0.088, 0.8956];

// PQ (Perceptual Quantizer, SMPTE ST 2084) constants
const PQ_M1 = 0.1593017578125; // 2610/16384
const PQ_M2 = 78.84375; // 2523/32 * 128
const PQ_C1 = 0.8359375; // 3424/4096
const PQ_C2 = 18.8515625; // 2413/128
const PQ_C3 = 18.6875; // 2392/128

/**
 * Apply PQ (SMPTE ST 2084) EOTF inverse (linear → PQ).
 * Input: linear luminance in [0, 1] range (where 1.0 = 10000 nits reference)
 * Output: PQ encoded value in [0, 1]
 *
 * For WebGL content, we assume scene-referred values where 1.0 maps
 * to ~100 nits (SDR white), so we scale by 100/10000 = 0.01 before PQ.
 */
function linearToPQ(L: number): number {
  // Scale scene-referred linear (1.0 = SDR white ~100 nits) to PQ absolute (1.0 = 10000 nits)
  const Lp = Math.max(0, L) * 0.01;
  const Lpm1 = Math.pow(Lp, PQ_M1);
  const numerator = PQ_C1 + PQ_C2 * Lpm1;
  const denominator = 1.0 + PQ_C3 * Lpm1;
  return Math.pow(numerator / denominator, PQ_M2);
}

/**
 * Convert linear sRGB float RGBA pixels to BT.2020 PQ YCbCr I420P10 format.
 *
 * Pipeline:
 * 1. Linear sRGB RGB → BT.2020 linear RGB (3x3 matrix)
 * 2. BT.2020 linear → PQ transfer function
 * 3. PQ RGB → YCbCr (BT.2020 non-constant luminance coefficients)
 * 4. Quantize to 10-bit (0-1023)
 * 5. 4:2:0 chroma subsampling
 * 6. Pack into I420P10 planar layout (Uint16Array, 16-bit LE per sample)
 *
 * @param rgba - Linear sRGB float pixel data (RGBA, 4 components per pixel)
 * @param width - Image width in pixels
 * @param height - Image height in pixels
 * @returns Uint16Array in I420P10 planar layout (Y plane + U plane + V plane)
 */
export function rgbaFloatToI420P10(rgba: Float32Array, width: number, height: number): Uint16Array {
  // BT.2020 NCL YCbCr coefficients (ITU-R BT.2020)
  const KR = 0.2627;
  const KB = 0.0593;
  const KG = 1.0 - KR - KB; // 0.6780

  const yPlaneSize = width * height;
  const uvWidth = width >> 1;
  const uvHeight = height >> 1;
  const uvPlaneSize = uvWidth * uvHeight;

  // I420P10: Y plane, then U plane, then V plane (each sample = 16-bit LE)
  const output = new Uint16Array(yPlaneSize + 2 * uvPlaneSize);

  // Temporary full-resolution Cb/Cr for subsampling
  const cbFull = new Float32Array(yPlaneSize);
  const crFull = new Float32Array(yPlaneSize);

  // Pass 1: Convert each pixel and write Y plane + full-res Cb/Cr
  for (let i = 0; i < yPlaneSize; i++) {
    const ri = i * 4;
    const rIn = rgba[ri];
    const gIn = rgba[ri + 1];
    const bIn = rgba[ri + 2];

    // 1. sRGB linear → BT.2020 linear
    const r2020 = SRGB_TO_BT2020[0] * rIn + SRGB_TO_BT2020[1] * gIn + SRGB_TO_BT2020[2] * bIn;
    const g2020 = SRGB_TO_BT2020[3] * rIn + SRGB_TO_BT2020[4] * gIn + SRGB_TO_BT2020[5] * bIn;
    const b2020 = SRGB_TO_BT2020[6] * rIn + SRGB_TO_BT2020[7] * gIn + SRGB_TO_BT2020[8] * bIn;

    // 2. BT.2020 linear → PQ
    const rPQ = linearToPQ(r2020);
    const gPQ = linearToPQ(g2020);
    const bPQ = linearToPQ(b2020);

    // 3. PQ RGB → YCbCr (BT.2020 NCL)
    // Y  = KR*R + KG*G + KB*B
    // Cb = (B - Y) / (2 * (1 - KB))
    // Cr = (R - Y) / (2 * (1 - KR))
    const y = KR * rPQ + KG * gPQ + KB * bPQ;
    const cb = (bPQ - y) / (2.0 * (1.0 - KB));
    const cr = (rPQ - y) / (2.0 * (1.0 - KR));

    // 4. Quantize to 10-bit
    // Y: [0, 1] → [64, 940] (limited range)
    // Cb/Cr: [-0.5, 0.5] → [64, 960] (limited range, centered at 512)
    const yQ = Math.round(clamp(y * 876 + 64, 64, 940));
    output[i] = yQ;

    cbFull[i] = cb;
    crFull[i] = cr;
  }

  // Pass 2: 4:2:0 chroma subsampling (average 2x2 blocks)
  const uOffset = yPlaneSize;
  const vOffset = yPlaneSize + uvPlaneSize;

  for (let j = 0; j < uvHeight; j++) {
    for (let k = 0; k < uvWidth; k++) {
      const srcRow = j * 2;
      const srcCol = k * 2;
      const i00 = srcRow * width + srcCol;
      const i01 = i00 + 1;
      const i10 = i00 + width;
      const i11 = i10 + 1;

      const cbAvg = (cbFull[i00] + cbFull[i01] + cbFull[i10] + cbFull[i11]) * 0.25;
      const crAvg = (crFull[i00] + crFull[i01] + crFull[i10] + crFull[i11]) * 0.25;

      const uvIdx = j * uvWidth + k;
      output[uOffset + uvIdx] = Math.round(clamp(cbAvg * 896 + 512, 64, 960));
      output[vOffset + uvIdx] = Math.round(clamp(crAvg * 896 + 512, 64, 960));
    }
  }

  return output;
}
