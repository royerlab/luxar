/**
 * Pure helpers for HDR pixel buffer manipulation.
 *
 * Extracted from `post-processing-manager.ts`'s captureHDRPixels /
 * captureHDRAsEXR / renderToImageData paths so the typed-array
 * conversions and the WebGL → ImageData vertical flip are testable
 * without a renderer.
 *
 * @module rendering/post-processing/hdr-pixel-utils
 */

import * as THREE from 'three';

/**
 * Convert an array of half-float values (encoded as Uint16) into
 * Float32 values via {@link THREE.DataUtils.fromHalfFloat}.
 *
 * Used after `gl.readPixels` against a HalfFloat render target — WebGL
 * gives us the raw 16-bit-encoded values; the rest of the HDR pipeline
 * (EXR encoding, downstream math) wants Float32.
 *
 * @param halfData - Source half-float-encoded buffer (Uint16Array view).
 * @returns Float32Array of the same length with decoded values.
 */
export function halfFloatToFloat32(halfData: Uint16Array): Float32Array {
  const out = new Float32Array(halfData.length);
  for (let i = 0; i < halfData.length; i++) {
    out[i] = THREE.DataUtils.fromHalfFloat(halfData[i]);
  }
  return out;
}

/**
 * Convert Float32 values into half-float encoding via
 * {@link THREE.DataUtils.toHalfFloat}.
 *
 * Used when exporting HDR data as a half-float EXR (smaller file size
 * than full Float32 with negligible quality loss for typical scenes).
 *
 * @param floatData - Source Float32Array.
 * @returns Uint16Array of the same length with half-float-encoded values.
 */
export function float32ToHalfFloat(floatData: Float32Array): Uint16Array {
  const out = new Uint16Array(floatData.length);
  for (let i = 0; i < floatData.length; i++) {
    out[i] = THREE.DataUtils.toHalfFloat(floatData[i]);
  }
  return out;
}

/**
 * Flip a row-major RGBA pixel buffer vertically.
 *
 * WebGL's `gl.readPixels` returns rows in bottom-up order (y=0 at the
 * bottom of the framebuffer); a `Canvas2D ImageData` expects top-down
 * (y=0 at the top). This routine swaps rows in-place by row-block
 * copy. It does not change RGBA component order — only the row order.
 *
 * The output is `Uint8ClampedArray` because that's what `ImageData`
 * requires; clamping is a no-op here since the input is already
 * Uint8 (no values outside [0, 255]).
 *
 * @param pixels - Source row-major RGBA pixel buffer (4 bytes/pixel).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns Vertically-flipped buffer as Uint8ClampedArray.
 */
export function flipPixelsVerticallyRGBA(
  pixels: Uint8Array,
  width: number,
  height: number
): Uint8ClampedArray {
  const rowSize = width * 4;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const srcOffset = y * rowSize;
    const dstOffset = (height - 1 - y) * rowSize;
    out.set(pixels.subarray(srcOffset, srcOffset + rowSize), dstOffset);
  }
  return out;
}
