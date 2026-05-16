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

const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;

/**
 * Drop WebGPU's per-row buffer padding from a readback typed array.
 *
 * WebGPU's `copyTextureToBuffer` (used internally by
 * `WebGPURenderer.readRenderTargetPixelsAsync`) requires `bytesPerRow`
 * to be a multiple of 256 (WebGPU spec § "Texture & buffer copy
 * alignment"). Three.js sizes the returned typed array to match the
 * padded GPU layout, so when `width * bytesPerTexel` isn't already a
 * multiple of 256 the array contains junk bytes at the end of every
 * row.
 *
 * This helper returns a row-compacted view sized exactly
 * `width * height * bytesPerTexel`. When the input already matches the
 * compact size (no padding required), the input is returned unchanged
 * to avoid an allocation. Both WebGL2-returned arrays (always compact)
 * and WebGPU-returned arrays for aligned widths hit the no-op path.
 *
 * Same math as the `PickingSystem.readbackAndVote` deinterlace loop;
 * extracted so the post-processing capture/screenshot paths can share
 * it (previously they assumed compact rows under WebGPU and produced
 * corrupted output for canvas widths whose row stride wasn't a multiple
 * of 256 bytes).
 *
 * @param raw - The typed array returned by
 *   `WebGPURenderer.readRenderTargetPixelsAsync`.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param bytesPerTexel - 4 for RGBA8/Uint8, 8 for RGBA16F/Uint16, 16
 *   for RGBA32F/Float32.
 * @returns Either `raw` itself (when no compaction is needed) or a new
 *   typed array of the same concrete type sized for the compact layout.
 */
export function compactWebGPUReadbackRows<T extends Uint8Array | Uint16Array | Float32Array>(
  raw: T,
  width: number,
  height: number,
  bytesPerTexel: number
): T {
  const bytesPerRowReal = width * bytesPerTexel;
  const bytesPerRowPadded =
    Math.ceil(bytesPerRowReal / WEBGPU_BYTES_PER_ROW_ALIGNMENT) * WEBGPU_BYTES_PER_ROW_ALIGNMENT;
  if (bytesPerRowPadded === bytesPerRowReal) {
    return raw;
  }
  const bytesPerElement = raw.BYTES_PER_ELEMENT;
  const elementsPerRowPadded = bytesPerRowPadded / bytesPerElement;
  const elementsPerRowReal = bytesPerRowReal / bytesPerElement;
  const compactElementCount = elementsPerRowReal * height;
  if (raw.length === compactElementCount) {
    return raw;
  }
  // Allocate the same concrete typed-array kind as the input. The
  // generic constraint guarantees the constructor lookup is safe.
  const Ctor = raw.constructor as new (length: number) => T;
  const out = new Ctor(compactElementCount);
  for (let row = 0; row < height; row++) {
    const srcStart = row * elementsPerRowPadded;
    const dstStart = row * elementsPerRowReal;
    out.set(raw.subarray(srcStart, srcStart + elementsPerRowReal), dstStart);
  }
  return out;
}
