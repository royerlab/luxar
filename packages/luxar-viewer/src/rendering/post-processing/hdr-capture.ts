/**
 * HDR-capture helpers for the post-processing pipeline.
 *
 * Pure helpers used by `captureHDRPixels`, `captureHDRAsEXR`, and the
 * memory estimator. The actual `composer.render()` and the WebGL
 * `readRenderTargetPixels` calls stay in `post-processing-manager.ts`
 * (they need a live renderer + composer); the *deterministic*
 * book-keeping that decides which ping-pong buffer holds the final
 * pixels and how big the post-processing pipeline is in VRAM is
 * extracted here.
 *
 * @module rendering/post-processing/hdr-capture
 */

/** Half-float (RGBA, 16 bits/channel) → 8 bytes per pixel. */
export const HDR_BYTES_PER_PIXEL = 8;

/**
 * Minimal slice of the EffectComposer pass we care about for buffer
 * picking. Each pass exposes `enabled` and `needsSwap`; only passes
 * with both true contribute a buffer swap.
 */
export interface SwapAwarePass {
  enabled: boolean;
  needsSwap: boolean;
}

/**
 * Count enabled passes that actually swap buffers. Pure.
 *
 * Some pmndrs passes (e.g. `ClearPass`, `MaskPass`) set `needsSwap =
 * false`; counting all enabled passes is wrong since the composer
 * only flips buffers on a `needsSwap` pass.
 */
export function countSwapPasses(passes: ReadonlyArray<SwapAwarePass>): number {
  let count = 0;
  for (const p of passes) {
    if (p.enabled && p.needsSwap) count++;
  }
  return count;
}

/**
 * Decide which of the two ping-pong buffers holds the final result of
 * `composer.render()` given how many swap-capable enabled passes were
 * applied: even count → inputBuffer; odd count → outputBuffer.
 *
 * Generic over the buffer type so callers can pass in
 * `THREE.WebGLRenderTarget` or any test stub. Pure.
 */
export function pickResultBuffer<T>(
  passes: ReadonlyArray<SwapAwarePass>,
  inputBuffer: T,
  outputBuffer: T
): T {
  return countSwapPasses(passes) % 2 === 0 ? inputBuffer : outputBuffer;
}

/**
 * Estimate the post-processing pipeline's VRAM footprint in MB.
 *
 * The composer keeps a fixed number of half-float RGBA render targets
 * (the two ping-pong buffers + the read target = 3 base targets), and
 * a few extras when bloom / AO are active (bloom keeps a downsample
 * pyramid; we approximate as +2; AO keeps a normal-depth pre-pass; we
 * approximate as +1). SSAA scales the per-target pixel count by the
 * multiplier squared; MSAA effectively multiplies the inner-target
 * memory by `msaaSamples`.
 *
 * Pure — used by `getPerformanceMetrics()`.
 */
export interface MemoryEstimateInputs {
  /** Display-resolution pixel count (width × height). */
  pixelCount: number;
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  msaaEnabled: boolean;
  msaaSamples: number;
  hasBloom: boolean;
  hasAO: boolean;
}

export function estimatePostProcMemoryMB(inputs: MemoryEstimateInputs): number {
  const ssaa = inputs.ssaaEnabled ? inputs.ssaaMultiplier * inputs.ssaaMultiplier : 1;
  const msaa = inputs.msaaEnabled ? inputs.msaaSamples : 1;
  const totalPixels = inputs.pixelCount * ssaa * msaa;
  const bufferCount = 3 + (inputs.hasBloom ? 2 : 0) + (inputs.hasAO ? 1 : 0);
  return (totalPixels * HDR_BYTES_PER_PIXEL * bufferCount) / (1024 * 1024);
}

/**
 * Format the size-tagged log line printed after a successful EXR
 * capture. Centralized so the test can pin the format.
 */
export function formatHDRExrLogLine(
  width: number,
  height: number,
  isHalfFloat: boolean,
  byteLength: number
): string {
  return (
    `HDR EXR captured: ${width}x${height}, ` +
    `${isHalfFloat ? 'half-float' : 'float'}, ` +
    `${(byteLength / (1024 * 1024)).toFixed(1)} MB`
  );
}
