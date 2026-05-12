/**
 * Pure helper for computing post-processing render-target dimensions.
 *
 * Extracted from `rendering/post-processing-manager.ts` so the SSAA
 * upscale arithmetic — multiply the screen size by `ssaaMultiplier` only
 * when SSAA is on, otherwise pass through — is testable without a
 * WebGL context.
 *
 * @module rendering/post-processing/render-target-sizing
 */

export interface RenderSize {
  width: number;
  height: number;
}

/**
 * Compute the effective render-target size given current SSAA settings.
 *
 * When SSAA is enabled, the off-screen render target runs at
 * `ssaaMultiplier` × the screen size in each axis (the post-processing
 * stack samples and downsamples back to the screen size). When SSAA is
 * disabled, the on-screen size is used directly. Width and height are
 * rounded to integers in both cases — render targets must be integral.
 *
 * @param renderSize - The on-screen render size in pixels.
 * @param ssaaEnabled - Whether SSAA is currently on.
 * @param ssaaMultiplier - The SSAA upscale factor (e.g. 1.5, 2, 4).
 *   Ignored when ssaaEnabled is false.
 * @returns Integer `{ width, height }` for the off-screen render target.
 */
export function computeEffectiveRenderSize(
  renderSize: RenderSize,
  ssaaEnabled: boolean,
  ssaaMultiplier: number
): RenderSize {
  if (!ssaaEnabled) {
    return { width: renderSize.width, height: renderSize.height };
  }
  return {
    width: Math.round(renderSize.width * ssaaMultiplier),
    height: Math.round(renderSize.height * ssaaMultiplier),
  };
}
