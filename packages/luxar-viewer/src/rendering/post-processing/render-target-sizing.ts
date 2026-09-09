/**
 * Pure helper for computing post-processing render-target dimensions.
 *
 * Extracted from `rendering/post-processing/post-processing-manager.ts` so the SSAA
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

export interface RenderTargetAllocation {
  logical: RenderSize;
  physical: RenderSize;
  limited: boolean;
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

/**
 * Compute matching renderer and render-target dimensions.
 *
 * Three.js floors `logical × pixelRatio` when it sizes the canvas. Use
 * the same rule here so every off-screen attachment exactly matches the
 * backbuffer at non-integer DPRs. If either physical axis exceeds the
 * framebuffer limit, reduce both logical axes by one common factor to
 * preserve the viewport aspect ratio.
 */
export function computeRenderTargetAllocation(
  renderSize: RenderSize,
  ssaaEnabled: boolean,
  ssaaMultiplier: number,
  pixelRatio: number,
  maxPhysicalDimension: number
): RenderTargetAllocation {
  const effective = computeEffectiveRenderSize(renderSize, ssaaEnabled, ssaaMultiplier);
  const requestedPhysical = {
    width: Math.max(1, Math.floor(effective.width * pixelRatio)),
    height: Math.max(1, Math.floor(effective.height * pixelRatio)),
  };
  const scale = Math.min(
    1,
    maxPhysicalDimension / requestedPhysical.width,
    maxPhysicalDimension / requestedPhysical.height
  );
  const logical = {
    width: Math.max(1 / pixelRatio, effective.width * scale),
    height: Math.max(1 / pixelRatio, effective.height * scale),
  };

  return {
    logical,
    physical: {
      width: Math.max(1, Math.floor(logical.width * pixelRatio)),
      height: Math.max(1, Math.floor(logical.height * pixelRatio)),
    },
    limited: scale < 1,
  };
}
