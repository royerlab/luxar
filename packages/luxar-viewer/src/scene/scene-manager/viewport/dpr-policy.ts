/**
 * Pixel-ratio (DPR) policy — pure functions for the SceneManager's
 * adaptive/manual DPR handling.
 *
 * Kept stateless so the policy can be unit-tested without a SceneManager.
 * The active override + post-processing manager (which the policy
 * informs when DPR changes) are passed in by callers.
 */

import type { PostProcessingManager } from '../../../rendering';

/** Return the DPR currently applied to renderer sizing. */
export function getActivePixelRatio(override: number | null): number {
  return (override ?? window.devicePixelRatio) || 1;
}

/**
 * Compute the next override + the resulting effective DPR for a
 * `setAdaptivePixelRatio(dpr)` call. Native DPR clears the override so
 * future monitor-DPI changes continue to track `window.devicePixelRatio`
 * automatically.
 */
export function computePixelRatioOverride(dpr: number): {
  override: number | null;
  active: number;
} {
  const nativeDPR = window.devicePixelRatio || 1;
  const safeDPR = Number.isFinite(dpr) && dpr > 0 ? dpr : nativeDPR;
  const override = Math.abs(safeDPR - nativeDPR) < 0.01 ? null : safeDPR;
  return { override, active: getActivePixelRatio(override) };
}

/**
 * Normalize active DPR relative to current native DPR for perceptual
 * effect scaling (e.g. detector-noise grain density should stay
 * perceptually constant across DPR overrides).
 */
export function getNormalizedDPRScale(activeDPR: number): number {
  const nativeDPR = window.devicePixelRatio || 1;
  return activeDPR / nativeDPR;
}

/**
 * Push the normalized DPR scale into the post-processing pipeline so
 * its DPR-dependent effects (noise grain, AA threshold) stay
 * consistent after DPR / resize changes.
 */
export function syncPostProcessingDPRScale(
  postProcessing: PostProcessingManager | null,
  override: number | null
): void {
  if (!postProcessing) return;
  postProcessing.setDPRScale(getNormalizedDPRScale(getActivePixelRatio(override)));
}
