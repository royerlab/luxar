/**
 * Pixel-ratio (DPR) policy — the SceneManager's adaptive/manual DPR
 * handling, and the single place a requested DPR is turned into the
 * number handed to `renderer.setPixelRatio`.
 *
 * Kept stateless so the policy can be unit-tested without a SceneManager.
 * The active override + post-processing manager (which the policy
 * informs when DPR changes) are passed in by callers; the one piece of
 * state these functions consult, the pixel-ratio CAP, lives in
 * `rendering/pixel-ratio-cap` (see that module for what the cap is, who
 * writes it, and why it is enforced at this seam).
 */

import type { PostProcessingManager } from '../../../rendering';
import { getMaxPixelRatio } from '../../../rendering/pixel-ratio-cap';

/**
 * Return the DPR currently applied to renderer sizing.
 *
 * A `null` override means "track the live ceiling" — the live
 * `window.devicePixelRatio` whenever high DPR is allowed, and the cap
 * otherwise. Keeping that meaning (rather than "track the live native")
 * is what preserves the no-reallocation fast path in
 * `AdaptiveDPRManager.syncNativeDPR`; see computePixelRatioOverride.
 */
export function getActivePixelRatio(override: number | null): number {
  const ceiling = getMaxPixelRatio();
  return Math.min((override ?? ceiling) || 1, ceiling);
}

/**
 * Compute the next override + the resulting effective DPR for a
 * `setAdaptivePixelRatio(dpr)` call.
 *
 * A request AT the ceiling clears the override, so future monitor-DPI
 * changes keep tracking the display automatically instead of forcing a
 * redundant render-target reallocation. Note this compares against the
 * CEILING and not the raw native DPR: under a 1.0 cap on a 2x display no
 * request would ever equal native, the override would be permanently
 * non-null, and that whole fast path would be lost.
 */
export function computePixelRatioOverride(dpr: number): {
  override: number | null;
  active: number;
} {
  const ceiling = getMaxPixelRatio();
  const requested = Number.isFinite(dpr) && dpr > 0 ? dpr : ceiling;
  const safeDPR = Math.min(requested, ceiling);
  const override = Math.abs(safeDPR - ceiling) < 0.01 ? null : safeDPR;
  return { override, active: getActivePixelRatio(override) };
}

/**
 * Normalize active DPR relative to the current ceiling for perceptual
 * effect scaling (e.g. detector-noise grain density should stay
 * perceptually constant across DPR overrides).
 *
 * Against the CEILING, not the native DPR: a scale of 1.0 must mean
 * "full quality for this viewer", which under a cap IS the cap.
 * Normalizing against native would peg the scale at 0.5 forever on a 2x
 * display and silently attenuate detector noise to half sigma / a
 * quarter photon gain in every capped session.
 */
export function getNormalizedDPRScale(activeDPR: number): number {
  return activeDPR / getMaxPixelRatio();
}

/**
 * Push the normalized DPR scale into the post-processing pipeline so its
 * DPR-dependent effects (detector-noise grain) stay consistent after
 * DPR / resize changes.
 */
export function syncPostProcessingDPRScale(
  postProcessing: PostProcessingManager | null,
  override: number | null
): void {
  if (!postProcessing) return;
  postProcessing.setDPRScale(getNormalizedDPRScale(getActivePixelRatio(override)));
}
