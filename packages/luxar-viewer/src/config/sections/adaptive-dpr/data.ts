import type { AdaptiveDPRConfig } from './types';

/**
 * Adaptive pixel ratio configuration for dynamic performance optimization.
 *
 * The AdaptiveDPRManager constructor merges this literal directly
 * (defaults) with `config.adaptiveDPR` (runtime/test overrides) so a
 * partial override — e.g. the fixed-shape config mock in unit tests —
 * can never leave a knob `undefined`.
 */
export const adaptiveDPRConfig: AdaptiveDPRConfig = {
  enabled: true, // Construction-time default; runtime toggle is renderingControls.defaults.adaptiveDPREnabled
  minDPR: 0.5, // Minimum DPR - lower bound before image becomes too pixelated
  scaleDownFactor: 0.9, // Reduce DPR by 10% when scaling down
  scaleUpFactor: 1.05, // Increase DPR by 5% when scaling up
  hysteresisSeconds: 3, // Wait 3 seconds of stable high FPS before scaling up
  evaluationIntervalMs: 500, // Evaluate FPS every 500ms

  // Refresh-rate-relative thresholds (60Hz → down<45, up>54; 120Hz → 90/108)
  scaleDownFpsRatio: 0.75,
  scaleUpFpsRatio: 0.9,
  refreshRateFallback: 60,
  midbandGraceSamples: 1,

  // U-shape probe (previously hardcoded class constants)
  probeWindowMs: 1500,
  probeImprovement: 1.05,
  probeMinSamples: 8,

  // Learned floor + exponential backoff (30s → 60s → 2min → capped 5min)
  floorTtlMs: 30_000,
  backoffMultiplier: 2,
  backoffMaxTtlMs: 300_000,

  // Evidence-based ceiling (native → 1.0 after repeated punished ascents)
  ceilingTtlMs: 60_000,
  punishedAscentWindowMs: 3000,
  punishedAscentThreshold: 2,

  // Session hygiene
  contentChangeRecheckMs: 5000,
  gapResetMs: 350,
};
