import type { DimensionAnimationConfig } from './types';
import { MAX_ANIMATION_PERIOD_S } from '../../../types/zarr';

/**
 * Dimension animation configuration
 */
export const dimensionAnimationConfig: DimensionAnimationConfig = {
  defaults: {
    targetFPS: 10,
    loop: 'loop' as const,
    direction: 'forward' as const,
    stepSize: null, // Auto: fps-derived (continuous) / authored step (discrete)
    ladderDepth: 'auto', // 'auto' = energy rule; N pins N rungs; null = time-budgeted ('Fast')
  },
  presets: {
    fps: [0.5, 1, 2, 5, 10, 15, 30, 60, 120],
    // The same rate read as SECONDS PER STEP (the Speed section's period view):
    // slow cadences — a kiosk tour advancing every 20 s, a time-lapse stepping
    // once a minute — are named by their period, not by a tiny fps.
    periodSeconds: [1, 2, 5, 10, 20, 30, 60, 120, 300, 600],
    // One step an hour: the floor has to sit below the longest period preset,
    // or an authored 20 s period (target_fps 0.05) would be clamped to 10 s.
    customMin: 1 / MAX_ANIMATION_PERIOD_S,
    customMax: 120,
    stepMultipliers: [0.1, 0.25, 0.5, 1, 2, 5],
    ladderDepths: [1, 2, 3, 4, 6, 8],
  },
  timing: {
    minFrameTimeMs: 8, // ~120fps absolute max (needs a ≥120 Hz display to be reached)
    continuousTraverseSeconds: 10, // Full range in 10s for continuous dims
    // Playback slower than this is a SLIDESHOW: each tick loads like a manual
    // step (full refinement) instead of under frame-paced playback directives.
    slideshowPeriodMs: 2000,
  },
  ui: {
    showFPSFeedback: true, // Log playback cadence/committed-quality feedback
    feedbackThreshold: 0.8, // Cadence is slow below 80% of target
  },
  playback: {
    budgetFraction: 0.6, // 60% of the frame window for LOD streaming
    minBudgetMs: 8, // even at 60fps targets, give loaders ≥8ms
    overheadReserveMs: 50, // slow FPS: budget = window − reserve (see types.ts)
    autoEnergyThreshold: 0.9, // 'auto' detail: pin the first rung whose e(k) reaches this
  },
};
