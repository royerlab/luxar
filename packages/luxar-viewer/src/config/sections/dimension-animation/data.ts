import type { DimensionAnimationConfig } from './types';

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
    customMin: 0.1,
    customMax: 120,
    stepMultipliers: [0.1, 0.25, 0.5, 1, 2, 5],
    ladderDepths: [1, 2, 3, 4, 6, 8],
  },
  timing: {
    minFrameTimeMs: 8, // ~120fps absolute max (needs a ≥120 Hz display to be reached)
    continuousTraverseSeconds: 10, // Full range in 10s for continuous dims
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
