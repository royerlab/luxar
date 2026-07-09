import type { DimensionAnimationConfig } from './types';

/**
 * Dimension animation configuration
 */
export const dimensionAnimationConfig: DimensionAnimationConfig = {
  defaults: {
    targetFPS: 10,
    loop: 'loop' as const,
    direction: 'forward' as const,
  },
  presets: {
    fps: [1, 2, 5, 10, 15, 30, 60],
    customMin: 0.1,
    customMax: 120,
  },
  timing: {
    minFrameTimeMs: 16, // ~60fps absolute max
    continuousTraverseSeconds: 10, // Full range in 10s for continuous dims
  },
  ui: {
    showFPSFeedback: true, // Show "target vs actual" fps
    feedbackThreshold: 0.8, // Warn if actual < 80% of target
  },
  playback: {
    budgetFraction: 0.6, // 60% of the frame window for LOD streaming
    minBudgetMs: 8, // even at 60fps targets, give loaders ≥8ms
    overheadReserveMs: 50, // slow FPS: budget = window − reserve (see types.ts)
  },
};
