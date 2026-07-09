/**
 * Dimension animation configuration
 * Controls FPS-based animation through dimension ranges with various loop modes
 */
export interface DimensionAnimationConfig {
  defaults: {
    targetFPS: number;
    loop: 'once' | 'loop' | 'bounce';
    direction: 'forward' | 'backward';
  };
  presets: {
    fps: number[];
    customMin: number;
    customMax: number;
  };
  timing: {
    minFrameTimeMs: number;
    continuousTraverseSeconds: number;
  };
  ui: {
    showFPSFeedback: boolean;
    feedbackThreshold: number;
  };
  playback: {
    /**
     * Fraction of the animation frame window (1000 / targetFPS) handed to
     * the progressive loaders as their per-tick LOD time budget. The
     * remainder covers projection, commit, and scheduling overhead. Loaders
     * stream sub-LODs until the budget runs out — whether because of level
     * count or a slow (cache-miss) level — then commit what they have.
     */
    budgetFraction: number;
    /** Floor for the per-tick budget so high target FPS still loads data. */
    minBudgetMs: number;
    /**
     * Fixed per-tick reserve (ms) for projection + commit + render. At slow
     * target FPS the budget is `frameWindow − overheadReserveMs` when that
     * exceeds `frameWindow × budgetFraction` — otherwise a 1 fps playback
     * would idle 40 % of every second with refinement disabled, capping
     * quality below what the window could deliver.
     */
    overheadReserveMs: number;
  };
}
