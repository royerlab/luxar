/**
 * Dimension animation configuration
 * Controls FPS-based animation through dimension ranges with various loop modes
 */
export interface DimensionAnimationConfig {
  defaults: {
    targetFPS: number;
    loop: 'once' | 'loop' | 'bounce';
    direction: 'forward' | 'backward';
    /** Per-tick step override; null = Auto (fps-derived / authored step). */
    stepSize: number | null;
    /**
     * Playback detail: `'auto'` pins each ladder at the first rung whose
     * cumulative energy e(k) reaches `playback.autoEnergyThreshold` (stamped
     * ladders; unstamped ones fall back to time-budgeted streaming), a number
     * pins that many rungs, `null` streams whatever fits the tick ("Fast").
     * See `DimensionAnimationState.ladderDepth`.
     */
    ladderDepth: number | 'auto' | null;
  };
  presets: {
    fps: number[];
    /**
     * Speed-section choices in PERIOD view, in seconds per step (the same
     * rate as an fps, `1 / period`). See `ui/dimension-sliders.ts`.
     */
    periodSeconds: number[];
    /** Slowest accepted rate, in fps (one step every `1 / customMin` s). */
    customMin: number;
    customMax: number;
    /**
     * Step-section choices in the per-dimension context menu, as multipliers
     * of the dimension's BASE step (authored step, else 1% of the range).
     */
    stepMultipliers: number[];
    /**
     * Detail-section choices in the per-dimension context menu: pinned rung
     * counts offered between Auto and All (the whole ladder).
     */
    ladderDepths: number[];
  };
  timing: {
    minFrameTimeMs: number;
    continuousTraverseSeconds: number;
    /**
     * Tick period (ms) at or above which playback is a SLIDESHOW rather than
     * frame-paced: each tick is loaded like a manual step and refines to the
     * full ladder, the playback directives (tick budget, pinned detail, t+1
     * prefetch, LOD aspiration cap) and the cadence feedback do not apply,
     * and the clock is held while a reveal-on-arrival waypoint flight is in
     * the air. See `DimensionAnimationManager`.
     */
    slideshowPeriodMs: number;
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
    /**
     * The `'auto'` playback detail pins a ladder at the FIRST rung whose
     * cumulative energy fraction e(k) (the `energy_fraction_cum` build stamps)
     * reaches this value; a ladder without stamps is left time-budgeted.
     */
    autoEnergyThreshold: number;
  };
}
