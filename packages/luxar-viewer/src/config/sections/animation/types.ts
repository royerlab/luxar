/**
 * Animation loop settings
 */
export interface AnimationConfig {
  idleTimeoutMs: number;
  /**
   * Frame pacing: after a pathologically slow frame the loop waits out a
   * bounded cooldown before scheduling the next one, so the main thread is
   * not held at a 100 % duty cycle of long tasks (#1724). See
   * `scene/animation/animation-controller.ts`.
   */
  pacing: {
    /** Master switch — false restores the back-to-back rAF loop. */
    enabled: boolean;
    /** Frame cost (ms) above which the next frame is paced. */
    slowFrameMs: number;
    /** Upper bound (ms) on the inserted cooldown. */
    maxCooldownMs: number;
  };
}
