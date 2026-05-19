/**
 * Settle-loop predicates for PickingSystem.
 *
 * Extracted in P6/step 7.4. The RAF state machine + lifecycle stays
 * on the orchestrator (it owns `_rafId`, `_pendingMouse`, etc.); this
 * module owns the pure two-axis-settle decision logic — when should
 * a pick actually fire based on the timestamps?
 *
 * @module rendering/picking/picking-system/settle-loop
 */

/**
 * Two-axis settle window in milliseconds. A pick fires only after BOTH
 * the mouse and the pick buffer (camera/geometry) have been stable for
 * this long. 120 ms matches the standard tooltip-appearance delay used
 * by browsers and IDEs — short enough to feel responsive, long enough
 * to filter out cursor jitter and per-frame camera ticks.
 */
export const HOVER_SETTLE_MS = 120;

/** Timestamps the settle-loop reads to decide whether a pick should fire. */
export interface SettleTimestamps {
  readonly now: number;
  readonly lastMouseMoveTime: number;
  readonly lastDirtyTime: number;
  readonly lastPickFiredTime: number;
}

/** Outcome of one settle evaluation. */
export type SettleDecision =
  | { action: 'fire' }
  | { action: 'wait' }
  | { action: 'idle' };

/**
 * Decide what the RAF tick should do given the current timestamps.
 * - `fire`  → both axes settled AND something changed since last pick.
 * - `wait`  → at least one axis still moving; reschedule for another rAF.
 * - `idle`  → both axes settled but nothing changed since last pick.
 *             Stop scheduling — a future markDirty / mousemove will
 *             rearm the loop.
 */
export function evaluateSettle(t: SettleTimestamps): SettleDecision {
  const mouseSettled = t.now - t.lastMouseMoveTime >= HOVER_SETTLE_MS;
  const cameraSettled = t.now - t.lastDirtyTime >= HOVER_SETTLE_MS;
  if (!mouseSettled || !cameraSettled) return { action: 'wait' };

  const newHover = t.lastMouseMoveTime > t.lastPickFiredTime;
  const newCamera = t.lastDirtyTime > t.lastPickFiredTime;
  if (!newHover && !newCamera) return { action: 'idle' };

  return { action: 'fire' };
}
