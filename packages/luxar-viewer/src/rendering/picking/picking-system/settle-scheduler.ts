/**
 * Two-axis settle scheduler for PickingSystem.
 *
 * Owns the RAF lifecycle and the mouse/dirty timestamps that drive
 * picking. The orchestrator builds a narrow `SettleSchedulerCtx`
 * (no `this` back-pointer) and lets the scheduler decide when to
 * actually invoke `firePick`. The decision policy itself is the
 * pure function `evaluateSettle()` in `./settle-loop`.
 *
 * The split keeps three concerns separate and unit-testable:
 *   - `settle-loop.ts`     pure decision (when to fire, given timestamps).
 *   - `settle-scheduler.ts` rAF lifecycle + timestamp ownership.
 *   - `picking-system.ts`   GPU pipeline + registration coordination.
 *
 * @module rendering/picking/picking-system/settle-scheduler
 */

import { evaluateSettle } from './settle-loop';

/** Narrow callback surface the orchestrator hands to the scheduler. */
export interface SettleSchedulerCtx {
  /** Current time in milliseconds — usually `performance.now()`. */
  now(): number;
  /** Gate the actual pick (orchestrator combines `_shouldPick` etc.). */
  shouldFire(): boolean;
  /** Trigger a pick at the supplied canvas-relative coordinates. */
  firePick(x: number, y: number): void;
}

export class SettleScheduler {
  private _pendingMouse: { x: number; y: number } | null = null;
  private _lastMouseMoveTime = 0;
  private _lastDirtyTime = 0;
  private _lastPickFiredTime = 0;
  private _rafId: number | null = null;
  private _suppressed = false;

  constructor(private ctx: SettleSchedulerCtx) {}

  /**
   * Record a cursor position from a mousemove. The position + timestamp
   * are tracked even while suppressed so the re-pick on resume
   * (orbit/pan/zoom release) uses where the cursor actually is — not a
   * stale pre-interaction position. While suppressed we still skip
   * scheduling, so no pick fires mid-interaction; `setSuppressed(false)`
   * re-arms the loop using the latest tracked position.
   */
  recordMouseMove(x: number, y: number): void {
    this._pendingMouse = { x, y };
    this._lastMouseMoveTime = this.ctx.now();
    if (this._suppressed) return;
    this.scheduleRaf();
  }

  /** Cursor left the canvas — drop the pending position and cancel rAF. */
  recordMouseLeave(): void {
    this.cancelPending();
  }

  /**
   * Drop the pending cursor and cancel the armed rAF without forgetting
   * the suppression state. Used when the stored canvas-local coordinate
   * is no longer valid — e.g. a page/ancestor scroll moved the canvas
   * after the rect was cached but before the settle pick fired, so the
   * pending coordinate would resolve against the stale frame. Picking
   * re-arms naturally on the next mousemove (which re-fetches the rect).
   */
  cancelPending(): void {
    this._pendingMouse = null;
    this.cancelRaf();
  }

  /** Camera/geometry/viewport changed — bump the dirty axis and re-arm. */
  markDirty(): void {
    this._lastDirtyTime = this.ctx.now();
    this.scheduleRaf();
  }

  /**
   * Suppress (`true`) or resume (`false`) picking. Resume re-arms the
   * rAF when there's a pending cursor so orbit-and-release picks once
   * the camera settles, without requiring a fresh mousemove.
   */
  setSuppressed(value: boolean): void {
    this._suppressed = value;
    if (value) {
      this.cancelRaf();
    } else if (this._pendingMouse) {
      this.scheduleRaf();
    }
  }

  /** Cancel any pending rAF and forget the pending cursor. Idempotent. */
  dispose(): void {
    this.cancelRaf();
    this._pendingMouse = null;
  }

  get lastMouseMoveTime(): number {
    return this._lastMouseMoveTime;
  }
  get lastDirtyTime(): number {
    return this._lastDirtyTime;
  }
  get lastPickFiredTime(): number {
    return this._lastPickFiredTime;
  }
  get isSuppressed(): boolean {
    return this._suppressed;
  }

  private scheduleRaf(): void {
    if (this._rafId !== null) return;
    if (this._suppressed) return;
    this._rafId = requestAnimationFrame(this.rafTick);
  }

  private cancelRaf(): void {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  /**
   * Per-frame settle check. Fires when both axes have been quiet for
   * HOVER_SETTLE_MS and at least one axis has changed since the last
   * pick. Reschedules itself while waiting; stops once idle.
   */
  private rafTick = (): void => {
    this._rafId = null;
    if (this._suppressed || !this._pendingMouse) return;

    const now = this.ctx.now();
    const decision = evaluateSettle({
      now,
      lastMouseMoveTime: this._lastMouseMoveTime,
      lastDirtyTime: this._lastDirtyTime,
      lastPickFiredTime: this._lastPickFiredTime,
    });

    if (decision.action === 'wait') {
      this.scheduleRaf();
      return;
    }
    if (decision.action === 'idle') return;
    if (!this.ctx.shouldFire()) return;

    this._lastPickFiredTime = now;
    this.ctx.firePick(this._pendingMouse.x, this._pendingMouse.y);
  };
}
