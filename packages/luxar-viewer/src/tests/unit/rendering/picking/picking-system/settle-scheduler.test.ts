/**
 * Unit tests for `SettleScheduler` in isolation.
 *
 * The scheduler is the bridge between the pure `evaluateSettle()`
 * predicate and the orchestrator's `performPick`. These tests inject
 * a fake `ctx` (controlled clock + recorded callbacks) and a manual
 * rAF queue so every settle path is exercised without spinning up
 * `PickingSystem` or a renderer.
 *
 * Integration of the scheduler with the orchestrator is covered by
 * the existing `picking-system.test.ts::PickingSystem — settle scheduler`
 * suite.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  SettleScheduler,
  type SettleSchedulerCtx,
} from '../../../../../rendering/picking/picking-system/settle-scheduler';
import { HOVER_SETTLE_MS } from '../../../../../rendering/picking/picking-system/settle-loop';

interface Harness {
  scheduler: SettleScheduler;
  firePick: ReturnType<typeof vi.fn>;
  shouldFire: ReturnType<typeof vi.fn>;
  setNow: (ms: number) => void;
  advance: (ms: number) => void;
  flushRaf: () => void;
  restore: () => void;
}

function setup(): Harness {
  let nowMs = 1000;
  let queue: Array<() => void> = [];
  const rafSpy = vi
    .spyOn(globalThis, 'requestAnimationFrame')
    .mockImplementation((cb: FrameRequestCallback) => {
      queue.push(() => cb(nowMs));
      return queue.length;
    });
  const cafSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});
  const firePick = vi.fn();
  const shouldFire = vi.fn(() => true);
  const ctx: SettleSchedulerCtx = {
    now: () => nowMs,
    shouldFire: () => shouldFire(),
    firePick: (x, y) => firePick(x, y),
  };
  const scheduler = new SettleScheduler(ctx);
  return {
    scheduler,
    firePick,
    shouldFire,
    setNow: (ms) => {
      nowMs = ms;
    },
    advance: (ms) => {
      nowMs += ms;
    },
    flushRaf: () => {
      const todo = queue;
      queue = [];
      todo.forEach((fn) => fn());
    },
    restore: () => {
      rafSpy.mockRestore();
      cafSpy.mockRestore();
    },
  };
}

describe('SettleScheduler', () => {
  let h: Harness;

  beforeEach(() => {
    h = setup();
  });

  afterEach(() => {
    h.restore();
  });

  it('fires after the mouse settle window elapses', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledTimes(1);
    expect(h.firePick).toHaveBeenCalledWith(100, 200);
    expect(h.scheduler.lastPickFiredTime).toBeGreaterThan(0);
  });

  it('camera-settle re-pick: markDirty after a fired pick re-arms the loop', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledTimes(1);

    h.advance(10);
    h.scheduler.markDirty();
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledTimes(2);
  });

  it('setSuppressed(true) cancels any pending rAF', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.scheduler.setSuppressed(true);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();
  });

  it('setSuppressed(false) with pending cursor re-arms the rAF', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.scheduler.setSuppressed(true);
    h.scheduler.markDirty();
    h.advance(50);
    h.scheduler.setSuppressed(false);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledTimes(1);
  });

  it('setSuppressed(false) with no pending cursor does NOT schedule rAF', () => {
    h.scheduler.setSuppressed(true);
    h.scheduler.setSuppressed(false);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();
  });

  it('recordMouseLeave clears the pending cursor so no pick fires', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.scheduler.recordMouseLeave();
    h.advance(HOVER_SETTLE_MS + 50);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();
  });

  it('cancelPending drops the armed cursor so no stale pick fires (scroll case)', () => {
    // C2 regression: a page/ancestor scroll invalidates the cached canvas
    // rect, so the already-converted canvas-local pending coordinate is
    // stale. cancelPending() must drop it; the re-pick comes from the next
    // fresh mousemove, not the stale armed one.
    h.scheduler.recordMouseMove(100, 200);
    h.scheduler.cancelPending();
    h.advance(HOVER_SETTLE_MS + 50);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();

    // A fresh mousemove re-arms and fires at the new coordinate.
    h.scheduler.recordMouseMove(300, 400);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledExactlyOnceWith(300, 400);
  });

  it('cancelPending preserves suppression state', () => {
    // Unlike a fresh suppress/resume cycle, cancelPending only drops the
    // pending cursor — it must not flip the suppressed flag.
    h.scheduler.setSuppressed(true);
    h.scheduler.cancelPending();
    expect(h.scheduler.isSuppressed).toBe(true);
  });

  it('recordMouseMove while suppressed tracks the cursor but does not schedule a pick', () => {
    // Contract: the position + timestamp are tracked even while suppressed
    // (so the resume re-pick uses the latest cursor), but no pick fires
    // mid-interaction because the rAF is never scheduled while suppressed.
    h.scheduler.setSuppressed(true);
    h.scheduler.recordMouseMove(100, 200);
    expect(h.scheduler.lastMouseMoveTime).toBe(1000); // tracked (clock starts at 1000)

    // Still suppressed → advancing past the settle window fires nothing.
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();

    // Resume → the tracked position drives the settle re-pick, no fresh
    // mousemove required.
    h.scheduler.setSuppressed(false);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).toHaveBeenCalledExactlyOnceWith(100, 200);
  });

  it('ctx.shouldFire() returning false suppresses the call AND leaves lastPickFiredTime unchanged', () => {
    h.shouldFire.mockReturnValue(false);
    h.scheduler.recordMouseMove(100, 200);
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();
    expect(h.scheduler.lastPickFiredTime).toBe(0);
  });

  it('dispose cancels pending rAF and is idempotent', () => {
    h.scheduler.recordMouseMove(100, 200);
    h.scheduler.dispose();
    expect(() => h.scheduler.dispose()).not.toThrow();
    h.advance(HOVER_SETTLE_MS + 10);
    h.flushRaf();
    expect(h.firePick).not.toHaveBeenCalled();
  });

  it('exposes lastMouseMoveTime / lastDirtyTime / isSuppressed for orchestrator getDiagnostics', () => {
    h.setNow(2000);
    h.scheduler.recordMouseMove(1, 2);
    expect(h.scheduler.lastMouseMoveTime).toBe(2000);

    h.setNow(3000);
    h.scheduler.markDirty();
    expect(h.scheduler.lastDirtyTime).toBe(3000);

    h.scheduler.setSuppressed(true);
    expect(h.scheduler.isSuppressed).toBe(true);
    h.scheduler.setSuppressed(false);
    expect(h.scheduler.isSuppressed).toBe(false);
  });
});
