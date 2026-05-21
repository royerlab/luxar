/**
 * Unit tests for the pure two-axis-settle decision predicate.
 *
 * `evaluateSettle()` is a pure function of four timestamps. These
 * tests exercise every branch directly with hand-crafted inputs;
 * the orchestrator's settle-scheduler integration tests
 * (`picking-system.test.ts`) cover the rAF lifecycle + wiring on top.
 */

import { describe, it, expect } from 'vitest';
import {
  evaluateSettle,
  HOVER_SETTLE_MS,
  type SettleTimestamps,
} from '../../../../../rendering/picking/picking-system/settle-loop';

function ts(partial: Partial<SettleTimestamps>): SettleTimestamps {
  return {
    now: 0,
    lastMouseMoveTime: 0,
    lastDirtyTime: 0,
    lastPickFiredTime: 0,
    ...partial,
  };
}

describe('evaluateSettle', () => {
  it('waits when mouse moved within the settle window', () => {
    expect(
      evaluateSettle(
        ts({ now: 1000, lastMouseMoveTime: 950, lastDirtyTime: 0, lastPickFiredTime: 0 })
      )
    ).toEqual({ action: 'wait' });
  });

  it('waits when camera dirtied within the settle window', () => {
    expect(
      evaluateSettle(
        ts({ now: 1000, lastMouseMoveTime: 0, lastDirtyTime: 950, lastPickFiredTime: 0 })
      )
    ).toEqual({ action: 'wait' });
  });

  it('fires when both settled and a new hover happened since the last pick', () => {
    // lastPickFiredTime = 100; mouse moved at 500; now = 700 (both axes >= 120 ms quiet).
    expect(
      evaluateSettle(
        ts({ now: 700, lastMouseMoveTime: 500, lastDirtyTime: 0, lastPickFiredTime: 100 })
      )
    ).toEqual({ action: 'fire' });
  });

  it('fires when both settled and a new camera dirty happened since the last pick (re-pick-on-camera-settle)', () => {
    expect(
      evaluateSettle(
        ts({ now: 700, lastMouseMoveTime: 100, lastDirtyTime: 500, lastPickFiredTime: 200 })
      )
    ).toEqual({ action: 'fire' });
  });

  it('is idle when both settled but nothing has changed since the last pick', () => {
    expect(
      evaluateSettle(
        ts({ now: 1000, lastMouseMoveTime: 100, lastDirtyTime: 200, lastPickFiredTime: 500 })
      )
    ).toEqual({ action: 'idle' });
  });

  it('treats the exact settle boundary as settled (closed bound)', () => {
    // Exactly HOVER_SETTLE_MS elapsed on each axis — should NOT wait.
    expect(
      evaluateSettle(
        ts({
          now: HOVER_SETTLE_MS,
          lastMouseMoveTime: 0,
          lastDirtyTime: 0,
          lastPickFiredTime: -1,
        })
      )
    ).toEqual({ action: 'fire' });
  });

  it('one nanosecond short of the boundary still waits', () => {
    expect(
      evaluateSettle(
        ts({
          now: HOVER_SETTLE_MS - 0.001,
          lastMouseMoveTime: 0,
          lastDirtyTime: 0,
          lastPickFiredTime: -1,
        })
      )
    ).toEqual({ action: 'wait' });
  });
});
