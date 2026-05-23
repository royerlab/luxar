/**
 * Unit tests for the data-monitor-cycle command body.
 *
 * input.md G3 fix: source module was entirely uncovered. The body
 * emits a `panel-cycle` event on the eventBus with the
 * `data-monitor` panel id and logs an info message. Tests subscribe
 * to the real event bus (it's the dependency-inversion seam) and
 * assert on the emitted payload.
 */

import { describe, it, expect, vi } from 'vitest';
import { cycleDataMonitor } from '../../../../../input/input-handler/commands/data-monitor-cycle';
import { eventBus } from '../../../../../utils/cross-layer/event-bus';

describe('cycleDataMonitor', () => {
  it('emits a panel-cycle event with panelId="data-monitor"', () => {
    const listener = vi.fn();
    const unsub = eventBus.on('panel-cycle', listener);
    try {
      cycleDataMonitor();
      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener).toHaveBeenCalledWith({ panelId: 'data-monitor' });
    } finally {
      unsub();
    }
  });

  it('emits exactly once per call (no duplicate dispatch)', () => {
    const listener = vi.fn();
    const unsub = eventBus.on('panel-cycle', listener);
    try {
      cycleDataMonitor();
      cycleDataMonitor();
      cycleDataMonitor();
      expect(listener).toHaveBeenCalledTimes(3);
      // All three calls carry the same payload (no mutation between dispatches).
      listener.mock.calls.forEach((args) => {
        expect(args[0]).toEqual({ panelId: 'data-monitor' });
      });
    } finally {
      unsub();
    }
  });

  it('does NOT throw when no listeners are registered', () => {
    // Sanity: the event bus emit path tolerates an empty subscriber list.
    expect(() => cycleDataMonitor()).not.toThrow();
  });
});
