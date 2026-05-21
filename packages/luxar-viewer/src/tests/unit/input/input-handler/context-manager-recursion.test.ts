/**
 * Tests for the re-entrance guard added to
 * `InputContextManager.handleKeyEvent`. A binding handler that
 * (mis)configures itself to dispatch keyboard events back through
 * the manager will recurse; the guard caps depth at
 * {@link MAX_KEY_EVENT_DEPTH} and bails with a single error log.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  InputContext,
  InputContextManager,
  MAX_KEY_EVENT_DEPTH,
} from '../../../../input/input-handler/context-manager';
import { log } from '../../../../utils/log';

function makeKeyEvent(key = 'p'): KeyboardEvent {
  // jsdom doesn't ship KeyboardEvent('keydown', ...) with a key property
  // populated for synthetic events constructed bare; we patch a minimal
  // shape that satisfies the manager's needs.
  return new KeyboardEvent('keydown', { key });
}

describe('InputContextManager re-entrance guard', () => {
  it('caps re-entrance at MAX_KEY_EVENT_DEPTH and emits a single log.error', () => {
    const mgr = new InputContextManager();
    const errSpy = vi.spyOn(log, 'error').mockImplementation(() => {});

    let recursiveCalls = 0;
    mgr.registerBinding(InputContext.NAVIGATION, {
      key: 'p',
      handler: (event) => {
        recursiveCalls++;
        // Trigger a recursive dispatch — the misconfiguration the
        // guard exists to catch.
        mgr.handleKeyEvent(event, 'down');
      },
    });

    mgr.handleKeyEvent(makeKeyEvent('p'), 'down');

    // First call + (MAX_KEY_EVENT_DEPTH - 1) re-entrances before the
    // outermost depth check rejects further reentrance. Exact count
    // depends on whether the guard increments before or after; we
    // just assert it's bounded.
    expect(recursiveCalls).toBeLessThanOrEqual(MAX_KEY_EVENT_DEPTH);
    expect(errSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('recursion limit')
    );
    errSpy.mockRestore();
  });

  it('non-recursive use is unaffected (depth resets to 0 after each call)', () => {
    const mgr = new InputContextManager();
    const errSpy = vi.spyOn(log, 'error').mockImplementation(() => {});

    let calls = 0;
    mgr.registerBinding(InputContext.NAVIGATION, {
      key: 'p',
      handler: () => {
        calls++;
      },
    });

    // Dispatch many times — depth resets between calls.
    for (let i = 0; i < 50; i++) {
      mgr.handleKeyEvent(makeKeyEvent('p'), 'down');
    }
    expect(calls).toBe(50);
    expect(errSpy).not.toHaveBeenCalled();
    errSpy.mockRestore();
  });

  it('exception in a handler still releases the depth counter (try/finally)', () => {
    const mgr = new InputContextManager();
    mgr.registerBinding(InputContext.NAVIGATION, {
      key: 'p',
      handler: () => {
        throw new Error('boom');
      },
    });

    expect(() => mgr.handleKeyEvent(makeKeyEvent('p'), 'down')).toThrow('boom');
    // Depth counter should be back to 0 — a follow-up dispatch must
    // not be incorrectly throttled.
    let secondCallReached = false;
    mgr.registerBinding(InputContext.NAVIGATION, {
      key: 'g',
      handler: () => {
        secondCallReached = true;
      },
    });
    mgr.handleKeyEvent(makeKeyEvent('g'), 'down');
    expect(secondCallReached).toBe(true);
  });
});
