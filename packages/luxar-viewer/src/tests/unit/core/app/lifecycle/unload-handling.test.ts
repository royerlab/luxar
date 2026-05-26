/**
 * Unit tests for core/app/lifecycle/unload-handling.ts (G4).
 *
 * `installUnloadHandler` wires a `beforeunload` listener on window
 * through the supplied EventGroup so it's cleaned up on dispose. The
 * contract is:
 *   - The supplied `dispose` callback fires when window emits
 *     `beforeunload`.
 *   - The listener is registered via `EventGroup.on` (so disposing the
 *     group removes the listener — no global window leak across reloads).
 *   - Disposing the EventGroup must stop the listener from firing.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { installUnloadHandler } from '../../../../../core/app/lifecycle/unload-handling';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { log } from '../../../../../utils/log';

describe('installUnloadHandler', () => {
  let events: EventGroup;
  let dispose: ReturnType<typeof vi.fn<() => void>>;

  beforeEach(() => {
    events = new EventGroup();
    dispose = vi.fn<() => void>();
  });

  afterEach(() => {
    // Disposing the group removes the registered listener so leaks
    // don't bleed into sibling tests (multiple installUnloadHandler
    // calls would otherwise stack up).
    events.dispose();
  });

  it('fires the supplied dispose callback when window beforeunload is dispatched', () => {
    installUnloadHandler({ events, dispose });

    window.dispatchEvent(new Event('beforeunload'));

    expect(dispose).toHaveBeenCalledOnce();
  });

  it('registers exactly one listener (idempotency guard if called once)', () => {
    installUnloadHandler({ events, dispose });
    window.dispatchEvent(new Event('beforeunload'));
    window.dispatchEvent(new Event('beforeunload'));

    // Two events → two listener invocations. The helper does NOT
    // debounce or auto-remove on first call; the unload semantics
    // require the listener to be alive until disposal.
    expect(dispose).toHaveBeenCalledTimes(2);
  });

  it('disposing the EventGroup unregisters the listener', () => {
    installUnloadHandler({ events, dispose });
    events.dispose();

    window.dispatchEvent(new Event('beforeunload'));

    // After dispose the callback must not fire — otherwise a re-mounted
    // LuxarApp on the same page would see the OLD instance's dispose
    // running on unload, double-disposing scene managers across the
    // page lifecycle.
    expect(dispose).not.toHaveBeenCalled();
  });

  it('installing twice on the same group results in two callback fires per event', () => {
    // Two installs = two listeners. The helper is intentionally
    // un-deduplicated — the caller (LuxarApp.init) is responsible for
    // calling it exactly once per init.
    installUnloadHandler({ events, dispose });
    installUnloadHandler({ events, dispose });

    window.dispatchEvent(new Event('beforeunload'));

    expect(dispose).toHaveBeenCalledTimes(2);
  });

  it('two separate EventGroups can each carry their own listener', () => {
    const otherEvents = new EventGroup();
    const otherDispose = vi.fn<() => void>();
    try {
      installUnloadHandler({ events, dispose });
      installUnloadHandler({ events: otherEvents, dispose: otherDispose });

      window.dispatchEvent(new Event('beforeunload'));

      expect(dispose).toHaveBeenCalledOnce();
      expect(otherDispose).toHaveBeenCalledOnce();
    } finally {
      otherEvents.dispose();
    }
  });

  // [core OOS] Un-skipped: the production helper now wraps the dispose
  // callback in try/catch + log.warning. The unload-handler can no
  // longer crash if dispose throws, so subsequent beforeunload listeners
  // (or browser-internal cleanup) still run.
  it('swallows errors when the supplied dispose throws AND log.warning is called', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const throwingDispose = vi.fn(() => {
      throw new Error('mid-unload crash');
    });
    installUnloadHandler({ events, dispose: throwingDispose });

    // The event dispatch must NOT throw — try/catch in the handler
    // swallows the failure.
    expect(() => window.dispatchEvent(new Event('beforeunload'))).not.toThrow();
    expect(throwingDispose).toHaveBeenCalledOnce();

    // log.warning fires with the expected shape (module, message, error).
    expect(warnSpy).toHaveBeenCalledWith(
      'Luxar',
      expect.stringMatching(/dispose\(\) threw during beforeunload.*mid-unload crash/),
      expect.any(Error)
    );

    warnSpy.mockRestore();
  });
});
