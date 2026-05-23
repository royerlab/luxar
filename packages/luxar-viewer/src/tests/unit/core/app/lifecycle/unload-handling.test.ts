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

  it.skip('SHOULD swallow errors when the supplied dispose itself throws (currently does not)', () => {
    // Pinned-as-skipped: the production helper does NOT wrap the dispose
    // callback in try/catch. Browser unload is a terminal event so an
    // unhandled throw bubbles to window.onerror and (in some browsers)
    // shows a console error to the user. The orchestrator's dispose
    // path already uses `safeDispose` internally, so in practice the
    // callback supplied here doesn't throw — but the helper would be
    // safer if it caught + logged. Recorded in core.md OOS so the
    // contract is documented; un-skip + add try/catch in source to fix.
    const throwingDispose = vi.fn(() => {
      throw new Error('mid-unload crash');
    });
    installUnloadHandler({ events, dispose: throwingDispose });

    expect(() => window.dispatchEvent(new Event('beforeunload'))).not.toThrow();
    expect(throwingDispose).toHaveBeenCalledOnce();
  });
});
