/**
 * Unit tests for core/app/dataset/browser-shortcut.ts (G5).
 *
 * `installBrowserShortcut` wires a window `open-dataset-browser` listener
 * through the supplied EventGroup. Contract:
 *   - Fires `showBrowser` when no browser is currently open
 *     (`hasOpenBrowser()` returns false).
 *   - SKIPS `showBrowser` when a browser is already open. This guards
 *     against double-opening that would stack modals over each other.
 *   - The hasOpenBrowser predicate is a LIVE accessor — checked every
 *     time the event fires, not snapshotted at install.
 *   - Disposing the EventGroup unregisters the listener.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { installBrowserShortcut } from '../../../../../core/app/dataset/browser-shortcut';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';

describe('installBrowserShortcut', () => {
  let events: EventGroup;
  let hasOpenBrowser: ReturnType<typeof vi.fn<() => boolean>>;
  let showBrowser: ReturnType<typeof vi.fn<() => void>>;

  beforeEach(() => {
    events = new EventGroup();
    hasOpenBrowser = vi.fn<() => boolean>().mockReturnValue(false);
    showBrowser = vi.fn<() => void>();
  });

  afterEach(() => {
    events.dispose();
  });

  it('fires showBrowser when the event fires and no browser is open', () => {
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });

    window.dispatchEvent(new Event('open-dataset-browser'));

    expect(showBrowser).toHaveBeenCalledOnce();
    expect(hasOpenBrowser).toHaveBeenCalledOnce();
  });

  it('skips showBrowser when the predicate says a browser is already open', () => {
    hasOpenBrowser.mockReturnValue(true);
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });

    window.dispatchEvent(new Event('open-dataset-browser'));

    expect(hasOpenBrowser).toHaveBeenCalledOnce();
    expect(showBrowser).not.toHaveBeenCalled();
  });

  it('uses a LIVE predicate — toggling hasOpenBrowser between events changes behavior', () => {
    // Critical guarantee: the closure reads `hasOpenBrowser()` at event
    // time, not at install time. Otherwise a "browser closed" state
    // change wouldn't let the next event open a fresh one.
    let isOpen = false;
    const pred = vi.fn<() => boolean>(() => isOpen);
    installBrowserShortcut({ events, hasOpenBrowser: pred, showBrowser });

    // First event: browser is closed → show.
    window.dispatchEvent(new Event('open-dataset-browser'));
    expect(showBrowser).toHaveBeenCalledTimes(1);

    // Simulate that the modal is now open.
    isOpen = true;
    window.dispatchEvent(new Event('open-dataset-browser'));
    expect(showBrowser).toHaveBeenCalledTimes(1); // unchanged: skipped

    // Simulate that the modal closed again.
    isOpen = false;
    window.dispatchEvent(new Event('open-dataset-browser'));
    expect(showBrowser).toHaveBeenCalledTimes(2);
  });

  it('disposing the EventGroup unregisters the listener', () => {
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });
    events.dispose();

    window.dispatchEvent(new Event('open-dataset-browser'));

    expect(hasOpenBrowser).not.toHaveBeenCalled();
    expect(showBrowser).not.toHaveBeenCalled();
  });

  it('does not respond to unrelated window events', () => {
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });

    window.dispatchEvent(new Event('focus'));
    window.dispatchEvent(new Event('resize'));
    window.dispatchEvent(new Event('blur'));

    expect(hasOpenBrowser).not.toHaveBeenCalled();
    expect(showBrowser).not.toHaveBeenCalled();
  });

  it('multiple events fire showBrowser each time when no browser is open', () => {
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });

    window.dispatchEvent(new Event('open-dataset-browser'));
    window.dispatchEvent(new Event('open-dataset-browser'));
    window.dispatchEvent(new Event('open-dataset-browser'));

    expect(showBrowser).toHaveBeenCalledTimes(3);
  });

  it('the showBrowser callback is invoked with no arguments', () => {
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser });

    window.dispatchEvent(new Event('open-dataset-browser'));

    // Listener wraps showBrowser in an arrow that drops its event arg
    // (`() => ports.showBrowser()`). Asserting `()` rules out a future
    // change that accidentally forwards the Event.
    expect(showBrowser).toHaveBeenCalledExactlyOnceWith();
  });
});
