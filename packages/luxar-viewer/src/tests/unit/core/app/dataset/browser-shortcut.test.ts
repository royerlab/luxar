// @vitest-environment jsdom
/**
 * Unit tests for core/app/dataset/browser-shortcut.ts (G5).
 *
 * `installBrowserShortcut` wires a window `OPEN_DATASET_BROWSER_EVENT` listener
 * through the supplied EventGroup. Contract:
 *   - TOGGLES the browser: fires `showBrowser` when none is open, and
 *     `closeBrowser` when one already is (so the dataset control behaves
 *     like every other panel toggle).
 *   - The hasOpenBrowser predicate is a LIVE accessor — checked every
 *     time the event fires, not snapshotted at install.
 *   - Disposing the EventGroup unregisters the listener.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { installBrowserShortcut } from '../../../../../core/app/dataset/browser-shortcut';
import { OPEN_DATASET_BROWSER_EVENT } from '../../../../../core/app/interaction/canvas-actions';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';

describe('installBrowserShortcut', () => {
  let events: EventGroup;
  let hasOpenBrowser: ReturnType<typeof vi.fn<() => boolean>>;
  let showBrowser: ReturnType<typeof vi.fn<() => void>>;
  let closeBrowser: ReturnType<typeof vi.fn<() => void>>;

  const install = () =>
    installBrowserShortcut({ events, hasOpenBrowser, showBrowser, closeBrowser });

  beforeEach(() => {
    events = new EventGroup();
    hasOpenBrowser = vi.fn<() => boolean>().mockReturnValue(false);
    showBrowser = vi.fn<() => void>();
    closeBrowser = vi.fn<() => void>();
  });

  afterEach(() => {
    events.dispose();
  });

  it('fires showBrowser when the event fires and no browser is open', () => {
    install();
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(showBrowser).toHaveBeenCalledOnce();
    expect(closeBrowser).not.toHaveBeenCalled();
    expect(hasOpenBrowser).toHaveBeenCalledOnce();
  });

  it('fires closeBrowser when a browser is already open (toggle)', () => {
    hasOpenBrowser.mockReturnValue(true);
    install();
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(hasOpenBrowser).toHaveBeenCalledOnce();
    expect(closeBrowser).toHaveBeenCalledOnce();
    expect(showBrowser).not.toHaveBeenCalled();
  });

  it('uses a LIVE predicate — toggling hasOpenBrowser between events flips open/close', () => {
    let isOpen = false;
    const pred = vi.fn<() => boolean>(() => isOpen);
    installBrowserShortcut({ events, hasOpenBrowser: pred, showBrowser, closeBrowser });

    // closed → open
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(showBrowser).toHaveBeenCalledTimes(1);
    expect(closeBrowser).toHaveBeenCalledTimes(0);

    // now open → close
    isOpen = true;
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(showBrowser).toHaveBeenCalledTimes(1);
    expect(closeBrowser).toHaveBeenCalledTimes(1);

    // closed again → open
    isOpen = false;
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(showBrowser).toHaveBeenCalledTimes(2);
    expect(closeBrowser).toHaveBeenCalledTimes(1);
  });

  it('disposing the EventGroup unregisters the listener', () => {
    install();
    events.dispose();
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(hasOpenBrowser).not.toHaveBeenCalled();
    expect(showBrowser).not.toHaveBeenCalled();
    expect(closeBrowser).not.toHaveBeenCalled();
  });

  it('does not respond to unrelated window events', () => {
    install();
    window.dispatchEvent(new Event('focus'));
    window.dispatchEvent(new Event('resize'));
    window.dispatchEvent(new Event('blur'));
    expect(hasOpenBrowser).not.toHaveBeenCalled();
    expect(showBrowser).not.toHaveBeenCalled();
    expect(closeBrowser).not.toHaveBeenCalled();
  });

  it('the toggle callbacks are invoked with no arguments', () => {
    install();
    window.dispatchEvent(new Event(OPEN_DATASET_BROWSER_EVENT));
    expect(showBrowser).toHaveBeenCalledExactlyOnceWith();
  });
});
