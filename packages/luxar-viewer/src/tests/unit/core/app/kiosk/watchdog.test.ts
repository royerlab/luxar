/**
 * The watchdog's whole value is that it does NOT fire when recovery works.
 *
 * A reload on an exhibit display throws away every warm cache it has built —
 * seconds of blank screen on a large scene — so "recovered in time" and
 * "flapping context" are the two cases worth pinning.
 */

import { describe, expect, it, vi } from 'vitest';

import { startKioskWatchdog } from '../../../../../core/app/kiosk/watchdog';
import { applyKioskMode } from '../../../../../core/app/kiosk/apply-kiosk';
import { KIOSK_MODE_OFF, resolveKioskMode } from '../../../../../config/kiosk';

/** A minimal EventTarget plus controllable timers. */
function harness(graceS = 10) {
  const canvas = new EventTarget();
  const reload = vi.fn();
  const onRecovered = vi.fn();
  let nextHandle = 1;
  const timers = new Map<number, () => void>();
  const watchdog = startKioskWatchdog({
    canvas,
    graceS,
    reload,
    onRecovered,
    setTimer: (handler) => {
      const handle = nextHandle++;
      timers.set(handle, handler);
      return handle;
    },
    clearTimer: (handle) => {
      timers.delete(handle);
    },
  });
  return {
    canvas,
    reload,
    onRecovered,
    watchdog,
    pending: () => timers.size,
    fireTimers: () => {
      for (const handler of [...timers.values()]) handler();
      timers.clear();
    },
  };
}

describe('startKioskWatchdog', () => {
  it('does nothing until the context is lost', () => {
    const h = harness();
    expect(h.pending()).toBe(0);
    h.fireTimers();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it('reloads when the grace period expires with no recovery', () => {
    const h = harness();
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(h.pending()).toBe(1);
    h.fireTimers();
    expect(h.reload).toHaveBeenCalledTimes(1);
  });

  it('does NOT reload when the context comes back in time', () => {
    // The case that matters: the viewer's own recovery gets first refusal.
    const h = harness();
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    h.canvas.dispatchEvent(new Event('webglcontextrestored'));
    expect(h.pending()).toBe(0);
    h.fireTimers();
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.onRecovered).toHaveBeenCalledTimes(1);
  });

  it('does not restart the clock on a second loss', () => {
    // A context that flaps is exactly what the reload is for; restarting the
    // timer on every flap could postpone it forever.
    const h = harness();
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(h.pending()).toBe(1);
    h.fireTimers();
    expect(h.reload).toHaveBeenCalledTimes(1);
  });

  it('cancels a pending reload on dispose', () => {
    // Otherwise a `switchDataset` reloads the page some seconds later for a
    // context that belonged to the previous scene.
    const h = harness();
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    h.watchdog.dispose();
    expect(h.pending()).toBe(0);
    h.fireTimers();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it('stops listening after dispose', () => {
    const h = harness();
    h.watchdog.dispose();
    h.canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(h.pending()).toBe(0);
  });

  it('is idempotent on dispose', () => {
    const h = harness();
    h.watchdog.dispose();
    expect(() => h.watchdog.dispose()).not.toThrow();
  });
});

describe('applyKioskMode', () => {
  function ports() {
    return {
      setInputEnabled: vi.fn(),
      hidePanels: vi.fn(),
      reload: vi.fn(),
    };
  }

  it('does nothing at all when kiosk mode is off', () => {
    const p = ports();
    const teardown = applyKioskMode(KIOSK_MODE_OFF, p);
    expect(p.setInputEnabled).not.toHaveBeenCalled();
    expect(p.hidePanels).not.toHaveBeenCalled();
    // The teardown must be safe to call unconditionally.
    expect(() => teardown()).not.toThrow();
  });

  it('disables input and hides panels when locked down', () => {
    const p = ports();
    applyKioskMode(resolveKioskMode({ enabled: true }, false), p);
    expect(p.setInputEnabled).toHaveBeenCalledWith(false);
    expect(p.hidePanels).toHaveBeenCalledTimes(1);
  });

  it('leaves input alone when the author allowed both', () => {
    // A touch-driven exhibit: locked panels, live canvas.
    const p = ports();
    applyKioskMode(
      resolveKioskMode({ enabled: true, allow_pointer: true, allow_keyboard: true }, false),
      p
    );
    expect(p.setInputEnabled).not.toHaveBeenCalled();
    expect(p.hidePanels).toHaveBeenCalledTimes(1);
  });

  it('keeps panels when the author asked for them', () => {
    const p = ports();
    applyKioskMode(resolveKioskMode({ enabled: true, show_panels: true }, false), p);
    expect(p.hidePanels).not.toHaveBeenCalled();
  });

  it('starts no watchdog without a canvas', () => {
    // Its absence is the normal case in a test and on a page that has not
    // built a renderer yet; it must not throw.
    const p = ports();
    const teardown = applyKioskMode(
      resolveKioskMode({ enabled: true, watchdog_reload: true }, false),
      p
    );
    expect(() => teardown()).not.toThrow();
  });

  it('wires the watchdog to the canvas when asked', () => {
    const p = ports();
    const canvas = new EventTarget();
    const teardown = applyKioskMode(
      resolveKioskMode({ enabled: true, watchdog_reload: true, watchdog_grace_s: 0 }, false),
      { ...p, canvas }
    );
    // grace 0 means the timer is scheduled with a zero delay; assert it was
    // armed rather than racing a real timer.
    canvas.dispatchEvent(new Event('webglcontextlost'));
    teardown();
    expect(p.reload).not.toHaveBeenCalled();
  });

  it('schedules on the platform timer when none is injected', async () => {
    // The DEFAULT port path, asserted on what it DOES rather than on what it
    // avoids. It used to call `window.setTimeout` in a suite whose default
    // environment is node, and the resulting `window is not defined` escaped
    // as an unhandled error from inside a dispatched listener: the run printed
    // "Errors 1" and still reported every test as passing. The test above
    // could not catch it either — it asserts `reload` was NOT called, which is
    // equally true of code that threw before arming anything.
    const reload = vi.fn();
    const canvas = new EventTarget();
    const watchdog = startKioskWatchdog({ canvas, graceS: 0, reload });

    canvas.dispatchEvent(new Event('webglcontextlost'));
    await new Promise((resolve) => setTimeout(resolve, 5));

    expect(reload).toHaveBeenCalledTimes(1);
    watchdog.dispose();
  });

  it('cancels the platform timer on recovery, with no ports injected', async () => {
    const reload = vi.fn();
    const onRecovered = vi.fn();
    const canvas = new EventTarget();
    const watchdog = startKioskWatchdog({ canvas, graceS: 0, reload, onRecovered });

    canvas.dispatchEvent(new Event('webglcontextlost'));
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    await new Promise((resolve) => setTimeout(resolve, 5));

    // The default `clearTimer` has to work too, or a recovered display would
    // reload anyway — which is the one thing this module exists to avoid.
    expect(reload).not.toHaveBeenCalled();
    expect(onRecovered).toHaveBeenCalledTimes(1);
    watchdog.dispose();
  });
});
