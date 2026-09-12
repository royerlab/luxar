/**
 * Reload an unattended display whose WebGL context never comes back.
 *
 * The last resort, and deliberately last. The viewer already recovers from
 * context loss on its own (`scene/scene-manager/render-pipeline/
 * webgl-context-recovery.ts`), and a reload throws away every warm cache the
 * display has built — on a large scene that is a visibly blank screen for
 * seconds. So this waits out a grace period and only fires if recovery has not
 * happened.
 *
 * It listens on the canvas ITSELF rather than hooking the recovery module.
 * That is not a shortcut: DOM events have many listeners, so observing the same
 * `webglcontextlost` / `webglcontextrestored` pair needs no plumbing through
 * three layers, cannot interfere with recovery, and inherently gives recovery
 * first refusal — the grace timer is simply cancelled if the context returns.
 *
 * Ports-injected (timers, reload, canvas) so a test can drive it without a
 * browser, and so nothing here reaches for a singleton it was not handed.
 *
 * @module core/app/kiosk/watchdog
 */

export interface KioskWatchdogPorts {
  /** The canvas whose context is being watched. */
  canvas: EventTarget;
  /** Seconds to wait for recovery before reloading. */
  graceS: number;
  /** Reload the page. Injected so a test never actually navigates. */
  reload: () => void;
  /** Defaults to `setTimeout`. */
  setTimer?: (handler: () => void, ms: number) => number;
  /** Defaults to `clearTimeout`. */
  clearTimer?: (handle: number) => void;
  /** Called instead of reloading when the context comes back in time. */
  onRecovered?: () => void;
}

export interface KioskWatchdog {
  /** Stop watching and cancel any pending reload. Idempotent. */
  dispose(): void;
}

/**
 * Start watching. Returns a handle whose `dispose` removes the listeners.
 *
 * A second loss while a reload is already pending does NOT restart the clock:
 * a context that flaps is exactly the case the reload exists for, and
 * restarting the timer on each flap could postpone it forever.
 */
export function startKioskWatchdog(ports: KioskWatchdogPorts): KioskWatchdog {
  const setTimer = ports.setTimer ?? ((handler, ms) => window.setTimeout(handler, ms));
  const clearTimer = ports.clearTimer ?? ((handle) => window.clearTimeout(handle));
  let pending: number | undefined;
  let disposed = false;

  const onLost = (): void => {
    if (disposed || pending !== undefined) return;
    pending = setTimer(() => {
      pending = undefined;
      if (!disposed) ports.reload();
    }, ports.graceS * 1000);
  };

  const onRestored = (): void => {
    if (pending === undefined) return;
    clearTimer(pending);
    pending = undefined;
    ports.onRecovered?.();
  };

  ports.canvas.addEventListener('webglcontextlost', onLost);
  ports.canvas.addEventListener('webglcontextrestored', onRestored);

  return {
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (pending !== undefined) {
        clearTimer(pending);
        pending = undefined;
      }
      ports.canvas.removeEventListener('webglcontextlost', onLost);
      ports.canvas.removeEventListener('webglcontextrestored', onRestored);
    },
  };
}
