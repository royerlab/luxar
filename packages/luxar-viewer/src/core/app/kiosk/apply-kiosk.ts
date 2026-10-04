/**
 * Apply a resolved {@link KioskMode} to a running viewer.
 *
 * Separate from `viewer-config/apply-state.ts` on purpose. That dispatcher
 * applies what the STORE says; kiosk mode is resolved from the store *and* the
 * URL, and the URL has to win. Folding it in would have meant handing the
 * dispatcher a second, differently-sourced input and hoping every future
 * reader noticed which fields came from where.
 *
 * Ports-injected and returning its own teardown, so the whole thing is
 * testable without a browser and a `switchDataset` cannot leave a previous
 * scene's input restrictions or watchdog running.
 *
 * @module core/app/kiosk/apply-kiosk
 */

import { resolveKioskMode, type KioskMode, type ZarrKioskConfig } from '../../../config/kiosk';
import { startKioskWatchdog, type KioskWatchdog } from './watchdog';

/** What applying kiosk mode needs from the app. */
export interface KioskPorts {
  /** Enable or disable routed keyboard handling. */
  setKeyboardEnabled: (enabled: boolean) => void;
  getKeyboardEnabled: () => boolean;
  /**
   * Enable or disable canvas gestures while leaving the camera update loop live.
   *
   * Separate from `setKeyboardEnabled`, and necessarily so: that one reaches
   * `InputContextManager.setEnabled`, whose entire effect is to drop keydown
   * dispatch (`context-manager.ts`: `if (!this.enabled && type === 'down')`) —
   * NOT pointer events, and not picking, which lives elsewhere. Orbit/fly
   * controls listen on the canvas themselves, so disabling only the keyboard
   * left the camera fully draggable under `?kiosk` — measured at 46x the
   * auto-rotate drift, so a visitor could still swing the view off the tour.
   */
  setPointerEnabled: (enabled: boolean) => void;
  getPointerEnabled: () => boolean;
  /** Hide every panel and the rail. */
  hidePanels?: () => void;
  /** The canvas, for the watchdog's context listeners. Absent in tests. */
  canvas?: EventTarget;
  /** Reload the page. Injected so a test never navigates. */
  reload?: () => void;
  /** Subscribe to an unrecoverable GPU loss (WebGPU `device.lost`); returns the unsubscribe. */
  onDeviceLost?: (listener: () => void) => () => void;
}

/**
 * Apply pointer and keyboard permissions independently.
 */
function applyInputPermissions(mode: KioskMode, ports: KioskPorts): () => void {
  const previousPointer = !mode.allowPointer ? ports.getPointerEnabled() : undefined;
  const previousKeyboard = !mode.allowKeyboard ? ports.getKeyboardEnabled() : undefined;
  if (!mode.allowPointer) ports.setPointerEnabled(false);
  if (!mode.allowKeyboard) ports.setKeyboardEnabled(false);
  return () => {
    if (previousPointer !== undefined) ports.setPointerEnabled(previousPointer);
    if (previousKeyboard !== undefined) ports.setKeyboardEnabled(previousKeyboard);
  };
}

/**
 * Apply `mode`. Returns a teardown that restores input and disposes the watchdog.
 *
 * A no-op when kiosk mode is off — including the teardown — so a caller can
 * apply unconditionally and not branch.
 */
export function applyKioskMode(mode: KioskMode, ports: KioskPorts): () => void {
  if (!mode.enabled) return () => undefined;

  const restoreInput = applyInputPermissions(mode, ports);
  if (!mode.showPanels) ports.hidePanels?.();

  const watchdog: KioskWatchdog | undefined =
    mode.watchdogReload && ports.canvas !== undefined
      ? startKioskWatchdog({
          canvas: ports.canvas,
          graceS: mode.watchdogGraceS,
          reload: ports.reload ?? (() => window.location.reload()),
          onUnrecoverableLoss: ports.onDeviceLost,
        })
      : undefined;
  let disposed = false;
  return () => {
    if (disposed) return;
    disposed = true;
    watchdog?.dispose();
    restoreInput();
  };
}

/**
 * Lock the display down when the scene or the URL asks for it: resolve the mode
 * from the authored `ui.kiosk` block and `?kiosk` (the URL wins — see
 * `config/kiosk.ts`), then apply it with the ports `ports()` builds. Returns
 * the input and watchdog teardown (a no-op when kiosk mode is off), which the
 * dataset session owns.
 */
export function applySceneKiosk(
  authored: ZarrKioskConfig | null | undefined,
  urlKiosk: boolean,
  ports: () => KioskPorts
): () => void {
  const mode = resolveKioskMode(authored, urlKiosk);
  // Ports are built only for a mode that is on: the viewer-config pass can run
  // on a partially constructed app, and "kiosk off" must not touch it.
  return mode.enabled ? applyKioskMode(mode, ports()) : () => undefined;
}
