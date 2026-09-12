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
 * scene's watchdog running.
 *
 * @module core/app/kiosk/apply-kiosk
 */

import type { KioskMode } from '../../../config/kiosk';
import { startKioskWatchdog, type KioskWatchdog } from './watchdog';

/** What applying kiosk mode needs from the app. */
export interface KioskPorts {
  /** Enable or disable routed keyboard handling. */
  setKeyboardEnabled: (enabled: boolean) => void;
  /**
   * Enable or disable the CAMERA CONTROLS.
   *
   * Separate from `setInputEnabled`, and necessarily so: the input handler
   * gates the viewer's own shortcuts and picking, while orbit/fly controls
   * listen on the canvas themselves. Disabling only the first left the camera
   * fully draggable under `?kiosk` — measured at 46x the auto-rotate drift, so
   * a visitor could still swing the view off the tour.
   */
  setControlsEnabled?: (enabled: boolean) => void;
  /** Hide every panel and the rail. */
  hidePanels?: () => void;
  /** The canvas, for the watchdog's context listeners. Absent in tests. */
  canvas?: EventTarget;
  /** Reload the page. Injected so a test never navigates. */
  reload?: () => void;
}

/**
 * Apply pointer and keyboard permissions independently.
 */
function applyInputPermissions(mode: KioskMode, ports: KioskPorts): void {
  if (!mode.allowPointer) ports.setControlsEnabled?.(false);
  if (!mode.allowKeyboard) ports.setKeyboardEnabled(false);
}

/**
 * Apply `mode`. Returns a teardown that undoes the watchdog.
 *
 * A no-op when kiosk mode is off — including the teardown — so a caller can
 * apply unconditionally and not branch.
 */
export function applyKioskMode(mode: KioskMode, ports: KioskPorts): () => void {
  if (!mode.enabled) return () => undefined;

  applyInputPermissions(mode, ports);
  if (!mode.showPanels) ports.hidePanels?.();

  if (!mode.watchdogReload || ports.canvas === undefined) return () => undefined;
  const watchdog: KioskWatchdog = startKioskWatchdog({
    canvas: ports.canvas,
    graceS: mode.watchdogGraceS,
    reload: ports.reload ?? (() => window.location.reload()),
  });
  return () => watchdog.dispose();
}
