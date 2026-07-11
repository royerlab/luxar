/**
 * Ctrl / Meta hold gate for the wheel-FOV path. Disables wheel zoom while
 * either modifier is held so Ctrl+wheel only adjusts FOV (the wheel
 * handler in `WindowEventHandler` reads the same modifier state).
 *
 * The gate OPENS via the context-gated keydown binding (NAVIGATION only —
 * Ctrl while typing in a panel must not hijack zoom), but it CLOSES by
 * reconciling against the live modifier flags on window-CAPTURE events
 * rather than counting keydown/keyup pairs. A pair-counter sticks: the
 * context manager swallows non-Escape keyups while a text input has focus,
 * and macOS suppresses key events while ⌘ is held, so a missed keyup used
 * to leave `enableZoom` false forever — the "wheel zoom stops working until
 * you alt-tab" bug. Reconciliation needs no pairing: any keyup / pointer
 * event that reports both modifiers up restores zoom immediately.
 */

import { InputContext } from '../context-manager';
import type { KeyBindingsDeps } from './register-all';

export function registerFovHoldGate(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager, cleanups } = deps;

  let gateActive = false;

  const disableZoom = (): void => {
    if (gateActive) return;
    gateActive = true;
    sceneManager.controls.setEnableZoom(false);
  };
  const restoreZoom = (): void => {
    if (!gateActive) return;
    gateActive = false;
    sceneManager.controls.setEnableZoom(true);
  };

  for (const key of ['Control', 'Meta']) {
    contextManager.registerBinding(InputContext.NAVIGATION, {
      key,
      handler: disableZoom,
      description: 'FOV control (hold Ctrl/Cmd + scroll to adjust field of view)',
    });
  }

  // Close the gate from window capture phase — upstream of the context
  // manager's typing-context keyup swallowing, so it can never be missed.
  // On the modifier's own keyup the released key's flag is already false in
  // the event, so "both flags up" is exactly "no modifier still held" (and
  // handles Ctrl+⌘ held together, which the old counter double-counted).
  // The pointer reconcilers cost one boolean check per event while the gate
  // is closed, and also self-heal after a keyup lost to focus changes.
  const reconcile = (e: KeyboardEvent | PointerEvent): void => {
    if (!gateActive) return;
    if (!e.ctrlKey && !e.metaKey) restoreZoom();
  };
  const resetFovKeyStateWhenHidden = (): void => {
    if (document.visibilityState === 'hidden') restoreZoom();
  };

  window.addEventListener('keyup', reconcile, true);
  window.addEventListener('pointermove', reconcile, true);
  window.addEventListener('pointerdown', reconcile, true);
  window.addEventListener('blur', restoreZoom);
  document.addEventListener('visibilitychange', resetFovKeyStateWhenHidden);
  cleanups.push(
    () => window.removeEventListener('keyup', reconcile, true),
    () => window.removeEventListener('pointermove', reconcile, true),
    () => window.removeEventListener('pointerdown', reconcile, true),
    () => window.removeEventListener('blur', restoreZoom),
    () => document.removeEventListener('visibilitychange', resetFovKeyStateWhenHidden)
  );
}
