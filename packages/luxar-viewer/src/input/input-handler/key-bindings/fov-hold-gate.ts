/**
 * Ctrl / Meta hold gate for the wheel-FOV path. Disables wheel zoom while
 * either modifier is held so Ctrl+wheel only adjusts FOV (the wheel
 * handler in `WindowEventHandler` reads the same modifier state). A
 * counter gates the toggle so releasing one key while the other is still
 * held doesn't re-enable zoom prematurely. A `blur` and a
 * `visibilitychange` listener reset the counter when the page loses
 * focus — otherwise a key release that happens off-page would leave
 * zoom permanently disabled.
 */

import { InputContext } from '../context-manager';
import type { KeyBindingsDeps } from './register-all';

export function registerFovHoldGate(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager, cleanups } = deps;

  let fovKeyHeldCount = 0;
  const resetFovKeyState = (): void => {
    if (fovKeyHeldCount === 0) return;
    fovKeyHeldCount = 0;
    sceneManager.controls.setEnableZoom(true);
  };
  const resetFovKeyStateWhenHidden = (): void => {
    if (document.visibilityState === 'hidden') resetFovKeyState();
  };

  for (const key of ['Control', 'Meta']) {
    contextManager.registerBinding(InputContext.NAVIGATION, {
      key,
      handler: () => {
        fovKeyHeldCount++;
        sceneManager.controls.setEnableZoom(false);
      },
      keyupHandler: () => {
        fovKeyHeldCount = Math.max(0, fovKeyHeldCount - 1);
        if (fovKeyHeldCount === 0) {
          sceneManager.controls.setEnableZoom(true);
        }
      },
      description: 'FOV control (hold Ctrl/Cmd + scroll to adjust field of view)',
    });
  }

  window.addEventListener('blur', resetFovKeyState);
  document.addEventListener('visibilitychange', resetFovKeyStateWhenHidden);
  cleanups.push(
    () => window.removeEventListener('blur', resetFovKeyState),
    () => document.removeEventListener('visibilitychange', resetFovKeyStateWhenHidden)
  );
}
