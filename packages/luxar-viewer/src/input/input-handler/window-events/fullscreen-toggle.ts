/**
 * Toggle fullscreen mode body extracted from input-handler.ts. Enters
 * fullscreen on the document element (true fullscreen, browser chrome
 * included); falls back to the WebGL canvas if the document request
 * rejects; exits fullscreen otherwise.
 *
 * @module input/input-handler/window-events/fullscreen-toggle
 */

import { log, Modules } from '../../../utils/log';
import type { SceneManager } from '../../../scene/scene-manager';

export interface FullscreenCtx {
  sceneManager: SceneManager;
}

export function toggleFullscreen(ctx: FullscreenCtx): void {
  if (!document.fullscreenElement) {
    // Enter fullscreen — target the document element for true fullscreen
    document.documentElement.requestFullscreen().catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to enable fullscreen:', err);
      // Fallback: try the canvas element
      ctx.sceneManager.renderer.domElement.requestFullscreen().catch((fallbackErr) => {
        log.error(Modules.INPUT, 'Fallback fullscreen also failed:', fallbackErr);
      });
    });
  } else {
    document.exitFullscreen().catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to exit fullscreen:', err);
    });
  }
}
