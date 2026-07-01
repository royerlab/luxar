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

/** Safari (< 16.4) exposes only the webkit-prefixed fullscreen API. */
interface WebkitFullscreenElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void> | void;
}
interface WebkitFullscreenDocument extends Document {
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => Promise<void> | void;
}

/** True when any element is currently fullscreen (standard or webkit). */
export function isDocumentFullscreen(): boolean {
  const d = document as WebkitFullscreenDocument;
  return !!(document.fullscreenElement || d.webkitFullscreenElement);
}

function requestFullscreen(el: HTMLElement): Promise<void> | void {
  const e = el as WebkitFullscreenElement;
  return el.requestFullscreen ? el.requestFullscreen() : e.webkitRequestFullscreen?.();
}

function exitFullscreen(): Promise<void> | void {
  const d = document as WebkitFullscreenDocument;
  return document.exitFullscreen ? document.exitFullscreen() : d.webkitExitFullscreen?.();
}

export function toggleFullscreen(ctx: FullscreenCtx): void {
  if (!isDocumentFullscreen()) {
    // Enter fullscreen — target the document element for true fullscreen.
    Promise.resolve(requestFullscreen(document.documentElement)).catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to enable fullscreen:', err);
      // Fallback: try the canvas element.
      Promise.resolve(requestFullscreen(ctx.sceneManager.renderer.domElement)).catch(
        (fallbackErr) => {
          log.error(Modules.INPUT, 'Fallback fullscreen also failed:', fallbackErr);
        }
      );
    });
  } else {
    Promise.resolve(exitFullscreen()).catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to exit fullscreen:', err);
    });
  }
}
