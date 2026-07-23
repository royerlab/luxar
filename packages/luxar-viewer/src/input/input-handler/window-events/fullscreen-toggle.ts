/**
 * Toggle fullscreen mode body extracted from input-handler.ts. Enters
 * fullscreen on the document element (true fullscreen, browser chrome
 * included); exits fullscreen otherwise.
 *
 * There is deliberately NO canvas-element fallback: the rejection causes for
 * `documentElement.requestFullscreen()` (no user gesture, permissions-policy,
 * `fullscreenEnabled` false) apply equally to the canvas, and a canvas-only
 * fullscreen would hide every DOM overlay (control rail, panels) — worse than
 * not entering fullscreen at all. (The historical fallback was a mechanical
 * carry-over from the pre-extraction handler, not a considered feature.)
 *
 * @module input/input-handler/window-events/fullscreen-toggle
 */

import { log, Modules } from '../../../utils/log';
import { isDocumentFullscreen } from '../../../utils/fullscreen';

/** Safari (< 16.4) exposes only the webkit-prefixed fullscreen API. */
interface WebkitFullscreenElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void> | void;
}
interface WebkitExitFullscreenDocument extends Document {
  webkitExitFullscreen?: () => Promise<void> | void;
}

function requestFullscreen(el: HTMLElement): Promise<void> | void {
  const e = el as WebkitFullscreenElement;
  return el.requestFullscreen ? el.requestFullscreen() : e.webkitRequestFullscreen?.();
}

function exitFullscreen(): Promise<void> | void {
  const d = document as WebkitExitFullscreenDocument;
  return document.exitFullscreen ? document.exitFullscreen() : d.webkitExitFullscreen?.();
}

export function toggleFullscreen(): void {
  if (!isDocumentFullscreen()) {
    // Enter fullscreen — target the document element for true fullscreen.
    Promise.resolve(requestFullscreen(document.documentElement)).catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to enable fullscreen:', err);
    });
  } else {
    Promise.resolve(exitFullscreen()).catch((err) => {
      log.error(Modules.INPUT, 'Error attempting to exit fullscreen:', err);
    });
  }
}
