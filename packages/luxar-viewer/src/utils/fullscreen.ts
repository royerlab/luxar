/**
 * Cross-browser fullscreen-state helper.
 *
 * Safari < 16.4 exposes only the webkit-prefixed fullscreen API, so a plain
 * `document.fullscreenElement` check misses fullscreen entered via
 * `webkitRequestFullscreen`. Every consumer that keys behaviour off "are we
 * fullscreen?" (canvas sizing, Escape handling, the control-rail auto-hide,
 * the window-event resize) must use this so the standard and webkit paths stay
 * in sync — otherwise one site takes the wrong branch on legacy Safari.
 *
 * @module utils/fullscreen
 */

interface WebkitFullscreenDocument extends Document {
  webkitFullscreenElement?: Element | null;
}

/** True when any element is currently fullscreen (standard or webkit). */
export function isDocumentFullscreen(): boolean {
  const d = document as WebkitFullscreenDocument;
  return !!(document.fullscreenElement || d.webkitFullscreenElement);
}
