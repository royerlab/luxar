/**
 * Native context-menu ownership — the secondary click belongs to the viewer on
 * every surface the viewer owns, not just on the canvas.
 *
 * Right-drag orbits the camera, so both controls implementations already
 * `preventDefault` the `contextmenu` event on the canvas they listen to
 * (`luxar-orbit-controls.ts`, `luxar-fly-controls/listeners.ts`), and
 * `canvas-actions.ts` repeats it. In Chromium that is enough: every scene
 * overlay floating above the canvas is `pointer-events: none`, so the hit test
 * walks past the overlay and the event target IS the canvas.
 *
 * WebKit resolves the context-menu target geometrically instead, so inside the
 * WKWebView of an exported `--native macos` app the target is whichever
 * overlay happens to sit under the cursor — a turntable's matte `<canvas>`,
 * its `<video>`, a logo `<img>`. None of those is the canvas and none of their
 * ancestors carried a handler, so the native "Copy Image / Share Image…" sheet
 * opened over the scene the moment a right-drag began, and the camera never
 * moved. Reported against the ESM kiosk export, 2026-09-09.
 *
 * Rather than chase each overlay, one delegated listener on the viewer
 * container covers everything the viewer mounts — present and future,
 * `pointer-events` or not. It is scoped two ways so it cannot eat a menu that
 * is not ours to eat:
 *
 *   - **Ownership.** Only a target that is the canvas, or that sits inside an
 *     element carrying a `luxar-` class, is suppressed. The container defaults
 *     to `document.body`, which an embedder may share with their own UI (see
 *     `utils/viewer-container`), so "inside the container" alone is too broad.
 *   - **Text fields.** An `input`, `textarea` or `contenteditable` keeps its
 *     native menu, which is the only way to paste — the layers panel already
 *     carves out the same exception for its filter box and bound editor, and
 *     the dataset browser's URL field needs it too.
 *
 * Capture phase, so an inner handler that calls `stopPropagation` (the
 * dimension sliders' play-button menu does) cannot leave the native menu
 * behind it.
 *
 * @module core/app/interaction/context-menu-ownership
 */

import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { getViewerContainer } from '../../../utils/viewer-container';

/**
 * Any element the viewer mounted. Every viewer-authored class is `luxar-`
 * prefixed (`luxar-overlay`, `luxar-control-rail__btn`, `luxar-gui`, …), and
 * an unclassed child — the `<img>` inside an image overlay, a matte `<canvas>`
 * — reaches one through `closest`.
 */
const VIEWER_OWNED = '[class*="luxar-"]';

/** Where the native menu is the only clipboard the user has. */
const TEXT_ENTRY = 'input, textarea, [contenteditable=""], [contenteditable="true"]';

/**
 * Suppress the browser's context menu across the viewer's own DOM.
 *
 * @param canvas - The 3D canvas, when the app owns one. It carries no
 *   `luxar-` class of its own (the standalone player styles `#app`), so it is
 *   matched by identity rather than by selector.
 * @param events - The app's listener group, so `dispose()` removes this too.
 */
export function installContextMenuOwnership(
  canvas: HTMLCanvasElement | null,
  events: EventGroup
): void {
  const container = getViewerContainer();
  // The app's own unit tests hand init a stub `document.body`; there is no
  // menu to suppress on something that cannot carry a listener.
  if (typeof container?.addEventListener !== 'function') return;
  events.on(
    container,
    'contextmenu',
    (event: Event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(TEXT_ENTRY)) return;
      if (target === canvas || target.closest(VIEWER_OWNED)) event.preventDefault();
    },
    true
  );
}
