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
 *     element carrying a `luxar-` class, is suppressed. Universal ancestors
 *     (the container, body and document element) do not establish ownership:
 *     the container defaults to `document.body`, which an embedder may share
 *     with their own UI (see `utils/viewer-container`).
 *   - **Text entry.** A typing surface keeps its native menu, which is the only
 *     way to paste — while range, checkbox and radio inputs remain viewer
 *     controls whose menu should be suppressed.
 *   - **Selected overlay text.** An interactive overlay with a live selection
 *     keeps the native Copy action. Its pointer events already prevent a
 *     right-drag from reaching the camera controls.
 *
 * Capture phase, so an inner handler that calls `stopPropagation` (the
 * dimension sliders' play-button menu does) cannot leave the native menu
 * behind it.
 *
 * @module core/app/interaction/context-menu-ownership
 */

import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { isTypingInInput } from '../../../utils/dom/focus';
import { getViewerContainer } from '../../../utils/viewer-container';

/**
 * Any element the viewer mounted. Every viewer-authored class is `luxar-`
 * prefixed (`luxar-overlay`, `luxar-control-rail__btn`, `luxar-gui`, …), and
 * an unclassed child — the `<img>` inside an image overlay, a matte `<canvas>`
 * — reaches one through `closest`.
 */
const VIEWER_OWNED = '[class^="luxar-"], [class*=" luxar-"]';

/** Candidate fields classified by the canonical typing-surface predicate. */
const FIELD = 'input, textarea, [contenteditable]';

function isTextEntry(target: Element): boolean {
  const field = target.closest(FIELD);
  return !!field && (isTypingInInput(field) || field.getAttribute('contenteditable') === '');
}

function hasSelectedInteractiveOverlay(target: Element): boolean {
  return (
    target.closest('.luxar-overlay--interactive') !== null &&
    window.getSelection()?.isCollapsed === false
  );
}

function isViewerOwned(
  target: Element,
  canvas: HTMLCanvasElement | null,
  container: Element
): boolean {
  if (target === canvas) return true;
  const owner = target.closest(VIEWER_OWNED);
  return (
    owner !== null &&
    owner !== container &&
    owner !== document.body &&
    owner !== document.documentElement
  );
}

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
      if (
        !isTextEntry(target) &&
        !hasSelectedInteractiveOverlay(target) &&
        isViewerOwned(target, canvas, container)
      ) {
        event.preventDefault();
      }
    },
    true
  );
}
