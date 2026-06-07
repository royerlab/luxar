/**
 * Viewer-container registry — the single DOM element the viewer mounts its
 * overlays, panels, toasts, dialogs, and injected SVG filters into.
 *
 * Defaults to `document.body` (the standalone-app behaviour). An embedder can
 * point it at a host-owned element via `LuxarApp.init({ container })` so that:
 *
 *   1. **DOM ownership** — every viewer node lives inside the embedder's
 *      element, so `app.dispose()` removes the whole subtree and the host's
 *      DOM is never littered with orphaned `document.body` children.
 *   2. **Visual containment** — a non-`body` container is promoted to a
 *      containing block for `position: fixed`/`absolute` descendants
 *      (`contain: layout` + `position: relative` when static), so the
 *      viewer's fixed overlays render inside the container box rather than
 *      against the viewport. The pre-existing inline `position`/`contain`
 *      values are saved and restored on {@link resetViewerContainer}.
 *
 * This is a page-level singleton, matching the viewer's other module
 * singletons (eventBus, ThemeManager, …). It is set at the start of
 * `LuxarApp.init()` and reset by the dispose pipeline, so the supported
 * contract remains **one viewer per page** (see README "Embedding").
 *
 * Importing this module is side-effect-free: no DOM is touched until
 * {@link setViewerContainer} runs.
 */

let viewerContainer: HTMLElement | null = null;

/** Inline styles we mutated on a custom container, to restore on reset. */
interface SavedContainerStyles {
  element: HTMLElement;
  position: string;
  contain: string;
}
let savedStyles: SavedContainerStyles | null = null;

/**
 * The element viewer overlays mount into. Falls back to `document.body`
 * when no container has been set (or after {@link resetViewerContainer}).
 */
export function getViewerContainer(): HTMLElement {
  return viewerContainer ?? document.body;
}

/**
 * Point the viewer container at `element`. When `element` is anything other
 * than `document.body`, promote it to a containing block so fixed/absolute
 * viewer overlays scope to it. Idempotent for repeat calls with the same
 * element.
 */
export function setViewerContainer(element: HTMLElement): void {
  if (viewerContainer === element) return;
  // Restore any element we previously adjusted before adopting a new one.
  restoreSavedStyles();
  viewerContainer = element;

  if (element === document.body || typeof window === 'undefined') return;

  // Promote the host container to a containing block for the viewer's
  // position:fixed / position:absolute overlays. Save the inline values
  // (NOT the computed ones) so resetViewerContainer restores exactly what the
  // embedder authored.
  const computedPosition = window.getComputedStyle(element).position;
  savedStyles = {
    element,
    position: element.style.position,
    contain: element.style.contain,
  };
  if (computedPosition === 'static') {
    element.style.position = 'relative';
  }
  // `contain: layout` makes the element the containing block for fixed
  // descendants without containing size (so it does not affect the host's
  // own layout sizing). See https://developer.mozilla.org/docs/Web/CSS/contain
  element.style.contain = 'layout';
}

/**
 * Reset the viewer container back to `document.body`, restoring any inline
 * styles {@link setViewerContainer} mutated on a custom container.
 */
export function resetViewerContainer(): void {
  restoreSavedStyles();
  viewerContainer = null;
}

function restoreSavedStyles(): void {
  if (!savedStyles) return;
  const { element, position, contain } = savedStyles;
  // Restore exactly: an empty saved value means there was no inline rule, so
  // remove ours rather than leaving an empty declaration behind.
  if (position) element.style.position = position;
  else element.style.removeProperty('position');
  if (contain) element.style.contain = contain;
  else element.style.removeProperty('contain');
  savedStyles = null;
}
