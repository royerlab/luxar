/**
 * Canvas gesture ownership — make the viewer, not the browser, the owner of
 * every touch gesture over the 3D canvas.
 *
 * Without this, a two-finger pinch over the canvas is claimed by the browser as
 * PAGE zoom: the engine fires `pointercancel`, the orbit controls' pinch
 * handler never runs, and the whole UI ends up zoomed with no way back
 * (measured under iPhone emulation: camera distance unchanged,
 * `visualViewport.scale` 1 → 5). A double-tap zooms the page too, and a
 * long-press raises iOS's image callout sheet over the canvas.
 *
 * Two layers, because the canvas has three owners:
 *   - The standalone page styles its own canvas in `index.html` /
 *     `styles/base/layout.css` (`touch-action: none` there is the fast path,
 *     applied before any script runs).
 *   - An npm-library embedder supplies the canvas (`LuxarAppOptions.canvas`),
 *     so this function stamps the same declarations from JS — but only when
 *     the computed `touch-action` is still the default `auto`, so an embedder
 *     who chose e.g. `pan-y` on the canvas itself deliberately keeps their
 *     choice (`touch-action` does not inherit from its wrapper).
 *   - A `LuxarLayer` host owns its renderer and canvas outright and is
 *     documented as responsible for `touch-action` itself.
 *
 * Safari additionally fires proprietary `gesturestart` / `gesturechange` /
 * `gestureend` events for a pinch, which `touch-action` does not silence in
 * every WebKit version; they are cancelled here, registered only on devices
 * that report touch points so a desktop Safari trackpad pinch (which also
 * emits them, and which the FOV handler consumes as a ctrl-wheel) is untouched.
 *
 * All listeners and inline style changes register through the caller's
 * {@link EventGroup}, so a single `dispose()` removes them with everything
 * else the app owns.
 *
 * @module core/app/interaction/canvas-gesture-ownership
 */

import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { getInputProfile } from '../../../utils/input-capabilities';

/** Safari-only pinch events (not in the standard event maps). */
const SAFARI_GESTURE_EVENTS = ['gesturestart', 'gesturechange', 'gestureend'] as const;
const installedCanvases = new WeakSet<HTMLCanvasElement>();

/**
 * True when the element's computed `touch-action` is the browser default, i.e.
 * nobody (page CSS, embedder, host) has expressed a choice yet. jsdom reports
 * `''` for properties it does not model, which counts as "no choice".
 */
function touchActionIsDefault(canvas: HTMLCanvasElement): boolean {
  if (typeof getComputedStyle !== 'function') return canvas.style.touchAction === '';
  const computed = getComputedStyle(canvas).touchAction;
  return computed === '' || computed === 'auto';
}

/**
 * Claim touch gestures over `canvas` for the viewer's controls. Idempotent;
 * safe on any element the app is handed (guards computed-style access).
 *
 * @param canvas - The canvas the controls listen on.
 * @param events - The app's listener group; the Safari gesture listeners are
 *   registered on it so `dispose()` removes them.
 */
export function installCanvasGestureOwnership(canvas: HTMLCanvasElement, events: EventGroup): void {
  if (installedCanvases.has(canvas)) return;
  installedCanvases.add(canvas);

  const previousTouchAction = canvas.style.touchAction;
  const previousTouchCallout = canvas.style.getPropertyValue('-webkit-touch-callout');
  const previousUserSelect = canvas.style.userSelect;
  const previousWebkitUserSelect = canvas.style.getPropertyValue('-webkit-user-select');

  if (touchActionIsDefault(canvas)) {
    canvas.style.touchAction = 'none';
  }
  // iOS long-press image/link callout and text selection are independent of
  // whether the embedder reserved a browser pan axis with touch-action.
  canvas.style.setProperty('-webkit-touch-callout', 'none');
  canvas.style.userSelect = 'none';
  canvas.style.setProperty('-webkit-user-select', 'none');

  events.add(() => {
    canvas.style.touchAction = previousTouchAction;
    canvas.style.setProperty('-webkit-touch-callout', previousTouchCallout);
    canvas.style.userSelect = previousUserSelect;
    canvas.style.setProperty('-webkit-user-select', previousWebkitUserSelect);
    installedCanvases.delete(canvas);
  });

  if (getInputProfile().touchPoints > 0) {
    const cancel = (event: Event): void => event.preventDefault();
    for (const type of SAFARI_GESTURE_EVENTS) {
      events.on(canvas, type, cancel, { passive: false });
    }
  }
}
