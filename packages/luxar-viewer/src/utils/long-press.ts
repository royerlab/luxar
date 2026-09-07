/**
 * Long-press → "secondary action" for touch-like pointers.
 *
 * Right-click is the sole route to several menus in the viewer (rail popovers,
 * animation settings, layer visibility verbs, the element menu). A finger has
 * no right button, and the platforms disagree on what a long press does: iOS
 * Safari never fires `contextmenu` for it (it raises a callout sheet instead,
 * which `-webkit-touch-callout: none` suppresses), while Android Chrome
 * synthesises a `contextmenu` event. This helper makes the viewer the SINGLE
 * opener on both: it fires its own callback after `durationMs`, swallows the
 * platform `contextmenu` that Android would otherwise also deliver (so one
 * press cannot open two menus), and swallows the `click` the browser emits when
 * the finger finally lifts (so the button's primary action does not run on top
 * of the menu that just opened).
 *
 * Only touch-like pointers arm it (`isTouchLikePointer`): a mouse keeps its
 * right button and never sees a timer, so desktop behaviour is unchanged.
 *
 * @module utils/long-press
 */

import { isTouchLikePointer } from './input-capabilities';

export interface LongPressOptions {
  /** Called once at the hold threshold; return true when it opened an action. */
  onLongPress: (clientX: number, clientY: number, event: PointerEvent) => boolean;
  /** Hold duration in ms (default 500 — the platforms' own long-press timing). */
  durationMs?: number;
  /**
   * Movement (CSS px) that cancels the press — a finger that travelled this
   * far is dragging, not holding (default 12, the touch click slop).
   */
  slopPx?: number;
}

/** Default hold duration, ms. */
export const LONG_PRESS_MS = 500;
/** Default movement tolerance, CSS px. */
export const LONG_PRESS_SLOP_PX = 12;
/** How long after firing the follow-up `click` is swallowed, ms. */
const CLICK_SWALLOW_MS = 600;

/**
 * Arm long-press detection on `el` (delegated: any touch-like `pointerdown`
 * inside it counts). Returns a disposer that removes every listener and any
 * pending timer.
 */
export function attachLongPress(el: HTMLElement, options: LongPressOptions): () => void {
  const duration = options.durationMs ?? LONG_PRESS_MS;
  const slop = options.slopPx ?? LONG_PRESS_SLOP_PX;

  let pressed: { id: number; x: number; y: number; event: PointerEvent } | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let firedAt = -Infinity;
  const activePointers = new Set<number>();

  const clearTimer = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  const cancel = (): void => {
    clearTimer();
    pressed = null;
  };
  const recentlyFired = (): boolean => performance.now() - firedAt < CLICK_SWALLOW_MS;

  const onPointerDown = (e: Event): void => {
    const ev = e as PointerEvent;
    if (!isTouchLikePointer(ev)) return;
    activePointers.add(ev.pointerId);
    // A second finger means a pinch or a two-finger gesture, not a press.
    if (activePointers.size > 1 || pressed !== null) {
      cancel();
      return;
    }
    pressed = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, event: ev };
    timer = setTimeout(() => {
      timer = null;
      const p = pressed;
      pressed = null;
      if (!p) return;
      if (options.onLongPress(p.x, p.y, p.event)) firedAt = performance.now();
    }, duration);
  };

  const onPointerMove = (e: Event): void => {
    const ev = e as PointerEvent;
    if (!pressed || ev.pointerId !== pressed.id) return;
    const dx = ev.clientX - pressed.x;
    const dy = ev.clientY - pressed.y;
    if (dx * dx + dy * dy > slop * slop) cancel();
  };

  const onPointerEnd = (e: Event): void => {
    const ev = e as PointerEvent;
    activePointers.delete(ev.pointerId);
    if (pressed && ev.pointerId === pressed.id) cancel();
  };

  // Capture phase, so the swallow runs before any listener on the target.
  const onContextMenu = (e: Event): void => {
    if (pressed !== null || recentlyFired()) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  };
  const onClick = (e: Event): void => {
    if (recentlyFired()) {
      e.preventDefault();
      e.stopImmediatePropagation();
      firedAt = -Infinity; // one click per press
    }
  };

  el.addEventListener('pointerdown', onPointerDown);
  el.addEventListener('pointermove', onPointerMove);
  el.addEventListener('pointerup', onPointerEnd);
  el.addEventListener('pointercancel', onPointerEnd);
  el.addEventListener('pointerleave', onPointerEnd);
  el.addEventListener('contextmenu', onContextMenu, true);
  el.addEventListener('click', onClick, true);

  return () => {
    cancel();
    activePointers.clear();
    el.removeEventListener('pointerdown', onPointerDown);
    el.removeEventListener('pointermove', onPointerMove);
    el.removeEventListener('pointerup', onPointerEnd);
    el.removeEventListener('pointercancel', onPointerEnd);
    el.removeEventListener('pointerleave', onPointerEnd);
    el.removeEventListener('contextmenu', onContextMenu, true);
    el.removeEventListener('click', onClick, true);
  };
}
