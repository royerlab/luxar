/**
 * DOM event-listener wiring for LuxarFlyControls.
 * Extracted from `luxar-fly-controls.ts` so the orchestrator stays
 * focused on lifecycle and the per-frame physics loop.
 *
 * `attachListeners` registers the listeners and returns a disposer that
 * removes them. The orchestrator stores the disposer and calls it from
 * `dispose()`.
 *
 * Pointer listeners carry the touch path (fly mode's finger vocabulary,
 * `input/touch.ts`); the mouse path stays on the legacy mouse events so a
 * mouse user's behaviour is byte-for-byte unchanged. Like the mouse move/up
 * pair, pointer move/up/cancel live on `window` so a drag may leave the
 * canvas.
 *
 * Keyboard listeners are skipped when `externalInputManagement` is set:
 * the caller (InputContextManager) routes keys via the orchestrator's
 * public `handleKeyDown` / `handleKeyUp` methods instead.
 */

export interface FlyListenersCtx {
  domElement: HTMLElement;
  externalInputManagement: boolean;

  onKeyDown: (e: KeyboardEvent) => void;
  onKeyUp: (e: KeyboardEvent) => void;
  onMouseDown: (e: MouseEvent) => void;
  onMouseUp: (e: MouseEvent) => void;
  onMouseMove: (e: MouseEvent) => void;
  onWheel: (e: WheelEvent) => void;
  /** Touch-like pointers (the orchestrator filters by pointer type). */
  onPointerDown: (e: PointerEvent) => void;
  onPointerMove: (e: PointerEvent) => void;
  /** Also receives `pointercancel`. */
  onPointerUp: (e: PointerEvent) => void;
}

/**
 * Attach listeners to `ctx.domElement` / `window`. Returns a disposer
 * that detaches the same set.
 */
export function attachListeners(ctx: FlyListenersCtx): () => void {
  const contextmenu = (e: Event): void => e.preventDefault();

  if (!ctx.externalInputManagement) {
    window.addEventListener('keydown', ctx.onKeyDown);
    window.addEventListener('keyup', ctx.onKeyUp);
  }

  // Mouse events are always handled internally
  ctx.domElement.addEventListener('mousedown', ctx.onMouseDown);
  window.addEventListener('mouseup', ctx.onMouseUp);
  window.addEventListener('mousemove', ctx.onMouseMove);
  ctx.domElement.addEventListener('wheel', ctx.onWheel, { passive: false });
  ctx.domElement.addEventListener('contextmenu', contextmenu);
  // Touch (pointer events; the orchestrator ignores mouse-typed ones).
  ctx.domElement.addEventListener('pointerdown', ctx.onPointerDown);
  window.addEventListener('pointermove', ctx.onPointerMove);
  window.addEventListener('pointerup', ctx.onPointerUp);
  window.addEventListener('pointercancel', ctx.onPointerUp);

  return () => {
    if (!ctx.externalInputManagement) {
      window.removeEventListener('keydown', ctx.onKeyDown);
      window.removeEventListener('keyup', ctx.onKeyUp);
    }
    ctx.domElement.removeEventListener('mousedown', ctx.onMouseDown);
    window.removeEventListener('mouseup', ctx.onMouseUp);
    window.removeEventListener('mousemove', ctx.onMouseMove);
    ctx.domElement.removeEventListener('wheel', ctx.onWheel);
    ctx.domElement.removeEventListener('contextmenu', contextmenu);
    ctx.domElement.removeEventListener('pointerdown', ctx.onPointerDown);
    window.removeEventListener('pointermove', ctx.onPointerMove);
    window.removeEventListener('pointerup', ctx.onPointerUp);
    window.removeEventListener('pointercancel', ctx.onPointerUp);
  };
}
