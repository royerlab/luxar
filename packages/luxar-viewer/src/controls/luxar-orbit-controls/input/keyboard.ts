/**
 * Keyboard arrow-key pan attachment for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on its sequenced update step and the lifecycle of its bound
 * handlers.
 *
 * `attachKeyboardPan` registers the listener and returns a disposer the
 * caller can keep and invoke at dispose-time.
 */

export interface OrbitKeyboardCtx {
  enabled: () => boolean;
  enablePan: () => boolean;
  /** Read as a getter so callers see live mutations to the field. */
  keyPanSpeed: () => number;
  pan: (deltaX: number, deltaY: number) => void;
}

/**
 * Attach an arrow-key pan handler to `element`. Returns a disposer
 * that removes the listener.
 */
export function attachKeyboardPan(
  element: HTMLElement | Window,
  ctx: OrbitKeyboardCtx
): () => void {
  const onKeyDown = (event: KeyboardEvent): void => {
    if (!ctx.enabled() || !ctx.enablePan()) return;

    const speed = ctx.keyPanSpeed();
    switch (event.code) {
      case 'ArrowUp':
        ctx.pan(0, speed);
        event.preventDefault();
        break;
      case 'ArrowDown':
        ctx.pan(0, -speed);
        event.preventDefault();
        break;
      case 'ArrowLeft':
        ctx.pan(speed, 0);
        event.preventDefault();
        break;
      case 'ArrowRight':
        ctx.pan(-speed, 0);
        event.preventDefault();
        break;
    }
  };

  element.addEventListener('keydown', onKeyDown as EventListener);

  return () => {
    element.removeEventListener('keydown', onKeyDown as EventListener);
  };
}
