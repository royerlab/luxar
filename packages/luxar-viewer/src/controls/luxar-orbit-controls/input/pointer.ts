/**
 * Pointer (mouse + pen) handler bodies for LuxarOrbitControls.
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator stays
 * focused on its sequenced update step and the lifecycle of its bound
 * handlers.
 *
 * Helpers operate on an `OrbitInputCtx` that the orchestrator builds
 * once per call. Mutable object refs (Vector2, Quaternion, arrays) are
 * shared; primitive state (`state`, `zoomDelta`) is read/written via
 * callbacks. Event dispatch goes back through the orchestrator via the
 * `dispatch` callback to preserve the manager as the dispatch site
 * (Non-Goal 2).
 */

import * as THREE from 'three';
import { computeArcballRotation } from '../math/trackball';
import { computeZoomScale } from '../math/zoom';
import { normalizeWheelDelta } from '../../../utils/wheel-delta';

/** The gesture the orbit controls are currently performing. */
export type ControlAction = 'rotate' | 'pan' | 'zoom' | 'none';

/** Event names the orbit controls dispatch through the orchestrator. */
export type ControlEventName = 'change' | 'start' | 'end';

/**
 * State and callbacks the pointer/touch handlers need, built once per call
 * by the `LuxarOrbitControls` orchestrator. Read-only flags and speeds are
 * copied by value; mutable object refs (pointer array/map, the rotate/pan/
 * dolly start vectors, rotationDelta) are shared so mutation flows back to
 * the instance; primitive state (`state`, zoom delta) is read/written via
 * accessors. Touch entry points and `pan`/`dispatch` are cross-cutting
 * callbacks the orchestrator supplies.
 */
export interface OrbitInputCtx {
  // Read-only flags
  enabled: boolean;
  enableRotate: boolean;
  enablePan: boolean;
  enableZoom: boolean;
  /**
   * True when the active camera is orthographic (ortho mode reuses this
   * controls class). Modifier-carrying wheel events are ceded to the
   * window-level FOV handler only for perspective cameras — adjustFOV
   * no-ops on ortho, so zoom keeps ownership there (trackpad pinch must
   * still zoom in ortho mode).
   */
  isOrthographic: boolean;
  mouseButtons: {
    LEFT: THREE.MOUSE | null;
    MIDDLE: THREE.MOUSE | null;
    RIGHT: THREE.MOUSE | null;
  };

  domElement: HTMLElement;
  trackballRadius: number;
  rotateSpeed: number;
  zoomSpeed: number;
  /**
   * Global per-machine multiplier on WHEEL zoom only (Settings > Input >
   * Zoom Sensitivity; `config.controls.wheelZoomSensitivity`). Pointer-drag
   * dolly is untouched: a drag's delta is screen pixels the user controls
   * directly, whereas a wheel notch's delta is whatever the mouse driver
   * decided, which is the thing this knob exists to tame.
   */
  wheelZoomSensitivity: number;

  boundOnPointerMove: (e: PointerEvent) => void;
  boundOnPointerUp: (e: PointerEvent) => void;

  // Mutable object refs (mutation flows back through the same instance)
  pointers: PointerEvent[];
  pointerPositions: Map<number, THREE.Vector2>;
  rotateStart: THREE.Vector2;
  panStart: THREE.Vector2;
  dollyStart: THREE.Vector2;
  rotationDelta: THREE.Quaternion;

  // Primitive-state accessors (orchestrator owns the underlying field)
  getState: () => ControlAction;
  setState: (s: ControlAction) => void;
  addZoomDelta: (delta: number) => void;

  // Cross-cutting callbacks
  pan: (deltaX: number, deltaY: number) => void;
  dispatch: (type: ControlEventName) => void;

  // Touch-handler entry points (orchestrator routes pointerType==='touch'
  // here; touch.ts implements them).
  onTouchStart: () => void;
  onTouchMove: (event: PointerEvent) => void;

  // setPointers preserves the orchestrator's pointer-array reassignment
  // (vs. in-place splice) for bit-for-bit allocation-pattern parity.
  setPointers: (pointers: PointerEvent[]) => void;
}

/** Convert a PointerEvent's clientX/Y into normalized device coordinates. */
export function pointerNDC(event: PointerEvent, domElement: HTMLElement): THREE.Vector2 {
  const rect = domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
}

/**
 * Translate a mouse button + shift state into a `ControlAction`, honoring
 * the `mouseButtons` mapping and the `enableRotate`/`enablePan`/`enableZoom`
 * gates.
 */
export function mouseAction(button: number, shiftKey: boolean, ctx: OrbitInputCtx): ControlAction {
  let mapping: THREE.MOUSE | null = null;
  if (button === 0) mapping = ctx.mouseButtons.LEFT;
  else if (button === 1) mapping = ctx.mouseButtons.MIDDLE;
  else if (button === 2) mapping = ctx.mouseButtons.RIGHT;

  if (mapping === null) return 'none';

  // Shift+left inverts the primary action:
  // If left=pan → Shift+left=rotate; if left=rotate → Shift+left=pan
  if (button === 0 && shiftKey) {
    if (mapping === THREE.MOUSE.PAN) return ctx.enableRotate ? 'rotate' : 'none';
    if (mapping === THREE.MOUSE.ROTATE) return ctx.enablePan ? 'pan' : 'none';
  }

  if (mapping === THREE.MOUSE.ROTATE) return ctx.enableRotate ? 'rotate' : 'none';
  if (mapping === THREE.MOUSE.PAN) return ctx.enablePan ? 'pan' : 'none';
  if (mapping === THREE.MOUSE.DOLLY) return ctx.enableZoom ? 'zoom' : 'none';

  return 'none';
}

/**
 * Begin a pointer interaction. On the first pointer, captures it and attaches
 * the move/up/cancel listeners; tracks the pointer in the array and position
 * map (reusing an existing Vector2 for the same id to avoid allocation).
 * Touch pointers are routed to `onTouchStart`; otherwise the mouse button +
 * shift state selects rotate/pan/zoom and seeds the matching start point.
 * Dispatches `start` when a gesture became active. No-op while disabled.
 */
export function handlePointerDown(ctx: OrbitInputCtx, event: PointerEvent): void {
  if (!ctx.enabled) return;

  if (ctx.pointers.length === 0) {
    ctx.domElement.setPointerCapture(event.pointerId);
    ctx.domElement.addEventListener('pointermove', ctx.boundOnPointerMove);
    ctx.domElement.addEventListener('pointerup', ctx.boundOnPointerUp);
    ctx.domElement.addEventListener('pointercancel', ctx.boundOnPointerUp);
  }

  ctx.pointers.push(event);
  // MED-36: reuse the Vector2 already in pointerPositions for this
  // pointerId when present (mutate via .set()) instead of allocating a
  // fresh one per pointerdown. handlePointerUp clears the entry on
  // release, so this is a no-op in the steady single-pointer case; for
  // re-entrant pointerdowns on the same id (e.g. dragged-into events)
  // it saves an allocation and preserves Vector2 identity across the
  // sequence, matching the file-level "no allocation" contract used in
  // pan.ts / update.ts.
  const existing = ctx.pointerPositions.get(event.pointerId);
  if (existing !== undefined) {
    existing.set(event.clientX, event.clientY);
  } else {
    ctx.pointerPositions.set(event.pointerId, new THREE.Vector2(event.clientX, event.clientY));
  }

  if (event.pointerType === 'touch') {
    ctx.onTouchStart();
  } else {
    const action = mouseAction(event.button, event.shiftKey, ctx);
    ctx.setState(action);

    if (action === 'rotate') {
      ctx.rotateStart.copy(pointerNDC(event, ctx.domElement));
    } else if (action === 'pan') {
      ctx.panStart.set(event.clientX, event.clientY);
    } else if (action === 'zoom') {
      ctx.dollyStart.set(event.clientX, event.clientY);
    }
  }

  if (ctx.getState() !== 'none') {
    ctx.dispatch('start');
  }
}

/**
 * Advance the active gesture from a pointer move. Updates the stored position
 * and the pointer in the array; touch moves are delegated to `onTouchMove`.
 * For a mouse: rotate accumulates an arcball quaternion into `rotationDelta`,
 * pan feeds the client delta to `ctx.pan`, and zoom converts the vertical
 * delta into a zoom-delta accumulation. No-op while disabled.
 */
export function handlePointerMove(ctx: OrbitInputCtx, event: PointerEvent): void {
  if (!ctx.enabled) return;

  // Update pointer position
  const pos = ctx.pointerPositions.get(event.pointerId);
  if (pos) pos.set(event.clientX, event.clientY);

  // Update the pointer in our array
  for (let i = 0; i < ctx.pointers.length; i++) {
    if (ctx.pointers[i].pointerId === event.pointerId) {
      ctx.pointers[i] = event;
      break;
    }
  }

  if (event.pointerType === 'touch') {
    ctx.onTouchMove(event);
    return;
  }

  const state = ctx.getState();
  if (state === 'rotate') {
    const endNDC = pointerNDC(event, ctx.domElement);
    const deltaQuat = computeArcballRotation(
      ctx.rotateStart,
      endNDC,
      ctx.trackballRadius,
      ctx.rotateSpeed
    );
    ctx.rotationDelta.multiply(deltaQuat);
    ctx.rotateStart.copy(endNDC);
  } else if (state === 'pan') {
    const deltaX = event.clientX - ctx.panStart.x;
    const deltaY = event.clientY - ctx.panStart.y;
    ctx.pan(deltaX, deltaY);
    ctx.panStart.set(event.clientX, event.clientY);
  } else if (state === 'zoom') {
    const deltaY = event.clientY - ctx.dollyStart.y;
    if (deltaY > 0) {
      ctx.addZoomDelta(computeZoomScale(deltaY, ctx.zoomSpeed) - 1);
    } else if (deltaY < 0) {
      ctx.addZoomDelta(-(computeZoomScale(-deltaY, ctx.zoomSpeed) - 1));
    }
    ctx.dollyStart.set(event.clientX, event.clientY);
  }
}

/**
 * End a pointer interaction: remove the pointer from the array (via
 * `setPointers`) and the position map. When no pointers remain, release the
 * capture and detach the move/up/cancel listeners. Resets the gesture state
 * to `'none'` and dispatches `end`.
 */
export function handlePointerUp(ctx: OrbitInputCtx, event: PointerEvent): void {
  // Remove this pointer. Use the filtered result directly — ctx.pointers
  // still references the pre-filter array after setPointers, so reading
  // its length here would check the wrong list.
  const remaining = ctx.pointers.filter((p) => p.pointerId !== event.pointerId);
  ctx.setPointers(remaining);
  ctx.pointerPositions.delete(event.pointerId);

  if (remaining.length === 0) {
    try {
      ctx.domElement.releasePointerCapture(event.pointerId);
    } catch {
      /* pointer capture may already be released on cancel */
    }
    ctx.domElement.removeEventListener('pointermove', ctx.boundOnPointerMove);
    ctx.domElement.removeEventListener('pointerup', ctx.boundOnPointerUp);
    ctx.domElement.removeEventListener('pointercancel', ctx.boundOnPointerUp);
  }

  ctx.setState('none');
  ctx.dispatch('end');
}

/**
 * Handle a scroll wheel zoom. Ctrl/Meta+scroll is ceded to the window-level
 * FOV handler for perspective cameras only (ortho has no FOV, so a pinch must
 * still zoom there). Normalizes `deltaY` to pixel-mode equivalent first
 * (`normalizeWheelDelta`, so a Firefox line-mode notch is not ~32x smaller
 * than a Chromium pixel-mode one), converts that into a zoom-scale (at
 * `zoomSpeed × wheelZoomSensitivity`, which therefore multiplies a
 * browser-independent delta) and accumulates a signed zoom delta (scroll up =
 * zoom in), then dispatches `change` so the damped zoom is picked up in the
 * next update. No-op while disabled or zoom is off.
 */
export function handleWheel(ctx: OrbitInputCtx, event: WheelEvent): void {
  if (!ctx.enabled || !ctx.enableZoom) return;

  // Ctrl/Meta+scroll belongs to the window-level FOV handler (which also
  // covers trackpad pinch — browsers synthesize those as ctrlKey wheel
  // events with no keydown). Deciding here, from the event's own live
  // modifier flags, keeps zoom-vs-FOV routing stateless: no keydown/keyup
  // bookkeeping that can stick when a modifier keyup is lost to a focus
  // change. Mirrors luxar-fly-controls/input/wheel.ts. Orthographic
  // cameras have no FOV (adjustFOV no-ops), so zoom keeps modifier
  // wheels in ortho mode — a pinch there must still zoom.
  if ((event.ctrlKey || event.metaKey) && !ctx.isOrthographic) return;

  event.preventDefault();

  const deltaY = normalizeWheelDelta(event, ctx.domElement);
  const scale = computeZoomScale(deltaY, ctx.zoomSpeed * ctx.wheelZoomSensitivity);
  if (deltaY < 0) {
    // Scroll up = zoom in
    ctx.addZoomDelta(scale - 1);
  } else if (deltaY > 0) {
    // Scroll down = zoom out
    ctx.addZoomDelta(-(scale - 1));
  }

  // Wake up animation loop (zoomDelta is applied with damping in update())
  ctx.dispatch('change');
}
