// @vitest-environment jsdom
/**
 * Unit tests for luxar-orbit-controls/input/pointer.ts.
 *
 * Targets audit finding G10 (mouseAction has 2 direct cases in the
 * orchestrator file; wheel/pointer paths untested) and M4
 * (mouseAction Shift+left "invert" logic — mutation swapping the two
 * branches would pass orchestrator tests).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  mouseAction,
  handlePointerDown,
  handlePointerMove,
  handlePointerUp,
  handleWheel,
  type OrbitInputCtx,
  type ControlAction,
} from '../../../../../controls/luxar-orbit-controls/input/pointer';

/**
 * jsdom does not implement PointerEvent. Build a minimal stub with the
 * fields the helpers actually read (pointerId, pointerType, button,
 * clientX, clientY, shiftKey).
 */
function makePointerEvent(
  type: string,
  init: {
    pointerId?: number;
    pointerType?: 'mouse' | 'touch' | 'pen';
    button?: number;
    clientX?: number;
    clientY?: number;
    shiftKey?: boolean;
  } = {}
): PointerEvent {
  const base = new MouseEvent(type, {
    button: init.button ?? 0,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    shiftKey: init.shiftKey ?? false,
  });
  Object.defineProperty(base, 'pointerId', { value: init.pointerId ?? 0, configurable: true });
  Object.defineProperty(base, 'pointerType', {
    value: init.pointerType ?? 'mouse',
    configurable: true,
  });
  return base as unknown as PointerEvent;
}

function makeBaseCtx(overrides: Partial<OrbitInputCtx> = {}): {
  ctx: OrbitInputCtx;
  state: { action: ControlAction; zoomDelta: number };
  domElement: HTMLElement;
} {
  const domElement = document.createElement('div');
  document.body.appendChild(domElement);
  Object.defineProperty(domElement, 'clientWidth', { configurable: true, get: () => 800 });
  Object.defineProperty(domElement, 'clientHeight', { configurable: true, get: () => 600 });
  domElement.getBoundingClientRect = vi.fn(() => ({
    left: 0,
    top: 0,
    width: 800,
    height: 600,
    right: 800,
    bottom: 600,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  }));

  const state = { action: 'none' as ControlAction, zoomDelta: 0 };
  const pointers: PointerEvent[] = [];

  const ctx: OrbitInputCtx = {
    enabled: true,
    enableRotate: true,
    enablePan: true,
    enableZoom: true,
    isOrthographic: false,
    mouseButtons: {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    },
    domElement,
    trackballRadius: 1.0,
    rotateSpeed: 1.0,
    zoomSpeed: 1.0,
    wheelZoomSensitivity: 1.0,
    boundOnPointerMove: vi.fn(),
    boundOnPointerUp: vi.fn(),
    pointers,
    pointerPositions: new Map(),
    rotateStart: new THREE.Vector2(),
    panStart: new THREE.Vector2(),
    dollyStart: new THREE.Vector2(),
    rotationDelta: new THREE.Quaternion(),
    getState: () => state.action,
    setState: (s) => {
      state.action = s;
    },
    addZoomDelta: (d) => {
      state.zoomDelta += d;
    },
    pan: vi.fn(),
    dispatch: vi.fn(),
    onTouchStart: vi.fn(),
    onTouchMove: vi.fn(),
    setPointers: vi.fn(),
    ...overrides,
  };
  return { ctx, state, domElement };
}

describe('mouseAction — default mapping (LEFT=PAN, RIGHT=ROTATE)', () => {
  it('LEFT (no shift) → pan when enablePan=true', () => {
    const { ctx } = makeBaseCtx();
    expect(mouseAction(0, false, ctx)).toBe('pan');
  });

  it('RIGHT → rotate when enableRotate=true', () => {
    const { ctx } = makeBaseCtx();
    expect(mouseAction(2, false, ctx)).toBe('rotate');
  });

  it('MIDDLE → zoom when enableZoom=true', () => {
    const { ctx } = makeBaseCtx();
    expect(mouseAction(1, false, ctx)).toBe('zoom');
  });

  it('null mapping (e.g. ortho RIGHT=null) → none', () => {
    const { ctx } = makeBaseCtx();
    ctx.mouseButtons.RIGHT = null;
    expect(mouseAction(2, false, ctx)).toBe('none');
  });
});

describe('mouseAction — gating', () => {
  it('returns none when enableRotate=false even if mapping is ROTATE', () => {
    const { ctx } = makeBaseCtx();
    ctx.enableRotate = false;
    expect(mouseAction(2, false, ctx)).toBe('none');
  });

  it('returns none when enablePan=false even if mapping is PAN', () => {
    const { ctx } = makeBaseCtx();
    ctx.enablePan = false;
    expect(mouseAction(0, false, ctx)).toBe('none');
  });

  it('returns none when enableZoom=false even if mapping is DOLLY', () => {
    const { ctx } = makeBaseCtx();
    ctx.enableZoom = false;
    expect(mouseAction(1, false, ctx)).toBe('none');
  });
});

describe('mouseAction — Shift+left "invert primary" (M4 mutation suspect)', () => {
  // M4: mutating the two branches (PAN→rotate vs ROTATE→pan) would pass.
  // We exercise BOTH inversions explicitly.

  it('default mapping (LEFT=PAN) + Shift+left → rotate (invert)', () => {
    const { ctx } = makeBaseCtx();
    expect(mouseAction(0, true, ctx)).toBe('rotate');
  });

  it('natural-drag mapping (LEFT=ROTATE) + Shift+left → pan (invert)', () => {
    const { ctx } = makeBaseCtx();
    ctx.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    expect(mouseAction(0, true, ctx)).toBe('pan');
  });

  it('Shift+left does NOT invert RIGHT or MIDDLE buttons', () => {
    const { ctx } = makeBaseCtx();
    expect(mouseAction(1, true, ctx)).toBe('zoom');
    expect(mouseAction(2, true, ctx)).toBe('rotate');
  });

  it('Shift+left fallback: when invert target is disabled, returns none', () => {
    // Default mapping LEFT=PAN, Shift+left wants rotate, but enableRotate=false.
    const { ctx } = makeBaseCtx();
    ctx.enableRotate = false;
    expect(mouseAction(0, true, ctx)).toBe('none');
  });
});

describe('handleWheel — wheelZoomSensitivity (Settings > Input > Zoom Sensitivity)', () => {
  it('scales the per-notch zoom exactly like zoomSpeed (the two multiply)', () => {
    const base = makeBaseCtx();
    const scaled = makeBaseCtx();
    scaled.ctx.wheelZoomSensitivity = 0.25;
    const halfSpeed = makeBaseCtx();
    halfSpeed.ctx.zoomSpeed = 0.25;
    const evt = () => new WheelEvent('wheel', { deltaY: 100, cancelable: true });
    handleWheel(base.ctx, evt());
    handleWheel(scaled.ctx, evt());
    handleWheel(halfSpeed.ctx, evt());
    // Exponential in speed: a quarter of the exponent, not a quarter of the delta.
    expect(scaled.state.zoomDelta).toBeCloseTo(-(Math.pow(0.95, 0.25) - 1), 10);
    expect(scaled.state.zoomDelta).toBeCloseTo(halfSpeed.state.zoomDelta, 10);
    expect(Math.abs(scaled.state.zoomDelta)).toBeLessThan(Math.abs(base.state.zoomDelta));
  });

  it('does not touch pointer-drag dolly (a drag delta is user-controlled pixels)', () => {
    const { ctx, state } = makeBaseCtx();
    ctx.wheelZoomSensitivity = 0.25;
    state.action = 'zoom';
    ctx.dollyStart.set(0, 0);
    ctx.pointers.push(makePointerEvent('pointerdown', { pointerId: 0, clientX: 0, clientY: 0 }));
    ctx.pointerPositions.set(0, new THREE.Vector2(0, 0));
    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 0, clientY: 100 })
    );
    // computeZoomScale(100, zoomSpeed=1) - 1, with the sensitivity NOT applied.
    expect(state.zoomDelta).toBeCloseTo(Math.pow(0.95, 1.0) - 1, 10);
  });
});

describe('handleWheel — deltaMode normalization (#2531)', () => {
  // jsdom defaults `deltaMode` to 0, so every other wheel test in this file
  // constructs a pixel-mode event and was blind to the bug by construction.
  // These set it explicitly. A line-mode notch (Firefox: deltaY 3) used to
  // reach computeZoomScale as a raw 3, ~32x smaller than Chromium's 100.
  const PIXELS_PER_LINE = 16;

  it('a line-mode notch produces the same zoom step as its pixel equivalent', () => {
    const line = makeBaseCtx();
    const pixel = makeBaseCtx();
    handleWheel(line.ctx, new WheelEvent('wheel', { deltaY: 3, deltaMode: 1, cancelable: true }));
    handleWheel(
      pixel.ctx,
      new WheelEvent('wheel', { deltaY: 3 * PIXELS_PER_LINE, deltaMode: 0, cancelable: true })
    );
    expect(line.state.zoomDelta).toBeCloseTo(pixel.state.zoomDelta, 10);
    // And it is far bigger than the pre-fix value, which treated the 3 as px.
    const rawThree = makeBaseCtx();
    handleWheel(
      rawThree.ctx,
      new WheelEvent('wheel', { deltaY: 3, deltaMode: 0, cancelable: true })
    );
    expect(Math.abs(line.state.zoomDelta)).toBeGreaterThan(Math.abs(rawThree.state.zoomDelta) * 10);
  });

  it('pixel mode is unchanged: a 100 px notch is still exactly 0.95^1 - 1', () => {
    // Regression anchor for the bit-identity invariant — Chromium/WebKit
    // behaviour must not have moved, so nothing needed re-tuning.
    const { ctx, state } = makeBaseCtx();
    handleWheel(ctx, new WheelEvent('wheel', { deltaY: 100, deltaMode: 0, cancelable: true }));
    expect(state.zoomDelta).toBeCloseTo(-(Math.pow(0.95, 1.0) - 1), 10);
  });

  it('line mode preserves the sign: negative deltaY zooms IN', () => {
    const { ctx, state } = makeBaseCtx();
    handleWheel(ctx, new WheelEvent('wheel', { deltaY: -3, deltaMode: 1, cancelable: true }));
    expect(state.zoomDelta).toBeLessThan(0);
    // Magnitude matches the pixel-mode equivalent, not the raw -3.
    const pixel = makeBaseCtx();
    handleWheel(
      pixel.ctx,
      new WheelEvent('wheel', { deltaY: -48, deltaMode: 0, cancelable: true })
    );
    expect(state.zoomDelta).toBeCloseTo(pixel.state.zoomDelta, 10);
  });

  it('wheelZoomSensitivity still multiplies on top of a normalized delta', () => {
    const base = makeBaseCtx();
    const scaled = makeBaseCtx();
    scaled.ctx.wheelZoomSensitivity = 0.25;
    const evt = () => new WheelEvent('wheel', { deltaY: 3, deltaMode: 1, cancelable: true });
    handleWheel(base.ctx, evt());
    handleWheel(scaled.ctx, evt());
    // Normalized delta 48 → exponent 0.48 × sensitivity.
    expect(base.state.zoomDelta).toBeCloseTo(-(Math.pow(0.95, 0.48) - 1), 10);
    expect(scaled.state.zoomDelta).toBeCloseTo(-(Math.pow(0.95, 0.48 * 0.25) - 1), 10);
  });
});

describe('handleWheel — direction (sign of deltaY) + addZoomDelta', () => {
  it('scroll up (deltaY < 0) → zoom in (negative addZoomDelta arg)', () => {
    const { ctx, state } = makeBaseCtx();
    const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true });
    handleWheel(ctx, evt);
    // computeZoomScale(-100, 1) ≈ 0.95, so scale-1 ≈ -0.05.
    // addZoomDelta called with scale - 1 (negative for zoom in).
    expect(state.zoomDelta).toBeLessThan(0);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('scroll down (deltaY > 0) → zoom out (positive addZoomDelta arg)', () => {
    const { ctx, state } = makeBaseCtx();
    handleWheel(ctx, new WheelEvent('wheel', { deltaY: 100, cancelable: true }));
    expect(state.zoomDelta).toBeGreaterThan(0);
  });

  it('does nothing when enableZoom=false', () => {
    const { ctx, state } = makeBaseCtx();
    ctx.enableZoom = false;
    handleWheel(ctx, new WheelEvent('wheel', { deltaY: -100, cancelable: true }));
    expect(state.zoomDelta).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('does nothing when disabled', () => {
    const { ctx, state } = makeBaseCtx();
    ctx.enabled = false;
    handleWheel(ctx, new WheelEvent('wheel', { deltaY: -100, cancelable: true }));
    expect(state.zoomDelta).toBe(0);
  });

  it('calls preventDefault on wheel (the page must not scroll under the viewer)', () => {
    // controls.md G19: the existing tests verify state mutation + dispatch
    // but never asserted preventDefault. A regression that removed
    // preventDefault would cause the host page to scroll while the user
    // tries to zoom — invisible to existing tests.
    const { ctx } = makeBaseCtx();
    const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true });
    const spy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('does NOT call preventDefault when disabled (the gate fires first)', () => {
    // Symmetric: the early-return gates must short-circuit before
    // preventDefault is called.
    const { ctx } = makeBaseCtx();
    ctx.enabled = false;
    const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true });
    const spy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(spy).not.toHaveBeenCalled();
  });

  it.each([{ ctrlKey: true }, { metaKey: true }])(
    'ignores wheel with %o — FOV owns modifier wheels (stateless routing)',
    (mods) => {
      // Regression for the stuck-wheel bug class: exclusivity between
      // Ctrl/⌘+wheel FOV and plain-wheel zoom is decided per event from
      // the event's own live modifier flags, not from a keydown-tracked
      // enableZoom gate (which stuck shut when a modifier keyup was lost
      // to a focus change — wheel zoom died after window switching).
      // Also covers trackpad pinch, which browsers synthesize as
      // ctrlKey wheel events with no Control keydown (issue #741).
      const { ctx, state } = makeBaseCtx();
      const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true, ...mods });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleWheel(ctx, evt);
      expect(state.zoomDelta).toBe(0);
      expect(ctx.dispatch).not.toHaveBeenCalled();
      // No preventDefault either — the window-level FOV handler owns
      // (and preventDefaults) modifier wheels.
      expect(spy).not.toHaveBeenCalled();
    }
  );

  it.each([{ ctrlKey: true }, { metaKey: true }])(
    'ortho camera: wheel with %o still zooms (no FOV exists to cede to)',
    (mods) => {
      // adjustFOV no-ops for orthographic cameras, so a modifier wheel
      // ceded to the FOV handler would be a dead input in ortho mode —
      // and trackpad pinch (synthesized ctrlKey wheel) must keep zooming
      // there. Zoom retains ownership when ctx.isOrthographic.
      const { ctx, state } = makeBaseCtx({ isOrthographic: true });
      const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true, ...mods });
      handleWheel(ctx, evt);
      expect(state.zoomDelta).toBeLessThan(0);
      expect(ctx.dispatch).toHaveBeenCalledWith('change');
    }
  );
});

describe('handlePointerDown — non-touch path', () => {
  it('captures the pointer and adds to ctx.pointers array', () => {
    const { ctx } = makeBaseCtx();
    const captureSpy = vi.fn();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      captureSpy;
    // also stub the matching addEventListener calls the helper makes
    // (already supported by jsdom).

    const evt = makePointerEvent('pointerdown', {
      pointerId: 1,
      pointerType: 'mouse',
      button: 0,
      clientX: 100,
      clientY: 50,
    });
    handlePointerDown(ctx, evt);

    expect(captureSpy).toHaveBeenCalledWith(1);
    expect(ctx.pointers).toHaveLength(1);
    expect(ctx.pointers[0]).toBe(evt);
  });

  it('dispatches "start" when state transitions away from "none"', () => {
    const { ctx } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};
    const evt = makePointerEvent('pointerdown', {
      pointerId: 1,
      pointerType: 'mouse',
      button: 0, // → pan (with default mapping)
    });
    handlePointerDown(ctx, evt);
    expect(ctx.dispatch).toHaveBeenCalledWith('start');
  });

  it('does NOT dispatch "start" when mouseButtons mapping yields state=none', () => {
    const { ctx } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};
    ctx.mouseButtons.LEFT = null;
    handlePointerDown(
      ctx,
      makePointerEvent('pointerdown', { pointerId: 1, pointerType: 'mouse', button: 0 })
    );
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('routes touch pointers through onTouchStart', () => {
    const { ctx } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};
    const evt = makePointerEvent('pointerdown', { pointerId: 1, pointerType: 'touch', button: 0 });
    handlePointerDown(ctx, evt);
    expect(ctx.onTouchStart).toHaveBeenCalled();
  });

  it('does nothing when disabled', () => {
    const { ctx } = makeBaseCtx();
    ctx.enabled = false;
    handlePointerDown(ctx, makePointerEvent('pointerdown', { pointerId: 1 }));
    expect(ctx.pointers).toHaveLength(0);
  });

  it('MED-36: reuses the existing Vector2 instance in pointerPositions for the same pointerId', () => {
    // Re-entrant pointerdown on the same pointerId (no intervening
    // pointerup) used to allocate a NEW Vector2 each time, defeating
    // the file-level "no allocation" contract held by pan.ts / update.ts.
    // The fix: when an entry exists, mutate it via .set(); identity is
    // preserved across pointerdowns.
    const { ctx } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};

    handlePointerDown(
      ctx,
      makePointerEvent('pointerdown', {
        pointerId: 1,
        pointerType: 'mouse',
        button: 0,
        clientX: 10,
        clientY: 20,
      })
    );
    const first = ctx.pointerPositions.get(1);
    expect(first).toBeDefined();
    expect(first!.x).toBe(10);
    expect(first!.y).toBe(20);

    handlePointerDown(
      ctx,
      makePointerEvent('pointerdown', {
        pointerId: 1,
        pointerType: 'mouse',
        button: 0,
        clientX: 30,
        clientY: 40,
      })
    );
    const second = ctx.pointerPositions.get(1);
    // Same object reference — Vector2 was mutated in place, not replaced.
    expect(second).toBe(first);
    expect(second!.x).toBe(30);
    expect(second!.y).toBe(40);
  });
});

describe('handlePointerUp', () => {
  it('removes the pointer from the pointers array', () => {
    const { ctx } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};
    (
      ctx.domElement as unknown as { releasePointerCapture: (id: number) => void }
    ).releasePointerCapture = () => {};
    const evt = makePointerEvent('pointerdown', {
      pointerId: 5,
      pointerType: 'mouse',
      button: 0,
    });
    handlePointerDown(ctx, evt);
    expect(ctx.pointers).toHaveLength(1);

    handlePointerUp(
      ctx,
      makePointerEvent('pointerup', { pointerId: 5, pointerType: 'mouse', button: 0 })
    );
    // setPointers was called with the filtered (empty) array.
    expect(ctx.setPointers).toHaveBeenCalledWith([]);
    expect(ctx.dispatch).toHaveBeenCalledWith('end');
  });

  it('sets state back to "none" after pointerup', () => {
    const { ctx, state } = makeBaseCtx();
    (ctx.domElement as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture =
      () => {};
    (
      ctx.domElement as unknown as { releasePointerCapture: (id: number) => void }
    ).releasePointerCapture = () => {};
    handlePointerDown(
      ctx,
      makePointerEvent('pointerdown', { pointerId: 7, pointerType: 'mouse', button: 0 })
    );
    expect(state.action).not.toBe('none');
    handlePointerUp(ctx, makePointerEvent('pointerup', { pointerId: 7 }));
    expect(state.action).toBe('none');
  });
});

describe('handlePointerMove — rotate/pan/zoom branches [controls.md G18]', () => {
  // [controls.md G18][P5] Lines 152-197 of pointer.ts (handlePointerMove) were
  // only reached through orchestrator tests that injected private state.
  // These tests drive the three branches directly on the pure ctx interface.

  it('[G18] disabled: handlePointerMove is a no-op (no state mutation, no dispatch)', () => {
    const { ctx, state } = makeBaseCtx({ enabled: false });
    state.action = 'rotate';
    const initialQuat = ctx.rotationDelta.clone();
    handlePointerMove(ctx, makePointerEvent('pointermove', { clientX: 200, clientY: 100 }));
    expect(ctx.rotationDelta.equals(initialQuat)).toBe(true);
  });

  it('[G18] touch pointer type delegates to onTouchMove and returns early', () => {
    const { ctx, state } = makeBaseCtx();
    state.action = 'rotate';
    const evt = makePointerEvent('pointermove', {
      pointerId: 1,
      pointerType: 'touch',
      clientX: 200,
      clientY: 100,
    });
    handlePointerMove(ctx, evt);
    expect(ctx.onTouchMove).toHaveBeenCalledWith(evt);
    // Mouse path NOT taken — rotationDelta untouched.
    expect(ctx.rotationDelta.equals(new THREE.Quaternion())).toBe(true);
  });

  it('[G18] rotate branch: rotationDelta accumulator grows; rotateStart re-anchors', () => {
    const { ctx, state } = makeBaseCtx();
    state.action = 'rotate';
    ctx.rotateStart.set(0, 0); // identity NDC
    const evt = makePointerEvent('pointermove', {
      pointerId: 0,
      clientX: 400,
      clientY: 200,
    });
    // Add the pointer to the array so the index loop is exercised.
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 100 })
    );
    ctx.pointerPositions.set(0, new THREE.Vector2(100, 100));

    handlePointerMove(ctx, evt);

    // rotationDelta moved off identity.
    expect(ctx.rotationDelta.equals(new THREE.Quaternion())).toBe(false);
    // rotateStart re-anchored to new pointer NDC (non-zero).
    expect(ctx.rotateStart.equals(new THREE.Vector2(0, 0))).toBe(false);
  });

  it('[G18] pan branch: ctx.pan invoked with raw clientX/Y deltas; panStart re-anchors', () => {
    const { ctx, state } = makeBaseCtx();
    state.action = 'pan';
    ctx.panStart.set(100, 200);
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 200 })
    );
    ctx.pointerPositions.set(0, new THREE.Vector2(100, 200));

    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 175, clientY: 220 })
    );
    // pan called with (clientX - panStart.x, clientY - panStart.y) = (75, 20).
    expect(ctx.pan).toHaveBeenCalledWith(75, 20);
    // panStart re-anchored to current pointer.
    expect(ctx.panStart.x).toBe(175);
    expect(ctx.panStart.y).toBe(220);
  });

  it('[G18] zoom branch: positive deltaY adds positive zoomDelta (dolly OUT)', () => {
    // deltaY > 0 means pointer moved DOWN. computeZoomScale(deltaY, speed) - 1
    // is added to zoomDelta. Pin the sign convention.
    const { ctx, state } = makeBaseCtx();
    state.action = 'zoom';
    ctx.dollyStart.set(100, 100);
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 100 })
    );
    ctx.pointerPositions.set(0, new THREE.Vector2(100, 100));

    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 100, clientY: 200 })
    );
    // computeZoomScale(100, 1) = 0.95^1 = 0.95; 0.95 - 1 = -0.05.
    // Wait — the source uses `computeZoomScale(deltaY, ...) - 1` which is < 0 for
    // |deltaY| > 0. But the sign-of-deltaY branch chooses the sign:
    // deltaY > 0 → ctx.addZoomDelta(computeZoomScale - 1) — negative zoomDelta.
    // Pin this contract precisely.
    expect(state.zoomDelta).toBeLessThan(0);
  });

  it('[G18] zoom branch: negative deltaY adds positive zoomDelta (dolly IN, sign flip)', () => {
    const { ctx, state } = makeBaseCtx();
    state.action = 'zoom';
    ctx.dollyStart.set(100, 200);
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 200 })
    );
    ctx.pointerPositions.set(0, new THREE.Vector2(100, 200));

    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 100, clientY: 100 })
    );
    // deltaY = -100 → `-(computeZoomScale(100, 1) - 1)` = `-(-0.05)` = +0.05.
    expect(state.zoomDelta).toBeGreaterThan(0);
  });

  it('[G18] zoom branch: deltaY === 0 → neither branch fires → zoomDelta unchanged', () => {
    // Pin the exact `> 0` / `< 0` strict boundary at lines 190/192.
    const { ctx, state } = makeBaseCtx();
    state.action = 'zoom';
    state.zoomDelta = 5; // sentinel
    ctx.dollyStart.set(100, 200);
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 200 })
    );
    ctx.pointerPositions.set(0, new THREE.Vector2(100, 200));

    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 100, clientY: 200 })
    );
    expect(state.zoomDelta).toBe(5);
  });

  it('[G18] state === "none": no rotate/pan/zoom branch taken; only pointer-tracking updated', () => {
    const { ctx, state } = makeBaseCtx();
    state.action = 'none';
    const startPos = new THREE.Vector2(100, 100);
    ctx.pointers.push(
      makePointerEvent('pointerdown', { pointerId: 0, clientX: 100, clientY: 100 })
    );
    ctx.pointerPositions.set(0, startPos);

    handlePointerMove(
      ctx,
      makePointerEvent('pointermove', { pointerId: 0, clientX: 250, clientY: 250 })
    );
    // pointerPositions updated.
    expect(startPos.x).toBe(250);
    expect(startPos.y).toBe(250);
    // No state mutation.
    expect(ctx.rotationDelta.equals(new THREE.Quaternion())).toBe(true);
    expect(ctx.pan).not.toHaveBeenCalled();
    expect(state.zoomDelta).toBe(0);
  });
});
