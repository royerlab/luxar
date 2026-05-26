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
    mouseButtons: {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    },
    domElement,
    trackballRadius: 1.0,
    rotateSpeed: 1.0,
    zoomSpeed: 1.0,
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

  it('[controls.md/G19] calls preventDefault on wheel (the page must not scroll under the viewer)', () => {
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

  it('[controls.md/G19] does NOT call preventDefault when disabled (the gate fires first)', () => {
    // Symmetric: the early-return gates must short-circuit before
    // preventDefault is called.
    const { ctx } = makeBaseCtx();
    ctx.enabled = false;
    const evt = new WheelEvent('wheel', { deltaY: -100, cancelable: true });
    const spy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(spy).not.toHaveBeenCalled();
  });
});

describe('handlePointerDown — non-touch path', () => {
  it('captures the pointer and adds to ctx.pointers array', () => {
    const { ctx } = makeBaseCtx();
    const captureSpy = vi.fn();
    (
      ctx.domElement as unknown as { setPointerCapture: (id: number) => void }
    ).setPointerCapture = captureSpy;
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
