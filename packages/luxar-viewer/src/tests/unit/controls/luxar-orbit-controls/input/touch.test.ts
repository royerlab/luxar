/**
 * Unit tests for luxar-orbit-controls/input/touch.ts.
 *
 * Targets audit finding G11 (single-finger and two-finger gestures
 * 100% untested in unit suite). The audit marks visual touch behavior
 * as E2E-only, but the gesture-state logic (single-finger rotate vs pan
 * fallback, two-finger pinch direction) is plain logic and worth
 * mutation coverage.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  handleTouchStart,
  handleTouchMove,
} from '../../../../../controls/luxar-orbit-controls/input/touch';
import type {
  OrbitInputCtx,
  ControlAction,
} from '../../../../../controls/luxar-orbit-controls/input/pointer';

function makePointerEvent(
  init: { pointerId: number; clientX: number; clientY: number } = {
    pointerId: 0,
    clientX: 0,
    clientY: 0,
  }
): PointerEvent {
  const base = new MouseEvent('pointerdown', { clientX: init.clientX, clientY: init.clientY });
  Object.defineProperty(base, 'pointerId', { value: init.pointerId, configurable: true });
  Object.defineProperty(base, 'pointerType', { value: 'touch', configurable: true });
  return base as unknown as PointerEvent;
}

function makeCtx(
  pointers: PointerEvent[],
  overrides: Partial<OrbitInputCtx> = {}
): { ctx: OrbitInputCtx; state: { action: ControlAction; zoomDelta: number } } {
  const domElement = document.createElement('div');
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
  const pointerPositions = new Map<number, THREE.Vector2>();
  for (const p of pointers) {
    pointerPositions.set(p.pointerId, new THREE.Vector2(p.clientX, p.clientY));
  }

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
    boundOnPointerMove: vi.fn(),
    boundOnPointerUp: vi.fn(),
    pointers,
    pointerPositions,
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
  return { ctx, state };
}

describe('handleTouchStart — single finger', () => {
  it('sets state = rotate when enableRotate=true', () => {
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx, state } = makeCtx([p]);
    handleTouchStart(ctx);
    expect(state.action).toBe('rotate');
  });

  it('falls back to pan when enableRotate=false (boundary)', () => {
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx, state } = makeCtx([p], { enableRotate: false });
    handleTouchStart(ctx);
    expect(state.action).toBe('pan');
  });

  it('does nothing when both enableRotate and enablePan are false', () => {
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx, state } = makeCtx([p], { enableRotate: false, enablePan: false });
    handleTouchStart(ctx);
    expect(state.action).toBe('none');
  });
});

describe('handleTouchStart — two fingers', () => {
  it('sets state = zoom (combined dolly+pan)', () => {
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    expect(state.action).toBe('zoom');
  });

  it('records initial two-finger distance in dollyStart.y', () => {
    // dx = 200, dy = 0 → distance = 200.
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    expect(ctx.dollyStart.y).toBeCloseTo(200, 5);
  });

  it('records the pan center between the two fingers', () => {
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 50 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 200, clientY: 150 });
    const { ctx } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    expect(ctx.panStart.x).toBeCloseTo(150, 5);
    expect(ctx.panStart.y).toBeCloseTo(100, 5);
  });
});

describe('handleTouchMove — two-finger pinch direction', () => {
  it('pinch-out (fingers spread) produces a NEGATIVE zoomDelta (zoom in)', () => {
    // Initial: distance=200; after move: distance=400 → dollyDelta = 2.
    // addZoomDelta(-(2-1)) = -1 (negative = zoom in, consistent with scroll-up).
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    // Update positions to simulate spread.
    ctx.pointerPositions.get(1)!.set(50, 100);
    ctx.pointerPositions.get(2)!.set(450, 100);
    state.action = 'zoom';

    handleTouchMove(ctx, p2);

    expect(state.zoomDelta).toBeLessThan(0);
  });

  it('pinch-in (fingers together) produces a POSITIVE zoomDelta (zoom out)', () => {
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    // Update positions to simulate pinch-in (closer together).
    ctx.pointerPositions.get(1)!.set(140, 100);
    ctx.pointerPositions.get(2)!.set(260, 100);
    state.action = 'zoom';

    handleTouchMove(ctx, p2);

    expect(state.zoomDelta).toBeGreaterThan(0);
  });
});

describe('handleTouchMove — single-finger pan', () => {
  it('single-finger pan invokes ctx.pan with the delta from panStart', () => {
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx, state } = makeCtx([p], { enableRotate: false });
    handleTouchStart(ctx); // state=pan, panStart=(100,100)
    expect(state.action).toBe('pan');

    // Move pointer to (150, 130).
    ctx.pointers[0] = makePointerEvent({ pointerId: 1, clientX: 150, clientY: 130 });

    handleTouchMove(ctx, ctx.pointers[0]);
    expect(ctx.pan).toHaveBeenCalledWith(50, 30);
  });
});

describe('handleTouchMove — single-finger rotate (controls.md G20)', () => {
  // [controls.md G20][P5] The rotate branch of handleTouchMove (touch.ts L40-49)
  // is structurally untested: previous coverage only asserted `pan` is NOT
  // called. Here we pin the rotation accumulator growth so a mutation that
  // dropped `rotationDelta.multiply(deltaQuat)` (line 48) would be caught.
  it('[G20] single-finger rotate accumulates rotationDelta (multiplies deltaQuat into accumulator)', () => {
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx, state } = makeCtx([p]);
    handleTouchStart(ctx); // state=rotate, rotateStart=NDC(100,100)
    expect(state.action).toBe('rotate');

    // Capture identity-baseline of rotationDelta.
    const identity = new THREE.Quaternion();
    expect(ctx.rotationDelta.equals(identity)).toBe(true);

    // Move finger to a different NDC location (non-trivial drag).
    ctx.pointers[0] = makePointerEvent({ pointerId: 1, clientX: 400, clientY: 200 });
    handleTouchMove(ctx, ctx.pointers[0]);

    // The accumulator must have moved off the identity quaternion.
    expect(ctx.rotationDelta.equals(identity)).toBe(false);
    // And remained a unit quaternion (no NaN/Infinity).
    const len = Math.sqrt(
      ctx.rotationDelta.x ** 2 +
        ctx.rotationDelta.y ** 2 +
        ctx.rotationDelta.z ** 2 +
        ctx.rotationDelta.w ** 2
    );
    expect(len).toBeCloseTo(1.0, 5);
  });

  it('[G20] rotateStart is advanced to the new pointer NDC after each move (pin re-anchor)', () => {
    // touch.ts L49: `ctx.rotateStart.copy(endNDC)` — without this, sequential
    // drags would compound from the original start, not the latest position.
    const p = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const { ctx } = makeCtx([p]);
    handleTouchStart(ctx);
    const startBefore = ctx.rotateStart.clone();

    ctx.pointers[0] = makePointerEvent({ pointerId: 1, clientX: 400, clientY: 200 });
    handleTouchMove(ctx, ctx.pointers[0]);

    // rotateStart must have moved (re-anchored to the new NDC).
    expect(ctx.rotateStart.equals(startBefore)).toBe(false);
  });
});

describe('handleTouchMove — two-finger pan component (controls.md G21)', () => {
  // [controls.md G21][P5] Two-finger handler does BOTH dolly AND pan; previous
  // coverage only asserted the dolly side. Pin the pan call so a mutation
  // that dropped lines 73-77 (the pan component) would be caught.
  it('[G21] two-finger drag (no pinch) translates ctx.pan with centerX/centerY deltas', () => {
    // Initial fingers at (100,100) and (300,100): distance=200, center=(200,100).
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    state.action = 'zoom';

    // Move BOTH fingers by the same offset (pure translation, no pinch):
    // (150,150) and (350,150) → distance still 200, center now (250,150).
    ctx.pointerPositions.get(1)!.set(150, 150);
    ctx.pointerPositions.get(2)!.set(350, 150);

    handleTouchMove(ctx, p2);

    // Pan delta = newCenter - panStart = (250-200, 150-100) = (50, 50).
    expect(ctx.pan).toHaveBeenCalledWith(50, 50);
  });

  it('[G21] panStart is re-anchored to the new center after each move', () => {
    // Pin touch.ts L77 — without this, panStart would stay at the original
    // center forever and subsequent pans would compound from gesture-start.
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    state.action = 'zoom';

    ctx.pointerPositions.get(1)!.set(150, 150);
    ctx.pointerPositions.get(2)!.set(350, 150);
    handleTouchMove(ctx, p2);

    expect(ctx.panStart.x).toBeCloseTo(250, 5);
    expect(ctx.panStart.y).toBeCloseTo(150, 5);
  });

  it('[G21] pointerPositions missing for one finger → early-return, no pan/zoom (defensive)', () => {
    // touch.ts L59: `if (!p0 || !p1) return;` — guards against a stale
    // pointer being removed from the map mid-gesture. Pin that early return.
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 300, clientY: 100 });
    const { ctx, state } = makeCtx([p1, p2]);
    handleTouchStart(ctx);
    state.action = 'zoom';
    state.zoomDelta = 0;
    (ctx.pan as ReturnType<typeof vi.fn>).mockClear();

    // Remove one finger from the position map (stale state).
    ctx.pointerPositions.delete(1);

    handleTouchMove(ctx, p2);

    expect(state.zoomDelta).toBe(0);
    expect(ctx.pan).not.toHaveBeenCalled();
  });
});

describe('handleTouchStart — zero-finger boundary (controls.md G22)', () => {
  // [controls.md G22][P5] handleTouchStart with pointers.length===0 is the
  // empty boundary — neither single-finger nor two-finger branch fires. State
  // must NOT change. A mutation that changed the gate `=== 1` to `>= 0` or
  // `>= 1` would be caught here.
  it('[G22] pointers.length === 0: state stays "none"', () => {
    const { ctx, state } = makeCtx([]);
    handleTouchStart(ctx);
    expect(state.action).toBe('none');
  });

  it('[G22] pointers.length === 0: rotateStart/panStart/dollyStart untouched', () => {
    const { ctx } = makeCtx([]);
    const r0 = ctx.rotateStart.clone();
    const p0 = ctx.panStart.clone();
    const d0 = ctx.dollyStart.clone();
    handleTouchStart(ctx);
    expect(ctx.rotateStart.equals(r0)).toBe(true);
    expect(ctx.panStart.equals(p0)).toBe(true);
    expect(ctx.dollyStart.equals(d0)).toBe(true);
  });

  it('[G22] pointers.length === 3 (three fingers): no branch matches, state stays "none"', () => {
    // Symmetric to G22 — the touch.ts gates only check `=== 1` and `=== 2`,
    // so >2 pointers also fall through. Lock this.
    const p1 = makePointerEvent({ pointerId: 1, clientX: 100, clientY: 100 });
    const p2 = makePointerEvent({ pointerId: 2, clientX: 200, clientY: 200 });
    const p3 = makePointerEvent({ pointerId: 3, clientX: 300, clientY: 300 });
    const { ctx, state } = makeCtx([p1, p2, p3]);
    handleTouchStart(ctx);
    expect(state.action).toBe('none');
  });
});
