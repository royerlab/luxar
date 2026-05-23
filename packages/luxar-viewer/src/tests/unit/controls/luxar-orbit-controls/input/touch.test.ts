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
import { handleTouchStart, handleTouchMove } from '../../../../../controls/luxar-orbit-controls/input/touch';
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
