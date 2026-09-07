// @vitest-environment jsdom
/**
 * Unit tests for luxar-fly-controls/input/touch.ts — fly mode's finger
 * vocabulary. Parity with the mouse handlers is asserted directly: a
 * one-finger drag must produce the SAME angular impulse as a right-drag of the
 * same delta, and a two-finger midpoint drag the SAME strafe as a left-drag.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
  handleTouchDown,
  handleTouchMove,
  handleTouchUp,
  FLY_TWIST_ROLL_GAIN,
  PINCH_THRUST_GAIN,
  type FlyPinchState,
  type FlyTouchCtx,
} from '../../../../../controls/luxar-fly-controls/input/touch';
import {
  handleMouseDown,
  handleMouseMove,
  type FlyMouseCtx,
} from '../../../../../controls/luxar-fly-controls/input/mouse';
import type { FlyMouseAction } from '../../../../../controls/luxar-fly-controls/input/keyboard';

function makeCtx(overrides: Partial<FlyTouchCtx> = {}): {
  ctx: FlyTouchCtx;
  state: { pinch: FlyPinchState | null };
} {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  const state = { pinch: null as FlyPinchState | null };
  const ctx: FlyTouchCtx = {
    enabled: true,
    inertialMode: true,
    lookSpeed: 0.005,
    movementSpeed: 1,
    camera,
    orientation: new THREE.Quaternion(),
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    pointers: new Map(),
    getPinch: () => state.pinch,
    setPinch: (p) => {
      state.pinch = p;
    },
    dispatch: vi.fn(),
    ...overrides,
  };
  return { ctx, state };
}

function touch(type: string, id: number, x: number, y: number): PointerEvent {
  return new PointerEvent(type, {
    pointerId: id,
    pointerType: 'touch',
    clientX: x,
    clientY: y,
    button: 0,
    bubbles: true,
    cancelable: true,
  });
}

/** Mouse ctx sharing the touch ctx's physics objects, for parity checks. */
function mouseCtxFrom(ctx: FlyTouchCtx): FlyMouseCtx {
  const st = { action: 'none' as FlyMouseAction, x: 0, y: 0 };
  return {
    enabled: true,
    inertialMode: ctx.inertialMode,
    lookSpeed: ctx.lookSpeed,
    movementSpeed: ctx.movementSpeed,
    camera: ctx.camera,
    orientation: ctx.orientation,
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    getActiveMouseAction: () => st.action,
    setActiveMouseAction: (v) => {
      st.action = v;
    },
    getMouseX: () => st.x,
    setMouseX: (v) => {
      st.x = v;
    },
    getMouseY: () => st.y,
    setMouseY: (v) => {
      st.y = v;
    },
    dispatch: vi.fn(),
  };
}

describe('handleTouchDown / handleTouchUp — gesture lifecycle', () => {
  it('the first finger starts the gesture and cancels the compatibility mouse events', () => {
    const { ctx } = makeCtx();
    const down = touch('pointerdown', 1, 100, 100);
    handleTouchDown(ctx, down);
    expect(down.defaultPrevented).toBe(true);
    expect(ctx.dispatch).toHaveBeenCalledWith('start');
    expect(ctx.pointers.size).toBe(1);
    // A second finger does not re-start.
    handleTouchDown(ctx, touch('pointerdown', 2, 300, 100));
    expect(ctx.dispatch).toHaveBeenCalledTimes(1);
  });

  it('the last finger ends the gesture; earlier lifts re-snapshot the pinch', () => {
    const { ctx, state } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 100, 100));
    handleTouchDown(ctx, touch('pointerdown', 2, 300, 100));
    expect(state.pinch?.distance).toBe(200);
    handleTouchUp(ctx, touch('pointerup', 2, 300, 100));
    expect(state.pinch).toBeNull();
    expect(ctx.dispatch).not.toHaveBeenCalledWith('end');
    handleTouchUp(ctx, touch('pointercancel', 1, 100, 100));
    expect(ctx.dispatch).toHaveBeenCalledWith('end');
    expect(ctx.pointers.size).toBe(0);
  });

  it('an unknown pointer release is ignored; disabled controls track nothing', () => {
    const { ctx } = makeCtx({ enabled: false });
    const down = touch('pointerdown', 1, 0, 0);
    handleTouchDown(ctx, down);
    expect(down.defaultPrevented).toBe(false);
    expect(ctx.pointers.size).toBe(0);
    handleTouchUp(ctx, touch('pointerup', 7, 0, 0));
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });
});

describe('one finger — look', () => {
  it('advances a tracked finger while disabled without applying an impulse', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 100, 100));
    vi.mocked(ctx.dispatch).mockClear();

    ctx.enabled = false;
    handleTouchMove(ctx, touch('pointermove', 1, 400, 100));
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();

    ctx.enabled = true;
    handleTouchMove(ctx, touch('pointermove', 1, 410, 100));
    expect(ctx.angularVelocity.y).toBeCloseTo(-10 * ctx.lookSpeed * 2.5);
  });

  it('ignores a move from a pointer that was never tracked', () => {
    const { ctx } = makeCtx();
    handleTouchMove(ctx, touch('pointermove', 7, 100, 100));
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('matches the mouse right-drag angular impulse for the same delta', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 100, 100));
    handleTouchMove(ctx, touch('pointermove', 1, 130, 80));

    const mouse = mouseCtxFrom(ctx);
    handleMouseDown(mouse, new MouseEvent('mousedown', { button: 2, clientX: 100, clientY: 100 }));
    handleMouseMove(mouse, new MouseEvent('mousemove', { clientX: 130, clientY: 80 }));

    expect(ctx.angularVelocity.toArray()).toEqual(mouse.angularVelocity.toArray());
    expect(ctx.angularVelocity.length()).toBeGreaterThan(0);
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('after a 2 → 1 lift the survivor looks from its own last position (no jump)', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 100, 100));
    handleTouchDown(ctx, touch('pointerdown', 2, 300, 100));
    handleTouchMove(ctx, touch('pointermove', 1, 120, 100)); // two-finger frame
    ctx.angularVelocity.set(0, 0, 0);
    ctx.velocity.set(0, 0, 0);
    handleTouchUp(ctx, touch('pointerup', 2, 300, 100));
    // A move that lands exactly where the survivor already is produces no impulse.
    handleTouchMove(ctx, touch('pointermove', 1, 120, 100));
    expect(ctx.angularVelocity.length()).toBe(0);
  });
});

describe('two fingers — strafe, thrust, roll', () => {
  it('advances the pinch snapshot while disabled without applying an impulse', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 300, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 500, 300));
    vi.mocked(ctx.dispatch).mockClear();

    ctx.enabled = false;
    handleTouchMove(ctx, touch('pointermove', 2, 700, 300));
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();

    ctx.enabled = true;
    handleTouchMove(ctx, touch('pointermove', 2, 710, 300));
    expect(ctx.velocity.z).toBeCloseTo(
      -Math.log(410 / 400) * ctx.movementSpeed * PINCH_THRUST_GAIN
    );
  });

  it('a midpoint drag strafes exactly like a mouse left-drag of the same delta', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 100, 100));
    handleTouchDown(ctx, touch('pointerdown', 2, 300, 100));
    // Translate both fingers by (+40, +10): distance and angle unchanged.
    handleTouchMove(ctx, touch('pointermove', 1, 140, 110));
    handleTouchMove(ctx, touch('pointermove', 2, 340, 110));

    const mouse = mouseCtxFrom(ctx);
    handleMouseDown(mouse, new MouseEvent('mousedown', { button: 0, clientX: 0, clientY: 0 }));
    handleMouseMove(mouse, new MouseEvent('mousemove', { clientX: 40, clientY: 10 }));

    expect(ctx.velocity.x).toBeCloseTo(mouse.velocity.x, 10);
    expect(ctx.velocity.y).toBeCloseTo(mouse.velocity.y, 10);
    expect(ctx.velocity.z).toBeCloseTo(0, 10);
    expect(ctx.angularVelocity.length()).toBeCloseTo(0, 10);
  });

  it('pinch-out thrusts forward, pinch-in exactly undoes it (logarithmic)', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 200, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 400, 300));
    handleTouchMove(ctx, touch('pointermove', 1, 100, 300)); // 200 → 300 px
    handleTouchMove(ctx, touch('pointermove', 2, 500, 300)); // 300 → 400 px
    // Identity orientation looks down -Z: forward thrust is negative z.
    expect(ctx.velocity.z).toBeLessThan(0);
    expect(ctx.velocity.z).toBeCloseTo(-Math.log(400 / 200) * PINCH_THRUST_GAIN, 10);
    expect(ctx.velocity.x).toBeCloseTo(0, 10);
    handleTouchMove(ctx, touch('pointermove', 1, 200, 300));
    handleTouchMove(ctx, touch('pointermove', 2, 400, 300)); // back to 200 px
    expect(ctx.velocity.z).toBeCloseTo(0, 10);
  });

  it('non-inertial mode moves the camera directly instead of the velocity', () => {
    const { ctx } = makeCtx({ inertialMode: false });
    handleTouchDown(ctx, touch('pointerdown', 1, 200, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 400, 300));
    handleTouchMove(ctx, touch('pointermove', 1, 100, 300));
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.camera.position.z).toBeLessThan(5);
  });

  it('a twist rolls about the view direction with the documented sign', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 300, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 500, 300));
    // Rotate the pair 90° about its midpoint (400, 300) at constant distance.
    handleTouchMove(ctx, touch('pointermove', 1, 400, 200));
    handleTouchMove(ctx, touch('pointermove', 2, 400, 400));
    // Forward for identity orientation is -Z; the roll impulse lies on it.
    expect(ctx.angularVelocity.x).toBeCloseTo(0, 10);
    expect(ctx.angularVelocity.y).toBeCloseTo(0, 10);
    // The roll impulse is along forward (-Z), so the -1 touch sign produces +Z.
    expect(ctx.angularVelocity.z).toBeCloseTo(FLY_TWIST_ROLL_GAIN * (Math.PI / 2), 6);
    expect(ctx.velocity.length()).toBeCloseTo(0, 6); // no net thrust at constant distance
  });

  it('does not dispatch change when only a non-anchor third finger moves', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 300, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 500, 300));
    handleTouchDown(ctx, touch('pointerdown', 3, 700, 300));
    vi.mocked(ctx.dispatch).mockClear();

    handleTouchMove(ctx, touch('pointermove', 3, 710, 300));

    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('coincident fingers (degenerate pinch) add no thrust, roll or NaN', () => {
    const { ctx } = makeCtx();
    handleTouchDown(ctx, touch('pointerdown', 1, 300, 300));
    handleTouchDown(ctx, touch('pointerdown', 2, 300, 300));
    handleTouchMove(ctx, touch('pointermove', 1, 400, 300));
    expect(Number.isFinite(ctx.velocity.z)).toBe(true);
    expect(ctx.velocity.z).toBe(0);
    expect(ctx.angularVelocity.z).toBe(0);
    // The next frame is measured against a valid separation again.
    handleTouchMove(ctx, touch('pointermove', 1, 500, 300));
    expect(ctx.velocity.z).toBeLessThan(0);
  });
});
