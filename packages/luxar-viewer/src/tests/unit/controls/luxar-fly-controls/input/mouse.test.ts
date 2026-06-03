/**
 * Unit tests for luxar-fly-controls/input/mouse.ts.
 *
 * Targets audit finding G15 (handleMouseDown/Up/Move pure helpers not
 * unit-tested). Direct ctx-driven tests verify left=strafe / right=rotate
 * routing, the start/end/change dispatch contract, and the
 * inertial-vs-non-inertial branching.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  handleMouseDown,
  handleMouseUp,
  handleMouseMove,
  type FlyMouseCtx,
} from '../../../../../controls/luxar-fly-controls/input/mouse';
import type { FlyMouseAction } from '../../../../../controls/luxar-fly-controls/input/keyboard';

function makeCtx(overrides: Partial<FlyMouseCtx> = {}): {
  ctx: FlyMouseCtx;
  state: {
    activeMouseAction: FlyMouseAction;
    mouseX: number;
    mouseY: number;
  };
} {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  const state = {
    activeMouseAction: 'none' as FlyMouseAction,
    mouseX: 0,
    mouseY: 0,
  };
  const ctx: FlyMouseCtx = {
    enabled: true,
    inertialMode: true,
    lookSpeed: 0.005,
    movementSpeed: 1,
    camera,
    orientation: new THREE.Quaternion(),
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    getActiveMouseAction: () => state.activeMouseAction,
    setActiveMouseAction: (v) => {
      state.activeMouseAction = v;
    },
    getMouseX: () => state.mouseX,
    setMouseX: (v) => {
      state.mouseX = v;
    },
    getMouseY: () => state.mouseY,
    setMouseY: (v) => {
      state.mouseY = v;
    },
    dispatch: vi.fn(),
    ...overrides,
  };
  return { ctx, state };
}

describe('handleMouseDown', () => {
  it('left-click (button=0) sets activeMouseAction = strafe and dispatches start', () => {
    const { ctx, state } = makeCtx();
    handleMouseDown(ctx, new MouseEvent('mousedown', { button: 0, clientX: 50, clientY: 30 }));
    expect(state.activeMouseAction).toBe('strafe');
    expect(state.mouseX).toBe(50);
    expect(state.mouseY).toBe(30);
    expect(ctx.dispatch).toHaveBeenCalledWith('start');
  });

  it('right-click (button=2) sets activeMouseAction = rotate and dispatches start', () => {
    const { ctx, state } = makeCtx();
    handleMouseDown(ctx, new MouseEvent('mousedown', { button: 2, clientX: 100, clientY: 80 }));
    expect(state.activeMouseAction).toBe('rotate');
    expect(state.mouseX).toBe(100);
    expect(state.mouseY).toBe(80);
    expect(ctx.dispatch).toHaveBeenCalledWith('start');
  });

  it('middle-click (button=1) is ignored', () => {
    const { ctx, state } = makeCtx();
    handleMouseDown(ctx, new MouseEvent('mousedown', { button: 1 }));
    expect(state.activeMouseAction).toBe('none');
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('does nothing when disabled (no state change, no dispatch)', () => {
    const { ctx, state } = makeCtx({ enabled: false });
    handleMouseDown(ctx, new MouseEvent('mousedown', { button: 0 }));
    expect(state.activeMouseAction).toBe('none');
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });
});

describe('handleMouseUp', () => {
  it('button=0 releases strafe and dispatches end', () => {
    const { ctx, state } = makeCtx();
    state.activeMouseAction = 'strafe';
    handleMouseUp(ctx, new MouseEvent('mouseup', { button: 0 }));
    expect(state.activeMouseAction).toBe('none');
    expect(ctx.dispatch).toHaveBeenCalledWith('end');
  });

  it('button=2 releases rotate and dispatches end', () => {
    const { ctx, state } = makeCtx();
    state.activeMouseAction = 'rotate';
    handleMouseUp(ctx, new MouseEvent('mouseup', { button: 2 }));
    expect(state.activeMouseAction).toBe('none');
    expect(ctx.dispatch).toHaveBeenCalledWith('end');
  });

  it('mismatched button (button=0 while in rotate) does NOT release', () => {
    // Boundary: the up-button must match the down-button to release.
    const { ctx, state } = makeCtx();
    state.activeMouseAction = 'rotate';
    handleMouseUp(ctx, new MouseEvent('mouseup', { button: 0 }));
    expect(state.activeMouseAction).toBe('rotate');
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('does nothing when disabled', () => {
    const { ctx, state } = makeCtx({ enabled: false });
    state.activeMouseAction = 'strafe';
    handleMouseUp(ctx, new MouseEvent('mouseup', { button: 0 }));
    expect(state.activeMouseAction).toBe('strafe'); // unchanged
  });
});

describe('handleMouseMove — rotate (right-drag)', () => {
  it('inertial mode adds torque to angularVelocity (pitch/yaw, signs P8/P2)', () => {
    // Formula:
    //   torquePitch = -deltaY * lookSpeed * 2.5
    //   torqueYaw   = -deltaX * lookSpeed * 2.5
    //   angularVelocity += pitchAxis * torquePitch + yawAxis * torqueYaw
    // With identity orientation: pitchAxis=(1,0,0), yawAxis=(0,1,0).
    const { ctx, state } = makeCtx({ inertialMode: true, lookSpeed: 0.01 });
    state.activeMouseAction = 'rotate';
    state.mouseX = 100;
    state.mouseY = 100;

    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 200, clientY: 250 }));

    // deltaX = 100, deltaY = 150 → torquePitch = -150*0.01*2.5 = -3.75, torqueYaw = -2.5.
    expect(ctx.angularVelocity.x).toBeCloseTo(-3.75, 5);
    expect(ctx.angularVelocity.y).toBeCloseTo(-2.5, 5);
    expect(ctx.angularVelocity.z).toBeCloseTo(0, 5);
    expect(state.mouseX).toBe(200);
    expect(state.mouseY).toBe(250);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('rotate path does NOT touch velocity (translation untouched)', () => {
    const { ctx, state } = makeCtx();
    state.activeMouseAction = 'rotate';
    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 10, clientY: 10 }));
    expect(ctx.velocity.length()).toBe(0);
  });
});

describe('handleMouseMove — strafe (left-drag)', () => {
  it('inertial mode adds velocity impulse along camera local axes (P8)', () => {
    // Formula:
    //   strafeScale = movementSpeed * 0.005
    //   inertial: velocity += rightAxis * (-deltaX * strafeScale) + upAxis * (deltaY * strafeScale)
    // With identity orientation: rightAxis=(1,0,0), upAxis=(0,1,0).
    const { ctx, state } = makeCtx({ inertialMode: true, movementSpeed: 4 });
    state.activeMouseAction = 'strafe';
    state.mouseX = 100;
    state.mouseY = 100;

    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 150, clientY: 80 }));
    // deltaX=50, deltaY=-20, strafeScale = 4*0.005 = 0.02.
    // velocity.x = -50 * 0.02 = -1.0, velocity.y = -20 * 0.02 = -0.4.
    expect(ctx.velocity.x).toBeCloseTo(-1.0, 5);
    expect(ctx.velocity.y).toBeCloseTo(-0.4, 5);
    expect(ctx.velocity.z).toBeCloseTo(0, 5);
  });

  it('non-inertial mode moves camera.position directly (no velocity)', () => {
    const { ctx, state } = makeCtx({ inertialMode: false, movementSpeed: 4 });
    state.activeMouseAction = 'strafe';
    state.mouseX = 100;
    state.mouseY = 100;
    const initialPos = ctx.camera.position.clone();

    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 150, clientY: 80 }));
    // position.x = 5 + (-50 * 0.02) = -1.0 + initial = same calc.
    expect(ctx.camera.position.x).toBeCloseTo(initialPos.x - 1.0, 5);
    expect(ctx.camera.position.y).toBeCloseTo(initialPos.y - 0.4, 5);
    expect(ctx.velocity.length()).toBe(0);
  });
});

describe('handleMouseMove — gating', () => {
  it('does nothing when activeMouseAction === "none"', () => {
    const { ctx } = makeCtx();
    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 100, clientY: 100 }));
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('does nothing when disabled even if a drag is "active"', () => {
    const { ctx, state } = makeCtx({ enabled: false });
    state.activeMouseAction = 'rotate';
    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 200, clientY: 200 }));
    expect(ctx.angularVelocity.length()).toBe(0);
  });
});

describe('handleMouseMove — rotated orientation [controls.md G17]', () => {
  // [controls.md G17][P5] Prior coverage used only identity quaternions. The
  // local-axis transformation `_v0.set(1,0,0).applyQuaternion(orientation)`
  // was never exercised. A 90° yaw orientation (rotation about Y) sends
  // the local-X axis to world-Z (negative). Strafing left/right should then
  // produce a Z-component, not an X-component.
  it('[G17] 90° yaw orientation: strafe-X maps to world-Z (local-X applyQuaternion exercised)', () => {
    // Rotate orientation by 90° about world Y.
    const orientation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      Math.PI / 2
    );
    const { ctx, state } = makeCtx({ inertialMode: true, movementSpeed: 1, orientation });
    state.activeMouseAction = 'strafe';
    state.mouseX = 100;
    state.mouseY = 100;

    // Drag +200 px in X, 0 in Y.
    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 300, clientY: 100 }));

    // _v0 was (1,0,0); after 90° Y-rotation → (0,0,-1).
    // strafeScale = 1 * 0.005 = 0.005; deltaX = 200; impulse = -200 * 0.005 = -1.0.
    // velocity.addScaledVector((0,0,-1), -1.0) → (0,0,+1.0).
    expect(ctx.velocity.x).toBeCloseTo(0, 4);
    expect(ctx.velocity.y).toBeCloseTo(0, 4);
    expect(ctx.velocity.z).toBeCloseTo(1.0, 4);
  });

  it('[G17] 90° yaw orientation: rotate (right-drag) angular impulse uses rotated local axes', () => {
    // For right-drag rotate: _v0=(1,0,0) and _v1=(0,1,0) are pitch/yaw axes
    // expressed in WORLD frame. After 90° Y-rotation:
    //   _v0 (was local-X) → world-(0,0,-1)
    //   _v1 (was local-Y) → world-(0,1,0)  (Y is the rotation axis, invariant)
    // Right-drag with deltaX=200, deltaY=0, lookSpeed=0.005:
    //   torquePitch = 0
    //   torqueYaw = -200 * 0.005 * 2.5 = -2.5
    //   angularVelocity += (0,0,-1)*0 + (0,1,0)*(-2.5) = (0,-2.5,0)
    const orientation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      Math.PI / 2
    );
    const { ctx, state } = makeCtx({ orientation });
    state.activeMouseAction = 'rotate';
    state.mouseX = 100;
    state.mouseY = 100;

    handleMouseMove(ctx, new MouseEvent('mousemove', { clientX: 300, clientY: 100 }));

    expect(ctx.angularVelocity.x).toBeCloseTo(0, 4);
    expect(ctx.angularVelocity.y).toBeCloseTo(-2.5, 4);
    expect(ctx.angularVelocity.z).toBeCloseTo(0, 4);
  });

  it('[G17] 180° yaw orientation: strafe-X flips sign vs identity (verifies orientation actually rotates the basis)', () => {
    // 180° yaw: local-X → world-(-1,0,0). With same drag, velocity flips sign vs identity.
    const orientationFlip = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      Math.PI
    );
    const a = makeCtx({ inertialMode: true, movementSpeed: 2 });
    const b = makeCtx({ inertialMode: true, movementSpeed: 2, orientation: orientationFlip });
    a.state.activeMouseAction = 'strafe';
    b.state.activeMouseAction = 'strafe';
    a.state.mouseX = 100;
    b.state.mouseX = 100;
    a.state.mouseY = 100;
    b.state.mouseY = 100;

    const evt = new MouseEvent('mousemove', { clientX: 150, clientY: 100 });
    handleMouseMove(a.ctx, evt);
    handleMouseMove(b.ctx, evt);

    // Identity: velocity.x = -(50 * 0.01) = -0.5; b: velocity.x = +0.5.
    expect(a.ctx.velocity.x).toBeCloseTo(-0.5, 4);
    expect(b.ctx.velocity.x).toBeCloseTo(0.5, 4);
  });
});
