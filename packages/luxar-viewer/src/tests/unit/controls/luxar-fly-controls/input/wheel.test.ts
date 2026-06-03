/**
 * Unit tests for luxar-fly-controls/input/wheel.ts.
 *
 * Targets audit finding G14 (handleWheel 100% untested) and M6
 * (Math.sign(event.deltaY) direction + magic numbers 0.06/0.3/0.2 —
 * zero direct test).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  handleWheel,
  type FlyWheelCtx,
} from '../../../../../controls/luxar-fly-controls/input/wheel';

function makeCtx(overrides: Partial<FlyWheelCtx> = {}): FlyWheelCtx {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  return {
    enabled: true,
    inertialMode: true,
    movementSpeed: 1,
    rotationSpeed: 1,
    camera,
    orientation: new THREE.Quaternion(), // identity → forward = -Z
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    dispatch: vi.fn(),
    ...overrides,
  };
}

function makeWheelEvent(
  deltaY: number,
  modifiers: { shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean } = {}
): WheelEvent {
  return new WheelEvent('wheel', {
    deltaY,
    shiftKey: modifiers.shiftKey ?? false,
    ctrlKey: modifiers.ctrlKey ?? false,
    metaKey: modifiers.metaKey ?? false,
    cancelable: true,
  });
}

describe('handleWheel — gating', () => {
  it('does nothing when disabled', () => {
    const ctx = makeCtx({ enabled: false });
    const evt = makeWheelEvent(-100);
    handleWheel(ctx, evt);
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('passes through to InputHandler when Ctrl is held (no preventDefault, no state change)', () => {
    const ctx = makeCtx();
    const evt = makeWheelEvent(-100, { ctrlKey: true });
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
    expect(pdSpy).not.toHaveBeenCalled();
  });

  it('passes through to InputHandler when Meta is held', () => {
    const ctx = makeCtx();
    const evt = makeWheelEvent(-100, { metaKey: true });
    handleWheel(ctx, evt);
    expect(ctx.velocity.length()).toBe(0);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });
});

describe('handleWheel — plain scroll (forward/backward)', () => {
  // Formula:
  //   delta = -Math.sign(event.deltaY)
  //   impulse = delta * movementSpeed * 0.3
  //   inertialMode: velocity += forward * impulse
  //   non-inertial: position += forward * impulse * 0.2
  // forward = (0,0,-1).applyQuaternion(orientation) → identity → (0,0,-1).

  it('scroll up (negative deltaY) in inertial mode adds forward impulse to velocity (M6)', () => {
    // M6: assert the exact magic-number formula. delta=+1, speed=2, impulse = 0.6.
    // velocity += (0,0,-1)*0.6 → (0,0,-0.6).
    const ctx = makeCtx({ inertialMode: true, movementSpeed: 2 });
    handleWheel(ctx, makeWheelEvent(-100));
    expect(ctx.velocity.x).toBeCloseTo(0, 5);
    expect(ctx.velocity.y).toBeCloseTo(0, 5);
    expect(ctx.velocity.z).toBeCloseTo(-0.6, 5);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('scroll down (positive deltaY) reverses direction', () => {
    // delta = -Math.sign(+100) = -1 → impulse = -0.6 → velocity = (0,0,+0.6).
    const ctx = makeCtx({ inertialMode: true, movementSpeed: 2 });
    handleWheel(ctx, makeWheelEvent(+100));
    expect(ctx.velocity.z).toBeCloseTo(+0.6, 5);
  });

  it('non-inertial mode moves camera.position directly by impulse * 0.2 (M6)', () => {
    // Non-inertial path: position += forward * impulse * 0.2 = -0.6 * 0.2 = -0.12.
    const ctx = makeCtx({ inertialMode: false, movementSpeed: 2 });
    handleWheel(ctx, makeWheelEvent(-100));
    expect(ctx.camera.position.z).toBeCloseTo(5 - 0.12, 5);
    expect(ctx.velocity.length()).toBe(0); // non-inertial doesn't touch velocity
  });

  it('only the SIGN of deltaY matters (impulse magnitude does not scale with |deltaY|)', () => {
    // Math.sign(d) means delta=10 and delta=10000 produce identical impulses.
    const ctxA = makeCtx({ movementSpeed: 1 });
    const ctxB = makeCtx({ movementSpeed: 1 });
    handleWheel(ctxA, makeWheelEvent(-10));
    handleWheel(ctxB, makeWheelEvent(-10000));
    expect(ctxA.velocity.z).toBeCloseTo(ctxB.velocity.z, 10);
  });
});

describe('handleWheel — shift+scroll (roll around viewing axis)', () => {
  // Formula:
  //   rollImpulse = delta * rotationSpeed * 0.06
  //   inertialMode: angularVelocity += forward * rollImpulse
  //   non-inertial: orientation *= setFromAxisAngle(forward, rollImpulse)

  it('shift+scroll in inertial mode adds to angularVelocity (M6)', () => {
    // delta=+1, rotationSpeed=2 → rollImpulse = 0.12. angularVelocity = (0,0,-0.12).
    const ctx = makeCtx({ inertialMode: true, rotationSpeed: 2 });
    handleWheel(ctx, makeWheelEvent(-100, { shiftKey: true }));
    expect(ctx.angularVelocity.x).toBeCloseTo(0, 5);
    expect(ctx.angularVelocity.y).toBeCloseTo(0, 5);
    expect(ctx.angularVelocity.z).toBeCloseTo(-0.12, 5);
    expect(ctx.velocity.length()).toBe(0); // translation untouched
  });

  it('shift+scroll in non-inertial mode pre-multiplies orientation directly', () => {
    // Non-inertial: orientation should change (not identity any more).
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    handleWheel(ctx, makeWheelEvent(-100, { shiftKey: true }));
    expect(ctx.orientation.equals(new THREE.Quaternion())).toBe(false);
    expect(ctx.angularVelocity.length()).toBe(0);
  });

  it('shift+scroll up vs shift+scroll down produces opposite roll signs', () => {
    const ctxUp = makeCtx({ rotationSpeed: 1 });
    const ctxDown = makeCtx({ rotationSpeed: 1 });
    handleWheel(ctxUp, makeWheelEvent(-100, { shiftKey: true }));
    handleWheel(ctxDown, makeWheelEvent(+100, { shiftKey: true }));
    // Z components opposite signs (forward = -Z).
    expect(Math.sign(ctxUp.angularVelocity.z)).not.toBe(Math.sign(ctxDown.angularVelocity.z));
  });
});

describe('handleWheel — preventDefault contract', () => {
  it('calls preventDefault for ordinary scroll (no Ctrl)', () => {
    const ctx = makeCtx();
    const evt = makeWheelEvent(-100);
    const spy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

describe('handleWheel — modifier precedence (controls.md G15)', () => {
  it('[G15] ctrl+shift together: ctrl takes precedence — early return, no state change, no preventDefault', () => {
    // controls.md G15: wheel.ts line 37 checks `event.ctrlKey || event.metaKey`
    // and returns BEFORE the shift branch. So ctrl+shift means ctrl wins —
    // the FOV passthrough behaviour. A mutation that swapped the branch
    // ordering would let shift+ctrl roll the camera instead of passing
    // through to the FOV controller.
    const ctx = makeCtx();
    const evt = makeWheelEvent(-100, { shiftKey: true, ctrlKey: true });
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.velocity.length()).toBe(0);
    expect(pdSpy).not.toHaveBeenCalled();
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });

  it('[G15] meta+shift together: meta takes precedence (macOS parity with ctrl)', () => {
    // Symmetric to ctrl+shift since metaKey is OR-ed with ctrlKey at the
    // gate. macOS users hold Cmd not Ctrl; pin the parity.
    const ctx = makeCtx();
    const evt = makeWheelEvent(-100, { shiftKey: true, metaKey: true });
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(pdSpy).not.toHaveBeenCalled();
  });
});

describe('handleWheel — deltaY = 0 boundary (controls.md G16)', () => {
  it('[G16] deltaY = 0: Math.sign(0) = 0 → delta = -0 → zero impulse, but dispatch still fires', () => {
    // controls.md G16: some trackpads emit zero-delta wheel events. The
    // function computes delta=-Math.sign(0)=-0 → impulse=0 → no velocity
    // gain. preventDefault and dispatch still fire (the gate was passed).
    const ctx = makeCtx({ movementSpeed: 5 });
    const evt = makeWheelEvent(0);
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    handleWheel(ctx, evt);
    expect(ctx.velocity.length()).toBe(0);
    expect(pdSpy).toHaveBeenCalledTimes(1);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('[G16] deltaY = 0 with shift: zero roll impulse, dispatch still fires', () => {
    // Symmetric path through the shift branch.
    const ctx = makeCtx({ rotationSpeed: 5 });
    const evt = makeWheelEvent(0, { shiftKey: true });
    handleWheel(ctx, evt);
    expect(ctx.angularVelocity.length()).toBe(0);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });
});
