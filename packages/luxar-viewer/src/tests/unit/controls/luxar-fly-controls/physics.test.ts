/**
 * Unit tests for luxar-fly-controls/physics.ts.
 *
 * Targets audit finding G16 (most numerically dense module in the
 * subpackage — boundary cases: delta=0, sub-threshold tick, Q+E
 * simultaneously, NaN-safe behavior), H6/H7 (linearity in movementSpeed,
 * sub-threshold zeroing, equal-duration Q+E returns angular velocity to
 * ~0), and M2 (mutation-suspect: dampingPower, velocityThreshold, sign
 * of _v0.set(0,0,-1), lookState signs, inertial/non-inertial constant).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  integrateTranslation,
  integrateRotation,
  type FlyPhysicsCtx,
} from '../../../../controls/luxar-fly-controls/physics';
import { config } from '../../../../config';

function makeCtx(overrides: Partial<FlyPhysicsCtx> = {}): FlyPhysicsCtx {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  return {
    camera,
    orientation: new THREE.Quaternion(), // identity → forward = -Z
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    moveState: { forward: 0, back: 0, left: 0, right: 0, up: 0, down: 0 },
    lookState: { horizontal: 0, vertical: 0, roll: 0 },
    inertialMode: true,
    damping: 0.999,
    rotationDamping: 0.99,
    movementSpeed: 1,
    rotationSpeed: 1,
    speedBoost: false,
    ...overrides,
  };
}

describe('integrateTranslation — direction signs (M2)', () => {
  // _v0 = (0,0,-1).applyQuaternion(orientation): for identity → forward = -Z.
  // M2 mutation: flipping this sign would survive orchestrator-level tests.

  it('moveState.forward = 1 produces -Z velocity (forward = -Z for identity orient)', () => {
    const ctx = makeCtx({ movementSpeed: 1 });
    ctx.moveState.forward = 1;
    integrateTranslation(ctx, 0.1);
    // After v += forward * speed * delta: v.z = -0.1 (then damped, but sign preserved).
    expect(ctx.velocity.z).toBeLessThan(0);
  });

  it('moveState.right = 1 produces +X velocity', () => {
    const ctx = makeCtx({ movementSpeed: 1 });
    ctx.moveState.right = 1;
    integrateTranslation(ctx, 0.1);
    expect(ctx.velocity.x).toBeGreaterThan(0);
  });

  it('moveState.up = 1 produces +Y velocity (world-up, not local-up)', () => {
    // M2: vertical uses _v2.set(0,1,0) — world Y, not local Y.
    const ctx = makeCtx({ movementSpeed: 1 });
    ctx.moveState.up = 1;
    // Even with rotated orientation, vertical should still produce +Y.
    ctx.orientation.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
    integrateTranslation(ctx, 0.1);
    expect(ctx.velocity.y).toBeGreaterThan(0);
  });
});

describe('integrateTranslation — H6: sub-threshold velocity zeroing', () => {
  it('zeroes velocity when below the configured velocity threshold (boundary)', () => {
    // H6 / M2: a velocity strictly below threshold must be set to zero.
    const t = config.controls.fly.physics.velocityThreshold;
    const ctx = makeCtx();
    ctx.velocity.set(0, 0, t * 0.1); // strictly sub-threshold
    const moved = integrateTranslation(ctx, 0.001);
    expect(ctx.velocity.x).toBe(0);
    expect(ctx.velocity.y).toBe(0);
    expect(ctx.velocity.z).toBe(0);
    expect(moved).toBe(false);
  });

  it('returns true (moved) when velocity is above threshold post-damping', () => {
    const ctx = makeCtx({ movementSpeed: 10, damping: 0.999 });
    ctx.moveState.forward = 1;
    const moved = integrateTranslation(ctx, 0.1);
    expect(moved).toBe(true);
  });
});

describe('integrateTranslation — H6: linearity in movementSpeed', () => {
  it('doubling movementSpeed doubles the resulting velocity (H6)', () => {
    const ctxA = makeCtx({ movementSpeed: 1, damping: 1.0 }); // damping=1 → no decay
    const ctxB = makeCtx({ movementSpeed: 2, damping: 1.0 });
    ctxA.moveState.forward = 1;
    ctxB.moveState.forward = 1;
    integrateTranslation(ctxA, 0.1);
    integrateTranslation(ctxB, 0.1);
    expect(ctxB.velocity.length()).toBeCloseTo(2 * ctxA.velocity.length(), 5);
  });

  it('speedBoost=true multiplies the effective speed by 2', () => {
    // M2: the speedBoost constant 2.0 must hold.
    const ctxBaseline = makeCtx({ movementSpeed: 1, damping: 1.0, speedBoost: false });
    const ctxBoost = makeCtx({ movementSpeed: 1, damping: 1.0, speedBoost: true });
    ctxBaseline.moveState.forward = 1;
    ctxBoost.moveState.forward = 1;
    integrateTranslation(ctxBaseline, 0.1);
    integrateTranslation(ctxBoost, 0.1);
    expect(ctxBoost.velocity.length()).toBeCloseTo(2 * ctxBaseline.velocity.length(), 5);
  });
});

describe('integrateTranslation — delta boundary cases (G16)', () => {
  it('delta = 0 is a no-op (no velocity gain, position unchanged)', () => {
    // P5 boundary: zero-time step must not produce movement.
    const ctx = makeCtx({ movementSpeed: 5 });
    ctx.moveState.forward = 1;
    const initialPos = ctx.camera.position.clone();
    integrateTranslation(ctx, 0);
    expect(ctx.camera.position.equals(initialPos)).toBe(true);
    // Velocity must remain zero (no accumulation since delta=0).
    expect(ctx.velocity.length()).toBe(0);
  });

  it('large delta does not produce NaN/Infinity', () => {
    // P5 boundary: 10s frame drop — sanity (no overflow).
    const ctx = makeCtx({ movementSpeed: 10 });
    ctx.moveState.forward = 1;
    integrateTranslation(ctx, 10);
    expect(Number.isFinite(ctx.camera.position.z)).toBe(true);
    expect(Number.isFinite(ctx.velocity.length())).toBe(true);
  });

  // [controls.md/G][P5] NaN / Infinity boundary cases. Delta typically
  // arrives from THREE.Timer (clamped), but a misbehaving caller could
  // pass NaN or Inf. Pinning the contract makes any future guard /
  // clamp introduction explicit.
  it('delta=NaN propagates NaN to camera position (current unguarded behaviour)', () => {
    // Pins the CURRENT behaviour: integrateTranslation does not guard
    // against NaN deltas. Multiplying any input by NaN propagates NaN.
    // If a future fix adds `if (!Number.isFinite(delta)) return false;`,
    // this test SHOULD fail and be updated to assert (a) position
    // unchanged AND (b) return value is false. The previous assertion
    // `isNaN || isUnchanged` was true-by-construction (NaN always
    // satisfies `isNaN`), so it killed no mutants.
    const ctx = makeCtx({ movementSpeed: 5 });
    ctx.moveState.forward = 1;
    integrateTranslation(ctx, Number.NaN);
    expect(Number.isNaN(ctx.camera.position.z)).toBe(true);
  });

  it('delta=+Infinity propagates non-finite values to camera position (current unguarded behaviour)', () => {
    const ctx = makeCtx({ movementSpeed: 5 });
    ctx.moveState.forward = 1;
    // Pin observable post-state: position becomes ±Inf (non-finite).
    // A future guard would make this finite — that change should
    // surface here, force an intentional contract update.
    expect(() => integrateTranslation(ctx, Number.POSITIVE_INFINITY)).not.toThrow();
    expect(Number.isFinite(ctx.camera.position.z)).toBe(false);
  });
});

describe('integrateTranslation — non-inertial mode uses high damping (M2)', () => {
  it('non-inertial mode hard-codes damping=0.5 (constant, ignores ctx.damping)', () => {
    // M2 mutation suspect: the 0.5 constant in `inertialMode ? damping : 0.5`.
    // Verify: a ctx with damping=0.999 but inertialMode=false should damp
    // hard like 0.5 (much faster decay than inertial path).
    const ctxInertial = makeCtx({ inertialMode: true, damping: 0.999 });
    const ctxNonInertial = makeCtx({ inertialMode: false, damping: 0.999 });
    ctxInertial.velocity.set(10, 0, 0);
    ctxNonInertial.velocity.set(10, 0, 0);
    integrateTranslation(ctxInertial, 0.1);
    integrateTranslation(ctxNonInertial, 0.1);
    // Non-inertial must have decayed substantially more.
    expect(ctxNonInertial.velocity.length()).toBeLessThan(ctxInertial.velocity.length());
  });
});

describe('integrateRotation — H7: equal-duration Q+E returns angular velocity to ~0', () => {
  it('Q-then-E (opposite roll inputs) leaves angular velocity ≈ 0 (inertial)', () => {
    // H7 invariant: equal-duration Q then E returns angular velocity to ~0.
    // First tick Q (roll=-1) accumulates -Z angular vel. Then second tick E
    // (roll=+1) should cancel it (approximately, since damping intervenes).
    const ctx = makeCtx({ inertialMode: true, rotationSpeed: 1, rotationDamping: 1.0 });
    ctx.lookState.roll = -1;
    integrateRotation(ctx, 0.1);
    const afterQ = ctx.angularVelocity.length();
    expect(afterQ).toBeGreaterThan(0);

    ctx.lookState.roll = 1;
    integrateRotation(ctx, 0.1);
    expect(ctx.angularVelocity.length()).toBeLessThan(afterQ * 0.5);
  });
});

describe('integrateRotation — sign conventions (M2)', () => {
  // physics.ts: pitch = -lookState.vertical * rotationSpeed, yaw = -lookState.horizontal.

  it('lookState.horizontal = 1 produces yaw with sign matching the formula (M2)', () => {
    // M2 mutation: flipping the negation would silently swap left/right turn.
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    ctx.lookState.horizontal = 1;
    integrateRotation(ctx, 0.05);
    // angularVelocity.y should be NEGATIVE (yaw axis = (0,1,0).applyQuat = (0,1,0)
    // for identity orientation; yaw = -1 * 1 = -1).
    expect(ctx.angularVelocity.y).toBeLessThan(0);
  });

  it('lookState.vertical = 1 produces pitch with sign matching the formula', () => {
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    ctx.lookState.vertical = 1;
    integrateRotation(ctx, 0.05);
    // angularVelocity.x should be NEGATIVE (pitch axis = (1,0,0); pitch = -1 * 1 = -1).
    expect(ctx.angularVelocity.x).toBeLessThan(0);
  });

  it('lookState.roll = 1 produces roll with sign matching the formula', () => {
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    ctx.lookState.roll = 1;
    integrateRotation(ctx, 0.05);
    // roll axis = (0,0,-1) (camera forward); roll = +1 * 1 = +1.
    // angularVelocity.z should be NEGATIVE (along -Z forward).
    expect(ctx.angularVelocity.z).toBeLessThan(0);
  });
});

describe('integrateRotation — delta boundary (G16)', () => {
  it('delta = 0 produces no orientation change', () => {
    const ctx = makeCtx({ inertialMode: true, rotationSpeed: 1 });
    ctx.lookState.horizontal = 1;
    const initialOrient = ctx.orientation.clone();
    integrateRotation(ctx, 0);
    expect(ctx.orientation.equals(initialOrient)).toBe(true);
  });

  it('sub-threshold angular velocity is zeroed', () => {
    const t = config.controls.fly.physics.angularVelocityThreshold;
    const ctx = makeCtx({ inertialMode: true });
    ctx.angularVelocity.set(t * 0.1, 0, 0);
    integrateRotation(ctx, 0.01);
    expect(ctx.angularVelocity.length()).toBe(0);
  });
});

describe('integrateRotation — Q+E simultaneous (boundary)', () => {
  it('Q+E held together with cancelling lookState.roll=0 produces no roll', () => {
    // P5 boundary: in the orchestrator, pressing Q then E with key release
    // clears lookState.roll. The integrator's contract is: if roll=0, no
    // accumulated roll torque.
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    ctx.lookState.roll = 0;
    integrateRotation(ctx, 0.05);
    expect(ctx.angularVelocity.length()).toBe(0);
  });
});

describe('integrateTranslation — input cancellation (controls.md G9)', () => {
  it('forward=1 AND back=1 simultaneously produces zero net acceleration', () => {
    // controls.md G9: the formula is `(forward - back) * speed`. With both
    // keys held, the net translation impulse is zero. A mutation that
    // replaced `-` with `+` would survive the existing single-key tests.
    const ctx = makeCtx({ movementSpeed: 5, damping: 1.0 });
    ctx.moveState.forward = 1;
    ctx.moveState.back = 1;
    integrateTranslation(ctx, 0.1);
    // velocity must remain at (0, 0, 0) — the acceleration cancelled and
    // damping=1 leaves any pre-existing velocity untouched (there was none).
    expect(ctx.velocity.length()).toBe(0);
  });

  it('left=1 AND right=1 simultaneously produces zero net acceleration', () => {
    // Symmetric to forward/back; same formula `(right - left) * speed`.
    const ctx = makeCtx({ movementSpeed: 5, damping: 1.0 });
    ctx.moveState.left = 1;
    ctx.moveState.right = 1;
    integrateTranslation(ctx, 0.1);
    expect(ctx.velocity.length()).toBe(0);
  });

  it('up=1 AND down=1 simultaneously produces zero net acceleration', () => {
    const ctx = makeCtx({ movementSpeed: 5, damping: 1.0 });
    ctx.moveState.up = 1;
    ctx.moveState.down = 1;
    integrateTranslation(ctx, 0.1);
    expect(ctx.velocity.length()).toBe(0);
  });
});

describe('integrateRotation — three-axis cross coupling (controls.md G10, G11)', () => {
  it('horizontal=1 + vertical=1 + roll=1 simultaneously accumulates on all three axes', () => {
    // controls.md G10: cross-axis coupling for full 3-axis input is not
    // covered by single-axis tests. With identity orientation, the three
    // local axes map to canonical (1,0,0), (0,1,0), (0,0,-1) — so each
    // angular-velocity component is non-zero with a sign matching the
    // signed-input formula (`-vertical * speed * v0` etc.).
    const ctx = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    ctx.lookState.horizontal = 1;
    ctx.lookState.vertical = 1;
    ctx.lookState.roll = 1;
    integrateRotation(ctx, 0.05);

    // Each component of angularVelocity must be non-zero (cross-coupling
    // would not zero any of them with identity orientation).
    expect(Math.abs(ctx.angularVelocity.x)).toBeGreaterThan(0);
    expect(Math.abs(ctx.angularVelocity.y)).toBeGreaterThan(0);
    expect(Math.abs(ctx.angularVelocity.z)).toBeGreaterThan(0);
  });

  it('doubling rotationSpeed doubles the resulting angular velocity magnitude', () => {
    // controls.md G11: integrateRotation linearity in rotationSpeed is the
    // symmetric counterpart to the translation-linearity test at line 92-101.
    // In non-inertial mode (deterministic), angularVelocity is set directly
    // to `-look * rotationSpeed * localAxis` — strictly linear in rotationSpeed.
    const ctxA = makeCtx({ inertialMode: false, rotationSpeed: 1 });
    const ctxB = makeCtx({ inertialMode: false, rotationSpeed: 2 });
    ctxA.lookState.horizontal = 1;
    ctxB.lookState.horizontal = 1;
    integrateRotation(ctxA, 0.05);
    integrateRotation(ctxB, 0.05);

    // Without damping interfering (one-step accumulation), the magnitudes
    // differ by exactly the speed ratio. A small tolerance covers any
    // damping that ran after the assignment.
    const magA = ctxA.angularVelocity.length();
    const magB = ctxB.angularVelocity.length();
    expect(magB / magA).toBeCloseTo(2, 4);
  });
});

describe('integrateTranslation — exact-threshold boundary (controls.md G12)', () => {
  it('velocity EXACTLY at velocityThreshold is NOT zeroed (strict-less-than gate)', () => {
    // controls.md G12: the zeroing test uses `velocity.length() < threshold`.
    // The boundary value itself must pass through unchanged — pinning the
    // strict `<` vs `<=` contract. We set the velocity precisely at the
    // threshold, then run integrate with delta=0 (no acceleration, no
    // damping decay below threshold) and assert the velocity survives.
    const t = config.controls.fly.physics.velocityThreshold;
    const ctx = makeCtx({ damping: 1.0 });
    ctx.velocity.set(t, 0, 0); // length === t exactly
    const moved = integrateTranslation(ctx, 0);
    // At the boundary value, the strict `<` gate does NOT trigger zeroing.
    expect(ctx.velocity.length()).toBeCloseTo(t, 12);
    expect(moved).toBe(true);
  });
});

describe('integrateTranslation — step-order [controls.md G27]', () => {
  // controls.md G27[P8]: orbit's update.test.ts has step-by-step coverage
  // (steps 1–9). The fly equivalent ordering was implicit:
  //    1. velocity.addScaledVector(_v3, delta)   ← accumulate input impulse
  //    2. velocity.multiplyScalar(damping^t)     ← damping
  //    3. camera.position.addScaledVector(velocity, delta) ← integrate to position
  // A mutation that swapped steps 2 and 3 (integrate first, then damp)
  // would mean position uses the UN-DAMPED velocity. The size of the
  // position change discriminates the two orders.
  it('[G27] damping applies BEFORE position integration (not after)', () => {
    // Pre-load velocity such that no input contributes (forward=back=0,
    // left=right=0, up=down=0). Then a single integrate step:
    //   step1 (impulse): velocity unchanged (zero impulse from no keys)
    //   step2 (damp):   velocity *= damping^(delta*dampingPower)
    //   step3 (move):   position += DAMPED velocity * delta
    // If steps 2/3 were swapped, position += UN-DAMPED velocity * delta —
    // a STRICTLY LARGER displacement (since damping ∈ (0,1)).
    const ctx = makeCtx({ damping: 0.9, inertialMode: true });
    ctx.velocity.set(1, 0, 0); // initial velocity, no input keys pressed
    const positionBefore = ctx.camera.position.clone();
    const delta = 0.1;

    integrateTranslation(ctx, delta);

    const displacement = ctx.camera.position.x - positionBefore.x;
    // Expected (correct order): position += (1 * 0.9^(0.1*dampingPower)) * 0.1.
    // Wrong order (swap 2/3): position += 1 * 0.1 = 0.1.
    expect(displacement).toBeLessThan(0.1); // damped order
    expect(displacement).toBeGreaterThan(0); // sanity
  });

  it('[G27] non-inertial mode (effectiveDamping=0.5) damps MORE aggressively than inertial', () => {
    // Pin the non-inertial branch at physics.ts L56:
    //   effectiveDamping = inertialMode ? ctx.damping : 0.5
    // Non-inertial uses 0.5 regardless of ctx.damping, producing a smaller
    // displacement than inertial with high damping.
    const ctxInertial = makeCtx({ damping: 0.99, inertialMode: true });
    const ctxNon = makeCtx({ damping: 0.99, inertialMode: false });
    ctxInertial.velocity.set(1, 0, 0);
    ctxNon.velocity.set(1, 0, 0);

    integrateTranslation(ctxInertial, 0.1);
    integrateTranslation(ctxNon, 0.1);

    const dInertial = ctxInertial.camera.position.x;
    const dNon = ctxNon.camera.position.x;
    expect(dNon).toBeLessThan(dInertial);
  });
});
