/**
 * Unit tests for luxar-orbit-controls/update.ts (runUpdateStep).
 *
 * Targets audit findings G9 (gates `Math.abs(rollDelta) > 1e-6` and
 * `Math.abs(zoomDelta) > 1e-8` untested) and H5 (pure damping at rest
 * preserves camera state exactly) and M3 (mutation-suspect: 9-step
 * ordering and gate thresholds).
 *
 * Direct ctx-driven tests avoid the orchestrator and exercise each of
 * the ten steps in isolation.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  runUpdateStep,
  type OrbitUpdateCtx,
} from '../../../../controls/luxar-orbit-controls/update';

function makeCtx(overrides: Partial<OrbitUpdateCtx> = {}): {
  ctx: OrbitUpdateCtx;
  state: {
    rollDelta: number;
    zoomDelta: number;
    distance: number;
    dollyPhase: number;
  };
} {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  camera.updateMatrixWorld();
  const state = {
    rollDelta: 0,
    zoomDelta: 0,
    distance: 5,
    dollyPhase: 0,
  };
  const ctx: OrbitUpdateCtx = {
    enableRotate: true,
    enableDamping: false, // no damping by default for deterministic asserts
    dampingFactor: 0.25,
    autoRotate: false,
    autoRotateSpeed: 0,
    autoRotateAxis: 'vertical',
    enableZoom: true,
    autoDolly: false,
    autoDollyAmplitude: 0.15,
    autoDollyPeriod: 10,
    orientation: new THREE.Quaternion(),
    rotationDelta: new THREE.Quaternion(),
    panDelta: new THREE.Vector3(),
    target: new THREE.Vector3(),
    getRollDelta: () => state.rollDelta,
    setRollDelta: (v) => {
      state.rollDelta = v;
    },
    getZoomDelta: () => state.zoomDelta,
    setZoomDelta: (v) => {
      state.zoomDelta = v;
    },
    getDistance: () => state.distance,
    setDistance: (v) => {
      state.distance = v;
    },
    getDollyPhase: () => state.dollyPhase,
    setDollyPhase: (v) => {
      state.dollyPhase = v;
    },
    camera,
    minDistance: 0.01,
    maxDistance: 1000,
    minZoom: 0.001,
    maxZoom: 1000,
    lastPosition: new THREE.Vector3(0, 0, 5),
    lastQuaternion: new THREE.Quaternion(),
    dispatch: vi.fn(),
    ...overrides,
  };
  return { ctx, state };
}

describe('runUpdateStep — H5: at rest preserves state exactly', () => {
  it('returns false when nothing changes (no auto-rotate, no deltas)', () => {
    // H5 invariant: when there is no pending motion, the update step
    // must be a no-op (camera unchanged → returns false).
    const { ctx } = makeCtx();
    // Force initial-state match by running once first.
    runUpdateStep(ctx);
    const moved = runUpdateStep(ctx);
    expect(moved).toBe(false);
    expect(ctx.dispatch).not.toHaveBeenCalled();
  });
});

describe('runUpdateStep — step 1: auto-rotation', () => {
  it('produces the deterministic per-frame swept angle', () => {
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateSpeed: 60, // 60 * (2π/60) * dt = 2π * dt
    });
    const dt = 1 / 60;
    runUpdateStep(ctx, dt);

    // orientation should have rotated by 2π/60 around screen-up.
    const angle = 2 * Math.acos(Math.min(1, Math.abs(ctx.orientation.w)));
    expect(angle).toBeCloseTo((2 * Math.PI) / 60, 4);
  });

  // controls.md O5 / Phase E11: previously a single `enableRotate=false`
  // gate test. The 2x2 grid of {enableRotate, autoRotate} ∈ {false, true}²
  // has 4 combos but only 2 produce rotation (enableRotate=true ∧
  // autoRotate=true). Parametrize so all 4 gate-product cells are
  // exercised — a regression that swapped the conjunction for a disjunction
  // (rotate when EITHER is true) surfaces as the `(false, true)` /
  // `(true, false)` cases failing rather than nothing breaking.
  it.each<{ enableRotate: boolean; autoRotate: boolean; rotates: boolean }>([
    { enableRotate: false, autoRotate: false, rotates: false },
    { enableRotate: false, autoRotate: true, rotates: false }, // gate blocks
    { enableRotate: true, autoRotate: false, rotates: false }, // nothing to rotate
    { enableRotate: true, autoRotate: true, rotates: true }, // only rotating cell
  ])(
    'auto-rotation gate: enableRotate=$enableRotate, autoRotate=$autoRotate → rotates=$rotates',
    ({ enableRotate, autoRotate, rotates }) => {
      const { ctx } = makeCtx({
        autoRotate,
        autoRotateSpeed: 60,
        enableRotate,
      });
      runUpdateStep(ctx, 1 / 60);
      if (rotates) {
        // 2π/60 sweep per frame at autoRotateSpeed=60.
        const angle = 2 * Math.acos(Math.min(1, Math.abs(ctx.orientation.w)));
        expect(angle).toBeCloseTo((2 * Math.PI) / 60, 4);
      } else {
        // Orientation should remain at identity.
        expect(ctx.orientation.w).toBeCloseTo(1, 5);
        expect(ctx.orientation.x).toBeCloseTo(0, 5);
      }
    }
  );
});

describe('runUpdateStep — step 1: auto-rotation axis', () => {
  // The three axes are distinguished by what stays FIXED, which is what makes
  // each one a stable turntable rather than a drift: the rotation axis is
  // re-derived from the live orientation every frame, and each axis is
  // invariant under its own rotation. Asserting the invariant (plus the
  // direction, so a sign flip cannot pass) pins all three behaviors —
  // a swapped axis table shows up as the wrong component moving.
  //
  // Start pose: orientation identity, target origin, distance 5 → camera at
  // (0, 0, 5), looking down -Z, up +Y, right +X.
  // angle = (2π/60) · speed · dt, so speed 60 at dt 0.25 sweeps exactly π/2.
  const QUARTER_TURN_SPEED = 60;

  it('vertical: holds up fixed, sweeps the camera through the y = 0 plane', () => {
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'vertical',
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    runUpdateStep(ctx, 0.25);

    // Screen-up is the axis, so it is untouched.
    expect(ctx.camera.up.x).toBeCloseTo(0, 6);
    expect(ctx.camera.up.y).toBeCloseTo(1, 6);
    // A quarter turn about +Y takes (0, 0, 5) → (5, 0, 0): the camera stays in
    // the horizontal plane and swings toward +X (right-hand rule about up).
    expect(ctx.camera.position.x).toBeCloseTo(5, 4);
    expect(ctx.camera.position.y).toBeCloseTo(0, 6);
    expect(ctx.camera.position.z).toBeCloseTo(0, 4);
  });

  it('horizontal: holds right fixed, tumbles the camera through the x = 0 plane', () => {
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'horizontal',
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    runUpdateStep(ctx, 0.25);

    // A quarter turn about +X takes (0, 0, 5) → (0, -5, 0), and carries up
    // from +Y to +Z: the camera tumbles under the target, x untouched.
    expect(ctx.camera.position.x).toBeCloseTo(0, 6);
    expect(ctx.camera.position.y).toBeCloseTo(-5, 4);
    expect(ctx.camera.position.z).toBeCloseTo(0, 4);
    expect(ctx.camera.up.z).toBeCloseTo(1, 4);
  });

  it('view: holds the camera POSITION fixed and rolls only the up vector', () => {
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'view',
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    const moved = runUpdateStep(ctx, 0.25);

    // The camera offset lies ALONG the view axis, so a roll cannot move it.
    expect(ctx.camera.position.x).toBeCloseTo(0, 6);
    expect(ctx.camera.position.y).toBeCloseTo(0, 6);
    expect(ctx.camera.position.z).toBeCloseTo(5, 6);
    // Up rolls a quarter turn about the view direction (0, 0, -1): +Y → +X,
    // the same sense as a positive Shift+scroll roll delta.
    expect(ctx.camera.up.x).toBeCloseTo(1, 4);
    expect(ctx.camera.up.y).toBeCloseTo(0, 4);
    // And it must still report movement: change detection compares the
    // quaternion as well as the position, which is what keeps
    // render-on-demand alive through a pure roll.
    expect(moved).toBe(true);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('world-y: holds the camera at its own latitude and preserves world y', () => {
    // The classic turntable. Start from a TILTED pose so world-Y differs from
    // screen-up: rotating about a world axis preserves that world coordinate
    // exactly, which is what makes the subject spin about its own axis instead
    // of precessing the way a screen-vertical turntable does.
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'world-y',
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    ctx.orientation.setFromEuler(new THREE.Euler(0.5, 0, 0)); // camera elevated
    runUpdateStep(ctx, 1 / 60);
    const yAfterOne = ctx.camera.position.y;
    const radiusAfterOne = Math.hypot(ctx.camera.position.x, ctx.camera.position.z);

    // Sweep the rest of a full turn: the latitude (world y and the radius in
    // the plane normal to the axis) must not drift at all.
    for (let i = 0; i < 120; i++) runUpdateStep(ctx, 1 / 60);
    expect(ctx.camera.position.y).toBeCloseTo(yAfterOne, 4);
    expect(Math.hypot(ctx.camera.position.x, ctx.camera.position.z)).toBeCloseTo(radiusAfterOne, 4);
    // Still 5 units from the target, and no NaN anywhere.
    expect(ctx.camera.position.length()).toBeCloseTo(5, 4);
  });

  it('world-y at a level camera is indistinguishable from vertical', () => {
    // Continuity: switching frames must not jump the view when the camera is
    // level, because there the two axes ARE the same axis.
    const runOne = (axis: 'vertical' | 'world-y') => {
      const { ctx } = makeCtx({ autoRotate: true, autoRotateAxis: axis, autoRotateSpeed: 60 });
      runUpdateStep(ctx, 0.25);
      return ctx.camera.position.clone();
    };
    expect(runOne('world-y').distanceTo(runOne('vertical'))).toBeLessThan(1e-5);
  });

  it('a world axis parallel to the view direction rolls instead of degenerating', () => {
    // The one degenerate world case: looking straight down world Y and
    // turntabling about world Y. The camera offset lies along the axis, so it
    // cannot move — this must be a benign roll, not a NaN or a pole flip.
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'world-y',
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    // Look down from +Y: view direction -Y, so the camera sits on the axis.
    ctx.orientation.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
    runUpdateStep(ctx, 1 / 60);
    const posBefore = ctx.camera.position.clone();
    const upBefore = ctx.camera.up.clone();

    runUpdateStep(ctx, 0.25);

    expect(ctx.camera.position.distanceTo(posBefore)).toBeLessThan(1e-4);
    expect(ctx.camera.up.angleTo(upBefore)).toBeCloseTo(Math.PI / 2, 3);
    for (const c of [...ctx.camera.position.toArray(), ...ctx.camera.up.toArray()]) {
      expect(Number.isFinite(c)).toBe(true);
    }
  });

  it('an unrecognized axis token degrades to vertical instead of throwing', () => {
    // A hand-edited scene attr or a newer file must not kill the render loop
    // mid-frame; validation lives at the settings boundary, this is the floor.
    const { ctx } = makeCtx({
      autoRotate: true,
      autoRotateAxis: 'sideways' as never,
      autoRotateSpeed: QUARTER_TURN_SPEED,
    });
    expect(() => runUpdateStep(ctx, 0.25)).not.toThrow();
    expect(ctx.camera.position.x).toBeCloseTo(5, 4);
    expect(ctx.camera.position.y).toBeCloseTo(0, 6);
  });
});

describe('runUpdateStep — step 2: rotation damping', () => {
  // [controls.md/C8 / Phase F] Damping decay was previously tested at the
  // orchestrator level by mutating `(controls as any).rotationDelta`
  // directly — a P1 private-state injection the audit flagged. The
  // contract being tested (rotation delta slerps toward identity each
  // frame when damping is enabled; commits in one frame when damping
  // is disabled) lives entirely inside `runUpdateStep`'s step 2, so
  // OrbitUpdateCtx is the correct seam. `rotationDelta` is a documented
  // public field of `OrbitUpdateCtx` — no privacy violation.
  it('decays rotationDelta toward identity over multiple frames when damping enabled', () => {
    const { ctx } = makeCtx({ enableDamping: true, dampingFactor: 0.1 });
    ctx.rotationDelta.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);

    for (let i = 0; i < 50; i++) runUpdateStep(ctx);

    expect(ctx.rotationDelta.x).toBeCloseTo(0, 2);
    expect(ctx.rotationDelta.y).toBeCloseTo(0, 2);
    expect(ctx.rotationDelta.z).toBeCloseTo(0, 2);
    expect(ctx.rotationDelta.w).toBeCloseTo(1, 2);
  });

  it('clears rotationDelta to identity in one frame when damping disabled', () => {
    const { ctx } = makeCtx({ enableDamping: false });
    ctx.rotationDelta.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);

    runUpdateStep(ctx);

    expect(ctx.rotationDelta.x).toBeCloseTo(0, 5);
    expect(ctx.rotationDelta.y).toBeCloseTo(0, 5);
    expect(ctx.rotationDelta.z).toBeCloseTo(0, 5);
    expect(ctx.rotationDelta.w).toBeCloseTo(1, 5);
  });
});

describe('runUpdateStep — step 3: view-axis roll gate (G9, M3)', () => {
  it('rollDelta = 1e-7 is BELOW the 1e-6 gate (no roll applied)', () => {
    // M3 mutation suspect: gate threshold 1e-6. We assert that values
    // strictly below the gate are silently dropped.
    const { ctx, state } = makeCtx();
    state.rollDelta = 1e-7; // strictly below 1e-6
    // Orientation should remain (almost) identity afterward.
    runUpdateStep(ctx);
    expect(ctx.orientation.equals(new THREE.Quaternion())).toBe(true);
    // rollDelta should NOT have been zeroed/modified by the gate-failure.
    expect(state.rollDelta).toBeCloseTo(1e-7, 12);
  });

  it('rollDelta = 1e-5 is ABOVE the gate (roll IS applied)', () => {
    const { ctx, state } = makeCtx();
    state.rollDelta = 1e-5;
    runUpdateStep(ctx);
    // With enableDamping=false, the roll is applied in full and rollDelta is zeroed.
    expect(state.rollDelta).toBe(0);
    // Orientation slightly different from identity.
    expect(ctx.orientation.equals(new THREE.Quaternion())).toBe(false);
  });

  it('rollDelta = 1e-6 EXACTLY is BELOW the strict-greater-than gate', () => {
    // controls.md G3: the gate is `Math.abs(rollDelta) > 1e-6` (strict
    // greater-than). The audit flagged that the boundary value itself —
    // exactly 1e-6 — is the most likely mutation site (`>` → `>=`). Pin
    // the contract: at the boundary value the roll is NOT applied and
    // rollDelta is preserved.
    const { ctx, state } = makeCtx();
    state.rollDelta = 1e-6;
    runUpdateStep(ctx);
    expect(ctx.orientation.equals(new THREE.Quaternion())).toBe(true);
    expect(state.rollDelta).toBe(1e-6);
  });
});

describe('runUpdateStep — step 6: zoom gate (G9, M3)', () => {
  it('zoomDelta = 1e-9 is BELOW the 1e-8 gate (distance unchanged)', () => {
    const { ctx, state } = makeCtx();
    state.zoomDelta = 1e-9;
    const distBefore = state.distance;
    runUpdateStep(ctx);
    expect(state.distance).toBe(distBefore);
    expect(state.zoomDelta).toBeCloseTo(1e-9, 15);
  });

  it('zoomDelta = 1e-7 is ABOVE the gate (distance changes)', () => {
    const { ctx, state } = makeCtx();
    state.zoomDelta = -0.1; // well above 1e-8, easy to verify
    const distBefore = state.distance;
    runUpdateStep(ctx);
    expect(state.distance).toBeCloseTo(distBefore * (1 + -0.1), 5);
    expect(state.zoomDelta).toBe(0);
  });

  it('zoomDelta = 1e-8 EXACTLY is BELOW the strict-greater-than gate', () => {
    // controls.md G4: gate is `Math.abs(zoomDelta) > 1e-8` (strict). The
    // boundary value itself must NOT trigger the zoom branch — pinning
    // the `>` vs `>=` contract.
    const { ctx, state } = makeCtx();
    state.zoomDelta = 1e-8;
    const distBefore = state.distance;
    runUpdateStep(ctx);
    expect(state.distance).toBe(distBefore);
    expect(state.zoomDelta).toBe(1e-8);
  });

  it('zoom step occurs BEFORE distance clamp (clamp catches over-zoom)', () => {
    // controls.md G2: step order must be (5) zoom → (6) clamp. A positive
    // zoomDelta sufficient to push distance past maxDistance must end up
    // clamped at maxDistance — proving the clamp runs AFTER the zoom step,
    // not before. With enableDamping=false and zoomDelta=10, the formula
    // is distance * (1 + 10) = 11 × initial; clamped to maxDistance.
    const { ctx, state } = makeCtx({ maxDistance: 7 });
    state.distance = 5;
    state.zoomDelta = 10; // 5 * 11 = 55 unclamped → clamp to 7
    runUpdateStep(ctx);
    expect(state.distance).toBeCloseTo(7, 10);
  });
});

describe('runUpdateStep — step 6: distance clamping', () => {
  it('clamps distance ≥ minDistance', () => {
    const { ctx, state } = makeCtx({ minDistance: 3 });
    state.distance = 1; // below min
    runUpdateStep(ctx);
    expect(state.distance).toBeCloseTo(3, 5);
  });

  it('clamps distance ≤ maxDistance', () => {
    const { ctx, state } = makeCtx({ maxDistance: 10 });
    state.distance = 20; // above max
    runUpdateStep(ctx);
    expect(state.distance).toBeCloseTo(10, 5);
  });
});

describe('runUpdateStep — step 7: ortho zoom clamping', () => {
  it('clamps ortho camera.zoom to [minZoom, maxZoom]', () => {
    const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    orthoCam.zoom = 100; // above max
    orthoCam.updateMatrixWorld();
    const { ctx } = makeCtx({ camera: orthoCam, maxZoom: 10, minZoom: 0.1 });
    runUpdateStep(ctx);
    expect(orthoCam.zoom).toBeCloseTo(10, 5);
  });
});

describe('runUpdateStep — step 10: change detection + dispatch', () => {
  it('dispatches "change" when camera moves', () => {
    const { ctx, state } = makeCtx();
    state.zoomDelta = -0.5; // forces a distance change
    runUpdateStep(ctx);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('updates lastPosition + lastQuaternion to current camera state when moved', () => {
    const { ctx } = makeCtx();
    ctx.target.set(5, 5, 5);
    runUpdateStep(ctx);
    // lastPosition should equal camera.position post-step.
    expect(ctx.lastPosition.equals(ctx.camera.position)).toBe(true);
    expect(ctx.lastQuaternion.equals(ctx.camera.quaternion)).toBe(true);
  });

  it('dispatches "change" for an ortho zoom-only frame (position unchanged)', () => {
    const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    orthoCam.position.set(0, 0, 5);
    orthoCam.updateMatrixWorld();
    const { ctx, state } = makeCtx({
      camera: orthoCam,
      lastPosition: new THREE.Vector3(0, 0, 5),
    });
    state.zoomDelta = -0.5; // ortho zoom mutates camera.zoom, not position
    const moved = runUpdateStep(ctx);
    // Sanity: distance (and thus position) is untouched by ortho zoom.
    expect(state.distance).toBeCloseTo(5, 5);
    expect(moved).toBe(true);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });

  it('returns false for an ortho camera at rest', () => {
    const orthoCam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    orthoCam.position.set(0, 0, 5);
    orthoCam.updateMatrixWorld();
    const { ctx } = makeCtx({
      camera: orthoCam,
      lastPosition: new THREE.Vector3(0, 0, 5),
    });
    runUpdateStep(ctx); // settle lastPosition/lastQuaternion
    const moved = runUpdateStep(ctx);
    expect(moved).toBe(false);
  });
});

describe('runUpdateStep — step 4: pan damping', () => {
  it('applies panDelta in full when enableDamping=false (and zeros panDelta)', () => {
    const { ctx } = makeCtx({ enableDamping: false });
    ctx.panDelta.set(1, 2, 3);
    runUpdateStep(ctx);
    expect(ctx.target.x).toBeCloseTo(1, 5);
    expect(ctx.target.y).toBeCloseTo(2, 5);
    expect(ctx.target.z).toBeCloseTo(3, 5);
    expect(ctx.panDelta.length()).toBe(0);
  });

  it('applies panDelta * dampingFactor when enableDamping=true (and decays panDelta)', () => {
    const { ctx } = makeCtx({ enableDamping: true, dampingFactor: 0.5 });
    ctx.panDelta.set(2, 0, 0);
    runUpdateStep(ctx);
    // target += panDelta * 0.5 = (1,0,0). panDelta *= (1 - 0.5) = (1,0,0).
    expect(ctx.target.x).toBeCloseTo(1, 5);
    expect(ctx.panDelta.x).toBeCloseTo(1, 5);
  });
});

describe('runUpdateStep — step 2: auto-dolly', () => {
  const PERIOD = 10;
  const STEPS_PER_PERIOD = 400;
  const DT = PERIOD / STEPS_PER_PERIOD;

  /** Advance `frames` frames of the oscillation and return the ctx state. */
  function run(
    overrides: Partial<OrbitUpdateCtx>,
    frames: number
  ): ReturnType<typeof makeCtx>['state'] {
    const { ctx, state } = makeCtx({
      autoDolly: true,
      autoDollyAmplitude: 0.15,
      autoDollyPeriod: PERIOD,
      ...overrides,
    });
    for (let i = 0; i < frames; i++) runUpdateStep(ctx, DT);
    return state;
  }

  it('is gated on autoDolly', () => {
    const state = run({ autoDolly: false }, STEPS_PER_PERIOD / 4);
    expect(state.distance).toBe(5);
  });

  it('is gated on enableZoom, NOT enableRotate', () => {
    // The asymmetry is the point: ortho disables rotation but keeps zoom, and
    // the dolly is alive there.
    const withoutZoom = run({ enableZoom: false }, STEPS_PER_PERIOD / 4);
    expect(withoutZoom.distance).toBe(5);

    const withoutRotate = run({ enableRotate: false }, STEPS_PER_PERIOD / 4);
    expect(withoutRotate.distance).toBeCloseTo(5 / 1.15, 6);
  });

  it('moves CLOSER first, then back out to the far extreme', () => {
    expect(run({}, STEPS_PER_PERIOD / 4).distance).toBeCloseTo(5 / 1.15, 6);
    expect(run({}, (STEPS_PER_PERIOD * 3) / 4).distance).toBeCloseTo(5 * 1.15, 6);
  });

  it('returns to the starting distance after a full period', () => {
    expect(run({}, STEPS_PER_PERIOD).distance).toBeCloseTo(5, 9);
  });

  it('reaches full amplitude even with damping ON (it bypasses the filter)', () => {
    // Routing the dolly through `zoomDelta` would low-pass it: the quarter-turn
    // extreme would fall short of 1/1.15 and lag the requested period. This
    // pins the "apply straight to the distance" decision.
    const damped = run({ enableDamping: true, dampingFactor: 0.25 }, STEPS_PER_PERIOD / 4);
    expect(damped.distance).toBeCloseTo(5 / 1.15, 6);
  });

  it('lets the user keep zooming: a wheel delta shifts the CENTRE, not the swing', () => {
    // The load-bearing interaction claim. Both the wheel and the dolly only
    // ever MULTIPLY the distance, and multiplication commutes — so a zoom
    // mid-oscillation moves the point the camera breathes around and the
    // oscillation carries on around it, rather than being fought or reset.
    const { ctx, state } = makeCtx({
      autoDolly: true,
      autoDollyAmplitude: 0.15,
      autoDollyPeriod: PERIOD,
    });
    for (let i = 0; i < STEPS_PER_PERIOD; i++) {
      // Halfway through, the user scrolls out by a factor of 1.5.
      if (i === STEPS_PER_PERIOD / 2) state.zoomDelta = 0.5;
      runUpdateStep(ctx, DT);
    }
    // One whole period of dolly (net factor 1) times the user's 1.5.
    expect(state.distance).toBeCloseTo(5 * 1.5, 6);
  });

  it('breathes ortho zoom instead of distance, and reports the frame as moved', () => {
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    camera.zoom = 1;
    const { ctx, state } = makeCtx({
      camera,
      autoDolly: true,
      autoDollyAmplitude: 0.15,
      autoDollyPeriod: PERIOD,
    });
    const moved = runUpdateStep(ctx, DT);
    expect(moved).toBe(true);
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
    // Closer in ortho means a LARGER zoom (zoom ~ 1/distance).
    expect(camera.zoom).toBeGreaterThan(1);
    expect(state.distance).toBe(5);
  });

  it('does not hold the camera still when only the dolly is running', () => {
    const { ctx } = makeCtx({
      autoDolly: true,
      autoDollyAmplitude: 0.15,
      autoDollyPeriod: PERIOD,
    });
    runUpdateStep(ctx, DT);
    expect(runUpdateStep(ctx, DT)).toBe(true);
  });

  it('is inert at zero amplitude or a zero period', () => {
    expect(run({ autoDollyAmplitude: 0 }, STEPS_PER_PERIOD / 4).distance).toBe(5);
    expect(run({ autoDollyPeriod: 0 }, STEPS_PER_PERIOD / 4).distance).toBe(5);
  });
});
