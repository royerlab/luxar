// @vitest-environment jsdom
/**
 * CameraFlight — the interruptible tween behind `LuxarApp.flyTo()`.
 *
 * Drives the per-frame callback by hand (the animation loop does not run
 * under jsdom) with an injected clock, so every assertion is about the pose
 * the flight WROTE, not about timing luck.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  CameraFlight,
  FLIGHT_CALLBACK_ID,
  buildFlightPath,
  easeFlight,
  keepOrientationPose,
  type FlightFrameDriver,
} from '../../../../../core/app/camera/camera-flight';
import type { CameraSnapshot } from '../../../../../core/app/snapshot/viewer-snapshot';
import type { SceneManager } from '../../../../../scene/scene-manager';

interface FakeControls {
  getFocusTarget: () => THREE.Vector3;
  setTarget: (v: THREE.Vector3) => void;
  reinitialize: ReturnType<typeof vi.fn>;
  dispatchEvent: ReturnType<typeof vi.fn>;
}

function makeSceneManager(dynamicClipping = false): {
  sm: SceneManager;
  camera: THREE.PerspectiveCamera;
  controls: FakeControls;
} {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(0, 0, 10);
  camera.up.set(0, 1, 0);
  const target = new THREE.Vector3(0, 0, 0);
  const controls: FakeControls = {
    getFocusTarget: () => target.clone(),
    setTarget: (v) => target.copy(v),
    reinitialize: vi.fn(),
    dispatchEvent: vi.fn(),
  };
  const sm = {
    camera,
    controls,
    // The SceneManager surface the clipping rule reads; planes as the
    // per-frame updater would have left them for the current view.
    getDynamicClippingState: () => ({
      enabled: dynamicClipping,
      near: camera.near,
      far: camera.far,
    }),
  } as unknown as SceneManager;
  return { sm, camera, controls };
}

/**
 * Animation-controller stub that models a STOPPED loop: callbacks only run
 * when the test ticks them, and `startAnimation` is observable. A stub that
 * fired callbacks on registration would hide the registration/start pairing
 * bug this module exists to avoid.
 */
function makeDriver(): FlightFrameDriver & { tick: () => void; has: (id: string) => boolean } {
  const callbacks = new Map<string, () => void>();
  return {
    addPerFrameCallback: vi.fn((id: string, cb: () => void) => {
      callbacks.set(id, cb);
    }),
    removePerFrameCallback: vi.fn((id: string) => callbacks.delete(id)),
    startAnimation: vi.fn(),
    tick: () => {
      for (const cb of [...callbacks.values()]) cb();
    },
    has: (id) => callbacks.has(id),
  };
}

const DEST: CameraSnapshot = {
  position: [20, 0, 0],
  target: [10, 0, 0],
  up: [0, 0, 1],
  isOrtho: false,
  fov: 40,
  near: 0.5,
  far: 500,
};

describe('easeFlight', () => {
  it('clamps to [0, 1] and is monotonic', () => {
    expect(easeFlight(-1, 'ease-in-out')).toBe(0);
    expect(easeFlight(2, 'ease-in-out')).toBe(1);
    expect(easeFlight(0.5, 'ease-in-out')).toBeCloseTo(0.5);
    expect(easeFlight(0.25, 'ease-in-out')).toBeLessThan(0.25);
    expect(easeFlight(0.75, 'ease-in-out')).toBeGreaterThan(0.75);
    expect(easeFlight(0.3, 'linear')).toBe(0.3);
  });
});

describe('buildFlightPath', () => {
  const from: CameraSnapshot = {
    position: [0, 0, 10],
    target: [0, 0, 0],
    up: [0, 1, 0],
    isOrtho: false,
    fov: 60,
    near: 0.1,
    far: 1000,
  };

  it('reproduces the end poses at s=0 and s=1', () => {
    const path = buildFlightPath(from, DEST);
    const a = path.at(0);
    const b = path.at(1);
    expect(a.position.map((v) => +v.toFixed(6))).toEqual([0, 0, 10]);
    expect(a.target).toEqual([0, 0, 0]);
    expect(b.position.map((v) => +v.toFixed(6))).toEqual([20, 0, 0]);
    expect(b.target).toEqual([10, 0, 0]);
    expect(b.fov).toBeCloseTo(40);
    expect(b.near).toBeCloseTo(0.5);
    expect(b.far).toBeCloseTo(500);
  });

  it('keeps the camera at orbit distance mid-flight instead of cutting through the target', () => {
    // Same distance (10) at both ends but a 90° swing in viewing direction:
    // a position lerp would pass within ~7 units of the target; the orbit
    // interpolation stays on the 10-unit arc.
    const path = buildFlightPath(from, DEST);
    const mid = path.at(0.5);
    const pos = new THREE.Vector3(...mid.position);
    const tgt = new THREE.Vector3(...mid.target);
    expect(pos.distanceTo(tgt)).toBeCloseTo(10, 5);
    // And the target itself moved linearly.
    expect(tgt.toArray().map((v) => +v.toFixed(6))).toEqual([5, 0, 0]);
  });

  it('interpolates distance geometrically and up spherically', () => {
    const far: CameraSnapshot = { ...from, position: [0, 0, 1000], up: [0, 0, 1] };
    const mid = buildFlightPath(from, far).at(0.5);
    const pos = new THREE.Vector3(...mid.position);
    expect(pos.length()).toBeCloseTo(100, 3); // sqrt(10 * 1000)
    const up = new THREE.Vector3(...mid.up);
    expect(up.length()).toBeCloseTo(1, 6);
    expect(up.y).toBeCloseTo(Math.SQRT1_2, 5);
    expect(up.z).toBeCloseTo(Math.SQRT1_2, 5);
  });

  it('carries ortho zoom when both ends have one', () => {
    const o0: CameraSnapshot = { ...from, isOrtho: true, fov: undefined, zoom: 1 };
    const o1: CameraSnapshot = { ...DEST, isOrtho: true, fov: undefined, zoom: 4 };
    expect(buildFlightPath(o0, o1).at(0.5).zoom).toBeCloseTo(2);
  });
});

describe('keepOrientationPose', () => {
  it('keeps the live direction and up, takes target, distance and projection from the path', () => {
    const pathPose: CameraSnapshot = { ...DEST, position: [20, 0, 0], target: [10, 0, 0] }; // dist 10, +x
    const reseated = keepOrientationPose(
      pathPose,
      new THREE.Vector3(0, 0, 4), // live camera 4 units along +z from its target
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0)
    );
    expect(reseated.target).toEqual([10, 0, 0]);
    expect(reseated.position.map((v) => +v.toFixed(6))).toEqual([10, 0, 10]); // +z kept, dist 10
    expect(reseated.up).toEqual([0, 1, 0]);
    expect(reseated.fov).toBe(DEST.fov);
    expect(reseated.near).toBe(DEST.near);
  });

  it('falls back to the path direction when the camera sits on its target', () => {
    const pathPose: CameraSnapshot = { ...DEST, position: [20, 0, 0], target: [10, 0, 0] };
    const reseated = keepOrientationPose(
      pathPose,
      new THREE.Vector3(3, 3, 3),
      new THREE.Vector3(3, 3, 3),
      new THREE.Vector3(0, 1, 0)
    );
    expect(reseated.position).toEqual([20, 0, 0]);
  });
});

describe('CameraFlight', () => {
  let now: number;
  let driver: ReturnType<typeof makeDriver>;
  let canvas: HTMLCanvasElement;

  beforeEach(() => {
    now = 1000;
    driver = makeDriver();
    canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
  });

  function makeFlight(inputElement: HTMLElement | null = canvas, dynamicClipping = false) {
    const { sm, camera, controls } = makeSceneManager(dynamicClipping);
    const flight = new CameraFlight({
      sceneManager: sm,
      animationController: driver,
      inputElement,
      now: () => now,
    });
    return { flight, camera, controls };
  }

  it('registers a continuous per-frame callback AND starts the loop', () => {
    const { flight } = makeFlight();
    void flight.flyTo(DEST, { durationMs: 1000 });
    expect(driver.addPerFrameCallback).toHaveBeenCalledWith(
      FLIGHT_CALLBACK_ID,
      expect.any(Function),
      { continuous: true }
    );
    expect(driver.startAnimation).toHaveBeenCalledTimes(1);
    expect(flight.isActive).toBe(true);
  });

  it('moves the camera along the path and lands exactly on the pose', async () => {
    const { flight, camera, controls } = makeFlight();
    const done = flight.flyTo(DEST, { durationMs: 1000, easing: 'linear' });

    now = 1500;
    driver.tick();
    // Mid-flight: still 10 units from the (moving) target, controls re-seated.
    const target = controls.getFocusTarget();
    expect(camera.position.distanceTo(target)).toBeCloseTo(10, 5);
    expect(target.x).toBeCloseTo(5, 5);
    expect(controls.reinitialize).toHaveBeenCalled();
    expect(controls.dispatchEvent).toHaveBeenCalledWith({ type: 'change' });
    expect(flight.isActive).toBe(true);

    now = 2001;
    driver.tick();
    await expect(done).resolves.toEqual({ completed: true });
    expect(camera.position.toArray()).toEqual([20, 0, 0]);
    expect(controls.getFocusTarget().toArray()).toEqual([10, 0, 0]);
    expect(camera.up.toArray()).toEqual([0, 0, 1]);
    expect(camera.fov).toBe(40);
    expect(camera.near).toBe(0.5);
    expect(camera.far).toBe(500);
    expect(flight.isActive).toBe(false);
    expect(driver.has(FLIGHT_CALLBACK_ID)).toBe(false);
  });

  it('applies the pose immediately for a non-positive duration', async () => {
    const { flight, camera } = makeFlight();
    await expect(flight.flyTo(DEST, { durationMs: 0 })).resolves.toEqual({ completed: true });
    expect(camera.position.toArray()).toEqual([20, 0, 0]);
    expect(driver.addPerFrameCallback).not.toHaveBeenCalled();
  });

  it('user input on the canvas cancels the flight where it is', async () => {
    const { flight, camera } = makeFlight();
    const done = flight.flyTo(DEST, { durationMs: 1000, easing: 'linear' });
    now = 1500;
    driver.tick();
    const midPosition = camera.position.clone();

    canvas.dispatchEvent(new Event('pointerdown', { bubbles: true }));

    await expect(done).resolves.toEqual({ completed: false });
    expect(camera.position.equals(midPosition)).toBe(true);
    expect(driver.has(FLIGHT_CALLBACK_ID)).toBe(false);
    // Listener released: a later pointerdown is not observed by a dead flight.
    now = 3000;
    driver.tick();
    expect(camera.position.equals(midPosition)).toBe(true);
  });

  it('keyboard input cancels too (dispatched at the document)', async () => {
    const { flight } = makeFlight();
    const done = flight.flyTo(DEST, { durationMs: 1000 });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    await expect(done).resolves.toEqual({ completed: false });
  });

  it('typing or adjusting an input does not cancel a flight', async () => {
    const { flight } = makeFlight();
    const done = flight.flyTo(DEST, { durationMs: 1000 });
    const text = document.createElement('input');
    const range = document.createElement('input');
    range.type = 'range';
    document.body.append(text, range);

    text.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true }));
    range.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));

    expect(flight.isActive).toBe(true);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    await expect(done).resolves.toEqual({ completed: false });
  });

  it('a newer flyTo supersedes the active one', async () => {
    const { flight, camera } = makeFlight();
    const first = flight.flyTo(DEST, { durationMs: 1000 });
    const second = flight.flyTo(
      { ...DEST, position: [0, 30, 0], target: [0, 0, 0] },
      { durationMs: 10 }
    );
    await expect(first).resolves.toEqual({ completed: false });
    now = 1011;
    driver.tick();
    await expect(second).resolves.toEqual({ completed: true });
    expect(camera.position.toArray()).toEqual([0, 30, 0]);
  });

  it('cancel() and dispose() are idempotent and resolve a pending flight', async () => {
    const { flight } = makeFlight(null);
    const done = flight.flyTo(DEST, { durationMs: 1000 });
    flight.cancel();
    flight.cancel();
    flight.dispose();
    await expect(done).resolves.toEqual({ completed: false });
    expect(driver.removePerFrameCallback).toHaveBeenCalledTimes(1);
  });

  it('keepOrientation: a turntable keeps spinning through the flight and the landing', async () => {
    const { flight, camera, controls } = makeFlight();
    // Start 10 units along +z of the origin; DEST is 10 units along +x of (10,0,0).
    const done = flight.flyTo(DEST, { durationMs: 1000, easing: 'linear', keepOrientation: true });

    // Emulate what controls.update() does before each flight frame under
    // auto-rotate: swing the camera about the current target by 90° per frame.
    const spin = (): void => {
      const target = controls.getFocusTarget();
      const offset = camera.position.clone().sub(target);
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
      camera.position.copy(target.add(offset));
    };

    now = 1500;
    spin(); // +z → +x
    driver.tick();
    let target = controls.getFocusTarget();
    let dir = camera.position.clone().sub(target).normalize();
    expect(target.x).toBeCloseTo(5, 5); // target still travels
    expect(camera.position.distanceTo(target)).toBeCloseTo(10, 5); // distance still travels
    expect(dir.x).toBeCloseTo(1, 5); // but the direction is the turntable's, not the path's
    expect(camera.up.toArray()).toEqual([0, 1, 0]); // and up is kept, not DEST's +z

    now = 2001;
    spin(); // +x → -z
    driver.tick();
    await expect(done).resolves.toEqual({ completed: true });
    target = controls.getFocusTarget();
    dir = camera.position.clone().sub(target).normalize();
    expect(target.toArray()).toEqual([10, 0, 0]);
    expect(camera.position.distanceTo(target)).toBeCloseTo(10, 5);
    expect(dir.z).toBeCloseTo(-1, 5); // landed where the spin had got to
    expect(camera.fov).toBe(40); // projection parameters still arrive
  });

  it('keepOrientation with a zero duration reseats the pose immediately', async () => {
    const { flight, camera, controls } = makeFlight();
    await expect(flight.flyTo(DEST, { durationMs: 0, keepOrientation: true })).resolves.toEqual({
      completed: true,
    });
    const target = controls.getFocusTarget();
    expect(target.toArray()).toEqual([10, 0, 0]);
    // Live direction was +z (camera at (0,0,10) looking at the origin).
    expect(camera.position.toArray().map((v) => +v.toFixed(6))).toEqual([10, 0, 10]);
  });

  it('under dynamic clipping the flight never writes near/far (the per-frame updater owns them)', async () => {
    const { flight, camera } = makeFlight(canvas, true);
    // Planes as the per-frame updater left them for the current view.
    camera.near = 0.02;
    camera.far = 80;

    const done = flight.flyTo(DEST, { durationMs: 1000, easing: 'linear' });
    now = 1500;
    driver.tick();
    // Mid-flight: planes untouched (DEST would have pulled them to 0.5 / 500).
    expect(camera.near).toBe(0.02);
    expect(camera.far).toBe(80);
    now = 2001;
    driver.tick();
    await expect(done).resolves.toEqual({ completed: true });
    // Landing (restoreCamera) honours the same rule; position still lands exactly.
    expect(camera.near).toBe(0.02);
    expect(camera.far).toBe(80);
    expect(camera.position.toArray()).toEqual([20, 0, 0]);
  });

  it('with no input element, pointer events cannot cancel', async () => {
    const { flight } = makeFlight(null);
    const done = flight.flyTo(DEST, { durationMs: 1000 });
    canvas.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(flight.isActive).toBe(true);
    now = 2001;
    driver.tick();
    await expect(done).resolves.toEqual({ completed: true });
  });
});
