/**
 * WaypointDriver — authored camera poses bound to hidden-dimension positions.
 *
 * The matcher must agree with the overlay manager's `visible_range` rule, and
 * the driver must act on a CHANGE of matched waypoint, not on every tick.
 */
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  WaypointDriver,
  matchWaypoint,
  resolveWaypointPose,
  waypointMatches,
  type WaypointDims,
  type WaypointPorts,
} from '../../../../../core/app/camera/waypoint-driver';
import type { CameraSnapshot } from '../../../../../core/app/snapshot/viewer-snapshot';
import type { ZarrWaypoint } from '../../../../../types/zarr';

function dims(currentStep: number[], names = ['x', 'y', 'z', 'story', 'time']): WaypointDims {
  return {
    currentStep,
    metadata: names.map((name) => ({ name, unit: '', display: false })),
  } as unknown as WaypointDims;
}

const LIVE: CameraSnapshot = {
  position: [0, 0, 10],
  target: [0, 0, 0],
  up: [0, 1, 0],
  isOrtho: false,
  fov: 60,
  near: 0.1,
  far: 1000,
};

const WAYPOINTS: ZarrWaypoint[] = [
  { when: { story: 0 }, camera: { position: [0, 0, 5] } },
  { when: { story: 1, time: [10, 20] }, camera: { position: [5, 0, 0] }, duration_ms: 800 },
  { when: { story: 1 }, camera: { position: [9, 9, 9] }, duration_ms: 0 },
  { when: { story: 2 }, camera: { target_node: 'clusterC' }, rendering: { exposure: 0.5 } },
];

describe('waypointMatches / matchWaypoint (the overlay visible_range rule)', () => {
  it('exact values match within ±0.5 of the current step', () => {
    expect(waypointMatches({ story: 1 }, dims([0, 0, 0, 1, 0]))).toBe(true);
    expect(waypointMatches({ story: 1 }, dims([0, 0, 0, 1.4, 0]))).toBe(true);
    expect(waypointMatches({ story: 1 }, dims([0, 0, 0, 1.6, 0]))).toBe(false);
  });

  it('ranges match inclusively and every clause must hold', () => {
    expect(waypointMatches({ story: 1, time: [10, 20] }, dims([0, 0, 0, 1, 10]))).toBe(true);
    expect(waypointMatches({ story: 1, time: [10, 20] }, dims([0, 0, 0, 1, 20]))).toBe(true);
    expect(waypointMatches({ story: 1, time: [10, 20] }, dims([0, 0, 0, 1, 21]))).toBe(false);
    expect(waypointMatches({ story: 1, time: [10, 20] }, dims([0, 0, 0, 2, 15]))).toBe(false);
  });

  it('skips dimension names the scene does not have, like the overlay manager', () => {
    expect(waypointMatches({ story: 0, nonexistent: 7 }, dims([0, 0, 0, 0, 0]))).toBe(true);
  });

  it('rejects a clause when none of its dimension names exist in the scene', () => {
    const typoFirst: ZarrWaypoint[] = [
      { when: { stroy: 0 }, camera: { position: [9, 9, 9] } },
      { when: { story: 1 }, camera: { position: [1, 2, 3] } },
    ];

    expect(waypointMatches({ stroy: 0 }, dims([0, 0, 0, 1, 0]))).toBe(false);
    expect(matchWaypoint(typoFirst, dims([0, 0, 0, 1, 0]))).toBe(1);
  });

  it('returns false with no metadata (dims not ready)', () => {
    expect(waypointMatches({ story: 0 }, { currentStep: [0, 0, 0, 0, 0] })).toBe(false);
  });

  it('first match in list order wins', () => {
    // story=1, time=15 matches both the ranged entry (index 1) and the broad
    // story=1 entry (index 2); the earlier, more specific one wins.
    expect(matchWaypoint(WAYPOINTS, dims([0, 0, 0, 1, 15]))).toBe(1);
    expect(matchWaypoint(WAYPOINTS, dims([0, 0, 0, 1, 50]))).toBe(2);
    expect(matchWaypoint(WAYPOINTS, dims([0, 0, 0, 7, 0]))).toBe(-1);
  });
});

describe('resolveWaypointPose', () => {
  const deps = {
    resolveNodeCenter: (name: string) => (name === 'clusterC' ? new THREE.Vector3(1, 2, 3) : null),
    fovPresets: { '85mm Portrait': 24 },
  };

  it('starts from the live pose so omitted fields keep their current value', () => {
    const pose = resolveWaypointPose({ position: [7, 7, 7] }, LIVE, deps);
    expect(pose.position).toEqual([7, 7, 7]);
    expect(pose.target).toEqual([0, 0, 0]);
    expect(pose.up).toEqual([0, 1, 0]);
    expect(pose.fov).toBe(60);
    expect(pose.near).toBe(0.1);
    // A copy, not the live object.
    expect(pose).not.toBe(LIVE);
  });

  it('resolves target_node over target, falling back to target when unknown', () => {
    const hit = resolveWaypointPose({ target: [4, 4, 4], target_node: 'clusterC' }, LIVE, deps);
    expect(hit.target).toEqual([1, 2, 3]);
    const miss = resolveWaypointPose({ target: [4, 4, 4], target_node: 'nope' }, LIVE, deps);
    expect(miss.target).toEqual([4, 4, 4]);
    const neither = resolveWaypointPose({ target_node: 'nope' }, LIVE, deps);
    expect(neither.target).toEqual([0, 0, 0]);
  });

  it('applies fov, fov_preset, near and far', () => {
    expect(resolveWaypointPose({ fov: 30 }, LIVE, deps).fov).toBe(30);
    expect(resolveWaypointPose({ fov_preset: '85mm Portrait' }, LIVE, deps).fov).toBe(24);
    expect(resolveWaypointPose({ fov_preset: 'unknown' }, LIVE, deps).fov).toBe(60);
    const clipped = resolveWaypointPose({ near: 1, far: 50 }, LIVE, deps);
    expect(clipped.near).toBe(1);
    expect(clipped.far).toBe(50);
  });

  it('carries an authored ortho zoom and ignores a non-positive one', () => {
    const ortho: CameraSnapshot = { ...LIVE, isOrtho: true, fov: undefined, zoom: 1 };
    expect(resolveWaypointPose({ zoom: 3 }, ortho, deps).zoom).toBe(3);
    expect(resolveWaypointPose({ zoom: 0 }, ortho, deps).zoom).toBe(1);
    expect(resolveWaypointPose({}, ortho, deps).zoom).toBe(1);
  });
});

describe('WaypointDriver', () => {
  function makePorts(current: { step: number[] | null; autoRotate?: boolean }) {
    const snapTo = vi.fn<WaypointPorts['snapTo']>();
    const flyTo = vi.fn<WaypointPorts['flyTo']>(() => Promise.resolve({ completed: true }));
    const applyRendering = vi.fn<WaypointPorts['applyRendering']>();
    const ports = {
      getDims: () => (current.step ? dims(current.step) : null),
      getLivePose: () => LIVE,
      resolvePose: (camera, live) =>
        resolveWaypointPose(camera, live, {
          resolveNodeCenter: () => new THREE.Vector3(1, 2, 3),
          fovPresets: {},
        }),
      snapTo,
      flyTo,
      applyRendering,
      autoRotateActive: () => current.autoRotate === true,
    } satisfies WaypointPorts;
    return ports;
  }

  it('while the turntable spins, a story step keeps the orientation (target + distance travel)', () => {
    const current = { step: [0, 0, 0, 0, 0], autoRotate: true };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);
    driver.evaluate('snap');
    // Load-time framing is still the authored pose, spin or no spin.
    expect(ports.snapTo).toHaveBeenCalledTimes(1);

    current.step = [0, 0, 0, 1, 15];
    driver.evaluate('fly');
    expect(ports.flyTo).toHaveBeenCalledWith(expect.objectContaining({ position: [5, 0, 0] }), {
      durationMs: 800,
      keepOrientation: true,
    });

    // Turntable off again: the author's orientation is honoured.
    current.autoRotate = false;
    current.step = [0, 0, 0, 2, 0];
    driver.evaluate('fly');
    expect(ports.flyTo).toHaveBeenLastCalledWith(expect.anything(), {});
  });

  it('snaps at load and flies on a change of matched waypoint', () => {
    const current = { step: [0, 0, 0, 0, 0] };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);

    expect(driver.evaluate('snap')).toBe(0);
    expect(ports.snapTo).toHaveBeenCalledWith(expect.objectContaining({ position: [0, 0, 5] }));
    expect(ports.flyTo).not.toHaveBeenCalled();

    current.step = [0, 0, 0, 1, 15];
    expect(driver.evaluate('fly')).toBe(1);
    expect(ports.flyTo).toHaveBeenCalledWith(expect.objectContaining({ position: [5, 0, 0] }), {
      durationMs: 800,
    });
  });

  it('does nothing while the matched waypoint is unchanged, or when none matches', () => {
    const current = { step: [0, 0, 0, 1, 12] };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);
    driver.evaluate('snap');
    ports.snapTo.mockClear();

    // Slider moves inside the same waypoint's time range: no re-fly.
    current.step = [0, 0, 0, 1, 18];
    driver.evaluate('fly');
    expect(ports.flyTo).not.toHaveBeenCalled();
    expect(ports.snapTo).not.toHaveBeenCalled();

    // Leaving every waypoint leaves the camera where it is.
    current.step = [0, 0, 0, 7, 0];
    expect(driver.evaluate('fly')).toBe(-1);
    expect(ports.flyTo).not.toHaveBeenCalled();

    // Coming back re-applies it (the match changed again).
    current.step = [0, 0, 0, 1, 18];
    expect(driver.evaluate('fly')).toBe(1);
    expect(ports.flyTo).toHaveBeenCalledTimes(1);
  });

  it('duration_ms 0 is a zero-length flight (so the turntable rule still applies), and rendering rides along', () => {
    const current = { step: [0, 0, 0, 0, 0] };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);
    driver.evaluate('snap');
    ports.snapTo.mockClear();

    current.step = [0, 0, 0, 1, 99];
    driver.evaluate('fly');
    expect(ports.flyTo).toHaveBeenCalledWith(expect.objectContaining({ position: [9, 9, 9] }), {
      durationMs: 0,
    });
    expect(ports.snapTo).not.toHaveBeenCalled();
    ports.flyTo.mockClear();

    current.step = [0, 0, 0, 2, 0];
    driver.evaluate('fly');
    expect(ports.flyTo).toHaveBeenCalledWith(expect.objectContaining({ target: [1, 2, 3] }), {});
    expect(ports.applyRendering).toHaveBeenCalledWith({ exposure: 0.5 });
  });

  it('is inert until the dims manager has a scene', () => {
    const current: { step: number[] | null } = { step: null };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);
    expect(driver.evaluate('snap')).toBe(-1);
    expect(ports.snapTo).not.toHaveBeenCalled();
    expect(driver.currentIndex).toBe(-1);
  });

  it('emits departed on leaving a waypoint and arrived when the flight resolves (or right after a snap)', async () => {
    const current = { step: [0, 0, 0, 0, 0] };
    const ports = makePorts(current);
    const events: Array<{ event: string; payload: unknown }> = [];
    const driver = new WaypointDriver(WAYPOINTS, {
      ...ports,
      emit: (event, payload) => events.push({ event, payload }),
    });
    driver.evaluate('snap');
    expect(events).toEqual([{ event: 'waypoint-arrived', payload: { index: 0, completed: true } }]);

    current.step = [0, 0, 0, 1, 15];
    driver.evaluate('fly');
    // Departure is synchronous; arrival waits for the flight.
    expect(events.at(-1)).toEqual({ event: 'waypoint-departed', payload: { index: 0 } });
    await Promise.resolve();
    await Promise.resolve();
    expect(events.at(-1)).toEqual({
      event: 'waypoint-arrived',
      payload: { index: 1, completed: true },
    });

    // Leaving every waypoint departs but never arrives.
    current.step = [0, 0, 0, 7, 0];
    driver.evaluate('fly');
    expect(events.at(-1)).toEqual({ event: 'waypoint-departed', payload: { index: 1 } });
    expect(events).toHaveLength(4);
  });

  it('a cancelled flight still arrives (completed: false); a superseded one never does', async () => {
    const current = { step: [0, 0, 0, 0, 0] };
    const ports = makePorts(current);
    const events: Array<{ event: string; payload: unknown }> = [];
    let resolveFirst!: (r: { completed: boolean }) => void;
    ports.flyTo
      .mockImplementationOnce(() => new Promise((r) => (resolveFirst = r)))
      .mockImplementationOnce(() => Promise.resolve({ completed: false }));
    const driver = new WaypointDriver(WAYPOINTS, {
      ...ports,
      emit: (event, payload) => events.push({ event, payload }),
    });
    driver.evaluate('snap');
    events.length = 0;

    current.step = [0, 0, 0, 1, 15];
    driver.evaluate('fly');
    // A newer waypoint wins before the first flight resolves.
    current.step = [0, 0, 0, 2, 0];
    driver.evaluate('fly');
    resolveFirst({ completed: false });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(events).toEqual([
      { event: 'waypoint-departed', payload: { index: 0 } },
      { event: 'waypoint-departed', payload: { index: 1 } },
      // Only the current waypoint arrives — cancelled by the visitor, but current.
      { event: 'waypoint-arrived', payload: { index: 3, completed: false } },
    ]);
    expect(driver.getWaypoint(3)).toBe(WAYPOINTS[3]);
    expect(driver.getWaypoint(9)).toBeUndefined();
  });

  it('reset() forgets the current match so it is re-applied', () => {
    const current = { step: [0, 0, 0, 0, 0] };
    const ports = makePorts(current);
    const driver = new WaypointDriver(WAYPOINTS, ports);
    driver.evaluate('snap');
    driver.reset();
    driver.evaluate('fly');
    expect(ports.flyTo).toHaveBeenCalledTimes(1);
  });

  describe('inTransit (reveal: "on_arrival")', () => {
    const REVEALING: ZarrWaypoint[] = [
      { when: { story: 0 }, camera: { position: [0, 0, 5] }, reveal: 'on_arrival' },
      { when: { story: 1 }, camera: { position: [5, 0, 0] }, reveal: 'on_arrival' },
      { when: { story: 2 }, camera: { position: [9, 9, 9] } },
      // A camera-less waypoint (rendering only) — cast because `camera` is
      // required by the type but optional for the driver.
      {
        when: { story: 3 },
        rendering: { exposure: 0.5 },
        reveal: 'on_arrival',
      } as unknown as ZarrWaypoint,
    ];

    it('is closed after a snap, open during the flight, closed again when it resolves — before the arrival event', async () => {
      const current = { step: [0, 0, 0, 0, 0] };
      const ports = makePorts(current);
      let resolveFlight!: (r: { completed: boolean }) => void;
      ports.flyTo.mockImplementationOnce(() => new Promise((r) => (resolveFlight = r)));
      const seenAtArrival: boolean[] = [];
      const driver = new WaypointDriver(REVEALING, {
        ...ports,
        emit: (event) => {
          if (event === 'waypoint-arrived') seenAtArrival.push(driver.inTransit);
        },
      });
      driver.evaluate('snap');
      expect(driver.inTransit).toBe(false); // the load-time framing never holds captions

      current.step = [0, 0, 0, 1, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(true);

      resolveFlight({ completed: true });
      await Promise.resolve();
      await Promise.resolve();
      expect(driver.inTransit).toBe(false);
      // The arrival listener already saw the gate open.
      expect(seenAtArrival).toEqual([false, false]);
    });

    it('a visitor-cancelled flight opens the gate (arrival); the default reveal never closes it', async () => {
      const current = { step: [0, 0, 0, 0, 0] };
      const ports = makePorts(current);
      ports.flyTo.mockImplementationOnce(() => Promise.resolve({ completed: false }));
      const driver = new WaypointDriver(REVEALING, ports);
      driver.evaluate('snap');

      current.step = [0, 0, 0, 1, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(true);
      await Promise.resolve();
      await Promise.resolve();
      expect(driver.inTransit).toBe(false);

      // reveal absent (immediate): captions show as the dimension changes.
      current.step = [0, 0, 0, 2, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(false);
    });

    it('a newer match re-decides the gate; the superseded flight resolving changes nothing', async () => {
      const current = { step: [0, 0, 0, 0, 0] };
      const ports = makePorts(current);
      let resolveFirst!: (r: { completed: boolean }) => void;
      ports.flyTo.mockImplementationOnce(() => new Promise((r) => (resolveFirst = r)));
      const driver = new WaypointDriver(REVEALING, ports);
      driver.evaluate('snap');

      current.step = [0, 0, 0, 1, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(true);
      // Story 2 has no reveal: its (immediate) flight opens the gate at once.
      current.step = [0, 0, 0, 2, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(false);
      resolveFirst({ completed: false });
      await Promise.resolve();
      await Promise.resolve();
      expect(driver.inTransit).toBe(false);

      // Leaving every waypoint, or a camera-less waypoint, never holds captions.
      current.step = [0, 0, 0, 1, 0];
      ports.flyTo.mockImplementationOnce(() => new Promise(() => {}));
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(true);
      current.step = [0, 0, 0, 7, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(false);
      current.step = [0, 0, 0, 3, 0];
      driver.evaluate('fly');
      expect(driver.inTransit).toBe(false);
    });
  });
});
