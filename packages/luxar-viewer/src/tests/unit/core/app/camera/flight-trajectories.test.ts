/**
 * The flight trajectories: each lands exactly on its end poses, has the
 * geometry its name promises, and is measured by the one perceived-length metric
 * that paces every flight. The wrong answers all fly SOMEWHERE plausible, so the
 * assertions pin the defining property of each path rather than its look.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  ARC_DEFAULT_LIFT,
  buildFlightPath,
  integrateLength,
  normalizeTrajectory,
  trajectoryOwnsOrientation,
  zoomPanProfile,
  viewHeight,
  type FlightTrajectorySpec,
} from '../../../../../core/app/camera/flight-trajectories';
import type { CameraSnapshot } from '../../../../../core/app/snapshot/viewer-snapshot';

const v = (a: readonly number[]): THREE.Vector3 => new THREE.Vector3(a[0], a[1], a[2]);

/** A camera `dist` along +z from `target`, perspective, 63° lens. */
function pose(
  target: [number, number, number],
  dist: number,
  axis = new THREE.Vector3(0, 0, 1)
): CameraSnapshot {
  const p = v(target).addScaledVector(axis.clone().normalize(), dist);
  return {
    position: [p.x, p.y, p.z],
    target,
    up: [0, 1, 0],
    isOrtho: false,
    fov: 63,
    near: 0.01,
    far: 1000,
  };
}

const A = pose([0, 0, 0], 2);
const B = pose([30, 0, 0], 2);

const ALL: FlightTrajectorySpec[] = [
  'orbit',
  'zoom-pan',
  'arc',
  'straight',
  'swing',
  'fly-through',
  { kind: 'via', via: pose([15, 0, 0], 40) },
];

describe('every trajectory', () => {
  it.each(ALL.map((t) => [typeof t === 'string' ? t : t.kind, t]))(
    '%s lands exactly on both end poses and has a positive length',
    (_name, spec) => {
      const path = buildFlightPath(A, B, spec);
      for (const [s, end] of [
        [0, A],
        [1, B],
      ] as const) {
        const q = path.at(s);
        expect(v(q.position).distanceTo(v(end.position))).toBeLessThan(1e-6);
        expect(v(q.target).distanceTo(v(end.target))).toBeLessThan(1e-6);
      }
      expect(path.length).toBeGreaterThan(0);
    }
  );

  it('an orthographic flight follows orbit whatever was asked', () => {
    const o0 = { ...A, isOrtho: true, fov: undefined, zoom: 1 };
    const o1 = { ...B, isOrtho: true, fov: undefined, zoom: 2 };
    const orbit = buildFlightPath(o0, o1, 'orbit').at(0.5);
    for (const spec of ALL) expect(buildFlightPath(o0, o1, spec).at(0.5)).toEqual(orbit);
  });
});

describe('the perceived-length metric', () => {
  it("agrees with van Wijk & Nuij's closed-form length on a zoom-pan path", () => {
    const path = buildFlightPath(A, B, 'zoom-pan');
    const S = zoomPanProfile(30, viewHeight(2, 63), viewHeight(2, 63)).S;
    expect(path.length).toBeCloseTo(S, 2);
  });

  it('counts zoom: a 100× zoom is twice a 10× one, wherever it happens', () => {
    const z10 = buildFlightPath(pose([0, 0, 0], 10), pose([0, 0, 0], 1), 'orbit').length;
    const z100 = buildFlightPath(pose([5, 5, 5], 100), pose([5, 5, 5], 1), 'orbit').length;
    expect(z100 / z10).toBeCloseTo(2, 3);
  });

  it('counts turning: the same spot seen from the side costs a quarter turn', () => {
    const side = pose([0, 0, 0], 5, new THREE.Vector3(1, 0, 0));
    expect(integrateLength(buildFlightPath(pose([0, 0, 0], 5), side, 'orbit').at)).toBeCloseTo(
      Math.PI / 2,
      2
    );
  });
});

describe('arc', () => {
  it('rises by lift × travel at mid-flight above the orbit path', () => {
    const dist = (q: CameraSnapshot): number => v(q.position).distanceTo(v(q.target));
    const orbitMid = dist(buildFlightPath(A, B, 'orbit').at(0.5));
    expect(dist(buildFlightPath(A, B, 'arc').at(0.5)) - orbitMid).toBeCloseTo(
      ARC_DEFAULT_LIFT * 30,
      6
    );
    expect(dist(buildFlightPath(A, B, { kind: 'arc', lift: 2 }).at(0.5)) - orbitMid).toBeCloseTo(
      60,
      6
    );
  });
});

describe('straight', () => {
  it('moves the camera along the segment between the two positions', () => {
    const path = buildFlightPath(A, B, 'straight');
    const p0 = v(A.position);
    const p1 = v(B.position);
    const line = new THREE.Line3(p0, p1);
    for (let s = 0; s <= 1; s += 0.125) {
      const p = v(path.at(s).position);
      expect(line.closestPointToPoint(p, true, new THREE.Vector3()).distanceTo(p)).toBeLessThan(
        1e-9
      );
    }
  });
});

describe('swing', () => {
  it('goes around the pivot instead of through it', () => {
    // Two views of a cloud of radius 10 about the origin, from opposite-ish sides.
    const from = pose([0, 0, 0], 12, new THREE.Vector3(1, 0, 0.1));
    const to = pose([0, 0, 0], 12, new THREE.Vector3(-1, 0, 0.1));
    const swing = buildFlightPath(from, to, { kind: 'swing', pivot: [0, 0, 0] });
    const straight = buildFlightPath(from, to, 'straight');
    let swingMin = Infinity;
    let straightMin = Infinity;
    for (let s = 0; s <= 1; s += 0.05) {
      swingMin = Math.min(swingMin, v(swing.at(s).position).length());
      straightMin = Math.min(straightMin, v(straight.at(s).position).length());
    }
    expect(swingMin).toBeCloseTo(12, 6); // stays on the 12-unit sphere
    expect(straightMin).toBeLessThan(10); // a dolly cuts through the cloud
  });

  it('defaults to the scene centre the flight is given, else the targets midpoint', () => {
    const from = pose([10, 0, 0], 3);
    const to = pose([-10, 0, 0], 3);
    const aboutOrigin = buildFlightPath(from, to, 'swing', new THREE.Vector3(0, 0, -50)).at(0.5);
    const aboutMid = buildFlightPath(from, to, 'swing').at(0.5);
    expect(v(aboutOrigin.position).distanceTo(v(aboutMid.position))).toBeGreaterThan(1);
  });
});

describe('fly-through', () => {
  it('looks along its travel mid-flight, and at the destination at the end', () => {
    const from = pose([0, 0, 0], 4);
    const to = pose([0, 0, -40], 4); // fly 40 units down -z, into the data
    const path = buildFlightPath(from, to, 'fly-through');
    const mid = path.at(0.5);
    const look = v(mid.target).sub(v(mid.position)).normalize();
    expect(look.z).toBeCloseTo(-1, 6);
    const end = path.at(1);
    expect(v(end.target).distanceTo(v(to.target))).toBeLessThan(1e-9);
  });
});

describe('via', () => {
  it('passes through the intermediate pose, at rest, with time split by leg length', () => {
    const via = pose([15, 0, 0], 40);
    const path = buildFlightPath(A, B, { kind: 'via', via });
    expect(path.atTime).toBeDefined();
    const ease = (x: number): number => x * x * (3 - 2 * x);
    // Find the split: the time at which the camera is closest to the via pose.
    let best = { t: 0, d: Infinity };
    for (let t = 0; t <= 1; t += 0.001) {
      const d = v((path.atTime as NonNullable<typeof path.atTime>)(t, ease).position).distanceTo(
        v(via.position)
      );
      if (d < best.d) best = { t, d };
    }
    expect(best.d).toBeLessThan(1e-3);
    // At rest there: a small step in time barely moves the camera.
    const at = (t: number): THREE.Vector3 =>
      v((path.atTime as NonNullable<typeof path.atTime>)(t, ease).position);
    expect(at(best.t + 0.002).distanceTo(at(best.t))).toBeLessThan(1e-2);
    // The two legs are symmetric here, so the split is at half time.
    expect(best.t).toBeCloseTo(0.5, 2);
  });
});

describe('names and ownership', () => {
  it('normalises names; a bare via or an unknown name means orbit', () => {
    expect(normalizeTrajectory('arc')).toEqual({ kind: 'arc' });
    expect(normalizeTrajectory('via')).toEqual({ kind: 'orbit' });
    expect(normalizeTrajectory(undefined)).toEqual({ kind: 'orbit' });
    expect(normalizeTrajectory({ kind: 'zoom-pan', rho: 2 })).toEqual({ kind: 'zoom-pan', rho: 2 });
  });

  it('straight, swing and fly-through set the view direction; the others leave it to the turntable', () => {
    const owns = ALL.filter((t) => trajectoryOwnsOrientation(normalizeTrajectory(t))).map((t) =>
      typeof t === 'string' ? t : t.kind
    );
    expect(owns).toEqual(['straight', 'swing', 'fly-through']);
  });
});
