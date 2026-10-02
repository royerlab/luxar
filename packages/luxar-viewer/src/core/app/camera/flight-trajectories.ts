/**
 * Flight trajectories — the camera paths a `CameraFlight` (`./camera-flight.ts`) can follow between
 * two poses, and the one measure of how far each of them travels.
 *
 * Every trajectory is a {@link FlightPath}: `at(s)` gives the pose at path
 * parameter `s ∈ [0, 1]`, and `length` is the path's PERCEIVED length — what a
 * paced flight (`speed`) divides by its speed to get its duration. The length is
 * integrated along the path with one metric for all of them (see
 * {@link perceivedStep}): panning measured in view heights, zooming in log scale
 * and turning in radians. A pan of one screen costs the same at any distance, a
 * 10× zoom the same at any scale, so durations follow what the viewer sees.
 *
 * The trajectories (Python: `luxar.core.trajectories`; user guide: "Waypoint
 * trajectories" in `VIEWER_GUIDE.md`):
 *
 * - `orbit` — the focus target slides in a straight line while the camera's
 *   offset from it turns (slerp) and its distance changes geometrically. The
 *   default: predictable, never passes through the target.
 * - `zoom-pan` — van Wijk & Nuij, "Smooth and efficient zooming and panning"
 *   (InfoVis 2003): the path through (target, view height) space that minimises
 *   perceived motion. Pulls back while it travels and dives in at the end, at a
 *   constant perceived speed. `rho` trades zoom for pan (√2, their preferred
 *   value; larger zooms out further).
 * - `arc` — `orbit` plus an explicit pull-back: the distance rises by
 *   `lift × (target travel)` at mid-flight, a sine bump. A tunable, predictable
 *   hop — unlike `zoom-pan`, the height does not depend on the zoom levels.
 * - `straight` — the camera position and the target both move in straight
 *   lines: a dolly. The camera may pass through the data on the way, which is
 *   the point of choosing it.
 * - `swing` — travels AROUND a pivot (default the scene centre): the camera and
 *   the target each follow a great circle about it, so the camera goes around
 *   the cloud rather than through it. Distances from the pivot change
 *   geometrically.
 * - `fly-through` — first-person travel: the camera moves in a straight line
 *   looking AHEAD along it (`lookAhead`, a fraction of the trip), turning from
 *   the start target over the first `turn` fraction and onto the destination over
 *   the last. The camera faces where it is going, as in a flight.
 * - `via` — two legs through an intermediate pose (`via`), each following the
 *   `leg` trajectory (default `zoom-pan`) and each eased on its own, so the
 *   camera comes to rest at the intermediate pose and sets off again: a return
 *   to the overview between stories, for instance.
 *
 * `straight`, `swing` and `fly-through` define the VIEW DIRECTION along the way,
 * as does `via` when its leg is one of them, so they take it over from the
 * turntable for the flight (see {@link trajectoryOwnsOrientation}); the others
 * carry the turntable's live direction when auto-rotate is on. All but `orbit`
 * need perspective at both ends; an orthographic flight follows `orbit`.
 *
 * @module core/app/camera/flight-trajectories
 */

import * as THREE from 'three';
import type { CameraSnapshot } from '../snapshot/viewer-snapshot';

export type Vec3Tuple = readonly [number, number, number];

/** The trajectory names (also the parameterless spelling of each). */
export type FlightTrajectoryKind =
  'orbit' | 'zoom-pan' | 'arc' | 'straight' | 'swing' | 'fly-through' | 'via';

/** The leg trajectory of a `via` (anything but another `via`). */
export type FlightLegKind = Exclude<FlightTrajectoryKind, 'via'>;

/** A trajectory with its parameters, resolved (poses in world space). */
export type FlightTrajectory =
  | { kind: 'orbit' }
  | { kind: 'zoom-pan'; rho?: number }
  | { kind: 'arc'; lift?: number }
  | { kind: 'straight' }
  | { kind: 'swing'; pivot?: Vec3Tuple }
  | { kind: 'fly-through'; lookAhead?: number; turn?: number }
  | { kind: 'via'; via: CameraSnapshot; leg?: FlightLegKind };

/** A trajectory name or a parameterised trajectory; names stand for the defaults. */
export type FlightTrajectorySpec = FlightTrajectoryKind | FlightTrajectory;

/**
 * Normalise a trajectory spec to its object form. `'via'` alone names no
 * intermediate pose, so it (and anything unknown) means `orbit`.
 */
export function normalizeTrajectory(spec: FlightTrajectorySpec | undefined): FlightTrajectory {
  if (spec == null) return { kind: 'orbit' };
  if (typeof spec === 'object')
    return Object.hasOwn(BUILDERS, spec.kind) ? spec : { kind: 'orbit' };
  if (spec === 'via') return { kind: 'orbit' };
  const known: readonly string[] = ['orbit', 'zoom-pan', 'arc', 'straight', 'swing', 'fly-through'];
  return known.includes(spec) ? ({ kind: spec } as FlightTrajectory) : { kind: 'orbit' };
}

/** Van Wijk & Nuij's ρ (their preferred value, and d3's default). */
export const ZOOM_PAN_RHO = Math.SQRT2;
/** `arc` default: the pull-back at mid-flight, as a fraction of the target travel. */
export const ARC_DEFAULT_LIFT = 0.5;
/** `fly-through` defaults: look this far ahead (fraction of the trip); turn over this fraction at each end. */
export const FLY_THROUGH_DEFAULT_LOOK_AHEAD = 0.2;
export const FLY_THROUGH_DEFAULT_TURN = 0.3;
/** Samples used to integrate a path's perceived length. */
const LENGTH_SAMPLES = 96;

/** A pre-computed interpolation between two poses. */
export interface FlightPath {
  /** The pose at path parameter `s ∈ [0, 1]` (time already eased by the caller). */
  at(s: number): CameraSnapshot;
  /** Perceived length (see the module doc); what a paced flight divides by its speed. */
  readonly length: number;
  /**
   * Optional own timing: the pose at normalised TIME `t`, given the easing. A
   * `via` uses it to ease each leg separately (rest at the intermediate pose).
   */
  atTime?(t: number, ease: (x: number) => number): CameraSnapshot;
}

/** Whether `kind` sets the view direction along the way (overrides the turntable). */
export function trajectoryOwnsOrientation(trajectory: FlightTrajectory): boolean {
  return (
    trajectory.kind === 'straight' ||
    trajectory.kind === 'swing' ||
    trajectory.kind === 'fly-through' ||
    (trajectory.kind === 'via' &&
      (trajectory.leg === 'straight' ||
        trajectory.leg === 'swing' ||
        trajectory.leg === 'fly-through'))
  );
}

// ─── small vector helpers ────────────────────────────────────────────────────

function vec3(a: Vec3Tuple): THREE.Vector3 {
  return new THREE.Vector3(a[0], a[1], a[2]);
}

function tuple(v: THREE.Vector3): [number, number, number] {
  return [v.x, v.y, v.z];
}

/** Exponential interpolation; linear when either end is not positive. */
export function lerpLog(a: number, b: number, s: number): number {
  if (a > 0 && b > 0) return Math.exp(THREE.MathUtils.lerp(Math.log(a), Math.log(b), s));
  return THREE.MathUtils.lerp(a, b, s);
}

/** Rotate unit `from` toward unit `to` by fraction `s` of the angle between them. */
export function slerpDirection(from: THREE.Vector3, to: THREE.Vector3, s: number): THREE.Vector3 {
  const q = new THREE.Quaternion().setFromUnitVectors(from, to);
  const partial = new THREE.Quaternion().slerp(q, s);
  return from.clone().applyQuaternion(partial).normalize();
}

/** View height at `dist` under a vertical field of view of `fovDeg` degrees. */
export function viewHeight(dist: number, fovDeg: number): number {
  return 2 * dist * Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2);
}

function smoothstep(e0: number, e1: number, x: number): number {
  if (e1 <= e0) return x < e0 ? 0 : 1;
  const t = THREE.MathUtils.clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Projection fields interpolated by `s` (fov linearly, ortho zoom geometrically). */
function lerpProjection(
  from: CameraSnapshot,
  to: CameraSnapshot,
  s: number
): Partial<CameraSnapshot> {
  const out: Partial<CameraSnapshot> = {};
  if (from.fov !== undefined && to.fov !== undefined)
    out.fov = THREE.MathUtils.lerp(from.fov, to.fov, s);
  else if (to.fov !== undefined) out.fov = to.fov;
  if (from.zoom !== undefined && to.zoom !== undefined) out.zoom = lerpLog(from.zoom, to.zoom, s);
  else if (to.zoom !== undefined) out.zoom = to.zoom;
  return out;
}

/** The two poses a path joins. */
interface Ends {
  from: CameraSnapshot;
  to: CameraSnapshot;
}

/**
 * Assemble a pose from a position and target, with up, projection and the
 * near/far planes taken as RATIOS to the camera–target distance (interpolated by
 * `s`), so the planes follow a pull-back instead of clipping the data at its top.
 */
function poseFrom(
  ends: Ends,
  s: number,
  position: THREE.Vector3,
  target: THREE.Vector3,
  up: THREE.Vector3
): CameraSnapshot {
  const { from, to } = ends;
  const d0 = Math.max(vec3(from.position).distanceTo(vec3(from.target)), 1e-12);
  const d1 = Math.max(vec3(to.position).distanceTo(vec3(to.target)), 1e-12);
  const dist = Math.max(position.distanceTo(target), 1e-12);
  return {
    position: tuple(position),
    target: tuple(target),
    up: tuple(up),
    isOrtho: to.isOrtho,
    near: THREE.MathUtils.lerp(from.near / d0, to.near / d1, s) * dist,
    far: THREE.MathUtils.lerp(from.far / d0, to.far / d1, s) * dist,
    ...lerpProjection(from, to, s),
  };
}

// ─── the perceived-length metric ─────────────────────────────────────────────

/**
 * Perceived motion between two nearby poses: √(ρ²·pan² + zoom²/ρ² + turn²) with
 * pan the target displacement in view heights (at the mean distance), zoom the
 * change in log distance (or log ortho zoom), turn the change of view direction
 * in radians. With ρ = √2 this is van Wijk & Nuij's metric, under which their
 * zoom-pan path moves at constant speed; turning is added at par with panning.
 */
export function perceivedStep(a: CameraSnapshot, b: CameraSnapshot): number {
  const ta = vec3(a.target);
  const tb = vec3(b.target);
  const offA = vec3(a.position).sub(ta);
  const offB = vec3(b.position).sub(tb);
  const da = Math.max(offA.length(), 1e-12);
  const db = Math.max(offB.length(), 1e-12);
  const fov = a.fov ?? b.fov;
  const height = fov !== undefined ? viewHeight((da + db) / 2, fov) : (da + db) / 2;
  const pan = ta.distanceTo(tb) / Math.max(height, 1e-12);
  const zoom =
    a.zoom !== undefined && b.zoom !== undefined && a.isOrtho
      ? Math.log(Math.max(b.zoom, 1e-12) / Math.max(a.zoom, 1e-12))
      : Math.log(db / da);
  const turn = offA.normalize().angleTo(offB.normalize());
  const rho = ZOOM_PAN_RHO;
  return Math.sqrt(rho * rho * pan * pan + (zoom * zoom) / (rho * rho) + turn * turn);
}

/** Integrate {@link perceivedStep} along `at` over `[0, 1]`. */
export function integrateLength(
  at: (s: number) => CameraSnapshot,
  samples = LENGTH_SAMPLES
): number {
  let total = 0;
  let prev = at(0);
  for (let i = 1; i <= samples; i++) {
    const next = at(i / samples);
    total += perceivedStep(prev, next);
    prev = next;
  }
  return total;
}

function withLength(at: (s: number) => CameraSnapshot): FlightPath {
  return { at, length: integrateLength(at) };
}

// ─── zoom-pan profile (van Wijk & Nuij) ──────────────────────────────────────

/**
 * Van Wijk & Nuij's optimal zoom-pan profile between a view of height `w0` and
 * one of height `w1` whose centres are `travel` apart. `at(f)` returns, for the
 * fraction `f` of the path's length, how far along the travel the view centre is
 * (`along`, 0 → 1) and the view height there. `S` is the path length in their
 * units; pure zooms and no-ops are handled. Mirrors d3-interpolate's
 * `interpolateZoom`.
 */
export function zoomPanProfile(
  travel: number,
  w0: number,
  w1: number,
  rho: number = ZOOM_PAN_RHO
): { S: number; at(f: number): { along: number; width: number } } {
  const rho2 = rho * rho;
  const rho4 = rho2 * rho2;
  if (travel < 1e-12 * Math.max(w0, w1, 1e-30)) {
    const S = Math.log(w1 / w0) / rho;
    return { S: Math.abs(S), at: (f) => ({ along: f, width: w0 * Math.exp(rho * f * S) }) };
  }
  const d2 = travel * travel;
  const b0 = (w1 * w1 - w0 * w0 + rho4 * d2) / (2 * w0 * rho2 * travel);
  const b1 = (w1 * w1 - w0 * w0 - rho4 * d2) / (2 * w1 * rho2 * travel);
  const r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0);
  const r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1);
  const S = (r1 - r0) / rho;
  const coshR0 = Math.cosh(r0);
  return {
    S,
    at(f: number) {
      const s = f * S;
      const along = (w0 / (rho2 * travel)) * (coshR0 * Math.tanh(rho * s + r0) - Math.sinh(r0));
      return { along, width: (w0 * coshR0) / Math.cosh(rho * s + r0) };
    },
  };
}

// ─── the trajectories ────────────────────────────────────────────────────────

interface OrbitParts {
  target0: THREE.Vector3;
  target1: THREE.Vector3;
  dist0: number;
  dist1: number;
  dirA: THREE.Vector3;
  dirB: THREE.Vector3;
  up0: THREE.Vector3;
  up1: THREE.Vector3;
}

function orbitParts(from: CameraSnapshot, to: CameraSnapshot): OrbitParts {
  const target0 = vec3(from.target);
  const target1 = vec3(to.target);
  const offset0 = vec3(from.position).sub(target0);
  const offset1 = vec3(to.position).sub(target1);
  const dist0 = offset0.length();
  const dist1 = offset1.length();
  // A degenerate (zero-length) offset has no direction: borrow the other end's.
  const dir0 = dist0 > 0 ? offset0.clone().divideScalar(dist0) : null;
  const dir1 = dist1 > 0 ? offset1.clone().divideScalar(dist1) : null;
  return {
    target0,
    target1,
    dist0,
    dist1,
    dirA: dir0 ?? dir1 ?? new THREE.Vector3(0, 0, 1),
    dirB: dir1 ?? dir0 ?? new THREE.Vector3(0, 0, 1),
    up0: vec3(from.up).normalize(),
    up1: vec3(to.up).normalize(),
  };
}

/** `orbit`: target lerp, offset slerp, distance geometric, up slerp; planes lerp. */
export function buildOrbitPath(from: CameraSnapshot, to: CameraSnapshot): FlightPath {
  const o = orbitParts(from, to);
  return withLength((s: number): CameraSnapshot => {
    const target = o.target0.clone().lerp(o.target1, s);
    const position = target
      .clone()
      .addScaledVector(slerpDirection(o.dirA, o.dirB, s), lerpLog(o.dist0, o.dist1, s));
    return {
      position: tuple(position),
      target: tuple(target),
      up: tuple(slerpDirection(o.up0, o.up1, s)),
      isOrtho: to.isOrtho,
      near: THREE.MathUtils.lerp(from.near, to.near, s),
      far: THREE.MathUtils.lerp(from.far, to.far, s),
      ...lerpProjection(from, to, s),
    };
  });
}

/** `zoom-pan`: van Wijk & Nuij target and distance; direction and up as `orbit`. */
function buildZoomPanPath(
  from: CameraSnapshot,
  to: CameraSnapshot,
  rho: number
): FlightPath | null {
  const o = orbitParts(from, to);
  if (!(o.dist0 > 0) || !(o.dist1 > 0)) return null;
  const fov0 = from.fov as number;
  const fov1 = to.fov as number;
  const profile = zoomPanProfile(
    o.target0.distanceTo(o.target1),
    viewHeight(o.dist0, fov0),
    viewHeight(o.dist1, fov1),
    rho
  );
  return withLength((s: number): CameraSnapshot => {
    const { along, width } = profile.at(s);
    const target = o.target0.clone().lerp(o.target1, along);
    const fov = THREE.MathUtils.lerp(fov0, fov1, s);
    const dist = width / viewHeight(1, fov);
    const position = target.clone().addScaledVector(slerpDirection(o.dirA, o.dirB, s), dist);
    return poseFrom({ from, to }, s, position, target, slerpDirection(o.up0, o.up1, s));
  });
}

/** `arc`: `orbit` with the distance raised by `lift × travel × sin(πs)`. */
function buildArcPath(from: CameraSnapshot, to: CameraSnapshot, lift: number): FlightPath {
  const o = orbitParts(from, to);
  const bump = lift * o.target0.distanceTo(o.target1);
  return withLength((s: number): CameraSnapshot => {
    const target = o.target0.clone().lerp(o.target1, s);
    const dist = lerpLog(o.dist0, o.dist1, s) + bump * Math.sin(Math.PI * s);
    const position = target.clone().addScaledVector(slerpDirection(o.dirA, o.dirB, s), dist);
    return poseFrom({ from, to }, s, position, target, slerpDirection(o.up0, o.up1, s));
  });
}

/** `straight`: position and target each in a straight line (a dolly). */
function buildStraightPath(from: CameraSnapshot, to: CameraSnapshot): FlightPath {
  const p0 = vec3(from.position);
  const p1 = vec3(to.position);
  const t0 = vec3(from.target);
  const t1 = vec3(to.target);
  const up0 = vec3(from.up).normalize();
  const up1 = vec3(to.up).normalize();
  return withLength((s: number): CameraSnapshot =>
    poseFrom(
      { from, to },
      s,
      p0.clone().lerp(p1, s),
      t0.clone().lerp(t1, s),
      slerpDirection(up0, up1, s)
    )
  );
}

/** A point carried around `pivot`: great-circle direction, geometric radius. */
function swingPoint(
  pivot: THREE.Vector3,
  a: THREE.Vector3,
  b: THREE.Vector3
): (s: number) => THREE.Vector3 {
  const ra = a.clone().sub(pivot);
  const rb = b.clone().sub(pivot);
  const la = ra.length();
  const lb = rb.length();
  // A point ON the pivot has no direction to swing: carry it in a straight line.
  if (la < 1e-9 || lb < 1e-9) return (s) => a.clone().lerp(b, s);
  const ua = ra.divideScalar(la);
  const ub = rb.divideScalar(lb);
  return (s) => pivot.clone().addScaledVector(slerpDirection(ua, ub, s), lerpLog(la, lb, s));
}

/** `swing`: camera and target each on a great circle about `pivot`. */
function buildSwingPath(
  from: CameraSnapshot,
  to: CameraSnapshot,
  pivot: THREE.Vector3
): FlightPath {
  const camera = swingPoint(pivot, vec3(from.position), vec3(to.position));
  const target = swingPoint(pivot, vec3(from.target), vec3(to.target));
  const up0 = vec3(from.up).normalize();
  const up1 = vec3(to.up).normalize();
  return withLength((s: number): CameraSnapshot =>
    poseFrom({ from, to }, s, camera(s), target(s), slerpDirection(up0, up1, s))
  );
}

/**
 * `fly-through`: the camera moves in a straight line looking `lookAhead` of the
 * trip ahead of itself, turning from the start target over the first `turn` of
 * the flight and onto the destination target over the last `turn`.
 */
function buildFlyThroughPath(
  from: CameraSnapshot,
  to: CameraSnapshot,
  lookAhead: number,
  turn: number
): FlightPath | null {
  const p0 = vec3(from.position);
  const p1 = vec3(to.position);
  const travel = p1.clone().sub(p0);
  if (travel.length() < 1e-9) return null; // nowhere to fly: let orbit turn on the spot
  const t0 = vec3(from.target);
  const t1 = vec3(to.target);
  const up0 = vec3(from.up).normalize();
  const up1 = vec3(to.up).normalize();
  const ramp = Math.min(Math.max(turn, 1e-6), 0.5);
  return withLength((s: number): CameraSnapshot => {
    const position = p0.clone().addScaledVector(travel, s);
    const ahead = position.clone().addScaledVector(travel, Math.max(lookAhead, 1e-3));
    const look = ahead.lerp(t0, 1 - smoothstep(0, ramp, s)).lerp(t1, smoothstep(1 - ramp, 1, s));
    return poseFrom({ from, to }, s, position, look, slerpDirection(up0, up1, s));
  });
}

/** `via`: two legs through `via`, each eased on its own (rest at the via pose). */
function buildViaPath(
  from: CameraSnapshot,
  to: CameraSnapshot,
  via: CameraSnapshot,
  leg: FlightLegKind,
  pivot: THREE.Vector3 | null
): FlightPath {
  const a = buildFlightPath(from, via, leg, pivot);
  const b = buildFlightPath(via, to, leg, pivot);
  const total = a.length + b.length;
  const split = total > 0 ? a.length / total : 0.5;
  const at = (s: number): CameraSnapshot =>
    s < split ? a.at(split > 0 ? s / split : 1) : b.at(split < 1 ? (s - split) / (1 - split) : 1);
  return {
    length: total,
    at,
    atTime(t: number, ease: (x: number) => number): CameraSnapshot {
      if (t < split) return a.at(ease(split > 0 ? t / split : 1));
      return b.at(ease(split < 1 ? (t - split) / (1 - split) : 1));
    },
  };
}

/** Builds one trajectory's path; `null` means its geometry is degenerate here. */
type PathBuilder = (
  t: FlightTrajectory,
  from: CameraSnapshot,
  to: CameraSnapshot,
  pivot: THREE.Vector3 | null
) => FlightPath | null;

const BUILDERS: Record<FlightTrajectoryKind, PathBuilder> = {
  orbit: (_t, from, to) => buildOrbitPath(from, to),
  'zoom-pan': (t, from, to) =>
    buildZoomPanPath(from, to, (t as { rho?: number }).rho ?? ZOOM_PAN_RHO),
  arc: (t, from, to) => buildArcPath(from, to, (t as { lift?: number }).lift ?? ARC_DEFAULT_LIFT),
  straight: (_t, from, to) => buildStraightPath(from, to),
  swing: (t, from, to, pivot) => {
    const named = (t as { pivot?: Vec3Tuple }).pivot;
    const centre =
      (named ? vec3(named) : pivot) ?? vec3(from.target).add(vec3(to.target)).multiplyScalar(0.5);
    return buildSwingPath(from, to, centre);
  },
  'fly-through': (t, from, to) => {
    const ft = t as { lookAhead?: number; turn?: number };
    return buildFlyThroughPath(
      from,
      to,
      ft.lookAhead ?? FLY_THROUGH_DEFAULT_LOOK_AHEAD,
      ft.turn ?? FLY_THROUGH_DEFAULT_TURN
    );
  },
  via: (t, from, to, pivot) => {
    const vt = t as { via: CameraSnapshot; leg?: FlightLegKind };
    return buildViaPath(from, to, vt.via, vt.leg ?? 'zoom-pan', pivot);
  },
};

/**
 * Build the path for `spec` between two snapshots. `pivot` is the resolved
 * centre for `swing` when the trajectory names none (the scene centre); without
 * either, `swing` turns about the midpoint of the two targets. Every trajectory
 * but `orbit` needs perspective at both ends and falls back to `orbit` otherwise
 * (and where its own geometry is degenerate).
 */
export function buildFlightPath(
  from: CameraSnapshot,
  to: CameraSnapshot,
  spec: FlightTrajectorySpec = 'orbit',
  pivot: THREE.Vector3 | null = null
): FlightPath {
  const trajectory = normalizeTrajectory(spec);
  const perspective =
    !from.isOrtho && !to.isOrtho && from.fov !== undefined && to.fov !== undefined;
  if (!perspective) return buildOrbitPath(from, to);
  return BUILDERS[trajectory.kind](trajectory, from, to, pivot) ?? buildOrbitPath(from, to);
}
