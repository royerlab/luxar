/**
 * Waypoint driver — binds authored camera poses to hidden-dimension positions.
 *
 * A scene tells a story by giving itself a hidden discrete "story" dimension
 * and authoring, per value, the colours (data), the captions (overlays with a
 * `visible_range`) and — through `viewer_config.waypoints` — the camera. The
 * first two already worked; this module adds the third with the SAME matching
 * rule the overlay manager uses for `visible_range`, so one vocabulary
 * describes both "show this caption here" and "look from here":
 *
 * - every known dimension in `when` must match; unknown names are skipped, but
 *   a clause containing no known dimensions never matches;
 * - an exact value matches within ±0.5 of the current step;
 * - a `[min, max]` range matches inclusively;
 * - the FIRST matching waypoint in list order wins.
 *
 * Behaviour is keyed on the MATCHED WAYPOINT changing, not on every slider
 * tick: a move that stays inside the same waypoint's ranges does nothing, and
 * leaving every waypoint leaves the camera where it is. At load the matched
 * waypoint is applied as a snap (the opening framing); afterwards a change of
 * match is a `flyTo` with the waypoint's own duration and easing, and the
 * optional `rendering` block rides the same override path an authored
 * `viewer_config` takes.
 *
 * Pure functions (`matchWaypoint`, `resolveWaypointPose`) carry the logic;
 * the class only sequences them against injected ports, so `LuxarApp` wires
 * it with the pieces it already owns (dims manager, camera flight, rendering
 * controls) and tests drive it with fakes.
 */

import * as THREE from 'three';
import type { ZarrCameraConfig, ZarrWaypoint, ZarrWaypointCondition } from '../../../types/zarr';
import type { SimpleDims } from '../../../types/dims';
import type { CameraSnapshot } from '../snapshot/viewer-snapshot';
import type { FlightResult, FlyToOptions } from './camera-flight';
import { log, Modules } from '../../../utils/log';

/** Tolerance for an exact-value clause — the overlay manager's rule. */
const EXACT_MATCH_TOLERANCE = 0.5;

/** The subset of the dims manager's state the matcher reads. */
export type WaypointDims = Pick<SimpleDims, 'currentStep' | 'metadata'>;

/**
 * Whether `when` holds for the current dimension state. Unknown dimension
 * names are skipped, but a clause containing no known dimensions never
 * matches. An exact value matches within ±0.5; a range matches inclusively.
 */
export function waypointMatches(when: ZarrWaypointCondition, dims: WaypointDims): boolean {
  const metadata = dims.metadata;
  if (!metadata) return false;
  let knownDimensions = 0;
  for (const [dimName, constraint] of Object.entries(when)) {
    const dimIndex = metadata.findIndex((m) => m.name === dimName);
    if (dimIndex < 0) continue;
    knownDimensions += 1;
    const current = dims.currentStep[dimIndex];
    if (current === undefined) continue;
    if (!constraintHolds(current, constraint)) return false;
  }
  return knownDimensions > 0;
}

/**
 * One `when` entry against one coordinate: a number matches within ±0.5, a
 * two-element range inclusively; anything else is ignored (the overlay rule).
 */
function constraintHolds(current: number, constraint: unknown): boolean {
  if (typeof constraint === 'number') {
    return Math.abs(current - constraint) <= EXACT_MATCH_TOLERANCE;
  }
  if (Array.isArray(constraint) && constraint.length === 2) {
    return current >= constraint[0] && current <= constraint[1];
  }
  return true;
}

/** Index of the first waypoint whose `when` matches, or -1. */
export function matchWaypoint(waypoints: readonly ZarrWaypoint[], dims: WaypointDims): number {
  for (let i = 0; i < waypoints.length; i++) {
    const when = waypoints[i]?.when;
    if (when && typeof when === 'object' && waypointMatches(when, dims)) return i;
  }
  return -1;
}

export interface ResolvePoseDeps {
  /** Named-node → bounding-box centre; `null` when the node is unknown or empty. */
  resolveNodeCenter: (name: string) => THREE.Vector3 | null;
  /** `fov_preset` name → degrees (the camera config's preset table). */
  fovPresets: Readonly<Record<string, number>>;
}

/**
 * Turn an authored camera block into a full `CameraSnapshot`, starting from
 * the LIVE pose so every field the author left out keeps its current value.
 * That is what lets a waypoint name only a `target_node` to re-aim without
 * moving, or only a `position` to move without re-aiming. `target_node` wins
 * over `target` (the scene-level camera block's rule); an unresolvable node
 * warns and falls back to `target`, then to the live target.
 */
export function resolveWaypointPose(
  camera: ZarrCameraConfig,
  live: CameraSnapshot,
  deps: ResolvePoseDeps
): CameraSnapshot {
  const pose: CameraSnapshot = { ...live };
  if (camera.position) pose.position = [...camera.position] as [number, number, number];
  if (camera.up) pose.up = [...camera.up] as [number, number, number];
  const target = resolveTarget(camera, deps);
  if (target) pose.target = target;
  const fov = resolveFov(camera, deps);
  if (fov !== undefined) pose.fov = fov;
  if (typeof camera.near === 'number') pose.near = camera.near;
  if (typeof camera.far === 'number') pose.far = camera.far;
  // Orthographic framing: distance changes nothing under an ortho projection,
  // only zoom does, so a waypoint that wants to frame tighter authors `zoom`.
  if (typeof camera.zoom === 'number' && camera.zoom > 0) pose.zoom = camera.zoom;
  return pose;
}

/** `target_node` (resolved) wins over `target`; neither → keep the live target. */
function resolveTarget(
  camera: ZarrCameraConfig,
  deps: ResolvePoseDeps
): [number, number, number] | undefined {
  if (camera.target_node) {
    const centre = deps.resolveNodeCenter(camera.target_node);
    if (centre) return [centre.x, centre.y, centre.z];
    log.warning(
      Modules.APP,
      `waypoint target_node '${camera.target_node}' not found in scene graph; using target`
    );
  }
  return camera.target ? ([...camera.target] as [number, number, number]) : undefined;
}

/** An explicit `fov` wins over a `fov_preset`; an unknown preset is ignored. */
function resolveFov(camera: ZarrCameraConfig, deps: ResolvePoseDeps): number | undefined {
  if (typeof camera.fov === 'number') return camera.fov;
  if (!camera.fov_preset) return undefined;
  const preset = deps.fovPresets[camera.fov_preset];
  return typeof preset === 'number' && preset > 0 ? preset : undefined;
}

export interface WaypointPorts {
  getDims: () => WaypointDims | null;
  getLivePose: () => CameraSnapshot;
  resolvePose: (camera: ZarrCameraConfig, live: CameraSnapshot) => CameraSnapshot;
  /** Instant application of the authored pose (the load-time framing). */
  snapTo: (pose: CameraSnapshot) => void;
  flyTo: (pose: CameraSnapshot, opts: FlyToOptions) => Promise<FlightResult>;
  /**
   * Whether the orbit turntable is currently spinning the camera. While it
   * is, a story step must not swing the camera to the author's azimuth: the
   * flight keeps the live direction (`keepOrientation`) and only the target
   * and distance travel, so the spin continues uninterrupted around the new
   * point of interest.
   */
  autoRotateActive: () => boolean;
  /** Snake_case rendering overrides — the authored `viewer_config` path. */
  applyRendering: (rendering: Record<string, unknown>) => void;
  /**
   * The two story events (`SOUND_SPEC.md` §4.3): `waypoint-departed` fires the
   * moment the matched waypoint changes away from one; `waypoint-arrived`
   * fires after the new waypoint's flight resolves (or immediately after a
   * snap, or immediately when it has no camera block). A flight the visitor
   * cancels still resolves and still arrives (`completed: false`); a flight a
   * NEWER waypoint superseded does not arrive at all — the visitor has moved
   * on, and its narration must not overlap the new story's.
   */
  emit?: (event: WaypointEvent['event'], payload: WaypointEvent['payload']) => void;
}

/** The waypoint driver's events on the embedder bus. */
export interface WaypointEventMap {
  'waypoint-departed': { index: number };
  'waypoint-arrived': { index: number; completed: boolean };
}
export type WaypointEventName = keyof WaypointEventMap;
/** One emitted event as a discriminated pair, for the port's non-generic signature. */
export type WaypointEvent = {
  [K in WaypointEventName]: { event: K; payload: WaypointEventMap[K] };
}[WaypointEventName];

/** How a newly matched waypoint is reached. */
export type WaypointArrival = 'snap' | 'fly';

/**
 * Sequences waypoint matching against dimension changes. One per loaded
 * scene; `LuxarApp` creates it in `applyViewerConfigState` and drops it on the
 * next load.
 */
export class WaypointDriver {
  private current = -1;

  constructor(
    private readonly waypoints: readonly ZarrWaypoint[],
    private readonly ports: WaypointPorts
  ) {}

  /** Index of the waypoint currently in effect, or -1. */
  get currentIndex(): number {
    return this.current;
  }

  /**
   * Re-match against the live dims and, if the matched waypoint changed,
   * reach it. `'snap'` is the load-time call; dimension changes use `'fly'`.
   * Returns the new index (or -1).
   */
  evaluate(arrival: WaypointArrival = 'fly'): number {
    const dims = this.ports.getDims();
    if (!dims) return this.current;
    const idx = matchWaypoint(this.waypoints, dims);
    if (idx === this.current) return idx;
    const previous = this.current;
    this.current = idx;
    if (previous >= 0) this.ports.emit?.('waypoint-departed', { index: previous });
    if (idx < 0) return idx;

    const wp = this.waypoints[idx];
    const flight = wp.camera && typeof wp.camera === 'object' ? this.reach(wp, wp.camera, arrival) : null;
    if (wp.rendering && typeof wp.rendering === 'object') {
      this.ports.applyRendering(wp.rendering);
    }
    log.info(Modules.APP, `Waypoint ${idx} reached (${arrival})`);
    this.announceArrival(idx, flight);
    return idx;
  }

  /**
   * `waypoint-arrived`: right away for a snap or a camera-less waypoint, else
   * when the flight resolves — unless a newer match superseded it meanwhile
   * (see the port docs).
   */
  private announceArrival(idx: number, flight: Promise<FlightResult> | null): void {
    if (!flight) {
      this.ports.emit?.('waypoint-arrived', { index: idx, completed: true });
      return;
    }
    void flight.then((result) => {
      if (this.current !== idx) return;
      this.ports.emit?.('waypoint-arrived', { index: idx, completed: result.completed });
    });
  }

  /** The waypoint at `index`, for a listener that needs its `when` clause. */
  getWaypoint(index: number): ZarrWaypoint | undefined {
    return this.waypoints[index];
  }

  /**
   * Apply a waypoint's camera: verbatim at load (returns null), as a flight on
   * a story step (returns the flight, for the arrival event).
   */
  private reach(
    wp: ZarrWaypoint,
    camera: ZarrCameraConfig,
    arrival: WaypointArrival
  ): Promise<FlightResult> | null {
    const pose = this.ports.resolvePose(camera, this.ports.getLivePose());
    if (arrival === 'snap') {
      // Load-time framing: the authored pose verbatim.
      this.ports.snapTo(pose);
      return null;
    }
    // A story step. `duration_ms: 0` is still a flight of zero length so the
    // turntable rule applies to it too.
    return this.ports.flyTo(pose, this.flightOptions(wp));
  }

  private flightOptions(wp: ZarrWaypoint): FlyToOptions {
    const opts: FlyToOptions = {};
    if (typeof wp.duration_ms === 'number') opts.durationMs = wp.duration_ms;
    if (wp.easing) opts.easing = wp.easing;
    if (this.ports.autoRotateActive()) opts.keepOrientation = true;
    return opts;
  }

  /** Forget the current match so the next `evaluate` re-applies it. */
  reset(): void {
    this.current = -1;
  }
}
