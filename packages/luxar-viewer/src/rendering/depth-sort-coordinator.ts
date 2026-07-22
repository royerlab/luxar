/**
 * Depth-sort coordinator — main-thread side of the SortWorker
 * (depth-sorting Phases 2-3,
 * `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5-§6).
 *
 * Module-scoped live authority (the `element-texture-layout.ts` pattern):
 * the commit path (`commit-gsplats-geometry.ts`) and the app lifecycle
 * are far apart, so both talk to this module instead of threading a
 * coordinator object through constructors.
 *
 * Responsibilities:
 * - Owns the single persistent Comlink SortWorker (spawn on first
 *   order-dependent commit, terminate on app teardown). NOT part of the
 *   round-robin data-worker pool — node registrations and their
 *   transferred center buffers must live in exactly one worker.
 * - Tracks the per-node **generation** counter: bumped on every non-noop
 *   commit (`noteGSplatsCommit`), never on stamp-only noops. An ordering
 *   is applied only when its generation still matches — a stale (shorter)
 *   permutation applied to a grown buffer would be corrupt.
 * - Enforces **at most one in-flight sort per node**; a commit landing
 *   mid-sort queues exactly one re-sort with the then-current generation.
 * - Applies resolved orderings via `writeSortedIndexOrdering` (the pick
 *   node shares the render mesh's geometry object, so one write covers
 *   both) and requests a frame.
 * - Phase 3: keeps the ordering tracking the camera via the per-frame
 *   scheduler `evaluateDepthSortPerFrame` (angle / view-axis-translation
 *   thresholds from `config.depthSort`; `?depthSort=0` disables the whole
 *   subsystem via {@link setDepthSortEnabled}).
 * - Cross-node draw order: the same per-frame pass collects every visible
 *   normal-mode gsplat mesh and assigns ONE global back-to-front
 *   `renderOrder` scale (wrapper groups by mean view-z, exact BSP ranks
 *   within a wrapper) — machinery owned by the
 *   `depth-sort-coordinator/render-order.ts` submodule, driven here via
 *   {@link clearRenderOrderFrameState} / {@link collectRenderOrderSlot} /
 *   {@link assignGlobalRenderOrder}.
 *
 * Ordering only matters for order-dependent blending (`normal`); all
 * other modes are commutative. Commits of non-`normal` nodes still bump
 * the generation (killing any in-flight sort) and release the node's
 * worker-side registration.
 */

import * as THREE from 'three';
import { wrap, transfer, type Remote } from 'comlink';
// Vite's `?worker` import emits a bundled worker chunk (see
// worker-pool.ts for why `new Worker(new URL(...))` is not used).
import SortWorker from '../workers/sort-worker?worker';
import type { SortWorkerAPI } from '../workers/sort-worker';
import { writeSortedIndexOrdering } from './element-storage';
import { needsDepthSort } from './blending-state';
import { clearCommittedData, hasCommittedData } from '../types/committed-data';
import type { BlendingMode } from './material-manager';
import {
  assignGlobalRenderOrder,
  clearRenderOrderFrameState,
  collectRenderOrderSlot,
} from './depth-sort-coordinator/render-order';
import { config } from '../config';
import type { UpdateProfiler } from '../profiling/update-profiler';
import { log, Modules } from '../utils/log';

/**
 * Optional override for the sort-worker module URL (embedders whose
 * bundler lacks `?worker` support). Mirrors {@link setDataWorkerUrl};
 * set via `LuxarAppOptions.workerPath` alongside the data worker's.
 */
let sortWorkerUrlOverride: string | undefined;

/** Override the URL used to construct the sort worker. Call before first use. */
export function setSortWorkerUrl(url: string): void {
  sortWorkerUrlOverride = url;
}

/**
 * WASM JS-shim URL forwarded into the worker's `initialize()` (the
 * main-thread `setWasmJsUrl` override does not cross the worker
 * boundary). Mirrors {@link setDataWorkerWasmPath}.
 */
let sortWorkerWasmPathOverride: string | undefined;

/** Override the WASM JS-shim URL used inside the sort worker. */
export function setSortWorkerWasmPath(url: string): void {
  sortWorkerWasmPathOverride = url;
}

interface NodeSortState {
  /** The node's render mesh (the pick node shares its geometry). */
  mesh: THREE.Mesh;
  /** Lifetime-unique non-noop commit stamp (see nextGeneration). */
  generation: number;
  /** True while a sort RPC is outstanding for this node. */
  inFlight: boolean;
  /** A newer commit landed mid-sort — re-sort once the current one resolves. */
  resortQueued: boolean;
  /**
   * Model-space view axis (the model-view matrix's z-row direction) at the
   * last DISPATCHED sort; null before the first dispatch. The sort kernel
   * orders by view-space z = axis·p + offset, so the resulting permutation
   * depends only on this axis direction and the offset below — the
   * per-frame scheduler (Phase 3) compares against them to decide when a
   * re-sort is due.
   */
  lastSortAxis: THREE.Vector3 | null;
  /** Normalized view-axis offset (m14 / |axis|) at the last dispatched sort. */
  lastSortOffset: number;
  /**
   * True iff centers for the CURRENT generation were dispatched to the
   * worker. Set where the register RPC is issued; cleared on every
   * release branch (empty/commutative commit, mode-switch-away). The
   * per-frame scheduler uses it to recover a node whose FIRST dispatch
   * raced a null camera: registered but `lastSortAxis === null` means
   * "worker has centers, no sort ever left" — dispatch one now.
   */
  registered: boolean;
}

let worker: Worker | null = null;
let api: Remote<SortWorkerAPI> | null = null;
let initPromise: Promise<void> | null = null;
let getCamera: (() => THREE.Camera | null) | null = null;
let requestRender: (() => void) | null = null;
let requestReprocess: (() => void) | null = null;
let isLoadInProgress: (() => boolean) | null = null;
let getProfiler: (() => UpdateProfiler | null) | null = null;
/**
 * Session master switch (Phase 3): config `depthSort.enabled` combined with
 * the `?depthSort=0` URL escape hatch at app init. When false the whole
 * subsystem is inert — commits keep the identity (storage) ordering, the
 * worker is never spawned, and the per-frame scheduler no-ops.
 */
let depthSortEnabled = true;
/** One-shot flag for the SortWorker-unavailable error (see noteGSplatsCommit). */
let warnedWorkerUnavailable = false;
const nodeStates = new Map<string, NodeSortState>();

/**
 * Wire the camera accessor + frame-request + reprocess callbacks. Called
 * once at app init (the commit path has none of these — the scene loader
 * deliberately owns no camera state). Safe to call again on renderer swap.
 */
export function configureDepthSort(options: {
  /**
   * Live camera accessor — a GETTER, not a captured reference: the
   * ortho-mode toggle REPLACES the scene manager's camera object, and a
   * value captured at init would keep sorting from the abandoned
   * perspective camera's frozen pose (the `lod-group-registry`
   * `getCamera` precedent).
   */
  getCamera: () => THREE.Camera | null;
  requestRender: () => void;
  /**
   * Force a full view reprocess (`SceneLoader.updateView({})`) — used by
   * the blending-mode-switch hook, see {@link noteGSplatsBlendingModeSwitch}.
   */
  requestReprocess?: () => void;
  /**
   * True while a view-update sweep is in flight (the same signal the
   * refinement loop consults). The per-frame scheduler skips dispatching
   * camera-motion re-sorts during loads — the pending commit will sort
   * from the then-current pose anyway.
   */
  isLoadInProgress?: () => boolean;
  /**
   * Update-profiler accessor for the 'Depth Sort' monitor line: each
   * SortWorker dispatch opens a detached pass whose duration is the
   * dispatch→applied round-trip latency.
   */
  getProfiler?: () => UpdateProfiler | null;
}): void {
  getCamera = options.getCamera;
  requestRender = options.requestRender;
  requestReprocess = options.requestReprocess ?? null;
  isLoadInProgress = options.isLoadInProgress ?? null;
  getProfiler = options.getProfiler ?? null;
}

/**
 * Session master switch, applied at app init from `config.depthSort.enabled`
 * combined with the `?depthSort=0` URL escape hatch. Disabling pins the
 * identity (storage) ordering for deterministic E2E/visual runs.
 */
export function setDepthSortEnabled(enabled: boolean): void {
  depthSortEnabled = enabled;
}

/** Lazily spawn + initialize the persistent sort worker. */
function ensureWorker(): Promise<void> {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const w = sortWorkerUrlOverride
      ? new Worker(sortWorkerUrlOverride, { type: 'module' })
      : new SortWorker();
    worker = w;
    api = wrap<SortWorkerAPI>(w);
    const result = await api.initialize(sortWorkerWasmPathOverride);
    log.info(
      Modules.WORKER_POOL,
      `SortWorker ready (${result.wasmFallback ? 'TypeScript fallback' : 'WASM'})`
    );
  })();
  initPromise.catch((error) => {
    log.error(Modules.WORKER_POOL, 'SortWorker failed to initialize', error);
  });
  return initPromise;
}

/** Read the live blending mode that actually drives the blend state. */
function liveBlendingMode(mesh: THREE.Mesh): BlendingMode | undefined {
  const material = mesh.material as THREE.Material | THREE.Material[];
  const single = Array.isArray(material) ? material[0] : material;
  return single?.userData?.blendingMode as BlendingMode | undefined;
}

/**
 * Record a non-noop gsplats commit. Always bumps the node's generation
 * (dropping any in-flight sort's result). When the node's LIVE blending
 * mode is order-dependent (`normal`), transfers the projected centers to
 * the SortWorker and requests one sort from the current camera pose.
 *
 * `centers3` is TRANSFERRED (detached) on the order-dependent path — the
 * commit's texture-write loops are the last main-thread readers, and the
 * memoized-concat noop identity keys on `sourceData`, never `processed.*`.
 */
/**
 * Module-scoped monotonic generation source. Generations must be unique
 * across a node's LIFETIMES, not just within one: `releaseDepthSortNode`
 * (LOD demotion) deletes the node state, and a re-promotion recommit
 * would otherwise restart the counter — letting a stale in-flight sort
 * from the previous life pass the `result.generation === current.generation`
 * guard and apply a CORRUPT permutation over the new (differently-sized)
 * commit (found by randomized interleaving fuzz; deterministic repro:
 * demote → re-promote within one sort round-trip). The worker echoes the
 * value opaquely, so uniqueness costs nothing.
 */
let nextGeneration = 0;

export function noteGSplatsCommit(mesh: THREE.Mesh, centers3: Float32Array, count: number): void {
  const nodeId = mesh.uuid;
  let state = nodeStates.get(nodeId);
  if (!state) {
    state = {
      mesh,
      generation: 0,
      inFlight: false,
      resortQueued: false,
      lastSortAxis: null,
      lastSortOffset: 0,
      registered: false,
    };
    nodeStates.set(nodeId, state);
  }
  state.generation = ++nextGeneration;

  const mode = liveBlendingMode(mesh);
  if (!depthSortEnabled || !mode || !needsDepthSort(mode) || count === 0) {
    // Depth sorting disabled (identity ordering pinned), commutative
    // blending, or an empty frame: no ordering needed. Drop any
    // worker-side registration so the worker doesn't hold stale
    // centers for a node that may not sort again for a long time —
    // and the recorded pose with it (see clearSortPose's invariant).
    clearSortPose(state);
    releaseWorkerNode(nodeId);
    return;
  }

  const generation = state.generation;
  void ensureWorker()
    .then(() => {
      if (!api) return;
      // A newer commit may have landed while the worker was spawning.
      const current = nodeStates.get(nodeId);
      if (current?.generation !== generation) return;
      current.registered = true;
      // The register RPC is its own promise — the surrounding .catch
      // only sees synchronous throws, so a transport/transfer rejection
      // here would otherwise float as an unhandled rejection.
      api
        .registerNode(transfer({ nodeId, generation, centers3, count }, [centers3.buffer]))
        .catch((error: unknown) => {
          log.error(Modules.WORKER_POOL, `SortWorker registerNode failed for ${nodeId}`, error);
          // The worker never received this generation's centers — clear
          // the flag (if still ours) so the per-frame scheduler doesn't
          // keep dispatching guaranteed-null sorts against the missing
          // registration until the next commit.
          const failed = nodeStates.get(nodeId);
          if (failed?.generation === generation) failed.registered = false;
        });
      scheduleSort(mesh, nodeId);
    })
    .catch((error) => {
      // Once per session: a failed worker init stays failed (the cached
      // initPromise is rejected), so EVERY later commit lands here —
      // per-commit error lines would flood a timelapse scrub. Rendering
      // degrades gracefully to unsorted normal mode.
      if (!warnedWorkerUnavailable) {
        warnedWorkerUnavailable = true;
        log.error(
          Modules.WORKER_POOL,
          'SortWorker unavailable — depth sorting disabled for this session ' +
            `(first failing node: ${nodeId})`,
          error
        );
      }
    });
}

/**
 * Fire-and-forget worker-side release. Swallows rejections: releases
 * run during teardown flows where the worker may already be
 * terminating, and a never-settling/rejected cleanup RPC is expected
 * there, not actionable.
 */
function releaseWorkerNode(nodeId: string): void {
  api?.releaseNode(nodeId).catch(() => {});
}

/**
 * Clear a node's recorded sort pose + registration + queued re-sort —
 * the invariant behind every release path: a kept `lastSortAxis` would
 * let the per-frame scheduler keep firing guaranteed-null sort RPCs
 * against a released worker registration on every threshold crossing,
 * and a kept `registered` flag would let the recovery branch do the
 * same. Callers pair this with {@link releaseWorkerNode} (and, on
 * mode-switch-away, a generation bump to kill in-flight results).
 */
function clearSortPose(state: NodeSortState): void {
  state.lastSortAxis = null;
  state.lastSortOffset = 0;
  state.registered = false;
  state.resortQueued = false;
}

/**
 * Record the pose a sort was dispatched from (the model-view z-row that
 * fully determines the resulting permutation — see NodeSortState). The
 * per-frame scheduler compares live poses against this.
 */
function recordSortPose(state: NodeSortState, modelView: THREE.Matrix4): void {
  const e = modelView.elements;
  if (!state.lastSortAxis) state.lastSortAxis = new THREE.Vector3();
  state.lastSortAxis.set(e[2], e[6], e[10]);
  const len = state.lastSortAxis.length();
  if (len > 0) {
    state.lastSortAxis.divideScalar(len);
    state.lastSortOffset = e[14] / len;
  } else {
    state.lastSortOffset = e[14];
  }
}

/**
 * Request one sort for a registered node, respecting the
 * single-in-flight rule. Queues a re-sort if one is already running.
 */
function scheduleSort(mesh: THREE.Mesh, nodeId: string): void {
  const state = nodeStates.get(nodeId);
  const camera = getCamera?.();
  if (!state || !api || !camera) return;
  if (state.inFlight) {
    state.resortQueued = true;
    return;
  }
  state.inFlight = true;

  const generation = state.generation;
  // Both matrices are normally renderer-maintained (updated during
  // render), but a commit can fire BEFORE the next frame — the first
  // commit of a load, or while the on-demand loop is idle-paused — and
  // would otherwise read a stale/identity pose. Refresh them here and
  // derive the view matrix locally (camera.matrixWorldInverse is only
  // refreshed by renderer.render, not by updateMatrixWorld).
  mesh.updateWorldMatrix(true, false);
  camera.updateMatrixWorld();
  const viewMatrix = new THREE.Matrix4().copy(camera.matrixWorld).invert();
  const modelView = viewMatrix.multiply(mesh.matrixWorld);
  recordSortPose(state, modelView);

  // One detached profiler pass per dispatch — its duration is the
  // dispatch→applied round-trip the monitor's 'Depth Sort' line shows.
  const session = getProfiler?.()?.beginDepthSortPass() ?? null;

  void api
    .sort({ nodeId, generation, modelView: new Float32Array(modelView.elements) })
    .then((result) => {
      const current = nodeStates.get(nodeId);
      if (!current) {
        session?.end();
        return; // released mid-sort
      }
      current.inFlight = false;

      // Stale-drop: apply only when the ordering matches the node's
      // CURRENT generation (the worker re-checked its own registration;
      // this re-check covers commits that raced the RPC). The
      // `committedData` check additionally covers LOD demotion: a demoted
      // level's geometry returned to the evictable pool (and may since
      // belong to another node) — the cleared stamp is exactly the signal
      // that the mesh's geometry no longer holds this commit's splats.
      const stillCommitted = hasCommittedData(mesh);
      if (result && result.generation === current.generation && stillCommitted) {
        const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
        if (geometry?.getAttribute?.('aSortedIndex')) {
          writeSortedIndexOrdering(geometry, result.ordering, result.ordering.length);
          // Ordering upload = 4 bytes/splat through the attribute
          // update-range machinery (the architecture's headline number).
          const bytes = result.ordering.length * 4;
          session?.setMetadata({
            splats: result.ordering.length,
            info:
              bytes >= 1_000_000
                ? `${(bytes / 1_000_000).toFixed(1)} MB up`
                : `${Math.round(bytes / 1000)} KB up`,
          });
          requestRender?.();
        }
      }
      session?.end();

      if (current.resortQueued) {
        current.resortQueued = false;
        scheduleSort(mesh, nodeId);
      }
    })
    .catch((error) => {
      session?.end();
      const current = nodeStates.get(nodeId);
      log.error(Modules.WORKER_POOL, `SortWorker sort failed for ${nodeId}`, error);
      if (!current) return;
      current.inFlight = false;
      // Drain a queued re-sort even on failure — a commit landed while
      // this sort was out, and dropping its request would leave the node
      // stale until the NEXT commit. Bounded: only a real commit sets
      // resortQueued, so a persistently failing worker cannot loop.
      if (current.resortQueued) {
        current.resortQueued = false;
        scheduleSort(mesh, nodeId);
      }
    });
}

// Per-frame scratch (no allocation on the hot path — the
// 'lod-group-selector' invariant). Allocated lazily on first use rather
// than at module load: several unit-test files partially mock 'three',
// and an import-time `new THREE.Matrix4()` would break every test that
// transitively imports this module.
interface EvaluateScratch {
  view: THREE.Matrix4;
  mv: THREE.Matrix4;
  axis: THREE.Vector3;
  /** Camera world position (feeds the render-order module's BSP traversal). */
  camPos: THREE.Vector3;
}
let scratch: EvaluateScratch | null = null;

/**
 * True when the mesh AND all its ancestors are visible. `mesh.visible`
 * alone misses a hidden ancestor: an LOD level can be a GROUP (partition
 * tiles), and the registry toggles `child.object.visible` on the group —
 * the member meshes' own flags stay true. Sorting a hidden mesh is wasted
 * worker time; when it re-shows, the next frame's pose comparison catches
 * any past-threshold camera motion immediately.
 */
function isEffectivelyVisible(mesh: THREE.Object3D): boolean {
  for (let o: THREE.Object3D | null = mesh; o; o = o.parent) {
    if (!o.visible) return false;
  }
  return true;
}

/**
 * Per-frame camera-motion re-sort scheduler (depth-sorting Phase 3, spec
 * §6). Registered as the 'depth-sort-scheduler' per-frame callback beside
 * 'lod-group-selector'.
 *
 * For each order-dependent node with a completed dispatch on record,
 * compare the live model-view z-row against the pose the last sort was
 * dispatched from and dispatch a re-sort when either
 * - the view axis has rotated past `config.depthSort.angleThresholdDeg`
 *   (relative to the node — a spinning node triggers it too), or
 * - the camera has translated ALONG the view axis past
 *   `config.depthSort.translationFraction` × the node's bounding-sphere
 *   radius (which changes the behind-camera set the kernel clamps to the
 *   far bucket).
 *
 * Translation orthogonal to the view axis is deliberately ignored: the
 * kernel sorts by view-space z = axis·p + offset, so the permutation
 * cannot change unless the axis direction or the offset does.
 *
 * Hysteresis is dispatch-updates-reference: `scheduleSort` records the
 * fresh pose, so a triggered node goes quiet until the camera moves past
 * the threshold AGAIN. Frames between dispatch and resolve render the
 * previous order — bounded staleness, standard 3DGS behavior. Skips:
 * pending view updates (the commit will sort anyway), in-flight sorts
 * (the resolve is at most a frame away), invisible/demoted meshes, and
 * nodes whose live mode is no longer order-dependent.
 */
export function evaluateDepthSortPerFrame(): void {
  // Drop the previous frame's render-order state FIRST — before any
  // early-return — so a disposed/dataset-switched frame can't leave the
  // module-scoped rank memo holding stale partition-wrapper subtrees alive.
  clearRenderOrderFrameState();
  // Deliberately NOT gated on `api`: the cross-node renderOrder pass is
  // pure main-thread and must keep ordering meshes back-to-front even
  // when the SortWorker was never constructed (`api` stays null forever
  // after a constructor throw — e.g. a CSP-blocked worker script — the
  // documented degrade-to-unsorted-normal mode). The within-mesh
  // re-sort triggers are worker-dependent, but `scheduleSort` guards
  // `!api` itself.
  if (!depthSortEnabled || nodeStates.size === 0) return;
  const camera = getCamera?.();
  if (!camera) return;
  if (isLoadInProgress?.()) return;

  if (!scratch) {
    scratch = {
      view: new THREE.Matrix4(),
      mv: new THREE.Matrix4(),
      axis: new THREE.Vector3(),
      camPos: new THREE.Vector3(),
    };
  }
  const cosThreshold = Math.cos((config.depthSort.angleThresholdDeg * Math.PI) / 180);
  let viewComputed = false;

  for (const [nodeId, state] of nodeStates) {
    const mesh = state.mesh;
    if (!isEffectivelyVisible(mesh)) continue;
    // LOD demotion returned the geometry to the pool — same signal the
    // resolve path checks; a sort dispatched now would be dropped there.
    if (!hasCommittedData(mesh)) continue;
    const mode = liveBlendingMode(mesh);
    if (!mode || !needsDepthSort(mode)) {
      // No longer order-dependent (e.g. switched to additive) — clear any
      // cross-part renderOrder bias so it doesn't strand a stale ordering.
      if (mesh.renderOrder !== 0) mesh.renderOrder = 0;
      continue;
    }

    if (!viewComputed) {
      // One-frame-stale matrices are fine for the TRIGGER test (the
      // dispatch itself re-derives fresh ones in scheduleSort), but the
      // camera's matrixWorld must at least exist post-move — cheap when
      // nothing changed.
      camera.updateMatrixWorld();
      scratch.view.copy(camera.matrixWorld).invert();
      scratch.camPos.setFromMatrixPosition(camera.matrixWorld);
      viewComputed = true;
    }

    scratch.mv.multiplyMatrices(scratch.view, mesh.matrixWorld);

    // === Cross-mesh (inter-node) back-to-front ordering: COLLECT ===
    // One order slot per surviving normal-mode mesh; the post-loop
    // assignGlobalRenderOrder() puts everything on ONE global integer
    // renderOrder scale (full rationale in
    // `depth-sort-coordinator/render-order.ts`).
    collectRenderOrderSlot(mesh, scratch.mv, scratch.camPos);
    const bs = (mesh.geometry as THREE.BufferGeometry | undefined)?.boundingSphere;

    // === Within-mesh re-sort trigger (Phase 3) ===
    if (state.inFlight) continue;
    if (!state.lastSortAxis) {
      // Registered with the worker but no sort ever dispatched — the
      // first commit raced a null camera (init ordering / renderer
      // swap window). The camera exists on this frame; recover with
      // one dispatch. No retry-loop risk: `scheduleSort` records the
      // pose BEFORE the RPC, so even a failing sort leaves this branch.
      if (state.registered) scheduleSort(mesh, nodeId);
      continue;
    }
    const e = scratch.mv.elements;
    scratch.axis.set(e[2], e[6], e[10]);
    const len = scratch.axis.length();
    if (len === 0) continue; // degenerate transform — nothing sortable
    scratch.axis.divideScalar(len);
    const offset = e[14] / len;

    let moved = scratch.axis.dot(state.lastSortAxis) < cosThreshold;
    if (!moved) {
      const radius = bs?.radius;
      // Without bounds (never computed / empty) the translation trigger
      // has no scale reference — rely on the angle trigger alone.
      if (radius !== undefined && radius > 0) {
        moved =
          Math.abs(offset - state.lastSortOffset) > config.depthSort.translationFraction * radius;
      }
    }

    if (moved) scheduleSort(mesh, nodeId);
  }

  assignGlobalRenderOrder();
}

/**
 * React to a gsplat layer's blending mode changing at runtime (the
 * LayersPanel compose chain — spec §5.4).
 *
 * Switching TO `normal` cannot simply "register+sort": the SortWorker has
 * no centers for a node that was order-independent at its last commit
 * (registration is gated on the live mode, and the staged arrays were
 * transferred/discarded). Instead, clear the node's noop stamp
 * (`userData.committedData`) and request a view reprocess — the
 * memoized-concat noop path would otherwise skip re-projection entirely.
 * The resulting standard commit registers + sorts like any other (the
 * SliceCache still holds the source nD data; one O(N) re-projection per
 * mode switch, a rare user action).
 *
 * Switching AWAY just stops future sorts (a sorted order is harmless
 * under commutative modes — no identity reset needed); the worker-side
 * centers are released as hygiene.
 */
export function noteGSplatsBlendingModeSwitch(
  mesh: THREE.Mesh,
  newMode: BlendingMode | undefined,
  prevMode: BlendingMode | undefined
): void {
  // Disabled session: identity ordering is pinned for every mode, so a
  // switch TO normal must not force the (expensive) reprocess; there is
  // also no worker-side state to release on a switch away.
  if (!depthSortEnabled) return;
  if (!newMode || newMode === prevMode) return;
  // Sorted modes = normal ∪ volumetric (needsDepthSort). A switch
  // BETWEEN two sorted modes (normal↔volumetric) is deliberately a
  // no-op here: the ordering stays valid; the projection/output change
  // is the material's problem (TSL rebuild / GLSL define recompile).
  const wasSorted = prevMode !== undefined && needsDepthSort(prevMode);
  if (needsDepthSort(newMode) && !wasSorted) {
    clearCommittedData(mesh);
    requestReprocess?.();
  } else if (!needsDepthSort(newMode) && wasSorted) {
    const state = nodeStates.get(mesh.uuid);
    if (state) {
      // Invalidate any in-flight sort's result; keep the counter
      // monotonic for the node's next order-dependent commit. The pose
      // clear is the same hygiene as the commit path's release branch.
      state.generation = ++nextGeneration;
      clearSortPose(state);
    }
    releaseWorkerNode(mesh.uuid);
  }
}

/**
 * Drop a node's sort state + worker-side registration. Wired to gsplats
 * node disposal (an in-flight sort resolves onto a missing state and is
 * discarded).
 */
export function releaseDepthSortNode(mesh: THREE.Mesh): void {
  const nodeId = mesh.uuid;
  if (!nodeStates.delete(nodeId)) return;
  releaseWorkerNode(nodeId);
}

/**
 * Drop every node's sort state + all worker-side registrations in one
 * sweep. Wired to the dataset-switch teardown (`clearLoadedSceneContent`)
 * ahead of the per-mesh walk — the walk's per-mesh releases then no-op,
 * and registrations whose mesh was never attached to the scene are
 * covered too.
 */
export function releaseAllDepthSortNodes(): void {
  nodeStates.clear();
  api?.releaseAllNodes().catch(() => {});
}

/**
 * Terminate the worker and reset all module state (app teardown; also
 * the test reset). Safe to call when never spawned.
 */
export function disposeDepthSort(): void {
  nodeStates.clear();
  // Module-state reset completeness: both per-frame containers can hold
  // THREE object references between calls (the rank memo until the next
  // evaluate's clear; the slots only if an evaluate threw mid-collect) —
  // an embedder that disposes and re-inits in one page must not have the
  // old scene pinned by them.
  clearRenderOrderFrameState();
  worker?.terminate();
  worker = null;
  api = null;
  initPromise = null;
  getCamera = null;
  requestRender = null;
  requestReprocess = null;
  isLoadInProgress = null;
  getProfiler = null;
  depthSortEnabled = true;
  warnedWorkerUnavailable = false;
}
