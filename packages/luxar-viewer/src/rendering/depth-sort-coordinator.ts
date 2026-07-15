/**
 * Depth-sort coordinator — main-thread side of the SortWorker
 * (depth-sorting Phase 2, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5).
 *
 * Module-scoped live authority (the `splat-texture-layout.ts` pattern):
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
import { writeSortedIndexOrdering } from './gsplat-geometry';
import { isNormalMode } from './blending-state';
import type { BlendingMode } from './material-manager';
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
  /** Per-node monotonic non-noop commit counter (the generation contract). */
  generation: number;
  /** True while a sort RPC is outstanding for this node. */
  inFlight: boolean;
  /** A newer commit landed mid-sort — re-sort once the current one resolves. */
  resortQueued: boolean;
}

let worker: Worker | null = null;
let api: Remote<SortWorkerAPI> | null = null;
let initPromise: Promise<void> | null = null;
let camera: THREE.Camera | null = null;
let requestRender: (() => void) | null = null;
let requestReprocess: (() => void) | null = null;
/** One-shot flag for the SortWorker-unavailable error (see noteGSplatsCommit). */
let warnedWorkerUnavailable = false;
const nodeStates = new Map<string, NodeSortState>();

/**
 * Wire the camera + frame-request + reprocess callbacks. Called once at
 * app init (the commit path has none of these — the scene loader
 * deliberately owns no camera state). Safe to call again on renderer swap.
 */
export function configureDepthSort(options: {
  camera: THREE.Camera;
  requestRender: () => void;
  /**
   * Force a full view reprocess (`SceneLoader.updateView({})`) — used by
   * the blending-mode-switch hook, see {@link noteGSplatsBlendingModeSwitch}.
   */
  requestReprocess?: () => void;
}): void {
  camera = options.camera;
  requestRender = options.requestRender;
  requestReprocess = options.requestReprocess ?? null;
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
export function noteGSplatsCommit(mesh: THREE.Mesh, centers3: Float32Array, count: number): void {
  const nodeId = mesh.uuid;
  let state = nodeStates.get(nodeId);
  if (!state) {
    state = { generation: 0, inFlight: false, resortQueued: false };
    nodeStates.set(nodeId, state);
  }
  state.generation++;

  const mode = liveBlendingMode(mesh);
  if (!mode || !isNormalMode(mode) || count === 0) {
    // Commutative blending (or an empty frame): no ordering needed. Drop
    // any worker-side registration so the worker doesn't hold stale
    // centers for a node that may not sort again for a long time.
    releaseWorkerNode(nodeId);
    return;
  }

  const generation = state.generation;
  void ensureWorker()
    .then(() => {
      if (!api) return;
      // A newer commit may have landed while the worker was spawning.
      if (nodeStates.get(nodeId)?.generation !== generation) return;
      // The register RPC is its own promise — the surrounding .catch
      // only sees synchronous throws, so a transport/transfer rejection
      // here would otherwise float as an unhandled rejection.
      api
        .registerNode(transfer({ nodeId, generation, centers3, count }, [centers3.buffer]))
        .catch((error: unknown) => {
          log.error(Modules.WORKER_POOL, `SortWorker registerNode failed for ${nodeId}`, error);
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
 * Request one sort for a registered node, respecting the
 * single-in-flight rule. Queues a re-sort if one is already running.
 */
function scheduleSort(mesh: THREE.Mesh, nodeId: string): void {
  const state = nodeStates.get(nodeId);
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

  void api
    .sort({ nodeId, generation, modelView: new Float32Array(modelView.elements) })
    .then((result) => {
      const current = nodeStates.get(nodeId);
      if (!current) return; // released mid-sort
      current.inFlight = false;

      // Stale-drop: apply only when the ordering matches the node's
      // CURRENT generation (the worker re-checked its own registration;
      // this re-check covers commits that raced the RPC). The
      // `committedData` check additionally covers LOD demotion: a demoted
      // level's geometry returned to the evictable pool (and may since
      // belong to another node) — the cleared stamp is exactly the signal
      // that the mesh's geometry no longer holds this commit's splats.
      const stillCommitted =
        (mesh.userData as { committedData?: unknown })?.committedData !== undefined;
      if (result && result.generation === current.generation && stillCommitted) {
        const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
        if (geometry?.getAttribute?.('aSortedIndex')) {
          writeSortedIndexOrdering(geometry, result.ordering, result.ordering.length);
          requestRender?.();
        }
      }

      if (current.resortQueued) {
        current.resortQueued = false;
        scheduleSort(mesh, nodeId);
      }
    })
    .catch((error) => {
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
  if (!newMode || newMode === prevMode) return;
  const wasNormal = prevMode !== undefined && isNormalMode(prevMode);
  if (isNormalMode(newMode) && !wasNormal) {
    delete (mesh.userData as { committedData?: unknown }).committedData;
    requestReprocess?.();
  } else if (!isNormalMode(newMode) && wasNormal) {
    const state = nodeStates.get(mesh.uuid);
    if (state) {
      // Invalidate any in-flight sort's result; keep the counter
      // monotonic for the node's next order-dependent commit.
      state.generation++;
      state.resortQueued = false;
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

/** Drop every node (dataset switch). */
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
  worker?.terminate();
  worker = null;
  api = null;
  initPromise = null;
  camera = null;
  requestRender = null;
  requestReprocess = null;
  warnedWorkerUnavailable = false;
}
