/**
 * The depth-sort coordinator's state and the small helpers every part of it
 * uses.
 *
 * The coordinator is split by concern — worker lifecycle
 * (`worker-lifecycle.ts`), ordering apply (`ordering-apply.ts`), the per-frame
 * scheduler (`scheduler.ts`), the offline-capture drain (`capture.ts`) and the
 * public `DepthSortCoordinator` class (`../depth-sort-coordinator.ts`) — over
 * TWO state objects with different lifetimes:
 *
 * - {@link CoordinatorState}: one per coordinator INSTANCE, i.e. per LuxarApp /
 *   LuxarLayer. Everything that is about a host — its tracked nodes, its camera,
 *   its frame wake-up, its capture suppression, the per-frame budget — lives
 *   here, so two hosts on one page sort their own nodes against their own
 *   camera and each wakes only its own render loop. Every module function that
 *   touches host state takes it as its first argument.
 * - {@link workerHost}: the ONE SortWorker the page shares, with its init
 *   bookkeeping (epoch, starved-retry budget, status). Node registrations are
 *   keyed by the render mesh's uuid, which is unique page-wide, and every sort
 *   RPC resolves through its own promise back into the coordinator that
 *   dispatched it, so sharing the worker needs no routing table. The host
 *   tracks which coordinators are attached so a starved-init retry can wake —
 *   and re-register the nodes of — every one of them, and so only the LAST
 *   coordinator's dispose terminates the worker.
 *
 * The docs on each field are the originals from the module-scoped coordinator.
 *
 * @module rendering/depth-sort-coordinator/state
 */

import type * as THREE from 'three';
import type { Remote } from 'comlink';
import type { SortWorkerAPI } from '../../workers/sort-worker';
import { releaseSortedIndexDrawHold, type SortedIndexApplyCallbacks } from '../element-storage';
import { needsDepthSort } from '../blending-state';
import { hasCommittedData, invalidateCommittedDataStamp } from '../../types/committed-data';
import type { BlendingMode } from '../../types/blending';
import type { UpdateProfiler } from '../../profiling/update-profiler';
import { config } from '../../config';
import { isEffectivelyVisible } from '../../utils/object-visibility';

export interface NodeSortState {
  /** The node's render mesh (the pick node shares its geometry). */
  mesh: THREE.Mesh;
  /** Lifetime-unique non-noop commit stamp (see nextGeneration). */
  generation: number;
  /**
   * Element count the CURRENT generation committed. A resolved ordering of
   * any other length is rejected: the worker clamps a registration whose
   * centers under-deliver and sorts the clamped count, and a short ordering
   * is neither a permutation of the population nor the length a held append
   * draw waits for.
   */
  count: number;
  /** Generation already warned about a short worker ordering. */
  warnedMismatchGeneration?: number;
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
  /**
   * MESH ONLY: the canonical (unpermuted, winding-corrected) index triples
   * of the current commit's visible faces, which a resolved ordering
   * permutes into `geometry.index`.
   *
   * The instanced types need no equivalent because their ordering is a
   * draw-slot indirection written from scratch each time — nothing composes.
   * A mesh's index buffer, by contrast, already holds the PREVIOUS
   * permutation, so permuting it again would compose the two and produce
   * garbage. Keeping the canonical list is what makes each apply
   * independent of the last.
   *
   * Set on every order-dependent mesh commit and cleared on every release
   * branch, so a mesh only doubles its index memory while it is actually
   * being sorted. The array is the projection's own output (already a fresh
   * copy per epoch — `ProjectedMeshData.indices`), so holding it costs a
   * reference rather than a copy.
   */
  triangleSource?: Uint32Array;
}

/**
 * The page's one SortWorker and its init bookkeeping (see the module doc for
 * why the worker is shared while everything else is per coordinator).
 */
export interface SortWorkerHost {
  worker: Worker | null;
  api: Remote<SortWorkerAPI> | null;
  initPromise: Promise<void> | null;
  /**
   * Attempt epoch: bumped by everything that INVALIDATES the init attempt
   * currently in flight (the host reset when the last coordinator is disposed,
   * and the retry when it starts a fresh attempt). `ensureWorker` captures it
   * and every host write past its first `await` is skipped when the captured
   * value no longer matches.
   *
   * This is the `worker === w` guard's precedent generalized, and it is not
   * optional: the init deadline (`config.depthSort.workerInitTimeoutMs`) lives
   * in a `setTimeout` inside the init closure and only `w.onerror` /
   * `initialize` settling clears it, so NOTHING outside the attempt can cancel
   * that timer. Without an epoch, app A's orphan timer firing one deadline
   * after A's dispose would classify a
   * deadline miss against app B's HEALTHY worker: `workerInitState` flips to
   * 'starved' with a retry armed, the next frame's retry nulls B's RESOLVED
   * `initPromise` and spawns a third worker over the live one (B's worker
   * leaked, its transferred centers buffers with it — tens of MB at 3M
   * points), and since every node still reads `registered === true` the
   * re-registration sweep skips them all, so the new worker holds no
   * registrations and `sortNode` returns null forever: depth sorting dead
   * for the session, i.e. the very bug #1694 is about, through a new door.
   * The SUCCESS path is symmetric — a stale attempt resolving after a
   * dispose would stamp 'ready' and clear the retry flags on behalf of a
   * worker that no longer exists.
   */
  initEpoch: number;
  /**
   * One-shot flag for the "commit could not reach the SortWorker" error (see
   * `DepthSortCoordinator.noteCommit`). Warn-once per EPISODE, not per
   * session: a successful (possibly retried) init clears it.
   *
   * What that re-arming does and does not buy: once init has succeeded the
   * cached `initPromise` is resolved forever, so no later commit can reach
   * that catch through a worker failure at all. The re-arm therefore only
   * matters for the catch's OTHER causes — a throwing lazy centers provider,
   * a `transfer()` of an already-detached buffer — and for a fresh
   * unavailability episode after the host was reset and the worker respawned.
   */
  warnedWorkerUnavailable: boolean;
  /**
   * Observable init state (see `getDepthSortWorkerStatus`). 'idle'
   * also covers "init in flight, never yet settled" — the terminal states
   * are what callers care about. 'starved' spans a retry ARMED **and one
   * already in flight** (it is only left when an attempt succeeds or the
   * budget runs out): sorting is off either way, so the two need no
   * distinction.
   */
  workerInitState: 'idle' | 'ready' | 'starved' | 'failed';
  /**
   * True while a RETRYABLE (deadline-miss) init failure is outstanding and
   * attempts remain. Cleared when a retry consumes it, when the attempts run
   * out, when a later failure turns out to be the permanent kind, and by the
   * host reset — but only `maybeRetryStarvedWorkerInit`
   * may clear the cached rejected `initPromise` on the strength of it (issue
   * #1694).
   */
  initTimeoutRetryPending: boolean;
  /** Deadline misses so far this session; bounds the total attempts. */
  initTimeoutCount: number;
  /** `performance.now()` before which the next retry must not be attempted. */
  initRetryNotBeforeMs: number;
  /**
   * Handle of the one-shot self-wake armed alongside a retry (see
   * `scheduleInitRetryWake`). At most one is pending at a time.
   */
  initRetryWakeTimer: ReturnType<typeof setTimeout> | null;
  /**
   * The coordinators sharing this worker. Attached by every path that may
   * need the worker (configure, warm-up, commit) and detached by dispose; the
   * last detach resets the host and terminates the worker.
   */
  coordinators: Set<CoordinatorState>;
}

/** The page's one {@link SortWorkerHost}. */
export const workerHost: SortWorkerHost = {
  worker: null,
  api: null,
  initPromise: null,
  initEpoch: 0,
  warnedWorkerUnavailable: false,
  workerInitState: 'idle',
  initTimeoutRetryPending: false,
  initTimeoutCount: 0,
  initRetryNotBeforeMs: 0,
  initRetryWakeTimer: null,
  coordinators: new Set(),
};

/** One coordinator instance's state (see the module doc). */
export interface CoordinatorState {
  /** Per-node sort state, keyed by the render mesh's uuid. */
  nodeStates: Map<string, NodeSortState>;
  getCamera: (() => THREE.Camera | null) | null;
  requestRender: (() => void) | null;
  requestReprocess: (() => void) | null;
  isLoadInProgress: (() => boolean) | null;
  getProfiler: (() => UpdateProfiler | null) | null;
  /** Live displayed-dimension indices, handed to each frame's render-order pass. */
  getDisplayDims: (() => readonly number[] | null) | null;
  /**
   * Master switch (Phase 3): config `depthSort.enabled` combined with the
   * `?depthSort=0` URL escape hatch at app init. When false this coordinator
   * is inert — its commits keep the identity (storage) ordering, it never
   * spawns the worker, and its per-frame scheduler no-ops.
   */
  depthSortEnabled: boolean;
  /**
   * Reentrancy guard for `resortForCapture`'s frame-request suppression.
   * The offline capture nulls `requestRender` so draining can't re-arm the rAF
   * loop it stopped; a depth counter ensures only the OUTERMOST capture call
   * snapshots and restores it, so an overlapping/nested call can never strand
   * `requestRender` at null. Non-zero also lifts the chunked apply's upload
   * back-pressure for THIS coordinator's streams (see `orderingApplyHooks`).
   */
  captureSuppressDepth: number;
  requestRenderBeforeCapture: (() => void) | null;
  // Shared across every commit between frame evaluations: the configured ceiling
  // bounds the whole main-thread batch, not each node independently.
  syncSortElementsRemaining: number;
  /**
   * Set when a per-frame pass changed drawn state (a slot uniform or a
   * renderOrder) — read and cleared by `evaluateDepthSortPerFrame`, whose
   * return value tells the render-on-change loop to redraw.
   */
  drawnStateChanged: boolean;
}

/** A fresh, unconfigured {@link CoordinatorState}. */
export function createCoordinatorState(): CoordinatorState {
  return {
    nodeStates: new Map(),
    getCamera: null,
    requestRender: null,
    requestReprocess: null,
    isLoadInProgress: null,
    getProfiler: null,
    getDisplayDims: null,
    depthSortEnabled: true,
    captureSuppressDepth: 0,
    requestRenderBeforeCapture: null,
    syncSortElementsRemaining: syncSortElementLimit(),
    drawnStateChanged: false,
  };
}

/**
 * The chunked-apply hooks every ordering THIS coordinator stages carries
 * (`SortedIndexApplyCallbacks` minus the profiler lifecycle): the owner tag the
 * per-coordinator cancel sweeps filter on, the frame wake-up a consumed slice
 * fires to resume an idle stream (issue #715 resume gap), and the
 * offline-capture back-pressure bypass.
 *
 * Both closures read the LIVE fields rather than capturing values:
 * `resortForCapture` nulls `requestRender` for its drain, and a slice
 * acknowledged mid-drain must not wake the loop it stopped.
 */
export function orderingApplyHooks(c: CoordinatorState): SortedIndexApplyCallbacks {
  return {
    owner: c,
    requestFrame: () => c.requestRender?.(),
    bypassBackPressure: () => c.captureSuppressDepth > 0,
  };
}

export function syncSortElementLimit(): number {
  return config?.depthSort?.syncSortMaxElements ?? 0;
}

/**
 * Force the next view sweep to RE-COMMIT a node that must (re-)deliver its
 * centers to the SortWorker, by clearing BOTH freshness stamps. Shared by
 * the two callers that need exactly this — the switch-to-sorted branch of
 * `noteDepthSortBlendingModeSwitch` and the post-retry
 * re-registration (`reregisterAfterLateWorkerInit`) — because the
 * two must not drift: either stamp left in place silently strands the node
 * unsorted.
 *
 * Stamp only — NOT `clearCommittedData`. The geometry stays on the GPU and
 * pickable throughout the async reprocess, so the picking `elementIdMap`
 * sibling still describes it exactly; dropping it would make every hover in
 * that window resolve labels through the raw storage slot (silently wrong
 * for a range-loaded or compacted labelled node).
 *
 * The per-slice freshness stamp goes TOO: the reprocess sweep re-commits
 * only sweep-registered (eager) loaders — a hidden resident LAZY LOD level
 * is structurally outside the sweep, and with only the noop stamp cleared
 * it stayed "ready + fresh" in the LOD registry, so nothing ever
 * re-committed it: on re-show it rendered the sorted mode UNSORTED until an
 * unrelated slice change. Marking it stale makes the registry's
 * settle-gated reload (`maybeKickReload`: ready-but-stale aspiration →
 * ensureLoaded) re-commit + register it. Eager nodes are unaffected (the
 * sweep re-commit re-stamps anyway).
 */
export function invalidateSortedNodeCommitStamps(mesh: THREE.Mesh): void {
  invalidateCommittedDataStamp(mesh);
  delete (mesh.userData as { loadedViewVersion?: number }).loadedViewVersion;
}

/**
 * True when something on screen actually WANTS sorting right now: a tracked
 * node that is effectively visible, still holds committed data, and whose LIVE
 * blending mode is order-dependent.
 *
 * One helper for the two consumers that must not drift — the retry gate below
 * (do not spend one of the few attempts on a scene that never sorts) and
 * `isDepthSortAvailable` (do not announce a degrade about a subsystem
 * this scene never uses).
 */
export function anyNodeWantsSorting(c: CoordinatorState): boolean {
  for (const state of c.nodeStates.values()) {
    const mesh = state.mesh;
    if (!isEffectivelyVisible(mesh)) continue;
    if (!hasCommittedData(mesh)) continue;
    if (!isLiveOrderDependent(liveBlendingMode(mesh))) continue;
    return true;
  }
  return false;
}

/** Read the live REQUESTED blending mode stamped by the material wrappers. */
export function liveBlendingMode(mesh: THREE.Mesh): BlendingMode | undefined {
  const material = mesh.material as THREE.Material | THREE.Material[];
  const single = Array.isArray(material) ? material[0] : material;
  return single?.userData?.blendingMode as BlendingMode | undefined;
}

/**
 * True when the mesh's LIVE mode is order-dependent as rendered.
 *
 * `userData.blendingMode` is the RESOLVED mode for every type, which is
 * what makes one predicate enough across four of them. The three emissive
 * types implement the real volumetric math (gsplats phase 1, points phase
 * 3, lines phase 4), so their requested mode IS the rendered mode. Mesh
 * has no volumetric (a triangle is a zero-thickness surface with no path
 * length to absorb over — §6.3) and the mesh material maps that request
 * onto `opaque` before stamping, so the sorted set for mesh is exactly
 * `normal`.
 */
export function isLiveOrderDependent(mode: BlendingMode | undefined): boolean {
  if (!mode) return false;
  return needsDepthSort(mode);
}

/**
 * End a held append draw (`holdSortedIndexDrawForAppend`) WITHOUT a sorted
 * ordering for the grown population: every path on which no such ordering
 * will arrive calls this, so a held draw can never outlive the sort it waits
 * for. A no-op when nothing is held (the common case, and every non-gsplat
 * geometry).
 *
 * A release changes what the next frame draws, and most of its callers run in
 * a promise callback long after the commit's own render request was consumed,
 * so it wakes a frame itself: under render-on-change the held draw would
 * otherwise stay on screen until something else moved.
 */
export function releaseHeldDraw(c: CoordinatorState, mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (!geometry || !releaseSortedIndexDrawHold(geometry)) return;
  c.drawnStateChanged = true;
  c.requestRender?.();
}

/**
 * Fire-and-forget worker-side release. Swallows rejections: releases
 * run during teardown flows where the worker may already be
 * terminating, and a never-settling/rejected cleanup RPC is expected
 * there, not actionable.
 */
export function releaseWorkerNode(nodeId: string): void {
  workerHost.api?.releaseNode(nodeId).catch(() => {});
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
export function clearSortPose(state: NodeSortState): void {
  state.lastSortAxis = null;
  state.lastSortOffset = 0;
  state.registered = false;
  state.resortQueued = false;
}
