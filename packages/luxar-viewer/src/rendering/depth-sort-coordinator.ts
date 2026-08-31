/**
 * Depth-sort coordinator — main-thread side of the SortWorker
 * (depth-sorting Phases 2-3,
 * `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5-§6).
 *
 * Serves all four geometry types. The mechanism is geometry-agnostic
 * (projected 3D centers in — segment midpoints for lines, face centroids
 * for mesh — back-to-front permutation out); the commit call sites and
 * the APPLY differ.
 *
 * Two apply paths, because there are two ways to draw a permutation:
 * - **Instanced** (gsplats, points, lines): rewrite the `aSortedIndex`
 *   draw-slot indirection, double-buffered and streamed in slices —
 *   `rendering/element-storage.ts`.
 * - **Indexed** (mesh): rewrite `geometry.index` itself, atomically —
 *   `depth-sort-coordinator/triangle-ordering.ts`, which explains why the
 *   double-buffered streaming trick does not transfer.
 *
 * Module-scoped live authority (the `element-texture-layout.ts` pattern):
 * the commit paths (`commit-gsplats-geometry.ts`,
 * `commit-points-geometry.ts`) and the app lifecycle are far apart, so
 * all talk to this module instead of threading a coordinator object
 * through constructors.
 *
 * Responsibilities:
 * - Owns the single persistent Comlink SortWorker (spawn at app init via
 *   {@link warmUpDepthSortWorker}, terminate on app teardown). NOT part of
 *   the round-robin data-worker pool — node registrations and their
 *   transferred center buffers must live in exactly one worker.
 * - Classifies an init failure, because the two causes need opposite
 *   answers (issue #1694). The init promise always SETTLES (the anti-leak
 *   contract — the deadline is `config.depthSort.workerInitTimeoutMs`), but
 *   a DEAD worker script fails permanently while a STARVED one — init
 *   losing the race against the deadline because the main thread was busy
 *   committing millions of elements — is retried a bounded number of times
 *   from the per-frame scheduler. {@link getDepthSortWorkerStatus} makes the
 *   resulting degrade state observable instead of console-only.
 * - Tracks the per-node **generation** counter: bumped on every non-noop
 *   commit (`noteDepthSortCommit`), never on stamp-only noops. An ordering
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
 *   sorted-mode mesh (any registered geometry type) and assigns ONE
 *   global back-to-front `renderOrder` scale (wrapper groups by mean
 *   view-z, exact BSP ranks within a wrapper) — machinery owned by the
 *   `depth-sort-coordinator/render-order.ts` submodule, driven here via
 *   {@link clearRenderOrderFrameState} / {@link collectRenderOrderSlot} /
 *   {@link assignGlobalRenderOrder}.
 *
 * Ordering only matters for order-dependent blending; all other modes
 * are commutative. Order-dependence is judged on `needsDepthSort(mode)`
 * uniformly for all three geometry types (gsplats phase 1, points
 * phase 3, lines phase 4 — each upgrade landed with zero coordinator
 * change, the designed chokepoint). Commits of order-independent nodes
 * still bump the generation (killing any in-flight sort) and release
 * the node's worker-side registration.
 */

import * as THREE from 'three';
import { wrap, transfer, type Remote } from 'comlink';
// Vite's `?worker` import emits a bundled worker chunk (see
// worker-pool.ts for why `new Worker(new URL(...))` is not used).
import SortWorker from '../workers/sort-worker?worker';
import type { SortWorkerAPI } from '../workers/sort-worker';
import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  cancelAllSortedIndexOrderingApplies,
  cancelSortedIndexOrderingApply,
  getActiveSortedIndexAttribute,
  hasPendingSortedIndexOrderingApply,
  pumpSortedIndexOrderingApply,
  setSortedIndexApplyBackPressureBypassed,
  setSortedIndexApplyRequestRender,
  writeSortedIndexOrdering,
  writeSortedIndexOrderingLive,
} from './element-storage';
// The TypeScript reference kernel, not the WASM one: the WASM module is
// instantiated inside the SortWorker, and the whole point of the synchronous
// path is to answer without touching the worker. Exact-parity with the Rust
// kernel (see its module doc), so the two agree on the permutation.
import { sort_splats_by_depth } from '../wasm/typescript/depth-sort';
import { needsDepthSort } from './blending-state';
import { hasCommittedData, invalidateCommittedDataStamp } from '../types/committed-data';
import type { BlendingMode } from '../types/blending';
import {
  assignGlobalRenderOrder,
  clearRenderOrderFrameState,
  collectRenderOrderSlot,
  orderGroupOf,
  setRenderOrderDisplayDimsAccessor,
  type ShardOrderInput,
} from './depth-sort-coordinator/render-order';
import {
  acknowledgeTriangleOrderingDraw,
  cancelAllTriangleOrderingApplies,
  cancelTriangleOrderingApply,
  writeSortedTriangleOrdering,
} from './depth-sort-coordinator/triangle-ordering';
import {
  depthShardChildren,
  effectiveInstanceCount,
  releaseDepthShards,
  syncDepthShards,
  syncShardMaterials,
} from './depth-sort-coordinator/depth-shards';
import {
  assignShardCounts,
  interleavedDrawCount,
  type ShardPolicyNode,
} from './depth-sort-coordinator/shard-policy';
import { config } from '../config';
import type { UpdateProfiler } from '../profiling/update-profiler';
import { withTimeout } from '../workers/worker-pool/timeout/with-timeout';
import { initializeWithGuard } from '../workers/worker-pool/lifecycle/init-with-guard';
import { WorkerInitTimeoutError } from '../workers/worker-pool/errors';
import { log, Modules } from '../utils/log';
// Ancestor-aware visibility (the single parent-chain walk, shared with the pick
// pass and the LOD registry/eviction pass). `mesh.visible` alone misses a hidden
// ancestor: an LOD level can be a GROUP (partition tiles) and the registry
// toggles `child.object.visible` on the group — the member meshes' own flags
// stay true; a hidden LAYER is likewise an ancestor flag. Sorting a hidden mesh
// is wasted worker time; when it re-shows, the next frame's pose comparison
// catches any past-threshold camera motion immediately.
import { isEffectivelyVisible } from '../utils/object-visibility';

/**
 * Optional override for the sort-worker module URL (embedders whose
 * bundler lacks `?worker` support). Mirrors {@link setDataWorkerUrl};
 * set via `LuxarAppOptions.workerPath` alongside the data worker's.
 */
let sortWorkerUrlOverride: string | undefined;

/**
 * Override the URL used to construct the sort worker.
 *
 * An INTERNAL entry point, reachable only from inside the source tree (a
 * vendored/bundled viewer, the standalone bootstrap): it is not re-exported
 * from `src/index.ts`, and `package.json`'s `exports` map publishes only `.`
 * and `./styles.css`, so an npm consumer cannot import it at all.
 *
 * Call before `LuxarApp.init()`. That window used to run to the first
 * order-dependent commit, but the worker is now warmed up during app init
 * ({@link warmUpDepthSortWorker}), so a later call would arrive after the
 * spawn it is meant to redirect. `LuxarAppOptions.workerPath` deliberately
 * does NOT reach here (it names the DATA worker bundle, a different chunk —
 * see `applyModuleOverrides`), so relocating the sort worker means calling
 * this setter yourself, before `init()`.
 */
export function setSortWorkerUrl(url: string): void {
  sortWorkerUrlOverride = url;
}

/**
 * WASM JS-shim URL forwarded into the worker's `initialize()` (the
 * main-thread `setWasmJsUrl` override does not cross the worker
 * boundary). Mirrors {@link setDataWorkerWasmPath}.
 */
let sortWorkerWasmPathOverride: string | undefined;

/**
 * Override the WASM JS-shim URL used inside the sort worker.
 *
 * Same timing contract as {@link setSortWorkerUrl}: before
 * `LuxarApp.init()`.
 */
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
  /**
   * Number of shards the LAST RESOLVED sort reported usable depth bounds for
   * (cross-node depth ordering — `CROSS_NODE_DEPTH_ORDERING_SPEC.md` §3.3).
   *
   * `0` means "merge this node as one whole-node interval": either nothing was
   * requested, or the kernel took its identity-ordering fallback and the ranges
   * are storage ranges rather than depth intervals.
   */
  shardCount: number;
  /**
   * Per-shard LOCAL-space AABB of the elements each shard draws,
   * `[shardCount * 3]` each; undefined while `shardCount === 0`.
   *
   * Local space, not view space, on purpose: the render-order pass re-projects
   * these every frame, so cross-node shard ordering tracks camera motion
   * BETWEEN re-sorts instead of going stale under the dispatch hysteresis.
   * A shard with no finite element on an axis carries `min = +Infinity` /
   * `max = -Infinity` there, so `min > max` is the "no usable bounds" test.
   */
  shardBoundsMin?: Float32Array;
  shardBoundsMax?: Float32Array;
}

let worker: Worker | null = null;
let api: Remote<SortWorkerAPI> | null = null;
let initPromise: Promise<void> | null = null;
/**
 * Attempt epoch: bumped by everything that INVALIDATES the init attempt
 * currently in flight ({@link disposeDepthSort}, and the retry when it
 * starts a fresh attempt). `ensureWorker` captures it and every module
 * write past its first `await` is skipped when the captured value no longer
 * matches.
 *
 * This is the `worker === w` guard's precedent generalized, and it is not
 * optional: the init deadline (`config.depthSort.workerInitTimeoutMs`) lives
 * in a `setTimeout` inside the init closure and only `w.onerror` /
 * `initialize` settling clears it, so NOTHING outside the attempt can cancel
 * that timer. Without an epoch, app A's orphan timer firing one deadline
 * after `disposeDepthSort()` would classify a
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
let initEpoch = 0;
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
/**
 * One-shot flag for the "commit could not reach the SortWorker" error (see
 * noteDepthSortCommit). Warn-once per EPISODE, not per session: a
 * successful (possibly retried) init clears it.
 *
 * What that re-arming does and does not buy: once init has succeeded the
 * cached `initPromise` is resolved forever, so no later commit can reach
 * that catch through a worker failure at all. The re-arm therefore only
 * matters for the catch's OTHER causes — a throwing lazy centers provider,
 * a `transfer()` of an already-detached buffer — and for a fresh
 * unavailability episode after a dispose/re-init.
 */
let warnedWorkerUnavailable = false;
/**
 * Observable init state (see {@link getDepthSortWorkerStatus}). 'idle'
 * also covers "init in flight, never yet settled" — the terminal states
 * are what callers care about. 'starved' spans a retry ARMED **and one
 * already in flight** (it is only left when an attempt succeeds or the
 * budget runs out): sorting is off either way, so the two need no
 * distinction.
 */
let workerInitState: 'idle' | 'ready' | 'starved' | 'failed' = 'idle';
/**
 * True while a RETRYABLE (deadline-miss) init failure is outstanding and
 * attempts remain. Cleared when a retry consumes it, when the attempts run
 * out, when a later failure turns out to be the permanent kind, and by
 * {@link disposeDepthSort} — but only {@link maybeRetryStarvedWorkerInit}
 * may clear the cached rejected `initPromise` on the strength of it (issue
 * #1694).
 */
let initTimeoutRetryPending = false;
/** Deadline misses so far this session; bounds the total attempts. */
let initTimeoutCount = 0;
/** `performance.now()` before which the next retry must not be attempted. */
let initRetryNotBeforeMs = 0;
/**
 * Handle of the one-shot self-wake armed alongside a retry (see
 * {@link scheduleInitRetryWake}). At most one is pending at a time.
 */
let initRetryWakeTimer: ReturnType<typeof setTimeout> | null = null;
/**
 * Reentrancy guard for {@link resortForCapture}'s frame-request suppression.
 * The offline capture nulls `requestRender` so draining can't re-arm the rAF
 * loop it stopped; a depth counter ensures only the OUTERMOST capture call
 * snapshots and restores it, so an overlapping/nested call can never strand
 * `requestRender` at null.
 */
let captureSuppressDepth = 0;
let requestRenderBeforeCapture: (() => void) | null = null;
const nodeStates = new Map<string, NodeSortState>();
// Shared across every commit between frame evaluations: the configured ceiling
// bounds the whole main-thread batch, not each node independently.
let syncSortElementsRemaining = 0;

function syncSortElementLimit(): number {
  return config?.depthSort?.syncSortMaxElements ?? 0;
}

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
   * the blending-mode-switch hook, see {@link noteDepthSortBlendingModeSwitch}.
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
  /**
   * Live displayed-dimension indices. A partition's serialized BSP `axis` is a
   * CENTER-COLUMN index, while the painter's-order traversal works in display
   * space (x/y/z = displayDims[0..2]); the two coincide only for `[0, 1, 2]`.
   * Injected rather than read from `sceneDimsManager` directly because
   * `rendering/` must not depend on `scene/` (`layer-rendering-no-upward`) —
   * the same inversion as `getCamera` above.
   */
  getDisplayDims?: () => readonly number[] | null;
}): void {
  getCamera = options.getCamera;
  requestRender = options.requestRender;
  requestReprocess = options.requestReprocess ?? null;
  isLoadInProgress = options.isLoadInProgress ?? null;
  getProfiler = options.getProfiler ?? null;
  setRenderOrderDisplayDimsAccessor(options.getDisplayDims ?? null);
  // A consumed slice's upload acknowledgement resumes an idle chunked
  // stream the instant the mesh is drawn again (issue #715 resume gap —
  // the pump never requests a render on a stall).
  setSortedIndexApplyRequestRender(options.requestRender);
  syncSortElementsRemaining = syncSortElementLimit();
}

/**
 * Session master switch, applied at app init from `config.depthSort.enabled`
 * combined with the `?depthSort=0` URL escape hatch. Disabling pins the
 * identity (storage) ordering for deterministic E2E/visual runs.
 */
export function setDepthSortEnabled(enabled: boolean): void {
  depthSortEnabled = enabled;
}

/**
 * Total init attempts for a STARVED (deadline-missing) worker, including
 * the first — so 3 means the initial attempt plus 2 retries. Bounded on
 * purpose: each attempt opens one more deadline-long window during which
 * commits attach continuations to a pending promise, and an unbounded retry
 * loop would restore exactly the leak the deadline was added to prevent.
 * Three attempts span at worst three init deadlines plus the 2 s and 4 s
 * backoffs between them (~96 s of wall clock at the default 30 s deadline),
 * which comfortably outlives a load-induced main-thread stall while staying
 * a fixed, small ceiling.
 */
const SORT_WORKER_INIT_MAX_ATTEMPTS = 3;

/**
 * Backoff before a starved retry, multiplied by the number of deadline
 * misses so far (2 s, then 4 s). Measured from the failure, on the
 * monotonic `performance.now()` clock — a wall-clock jump must not skip or
 * freeze a backoff.
 */
const SORT_WORKER_INIT_RETRY_BASE_MS = 2_000;

/**
 * Slack added to the self-wake timer on top of the backoff (see
 * {@link scheduleInitRetryWake}). The retry is gated on
 * `performance.now() >= initRetryNotBeforeMs`, so a wake scheduled for
 * EXACTLY that instant is a coin flip between the timer's own rounding and
 * the clock read — a lost toss costs a whole extra idle window. Small
 * enough to be invisible next to a 2 s backoff.
 */
const SORT_WORKER_INIT_RETRY_WAKE_SLACK_MS = 50;

/**
 * Per-sort RPC deadline (see scheduleSort). Generous — a sort is
 * O(N + buckets) over at most a few million centers, milliseconds on any
 * live worker; the deadline only trips on a crashed/wedged worker thread.
 */
const SORT_RPC_TIMEOUT_MS = 30_000;

/**
 * Shared zero-length stand-in for the kernel's shard-bounds outputs on the
 * main-thread sort path, which never requests them (`shardCount = 0`).
 */
const EMPTY_SHARD_BOUNDS = new Float32Array(0);

/**
 * How many depth shards this node's next sort should report bounds for.
 *
 * The single place the shard-count policy will be read from (spec §4). Until it
 * lands this is always 0 — no node is sharded, the kernel skips the bounds pass,
 * and the whole feature is inert. Reads a stamp rather than owning state so the
 * policy can live where the overlap information is.
 */
function requestedShardCount(mesh: THREE.Mesh): number {
  const requested = (mesh.userData as { depthShardCount?: number }).depthShardCount;
  return typeof requested === 'number' && requested > 1 ? Math.trunc(requested) : 0;
}

/**
 * Live depth-shard policy overrides (`?depthShards=N`), injected at app init.
 * `enabled` false means no node is ever split, so draw order is exactly the
 * group-major assignment — the determinism escape hatch.
 */
let shardPolicy: { enabled: boolean; pinnedShardsPerNode?: number } = { enabled: false };

/** Interleaved draws the last policy evaluation established (monitor + gates). */
let shardInterleavedDraws = 0;

/**
 * Configure cross-node depth ordering. Called once at app init from the URL /
 * options / config resolution; see `CROSS_NODE_DEPTH_ORDERING_SPEC.md` §4.
 */
export function setDepthShardPolicy(policy: {
  enabled: boolean;
  pinnedShardsPerNode?: number;
}): void {
  shardPolicy = policy;
  if (!policy.enabled) {
    // Tear every split down immediately rather than waiting for the next commit:
    // this is the escape hatch, and it has to pin the output NOW.
    for (const state of nodeStates.values()) {
      delete (state.mesh.userData as { depthShardCount?: number }).depthShardCount;
      releaseDepthShards(state.mesh);
      clearShardBounds(state);
    }
    shardInterleavedDraws = 0;
  }
}

/** Interleaved draw count from the live shard assignment (0 when disabled). */
export function getDepthShardDrawCount(): number {
  return shardInterleavedDraws;
}

/**
 * Re-evaluate which tracked nodes should be split, and apply any change.
 *
 * Runs per frame but is nearly free and nearly always a no-op: the policy is a
 * pure function of world-space BOUNDS, so its answer changes only when a node
 * commits or the tracked set changes — never as the camera moves. Applying a
 * change rebuilds that node's shard meshes and dispatches a sort, because the
 * new shard count is what the next sort must report bounds for; until that sort
 * resolves the node merges as one whole-node interval, which is simply today's
 * behaviour.
 */
function evaluateShardPolicy(): void {
  if (!shardPolicy.enabled) return;

  const nodes: ShardPolicyNode[] = [];
  for (const state of nodeStates.values()) {
    const mesh = state.mesh;
    if (!isEffectivelyVisible(mesh) || !hasCommittedData(mesh)) continue;
    if (!isLiveOrderDependent(liveBlendingMode(mesh))) continue;
    const bs = (mesh.geometry as THREE.BufferGeometry | undefined)?.boundingSphere;
    const center = new THREE.Vector3();
    let radius = -1;
    if (bs && Number.isFinite(bs.radius) && bs.radius >= 0) {
      mesh.updateWorldMatrix(true, false);
      center.copy(bs.center).applyMatrix4(mesh.matrixWorld);
      const scaled = bs.radius * mesh.matrixWorld.getMaxScaleOnAxis();
      if (Number.isFinite(center.x + center.y + center.z) && Number.isFinite(scaled)) {
        radius = scaled;
      }
    }
    nodes.push({
      mesh,
      center,
      radius,
      group: orderGroupOf(mesh),
      elements: effectiveInstanceCount(mesh),
    });
  }

  const counts = assignShardCounts(nodes, {
    enabled: true,
    shardsPerNode: config.depthShards.shardsPerNode,
    maxInterleavedDraws: config.depthShards.maxInterleavedDraws,
    minElements: config.depthShards.minElements,
    pinnedShardsPerNode: shardPolicy.pinnedShardsPerNode,
  });
  shardInterleavedDraws = interleavedDrawCount(counts);

  for (const [mesh, count] of counts) {
    const userData = mesh.userData as { depthShardCount?: number };
    if ((userData.depthShardCount ?? 1) === count) continue;
    userData.depthShardCount = count;
    const state = nodeStates.get(mesh.uuid);
    if (!state) continue;
    const established = syncDepthShards(mesh, count, effectiveInstanceCount(mesh));
    // The bounds on hand describe the OLD shard boundaries, so they must go
    // before the next frame reads them — the count guard in `shardOrderInputFor`
    // would reject them anyway, but leaving them would rely on that.
    clearShardBounds(state);
    if (established > 1 || count === 1) scheduleSort(mesh, mesh.uuid);
  }
}

/**
 * The node's depth shards in the shape the render-order pass consumes, or
 * `undefined` when it should be ordered as ONE whole-node interval.
 *
 * Three conditions must all hold, and each one failing means the same thing —
 * the ranges carry no usable depth meaning, so fall back to today's behaviour:
 * the last resolved sort reported bounds (`shardCount > 1`; it reports 0 for the
 * kernel's identity-ordering fallback), the bound arrays are present, and the
 * shard MESHES still exist. The last is not redundant: a release or a re-commit
 * tears the meshes down independently of the bounds, and ordering by a box whose
 * mesh is gone would silently skip elements the parent is now drawing itself.
 */
function shardOrderInputFor(state: NodeSortState): ShardOrderInput | undefined {
  if (state.shardCount <= 1 || !state.shardBoundsMin || !state.shardBoundsMax) return undefined;
  const meshes = depthShardChildren(state.mesh);
  if (meshes.length + 1 !== state.shardCount) return undefined;
  return {
    meshes,
    count: state.shardCount,
    boundsMin: state.shardBoundsMin,
    boundsMax: state.shardBoundsMax,
  };
}

/**
 * Spawn + initialize the persistent sort worker.
 *
 * Init settle guard: a worker whose script dies during ASYNC module
 * evaluation (before `expose()` runs) emits an `error` event but never
 * settles the Comlink `initialize` RPC — and every order-dependent
 * commit attaches a continuation (closing over its centers provider,
 * which for points pins the full `LoadedPointsData`) to the cached
 * `initPromise`. Left pending forever, those closures accumulate one
 * per commit, unbounded. The shared `initializeWithGuard` races the RPC
 * against the `config.depthSort.workerInitTimeoutMs` deadline AND the
 * worker's own `error` / `messageerror` events, so the promise SETTLES on
 * every attempt, draining all queued continuations into the warn-once
 * degrade path. (It is shared with the data pool precisely so the two
 * startup paths cannot drift; the coordinator's hand-rolled copy never
 * grew the `messageerror` arm.)
 *
 * The settle contract is unconditional, but the DEGRADE is not (issue
 * #1694): only the deadline miss is retryable.
 * - A dead script (`error`/`messageerror`), a rejected `initialize`, or a
 *   constructor throw is PERMANENT: nothing about waiting longer would
 *   help, so the cached rejection stands for the session.
 * - Missing the deadline means only that init did not finish in time. At
 *   ~3M points the main thread is saturated long enough during a load for
 *   worker startup to lose that race, and the old unconditional
 *   stays-failed degrade then disabled depth sorting for the whole
 *   session over a condition that would have cleared in seconds. That
 *   case is retried, BOUNDED (see {@link SORT_WORKER_INIT_MAX_ATTEMPTS}),
 *   so the unbounded closure pile-up the deadline exists to stop cannot
 *   come back: between attempts the cached rejection is still what every
 *   commit sees, and only a finite number of fresh pending promises can
 *   ever exist.
 */
function ensureWorker(): Promise<void> {
  if (initPromise) return initPromise;
  // Snapshot for the epoch guard on every write past the awaits below — see
  // {@link initEpoch} for the hazard this closes.
  const epoch = initEpoch;
  initPromise = (async () => {
    let w: Worker;
    try {
      w = sortWorkerUrlOverride
        ? new Worker(sortWorkerUrlOverride, { type: 'module' })
        : new SortWorker();
    } catch (error) {
      // A constructor throw (CSP-blocked script, an embedder bundler with
      // no `?worker` support) is permanently fatal — classify it here so it
      // can never be mistaken for a starved deadline and retried.
      noteWorkerInitFailure(error);
      throw error;
    }
    worker = w;
    api = wrap<SortWorkerAPI>(w);
    try {
      const result = await initializeWithGuard(
        w,
        api!,
        'SortWorker',
        config.depthSort.workerInitTimeoutMs,
        // The guard's own handlers are scoped to the init race; this runs on
        // settle, and the coordinator has no permanent ones to restore.
        () => {
          w.onerror = null;
          w.onmessageerror = null;
        },
        sortWorkerWasmPathOverride
      );
      if (epoch !== initEpoch) {
        // Stale attempt: whatever started it is gone (a dispose, or a retry
        // that superseded it) while this init was awaited. Terminate the
        // worker so neither the thread nor its transferred centers buffers
        // leak, and write NOTHING module-scoped — the 'ready' stamp, the
        // warn-once re-arm, the retry clear and even the log line all belong
        // to whichever attempt is current NOW. `worker === w` is the same
        // identity check as the catch below: false whenever a live attempt
        // has already re-homed the fields, which is what keeps the live
        // worker/api pair intact. (`terminate()` is idempotent, so racing a
        // dispose that already terminated this worker is harmless.)
        w.terminate();
        if (worker === w) {
          worker = null;
          api = null;
        }
        return;
      }
      workerInitState = 'ready';
      // The unavailability episode is over: re-arm the warn-once so a
      // LATER genuine failure is reported rather than swallowed by the flag
      // an earlier starved attempt set. Also drop any armed retry — this
      // worker is live, there is nothing left to retry.
      warnedWorkerUnavailable = false;
      initTimeoutRetryPending = false;
      log.info(
        Modules.WORKER_POOL,
        `SortWorker ready (${result.wasmFallback ? 'TypeScript fallback' : 'WASM'})`
      );
    } catch (error) {
      // Terminate the wedged/failed worker so it can't hold resources.
      // `initPromise` stays rejected either way — every later commit keeps
      // landing in the warn-once catch, which is what BOUNDS the closure
      // accumulation the deadline exists to prevent. Whether that rejection
      // is the end of the story is noteWorkerInitFailure's call; only
      // maybeRetryStarvedWorkerInit ever clears it.
      w.terminate();
      if (worker === w) {
        worker = null;
        api = null;
      }
      // Epoch guard (see {@link initEpoch}): a STALE attempt's failure —
      // classically its orphaned init deadline firing long after a dispose —
      // must not classify, log, or arm anything against the attempt that is
      // current now. The worker above is still terminated; only the
      // bookkeeping is skipped.
      if (epoch === initEpoch) noteWorkerInitFailure(error);
      throw error;
    }
  })();
  initPromise.catch(() => {
    // Classified + logged by noteWorkerInitFailure above; this handler
    // exists only so the CACHED rejection is never an unhandled one (the
    // cache is deliberately kept — see the catch block).
  });
  return initPromise;
}

/** Cancel a pending self-wake, if any (idempotent). */
function clearInitRetryWake(): void {
  if (initRetryWakeTimer !== null) {
    clearTimeout(initRetryWakeTimer);
    initRetryWakeTimer = null;
  }
}

/**
 * Arm exactly ONE wake-up so an armed retry's backoff expiry is guaranteed
 * to be observed.
 *
 * {@link maybeRetryStarvedWorkerInit} runs only from the per-frame
 * scheduler, which `core/app/init/pipeline.ts` registers as a
 * NON-continuous per-frame callback — so the on-demand render loop
 * `stopAnimation()`s `config.animation.idleTimeoutMs` (default 2000 ms,
 * user-settable down to 500 ms) after the last `requestRender`, taking the
 * scheduler with it. Nothing else asks for a frame once a load finishes, and
 * the first backoff is 2000 ms: on a static "load it and look at it" scene
 * the loop can pause at or before the first eligible retry instant, so the
 * recovery would simply never fire (and the second attempt's 4 s backoff
 * would be unreachable without user interaction). `requestRender` is what
 * re-arms the loop, hence the per-frame callback, hence the retry.
 *
 * Bounded by construction: one pending wake at most, replaced on each
 * arming, cancelled when an attempt starts, when the retry is abandoned, and
 * by {@link disposeDepthSort} — so it can never outlive its arming nor wake
 * a disposed app. A wake that cannot be DELIVERED re-arms itself instead of
 * being spent (see below).
 */
function scheduleInitRetryWake(delayMs: number): void {
  clearInitRetryWake();
  initRetryWakeTimer = setTimeout(() => {
    initRetryWakeTimer = null;
    // Read the CURRENT `requestRender` (a dispose nulls it), never a
    // captured one — an old app's closure must not be resurrected here.
    if (requestRender) {
      requestRender();
      return;
    }
    // There is nobody to wake: `requestRender` is null before
    // `configureDepthSort` has run, and {@link resortForCapture} nulls it
    // DELIBERATELY for the duration of an offline capture. Firing into the
    // void would consume the one wake while the retry stays armed — exactly
    // the never-recovers hole this wake exists to close — so re-arm the same
    // delay instead. Bounded: an undeliverable wake is a no-op tick (one
    // timer, still at most one pending), and delivery resumes as soon as the
    // capture's `finally` restores `requestRender`; a dispose clears
    // `initTimeoutRetryPending`, which stops the chain for good.
    //
    // The LOAD case needs no such re-arm, and deliberately gets none: there
    // `requestRender` IS wired, so the wake is delivered and the frame simply
    // declines to spend an attempt while `isLoadInProgress`. That sweep's own
    // commits each call the render wake-up the app installed via
    // `SceneLoaderManager.setRequestRender` (`core/app/init/pipeline.ts`;
    // `SceneLoader` fires it on every commit), so a natural frame — and with
    // it another retry chance — arrives when the sweep ends.
    if (initTimeoutRetryPending) scheduleInitRetryWake(delayMs);
  }, delayMs);
}

/**
 * Classify an init failure and arm — or refuse — the bounded retry
 * (issue #1694). The starved/dead distinction is the whole point: a
 * deadline miss says nothing about the worker's health, every other
 * failure says the script will never run.
 */
function noteWorkerInitFailure(error: unknown): void {
  if (error instanceof WorkerInitTimeoutError) {
    initTimeoutCount++;
    if (initTimeoutCount < SORT_WORKER_INIT_MAX_ATTEMPTS) {
      initTimeoutRetryPending = true;
      const backoffMs = SORT_WORKER_INIT_RETRY_BASE_MS * initTimeoutCount;
      initRetryNotBeforeMs = performance.now() + backoffMs;
      workerInitState = 'starved';
      // Arming a retry is useless if no frame ever comes to run it.
      scheduleInitRetryWake(backoffMs + SORT_WORKER_INIT_RETRY_WAKE_SLACK_MS);
      log.warning(
        Modules.WORKER_POOL,
        `SortWorker init missed its ${config.depthSort.workerInitTimeoutMs}ms deadline ` +
          `(attempt ${initTimeoutCount}/${SORT_WORKER_INIT_MAX_ATTEMPTS}) — the main thread was ` +
          'likely starved by a large load; depth sorting is off (identity order drawn) until a ' +
          `retry succeeds, next attempt in ${backoffMs}ms`
      );
      return;
    }
    initTimeoutRetryPending = false;
    clearInitRetryWake();
    workerInitState = 'failed';
    log.error(
      Modules.WORKER_POOL,
      `SortWorker init missed its deadline ${initTimeoutCount} times — giving up; depth ` +
        'sorting stays off (identity order drawn) until the next app re-init'
    );
    return;
  }
  // Not a deadline miss: the worker script is dead / unusable. Nothing to
  // retry — the cached rejection is the final answer, so no wake either.
  initTimeoutRetryPending = false;
  clearInitRetryWake();
  workerInitState = 'failed';
  log.error(Modules.WORKER_POOL, 'SortWorker failed to initialize', error);
}

/**
 * Observable depth-sort worker state, for the debug surface and tests —
 * the degrade used to be visible only as a console line (issue #1694).
 *
 * - `state`: `'idle'` = never spawned, or an init still in flight;
 *   `'ready'` = the worker initialized and sorts are flowing;
 *   `'starved'` = init missed its deadline and a bounded retry is armed or
 *   in flight (depth sorting is off MEANWHILE, not for the session);
 *   `'failed'` = permanently unavailable (dead script, or the starved
 *   retries were exhausted).
 * - `initTimeouts`: how many init attempts missed the startup deadline
 *   (`config.depthSort.workerInitTimeoutMs`, default 30 s) this session.
 *
 * The verdict describes INIT state only: a worker that dies AFTER a successful
 * init keeps reporting `'ready'` while every sort silently burns the
 * `SORT_RPC_TIMEOUT_MS` deadline instead (a code span, not a `{@link}` — it is
 * module-private, and this function is exported).
 */
export function getDepthSortWorkerStatus(): {
  state: 'idle' | 'ready' | 'starved' | 'failed';
  initTimeouts: number;
} {
  return { state: workerInitState, initTimeouts: initTimeoutCount };
}

/**
 * Force the next view sweep to RE-COMMIT a node that must (re-)deliver its
 * centers to the SortWorker, by clearing BOTH freshness stamps. Shared by
 * the two callers that need exactly this — the switch-to-sorted branch of
 * {@link noteDepthSortBlendingModeSwitch} and the post-retry
 * re-registration ({@link reregisterAfterLateWorkerInit}) — because the
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
function invalidateSortedNodeCommitStamps(mesh: THREE.Mesh): void {
  invalidateCommittedDataStamp(mesh);
  delete (mesh.userData as { loadedViewVersion?: number }).loadedViewVersion;
}

/**
 * After a LATE (retried) init succeeds, get every sorted node's centers to
 * the worker again.
 *
 * The worker is brand new and holds no registrations, and the coordinator
 * retains no centers by design (the buffers were transferred, or their
 * thunks were never paid because the commit's continuation drained into
 * the failure catch). So the only way back is a re-commit — exactly the
 * situation {@link noteDepthSortBlendingModeSwitch}'s switch-to-sorted
 * branch solves, and by exactly the same means: invalidate the stamps that
 * would let the memoized-concat fast path skip re-projection, then request
 * ONE reprocess for the whole set.
 *
 * Gated on `requestReprocess` being wired: with nothing able to re-commit,
 * invalidating stamps would strand the nodes stamp-less for no gain. (The
 * RETRY itself is deliberately NOT gated on it — a fresh worker still lets
 * future commits register naturally.)
 */
function reregisterAfterLateWorkerInit(): void {
  if (!requestReprocess) return;
  let anyInvalidated = false;
  for (const state of nodeStates.values()) {
    // A registered node already has its centers in the worker, so
    // re-committing it would be pure waste. This is a cheap guard, not a
    // race guard: `maybeRetryStarvedWorkerInit` attaches this function to
    // the fresh `initPromise` synchronously, so it always runs BEFORE any
    // commit's own continuation on that promise — no commit can have
    // registered by now. What the ordering does mean is that a commit
    // landing inside the retry window has its stamps invalidated one
    // microtask before its own continuation registers + sorts: it pays one
    // discarded sort and is re-committed by the reprocess below. Self-
    // healing waste, not corruption.
    if (state.registered) continue;
    // A tracked node with no `committedData` stamp is one whose stamps were
    // ALREADY invalidated and whose re-commit is still pending — the
    // switch-to-sorted branch of {@link noteDepthSortBlendingModeSwitch}
    // firing inside the retry window, or an earlier run of this very sweep.
    // Both have already requested the reprocess that re-commits (and
    // re-registers) the node, so there is nothing to add here.
    // Notably NOT the LOD-demotion case, tempting as that reading is: every
    // demotion path pairs its `clearCommittedDataStamp` with
    // `releaseDepthSortNode` (`data/scene-loader.ts`'s `releaseLazy*`
    // callbacks), which deletes the node from `nodeStates` entirely — a
    // demoted level is never seen by this loop. The guard therefore stays as
    // the defensive peer of the module's other `!hasCommittedData` skips
    // rather than as the demotion filter.
    if (!hasCommittedData(state.mesh)) continue;
    if (!isLiveOrderDependent(liveBlendingMode(state.mesh))) continue;
    invalidateSortedNodeCommitStamps(state.mesh);
    anyInvalidated = true;
  }
  // `requestReprocess` is `SceneLoader.updateView({})`, and by now a view
  // sweep may well be in flight — the retry is only dispatched from a frame
  // where none was, and the init it awaited took real time. That is not a lost
  // call: the loader's serialization branch ABORTS the in-flight pass (its
  // commit is skipped), parks this state as pending, and re-enters
  // `updateView` with it as the aborted pass unwinds, so the re-commit these
  // stamp-less nodes need always happens. The accepted cost is the aborted
  // pass's fetch/decode work, which the winning pass redoes. Gating on
  // `isLoadInProgress` instead would be the worse trade: the stamps are
  // already cleared at this point, so a skipped reprocess leaves the nodes
  // stamp-less and unsorted indefinitely — the bug itself.
  if (anyInvalidated) requestReprocess();
}

/**
 * True when something on screen actually WANTS sorting right now: a tracked
 * node that is effectively visible, still holds committed data, and whose LIVE
 * blending mode is order-dependent.
 *
 * One helper for the two consumers that must not drift — the retry gate below
 * (do not spend one of the few attempts on a scene that never sorts) and
 * {@link isDepthSortAvailable} (do not announce a degrade about a subsystem
 * this scene never uses).
 */
function anyNodeWantsSorting(): boolean {
  for (const state of nodeStates.values()) {
    const mesh = state.mesh;
    if (!isEffectivelyVisible(mesh)) continue;
    if (!hasCommittedData(mesh)) continue;
    if (!isLiveOrderDependent(liveBlendingMode(mesh))) continue;
    return true;
  }
  return false;
}

/**
 * Retry a STARVED init, at most {@link SORT_WORKER_INIT_MAX_ATTEMPTS}
 * times per session (issue #1694). Driven from the per-frame scheduler
 * rather than a timer: the frame loop is precisely where "the main thread
 * has room again" becomes observable, and it is already gated on the
 * conditions a retry must respect.
 *
 * Every gate below is about not WASTING one of the few attempts:
 * - the failure must be the retryable (deadline) kind and attempts must
 *   remain — `initTimeoutRetryPending` carries both,
 * - the backoff must have elapsed (a retry issued into the same stall
 *   would just miss the deadline again),
 * - no offline capture may be in flight: it drains synchronously against a
 *   time bound and cannot await an init that may run to its deadline,
 * - and SOMETHING must actually want sorting right now — a visible, still
 *   committed, live-order-dependent node (`anyNodeWantsSorting`, shared with
 *   {@link isDepthSortAvailable} so the gate and the monitor's note can never
 *   disagree about what "wants sorting" means). An idle or all-additive scene
 *   would otherwise burn the budget before the scene that needs it loads.
 *
 * Deliberately NOT gated on `requestReprocess`: a recovered worker is
 * worth having even when nothing can force a re-commit, because every
 * FUTURE commit then registers naturally. Only the re-registration sweep
 * needs that callback, and it checks for itself.
 *
 * (The caller explicitly suppresses this helper while `isLoadInProgress`
 * is true, so a retry never fires into an in-flight load sweep — the very
 * condition that starves init in the first place.)
 */
function maybeRetryStarvedWorkerInit(): void {
  if (!initTimeoutRetryPending) return;
  if (initTimeoutCount >= SORT_WORKER_INIT_MAX_ATTEMPTS) return;
  if (performance.now() < initRetryNotBeforeMs) return;
  if (captureSuppressDepth > 0) return;
  if (!anyNodeWantsSorting()) return;

  // Consume the arm-flag and the cached rejection TOGETHER: this is the only
  // place the cached rejection is dropped, and consuming the flag in the same
  // synchronous step means a fresh attempt can never be started twice — the
  // failure bookkeeping then re-arms itself from the new attempt's own
  // outcome. Dropping the cached rejection INVALIDATES the previous
  // attempt, so bump the epoch with it (see {@link initEpoch}) — and the
  // self-wake armed for this backoff has done its job.
  initTimeoutRetryPending = false;
  initPromise = null;
  initEpoch++;
  clearInitRetryWake();
  const epoch = initEpoch;
  log.warning(
    Modules.WORKER_POOL,
    `Retrying the starved SortWorker init (attempt ${initTimeoutCount + 1}/` +
      `${SORT_WORKER_INIT_MAX_ATTEMPTS})`
  );
  // The re-registration below invalidates both freshness stamps on several
  // nodes at once, with one accepted, transient cost. A cleared
  // `loadedViewVersion` makes the node STALE for the LOD freshness check
  // (`scene/lod-freshness.ts::isFresh`), so a substitutive-LOD group's
  // display falls back to its coarsest ready level (`coarsestFreshOrReadyIndex`
  // in `scene/lod-group-registry.ts`) until the settle-gated `maybeKickReload`
  // climbs back. While `committedData` is absent, the node keeps its exact
  // cross-node `renderOrder`; only its within-mesh permutation stays stale
  // until the re-commit. The LOD fallback is exactly what the switch-to-sorted
  // blending-mode hook has always done for ONE node; an automatic recovery
  // just does it for several, which is why it is written down here rather than
  // left to be discovered.
  void ensureWorker().then(
    () => {
      // Epoch guard: a dispose (or another retry) between the dispatch and
      // this resolve means these nodes belong to a different session — see
      // {@link initEpoch}.
      if (epoch !== initEpoch) return;
      reregisterAfterLateWorkerInit();
    },
    () => {
      // Swallowed: noteWorkerInitFailure already logged and either re-armed
      // the retry or marked the worker permanently failed.
    }
  );
}

/**
 * Spawn + initialize the sort worker AHEAD of any data, at app init.
 *
 * The worker used to be spawned lazily by the first order-dependent commit
 * — which is the worst possible moment, because that commit lands exactly
 * when the main thread and the data-worker pool are saturated decoding the
 * scene. Starting here instead means `initWasm()` runs while the app is
 * still idle and finishes long before a million-element commit exists.
 *
 * Deliberately NOT folded into {@link configureDepthSort}: that function is
 * pure wiring, every unit test calls it, and spawning there would change
 * observable behaviour across the whole suite.
 *
 * Fire-and-forget and idempotent — the commit path's own `ensureWorker()`
 * remains the correctness path (it dedupes on `initPromise`) and still
 * covers embedders that configure late.
 *
 * This spends the FIRST of the bounded init attempts
 * (`SORT_WORKER_INIT_MAX_ATTEMPTS`), before any node has committed. If it
 * misses the deadline, recovery is not immediate: nothing retries here, and
 * `maybeRetryStarvedWorkerInit` only spends an attempt once something visible
 * actually wants sorting — which is the point, since burning the budget on an
 * empty scene would leave none for the load that needs it. (Both names stay
 * code spans: they are module-private, and a doc link from an EXPORTED symbol
 * to one of those trips the TypeDoc warning ratchet.)
 *
 * The cost is unconditional: the sort-worker chunk and its WASM are fetched on
 * every page load, including scenes that never sort (all-additive points, an
 * opaque mesh, `?debug` with no dataset). `?depthSort=0` is the opt-out — it
 * spawns nothing at all.
 */
export function warmUpDepthSortWorker(): void {
  if (!depthSortEnabled) return;
  // No `Worker` constructor in this environment (the unit suite's default
  // `node` env, jsdom, SSR): there is nothing to warm up, and trying anyway
  // logs a red 'SortWorker failed to initialize ReferenceError: Worker is not
  // defined' and latches `workerInitState` at 'failed' module-wide — for the
  // rest of that test file, since the module is a singleton. The COMMIT path is
  // deliberately left unguarded: a node that actually asks for sorting must
  // still report honestly.
  if (typeof Worker === 'undefined') return;
  void ensureWorker().catch(() => {
    // Already logged by ensureWorker; a failed warm-up must not become an
    // unhandled rejection, and the commit path will retry.
  });
}

/**
 * False only when BOTH halves hold: depth sorting has GIVEN UP for this
 * session (`'failed'` — a dead worker script, or the starved retry budget
 * exhausted) AND something visible actually wants sorting right now
 * (`anyNodeWantsSorting`). A `'starved'` init still reports available, because
 * a retry is armed and expected to recover.
 *
 * The demand half is not cosmetic. The worker is warmed up unconditionally at
 * app init, so a CSP-blocked chunk latches `'failed'` on a scene with NO
 * order-dependent geometry at all (all-additive points, an opaque mesh,
 * `?debug` with no dataset) — announcing a degrade there would report on a
 * subsystem that session never uses, and drag the performance panel out of its
 * "No timing data yet" empty state to do it. Sharing the predicate with the
 * retry gate is what keeps the two readings of "wants sorting" identical.
 *
 * Exact contract, and it is asymmetric: a SHOWN note proves that visible,
 * committed, order-dependent geometry is being drawn in storage order. Its
 * ABSENCE proves nothing — a `'starved'` init reports available for the whole
 * retry window while identity order is drawn, `?depthSort=0` pins identity
 * order and reports available by definition, and a node whose commit could not
 * reach a HEALTHY worker (detached buffer, throwing centers thunk) is
 * invisible here. {@link getDepthSortWorkerStatus} is the finer-grained read.
 */
export function isDepthSortAvailable(): boolean {
  return !(workerInitState === 'failed' && anyNodeWantsSorting());
}

/**
 * Push the geometry's active ordering slot onto a material's
 * `uSortedIndexSlot` uniform. Covers both backends: the GLSL materials
 * expose a plain `IUniform`, and the TSL materials expose a
 * `proxyIUniform` that writes straight through to the node — neither
 * rebuilds or recompiles on a value change.
 *
 * `slot` is `0 | 1`, not `number`, and that is load-bearing: this is the
 * ONLY path by which a value reaches `uSortedIndexSlot`, and the two
 * backends do not agree outside that domain. GLSL selects with
 * `slot == 1 ? back : front`, so anything else reads the FRONT buffer;
 * the TSL twin is branchless (`a·(1-slot) + b·slot`, forced by
 * `.select()` being a statement — see `sortedIndexNode`), so a slot of 2
 * would evaluate to `2b - a`: garbage indices, not a fallback. Rather
 * than clamp on every vertex for a state nothing can produce, the type
 * keeps it unrepresentable at the one entry point. Widening this
 * signature — or adding a third ordering buffer — means giving the two
 * shaders a shared, tested selection rule first.
 */
function applySortedIndexSlotToMaterial(
  material: THREE.Material | THREE.Material[] | undefined,
  slot: 0 | 1
): void {
  if (!material) return;
  // Scalar and array handled without a temporary wrapper array: this runs
  // for every tracked node on every frame (twice with a pick material), so
  // it must stay allocation-free — the per-frame scratch invariant below.
  if (Array.isArray(material)) {
    for (const m of material) setSortedIndexSlotUniform(m, slot);
  } else {
    setSortedIndexSlotUniform(material, slot);
  }
}

/** Write one material's `uSortedIndexSlot`, if it has one. */
function setSortedIndexSlotUniform(material: THREE.Material, slot: 0 | 1): void {
  const uniform = (material as THREE.ShaderMaterial | undefined)?.uniforms?.uSortedIndexSlot;
  if (uniform) uniform.value = slot;
}

/**
 * Point a node's shaders at whichever ordering buffer is currently
 * complete. The PICK material must move with the visual one: it shares
 * the geometry and emits `vElementId` from the same index, so a pick
 * pass reading the other buffer would resolve hovers against a stale
 * permutation.
 */
function syncSortedIndexSlot(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (!geometry) return;
  const slot = activeSortedIndexSlot(geometry);
  applySortedIndexSlotToMaterial(mesh.material, slot);
  const pickNode = (mesh.userData as { pickNode?: THREE.Mesh } | undefined)?.pickNode;
  if (pickNode) applySortedIndexSlotToMaterial(pickNode.material, slot);
}

/** Installed post-render acknowledgement hook for each sortable mesh. */
const drawAcknowledgementHooks = new WeakMap<THREE.Mesh, THREE.Mesh['onAfterRender']>();

/**
 * Chain a mesh-local post-render hook that acknowledges the selected ordering
 * only after THREE has consumed its pending attribute ranges and drawn it.
 * The hook uses the render callback's geometry argument (rather than
 * `mesh.geometry`) so a pool swap inside another callback cannot acknowledge
 * the wrong buffer. Existing user callbacks are preserved.
 */
function ensureDrawAcknowledgementHook(mesh: THREE.Mesh): void {
  const installed = drawAcknowledgementHooks.get(mesh);
  if (installed && mesh.onAfterRender === installed) return;

  const previous = mesh.onAfterRender;
  const hook: THREE.Mesh['onAfterRender'] = (...args) => {
    const geometry = args[3];
    acknowledgeSortedIndexOrderingDraw(geometry as THREE.InstancedBufferGeometry);
    // The indexed (mesh) apply acknowledges at the same point and for the
    // same reason. Both are keyed by geometry and both no-op when the
    // geometry has nothing pending, so calling each unconditionally is
    // cheaper than discriminating the node type on every rendered frame.
    acknowledgeTriangleOrderingDraw(geometry);
    previous.apply(mesh, args);
  };
  drawAcknowledgementHooks.set(mesh, hook);
  mesh.onAfterRender = hook;
}

/** Read the live REQUESTED blending mode stamped by the material wrappers. */
function liveBlendingMode(mesh: THREE.Mesh): BlendingMode | undefined {
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
function isLiveOrderDependent(mode: BlendingMode | undefined): boolean {
  if (!mode) return false;
  return needsDepthSort(mode);
}

/**
 * Record a non-noop commit of a sortable node (any of the four geometry
 * types). Always bumps the node's generation
 * (dropping any in-flight sort's result). When the node's LIVE effective
 * blending mode is order-dependent, transfers the projected centers to
 * the SortWorker and requests one sort from the current camera pose.
 *
 * `centers3` — projected 3D centers, `count * 3` floats:
 * - As a `Float32Array` it is TRANSFERRED (detached) on the
 *   order-dependent path, so the caller must hand over a buffer with no
 *   other readers (gsplats pass `processed.centers3D`: the commit's
 *   texture-write loops are the last main-thread readers, and the
 *   memoized-concat noop identity keys on `sourceData`, never
 *   `processed.*`).
 * - As a THUNK it is invoked lazily, only when the node actually
 *   registers (order-dependent mode, non-empty, latest generation) — the
 *   points commit uses this to pay the O(N) fresh-copy of
 *   `data.positions` only on the sorted path. The thunk MUST return a
 *   freshly allocated array: the returned buffer is transferred, and a
 *   `subarray` view of a live array would detach that array with it.
 *
 * `triangleSource` — MESH ONLY: the canonical visible index triples a
 * resolved ordering permutes into `geometry.index`. Omitted by the three
 * instanced types, whose ordering is a draw-slot indirection with nothing
 * to permute FROM. See {@link NodeSortState.triangleSource}.
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

export function noteDepthSortCommit(
  mesh: THREE.Mesh,
  centers3: Float32Array | (() => Float32Array),
  count: number,
  triangleSource?: Uint32Array
): void {
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
      shardCount: 0,
    };
    nodeStates.set(nodeId, state);
  }
  ensureDrawAcknowledgementHook(mesh);
  state.generation = ++nextGeneration;
  // A commit SUPERSEDES any indexed ordering that was written but not yet
  // drawn — `updateMeshGeometry` has already overwritten the index buffer by
  // the time this runs, so that ordering will never reach a frame. Without
  // this the next render would acknowledge it and report an obsolete
  // permutation as uploaded (a false 'Depth Sort' completion sample), or leave
  // the session open until some later draw. The instanced path does the same
  // thing from its identity write; this is the indexed peer of it, and it must
  // fire on EVERY commit, not only the release branch below — a commit that
  // stays order-dependent overwrites the buffer just as thoroughly.
  cancelTriangleOrderingApply(mesh.geometry);
  // Rebound to THIS commit's triples before the release branch below, which
  // drops it again: a stale source outliving its commit is the one way the
  // indexed apply can write a corrupt permutation, and the generation check
  // alone would not catch it (the ordering and the source would be from
  // different commits while the generation matched the newer one).
  state.triangleSource = triangleSource;

  // Push the geometry's (possibly just-normalised) slot to the materials
  // NOW, not only on the next per-frame pump: the commit's writers may
  // have re-homed the geometry on slot 0 (identity write) or handed the
  // mesh a different geometry entirely (pool acquire), and a render that
  // does not go through the frame loop — the settle-scheduled pick pass —
  // can fire before the pump's per-frame re-assert runs. Idempotent.
  syncSortedIndexSlot(mesh);

  const mode = liveBlendingMode(mesh);
  if (!depthSortEnabled || !isLiveOrderDependent(mode) || count === 0) {
    // Depth sorting disabled (identity ordering pinned), commutative
    // blending, or an empty frame: no ordering needed. Drop any
    // worker-side registration so the worker doesn't hold stale
    // centers for a node that may not sort again for a long time —
    // and the recorded pose with it (see clearSortPose's invariant).
    clearSortPose(state);
    // Drop the retained triples too: an unsorted mesh must not keep a
    // second copy of its index alive for the rest of the session. (The
    // pending indexed apply was already cancelled above, for every commit.)
    state.triangleSource = undefined;
    // Unshard too: a node that no longer sorts must go back to one draw, and
    // its shard views alias an ordering that is about to stop being maintained.
    releaseDepthShards(mesh);
    releaseWorkerNode(nodeId);
    return;
  }

  // Re-establish the shard set against THIS commit. The element count and the
  // pooled geometry can both have changed, and either invalidates the shard
  // boundaries and the views that alias them; `syncDepthShards` rebuilds
  // wholesale and is a no-op while the count is 1 (every node today).
  syncDepthShards(mesh, requestedShardCount(mesh), count);

  // Sort NOW when the node is small enough, so the first frame after this
  // commit is already ordered instead of showing the fallback the commit path
  // just wrote. Returns the resolved centers so the copy is paid once.
  const syncedCenters = trySynchronousFirstSort(mesh, centers3, count);
  const centersForWorker: Float32Array | (() => Float32Array) = syncedCenters ?? centers3;

  const generation = state.generation;
  void ensureWorker()
    .then(() => {
      if (!api) return;
      // A newer commit may have landed while the worker was spawning.
      const current = nodeStates.get(nodeId);
      if (current?.generation !== generation) return;
      // Resolve a lazy centers provider only now — past the generation
      // re-check, so a superseded commit never pays the copy. Resolved
      // BEFORE the `registered` flag flips: a throwing provider then
      // leaves the node unregistered (same semantics as a registerNode
      // rejection) instead of stranding a phantom registration the
      // per-frame recovery branch would dispatch guaranteed-null sorts
      // against. (The throw lands in the outer catch below.)
      const buffer = typeof centersForWorker === 'function' ? centersForWorker() : centersForWorker;
      current.registered = true;
      // The register RPC is its own promise — the surrounding .catch
      // only sees synchronous throws, so a transport/transfer rejection
      // here would otherwise float as an unhandled rejection.
      api
        .registerNode(transfer({ nodeId, generation, centers3: buffer, count }, [buffer.buffer]))
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
      // Once per EPISODE: while init is failing the cached initPromise stays
      // rejected, so EVERY later commit lands here — per-commit error lines
      // would flood a timelapse scrub. Rendering degrades gracefully to the
      // identity (storage) order.
      //
      // The message must not claim more than it knows, and this catch has
      // MORE than one cause: an unavailable worker, but also a throwing lazy
      // centers provider or a `transfer()` of an already-detached buffer —
      // both of which can fire while the worker is perfectly healthy. So it
      // reports the symptom (this commit did not reach the SortWorker, the
      // identity order is drawn) and quotes the worker's actual state
      // instead of asserting one (issue #1694).
      if (!warnedWorkerUnavailable) {
        warnedWorkerUnavailable = true;
        const status = getDepthSortWorkerStatus();
        log.error(
          Modules.WORKER_POOL,
          'Depth-sort commit could not reach the SortWorker — drawing the identity ' +
            `(unsorted) order (worker init state: ${status.state}, deadline misses: ` +
            `${status.initTimeouts}; first failing node: ${nodeId})`,
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
  clearShardBounds(state);
}

/**
 * Drop a node's per-shard depth bounds.
 *
 * Called from every branch that invalidates the ordering they describe, because
 * a box retained past its permutation would place a shard by where its elements
 * USED to be — a silent wrong ordering rather than a visible failure.
 */
function clearShardBounds(state: NodeSortState): void {
  state.shardCount = 0;
  state.shardBoundsMin = undefined;
  state.shardBoundsMax = undefined;
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
 * Format an ordering-upload byte count for the 'Depth Sort' monitor line.
 * `uploaded` distinguishes bytes that have reached the GPU (`up`, after the
 * selected mesh completes a render) from bytes merely STAGED for upload
 * (`sched`, at worker resolve) — issue #713. The panel shows this as the
 * pass's `info` tag.
 */
function formatOrderingBytes(bytes: number, uploaded: boolean): string {
  const suffix = uploaded ? 'up' : 'sched';
  return bytes >= 1_000_000
    ? `${(bytes / 1_000_000).toFixed(1)} MB ${suffix}`
    : `${Math.round(bytes / 1000)} KB ${suffix}`;
}

/**
 * The model-view matrix to sort against.
 *
 * Both matrices are normally renderer-maintained (updated during render), but
 * a commit can fire BEFORE the next frame — the first commit of a load, or
 * while the on-demand loop is idle-paused — and would otherwise read a
 * stale/identity pose. Refresh them here and derive the view matrix locally
 * (`camera.matrixWorldInverse` is only refreshed by `renderer.render`, not by
 * `updateMatrixWorld`).
 */
function computeModelView(mesh: THREE.Mesh, camera: THREE.Camera): THREE.Matrix4 {
  mesh.updateWorldMatrix(true, false);
  camera.updateMatrixWorld();
  return new THREE.Matrix4().copy(camera.matrixWorld).invert().multiply(mesh.matrixWorld);
}

/**
 * Sort a just-committed node's ordering RIGHT NOW, on the main thread, so the
 * very first frame after the commit is already ordered.
 *
 * Returns the resolved centers buffer when the sort landed (so the caller can
 * hand the SAME copy to the worker registration instead of paying the lazy
 * provider twice), or `undefined` when the synchronous path was declined — in
 * which case the caller must behave exactly as it did before.
 *
 * WHY this exists. Every commit of an order-dependent node writes a fallback
 * ordering and then waits for the worker's answer, which is a Comlink round
 * trip through `ensureWorker` + `registerNode` + `sort` — measured at ~5 ms,
 * so at least one frame renders on the fallback. On a one-off load that frame
 * is invisible. During nD playback it is not, because a commit lands at EVERY
 * timepoint: on the `cloud` demo (4D Points, `volumetric` blending) the
 * fallback was reached 14 times a second, and it composited only 61.7% of
 * sampled element pairs in correct back-to-front order against 100% for a real
 * sort. That is the flash reported in #2290.
 *
 * WHY it is bounded. The kernel is a counting sort — two O(n) passes and a
 * 65,536-bucket histogram — so its cost is linear and measurable rather than
 * data-dependent: 0.8 ms at 34k elements, 2.5 ms at 250k, 16.2 ms at 1M.
 * `config.depthSort.syncSortMaxElements` is both the per-node ceiling and the
 * shared element budget for every commit between frame evaluations; once spent,
 * the async path stays the only sane answer. `repairSortedIndexForCount` (the
 * commit paths' fallback) keeps that one frame far closer to sorted than storage
 * order was.
 *
 * The result is written LIVE rather than staged, because a permutation
 * computed in one shot has no partial state to hide — see
 * {@link writeSortedIndexOrderingLive}.
 *
 * The async pipeline behind this is deliberately left ALONE: the node is still
 * registered and still dispatches its usual first sort. Suppressing that would
 * save a redundant sort, but the kernels are exact-parity so the worker's
 * answer is the SAME permutation, landing as a no-op overwrite — not worth
 * changing the coordinator's dispatch contract for. The whole of this
 * function's job is to make the FIRST frame correct; the pipeline's job, of
 * keeping the ordering current as the camera moves, is unchanged.
 */
function trySynchronousFirstSort(
  mesh: THREE.Mesh,
  centers3: Float32Array | (() => Float32Array),
  count: number
): Float32Array | undefined {
  const limit = syncSortElementLimit();
  if (limit <= 0 || count > limit || count > syncSortElementsRemaining) return undefined;
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  if (!getActiveSortedIndexAttribute(geometry)) return undefined;
  const camera = getCamera?.();
  if (!camera) return undefined;

  let buffer: Float32Array;
  try {
    buffer = typeof centers3 === 'function' ? centers3() : centers3;
  } catch {
    // A throwing lazy provider must land where it always did: the async
    // path's outer `.catch`, which owns the once-per-episode report. Decline
    // and let it be called again there.
    return undefined;
  }
  // A provider that under-delivers (or a detached buffer) is not sortable —
  // decline rather than sort garbage; the async path re-checks the same way.
  if (buffer.length < count * 3) return undefined;

  const modelView = computeModelView(mesh, camera);
  const ordering = new Uint32Array(count);
  // No shard bounds on the synchronous first sort: nothing is sharded before
  // the node has been registered, and this path exists to get SOMETHING drawn
  // on the first frame as cheaply as possible.
  sort_splats_by_depth(
    buffer,
    new Float32Array(modelView.elements),
    ordering,
    count,
    0,
    EMPTY_SHARD_BOUNDS,
    EMPTY_SHARD_BOUNDS
  );
  if (writeSortedIndexOrderingLive(geometry, ordering, count) !== count) return undefined;
  syncSortElementsRemaining -= count;
  requestRender?.();
  return buffer;
}

/**
 * Request one sort for a registered node, respecting the
 * single-in-flight rule. Queues a re-sort if one is already running.
 */
function scheduleSort(mesh: THREE.Mesh, nodeId: string): void {
  const state = nodeStates.get(nodeId);
  const camera = getCamera?.();
  // `api` is wrapped BEFORE `initializeWithGuard` is awaited, so a live handle
  // is not the same thing as a usable worker: between the spawn and 'ready'
  // there is a window in which `sort` would reject NOT_INITIALIZED worker-side
  // (`requireWasm`). Gating here rather than at each caller keeps it a single
  // chokepoint — the capture drain's force loop dispatches without consulting
  // `state.registered`, and a commit stamps `nodeStates` + `committedData`
  // synchronously, so it can reach this during startup warm-up.
  //
  // Skipping beats letting it reject, and not only for the error line:
  // `recordSortPose` below runs BEFORE the RPC, so a dispatch that rejects
  // still leaves a pose on record for a sort that never ran — which is exactly
  // what silences the per-frame `!lastSortAxis` recovery dispatch for a node
  // whose first real dispatch raced a null camera.
  if (!state || !api || workerInitState !== 'ready' || !camera) return;
  if (state.inFlight) {
    state.resortQueued = true;
    return;
  }
  // NO apply-gate. Sorting and applying run CONCURRENTLY: a stream writes
  // into the inactive buffer, so the displayed ordering stays a complete
  // permutation throughout and a fresher sort is never wasted — it is
  // held and swapped in at the next flip. (The gate existed because a
  // stream used to write into the LIVE attribute, where sorting faster
  // than the apply cadence only prolonged the mixed state.)
  //
  // A throttle on "an ordering is already queued" was built and MEASURED
  // on the 10M orbit bench: it cut sorts 24 -> ~15 but moved the
  // sort-adjacent frame p99 only ~92 -> ~89 ms (inside run-to-run noise)
  // while costing 41% more staleness on the 8M-point orbit (fitted
  // sort-axis lag, fast orbit: mean 44 -> 63 deg). Freshness is the whole
  // point of re-sorting, so it was dropped. Sort traffic stays bounded by
  // one-in-flight-per-node.
  state.inFlight = true;

  const generation = state.generation;
  // Both matrices are normally renderer-maintained (updated during
  // render), but a commit can fire BEFORE the next frame — the first
  // commit of a load, or while the on-demand loop is idle-paused — and
  // would otherwise read a stale/identity pose. Refresh them here and
  // derive the view matrix locally (camera.matrixWorldInverse is only
  // refreshed by renderer.render, not by updateMatrixWorld).
  const modelView = computeModelView(mesh, camera);
  recordSortPose(state, modelView);

  // One detached profiler pass per dispatch — its duration is the whole
  // dispatch→applied lifecycle the monitor's 'Depth Sort' line shows. The
  // session is opened here at dispatch but ended at APPLY completion, not
  // at RPC resolve: large orderings apply CHUNKED over many later frames,
  // so the resolve handler hands the session to writeSortedIndexOrdering's
  // lifecycle callbacks (issue #713). It ends immediately only when no
  // ordering is staged (stale/demoted/rejected) or the RPC fails.
  //
  // Capture the profiler and its generation HERE too, so the dedicated
  // completion stream records against the same profiler the session merges
  // into and honors the same reset-isolation contract (issue #711).
  const profiler = getProfiler?.() ?? null;
  const session = profiler?.beginDepthSortPass() ?? null;
  const profilerGeneration = profiler?._currentGeneration() ?? 0;
  // The profiler session does not expose its elapsed time (SessionImpl's
  // startTime is private), so time the dispatch→resolve round-trip locally
  // for the queueMs derivation below.
  const dispatchedAt = performance.now();
  // Ownership latch shared by the .then/.catch handlers: once the session is
  // handed to the chunked-apply callbacks (below), NEITHER handler may end
  // it — the callbacks own its close. Without this, a throw from the
  // post-handoff `requestRender()` / re-sort drain would reject the .then
  // and route into .catch's `session?.end()`, closing the pass early and
  // mislabeling the eventual applied sample as scheduled (issue #713).
  let applyOwnsSession = false;

  // Timeout-raced: a worker that CRASHES mid-session leaves the Comlink
  // RPC pending forever, and a stuck `inFlight` is unrecoverable — the
  // per-frame scheduler skips in-flight nodes and later commits only set
  // `resortQueued`, which never drains. Routing the timeout through the
  // existing .catch clears `inFlight` and drains the queue (bounded
  // staleness degrade instead of a permanently unsorted node). A merely
  // SLOW sort that resolves after the deadline is harmless too: the race
  // has already rejected, so its late resolve is dropped — the node just
  // stays unsorted until the next commit/camera trigger re-sorts it.
  void withTimeout(
    'depth-sort',
    api.sort({
      nodeId,
      generation,
      modelView: new Float32Array(modelView.elements),
      // Request per-shard depth bounds only for a node that is actually
      // sharded. Until the shard-count policy lands (spec §4) nothing is, so
      // this is 0 everywhere and the kernel skips the work entirely.
      shardCount: requestedShardCount(mesh),
    }),
    SORT_RPC_TIMEOUT_MS
  )
    .then((result) => {
      const roundTripMs = performance.now() - dispatchedAt;
      const current = nodeStates.get(nodeId);
      if (!current) {
        session?.end();
        return; // released mid-sort — no ordering staged
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
        // Which apply path this node uses. A retained `triangleSource` is
        // the mesh signal and it is set by the SAME commit whose generation
        // just matched, so the two can never describe different face sets.
        const triangleSource = current.triangleSource;
        const applicable =
          triangleSource !== undefined || !!geometry?.getAttribute?.('aSortedIndex');
        // Adopt the shard bounds under the SAME generation guard as the
        // ordering, and only when that ordering will actually be applied —
        // boxes describing a permutation the node never draws would place its
        // shards by where their elements were about to be, not where they are.
        if (applicable && result.shardCount > 0 && result.shardBoundsMin && result.shardBoundsMax) {
          current.shardCount = result.shardCount;
          current.shardBoundsMin = result.shardBoundsMin;
          current.shardBoundsMax = result.shardBoundsMax;
        } else {
          clearShardBounds(current);
        }
        if (applicable) {
          // Ordering upload: 4 bytes/splat through the attribute
          // update-range machinery (the architecture's headline number) on
          // the instanced path. The indexed path uploads THREE index
          // entries per face instead, at the index buffer's own width —
          // Uint16 under 65536 vertices on either WebGL path (see
          // `createMeshIndexAttribute`), but the NATIVE WebGPU backend widens
          // that same buffer to Uint32 in place at first upload and it stays
          // widened from then on (see `applyMeshIndices`), so this reads
          // Uint32 there. Hence the live `BYTES_PER_ELEMENT` rather than a
          // width assumed from the vertex count.
          const bytes = triangleSource
            ? result.ordering.length * 3 * (geometry.index?.array.BYTES_PER_ELEMENT ?? 4)
            : result.ordering.length * 4;
          // Timing split (perf campaign): kernelMs = inside the backend
          // call (incl. wasm-bindgen boundary copies for compiled WASM);
          // boundaryMs = worker-side overhead around it; queueMs =
          // round-trip minus worker time (Comlink RPC + structured clone
          // + event-loop queueing). All durations, so worker-clock vs
          // main-clock is safe; clamped at 0 against timer granularity.
          // These keys are LAST-WRITE on profiler metadata merge (each
          // sort pass has its own seq, so the 'Depth Sort' root always
          // shows the latest sort's split). The byte label starts as
          // SCHEDULED here and is upgraded to uploaded only after THREE
          // renders the selected buffer (issue #713).
          session?.setMetadata({
            elements: result.ordering.length,
            kernelMs: result.kernelMs,
            boundaryMs: Math.max(0, result.workerMs - result.kernelMs),
            queueMs: Math.max(0, roundTripMs - result.workerMs),
            info: formatOrderingBytes(bytes, false),
          });
          // Hand the profiler session to the apply's lifecycle so the pass
          // spans dispatch→applied: onApplied (post-render acknowledgement)
          // ends it as UPLOADED; onAbandoned
          // (superseded/cancelled/demoted/released/disposed) ends it as
          // scheduled. Neither writer invokes either unless it ACCEPTS the
          // ordering (returns > 0) — issue #713.
          const resolvedAt = performance.now();
          const applyCallbacks = session
            ? {
                onApplied: () => {
                  const appliedAt = performance.now();
                  session.setMetadata({
                    applyMs: Math.max(0, appliedAt - resolvedAt),
                    info: formatOrderingBytes(bytes, true),
                  });
                  // The dedicated MONOTONIC completion stream records only
                  // orderings that actually became drawable — not merely
                  // worker resolves that were staged and later abandoned.
                  // It uses the dispatch-time profiler/generation, matching
                  // the session's reset-isolation contract (issue #711).
                  if (profiler && profiler._currentGeneration() === profilerGeneration) {
                    profiler.recordDepthSortCompletion({
                      lastMs: Math.max(0, appliedAt - dispatchedAt),
                      kernelMs: result.kernelMs,
                      boundaryMs: Math.max(0, result.workerMs - result.kernelMs),
                      queueMs: Math.max(0, roundTripMs - result.workerMs),
                      elements: result.ordering.length,
                    });
                  }
                  session.end();
                },
                onAbandoned: () => session.end(),
              }
            : undefined;
          // Instanced: large orderings apply CHUNKED across frames
          // (element-storage routes internally, perf lever L8) — this call
          // only STAGES the pending state and the per-frame pump streams
          // the slices. Indexed: the write is atomic and complete on
          // return, because a half-permuted index buffer is not a
          // permutation (triangle-ordering.ts).
          const staged = triangleSource
            ? writeSortedTriangleOrdering(
                geometry,
                triangleSource,
                result.ordering,
                result.ordering.length,
                applyCallbacks
              )
            : writeSortedIndexOrdering(
                geometry,
                result.ordering,
                result.ordering.length,
                applyCallbacks
              );
          if (staged > 0) {
            // Handed off BEFORE the throwing `requestRender()` below, so a
            // throw there can't route into .catch and close the pass early.
            applyOwnsSession = true;
            // Instanced: bootstrap the pump's requestRender chain (the
            // per-frame pump keeps the on-demand loop alive between
            // slices). Indexed: the permutation is already in the buffer,
            // so this is the one frame it needs to reach the screen.
            requestRender?.();
          }
        }
      }
      // No ordering staged (stale generation, demotion, rejected write, or
      // missing attribute): the pass is just the dispatch→resolve
      // round-trip — close it now (issue #713).
      if (!applyOwnsSession) session?.end();

      if (current.resortQueued) {
        current.resortQueued = false;
        // Demoted mid-sort (stamp cleared): the queued request describes a
        // population the geometry no longer holds, and re-promotion always
        // re-commits — which schedules the sort it actually needs. Dispatching
        // here would burn worker time on an ordering the resolve path is
        // guaranteed to discard.
        if (stillCommitted) scheduleSort(mesh, nodeId);
      }
    })
    .catch((error) => {
      // Only close the pass here for a genuine RPC failure — NOT when the
      // ordering was already handed to the chunked-apply callbacks and a
      // post-handoff step (e.g. requestRender) threw: those callbacks own
      // the close, and ending here would mislabel the applied sample (#713).
      if (!applyOwnsSession) session?.end();
      const current = nodeStates.get(nodeId);
      log.error(Modules.WORKER_POOL, `SortWorker sort failed for ${nodeId}`, error);
      if (!current) return;
      current.inFlight = false;
      // Drain a queued re-sort even on failure — a commit landed while
      // this sort was out, and dropping its request would leave the node
      // stale until the NEXT commit. Bounded: only a real commit sets
      // resortQueued, so a persistently failing worker cannot loop. Same
      // demotion guard as the resolve path: a cleared stamp means the
      // re-promotion commit will schedule the sort that matters.
      if (current.resortQueued) {
        current.resortQueued = false;
        if (hasCommittedData(mesh)) scheduleSort(mesh, nodeId);
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
 * Advance every in-flight chunked ordering application by one slice
 * (perf lever L8 — chunk mechanics + the transient-mix trade are
 * documented in `element-storage.ts`). Runs once per rendered frame
 * from {@link evaluateDepthSortPerFrame}, ahead of its early-returns.
 *
 * - A cleared `committedData` stamp (LOD demotion — the geometry went
 *   back to the evictable pool, possibly already serving another node)
 *   ABORTS the apply: the remaining slices describe the demoted
 *   commit's population. (The commit path's identity writers cancel
 *   independently; this catches demotions with no follow-up write.)
 * - A node that is not effectively visible (hidden ancestor/layer,
 *   `visible=false`, LOD-hidden) is SKIPPED without pumping — it must
 *   not advance/accumulate a stream while not drawn (issue #715). It
 *   resumes when shown (the visibility change requests its own render).
 * - While slices remain, request another frame — the pump is the only
 *   thing keeping the on-demand loop alive between slices. The slice
 *   just written rides THIS frame's flush (per-frame callbacks run
 *   before render), so the final slice needs no extra frame. EXCEPT a
 *   STALLED pump (previous slice unflushed — frustum-culled or otherwise
 *   not drawn; issue #715): it made no progress, so the loop idles until
 *   a DRAWN frame consumes the pending slice and the next pump advances.
 * - The final slice SELECTS the complete buffer for this frame; the mesh's
 *   `onAfterRender` hook acknowledges upload/draw and closes its profiler
 *   lifecycle only after THREE actually renders it.
 * - On COMPLETION, drain a queued re-sort (a commit that landed while a
 *   sort was in flight parked it) — the counterpart of the resolve
 *   path's drain, restoring the natural cadence: sort → apply N frames
 *   → next sort.
 *
 * Per-node bound: one slice per pending node per frame — several large
 * nodes resolving simultaneously each add one slice's cost to a frame
 * (bounded per node, not globally; simultaneous 10M-scale resolves are
 * already serialized by the per-node single-in-flight sort rule).
 */
function pumpChunkedOrderingApplies(): void {
  for (const [nodeId, state] of nodeStates) {
    const geometry = state.mesh.geometry as THREE.InstancedBufferGeometry | undefined;
    if (!geometry) continue;

    // Re-assert the slot on EVERY tracked node every frame, not just on
    // the frame it flips. The uniform lives on the materials while the
    // slot lives on the geometry, and the two are re-paired behind our
    // back: the pool hands a geometry (slot included) to another node,
    // a TSL graph rebuild replaces the uniform leaves, a pick material
    // is created after the flip. Idempotent and a couple of property
    // writes per node, so re-asserting is cheaper than tracking every
    // way they can desync.
    syncSortedIndexSlot(state.mesh);
    // Re-point any shard children at the parent's CURRENT material, for exactly
    // the same reason: the LayersPanel's clone-on-first-use and the LOD
    // cross-fade both REPLACE `mesh.material`, and a shard still holding the old
    // object would silently stop tracking every appearance edit. A pointer
    // compare per shard, and a no-op while the node is unsharded.
    syncShardMaterials(state.mesh);

    if (!hasPendingSortedIndexOrderingApply(geometry)) {
      // The indexed (mesh) path has nothing to stream — its write is atomic
      // — but a DEMOTED node's written-but-undrawn ordering still owns a
      // profiler session nobody else will close. A no-op unless one is
      // actually pending, so this costs a map miss on every other node.
      if (!hasCommittedData(state.mesh)) cancelTriangleOrderingApply(geometry);
      continue;
    }
    if (!hasCommittedData(state.mesh)) {
      // LOD demotion — the geometry went back to the pool, so the
      // remaining slices describe a population this mesh no longer
      // holds. A queued re-sort is CLEARED, not drained: dispatching it
      // would sort the released registration's old centers only for the
      // resolve path to discard the result (an invisible multi-million-
      // element sort delaying live nodes), and re-promotion always
      // re-commits, which schedules the sort the node actually needs.
      cancelSortedIndexOrderingApply(geometry);
      state.resortQueued = false;
      continue;
    }
    // Issue #715: a hidden/off-screen node must not advance its stream —
    // writing while not drawn would accumulate an unflushed union (the
    // per-slice bound only holds when each slice is uploaded before the
    // next). Skip the pump entirely (write nothing while hidden → zero
    // accumulation) and request no render. This resumes cleanly when the
    // node is shown: the last slice it wrote while visible was already
    // consumed, double-buffering keeps the drawn buffer a complete valid
    // permutation throughout, and the visibility change requests its own
    // render, which re-runs the pump to resume the drain. (Visibility
    // gating alone is insufficient for frustum culling — a culled mesh
    // stays `visible=true` — which is what the pump's back-pressure
    // covers; this branch handles the hidden case.)
    if (!isEffectivelyVisible(state.mesh)) continue;
    const { more, flipped, stalled } = pumpSortedIndexOrderingApply(geometry);
    if (flipped) {
      // The just-completed buffer becomes selected for this frame. This runs
      // before render; the mesh's onAfterRender hook is the separate proof
      // that THREE consumed the update ranges and issued a draw with it.
      syncSortedIndexSlot(state.mesh);
      requestRender?.();
    }
    if (more) {
      // A STALLED pump made no progress (the previous slice is still
      // unflushed — the mesh was not drawn since; issue #715). Do NOT
      // spin the on-demand loop on it: the node resumes one slice per
      // DRAWN frame, since each drawn frame's render consumes the prior
      // slice and any subsequent render — camera motion, commit,
      // visibility change — re-runs the pump, which then advances.
      if (!stalled) requestRender?.();
    } else if (state.resortQueued) {
      state.resortQueued = false;
      scheduleSort(state.mesh, nodeId);
    }
  }
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
 * nodes whose live mode is no longer order-dependent. A streaming
 * chunked apply does NOT skip — a fresher sort fills the inactive
 * buffer concurrently.
 *
 * Work is tiered by dependency: frame-state cleanup runs above every gate,
 * pure main-thread cross-node ordering runs above the loader gate, and only
 * work that touches the SortWorker stays below that gate.
 */
export function evaluateDepthSortPerFrame(): void {
  syncSortElementsRemaining = Math.max(0, syncSortElementLimit());
  // Drop the previous frame's render-order state FIRST — before any
  // early-return — so a disposed/dataset-switched frame can't leave the
  // module-scoped rank memo holding stale partition-wrapper subtrees alive.
  clearRenderOrderFrameState();
  // Chunked ordering applies advance BEFORE every early-return below:
  // they need neither a camera nor an idle loader (stalling them during
  // a load would postpone convergence to the newest complete ordering),
  // and a paused stream must always resume its bounded drain.
  pumpChunkedOrderingApplies();
  // Deliberately NOT gated on `api`: the cross-node renderOrder pass is
  // pure main-thread and must keep ordering meshes back-to-front even
  // when the SortWorker was never constructed (`api` stays null forever
  // after a constructor throw — e.g. a CSP-blocked worker script — the
  // documented degrade-to-unsorted-normal mode). The within-mesh
  // re-sort triggers are worker-dependent, but `scheduleSort` guards
  // both `api` and init readiness itself.
  if (!depthSortEnabled || nodeStates.size === 0) return;
  const camera = getCamera?.();
  if (!camera) return;
  const loadInProgress = isLoadInProgress?.() ?? false;
  // Keep worker retry behind the load gate: a starved init must not run while
  // a view-update sweep is in flight, since that sweep IS the main-thread
  // saturation that starved it. Everything else the retry needs to know
  // (something visible actually wants sorting, the backoff, an offline
  // capture) it checks itself.
  if (!loadInProgress) maybeRetryStarvedWorkerInit();

  // Which nodes should be split, BEFORE the collect loop reads their shards.
  // Kept behind the load gate for the same reason worker retry is: a sweep in
  // flight means bounds are still moving, and re-splitting on each intermediate
  // commit would churn shard meshes for orderings about to be replaced.
  if (!loadInProgress) evaluateShardPolicy();

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
    const mode = liveBlendingMode(mesh);
    if (!isLiveOrderDependent(mode)) {
      // No longer order-dependent (e.g. switched to additive) — clear any
      // cross-part renderOrder bias so it doesn't strand a stale ordering.
      // Shard children too: they are never collected again from here, so a
      // positive rank left on one would keep drawing it out of position among
      // the commutative content it now belongs with.
      if (mesh.renderOrder !== 0) mesh.renderOrder = 0;
      for (const shard of depthShardChildren(mesh)) {
        if (shard.renderOrder !== 0) shard.renderOrder = 0;
      }
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
    collectRenderOrderSlot(mesh, scratch.mv, scratch.camPos, shardOrderInputFor(state));

    // Everything below dispatches or evaluates a within-mesh worker sort.
    // Keep that work paused during a loader sweep, but do not pause the
    // pure-main-thread cross-mesh ordering collected above.
    if (loadInProgress) continue;
    // Mode switches and late-worker re-registration deliberately invalidate
    // commit stamps while the existing geometry stays visible. Those meshes
    // still receive their cross-mesh rank above, but cannot dispatch a worker
    // sort until re-commit. LOD demotion is only a defensive peer case here:
    // its synchronous release removes the mesh from nodeStates first.
    if (!hasCommittedData(mesh)) continue;
    const bs = (mesh.geometry as THREE.BufferGeometry | undefined)?.boundingSphere;

    // === Within-mesh re-sort trigger (Phase 3) ===
    if (state.inFlight) continue;
    // No apply-gate here either (see scheduleSort): a streaming apply no
    // longer blocks a fresher sort, so camera motion is answered as soon
    // as the threshold is crossed rather than after the stream drains.
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
 * True when the depth-sort subsystem has settled: no tracked node has a
 * sort RPC outstanding (`inFlight`), a re-sort queued behind one
 * (`resortQueued`), or a chunked ordering apply still streaming into its
 * inactive buffer (`hasPendingSortedIndexOrderingApply`). This is the
 * termination condition {@link resortForCapture} drains toward — while
 * ANY of those hold, the drawn permutation is not yet the pose-fresh one.
 * Module-private: the drain loop is the only caller.
 */
function isCaptureQuiescent(): boolean {
  for (const state of nodeStates.values()) {
    if (state.inFlight || state.resortQueued) return false;
    const geometry = state.mesh.geometry as THREE.InstancedBufferGeometry | undefined;
    if (geometry && hasPendingSortedIndexOrderingApply(geometry)) return false;
  }
  return true;
}

/**
 * Produce a fresh, fully-settled depth ordering for the CURRENT camera
 * pose and return only once it is drawn — the offline-capture entry point.
 *
 * WHY this exists: the Phase-3 per-frame scheduler
 * ({@link evaluateDepthSortPerFrame}) is wired ONLY as an
 * AnimationController per-frame callback, so it runs exclusively inside
 * the rAF loop. Offline capture (the gallery orbit-video pass) deliberately
 * STOPS that loop, then per frame moves the camera and renders
 * synchronously. With the loop dead the scheduler never fires: across the
 * whole orbit there are zero re-sorts and zero cross-node renderOrder
 * updates, so any order-dependent node (`normal` / `volumetric`) is filmed
 * with the back-to-front permutation frozen at the pre-orbit pose. This
 * helper drives the scheduler + worker round-trip + chunked apply by hand
 * so each captured frame is ordered for its own pose.
 *
 * It is a NO-OP (returns as soon as it observes quiescence) when depth sort
 * is disabled, no order-dependent node exists, or nothing is pending. It
 * also degrades gracefully when the SortWorker is unavailable — or merely not
 * READY yet, a capture launched during startup warm-up: the cross-node
 * renderOrder pass inside `evaluateDepthSortPerFrame` is pure main-thread and
 * still runs, and `scheduleSort` guards both `api` and init readiness itself,
 * so no fresh sort is dispatched but the renderOrder assignment is still
 * refreshed for the pose.
 *
 * `maxWaitMs` bounds the drain so a crashed / wedged worker can never hang
 * the capture — the loop exits and the frame is captured with whatever
 * ordering is current.
 */
export async function resortForCapture(maxWaitMs = 3000): Promise<void> {
  // The capture stopped the rAF loop on purpose; `requestRender` is wired
  // to `animationController.startAnimation()`, and the sort resolve/pump
  // paths call `requestRender?.()`. Suppress it for the duration so
  // draining (which we drive ourselves) can't silently re-arm the frozen
  // loop. Safe offline: there are no concurrent commits to lose a frame
  // request from. The suppression is depth-counted / reentrancy-safe: this
  // helper is exposed on `__luxarDebug`, so an overlapping (nested) call
  // could otherwise snapshot `null` and restore `null` permanently, wedging
  // the render loop forever. Only the OUTERMOST call snapshots and restores.
  if (captureSuppressDepth === 0) requestRenderBeforeCapture = requestRender;
  captureSuppressDepth++;
  requestRender = null;
  // The drain below never draws, but on the chunked (WebGL) path a
  // multi-slice apply stalls after its first slice until a DRAW's upload
  // ack releases the #715 back-pressure — so without this bypass any
  // order-dependent node past one slice (>1M elements, e.g. the 3M-star
  // gaia demo) could never reach quiescence: every captured frame would
  // burn the full maxWaitMs and still film a stale ordering. Offline,
  // folding the slices into one upload on the capture's own render is
  // exactly acceptable (the union range stays contiguous and current).
  setSortedIndexApplyBackPressureBypassed(true);
  try {
    // FORCE a fresh sort on every eligible node — offline capture can
    // afford a full sort per frame, so the ordering is exact for THIS pose
    // rather than only when a per-frame threshold happens to trip.
    // scheduleSort already queues a re-sort if one is in flight.
    //
    // Order matters: the force loop runs BEFORE the per-frame pass below.
    // Its motion trigger dispatches for the same pose whenever the camera
    // moved past a threshold since the last sort (the first orbit frame
    // after repositioning, a coarse-threshold config), and a force-call on
    // a node that pass just put in flight would only set `resortQueued` —
    // a SECOND, identical full sort run serially after the first. Force-
    // first, the pass's in-flight skip makes the two compose to one sort.
    if (depthSortEnabled && getCamera?.() && api) {
      for (const [nodeId, state] of nodeStates) {
        const mesh = state.mesh;
        if (!isEffectivelyVisible(mesh)) continue;
        if (!hasCommittedData(mesh)) continue;
        if (!isLiveOrderDependent(liveBlendingMode(mesh))) continue;
        scheduleSort(mesh, nodeId);
      }
    }

    // Cross-node renderOrder pass + pump any pending chunked applies, all
    // for the current pose (its re-sort trigger skips the in-flight nodes
    // the force loop just dispatched).
    evaluateDepthSortPerFrame();

    // Nothing to wait for (disabled / no order-dependent node / worker
    // unavailable): return before opening the drain loop.
    if (isCaptureQuiescent()) return;

    // Drain to quiescence, bounded by maxWaitMs. Each iteration yields a
    // macrotask (setTimeout(0)) so worker resolutions land, then pumps one
    // chunked-apply slice and re-asserts renderOrder via
    // evaluateDepthSortPerFrame.
    const start = performance.now();
    while (performance.now() - start < maxWaitMs) {
      await new Promise<void>((r) => setTimeout(r, 0));
      evaluateDepthSortPerFrame();
      if (isCaptureQuiescent()) return;
    }
  } finally {
    // Clamped, not a bare decrement: disposeDepthSort resets the depth to
    // 0, and a capture that was in flight across that dispose must not
    // drive it negative (a later capture would then decrement back to a
    // non-zero exit and strand `requestRender` at null forever).
    captureSuppressDepth = Math.max(0, captureSuppressDepth - 1);
    if (captureSuppressDepth === 0) {
      requestRender = requestRenderBeforeCapture;
      // Drop the snapshot so it can't pin the app's closure between
      // captures (mirrors the dispose-path hygiene below).
      requestRenderBeforeCapture = null;
      setSortedIndexApplyBackPressureBypassed(false);
    }
  }
}

/**
 * React to a sortable layer's blending mode changing at runtime (the
 * LayersPanel compose chain — spec §5.4). Wired for all four geometry
 * types.
 *
 * Switching TO a sorted mode cannot simply "register+sort": the
 * SortWorker has no centers for a node that was order-independent at its
 * last commit (registration is gated on the live mode, and the staged
 * arrays were transferred/discarded). Instead, clear the node's noop
 * stamp (`userData.committedData`) and request a view reprocess — the
 * memoized-concat noop path would otherwise skip re-projection entirely.
 * The resulting standard commit registers + sorts like any other (the
 * SliceCache still holds the source nD data; one O(N) re-projection per
 * mode switch, a rare user action).
 *
 * Switching AWAY just stops future sorts (a sorted order is harmless
 * under commutative modes — no identity reset needed); the worker-side
 * centers are released as hygiene.
 */
export function noteDepthSortBlendingModeSwitch(
  mesh: THREE.Mesh,
  newMode: BlendingMode | undefined,
  prevMode: BlendingMode | undefined
): void {
  // Disabled session: identity ordering is pinned for every mode, so a
  // switch TO a sorted mode must not force the (expensive) reprocess;
  // there is also no worker-side state to release on a switch away.
  if (!depthSortEnabled) return;
  if (!newMode || newMode === prevMode) return;
  // Sorted modes = normal ∪ volumetric (needsDepthSort), for all three
  // geometry types. A switch BETWEEN two sorted modes (e.g.
  // normal→volumetric) is deliberately a no-op here: the ordering stays
  // valid; the projection/output change is the material's problem (TSL
  // rebuild / GLSL define recompile).
  const wasSorted = prevMode !== undefined && needsDepthSort(prevMode);
  const isSorted = needsDepthSort(newMode);
  if (isSorted && !wasSorted) {
    // Both freshness stamps go, so the reprocess actually re-projects and
    // re-commits (including a hidden resident LAZY LOD level, which no
    // sweep visits) — the full reasoning, shared with the post-retry
    // re-registration path, lives on the helper.
    invalidateSortedNodeCommitStamps(mesh);
    requestReprocess?.();
  } else if (!isSorted && wasSorted) {
    const state = nodeStates.get(mesh.uuid);
    if (state) {
      // Invalidate any in-flight sort's result; keep the counter
      // monotonic for the node's next order-dependent commit. The pose
      // clear is the same hygiene as the commit path's release branch.
      state.generation = ++nextGeneration;
      clearSortPose(state);
      // …and so is dropping the retained triples: an opaque mesh has no
      // reason to keep a second copy of its index alive. A switch back
      // re-commits (the branch above), which re-supplies them.
      state.triangleSource = undefined;
    }
    cancelTriangleOrderingApply(mesh.geometry);
    releaseWorkerNode(mesh.uuid);
  }
}

/**
 * Drop a node's sort state + worker-side registration. Wired to node
 * disposal and lazy-LOD release for every sortable type (an in-flight
 * sort resolves onto a missing state and is discarded; a no-op for
 * never-registered nodes).
 */
export function releaseDepthSortNode(mesh: THREE.Mesh): void {
  const nodeId = mesh.uuid;
  if (!nodeStates.delete(nodeId)) return;
  // Shards are a rendering-time detail of a SORTED node, so this is their single
  // teardown point — LOD demotion, layer detach and scene disposal all already
  // route here. Restores the parent to one full draw and drops the views, whose
  // ordering is about to stop being maintained.
  releaseDepthShards(mesh);
  // With the node state gone the per-frame pump would never visit this
  // geometry again — abort any in-flight chunked apply so the map
  // doesn't pin the geometry + its (up to 40 MB) ordering until the
  // pool's next identity write.
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry | undefined;
  if (geometry) {
    cancelSortedIndexOrderingApply(geometry);
    // Indexed peer: closes a written-but-undrawn ordering's profiler
    // session. Nothing to undo in the buffer itself — see
    // `cancelTriangleOrderingApply`.
    cancelTriangleOrderingApply(geometry);
  }
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
  // Same orphan hazard as releaseDepthSortNode, swept globally (a
  // dataset switch tears everything down anyway).
  cancelAllSortedIndexOrderingApplies();
  cancelAllTriangleOrderingApplies();
  api?.releaseAllNodes().catch(() => {});
}

/**
 * Terminate the worker and reset all module state (app teardown; also
 * the test reset). Safe to call when never spawned.
 */
export function disposeDepthSort(): void {
  nodeStates.clear();
  // Module-state reset completeness: in-flight chunked applies hold
  // geometry + ordering references in element-storage's map, and the
  // indexed path's acknowledgement map holds geometry + profiler closures.
  cancelAllSortedIndexOrderingApplies();
  cancelAllTriangleOrderingApplies();
  // Drop the per-slice render-continuation hook so a dispose/re-init does
  // not keep the old app's requestRender closure alive.
  setSortedIndexApplyRequestRender(null);
  // Same hygiene for the offline-capture suppression state: a capture in
  // flight across this dispose must not later restore the old app's
  // requestRender closure from its snapshot (its clamped `finally` then
  // restores the null set below), and the snapshot itself must not pin
  // the closure. The back-pressure bypass is module state in
  // element-storage — reset it too.
  captureSuppressDepth = 0;
  requestRenderBeforeCapture = null;
  setSortedIndexApplyBackPressureBypassed(false);
  // Module-state reset completeness: both per-frame containers can hold
  // THREE object references between calls (the rank memo until the next
  // evaluate's clear; the slots only if an evaluate threw mid-collect) —
  // an embedder that disposes and re-inits in one page must not have the
  // old scene pinned by them.
  clearRenderOrderFrameState();
  // The display-dims accessor lives in the render-order submodule, not the
  // locals below; drop it too so a dispose/re-init doesn't keep the old app's
  // closure alive (re-init overwrites it via configureDepthSort regardless).
  setRenderOrderDisplayDimsAccessor(null);
  worker?.terminate();
  worker = null;
  api = null;
  initPromise = null;
  // ORPHAN any init attempt still in flight — kept adjacent to the
  // `initPromise = null` it invalidates, because the two are one invariant.
  // This is the reset that cannot be done by nulling a variable: the attempt's
  // init deadline timer lives in its own closure and nothing here can cancel
  // it, so the epoch is what stops it from classifying a miss against the NEXT
  // app's healthy worker (see {@link initEpoch} for the full failure chain).
  initEpoch++;
  getCamera = null;
  requestRender = null;
  requestReprocess = null;
  isLoadInProgress = null;
  getProfiler = null;
  depthSortEnabled = true;
  syncSortElementsRemaining = syncSortElementLimit();
  warnedWorkerUnavailable = false;
  // Init-failure bookkeeping is module state too: an embedder that disposes
  // and re-inits in one page must start with a full retry budget and a
  // truthful status, not a stale 'failed'/'starved' verdict about the worker
  // just terminated (issue #1694).
  workerInitState = 'idle';
  initTimeoutRetryPending = false;
  initTimeoutCount = 0;
  initRetryNotBeforeMs = 0;
  // The armed self-wake must not survive the app that armed it (it would
  // request a frame from a re-inited app for a retry that no longer exists).
  clearInitRetryWake();
}
