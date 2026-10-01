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
 * This file is the public entry point (configuration, commit, release,
 * blending-mode switch, glass registry, dispose) and re-exports the rest.
 * The machinery lives in `depth-sort-coordinator/`, split by concern over
 * ONE shared state object (`state.ts`):
 * - `worker-lifecycle.ts` — SortWorker spawn, guarded init, starved-init
 *   retry, status;
 * - `ordering-apply.ts` — slot sync, draw acknowledgement, the commit-time
 *   synchronous first sort;
 * - `scheduler.ts` — sort dispatch + resolve, the chunked apply pump, the
 *   per-frame camera-motion pass;
 * - `capture.ts` — the offline-capture drain ({@link resortForCapture});
 * - `render-order.ts` / `triangle-ordering.ts` — cross-node renderOrder and
 *   the indexed (mesh) apply.
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
import { transfer } from 'comlink';
import {
  cancelAllSortedIndexOrderingApplies,
  cancelSortedIndexOrderingApply,
  setSortedIndexApplyBackPressureBypassed,
  setSortedIndexApplyRequestRender,
} from './element-storage';
import { needsDepthSort } from './blending-state';
import type { BlendingMode } from '../types/blending';
import {
  clearRenderOrderFrameState,
  drawsAfterEmissive,
  setRenderOrderDisplayDimsAccessor,
} from './depth-sort-coordinator/render-order';
import {
  cancelAllTriangleOrderingApplies,
  cancelTriangleOrderingApply,
} from './depth-sort-coordinator/triangle-ordering';
import type { UpdateProfiler } from '../profiling/update-profiler';
import { log, Modules } from '../utils/log';
import { isEffectivelyVisible } from '../utils/object-visibility';
import { isPhysicalMeshMaterial } from './materials/mesh-physical/config';
import { setGlassPartition, type GlassPartition } from './materials/_shared/glass-partition';
import {
  clearSortPose,
  invalidateSortedNodeCommitStamps,
  isLiveOrderDependent,
  liveBlendingMode,
  nodeStates,
  releaseHeldDraw,
  releaseWorkerNode,
  shared,
  syncSortElementLimit,
} from './depth-sort-coordinator/state';
import {
  clearInitRetryWake,
  ensureWorker,
  getDepthSortWorkerStatus,
} from './depth-sort-coordinator/worker-lifecycle';
import {
  ensureDrawAcknowledgementHook,
  syncSortedIndexSlot,
  trySynchronousFirstSort,
} from './depth-sort-coordinator/ordering-apply';
import { scheduleSort } from './depth-sort-coordinator/scheduler';

export {
  getDepthSortWorkerStatus,
  isDepthSortAvailable,
  setSortWorkerUrl,
  setSortWorkerWasmPath,
  warmUpDepthSortWorker,
} from './depth-sort-coordinator/worker-lifecycle';
export { evaluateDepthSortPerFrame } from './depth-sort-coordinator/scheduler';
export { resortForCapture } from './depth-sort-coordinator/capture';

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
   * True while the loader is doing work of any kind (a view-update sweep or
   * its progressive-refinement drain). Gates ONLY the starved-worker init
   * retry — a retry fired into that main-thread saturation spends an attempt
   * on a guaranteed miss. Camera-motion re-sorts are deliberately NOT gated
   * on it: the refinement drain holds this true for seconds on a hosted
   * laddered scene, and a gated orbit left the drawn order up to ~180 deg
   * stale until the next rung commit. A re-sort racing a commit is safe —
   * the commit bumps the generation, so the stale result is dropped.
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
  shared.getCamera = options.getCamera;
  shared.requestRender = options.requestRender;
  shared.requestReprocess = options.requestReprocess ?? null;
  shared.isLoadInProgress = options.isLoadInProgress ?? null;
  shared.getProfiler = options.getProfiler ?? null;
  setRenderOrderDisplayDimsAccessor(options.getDisplayDims ?? null);
  // A consumed slice's upload acknowledgement resumes an idle chunked
  // stream the instant the mesh is drawn again (issue #715 resume gap —
  // the pump never requests a render on a stall). Through the LIVE binding,
  // not the configured closure: resortForCapture suppresses `requestRender`
  // for its drain, and a slice acknowledged mid-drain must not wake the loop.
  setSortedIndexApplyRequestRender(() => shared.requestRender?.());
  shared.syncSortElementsRemaining = syncSortElementLimit();
}

/**
 * Session master switch, applied at app init from `config.depthSort.enabled`
 * combined with the `?depthSort=0` URL escape hatch. Disabling pins each mesh's
 * identity (storage) ordering for deterministic E2E/visual runs; authored
 * cross-layer bands and the physical-glass draw-first rule still apply.
 */
export function setDepthSortEnabled(enabled: boolean): void {
  shared.depthSortEnabled = enabled;
}

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
 * to permute FROM. See `NodeSortState.triangleSource` (`depth-sort-coordinator/state.ts`).
 */
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
      count: 0,
      inFlight: false,
      resortQueued: false,
      lastSortAxis: null,
      lastSortOffset: 0,
      registered: false,
    };
    nodeStates.set(nodeId, state);
  }
  ensureDrawAcknowledgementHook(mesh);
  state.generation = ++nextGeneration;
  state.count = count;
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
  if (!shared.depthSortEnabled || !isLiveOrderDependent(mode) || count === 0) {
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
    releaseWorkerNode(nodeId);
    // No ordering will come for this commit, so a held append draw shows the
    // whole population now (the order is irrelevant or pinned here).
    releaseHeldDraw(mesh);
    return;
  }

  // Sort NOW when the node is small enough, so the first frame after this
  // commit is already ordered instead of showing the fallback the commit path
  // just wrote. Returns the resolved centers so the copy is paid once.
  const syncedCenters = trySynchronousFirstSort(mesh, centers3, count, triangleSource);
  const centersForWorker: Float32Array | (() => Float32Array) = syncedCenters ?? centers3;

  const generation = state.generation;
  void ensureWorker()
    .then(() => {
      if (!shared.api) {
        releaseHeldDraw(mesh);
        return;
      }
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
      shared.api
        .registerNode(transfer({ nodeId, generation, centers3: buffer, count }, [buffer.buffer]))
        .catch((error: unknown) => {
          log.error(Modules.WORKER_POOL, `SortWorker registerNode failed for ${nodeId}`, error);
          // The worker never received this generation's centers — clear
          // the flag (if still ours) so the per-frame scheduler doesn't
          // keep dispatching guaranteed-null sorts against the missing
          // registration until the next commit.
          const failed = nodeStates.get(nodeId);
          if (failed?.generation === generation) {
            failed.registered = false;
            releaseHeldDraw(mesh);
          }
        });
      scheduleSort(mesh, nodeId);
    })
    .catch((error) => {
      // This commit's population will not be sorted: show all of it.
      if (nodeStates.get(nodeId)?.generation === generation) releaseHeldDraw(mesh);
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
      if (!shared.warnedWorkerUnavailable) {
        shared.warnedWorkerUnavailable = true;
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
  if (!shared.depthSortEnabled) return;
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
    shared.requestReprocess?.();
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
    // The generation bump above drops the sort a held append draw waits for.
    releaseHeldDraw(mesh);
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
    // No sort will ever raise a held append draw now.
    releaseHeldDraw(mesh);
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
  shared.api?.releaseAllNodes().catch(() => {});
}

/**
 * The visible meshes that ask to draw AFTER the emissive data — physical glass
 * authored with `refract_data` (spec MESH_PHYSICAL_MATERIALS §3.4, Phase 3).
 *
 * The post-processing pipeline splits its scene pass around exactly these meshes, so
 * it asks here rather than traversing the scene: every data mesh of every geometry
 * type registers with this coordinator on commit and is released on disposal
 * (`releaseDepthSortNode` runs from the disposal walk and every LOD demotion path),
 * so `nodeStates` already IS the registry, with the visibility walk the pick pass
 * uses. A hidden LOD level or a hidden layer is excluded the same way it is excluded
 * from ranking. The caller passes a reusable array so a frame allocates nothing.
 */
export function collectRefractingGlass(out: THREE.Mesh[] = []): THREE.Mesh[] {
  out.length = 0;
  for (const state of nodeStates.values()) {
    const mesh = state.mesh;
    if (drawsAfterEmissive(mesh) && isEffectivelyVisible(mesh) && drawsAnyFace(mesh)) {
      out.push(mesh);
    }
  }
  return out;
}

/**
 * Whether a mesh has anything to draw. The nD slice cull does not hide a mesh whose
 * faces all fall outside the slab — it leaves `visible` alone and empties the geometry's
 * `drawRange` (`mesh-geometry.ts`: the index buffer is capacity-sized, `drawRange.count`
 * is the drawn quantity). A refracting glass pinned to another coordinate of a hidden
 * dimension is therefore "visible" yet draws nothing, and must not make the refraction
 * split pay its glass pass on every frame.
 */
function drawsAnyFace(mesh: THREE.Mesh): boolean {
  return mesh.geometry.drawRange.count !== 0;
}

/**
 * The visible meshes drawn by three's OWN materials that do not refract the data —
 * physical glass without `refract_data`, physical opaque surfaces. They carry none of
 * Luxar's shader code, so they cannot classify their fragments against the refracting
 * glass's depth (`glass-partition.ts`); the refraction split draws them whole in pass A
 * and keeps them out of pass C, where a second draw would repaint the data composited
 * over them. Same registry, same visibility walk, same reusable array as
 * {@link collectRefractingGlass}.
 */
export function collectUnpartitionedMeshes(out: THREE.Mesh[] = []): THREE.Mesh[] {
  out.length = 0;
  for (const state of nodeStates.values()) {
    const mesh = state.mesh;
    if (
      isPhysicalMeshMaterial(mesh.material) &&
      !drawsAfterEmissive(mesh) &&
      isEffectivelyVisible(mesh) &&
      drawsAnyFace(mesh)
    ) {
      out.push(mesh);
    }
  }
  return out;
}

/**
 * Broadcast a glass-partition mode to every registered data mesh's material (the
 * refraction split's per-pass write; `glass-partition.ts`). Materials without the
 * uniform — three's own, and the pick materials, which live in the pick scene and never
 * register here — are left alone. Shared materials are written more than once, which
 * `setGlassPartition` makes a cheap no-op. Returns how many materials changed.
 */
export function applyGlassPartition(mode: GlassPartition): number {
  let changed = 0;
  for (const state of nodeStates.values()) {
    if (setGlassPartition(state.mesh.material, mode)) changed++;
  }
  return changed;
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
  shared.captureSuppressDepth = 0;
  shared.requestRenderBeforeCapture = null;
  setSortedIndexApplyBackPressureBypassed(false);
  // Module-state reset completeness: both per-frame containers can hold
  // THREE object references between calls (the rank lookup until the next
  // evaluate's clear; the slots only if an evaluate threw mid-collect) —
  // an embedder that disposes and re-inits in one page must not have the
  // old scene pinned by them. The cross-frame rank memo is weak-keyed.
  clearRenderOrderFrameState();
  // The display-dims accessor lives in the render-order submodule, not the
  // locals below; drop it too so a dispose/re-init doesn't keep the old app's
  // closure alive (re-init overwrites it via configureDepthSort regardless).
  setRenderOrderDisplayDimsAccessor(null);
  shared.worker?.terminate();
  shared.worker = null;
  shared.api = null;
  shared.initPromise = null;
  // ORPHAN any init attempt still in flight — kept adjacent to the
  // `initPromise = null` it invalidates, because the two are one invariant.
  // This is the reset that cannot be done by nulling a variable: the attempt's
  // init deadline timer lives in its own closure and nothing here can cancel
  // it, so the epoch is what stops it from classifying a miss against the NEXT
  // app's healthy worker (see {@link initEpoch} for the full failure chain).
  shared.initEpoch++;
  shared.getCamera = null;
  shared.requestRender = null;
  shared.requestReprocess = null;
  shared.isLoadInProgress = null;
  shared.getProfiler = null;
  shared.depthSortEnabled = true;
  shared.syncSortElementsRemaining = syncSortElementLimit();
  shared.warnedWorkerUnavailable = false;
  // Init-failure bookkeeping is module state too: an embedder that disposes
  // and re-inits in one page must start with a full retry budget and a
  // truthful status, not a stale 'failed'/'starved' verdict about the worker
  // just terminated (issue #1694).
  shared.workerInitState = 'idle';
  shared.initTimeoutRetryPending = false;
  shared.initTimeoutCount = 0;
  shared.initRetryNotBeforeMs = 0;
  // The armed self-wake must not survive the app that armed it (it would
  // request a frame from a re-inited app for a retry that no longer exists).
  clearInitRetryWake();
}
