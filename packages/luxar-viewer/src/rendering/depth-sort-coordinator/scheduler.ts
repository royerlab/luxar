/**
 * The depth-sort coordinator's scheduler: the single-in-flight sort dispatch
 * and its resolve/apply handling ({@link scheduleSort}), the per-frame chunked
 * apply pump, and the Phase-3 camera-motion re-sort pass
 * ({@link evaluateDepthSortPerFrame}) that also drives the cross-node
 * renderOrder assignment.
 *
 * @module rendering/depth-sort-coordinator/scheduler
 */

import * as THREE from 'three';
import {
  cancelSortedIndexOrderingApply,
  hasPendingSortedIndexOrderingApply,
  pumpSortedIndexOrderingApply,
  writeDeviceSortedIndexOrdering,
  writeSortedIndexOrdering,
  type SortedIndexApplyCallbacks,
} from '../element-storage';
import { hasCommittedData } from '../../types/committed-data';
import {
  assignGlobalRenderOrder,
  authoredLayerOrder,
  beginRenderOrderFrame,
  collectRenderOrderSlot,
  drawsAfterEmissive,
  drawsBeforeEmissive,
} from './render-order';
import { cancelTriangleOrderingApply, writeSortedTriangleOrdering } from './triangle-ordering';
import { config } from '../../config';
import { log, Modules } from '../../utils/log';
import { isEffectivelyVisible } from '../../utils/object-visibility';
import {
  isLiveOrderDependent,
  liveBlendingMode,
  orderingApplyHooks,
  releaseHeldDraw,
  syncSortElementLimit,
  type CoordinatorState,
  type NodeSortState,
} from './state';
import {
  computeModelView,
  formatOrderingBytes,
  orderingCoversCommit,
  recordSortPose,
  syncSortedIndexSlot,
} from './ordering-apply';
import { maybeRetryStarvedWorkerInit } from './worker-lifecycle';
import {
  selectSortProducer,
  type SortJob,
  type SortOutcome,
  type SortProducer,
  type SortRoute,
} from './sort-producers';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';

/**
 * How far behind an orthographic camera the BSP eye is placed (world units):
 * beyond any scene extent, so every split plane the view crosses at an angle
 * is decided by the view direction, and small enough to stay finite through
 * the wrapper's inverse matrix.
 */
const ORTHO_BSP_EYE_DISTANCE = 1e15;

/**
 * Request one sort for a registered node, respecting the
 * single-in-flight rule. Queues a re-sort if one is already running.
 *
 * The sort goes to the first ready producer that accepts the node
 * (`sort-producers.ts`); its outcome, whenever it arrives, is settled by
 * `settleSort` below.
 */
export function scheduleSort(c: CoordinatorState, mesh: THREE.Mesh, nodeId: string): void {
  const state = c.nodeStates.get(nodeId);
  const camera = c.getCamera?.();
  if (!state || !camera) {
    if (state) releaseHeldDraw(c, mesh);
    return;
  }
  // Fresh matrices even between frames (see computeModelView).
  const job: SortJob = {
    nodeId,
    mesh,
    state,
    generation: state.generation,
    modelView: computeModelView(mesh, camera),
  };
  // Skipping a dispatch no producer can serve beats letting it reject:
  // `recordSortPose` below runs BEFORE the sort, so a dispatch that fails still
  // leaves a pose on record for a sort that never ran — which is exactly what
  // silences the per-frame `!lastSortAxis` recovery dispatch for a node whose
  // first real dispatch raced a null camera.
  const producer = selectSortProducer(c, job);
  if (!producer) {
    // Nothing will sort this population, so a held append draw must not wait.
    releaseHeldDraw(c, mesh);
    return;
  }
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
  recordSortPose(state, job.modelView);
  const pass = beginSortPass(c, producer.route);
  let outcome: ReturnType<SortProducer['run']>;
  try {
    outcome = producer.run(job);
  } catch (error) {
    failSort(c, job, pass, error);
    return;
  }
  if (outcome instanceof Promise) {
    void outcome
      .then((result) => settleSort(c, job, pass, result))
      .catch((error: unknown) => failSort(c, job, pass, error));
    return;
  }
  try {
    settleSort(c, job, pass, outcome);
  } catch (error) {
    failSort(c, job, pass, error);
  }
}

/**
 * One dispatch's 'Depth Sort' profiler pass. Its duration is the whole
 * dispatch→applied lifecycle the monitor's 'Depth Sort' line shows. The
 * session opens at dispatch but ends at APPLY completion, not when the
 * producer answers: large orderings apply CHUNKED over many later frames, so
 * the settle hands the session to the apply's lifecycle callbacks (issue
 * #713). It ends immediately only when nothing is staged (stale, demoted,
 * rejected) or the sort fails.
 *
 * The profiler and its generation are captured at dispatch, so the dedicated
 * completion stream records against the profiler the session merges into and
 * honours the same reset-isolation contract (issue #711).
 */
interface SortPass {
  readonly route: SortRoute;
  readonly session: UpdateSession | null;
  readonly profiler: UpdateProfiler | null;
  readonly profilerGeneration: number;
  readonly dispatchedAt: number;
  /**
   * Set once the staged ordering's callbacks own closing the session. Neither
   * the settle nor the failure path may end it after that: a throw from a
   * post-handoff step (`requestRender`, the re-sort drain) routes into
   * {@link failSort}, and ending there would close the pass early and
   * mislabel the eventual applied sample as scheduled (issue #713).
   */
  handedOff: boolean;
}

function beginSortPass(c: CoordinatorState, route: SortRoute): SortPass {
  const profiler = c.getProfiler?.() ?? null;
  return {
    route,
    session: profiler?.beginDepthSortPass() ?? null,
    profiler,
    profilerGeneration: profiler?._currentGeneration() ?? 0,
    dispatchedAt: performance.now(),
    handedOff: false,
  };
}

/**
 * Settle a producer's outcome: stage it if it still describes the node's
 * committed population, close or hand off the profiler pass, release a held
 * draw nothing will satisfy, and drain a queued re-sort.
 */
function settleSort(
  c: CoordinatorState,
  job: SortJob,
  pass: SortPass,
  outcome: SortOutcome | null
): void {
  const current = c.nodeStates.get(job.nodeId);
  if (!current) {
    pass.session?.end();
    return; // released mid-sort — no ordering staged
  }
  current.inFlight = false;

  // Stale-drop: apply only when the ordering matches the node's CURRENT
  // generation (the producer checked its own registration; this re-check
  // covers commits that raced the sort). The `committedData` check
  // additionally covers LOD demotion: a demoted level's geometry returned to
  // the evictable pool (and may since belong to another node) — the cleared
  // stamp is exactly the signal that the mesh's geometry no longer holds this
  // commit's elements.
  const stillCommitted = hasCommittedData(job.mesh);
  if (
    outcome &&
    outcome.generation === current.generation &&
    stillCommitted &&
    orderingCoversCommit(outcomeLength(outcome), current, job.nodeId)
  ) {
    stageOutcome(c, job, pass, current, outcome);
  }
  // No ordering staged (stale generation, demotion, rejected write, or missing
  // attribute): the pass is just the dispatch→result round trip (issue #713).
  if (!pass.handedOff) pass.session?.end();
  // A sort for the CURRENT population that staged nothing leaves no ordering
  // for a held append draw to wait for. (A stale generation's does not: the
  // newer commit's own sort is queued behind this one.)
  if (!pass.handedOff && current.generation === job.generation) releaseHeldDraw(c, job.mesh);

  if (current.resortQueued) {
    current.resortQueued = false;
    // Demoted mid-sort (stamp cleared): the queued request describes a
    // population the geometry no longer holds, and re-promotion always
    // re-commits — which schedules the sort it actually needs. Dispatching
    // here would burn a sort on an ordering the settle is guaranteed to
    // discard.
    if (stillCommitted) scheduleSort(c, job.mesh, job.nodeId);
  }
}

/** The element count an outcome orders. */
function outcomeLength(outcome: SortOutcome): number {
  return outcome.kind === 'host' ? outcome.ordering.length : outcome.write.count;
}

/**
 * Stage a current outcome through the node's apply path, with the profiler
 * pass handed to the apply's lifecycle.
 */
function stageOutcome(
  c: CoordinatorState,
  job: SortJob,
  pass: SortPass,
  current: NodeSortState,
  outcome: SortOutcome
): void {
  const geometry = job.mesh.geometry as THREE.InstancedBufferGeometry;
  // Which apply path this node uses. A retained `triangleSource` is the mesh
  // signal and it is set by the SAME commit whose generation just matched, so
  // the two can never describe different face sets.
  const triangleSource = current.triangleSource;
  if (!(triangleSource ?? geometry?.getAttribute?.('aSortedIndex'))) return;
  const info = orderingInfo(outcome, triangleSource, geometry);
  // The byte label starts as SCHEDULED and is upgraded to uploaded only after
  // THREE renders the selected buffer (issue #713). The timing keys are
  // LAST-WRITE on profiler metadata merge (each sort pass has its own seq, so
  // the 'Depth Sort' root always shows the latest sort's split).
  const { kernelMs, boundaryMs, queueMs } = outcome.timing;
  pass.session?.setMetadata({
    elements: outcomeLength(outcome),
    kernelMs,
    boundaryMs,
    queueMs,
    info: info(false),
  });
  const callbacks = sortApplyCallbacks(c, pass, outcome, info);
  // Instanced host orderings apply CHUNKED across frames on classic WebGL
  // (element-storage routes internally, perf lever L8) — this call only STAGES
  // and the per-frame pump streams the slices. Indexed: the write is atomic
  // and complete on return, because a half-permuted index buffer is not a
  // permutation (triangle-ordering.ts). Device: the pump has the producer
  // write the inactive buffer itself, then flips.
  const staged = writeOutcome(geometry, triangleSource, outcome, callbacks);
  if (staged > 0) {
    // Handed off BEFORE the throwing `requestRender()` below, so a throw there
    // can't route into failSort and close the pass early.
    pass.handedOff = true;
    // Instanced: bootstrap the pump's requestRender chain (the per-frame pump
    // keeps the on-demand loop alive between slices). Indexed: the
    // permutation is already in the buffer, so this is the one frame it needs
    // to reach the screen.
    c.requestRender?.();
  }
}

/** Stage `outcome` into the node's ordering storage; the staged count, or 0. */
function writeOutcome(
  geometry: THREE.InstancedBufferGeometry,
  triangleSource: NodeSortState['triangleSource'],
  outcome: SortOutcome,
  callbacks: SortedIndexApplyCallbacks
): number {
  if (outcome.kind === 'device') {
    // A mesh orders its index buffer on the CPU; no device writer serves it.
    return triangleSource ? 0 : writeDeviceSortedIndexOrdering(geometry, outcome.write, callbacks);
  }
  const { ordering } = outcome;
  return triangleSource
    ? writeSortedTriangleOrdering(geometry, triangleSource, ordering, ordering.length, callbacks)
    : writeSortedIndexOrdering(geometry, ordering, ordering.length, callbacks, {
        adoptOrdering: true,
      });
}

/**
 * The profiler's label for an ordering: the bytes it uploads, or that the GPU
 * wrote it in place.
 *
 * A host ordering uploads 4 bytes/element through the attribute update-range
 * machinery on the instanced path. The indexed path uploads THREE index entries
 * per face instead, at the index buffer's own width — Uint16 under 65536
 * vertices on either WebGL path (see `createMeshIndexAttribute`), but the
 * NATIVE WebGPU backend widens that same buffer to Uint32 in place at first
 * upload and it stays widened from then on (see `applyMeshIndices`). Hence the
 * live `BYTES_PER_ELEMENT` rather than a width assumed from the vertex count.
 */
function orderingInfo(
  outcome: SortOutcome,
  triangleSource: NodeSortState['triangleSource'],
  geometry: THREE.InstancedBufferGeometry
): (uploaded: boolean) => string {
  if (outcome.kind === 'device') return (applied) => (applied ? 'GPU sort applied' : 'GPU sort');
  const n = outcome.ordering.length;
  const bytes = triangleSource ? n * 3 * (geometry.index?.array.BYTES_PER_ELEMENT ?? 4) : n * 4;
  return (uploaded) => formatOrderingBytes(bytes, uploaded);
}

/**
 * The staged ordering's lifecycle hooks: the coordinator's owner hooks on
 * every ordering (cancel-sweep tag, slice-ack frame wake-up, capture bypass),
 * plus the profiler lifecycle when a session is open. `onApplied`
 * (post-render acknowledgement) ends the pass as applied; `onAbandoned`
 * (superseded/cancelled/demoted/released/disposed) ends it as scheduled.
 * Neither writer invokes either unless it ACCEPTS the ordering (issue #713).
 */
function sortApplyCallbacks(
  c: CoordinatorState,
  pass: SortPass,
  outcome: SortOutcome,
  info: (applied: boolean) => string
): SortedIndexApplyCallbacks {
  const hooks = orderingApplyHooks(c);
  const session = pass.session;
  if (!session) return hooks;
  const resolvedAt = performance.now();
  return {
    ...hooks,
    onApplied: () => {
      const appliedAt = performance.now();
      session.setMetadata({ applyMs: Math.max(0, appliedAt - resolvedAt), info: info(true) });
      // The dedicated MONOTONIC completion stream records only orderings that
      // actually became drawable — not merely results that were staged and
      // later abandoned. It uses the dispatch-time profiler/generation,
      // matching the session's reset-isolation contract (issue #711).
      const { profiler, profilerGeneration } = pass;
      if (profiler && profiler._currentGeneration() === profilerGeneration) {
        profiler.recordDepthSortCompletion({
          lastMs: Math.max(0, appliedAt - pass.dispatchedAt),
          ...outcome.timing,
          elements: outcomeLength(outcome),
        });
      }
      session.end();
    },
    onAbandoned: () => session.end(),
  };
}

/**
 * A sort that threw or rejected (a worker RPC failure or timeout, or a throw
 * from a post-handoff step of the settle). Clears `inFlight`, releases a held
 * draw for the current population, and drains a queued re-sort.
 */
function failSort(c: CoordinatorState, job: SortJob, pass: SortPass, error: unknown): void {
  // Only close the pass for a genuine sort failure — NOT when the ordering was
  // already handed to the apply callbacks, which own the close (#713).
  if (!pass.handedOff) pass.session?.end();
  const current = c.nodeStates.get(job.nodeId);
  const who = pass.route === 'worker' ? 'SortWorker' : 'GPU';
  log.error(Modules.WORKER_POOL, `${who} sort failed for ${job.nodeId}`, error);
  if (!current) return;
  current.inFlight = false;
  if (current.generation === job.generation) releaseHeldDraw(c, job.mesh);
  // Drain a queued re-sort even on failure — a commit landed while this sort
  // was out, and dropping its request would leave the node stale until the NEXT
  // commit. Bounded: only a real commit sets resortQueued, so a persistently
  // failing producer cannot loop. Same demotion guard as the settle: a cleared
  // stamp means the re-promotion commit will schedule the sort that matters.
  if (current.resortQueued) {
    current.resortQueued = false;
    if (hasCommittedData(job.mesh)) scheduleSort(c, job.mesh, job.nodeId);
  }
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
function pumpChunkedOrderingApplies(c: CoordinatorState): void {
  for (const [nodeId, state] of c.nodeStates) {
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
    syncSortedIndexSlot(c, state.mesh);

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
      syncSortedIndexSlot(c, state.mesh);
      c.requestRender?.();
    }
    if (more) {
      // A STALLED pump made no progress (the previous slice is still
      // unflushed — the mesh was not drawn since; issue #715). Do NOT
      // spin the on-demand loop on it: the node resumes one slice per
      // DRAWN frame, since each drawn frame's render consumes the prior
      // slice and any subsequent render — camera motion, commit,
      // visibility change — re-runs the pump, which then advances.
      if (!stalled) c.requestRender?.();
    } else if (state.resortQueued) {
      state.resortQueued = false;
      scheduleSort(c, state.mesh, nodeId);
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
 * in-flight sorts
 * (the resolve is at most a frame away), invisible/demoted meshes, and
 * nodes whose live mode is no longer order-dependent. A streaming
 * chunked apply does NOT skip — a fresher sort fills the inactive
 * buffer concurrently.
 *
 * Work is tiered by dependency: frame-state cleanup runs above every gate,
 * and the only thing the loader-busy signal gates is the starved-worker init
 * retry. Cross-node ordering and within-mesh re-sorts both run during loads.
 *
 * @returns true when this pass changed what the next render draws (a
 *   renderOrder or an ordering-buffer slot) — the render-on-change loop's
 *   per-frame callback contract. A chunked slice write or slot flip ALSO
 *   calls `requestRender` (the pump must keep the loop alive), so both
 *   routes agree.
 */
export function evaluateDepthSortPerFrame(c: CoordinatorState): boolean {
  c.syncSortElementsRemaining = Math.max(0, syncSortElementLimit());
  // Drop the previous frame's render-order state FIRST — before any
  // early-return — so a disposed/dataset-switched frame can't leave the
  // module-scoped rank memo holding stale partition-wrapper subtrees alive —
  // and hand this coordinator's display dims to its pass.
  beginRenderOrderFrame(c.getDisplayDims);
  // Chunked ordering applies advance BEFORE every early-return below:
  // they need neither a camera nor an idle loader (stalling them during
  // a load would postpone convergence to the newest complete ordering),
  // and a paused stream must always resume its bounded drain.
  pumpChunkedOrderingApplies(c);
  // Deliberately NOT gated on `api`: the cross-node renderOrder pass is
  // pure main-thread and must keep ordering meshes back-to-front even
  // when the SortWorker was never constructed (`api` stays null forever
  // after a constructor throw — e.g. a CSP-blocked worker script — the
  // documented degrade-to-unsorted-normal mode). The within-mesh
  // re-sort triggers are worker-dependent, but `scheduleSort` guards
  // both `api` and init readiness itself.
  if (c.nodeStates.size === 0) return takeDrawnStateChanged(c);
  const camera = c.getCamera?.();
  if (!camera) return takeDrawnStateChanged(c);
  const loadInProgress = c.isLoadInProgress?.() ?? false;
  // Keep worker retry behind the load gate: a starved init must not run while
  // a view-update sweep is in flight, since that sweep IS the main-thread
  // saturation that starved it. Everything else the retry needs to know
  // (something visible actually wants sorting, the backoff, an offline
  // capture) it checks itself.
  if (c.depthSortEnabled && !loadInProgress) maybeRetryStarvedWorkerInit(c);

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

  for (const [nodeId, state] of c.nodeStates) {
    const mesh = state.mesh;
    if (!isEffectivelyVisible(mesh)) continue;
    const mode = liveBlendingMode(mesh);
    const orderDependent = c.depthSortEnabled && isLiveOrderDependent(mode);
    // A commutative layer carrying an AUTHORED layer_order still takes part in
    // the cross-layer band ordering (`LAYER_ORDER_SPEC.md` D4): its order
    // against other commutative layers is a no-op (addition commutes), but its
    // position relative to `normal`/`volumetric` layers is exactly what the
    // level exists to state — and without this it would stay pinned at
    // renderOrder 0 and always draw first.
    //
    // ONE predicate drives both branches on purpose. Written as two conditions
    // (collect here, reset there) they can disagree, and a layer whose level is
    // removed at runtime would keep a stale positive rank forever — the same
    // class of bug `computeDrawOrder` already documents for opaque meshes after
    // a live blending switch.
    //
    // A refracting glass (`drawsAfterEmissive`, spec MESH_PHYSICAL_MATERIALS §3.4
    // Phase 3) ALWAYS takes a rank: the unranked emissive layers sit at
    // renderOrder 0, ranks start at 1, and "after them, within my band" is what
    // the `last` slot flag then expresses — no sentinel value could.
    const wantsRank =
      orderDependent || authoredLayerOrder(mesh) !== undefined || drawsAfterEmissive(mesh);
    if (!wantsRank) {
      // Neither order-dependent nor authored — clear any cross-part renderOrder
      // bias so it doesn't strand a stale ordering. The one exception is glass
      // (`drawsBeforeEmissive`): unranked meshes all sit at three's default 0 and
      // sort by depth among themselves, so -1 is what puts a transmissive mesh
      // ahead of the unranked emissive layers it shares band 0 with (and ahead of
      // every ranked group, which is what band 0 means).
      const target = drawsBeforeEmissive(mesh) ? -1 : 0;
      if (mesh.renderOrder !== target) {
        mesh.renderOrder = target;
        c.drawnStateChanged = true;
      }
      continue;
    }

    if (!viewComputed) {
      // Derived here from the live camera rather than read from the frame's
      // view snapshot: resortForCapture runs this pass straight after a pose
      // is set, outside the frame loop, where a snapshot would be the
      // previous frame's. (The dispatch re-derives fresh matrices in
      // scheduleSort either way.)
      camera.updateMatrixWorld();
      scratch.view.copy(camera.matrixWorld).invert();
      scratch.camPos.setFromMatrixPosition(camera.matrixWorld);
      if ((camera as THREE.OrthographicCamera).isOrthographicCamera) {
        // Parallel rays: which side of a BSP split plane is near depends on
        // the view direction alone, not on where the eye sits (the ortho
        // camera keeps the perspective pose's position, often on the far
        // side of a plane the view looks back across). An eye pushed
        // effectively to infinity behind the camera gives that answer through
        // the same eye-side test. camPos feeds only the BSP ranks.
        const e = camera.matrixWorld.elements;
        scratch.axis.set(e[8], e[9], e[10]).normalize();
        scratch.camPos.addScaledVector(scratch.axis, ORTHO_BSP_EYE_DISTANCE);
      }
      viewComputed = true;
    }

    scratch.mv.multiplyMatrices(scratch.view, mesh.matrixWorld);

    // === Cross-mesh (inter-node) back-to-front ordering: COLLECT ===
    // One order slot per surviving normal-mode mesh; the post-loop
    // assignGlobalRenderOrder() puts everything on ONE global integer
    // renderOrder scale (full rationale in
    // `depth-sort-coordinator/render-order.ts`).
    collectRenderOrderSlot(mesh, scratch.mv, scratch.camPos);

    // A commutative layer earns a cross-layer rank (above) but must never
    // request a within-mesh sort: its elements composite in any order, so the
    // worker round-trip would buy nothing. Bail before the dispatch block.
    if (!orderDependent) continue;

    // Everything below dispatches or evaluates a within-mesh worker sort.
    // NOT paused while the loader is busy. "The pending commit will sort
    // anyway" holds only for the nodes a pass actually re-commits, and only
    // once it lands: through a progressive-refinement drain (seconds on a
    // hosted scene) and through a view pass that leaves a node untouched,
    // the orbit would otherwise be answered only by rung commits — measured
    // at up to 176 deg of sort-axis lag. Racing a commit is safe: it bumps
    // the generation (the resolve drops the stale ordering) and the commit's
    // own sort queues behind this one via `resortQueued`.
    // Mode switches and late-worker re-registration deliberately invalidate
    // commit stamps while the existing geometry stays visible. Those meshes
    // still receive their cross-mesh rank above, but cannot dispatch a worker
    // sort until re-commit. LOD demotion is only a defensive peer case here:
    // its synchronous release removes the mesh from `nodeStates` first.
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
      if (state.registered) scheduleSort(c, mesh, nodeId);
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

    if (moved) scheduleSort(c, mesh, nodeId);
  }

  const ordered = assignGlobalRenderOrder();
  return takeDrawnStateChanged(c) || ordered;
}

/** Read and clear `drawnStateChanged`. */
function takeDrawnStateChanged(c: CoordinatorState): boolean {
  const changed = c.drawnStateChanged;
  c.drawnStateChanged = false;
  return changed;
}
