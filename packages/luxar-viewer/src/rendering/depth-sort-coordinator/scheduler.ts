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
import { withTimeout } from '../../workers/worker-pool/timeout/with-timeout';
import { log, Modules } from '../../utils/log';
import { isEffectivelyVisible } from '../../utils/object-visibility';
import {
  isLiveOrderDependent,
  liveBlendingMode,
  orderingApplyHooks,
  releaseHeldDraw,
  syncSortElementLimit,
  workerHost,
  type CoordinatorState,
} from './state';
import {
  computeModelView,
  formatOrderingBytes,
  orderingCoversCommit,
  recordSortPose,
  syncSortedIndexSlot,
} from './ordering-apply';
import { SORT_RPC_TIMEOUT_MS, maybeRetryStarvedWorkerInit } from './worker-lifecycle';

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
 */
export function scheduleSort(c: CoordinatorState, mesh: THREE.Mesh, nodeId: string): void {
  const state = c.nodeStates.get(nodeId);
  const camera = c.getCamera?.();
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
  if (!state || !workerHost.api || workerHost.workerInitState !== 'ready' || !camera) {
    // Nothing will sort this population, so a held append draw must not wait.
    if (state) releaseHeldDraw(c, mesh);
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

  const generation = state.generation;
  // Fresh matrices even between frames (see computeModelView).
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
  const profiler = c.getProfiler?.() ?? null;
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
    workerHost.api.sort({ nodeId, generation, modelView: new Float32Array(modelView.elements) }),
    SORT_RPC_TIMEOUT_MS
  )
    .then((result) => {
      const roundTripMs = performance.now() - dispatchedAt;
      const current = c.nodeStates.get(nodeId);
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
      if (
        result &&
        result.generation === current.generation &&
        stillCommitted &&
        orderingCoversCommit(result.ordering, current, nodeId)
      ) {
        const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
        // Which apply path this node uses. A retained `triangleSource` is
        // the mesh signal and it is set by the SAME commit whose generation
        // just matched, so the two can never describe different face sets.
        const triangleSource = current.triangleSource;
        const applicable =
          triangleSource !== undefined || !!geometry?.getAttribute?.('aSortedIndex');
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
          // The owner hooks ride every staged ordering (cancel-sweep tag,
          // slice-ack frame wake-up, capture bypass); the profiler lifecycle
          // only when a session is open.
          const hooks = orderingApplyHooks(c);
          const applyCallbacks: SortedIndexApplyCallbacks = session
            ? {
                ...hooks,
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
            : hooks;
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
            c.requestRender?.();
          }
        }
      }
      // No ordering staged (stale generation, demotion, rejected write, or
      // missing attribute): the pass is just the dispatch→resolve
      // round-trip — close it now (issue #713).
      if (!applyOwnsSession) session?.end();
      // A sort for the CURRENT population that staged nothing leaves no
      // ordering for a held append draw to wait for. (A stale generation's
      // does not: the newer commit's own sort is queued behind this one.)
      if (!applyOwnsSession && current.generation === generation) releaseHeldDraw(c, mesh);

      if (current.resortQueued) {
        current.resortQueued = false;
        // Demoted mid-sort (stamp cleared): the queued request describes a
        // population the geometry no longer holds, and re-promotion always
        // re-commits — which schedules the sort it actually needs. Dispatching
        // here would burn worker time on an ordering the resolve path is
        // guaranteed to discard.
        if (stillCommitted) scheduleSort(c, mesh, nodeId);
      }
    })
    .catch((error) => {
      // Only close the pass here for a genuine RPC failure — NOT when the
      // ordering was already handed to the chunked-apply callbacks and a
      // post-handoff step (e.g. requestRender) threw: those callbacks own
      // the close, and ending here would mislabel the applied sample (#713).
      if (!applyOwnsSession) session?.end();
      const current = c.nodeStates.get(nodeId);
      log.error(Modules.WORKER_POOL, `SortWorker sort failed for ${nodeId}`, error);
      if (!current) return;
      current.inFlight = false;
      if (current.generation === generation) releaseHeldDraw(c, mesh);
      // Drain a queued re-sort even on failure — a commit landed while
      // this sort was out, and dropping its request would leave the node
      // stale until the NEXT commit. Bounded: only a real commit sets
      // resortQueued, so a persistently failing worker cannot loop. Same
      // demotion guard as the resolve path: a cleared stamp means the
      // re-promotion commit will schedule the sort that matters.
      if (current.resortQueued) {
        current.resortQueued = false;
        if (hasCommittedData(mesh)) scheduleSort(c, mesh, nodeId);
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
