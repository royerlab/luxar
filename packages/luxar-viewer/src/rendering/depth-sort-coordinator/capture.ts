/**
 * The depth-sort coordinator's offline-capture drain: {@link resortForCapture}
 * forces a pose-fresh sort of every order-dependent node and drives the
 * scheduler, the worker round-trip and the chunked apply by hand until the
 * drawn ordering is the one for the current pose.
 *
 * @module rendering/depth-sort-coordinator/capture
 */

import type * as THREE from 'three';
import { hasPendingSortedIndexOrderingApply } from '../element-storage';
import { hasCommittedData } from '../../types/committed-data';
import { isEffectivelyVisible } from '../../utils/object-visibility';
import { isLiveOrderDependent, liveBlendingMode, workerHost, type CoordinatorState } from './state';
import { evaluateDepthSortPerFrame, scheduleSort } from './scheduler';

/**
 * True when the depth-sort subsystem has settled: no tracked node has a
 * sort RPC outstanding (`inFlight`), a re-sort queued behind one
 * (`resortQueued`), or a chunked ordering apply still streaming into its
 * inactive buffer (`hasPendingSortedIndexOrderingApply`). This is the
 * termination condition {@link resortForCapture} drains toward — while
 * ANY of those hold, the drawn permutation is not yet the pose-fresh one.
 * Module-private: the drain loop is the only caller.
 */
function isCaptureQuiescent(c: CoordinatorState): boolean {
  for (const state of c.nodeStates.values()) {
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
 * AnimationController per-frame callback, and re-sorts only past its
 * motion thresholds. Two offline captures need an ordering exact for each
 * frame's pose instead:
 * - the gallery orbit-video pass (`__luxarDebug`) STOPS the rAF loop and
 *   renders each frame synchronously, so the scheduler never fires at all;
 * - the recording panel's frame-by-frame capture
 *   (`ui/recording-panel/offline-capture-frame-loop.ts`) keeps the loop
 *   running but suppresses its own render, and steps the turntable by less
 *   than the angle threshold, so the scheduler fires too rarely.
 * Either way an order-dependent node (`normal` / `volumetric`) would be
 * filmed with a permutation from an earlier pose. This helper drives the
 * scheduler + worker round-trip + chunked apply by hand so each captured
 * frame is ordered for its own pose.
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
export async function resortForCapture(c: CoordinatorState, maxWaitMs = 3000): Promise<void> {
  // `requestRender` is wired to `animationController.requestRender('depthSort')`
  // (core/app/init/pipeline.ts), and the sort resolve/pump paths call
  // `requestRender?.()`. Suppress it for the duration so draining (which we
  // drive ourselves) can't silently re-arm a loop the gallery capture
  // stopped on purpose. Safe offline: nothing but the capture's own pass
  // draws (the panel capture suppresses the loop's render), so there is no
  // frame request to lose. The suppression is depth-counted / reentrancy-safe: this
  // helper is exposed on `__luxarDebug`, so an overlapping (nested) call
  // could otherwise snapshot `null` and restore `null` permanently, wedging
  // the render loop forever. Only the OUTERMOST call snapshots and restores.
  //
  // A non-zero depth ALSO lifts the #715 upload back-pressure for this
  // coordinator's streams (`orderingApplyHooks`). The drain below never draws,
  // but on the chunked (WebGL) path a multi-slice apply stalls after its first
  // slice until a DRAW's upload ack releases it — so without the bypass any
  // order-dependent node past one slice (>1M elements, e.g. the 3M-star gaia
  // demo) could never reach quiescence: every captured frame would burn the
  // full maxWaitMs and still film a stale ordering. Offline, folding the slices
  // into one upload on the capture's own render is exactly acceptable (the
  // union range stays contiguous and current). Scoped to THIS coordinator: a
  // second host's interactive streams on the same page keep their bound.
  if (c.captureSuppressDepth === 0) c.requestRenderBeforeCapture = c.requestRender;
  c.captureSuppressDepth++;
  c.requestRender = null;
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
    if (c.depthSortEnabled && c.getCamera?.() && workerHost.api) {
      for (const [nodeId, state] of c.nodeStates) {
        const mesh = state.mesh;
        if (!isEffectivelyVisible(mesh)) continue;
        if (!hasCommittedData(mesh)) continue;
        if (!isLiveOrderDependent(liveBlendingMode(mesh))) continue;
        scheduleSort(c, mesh, nodeId);
      }
    }

    // Cross-node renderOrder pass + pump any pending chunked applies, all
    // for the current pose (its re-sort trigger skips the in-flight nodes
    // the force loop just dispatched).
    evaluateDepthSortPerFrame(c);

    // Nothing to wait for (disabled / no order-dependent node / worker
    // unavailable): return before opening the drain loop.
    if (isCaptureQuiescent(c)) return;

    // Drain to quiescence, bounded by maxWaitMs. Each iteration yields a
    // macrotask (setTimeout(0)) so worker resolutions land, then pumps one
    // chunked-apply slice and re-asserts renderOrder via
    // evaluateDepthSortPerFrame.
    const start = performance.now();
    while (performance.now() - start < maxWaitMs) {
      await new Promise<void>((r) => setTimeout(r, 0));
      evaluateDepthSortPerFrame(c);
      if (isCaptureQuiescent(c)) return;
    }
  } finally {
    // Clamped, not a bare decrement: dispose resets the depth to 0, and a
    // capture that was in flight across that dispose must not drive it
    // negative (a later capture would then decrement back to a non-zero exit
    // and strand `requestRender` at null forever).
    c.captureSuppressDepth = Math.max(0, c.captureSuppressDepth - 1);
    if (c.captureSuppressDepth === 0) {
      c.requestRender = c.requestRenderBeforeCapture;
      // Drop the snapshot so it can't pin the app's closure between
      // captures (mirrors the dispose-path hygiene).
      c.requestRenderBeforeCapture = null;
    }
  }
}
