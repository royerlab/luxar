/**
 * `__luxarDebug.getPerf()` — one read-only snapshot of everything a
 * performance probe needs, available from BOOTSTRAP (before the dataset
 * loads) and enriched once the runtime components exist.
 *
 * Why a separate surface from `getState()`: `getState` is installed only
 * after the dataset load and the blend warm-up settle, so a probe waiting
 * for it under-observes the load; and `getState().isLoading` has a narrow,
 * E2E-relied-upon meaning (a load PASS is outstanding) that must not be
 * widened. `isSettled` here is the wide predicate — no update pass, no
 * refinement drain, no lazy LOD promotion in flight, and the post-load
 * refinement ran to completion.
 *
 * @module core/app/debug/perf-snapshot
 */

import { getLoadTimeline, type LoadTimelineSnapshot } from '../../../profiling/load-timeline';
import type { AdaptiveDPRDiagnostics } from '../../../rendering/adaptive-dpr-manager';
import type { RendererInfoSnapshot } from './renderer-info-sampler';

/** Optional runtime hooks; every field is absent before `init()` finishes. */
export interface PerfSnapshotContext {
  rendererInfo?: () => RendererInfoSnapshot | null;
  adaptiveDpr?: () => AdaptiveDPRDiagnostics;
  workers?: () => unknown;
  /** `SceneLoader.isUpdateInProgress()` — true during passes AND the refinement drain. */
  isUpdateInProgress?: () => boolean;
  /** `SceneLoaderManager.isAnyLoadPassInProgress()` — the `getState().isLoading` predicate. */
  isAnyLoadPassInProgress?: () => boolean;
  /** Any lazy substitutive-LOD / deferred-partition level fetch in flight. */
  isAnyLodLevelLoading?: () => boolean;
  /** Per-node projected density (`scene/projected-density.ts` snapshot). */
  density?: () => unknown;
  /** Multi-level cache stats (`SceneLoader.getCacheStats()`; includes the L2 write queue). */
  cache?: () => unknown;
  /** Blend-mode program warm-up counters (`rendering/webgl-blend-warmup.ts`). */
  blendWarmup?: () => unknown;
}

export interface PerfSnapshot {
  /** Always true — lets a probe distinguish "getPerf exists" from "runtime ready". */
  perfReady: true;
  /** True once the runtime hooks (renderer, loaders, DPR) are wired. */
  runtimeReady: boolean;
  timeline: LoadTimelineSnapshot;
  rendererInfo: RendererInfoSnapshot | null;
  adaptiveDpr: AdaptiveDPRDiagnostics | null;
  workers: unknown;
  /** Per-node projected density records, keyed by scene path (null before init). */
  density: unknown;
  /** Cache tier stats incl. `l2WriteQueue.{pending,inFlight,dropped}` (null before init). */
  cache: unknown;
  /** Warm-up counters: variants queued / compiled / deduped (null before init). */
  blendWarmup: unknown;
  /**
   * The wide "nothing is in flight" predicate, or `null` before the runtime
   * hooks exist. See the module doc for what it covers.
   */
  isSettled: boolean | null;
  /** Component predicates behind `isSettled`, for diagnosing what is still busy. */
  settle: {
    updateInProgress: boolean | null;
    loadPassInProgress: boolean | null;
    lodLevelLoading: boolean | null;
    refinementComplete: boolean;
  };
}

function read(fn: (() => boolean) | undefined): boolean | null {
  if (!fn) return null;
  try {
    return fn();
  } catch {
    return null;
  }
}

function readOrNull<T>(fn: (() => T) | undefined): T | null {
  if (!fn) return null;
  try {
    return fn();
  } catch {
    return null;
  }
}

/** Build the snapshot. Never throws: a failing hook reads as `null`. */
export function computePerfSnapshot(ctx: PerfSnapshotContext = {}): PerfSnapshot {
  const timeline = getLoadTimeline();
  const updateInProgress = read(ctx.isUpdateInProgress);
  const loadPassInProgress = read(ctx.isAnyLoadPassInProgress);
  const lodLevelLoading = read(ctx.isAnyLodLevelLoading);
  const runtimeReady = ctx.isUpdateInProgress !== undefined;
  const refinementComplete = timeline.refinement.complete;
  const isSettled = runtimeReady
    ? updateInProgress === false &&
      loadPassInProgress !== true &&
      lodLevelLoading !== true &&
      refinementComplete
    : null;
  return {
    perfReady: true,
    runtimeReady,
    timeline,
    rendererInfo: readOrNull(ctx.rendererInfo),
    adaptiveDpr: readOrNull(ctx.adaptiveDpr),
    workers: readOrNull(ctx.workers),
    density: readOrNull(ctx.density),
    cache: readOrNull(ctx.cache),
    blendWarmup: readOrNull(ctx.blendWarmup),
    isSettled,
    settle: { updateInProgress, loadPassInProgress, lodLevelLoading, refinementComplete },
  };
}
