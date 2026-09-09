/**
 * The adaptive-DPR controller's "data is loading" predicate.
 *
 * While the viewer is not settled, frame times measure decode/upload/commit
 * work, not steady-state render cost, so the controller must not learn from
 * them (no probes armed or settled, no refresh-cap samples, no floor
 * evidence). The controller used to read only `isUpdateInProgress()` — the
 * update-view lock — which excludes the post-load progressive refinement
 * drain (each rung is a fetch + decode + commit with the lock RELEASED between
 * passes), lazy `lod_group` level loads, held partition rising-edge resyncs on
 * the default loader, and load passes on non-default loaders. The 2026-09 audit
 * caught the controller crediting the end of a refinement drain to a DPR step
 * it had just taken.
 *
 * This is the same settledness the perf probes read through
 * `__luxarDebug.getPerf().isSettled` (`debug/perf-snapshot.ts`), inverted.
 *
 * @module core/app/init/load-activity
 */

import { getLoadTimeline } from '../../../profiling/load-timeline';

export interface LoadActivityDeps {
  /** The default loader's update-view lock. */
  isUpdateInProgress(): boolean;
  /** Any loader's load pass (SceneLoaderManager.isAnyLoadPassInProgress). */
  isAnyLoadPassInProgress(): boolean;
  /** Any lazy LOD-group level currently loading. */
  isAnyLodLevelLoading(): boolean;
  /** A visible partition rising edge is waiting to be handed to the loader. */
  hasVisiblePendingPartitionResync(): boolean;
  /** The current load's refinement drain has run to completion. */
  isRefinementComplete(): boolean;
}

/** True while ANY loading activity makes frame samples unrepresentative. */
export function isLoadActivity(deps: LoadActivityDeps): boolean {
  return (
    deps.isUpdateInProgress() ||
    deps.isAnyLoadPassInProgress() ||
    deps.isAnyLodLevelLoading() ||
    deps.hasVisiblePendingPartitionResync() ||
    !deps.isRefinementComplete()
  );
}

/** Default `isRefinementComplete` source: the load timeline's counter. */
export function refinementCompleteFromTimeline(): boolean {
  return getLoadTimeline().refinement.complete;
}

/** The live loader handles the predicate reads each tick (a dataset switch replaces them). */
export interface LoadActivitySources {
  getDefaultLoader(): {
    isUpdateInProgress(): boolean;
    lodGroupRegistry?: {
      isAnyLevelLoading(): boolean;
      hasVisiblePendingPartitionResync(): boolean;
    } | null;
  } | null;
  isAnyLoadPassInProgress(): boolean;
}

/**
 * The production predicate: every source is read at call time through the
 * loader manager, never captured, so it follows dataset switches.
 */
export function buildLoadActivityPredicate(sources: LoadActivitySources): () => boolean {
  return () =>
    isLoadActivity({
      isUpdateInProgress: () => sources.getDefaultLoader()?.isUpdateInProgress() ?? false,
      isAnyLoadPassInProgress: () => sources.isAnyLoadPassInProgress(),
      isAnyLodLevelLoading: () =>
        sources.getDefaultLoader()?.lodGroupRegistry?.isAnyLevelLoading() === true,
      hasVisiblePendingPartitionResync: () =>
        sources.getDefaultLoader()?.lodGroupRegistry?.hasVisiblePendingPartitionResync() === true,
      isRefinementComplete: refinementCompleteFromTimeline,
    });
}
