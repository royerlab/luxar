/**
 * Retry path for failed loaders — partial-scene-resilience recovery.
 *
 * Exposes lock-free helpers (`retryFailedLoaderUnlocked`,
 * `retryAllFailedLoadersUnlocked`) that re-trigger the data fetch +
 * commit for paths that recorded a failure. The orchestrator (SceneLoader)
 * wraps each call in its `_updateInProgress` lock dance — these helpers
 * never touch that lock so retry-from-inside-retry doesn't deadlock.
 *
 * Each retry repeats the same query-shape logic the main update path
 * uses (deriveNodeViewState + the per-type viewState fallback for
 * extend_to_all skip) so retry / update / initial-load are never out
 * of sync.
 */

import type * as THREE from 'three';
import type { LoadedPointsData, ViewState } from '../../data-loader-types';
import type { LinesDataLoader, LinesViewState } from '../../../types/lines';
import type { GSplatsDataLoader, GSplatsViewState } from '../../../types/gsplats';
import { log, Modules } from '../../../utils/log';
import type { LoaderRegistry } from '../loaders/loader-registry';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { GEOMETRY_DESCRIPTORS } from '../geometry-descriptors';
import type { StagedLinesCommit } from '../process/data-processor-lines';
import type { StagedPointsCommit } from '../process/data-processor-points';
import type { StagedGSplatsCommit } from '../process/data-processor-gsplats';

/**
 * Result of `deriveNodeViewState` — `skip` is true when the helper
 * decides the node should fall back to the base view state (e.g. full
 * extend_to_all coverage where deriving would empty the query).
 */
type DerivedViewState = { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };

/**
 * Narrow context the retry helpers need from the orchestrator. Keeps
 * the helper independent of SceneLoader internals.
 */
export interface RetryCtx {
  registry: LoaderRegistry;
  /**
   * Optional LOD-group registry for the lazy-level fallback: lazy
   * substitutive levels never join the loader maps (they're registry-driven
   * — see load-lod-group-node.ts), so when a failed path resolves to no
   * map entry the retry re-kicks the level's ``ensureLoaded`` through
   * ``retryLazyChildByLeafPath`` instead of discarding the failure.
   * Optional so headless/test ctxs without a registry keep working.
   */
  lodGroupRegistry?: LODGroupRegistry | null;
  rootGroup: THREE.Group | null;
  viewState: ViewState;
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: { applyPartialExtendTolerance: boolean }
  ): DerivedViewState;
  processPointsData(path: string, data: LoadedPointsData): StagedPointsCommit;
  commitPointsGeometry(staged: StagedPointsCommit): void;
  processLinesData(
    path: string,
    data: Awaited<ReturnType<LinesDataLoader['updateView']>>,
    viewState: LinesViewState
  ): Promise<StagedLinesCommit | null>;
  commitLinesGeometry(staged: StagedLinesCommit): void;
  processGSplatsData(
    path: string,
    data: Awaited<ReturnType<GSplatsDataLoader['updateView']>>,
    viewState: GSplatsViewState
  ): Promise<StagedGSplatsCommit | null>;
  commitGSplatsGeometry(staged: StagedGSplatsCommit): void;
}

/**
 * Retry a single failed loader without touching the orchestrator's
 * update lock. Returns true on success (failure cleared from the
 * registry — except for a lazy LOD level, where true means "retry
 * kicked" and the record is kept until the thunk settles), false on
 * continued failure (registry updated with new retry count), or false
 * if the path is no longer in failed-loaders.
 */
export async function retryFailedLoaderUnlocked(path: string, ctx: RetryCtx): Promise<boolean> {
  const { registry } = ctx;
  if (!registry.failedLoaders.has(path)) return false;

  log.info(Modules.SCENE_LOADER, `Retrying failed loader: ${path}`);

  try {
    // Look up the per-node attrs so retry applies the same
    // extend_to_all / nd_transform adjustments as the main update path.
    // Passing a raw view state here silently renders an incorrect query
    // region for transformed or extended nodes.
    const obj = ctx.rootGroup?.getObjectByName(path) as
      THREE.Object3D | THREE.Mesh | THREE.Points | undefined;
    const attrs = obj?.userData?.attrs as { extend_to_all?: string[] } | undefined;

    // Defensive guard — only clear `failedLoaders` if the named object
    // still exists in the scene. The placeholder model should make
    // commit always succeed when retry runs in normal conditions, but
    // a scene reload or programmatic node removal between failure and
    // retry could leave us fetching data that has nowhere to land.
    // Without this guard, retry would falsely report success ("data
    // fetched + commit silently no-op'd") and clear the failure,
    // hiding the broken state from `hasFailures()`.
    const verifyAndClear = (kind: string): boolean => {
      if (!ctx.rootGroup?.getObjectByName(path)) {
        log.warning(
          Modules.SCENE_LOADER,
          `Retry of ${path} fetched data but no scene object exists; not clearing failure`
        );
        return false;
      }
      registry.failedLoaders.delete(path);
      log.success(Modules.SCENE_LOADER, `Successfully retried ${kind} loader: ${path}`);
      return true;
    };

    const kind = registry.getLoaderType(path);
    if (kind) {
      const descriptor = GEOMETRY_DESCRIPTORS[kind];
      const derived = ctx.deriveNodeViewState(path, attrs, {
        applyPartialExtendTolerance: descriptor.applyPartialExtendTolerance,
      });
      // Mirror the initial-load fallback: when `derived.skip` is true
      // (extend_to_all fully covers), the placeholder still needs data
      // committed — skipping the load and clearing failedLoaders would
      // falsely report success against an empty placeholder.
      const viewState = derived.skip ? ctx.viewState : derived.viewState;
      await descriptor.retryCommit(ctx, path, registry.loadersOf(kind).get(path)!, viewState);
      return verifyAndClear(kind);
    } else {
      // Not in the sweep maps. Lazy substitutive LOD levels are never
      // registered there (registry-driven lifecycle) but DO record failures;
      // without this fallback their records would be discarded below and
      // lazy levels would be unretryable through this API. Re-kick the
      // level's deferred loader instead.
      //
      // Fire-and-forget semantics: `true` means "retry started" (the thunk
      // owns the ready/failed outcome). KEEP the failure record across the
      // kick — deleting it here reset `autoRetryCount` to 0 on the next
      // `recordFailure`, so `MAX_AUTO_RETRY_ATTEMPTS` never bound a lazy
      // level and a permanently-failing one (e.g. a 404 that classifies
      // `Network`) was re-kicked on every `online` transition forever. On
      // success the lazy loader clears the record
      // (load-{points,lines,gsplats}-node.ts); on a repeat failure
      // `recordFailure` preserves the accumulated counter.
      if (ctx.lodGroupRegistry?.retryLazyChildByLeafPath(path)) {
        log.info(Modules.SCENE_LOADER, `Retry kicked for lazy LOD level: ${path}`);
        return true;
      }
      // Loader not found - it may have been disposed
      log.warning(Modules.SCENE_LOADER, `No loader found for path: ${path}`);
      registry.failedLoaders.delete(path); // Clean up stale entry
      return false;
    }
  } catch (error) {
    // Update error tracking with the new attempt. Routed through
    // `recordFailure` (rather than an inline `.set`) so the classified `kind` is
    // refreshed and the counter has one owner — this call site used to baseline
    // `retryCount` at 1 while the update sweep baselined it at 0.
    registry.recordFailure(path, error as Error);
    const retryCount = registry.failedLoaders.get(path)?.retryCount ?? 0;
    log.error(
      Modules.SCENE_LOADER,
      `Retry failed for ${path} (attempt ${retryCount}): ${(error as Error).message}`
    );
    return false;
  }
}

/**
 * Retry the supplied set of failed paths in parallel without touching
 * the orchestrator's update lock. Caller is expected to hold the lock
 * (parity with the single-path helper above). Returns the path split
 * into succeeded / still-failing buckets.
 */
export async function retryAllFailedLoadersUnlocked(
  failedPaths: string[],
  ctx: RetryCtx
): Promise<{ succeeded: string[]; failed: string[] }> {
  const succeeded: string[] = [];
  const failed: string[] = [];

  const results = await Promise.all(
    failedPaths.map(async (path) => {
      const success = await retryFailedLoaderUnlocked(path, ctx);
      return { path, success };
    })
  );

  for (const { path, success } of results) {
    if (success) {
      succeeded.push(path);
    } else {
      failed.push(path);
    }
  }

  return { succeeded, failed };
}
