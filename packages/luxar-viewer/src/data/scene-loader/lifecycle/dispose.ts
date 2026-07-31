/**
 * SceneLoader dispose body — releases dataset-scoped resources:
 *
 *   - the dataset abort controller (so in-flight worker tasks settle
 *     immediately and the worker pool's signal reference is cleared),
 *   - every geometry loader registered with the LoaderRegistry,
 *   - the GPU buffer pool (without this, instanced buffer geometries
 *     leak hundreds of MB of GPU memory across dataset switches),
 *   - the L1/L2 caching store (awaited so L2 metadata flushes before
 *     the next loader constructs its caching store),
 *   - the L0 decompressed-chunk cache,
 *   - the predictive-prefetch baseline,
 *   - dataset-scoped custom colormap LUT textures, and
 *   - the data-loading monitor's loader-bound closures.
 *
 * Worker pool itself is NOT terminated here — it's bounded and lives
 * across dataset switches. `disposeWorkerPool()` in core/app.ts owns
 * that lifecycle at app shutdown.
 */

import { log, Modules } from '../../../utils/log';
import { disposeCustomColormapTextures } from '../../../rendering/colormap-textures';
import { getWorkerPool } from '../../../workers/worker-pool';
import type { LoaderRegistry } from '../loaders/loader-registry';
import type { ViewStateQueue } from '../view-state/view-state-queue';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { SliceCache } from '../../../cache/slice-cache';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';

/**
 * Container of nullable resources the SceneLoader owns. The orchestrator
 * passes its fields in, the helper does the disposal work, then writes
 * the cleared state back via the `clear*` callbacks.
 */
export interface DisposeCtx {
  datasetAbortController: AbortController | null;
  /** Live per-update controller (if an updateView is in flight); aborted so its reads bail. */
  updateAbortController: AbortController | null;
  registry: LoaderRegistry;
  gpuBufferPool: GPUBufferPool | null;
  cachingStore: MultiLevelCachingStore | null;
  l0Cache: DecompressedChunkCache | null;
  sliceCache: SliceCache | null;
  viewStateQueue: ViewStateQueue;
  monitor: SceneLoaderMonitorPort | null;
}

/**
 * Async dispose path — awaits the caching-store flush so dataset
 * switches see L2 fully drained before the next caching store is
 * constructed. Returns the cleared resources so the orchestrator can
 * null-out its corresponding fields.
 */
export async function disposeSceneLoader(ctx: DisposeCtx): Promise<{
  rootGroupCleared: true;
  sceneGraphCleared: true;
}> {
  // Abort the dataset-scoped signal first so any in-flight worker
  // `runWithTimeout` callers settle immediately instead of waiting
  // for their tasks to complete (WASM tasks themselves keep running
  // but their results are discarded). Clear the pool's reference
  // afterwards so future workers don't get an already-aborted
  // signal from this disposed loader.
  if (ctx.datasetAbortController) {
    ctx.datasetAbortController.abort();
  }
  // Also abort the live per-update controller (if an updateView is in flight)
  // so its chunk reads bail instead of resolving against a torn-down loader.
  if (ctx.updateAbortController) {
    ctx.updateAbortController.abort();
  }
  getWorkerPool().setAbortSignal(undefined);

  // Dispose all geometry loaders via registry
  ctx.registry.disposeAll();

  // Also drop failure tracking. `disposeAll` clears only the loader maps, but
  // `loadScene` supports same-instance reuse, and a stale failure record from
  // the previous dataset would otherwise be counted by the new load's outcome
  // report and retried against the new scene.
  ctx.registry.clearAllFailures();

  // dispose GPU buffer pool. Without this, the pool retains
  // active+pooled InstancedBufferGeometry references after a dataset
  // switch — at million-element scale this can leak hundreds of MB
  // of GPU memory until the page is refreshed. The pool's internal
  // dispose() is idempotent.
  if (ctx.gpuBufferPool) {
    try {
      ctx.gpuBufferPool.dispose();
    } catch (error) {
      log.warning(Modules.SCENE_LOADER, 'GPU buffer pool disposal failed', error);
    }
  }

  // Dispose caching store (flushes L2 metadata, clears L1).
  // Awaited so a dataset switch sees the previous L2 fully drained
  // before the next caching store is constructed.
  if (ctx.cachingStore) {
    try {
      await ctx.cachingStore.dispose();
    } catch (error) {
      log.warning(Modules.SCENE_LOADER, 'Caching store disposal failed', error);
    }
  }

  // Clear L0 decompressed chunk cache
  if (ctx.l0Cache) {
    const stats = ctx.l0Cache.getStats();
    log.info(
      Modules.SCENE_LOADER,
      `L0 cache stats at dispose: ${stats.count} chunks, ${(stats.size / 1024 / 1024).toFixed(1)}MB, ` +
        `hit rate: ${(stats.hitRate * 100).toFixed(1)}%`
    );
    ctx.l0Cache.clear();
  }

  // Clear the SliceCache (S-cache) so a reused SceneLoader / dataset switch
  // never serves decoded geometry from the previous dataset.
  if (ctx.sliceCache) {
    const s = ctx.sliceCache.getStats();
    log.info(
      Modules.SCENE_LOADER,
      `SliceCache stats at dispose: ${s.count} slices, ${(s.size / 1024 / 1024).toFixed(1)}MB, ` +
        `hit rate: ${(s.hitRate * 100).toFixed(1)}%`
    );
    ctx.sliceCache.clear();
  }

  // S6: clear per-loader prefetch predictor state on dispose so a
  // reused SceneLoader doesn't extrapolate from a prior dataset.
  ctx.viewStateQueue.clearPrev();

  // Dispose dataset-scoped custom colormap LUTs. The custom-LUT cache
  // is keyed by content hash and shared across all scenes, but entries
  // from an unloaded dataset have no value and would accumulate in a
  // long-lived app that swaps many unique LUTs. Built-ins survive
  // because they're shared with all scenes and cheap to keep.
  try {
    disposeCustomColormapTextures();
  } catch (error) {
    log.warning(Modules.SCENE_LOADER, 'Custom colormap disposal failed', error);
  }

  // Tell the monitor to drop its scene-loader-bound closures (cache
  // stats, L0 cache, GPU buffer pool, accumulators, profiler) before
  // we release our reference. Without this, the monitor outlives the
  // loader with closures that capture our nulled-out fields and NPE
  // on the next stats poll. The monitor's lifecycle itself is owned
  // by core/app.ts via DataMonitorManager.
  ctx.monitor?.disconnectAllLoaders();

  return { rootGroupCleared: true, sceneGraphCleared: true } as const;
}
