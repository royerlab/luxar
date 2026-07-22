/**
 * Material-manager diagnostics: cache-size + eviction + create-time
 * snapshot.
 *
 * Counters stay on MaterialManager because eviction and cache-miss paths
 * mutate them; this module builds the returned object shape for
 * `getCacheStats()`.
 *
 * @module rendering/material-manager/stats
 */

import { config } from '../../config';

/** Snapshot view of the orchestrator state needed to build stats. */
export interface StatsCtx {
  readonly pointMaterialCache: Map<string, unknown>;
  readonly lineMaterialCache: Map<string, unknown>;
  readonly gsplatMaterialCache: Map<string, unknown>;
  readonly ownedMaterials: Set<unknown>;
  readonly registeredMaterials: Set<unknown>;
  readonly evictionCount: number;
  readonly totalCreateMs: number;
  readonly createCount: number;
}

/** Build a stats snapshot for diagnostics / tests. */
export function getCacheStats(ctx: StatsCtx) {
  return {
    pointMaterials: ctx.pointMaterialCache.size,
    lineMaterials: ctx.lineMaterialCache.size,
    gsplatMaterials: ctx.gsplatMaterialCache.size,
    ownedMaterials: ctx.ownedMaterials.size,
    cachedMaterials:
      ctx.pointMaterialCache.size + ctx.lineMaterialCache.size + ctx.gsplatMaterialCache.size,
    totalRegistered: ctx.registeredMaterials.size,
    /** Cumulative LRU evictions since creation (only the line cache evicts). */
    evictions: ctx.evictionCount,
    /** Configured cache bound (`0` = disabled). */
    maxSize: config.dataLoading.performance.materialCacheMaxSize,
    /**
     * Cumulative wall-clock ms spent inside `new XMaterial(...)`
     * calls (cache-miss path). Excludes WebGL program compilation,
     * which happens lazily on first render.
     */
    totalCreateMs: ctx.totalCreateMs,
    /** Number of `new XMaterial(...)` calls (cache misses). */
    createCount: ctx.createCount,
    keys: [
      ...Array.from(ctx.pointMaterialCache.keys()),
      ...Array.from(ctx.lineMaterialCache.keys()),
      ...Array.from(ctx.gsplatMaterialCache.keys()),
    ],
  };
}
