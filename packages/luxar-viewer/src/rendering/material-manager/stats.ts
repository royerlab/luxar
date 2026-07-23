/**
 * Material-manager diagnostics: registry-size + create-time snapshot.
 *
 * Counters stay on MaterialManager because the creation paths mutate
 * them; this module builds the returned object shape for
 * `getCacheStats()`.
 *
 * @module rendering/material-manager/stats
 */

/** Snapshot view of the orchestrator state needed to build stats. */
export interface StatsCtx {
  readonly pointMaterialCache: Map<string, unknown>;
  readonly lineMaterialCache: Map<string, unknown>;
  readonly gsplatMaterialCache: Map<string, unknown>;
  readonly ownedMaterials: Set<unknown>;
  readonly registeredMaterials: Set<unknown>;
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
    /**
     * Cumulative LRU evictions — always 0 since the line-material LRU
     * (the last cached kind) died with the lines texture-storage
     * migration; kept so stats consumers don't break.
     */
    evictions: 0,
    /**
     * Cumulative wall-clock ms spent inside `new XMaterial(...)`
     * calls. Excludes WebGL program compilation, which happens lazily
     * on first render.
     */
    totalCreateMs: ctx.totalCreateMs,
    /** Number of `new XMaterial(...)` calls. */
    createCount: ctx.createCount,
    keys: [
      ...Array.from(ctx.pointMaterialCache.keys()),
      ...Array.from(ctx.lineMaterialCache.keys()),
      ...Array.from(ctx.gsplatMaterialCache.keys()),
    ],
  };
}
