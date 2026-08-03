/**
 * Material-manager diagnostics: registry-size + create-time snapshot.
 *
 * Counters stay on MaterialManager because the creation paths mutate
 * them; this module builds the returned object shape for
 * `getCacheStats()`.
 *
 * Nothing here reports a cache size. Every material is per-node — each carries
 * its own texture uniform, so two nodes can never share one — which means there
 * is no material cache left to measure. See `getPointMaterial` for the full
 * reasoning.
 *
 * @module rendering/material-manager/stats
 */

/** Snapshot view of the orchestrator state needed to build stats. */
export interface StatsCtx {
  readonly ownedMaterials: Set<unknown>;
  readonly registeredMaterials: Set<unknown>;
  readonly totalCreateMs: number;
  readonly createCount: number;
}

/** Build a stats snapshot for diagnostics / tests. */
export function getCacheStats(ctx: StatsCtx) {
  return {
    ownedMaterials: ctx.ownedMaterials.size,
    totalRegistered: ctx.registeredMaterials.size,
    /**
     * Cumulative wall-clock ms spent inside `new XMaterial(...)`
     * calls. Excludes WebGL program compilation, which happens lazily
     * on first render.
     */
    totalCreateMs: ctx.totalCreateMs,
    /** Number of `new XMaterial(...)` calls. */
    createCount: ctx.createCount,
  };
}
