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
  /** Tracked-for-disposal materials that take no camera broadcast (mesh). */
  readonly staticMaterials: Set<unknown>;
  readonly totalCreateMs: number;
  readonly createCount: number;
}

/** Build a stats snapshot for diagnostics / tests. */
export function getCacheStats(ctx: StatsCtx) {
  return {
    ownedMaterials: ctx.ownedMaterials.size,
    /**
     * Every material the manager is tracking, camera-aware or not — so a leak in
     * mesh materials is as visible here as one in the other three types. Materials
     * in `staticMaterials` are counted but never receive `updateCameraParams`.
     */
    totalRegistered: ctx.registeredMaterials.size + ctx.staticMaterials.size,
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
