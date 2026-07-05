/**
 * Shared helpers for the per-geometry commit concerns: stamp commit-time
 * facts about a mesh's committed geometry onto its `userData`.
 *
 * The LOD registry (`scene/lod-freshness.ts` / `scene/lod-group-registry.ts`)
 * reads two stamps:
 *
 *   - `loadedViewVersion` — whether a level's committed geometry is *fresh
 *     for the current view (slice / displayDims) version*; distinct from
 *     "ready" (geometry committed), since a re-slice overwrites the geometry
 *     buffers in place without changing readiness.
 *   - `committedLadderComplete` — whether the committed geometry is the FULL
 *     additive ladder for the current view, or a partial prefix that is
 *     still streaming. Read by the never-downgrade display gate
 *     (`shouldHoldPreviousDisplay` / `subtreeDisplayProgress`). Stamped at
 *     commit time — unlike the loaders' live `hasMoreLODs` getters, which
 *     flip the moment the final LOD's *fetch* resolves, frames before its
 *     processing + commit land — so a reader never sees "complete" paired
 *     with a stale partial count.
 *
 * Centralised here so GSplats, Points, and Lines stamp identically
 * (three-geometry symmetry).
 *
 * @module data/scene-loader/commit/stamp-view-version
 */

/** Minimal shape of the leaf userData objects that carry the stamps. */
export interface ViewVersionStampable {
  loadedViewVersion?: number;
}

/**
 * Minimal shape for {@link stampLadderComplete}: every leaf mesh's userData
 * carries a live reference to the loader that produced its data (written at
 * node creation by the node factory / create-*-node helpers), which is how
 * the stamp reads ladder state with zero extra plumbing.
 */
export interface LadderStampable {
  loader?: unknown;
  committedLadderComplete?: boolean;
}

/**
 * Write `version` onto `userData.loadedViewVersion`. No-op when `userData` is
 * absent. Called by every leaf commit (gsplats / points / lines) so the stamp
 * lives in exactly one place.
 */
export function stampLoadedViewVersion(
  userData: ViewVersionStampable | undefined | null,
  version: number
): void {
  if (userData) userData.loadedViewVersion = version;
}

/**
 * Stamp whether the committing loader's additive ladder is complete for the
 * current view. Reads `hasMoreLODs` off `userData.loader` (the same loader
 * whose just-returned data is being committed — one source of truth). A
 * non-progressive loader has no `hasMoreLODs` getter and stamps `true`
 * (complete), matching the structural-probe idiom used by the refinement
 * scheduling (`queue-next.ts`). No-op when `userData` is absent. Called by
 * every leaf commit, INCLUDING the stamp-only no-op branches, so the stamp is
 * exactly as current as the freshness stamp beside it.
 */
export function stampLadderComplete(userData: LadderStampable | undefined | null): void {
  if (!userData) return;
  userData.committedLadderComplete =
    (userData.loader as { hasMoreLODs?: boolean } | undefined)?.hasMoreLODs !== true;
}
