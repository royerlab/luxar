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
  /**
   * Cumulative energy fraction e(k) of the COMMITTED ladder prefix — how much
   * of the ladder's total self-energy is on screen (from the progressive
   * loaders' `committedEnergyFraction`, sourced from the build-time
   * `lod_stats.energy_fraction_cum` stamps). `1` for non-progressive
   * (complete single-set) leaves; ABSENT on unstamped (legacy) datasets —
   * the display gate falls back to count crossover then.
   */
  committedEnergyFraction?: number;
}

/**
 * Whether a committed node may later need a previously superseded capacity.
 * Progressive nD nodes can shrink on a slice change and regrow into an old
 * bucket; an unsliced ladder grows monotonically, so every successful grow
 * makes its released geometry permanently obsolete.
 */
export function canLadderRegrow(
  userData: LadderStampable | undefined | null,
  ndim: number
): boolean {
  if (ndim > 3) return true;
  const loader = userData?.loader;
  if (!loader || typeof loader !== 'object' || !('hasMoreLODs' in loader)) return true;
  return loader.hasMoreLODs === true;
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
 * Stamp the committing loader's ladder state for the current view: whether
 * the committed ladder is complete (`committedLadderComplete`) and how much
 * of its total self-energy the committed prefix carries
 * (`committedEnergyFraction`). Reads both off `userData.loader` (the same
 * loader whose just-returned data is being committed — one source of truth).
 *
 * A non-progressive loader has no `hasMoreLODs` getter and stamps
 * complete/energy-1 (a single-set leaf IS its full content), matching the
 * structural-probe idiom used by the refinement scheduling (`queue-next.ts`).
 * A progressive loader whose dataset carries no `energy_fraction_cum` build
 * stamps reports `committedEnergyFraction: null` — the mesh stamp is then
 * REMOVED (absence = unstamped), so the display gate falls back to committed-
 * count crossover instead of blending measured and guessed energies.
 *
 * No-op when `userData` is absent. Called by every leaf commit, INCLUDING the
 * stamp-only no-op branches, so the stamps are exactly as current as the
 * freshness stamp beside them.
 */
export function stampLadderComplete(userData: LadderStampable | undefined | null): void {
  if (!userData) return;
  const loader = userData.loader as
    { hasMoreLODs?: boolean; committedEnergyFraction?: number | null } | undefined;
  userData.committedLadderComplete = loader?.hasMoreLODs !== true;
  if (!loader || !('committedEnergyFraction' in loader)) {
    // Non-progressive loader: the committed geometry is the leaf's complete
    // content — all of its energy is on screen.
    userData.committedEnergyFraction = 1;
    return;
  }
  const e = loader.committedEnergyFraction;
  if (e == null) delete userData.committedEnergyFraction;
  else userData.committedEnergyFraction = e;
}
