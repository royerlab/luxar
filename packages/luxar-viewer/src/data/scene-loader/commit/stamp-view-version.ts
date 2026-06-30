/**
 * Shared helper for the per-geometry commit concerns: stamp the view-update
 * version a mesh's committed geometry was loaded for onto its `userData`.
 *
 * The LOD registry (`scene/lod-freshness.ts` / `scene/lod-group-registry.ts`)
 * reads `userData.loadedViewVersion` to tell whether a level's committed
 * geometry is *fresh for the current view (slice / displayDims) version* —
 * distinct from "ready" (geometry committed), since a re-slice overwrites the
 * geometry buffers in place without changing readiness. Centralised here so
 * GSplats, Points, and Lines stamp it identically (three-geometry symmetry).
 *
 * @module data/scene-loader/commit/stamp-view-version
 */

/** Minimal shape of the leaf userData objects that carry the stamp. */
export interface ViewVersionStampable {
  loadedViewVersion?: number;
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
