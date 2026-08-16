/**
 * Convert a `SimpleDims` snapshot (the high-level "user is navigating
 * dimensions" view) into a `ViewState` (the per-dimension query the
 * spatial-index loaders consume).
 *
 * Extracted from `zarr-loader.ts::updateSceneForDimensions` so it can be
 * tested without standing up the SceneLoaderManager / SceneLoader / zarr
 * stack. The rules below define the navigation-facing ViewState contract.
 *
 * Per-dimension tolerance rules:
 * - Displayed dimensions: `0` (the slice is in the viewing plane, so no
 *   thickness is queried along it).
 * - Discrete non-spatial non-displayed dimensions: `0.5` (exact-match with
 *   float-comparison slack — channels, time-as-index, etc.).
 * - Continuous / spatial non-displayed dimensions: `maxRadius` (so all
 *   geometry whose extent crosses the slice is included).
 *
 * Note: this is a different layer from `tolerance-computer.ts`, which is
 * the per-loader spatial-query layer that uses `1e10` for displayed dims.
 * This high-level converter uses `0` for displayed because that's the
 * shape the public ViewState contract uses for navigation events.
 */

import type { SimpleDims } from '../types/dims';
import type { ViewState } from './data-loader-types';

/**
 * Options controlling how {@link simpleDimsToViewState} fills the per-dimension
 * tolerance array when converting a `SimpleDims` snapshot into a `ViewState`.
 */
export interface DimsToViewStateOptions {
  /** Per-non-displayed continuous dim tolerance (typical: scene maxRadius). */
  maxRadius: number;
  /**
   * Initial filler for the tolerance array before the per-dim rules below
   * are applied. Every entry is overwritten — a dim with no metadata falls
   * through to `maxRadius`, not to this value — so it does not currently
   * reach the returned `ViewState`.
   */
  defaultTolerance: number;
}

/**
 * Convert a `SimpleDims` navigation snapshot into the `ViewState` query the
 * spatial-index loaders consume, applying the per-dimension tolerance rules
 * described in this module's header (0 for displayed dims, 0.5 for discrete
 * non-spatial dims, `maxRadius` otherwise).
 *
 * @param dims - Current dimension snapshot (displayed set, current step, metadata).
 * @param options - Tolerance fill options; see {@link DimsToViewStateOptions}.
 * @returns The `ViewState` describing displayed dims, slice position, and tolerances.
 */
export function simpleDimsToViewState(
  dims: SimpleDims,
  { maxRadius, defaultTolerance }: DimsToViewStateOptions
): ViewState {
  const tolerance = new Array(dims.ndim).fill(defaultTolerance).map((_, i) => {
    if (dims.displayed.includes(i)) {
      return 0;
    }
    const meta = dims.metadata?.[i];
    if (meta?.discrete && !meta?.spatial) {
      // Query-irrelevant: the spatial-index builders recompute the discrete
      // tolerance via `computeTolerance` (0.25×step) and ignore this value. It
      // only rides along in `viewState.tolerance` (e.g. the SliceCache key),
      // where a stable constant is all that's required.
      return 0.5;
    }
    // Continuous / spatial: no QUERY path reads this value any more. Every one
    // recomputes it: gsplats/lines/points-with-config via `computeTolerance`,
    // and `fallbackQueryTolerance` (the points path when a node has no
    // `EffectiveRadiusConfig`) now uses the node's own `maxRadius` instead of
    // the ride-along (issue #1183).
    //
    // The MAGNITUDE is still read, though — by two non-query consumers, unlike
    // the discrete case above where both collapse it away: `buildSliceViewSig`
    // keys a continuous dim on the raw tolerance, and `viewStatesEqual` skips a
    // tolerance difference only for a discrete non-spatial dim. So a divergence
    // between this builder and the init-time one would cost SliceCache misses
    // and progressive-loader resets — not wrong data.
    //
    // It NAMES a different constant than the init-time builder
    // (`ViewStateManager.buildToleranceArray`, which uses
    // `config.dataLoading.spatial.defaultTolerance`), and today the two are inert
    // only because they agree NUMERICALLY: `scene.userData.maxRadius` is never set
    // in production, so `zarr-loader.ts` passes
    // `config.dataLoading.spatial.defaultMaxRadius`, and both config values are
    // 0.1. Change either config value and the churn above becomes real.
    return maxRadius;
  });

  return {
    displayDims: [...dims.displayed],
    slicePosition: [...dims.currentStep],
    tolerance,
    dimensions: dims.metadata,
  };
}
