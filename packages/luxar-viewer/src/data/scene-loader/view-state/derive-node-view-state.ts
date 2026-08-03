/**
 * Per-node view-state derivation — folds `extend_to_all` tolerance
 * overrides and the inverse `nd_transform` for the node's path into
 * the base ViewState.
 *
 * Single source of truth for query-state derivation: both the main
 * update path and `retryFailedLoader` go through this helper, so a
 * retry cannot load a different query region than a fresh update
 * would. The pre-extraction retry skipped both adjustments, which
 * could produce a "successful" retry rendering incorrect data.
 *
 * ALWAYS returns a derived `viewState`. `skip` is an OPTIMIZATION HINT
 * (`'extend_to_all'`) that a node's `extend_to_all` dims fully cover all
 * non-displayed dims, so the query never changes across slices and the
 * per-slice update handler may short-circuit. It is NOT a signal that no
 * view state was produced: a fully-extended node still carries its derived
 * state — the base slice with the 1e10 tolerance sentinels applied to the
 * extended dims — so the initial-load / refinement / retry paths fetch the
 * correct "ignore the hidden dims" query (the prefetcher, by contrast,
 * correctly skips a fully-extended node on the hint) (issue #1157). A prior
 * bug early-returned a bare `{ skip: 'extend_to_all' }` before the tolerance
 * override ran, so consumers fell back to the raw live slice (points sliced
 * away, gsplats filtered out, ladder frozen).
 */

import type { SceneNode, ViewState } from '../../data-loader-types';
import {
  hasOwnProperties,
  getOrComputeExtendedTolerance,
  validateExtendDims,
} from './extend-tolerance';
import { computeWorldNdTransform, invertNdTransformForQuery } from '../../transforms/nd-transform';

export type DerivedNodeViewState = { skip: 'extend_to_all' | false; viewState: ViewState };

export interface DeriveOpts {
  applyPartialExtendTolerance: boolean;
  /**
   * Optional cross-node cache for the partial-extend tolerance array.
   * Main update path passes one cache per cycle; retry passes nothing.
   */
  extendedToleranceCache?: Map<string, number[]>;
}

/**
 * Derive a per-node view state. Pure function; takes the base viewState
 * + the scene-graph root (for computing the world `nd_transform` from
 * the node up to the root) as inputs.
 */
export function deriveNodeViewState(
  path: string,
  attrs: { extend_to_all?: string[] } | undefined,
  baseViewState: ViewState,
  sceneGraph: SceneNode | null,
  opts: DeriveOpts
): DerivedNodeViewState {
  const extendDims: string[] = attrs?.extend_to_all ?? [];

  let derived: ViewState = {
    displayDims: baseViewState.displayDims,
    slicePosition: baseViewState.slicePosition,
    tolerance: baseViewState.tolerance,
    dimensions: baseViewState.dimensions,
  };

  // Whether the node's extend_to_all dims cover EVERY non-displayed dim. This
  // is the `skip` optimization hint — it does NOT short-circuit derivation, so
  // the tolerance override below still runs (issue #1157).
  let fullyExtended = false;

  // Step 1: extend_to_all analysis + tolerance override.
  if (extendDims.length > 0 && baseViewState.dimensions) {
    const dims = baseViewState.dimensions;
    validateExtendDims(extendDims, dims);
    const nonDisplayedDims = dims
      .filter((_: { name?: string }, idx: number) => !baseViewState.displayDims.includes(idx))
      .map((d: { name?: string }) => d.name)
      .filter((name: string | undefined): name is string => !!name);

    fullyExtended = nonDisplayedDims.every((dimName: string) => extendDims.includes(dimName));

    // Step 2: extend tolerance override (Points + GSplats only). Applied for
    // BOTH the fully- and partially-extended cases: a fully-extended node must
    // still carry the 1e10 sentinels on its hidden dims so its query ignores
    // them (data-processor-gsplats derives its extended-dims set from exactly
    // these sentinels). This used to sit behind the fully-extended early return,
    // so a fully-extended node never got the override and fell back to the raw
    // live slice — the #1157 bug.
    if (opts.applyPartialExtendTolerance) {
      const tolerance = getOrComputeExtendedTolerance(
        baseViewState.tolerance,
        extendDims,
        baseViewState.dimensions,
        opts.extendedToleranceCache ?? new Map<string, number[]>()
      );
      derived = { ...derived, tolerance };
    }
  }

  // Step 3: nd_transform inverse for world→local query mapping.
  if (sceneGraph && derived.dimensions) {
    const worldNdT = computeWorldNdTransform(sceneGraph, path);
    if (hasOwnProperties(worldNdT)) {
      const inverted = invertNdTransformForQuery(
        derived.slicePosition,
        derived.tolerance,
        worldNdT,
        derived.dimensions,
        derived.displayDims,
        // Pass the node's extend_to_all NAMES: the no-preimage rule must not
        // fire on a dimension the node extends, and the 1e10 tolerance sentinel
        // is not a reliable proxy — every Lines call site derives with
        // `applyPartialExtendTolerance: false`, so a lines node's extended dims
        // never carry it.
        extendDims
      );
      // `noPreimage` rides the derived state only when set, so nodes with an
      // ordinary transform keep a view state that is shape-identical to before
      // (see ViewState.noPreimage for why it must stay out of the cache key).
      derived = inverted.noPreimage
        ? { ...derived, ...inverted }
        : {
            ...derived,
            slicePosition: inverted.slicePosition,
            tolerance: inverted.tolerance,
          };
    }
  }

  return { skip: fullyExtended ? 'extend_to_all' : false, viewState: derived };
}
