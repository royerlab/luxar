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
 * Returns `{ skip: 'extend_to_all' }` when the node's `extend_to_all`
 * dims fully cover all non-displayed dims (the work is a no-op),
 * otherwise the derived view state.
 */

import type { SceneNode, ViewState } from '../../data-loader-types';
import {
  hasOwnProperties,
  getOrComputeExtendedTolerance,
  validateExtendDims,
} from './extend-tolerance';
import { computeWorldNdTransform, invertNdTransformForQuery } from '../../transforms/nd-transform';

export type DerivedNodeViewState =
  { skip: 'extend_to_all' } | { skip: false; viewState: ViewState };

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

  // Step 1: full-extend skip check.
  if (extendDims.length > 0 && baseViewState.dimensions) {
    const dims = baseViewState.dimensions;
    validateExtendDims(extendDims, dims);
    const nonDisplayedDims = dims
      .filter((_: { name?: string }, idx: number) => !baseViewState.displayDims.includes(idx))
      .map((d: { name?: string }) => d.name)
      .filter((name: string | undefined): name is string => !!name);

    const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
      extendDims.includes(dimName)
    );
    if (isFullyExtended) {
      return { skip: 'extend_to_all' };
    }

    // Step 2: partial-extend tolerance override (Points + GSplats only).
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

  return { skip: false, viewState: derived };
}
