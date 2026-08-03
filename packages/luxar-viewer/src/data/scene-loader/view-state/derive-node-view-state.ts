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
 * A node whose `extend_to_all` covers ALL non-displayed dims (a
 * slice-independent "always visible" context layer) is NOT special-cased
 * into a skip: it is derived as a NORMAL node whose query is made
 * slice-INVARIANT. Its view state gets the `1e10` extend-to-all tolerance
 * sentinel on every extended dim AND its extended dims' `slicePosition`
 * pinned to a constant `0`, so the derived view state is byte-identical
 * across every scrub. Each per-sweep `updateView` then hits the loader's
 * memoized same-view no-op path (`viewStatesEqual`), which delegates all
 * convergence / empty-ladder / abort / playback-budget handling to the
 * well-tested normal-node path — no fragile convergence detection.
 */

import type { SceneNode, ViewState } from '../../data-loader-types';
import {
  hasOwnProperties,
  getOrComputeExtendedTolerance,
  validateExtendDims,
} from './extend-tolerance';
import { computeWorldNdTransform, invertNdTransformForQuery } from '../../transforms/nd-transform';

export type DerivedNodeViewState = { skip: false; viewState: ViewState };

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

  let isFullyExtended = false;
  let derived: ViewState = {
    displayDims: baseViewState.displayDims,
    slicePosition: baseViewState.slicePosition,
    tolerance: baseViewState.tolerance,
    dimensions: baseViewState.dimensions,
  };

  // Step 1: extend_to_all detection + tolerance override.
  if (extendDims.length > 0 && baseViewState.dimensions) {
    const dims = baseViewState.dimensions;
    validateExtendDims(extendDims, dims);
    const nonDisplayedDims = dims
      .filter((_: { name?: string }, idx: number) => !baseViewState.displayDims.includes(idx))
      .map((d: { name?: string }) => d.name)
      .filter((name: string | undefined): name is string => !!name);

    isFullyExtended = nonDisplayedDims.every((dimName: string) => extendDims.includes(dimName));
    if (isFullyExtended) {
      // Fully extended: compute the extended tolerance UNCONDITIONALLY. The
      // query that loads the whole node needs the `1e10` sentinel on every
      // extended dim regardless of the `applyPartialExtendTolerance` opt — that
      // opt only governs the PARTIAL case; the full-extend query must always
      // ignore the non-displayed dims (this is also what lets a lines node,
      // which derives with the opt off, still load when fully extended).
      const tolerance = getOrComputeExtendedTolerance(
        baseViewState.tolerance,
        extendDims,
        baseViewState.dimensions,
        opts.extendedToleranceCache ?? new Map<string, number[]>()
      );
      derived = { ...derived, tolerance };
    } else if (opts.applyPartialExtendTolerance) {
      // Step 2: partial-extend tolerance override (Points + GSplats only).
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
        // fire on a dimension the node extends. (The `1e10` tolerance sentinel
        // is not a reliable proxy — a Lines node derives with
        // `applyPartialExtendTolerance: false`, so a PARTIALLY-extended lines
        // node's extended dims never carry it; only the full-extend case sets
        // the sentinel unconditionally. Keying off the names covers both.)
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

  // Step 4: pin the extended dims' slicePosition to a constant so a
  // fully-extended node's derived view state is INVARIANT across scrubs (the
  // whole point — per-sweep re-queries then hit the loader's same-view no-op).
  // Done AFTER Step 3 so the nd_transform inverse can't perturb the pin.
  // Correctness is unaffected: the projection skips extended dims entirely
  // (`hidden-dims.ts`: `if (extendToAllDims.has(dim)) continue`) and the `1e10`
  // tolerance covers everything regardless of center — the pin is purely to
  // stabilize `viewStatesEqual` / the slice-cache key.
  if (isFullyExtended && derived.dimensions) {
    const pinned = [...derived.slicePosition];
    for (const dimName of extendDims) {
      const idx = derived.dimensions.findIndex((d) => d.name === dimName);
      if (idx >= 0 && idx < pinned.length) pinned[idx] = 0;
    }
    derived = { ...derived, slicePosition: pinned };
  }

  return { skip: false, viewState: derived };
}
