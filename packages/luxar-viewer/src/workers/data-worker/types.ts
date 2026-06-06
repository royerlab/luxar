/**
 * Worker-side types passed across the Comlink RPC boundary.
 *
 * `EffectiveRadiusConfig` is sourced from the canonical definition
 * in `types/points.ts` so worker and main-thread paths can't drift.
 */

import type { EffectiveRadiusConfig } from '../../types/points';

export type { EffectiveRadiusConfig };

/**
 * View state for projection (subset of main thread ViewState).
 *
 * The worker accepts the minimal subset of ViewState fields its
 * projection kernels actually consume — keeping the worker payload
 * narrow rather than serializing the full main-thread ViewState.
 */
export interface ProjectionViewState {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
}
