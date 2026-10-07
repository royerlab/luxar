/**
 * Worker-side types passed across the Comlink RPC boundary.
 */

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
