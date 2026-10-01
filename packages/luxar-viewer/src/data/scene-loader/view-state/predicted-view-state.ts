/**
 * R4: Predict the next view state for prefetch hints.
 *
 * `SceneLoader.updateView()` runs once per view change (slider drag,
 * keyboard step, dimension animation tick). The cache only sees the
 * current view; without a prediction it has no way to know which
 * chunk the user will need on the *next* tick. By extrapolating the
 * per-dimension delta between the previous and current view states,
 * we can dispatch a predicted transition to each loader so the cache
 * warms while the user keeps scrubbing. Spatial-index loaders use their
 * array chunk shapes plus index bounds to warm the next chunk boundary;
 * other loaders retain the single-step fallback.
 *
 * Single-step extrapolation: for each non-displayed dimension d,
 *
 *     delta_d = current.slicePosition[d] - prev.slicePosition[d]
 *     predicted.slicePosition[d] = current.slicePosition[d] + delta_d
 *
 * Display axes are never extrapolated — `displayDims` changes are
 * user-driven, not animation-driven, and don't warrant predictive
 * prefetch. NaN-protected: a NaN delta would otherwise propagate into
 * the predicted slice and produce wild prefetch targets.
 *
 * Returns a fresh `ViewState` with copied arrays so callers can hand
 * it to loaders without worrying about cross-mutation with the
 * SceneLoader's own viewState.
 */

import type { ViewState } from '../../data-loader-types';
import { tagSignalPriority } from '../../../utils/fetch-concurrency';

/**
 * Structural type for a loader that supports predictive prefetch.
 * The concrete spatial-index loaders implement `prefetchChunkBoundary`;
 * loaders without chunk metadata can expose `prefetchChunks` for the
 * historical one-step behavior. The base loader interfaces don't declare
 * either method, so this structural type avoids coupling scene-loader.ts
 * to concrete loader classes.
 */
export interface PrefetchableLoader {
  prefetchChunks?: (vs: ViewState, signal?: AbortSignal) => Promise<void>;
  prefetchChunkBoundary?: (
    current: ViewState,
    predicted: ViewState,
    signal?: AbortSignal
  ) => Promise<void>;
}

/**
 * Predict the next view state and dispatch a chunk-boundary-aware transition,
 * falling back to `prefetchChunks(predicted)`. Returns `true` when a prefetch
 * was dispatched, `false` when the predictor decided no axis moved.
 *
 * The dispatcher itself is synchronous (returns after calling each
 * loader's prefetch method once); the underlying prefetch is async
 * and not awaited — that's the point of prefetch. Every warm-up runs under
 * `signal`, tagged `'speculative'` here; the caller aborts it when the next
 * prediction for the same node supersedes this one (`ViewStateQueue`).
 *
 * Errors from either prefetch method are caught and silently swallowed —
 * prefetch is best-effort cache warming, not a demand fetch, so a
 * failed prefetch must never block the next updateView.
 */
export function dispatchPredictivePrefetch(
  prev: ViewState | null,
  current: ViewState,
  loaders: Iterable<PrefetchableLoader>,
  signal: AbortSignal = new AbortController().signal
): boolean {
  const predicted = predictNextViewState(prev, current);

  let moved = false;
  for (let i = 0; i < predicted.slicePosition.length; i++) {
    if (predicted.slicePosition[i] !== current.slicePosition[i]) {
      moved = true;
      break;
    }
  }
  if (!moved) return false;

  // A prediction is speculative by definition: tag it so the fetch gate and
  // the decode cache queue it behind frame-blocking reads (an untagged signal
  // resolves to `demand`). The caller owns the signal and aborts it when a
  // newer prediction supersedes this one.
  tagSignalPriority(signal, 'speculative');
  for (const loader of loaders) {
    dispatchToLoader(loader, current, predicted, signal);
  }
  return true;
}

/**
 * Fire one loader's predicted-view warm-up: the chunk-boundary-aware transition
 * when it has one, else the one-step `prefetchChunks(predicted)` fallback.
 * Best-effort — a synchronous throw or a rejection is swallowed so prefetch
 * noise never masks a demand-path failure.
 */
function dispatchToLoader(
  loader: PrefetchableLoader,
  current: ViewState,
  predicted: ViewState,
  signal: AbortSignal
): void {
  try {
    const result = loader.prefetchChunkBoundary
      ? loader.prefetchChunkBoundary(current, predicted, signal)
      : loader.prefetchChunks?.(predicted, signal);
    if (result && typeof (result as Promise<unknown>).catch === 'function') {
      (result as Promise<unknown>).catch(() => {
        // Prefetch is best-effort; swallow errors so demand-path
        // failures aren't masked by prefetch noise in the console.
      });
    }
  } catch {
    // Synchronous throw — same policy.
  }
}

export function predictNextViewState(prev: ViewState | null, current: ViewState): ViewState {
  const predicted: ViewState = {
    displayDims: [...current.displayDims],
    slicePosition: [...current.slicePosition],
    tolerance: [...current.tolerance],
    // Dimension metadata is treated as read-only by all consumers, so
    // sharing the reference is safe and avoids deep-copy churn.
    dimensions: current.dimensions,
  };
  if (!prev) return predicted;

  const slice = predicted.slicePosition as number[];
  for (let d = 0; d < current.slicePosition.length; d++) {
    if (current.displayDims.includes(d)) continue;
    const prevVal = prev.slicePosition[d];
    const currVal = current.slicePosition[d];
    if (prevVal === undefined || !Number.isFinite(prevVal) || !Number.isFinite(currVal)) {
      continue;
    }
    const delta = currVal - prevVal;
    if (delta === 0) continue;
    slice[d] = currVal + delta;
  }
  return predicted;
}
