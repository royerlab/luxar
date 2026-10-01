/**
 * View-state queue for serialized SceneLoader updates.
 *
 * Owns two pieces of update-time state:
 *
 *   1. `_pendingViewState` — the single Partial<ViewState> queued by an
 *      `updateView()` call while the serialization lock is held (a pass, the
 *      frame between passes, a refinement drain or a retry). Its lifecycle —
 *      when it is taken, drained or flushed — is `PassScheduler`'s
 *      (`update-view/pass-scheduler.ts`).
 *
 *   2. `_prevPerNodeViewState` — per-node snapshots of the last
 *      successful view-state, used to extrapolate the next-frame view
 *      for predictive prefetch.
 *
 * SceneLoader holds one queue instance and delegates pending-state
 * set/take/has, prefetch dispatch, and the drain trigger to it.
 *
 * Drain ordering is load-bearing: it MUST fire on a microtask
 * (`Promise.resolve().then(...)`) after the caller's own promise
 * resolves — the retry path that triggers the drain expects to settle
 * its own continuation first. Tolerance / extend_to_all semantics
 * remain unchanged because the queue stores only the raw view-state and
 * passes it back through the trigger callback unchanged.
 */

import { log, Modules } from '../../../utils/log';
import { getErrorMessage } from '../../../utils/format-error';
import type { ViewState } from '../../data-loader-types';
import { dispatchPredictivePrefetch, type PrefetchableLoader } from './predicted-view-state';

/**
 * How many predicted-view prefetches stay live per node: the newest plus its
 * predecessor, whose in-flight fetches the newest may still be about to join.
 */
const LIVE_PREDICTIONS_PER_NODE = 2;

export class ViewStateQueue {
  private _pendingViewState: Partial<ViewState> | null = null;
  private _prevPerNodeViewState: Map<string, ViewState> = new Map();
  /**
   * The live predicted-view prefetches per node, oldest first, at most
   * {@link LIVE_PREDICTIONS_PER_NODE}. A new prediction aborts the oldest — the
   * user has moved past it — but NOT its immediate predecessor: consecutive
   * predictions overlap (one chunk spans several slices), and aborting the
   * predecessor before the new one's reads join its in-flight fetches would
   * cancel those fetches outright and make the new prediction refetch them.
   * A dataset switch / retry baseline ({@link clearPrev}) and
   * {@link forgetPath} abort them all.
   */
  private _predictionControllers: Map<string, AbortController[]> = new Map();

  /** Queue a view-state for a future drain. Overwrites any previous pending state. */
  setPending(state: Partial<ViewState>): void {
    this._pendingViewState = state;
  }

  /** Returns the pending view-state and clears it, or `null` if nothing was queued. */
  takePending(): Partial<ViewState> | null {
    const pending = this._pendingViewState;
    this._pendingViewState = null;
    return pending;
  }

  /** Whether a view-state is currently queued. */
  hasPending(): boolean {
    return this._pendingViewState !== null;
  }

  /**
   * Process any pending view-state queued during a retry / refinement
   * loop. Fires on a microtask so the caller's promise resolves first,
   * then re-enters `updateView` via the supplied trigger callback.
   *
   * @returns True when a state was queued and a re-entry has been scheduled;
   *   false when the slot was empty and nothing will run. Callers that also
   *   settle the pass waiters need this: the re-entered pass carries the parked
   *   waiters to its own commit, so resolving them as well would release them
   *   before the view they asked for has landed.
   */
  drain(triggerUpdate: (state: Partial<ViewState>) => Promise<unknown>): boolean {
    const pendingState = this.takePending();
    if (pendingState === null) return false;
    Promise.resolve()
      .then(() => triggerUpdate(pendingState))
      .catch((error: unknown) => {
        log.warning(
          Modules.SCENE_LOADER,
          `Drained updateView after retry failed: ${getErrorMessage(error)}`,
          error
        );
      });
    return true;
  }

  /**
   * S6: per-loader predictive prefetch. Extrapolates the next-frame
   * view state from the path's previous derived view-state to the
   * current one and fires `prefetchChunks(predicted)` on the loader.
   * Honors per-node tolerance extension (incl. a fully-extended node's
   * slice-invariant query) by accepting the *derived* view-state (caller
   * computes it) — so extended nodes don't get over-prefetched.
   *
   * Fire-and-forget in a microtask so a slow prefetch cannot delay the
   * commit path.
   */
  dispatchPrefetch(path: string, current: ViewState, loader: unknown): void {
    const prev = this._prevPerNodeViewState.get(path) ?? null;
    // Snapshot current — keeps the saved value immune to later in-place
    // mutation by downstream loader work.
    const snapshot: ViewState = {
      displayDims: [...current.displayDims],
      slicePosition: [...current.slicePosition],
      tolerance: [...current.tolerance],
      dimensions: current.dimensions,
    };
    this._prevPerNodeViewState.set(path, snapshot);

    // First call for this path (no prev) — there's nothing to extrapolate,
    // the dispatcher would return `false`. Skip the microtask so we don't
    // pay queue overhead for a guaranteed no-op. The next updateView
    // observes this snapshot as `prev` and the user's first
    // scrub-direction delta is captured then.
    if (prev === null) return;

    const controller = this.startPrediction(path);
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      dispatchPredictivePrefetch(prev, snapshot, [loader as PrefetchableLoader], controller.signal);
    });
  }

  /** Register a new prediction for `path`, aborting the ones it supersedes. */
  private startPrediction(path: string): AbortController {
    const live = this._predictionControllers.get(path) ?? [];
    while (live.length >= LIVE_PREDICTIONS_PER_NODE) live.shift()?.abort();
    const controller = new AbortController();
    live.push(controller);
    this._predictionControllers.set(path, live);
    return controller;
  }

  /** Abort every live prediction for `path`. */
  private abortPredictions(path: string): void {
    for (const controller of this._predictionControllers.get(path) ?? []) controller.abort();
    this._predictionControllers.delete(path);
  }

  /** Drop the saved per-node previous view-state for a single path. */
  forgetPath(path: string): void {
    this._prevPerNodeViewState.delete(path);
    this.abortPredictions(path);
  }

  /**
   * Clear the per-loader prefetch predictor state. Called on dataset
   * switch + on retry-baseline failures so a reused SceneLoader doesn't
   * extrapolate from a prior dataset.
   */
  clearPrev(): void {
    this._prevPerNodeViewState.clear();
    for (const path of [...this._predictionControllers.keys()]) this.abortPredictions(path);
  }
}
