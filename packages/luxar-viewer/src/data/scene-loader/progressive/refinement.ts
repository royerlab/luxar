/**
 * Generic progressive-LOD refinement loop.
 *
 * Factored from `data/gsplats/lod-refinement.ts` so all three leaf
 * types (Points, Lines, GSplats) can drive the same
 * `requestAnimationFrame()` yield + cancellable-via-queue contract
 * with their own typed loader / processor / commit closures.
 *
 * Per-frame contract:
 *   1. Yield via `requestAnimationFrame` (synchronous in test envs).
 *   2. Check the view-state queue for a pending update from the user.
 *      If pending → hand off to `retriggerUpdate` and return.
 *   3. For each loader, call the user-supplied `processLoader`
 *      closure — which derives view-state, fetches the next LOD,
 *      processes, and commits.
 *   4. Call `updateVisibleCountsInMonitor` once per pass.
 *   5. If any loader still has more LODs to load, repeat. Otherwise
 *      release the SceneLoader lock and exit.
 *
 * @module data/scene-loader/progressive/refinement
 */

import type { ViewState } from '../../data-loader-types';
import type { ViewStateQueue } from '../view-state/view-state-queue';

/**
 * Context object the generic loop needs from the per-type caller.
 *
 * `TLoader` is intentionally loose — the loop only reads loaders out
 * of `loaders` and passes them through `processLoader` / checks
 * `anyHasMoreLODs`. All type-specific work happens inside the
 * closures.
 */
export interface ProgressiveRefinementCtx<TLoader> {
  /** Currently-loaded progressive loaders, keyed by scene path. */
  loaders: Map<string, TLoader>;
  /** Queue checked for user-initiated cancellation each rAF. */
  viewStateQueue: ViewStateQueue;
  /**
   * Per-iteration processor. Handles ONE loader: derive query state,
   * fetch the next LOD via the loader's `updateView`, process, commit.
   * If the loader has no more LODs (or query state says skip), this
   * is a no-op. Exceptions are caught + logged by the caller's
   * closure so the loop continues on per-loader failure.
   */
  processLoader(path: string, loader: TLoader): Promise<void>;
  /**
   * After each pass, the loop asks whether ANY loader still has more
   * LODs to refine. Once all return false, the loop exits.
   */
  anyHasMoreLODs(): boolean;
  /** Aggregate visible counts into the monitor after each pass. */
  updateVisibleCountsInMonitor(): void;
  /**
   * Release the SceneLoader's `_updateInProgress` lock when the loop
   * completes normally. On cancellation the loop hands the lock to
   * `retriggerUpdate` instead.
   */
  releaseLock(): void;
  /**
   * Re-enter `SceneLoader.updateView()` with the pending state.
   * Owns the lock from this point. Typically schedules the new
   * update at the next rAF boundary.
   */
  retriggerUpdate(pendingState: Partial<ViewState>): void;
  /**
   * Optional liveness check. Returns false once the owning SceneLoader
   * has been disposed (e.g. a dataset switch tore it down while this loop
   * was mid-flight). When provided and false, the loop aborts immediately
   * instead of continuing to fetch/decode/prefetch against a dead dataset.
   * Omitted by callers that have no separate disposal signal.
   */
  isActive?(): boolean;
  /**
   * Optional handler for an error thrown inside the per-pass loop body
   * (outside the per-loader processing, which the caller's closure already
   * guards). Called just before the loop breaks; typically logs.
   */
  onError?(error: unknown): void;
}

/**
 * Run the refinement loop. Returns when no progressive loader has
 * additional LODs (normal completion) or when a pending view-state
 * was observed and the loop handed off to `retriggerUpdate`
 * (cancellation).
 */
export async function runProgressiveRefinement<TLoader>(
  ctx: ProgressiveRefinementCtx<TLoader>
): Promise<void> {
  let lockHandedOff = false;
  try {
    while (true) {
      // Yield: let the browser paint the current LOD level so the
      // refinement is *visible* rather than batched into one frame.
      await new Promise<void>((resolve) => {
        if (typeof requestAnimationFrame !== 'undefined') {
          requestAnimationFrame(() => resolve());
        } else {
          // Test environment without rAF: proceed synchronously.
          resolve();
        }
      });

      // Abort if the owning SceneLoader was disposed (e.g. a dataset
      // switch) while this loop was mid-flight — stop touching the dead
      // dataset rather than burning fetch/decode/prefetch cycles on it.
      if (ctx.isActive && !ctx.isActive()) {
        return;
      }

      // Cancellation check: did the user navigate while we were
      // mid-refinement? If so, hand the lock to the new update.
      const pendingState = ctx.viewStateQueue.takePending();
      if (pendingState !== null) {
        lockHandedOff = true;
        ctx.retriggerUpdate(pendingState);
        return;
      }

      // One refinement pass: each loader processes one more LOD. The whole
      // pass body is guarded so a throw from the monitor/visibility helpers
      // (e.g. walking a half-disposed scene) is logged and stops the loop
      // rather than escaping as an unhandled promise rejection — the loop is
      // kicked fire-and-forget.
      try {
        for (const [path, loader] of ctx.loaders) {
          await ctx.processLoader(path, loader);
        }

        // Aggregate counts once per pass.
        ctx.updateVisibleCountsInMonitor();

        // Exit when all loaders have finished their ladders.
        if (!ctx.anyHasMoreLODs()) break;
      } catch (error) {
        ctx.onError?.(error);
        break;
      }
    }
  } finally {
    if (!lockHandedOff) {
      ctx.releaseLock();
    }
  }
}
