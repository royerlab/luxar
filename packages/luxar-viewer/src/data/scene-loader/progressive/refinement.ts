/**
 * Generic progressive-LOD refinement loop.
 *
 * Factored from `data/gsplats/lod-refinement.ts` so all four geometry
 * types (Points, Lines, GSplats, Mesh) can drive the same
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
 *   5. If loaders still have more LODs and every one of them succeeded
 *      but none advanced, report the stalled rung(s), release the
 *      SceneLoader lock, and exit.
 *   6. Otherwise repeat until every loader finishes its ladder.
 *
 * @module data/scene-loader/progressive/refinement
 */

import type { ViewState } from '../../data-loader-types';
import type { ViewStateQueue } from '../view-state/view-state-queue';
import { scheduleFrame } from '../../../utils/schedule-frame';
import { noteRefinementPass } from '../../../profiling/load-timeline';

/**
 * Consecutive per-loader failures a refinement run tolerates before giving
 * up on that loader for the rest of the run. Without a cap, a persistently
 * failing LOD level (e.g. a hard 404 / decode error) kept `hasMoreLODs`
 * true forever and the loop retried at frame rate indefinitely — a network
 * retry storm with the update lock held. Scope is per loader instance, so a
 * partition resync cannot reset the cap; replacing the loader starts fresh.
 */
export const MAX_CONSECUTIVE_REFINEMENT_FAILURES = 3;

/**
 * Consecutive-failure bookkeeping shared by the four per-geometry refinement
 * wrappers (Points / Lines / GSplats / Mesh). Each loader owns one tracker and:
 *
 *   - skips loaders whose path {@link isExhausted},
 *   - calls {@link recordSuccess} after a loader's step completes,
 *   - calls {@link recordFailure} in its catch — when it returns `true`
 *     the loader just crossed the cap and the wrapper logs a final
 *     "giving up" line,
 *   - excludes exhausted paths from its `anyHasMoreLODs` aggregation so
 *     the generic loop can terminate and release the lock.
 */
export class RefinementFailureTracker {
  private failCounts = new Map<string, number>();
  private exhaustedPaths = new Set<string>();

  constructor(private maxConsecutiveFailures: number = MAX_CONSECUTIVE_REFINEMENT_FAILURES) {}

  /** True once `path` has failed `maxConsecutiveFailures` times in a row. */
  isExhausted(path: string): boolean {
    return this.exhaustedPaths.has(path);
  }

  /** Reset the consecutive-failure count for `path` (a step succeeded). */
  recordSuccess(path: string): void {
    this.failCounts.delete(path);
  }

  /**
   * Record one failure for `path`. Returns `true` exactly when this failure
   * crosses the cap (the caller logs the one-time "giving up" line).
   */
  recordFailure(path: string): boolean {
    const count = (this.failCounts.get(path) ?? 0) + 1;
    this.failCounts.set(path, count);
    if (count >= this.maxConsecutiveFailures && !this.exhaustedPaths.has(path)) {
      this.exhaustedPaths.add(path);
      return true;
    }
    return false;
  }
}

/** One eligible loader's loaded/total rung snapshot for a refinement pass. */
export interface ProgressiveRefinementProgress {
  path: string;
  loaded: number;
  total: number;
}

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
   * returns false without work. A successful attempt returns true;
   * failures and cancellations return false so their existing retry
   * policies take precedence over the no-progress guard.
   */
  processLoader(path: string, loader: TLoader): Promise<boolean>;
  /**
   * Snapshot a loader that is still eligible for refinement. Return `null`
   * for completed, non-progressive, or failure-exhausted loaders.
   */
  getLoaderProgress(
    path: string,
    loader: TLoader
  ): Omit<ProgressiveRefinementProgress, 'path'> | null;
  /**
   * After each pass, the loop asks whether ANY loader still has more
   * LODs to refine. Once all return false, the loop exits.
   */
  anyHasMoreLODs(): boolean;
  /** Report loaders that remained pending without advancing during a pass. */
  onNoProgress?(stalled: ProgressiveRefinementProgress[]): void;
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
      // scheduleFrame (not bare rAF) matters here: rAF is suspended in
      // hidden tabs, and this loop HOLDS the update lock — with the
      // queued-updateView pacing waiters, a suspended yield would hang
      // waitForUpdate()/awaitDimensionUpdate() callers until the tab is
      // foregrounded. scheduleFrame degrades to a timer when hidden and
      // runs synchronously in rAF-less test environments.
      await new Promise<void>((resolve) => scheduleFrame(resolve));

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
        const progressBefore = collectProgress(ctx);
        const successfulPaths = new Set<string>();
        for (const [path, loader] of ctx.loaders) {
          if (await ctx.processLoader(path, loader)) successfulPaths.add(path);
        }

        noteRefinementPass(successfulPaths.size);

        // Aggregate counts once per pass.
        ctx.updateVisibleCountsInMonitor();

        // Exit when all loaders have finished their ladders.
        if (!ctx.anyHasMoreLODs()) break;

        const progressAfter = collectProgress(ctx);
        const madeProgress =
          [...progressBefore].some(([path, before]) => {
            const after = progressAfter.get(path);
            return after === undefined || after.loaded > before.loaded;
          }) || [...progressAfter.keys()].some((path) => !progressBefore.has(path));
        const everyStalledLoaderSucceeded = [...progressAfter.keys()].every((path) =>
          successfulPaths.has(path)
        );
        if (!madeProgress && progressAfter.size > 0 && everyStalledLoaderSucceeded) {
          ctx.onNoProgress?.([...progressAfter.values()]);
          break;
        }
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

function collectProgress<TLoader>(
  ctx: ProgressiveRefinementCtx<TLoader>
): Map<string, ProgressiveRefinementProgress> {
  const progress = new Map<string, ProgressiveRefinementProgress>();
  for (const [path, loader] of ctx.loaders) {
    const loaderProgress = ctx.getLoaderProgress(path, loader);
    if (loaderProgress !== null) {
      progress.set(path, { path, ...loaderProgress });
    }
  }
  return progress;
}
