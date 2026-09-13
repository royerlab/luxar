/**
 * Scene-owned orchestration for loading the current nD dimension slice.
 *
 * @module scene/dimension-loading
 */

import * as THREE from 'three';
import {
  prefetchSceneForDimensions,
  releasePrefetchResources,
  updateSceneForDimensions,
} from '../data';
import { sceneDimsManager } from './scene-dims-manager';
import type { AnimationController } from './animation/animation-controller';
import type { DimensionAnimationManager } from './animation/dimension-animation-manager';
import type { SceneManager } from './scene-manager';
import { markLoad } from '../profiling/load-timeline';

/** Dependencies needed to load the current nD slice. */
export interface DimensionLoadingContext {
  sceneManager: SceneManager;
  animationController: AnimationController;
  getAnimationManager(): DimensionAnimationManager | undefined;
}

/**
 * The per-pass playback directives the loaders receive: the per-tick LOD time
 * budget, and the playback "detail" — a pinned rung count that makes the
 * loaders load exactly that many rungs per frame (cold or not) instead of
 * whatever fits the budget, the tick waiting for it (consistent frames at an
 * adaptive rate). Both undefined outside playback.
 */
function playbackDirectives(anim: DimensionAnimationManager | undefined): {
  frameBudgetMs: number | undefined;
  ladderDepth: number | 'auto' | undefined;
} {
  if (!anim) return { frameBudgetMs: undefined, ladderDepth: undefined };
  const frameBudgetMs = anim.getFrameBudgetMs() ?? undefined;
  if (frameBudgetMs !== undefined) {
    return { frameBudgetMs, ladderDepth: anim.getPlaybackLadderDepth() ?? undefined };
  }
  // Not playing: a SCRUB (slider drag, keyboard step, pause re-trigger). Pin it
  // to the scrub detail so a heavy time-lapse reads consistently while it is
  // dragged; the settle pass below refines to the full ladder once quiet. The
  // settle pass itself carries no directive.
  if (scrubSettlePassPending) return { frameBudgetMs: undefined, ladderDepth: undefined };
  return { frameBudgetMs: undefined, ladderDepth: anim.getScrubLadderDepth() ?? undefined };
}

/**
 * Quiet period after the last scrub before the unpinned settle pass fires.
 * Long enough to swallow a slider drag's event stream, short enough that a
 * paused view starts refining to its full ladder almost immediately.
 */
export const SCRUB_SETTLE_MS = 400;

let scrubSettleTimer: ReturnType<typeof setTimeout> | null = null;
/** True while the settle pass's own dimension notification is being served. */
let scrubSettlePassPending = false;

/**
 * Schedule the unpinned settle pass that follows a pinned scrub pass: after
 * `SCRUB_SETTLE_MS` without another slice change, re-notify the dimension
 * listeners at the current position with no playback directive, so the normal
 * refine pass and refinement run complete the ladder. Skipped if playback has
 * started meanwhile (the pause re-trigger covers that path).
 */
function scheduleScrubSettle(ctx: DimensionLoadingContext): void {
  if (scrubSettleTimer !== null) clearTimeout(scrubSettleTimer);
  scrubSettleTimer = setTimeout(() => {
    scrubSettleTimer = null;
    if (ctx.getAnimationManager()?.isAnyPlaying()) return;
    const dims = sceneDimsManager.getDims();
    if (!dims || dims.ndim === 0) return;
    scrubSettlePassPending = true;
    try {
      // Same-value write: setDimensionValue notifies listeners unconditionally.
      sceneDimsManager.setDimensionValue(0, dims.currentStep[0]);
    } finally {
      // The listener ran synchronously up to its first await and has already
      // read the (empty) directives; clear for the next scrub.
      scrubSettlePassPending = false;
    }
  }, SCRUB_SETTLE_MS);
}

/** Test hook: drop a pending settle timer. */
export function cancelScrubSettleForTests(): void {
  if (scrubSettleTimer !== null) clearTimeout(scrubSettleTimer);
  scrubSettleTimer = null;
  scrubSettlePassPending = false;
}

/**
 * Update all nD nodes (points, lines, splats) for the current slice.
 * Called from the scene-dims listener and once after dimension UI setup.
 */
export async function updateAllNDNodes(ctx: DimensionLoadingContext): Promise<void> {
  const dims = sceneDimsManager.getDims();
  if (!dims) {
    return;
  }

  // During dimension-animation playback, hand the progressive loaders a
  // per-tick LOD time budget so each update pass fits the animation frame
  // window (they stream sub-LODs until the budget runs out, then commit).
  // Null/undefined outside playback → normal full-refinement behavior.
  const { frameBudgetMs, ladderDepth } = playbackDirectives(ctx.getAnimationManager());
  const pinnedScrub = frameBudgetMs === undefined && ladderDepth !== undefined;

  // Use the new loader architecture's update mechanism
  await updateSceneForDimensions(
    dims,
    ctx.sceneManager.scene as unknown as THREE.Group,
    undefined,
    {
      frameBudgetMs,
      ladderDepth,
    }
  );
  // Once per load: the first slice update after `loadScene` has landed.
  markLoad('initUpdateDone');

  // Projected t+1 prefetch: the await above resolved at the pass's COMMIT
  // (the pass-waiter contract), i.e. the start of the idle window before
  // the next playback tick — refinement is suppressed during play, so the
  // window is free. Predict the next value per playing dim (loop/bounce/
  // backward aware via peekNextValue) and warm the S-cache for it on
  // SHADOW loaders (fire-and-forget; aborted by the next foreground pass).
  const anim = ctx.getAnimationManager();
  if (anim?.isAnyPlaying()) {
    const nextStep = [...dims.currentStep];
    let predictedAny = false;
    for (const dimIndex of anim.getPlayingDimIndices()) {
      const next = anim.peekNextValue(dimIndex);
      if (next !== null) {
        nextStep[dimIndex] = next;
        predictedAny = true;
      }
    }
    const budgetMs = anim.getFrameBudgetMs();
    if (predictedAny && budgetMs !== null) {
      prefetchSceneForDimensions(
        { ...dims, currentStep: nextStep },
        ctx.sceneManager.scene as unknown as THREE.Group,
        undefined,
        { budgetMs, ladderDepth }
      );
    }
  } else {
    // Playback ended (this branch includes the pause refine re-trigger
    // pass): free the shadow loaders' accumulators until the next play.
    releasePrefetchResources();
    if (pinnedScrub) scheduleScrubSettle(ctx);
  }

  // Trigger re-render after update
  ctx.animationController.startAnimation();
}
