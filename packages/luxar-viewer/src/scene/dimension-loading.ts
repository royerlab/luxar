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
  const frameBudgetMs = ctx.getAnimationManager()?.getFrameBudgetMs() ?? undefined;

  // Use the new loader architecture's update mechanism
  await updateSceneForDimensions(
    dims,
    ctx.sceneManager.scene as unknown as THREE.Group,
    undefined,
    {
      frameBudgetMs,
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
        { budgetMs }
      );
    }
  } else {
    // Playback ended (this branch includes the pause refine re-trigger
    // pass): free the shadow loaders' accumulators until the next play.
    releasePrefetchResources();
  }

  // Trigger re-render after update
  ctx.animationController.startAnimation();
}
