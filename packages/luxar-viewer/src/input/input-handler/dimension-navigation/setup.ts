/**
 * Dimension-navigation lifecycle bodies extracted from input-handler.ts.
 *
 * Four functions live here:
 *   - clearDimensionUI(ctx)     — dispose sliders + animation manager + listener
 *   - initDimensionSliders(ctx) — wire scene-dims manager → optional slider
 *                                 panel → animation manager → reactive listener
 *   - initAnimationManager(ctx) — construct DimensionAnimationManager + register
 *                                 AnimationShortcuts on the NAVIGATION context
 *   - updateAllNDNodes(ctx)     — push current slice through the loader
 *                                 architecture and kick the animation loop
 *
 * The orchestrator owns the mutable state (animationManager,
 * dimensionSliders, sceneDimsListener, selectedDimension) and exposes
 * it through getter/setter callbacks on `DimNavSetupCtx`. The ctx
 * never carries `this`; setup functions read live state at call time.
 *
 * @module input/input-handler/dimension-navigation/setup
 */

import * as THREE from 'three';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import {
  updateSceneForDimensions,
  prefetchSceneForDimensions,
  releasePrefetchResources,
} from '../../../data';
import { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { log, Modules } from '../../../utils/log';
import { getViewerContainer } from '../../../utils/viewer-container';
import { AnimationShortcuts } from '../key-bindings/animation-shortcuts';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { DimensionSliders, SliderConfig } from '../../../ui/dimension-sliders';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { InputContextManager } from '../context-manager';
import type { PanelCoordinator } from '../commands/panel-coordinator';

/**
 * Factory used by `initDimensionSliders` to construct the slider panel.
 * Injected from `core/app.ts` (via `InputHandler`'s constructor) so the
 * input layer never imports the concrete UI class at runtime — it only
 * knows the shape via `import type`. Closes the input → ui layer-cruiser
 * exception (see `.dependency-cruiser.cjs`'s `KNOWN_LAYER_EXCEPTIONS`).
 *
 * Re-exported from `input/input-handler.ts` as the public name.
 */
export type DimensionSlidersFactory = (config: SliderConfig) => DimensionSliders;

export interface DimNavSetupCtx {
  sceneManager: SceneManager;
  animationController: AnimationController;
  dimensionSlidersFactory: DimensionSlidersFactory | undefined;
  contextManager: InputContextManager;
  panelCoordinator: PanelCoordinator;
  recordingPanel: RecordingPanel | undefined;
  getSelectedDimension(): number;
  setSelectedDimension(value: number): void;
  getAnimationManager(): DimensionAnimationManager | undefined;
  setAnimationManager(manager: DimensionAnimationManager | undefined): void;
  getDimensionSliders(): DimensionSliders | undefined;
  setDimensionSliders(sliders: DimensionSliders | undefined): void;
  getSceneDimsListener(): (() => Promise<void>) | undefined;
  setSceneDimsListener(listener: (() => Promise<void>) | undefined): void;
}

/**
 * Clear dimension UI and reset the scene dimension manager. Called before
 * loading a new scene so the next initDimensionSliders runs against
 * fresh metadata.
 */
export function clearDimensionUI(ctx: DimNavSetupCtx): void {
  const sliders = ctx.getDimensionSliders();
  if (sliders) {
    sliders.dispose();
    ctx.setDimensionSliders(undefined);
    ctx.panelCoordinator.setDimensionSliders(undefined);
  }

  const animManager = ctx.getAnimationManager();
  if (animManager) {
    animManager.dispose();
    ctx.setAnimationManager(undefined);
  }

  // Remove the listener before resetting so a stale closure can't
  // observe a half-reset state.
  const listener = ctx.getSceneDimsListener();
  if (listener) {
    sceneDimsManager.removeListener(listener);
    ctx.setSceneDimsListener(undefined);
  }

  sceneDimsManager.reset();
  ctx.setSelectedDimension(0);
}

/**
 * Update all nD nodes (points, lines, splats) for the current slice.
 * Called from the scene-dims listener; also fired once at the end of
 * initDimensionSliders to load the initial slice.
 */
export async function updateAllNDNodes(ctx: DimNavSetupCtx): Promise<void> {
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

/**
 * Construct DimensionAnimationManager (idempotent) and register the
 * five animation-shortcut bindings on the NAVIGATION context. Reads
 * `selectedDimension` and `animationManager` live at dispatch time so
 * subsequent dim selections / manager swaps Just Work.
 */
export function initAnimationManager(ctx: DimNavSetupCtx): void {
  if (ctx.getAnimationManager()) return;

  const manager = new DimensionAnimationManager(sceneDimsManager, ctx.animationController);
  ctx.setAnimationManager(manager);

  const shortcuts = new AnimationShortcuts(ctx.contextManager, {
    getSelectedDimension: () => ctx.getSelectedDimension(),
    getAnimationManager: () => ctx.getAnimationManager(),
  });
  shortcuts.register();
}

/**
 * Initialize dimension navigation UI after scene load. Wires scene-dims
 * manager → optional slider panel → animation manager → reactive
 * listener, then triggers an initial update so data loads for the
 * starting slice. Returns early when no nD objects are found (3D-only
 * scene).
 */
export function initDimensionSliders(ctx: DimNavSetupCtx): void {
  // Initialize scene dims manager
  if (!sceneDimsManager.initFromScene(ctx.sceneManager.scene)) {
    return; // No nD objects found
  }

  const dims = sceneDimsManager.getDims();
  const dimensionRanges = sceneDimsManager.getDimensionRanges();

  if (!dims || !dimensionRanges) {
    return;
  }

  // Clean up existing sliders if any. Clearing the ref here (rather than
  // relying on the factory branch to overwrite it) keeps the no-factory
  // branch consistent — otherwise re-initialising without a factory
  // would leave `dimensionSliders` pointing at a disposed instance and
  // `showDimensionSliders()` would call `.setVisible(true)` on a corpse.
  const existingSliders = ctx.getDimensionSliders();
  if (existingSliders) {
    existingSliders.dispose();
    ctx.setDimensionSliders(undefined);
  }

  // Build the slider panel only if a factory is injected. Listener
  // wiring + animation manager + initial update are hoisted out of
  // this branch so embed callers without a slider factory still get
  // keyboard nD navigation that actually loads data.
  if (ctx.dimensionSlidersFactory) {
    const dimensionNames = sceneDimsManager.getDimensionNames();
    const dimensionUnits = sceneDimsManager.getDimensionUnits();

    const sliders = ctx.dimensionSlidersFactory({
      container: getViewerContainer(),
      dims,
      dimensionRanges,
      dimensionNames,
      dimensionUnits,
    });
    ctx.setDimensionSliders(sliders);
    ctx.panelCoordinator.setDimensionSliders(sliders);

    // Show sliders only if we have non-displayed dimensions
    sliders.setVisible(sceneDimsManager.hasNonDisplayedDimensions());
  } else {
    log.warning(
      Modules.INPUT,
      'No DimensionSliders factory provided; skipping slider construction'
    );
    ctx.panelCoordinator.setDimensionSliders(undefined);
  }

  // Initialize animation manager and register keyboard shortcuts —
  // these don't depend on the slider panel existing.
  initAnimationManager(ctx);

  // Cross-link animation manager. Slider link is null-guarded; the
  // recording-panel link runs unconditionally.
  const animManager = ctx.getAnimationManager();
  if (animManager) {
    ctx.getDimensionSliders()?.setAnimationManager(animManager);
    ctx.recordingPanel?.setAnimationManager(animManager);
  }

  // Listen for dimension changes (returns Promise for animation
  // synchronization). The slider .update() inside the callback is
  // null-guarded, so this listener works fine without a slider
  // panel. Stored on the instance so clearDimensionUI / dispose
  // can remove it cleanly. Replace any prior listener instead of
  // stacking when initDimensionSliders runs more than once.
  const prior = ctx.getSceneDimsListener();
  if (prior) {
    sceneDimsManager.removeListener(prior);
  }
  const listener = async (): Promise<void> => {
    ctx.getDimensionSliders()?.update();
    ctx.animationController.startAnimation();
    await updateAllNDNodes(ctx);
  };
  ctx.setSceneDimsListener(listener);
  sceneDimsManager.addListener(listener);

  // Trigger initial update now that listener is registered — ensures
  // data loads at the correct initial slice position whether or not
  // a slider panel exists. The result is intentionally not awaited
  // (initDimensionSliders is synchronous), but an explicit `.catch`
  // ensures rejected updates surface in the console instead of being
  // swallowed by the unhandled-rejection global.
  updateAllNDNodes(ctx).catch((err) => {
    log.error(Modules.INPUT, 'updateAllNDNodes failed', err);
  });
  ctx.animationController.startAnimation();
}
