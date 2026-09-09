/**
 * Dimension-navigation lifecycle bodies extracted from input-handler.ts.
 *
 * Three functions live here:
 *   - clearDimensionUI(ctx)     — dispose sliders + animation manager + listener
 *   - initDimensionSliders(ctx) — wire scene-dims manager → optional slider
 *                                 panel → animation manager → reactive listener
 *   - initAnimationManager(ctx) — construct DimensionAnimationManager
 *
 * The orchestrator owns the mutable state (animationManager,
 * dimensionSliders, sceneDimsListener, selectedDimension) and exposes
 * it through getter/setter callbacks on `DimNavSetupCtx`. The ctx
 * never carries `this`; setup functions read live state at call time.
 *
 * @module input/input-handler/dimension-navigation/setup
 */

import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { updateAllNDNodes, type DimensionLoadingContext } from '../../../scene/dimension-loading';
import { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { worstCommittedEnergy, type QualityNode } from '../../../scene/animation/committed-quality';
import { log, Modules } from '../../../utils/log';
import { getViewerContainer } from '../../../utils/viewer-container';
import type { PanelCoordinator } from '../commands/panel-coordinator';
import type {
  DimensionSlidersFactory,
  DimensionSlidersHandle,
  RecordingPanelHandle,
} from '../panel-capabilities';

/** Mutable dependencies and state accessors for dimension-navigation setup. */
export interface DimNavSetupCtx extends DimensionLoadingContext {
  dimensionSlidersFactory: DimensionSlidersFactory | undefined;
  panelCoordinator: PanelCoordinator;
  recordingPanel: RecordingPanelHandle | undefined;
  getSelectedDimension(): number;
  setSelectedDimension(value: number): void;
  setAnimationManager(manager: DimensionAnimationManager | undefined): void;
  getDimensionSliders(): DimensionSlidersHandle | undefined;
  setDimensionSliders(sliders: DimensionSlidersHandle | undefined): void;
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
 * Construct DimensionAnimationManager (idempotent).
 */
export function initAnimationManager(ctx: DimNavSetupCtx): void {
  if (ctx.getAnimationManager()) return;

  // The pacing feedback needs to tell "loaders cannot keep up" from "the
  // playhead slowed to wait for them", and only the scene knows which (#2374).
  // Read live rather than captured: the scene is rebuilt across dataset loads.
  const manager = new DimensionAnimationManager(sceneDimsManager, ctx.animationController, () =>
    worstCommittedEnergy(ctx.sceneManager.scene as unknown as QualityNode)
  );
  ctx.setAnimationManager(manager);
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
      selectedDimension: ctx.getSelectedDimension(),
      onSelectDimension: (navigableIndex) => ctx.setSelectedDimension(navigableIndex),
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
