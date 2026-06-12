import * as THREE from 'three';
import * as zarr from '../../../data/zarr';
import { log, Modules } from '../../../utils/log';
import { getSceneLoader } from '../../../data/scene-loader-manager';
import { PickingSystem } from '../../../rendering/picking/picking-system';
import { LabelLoader, ImageLabelLoader } from '../../../data/loaders';
import { buildPickResultHandler } from './pick-result-handler';
import type { SceneManager } from '../../../scene/scene-manager';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Result of {@link initPicking}. All three fields are `undefined` when
 * the scene has no labels / no zarr store / no LuxarScene root, in which
 * case picking is intentionally inactive for this session.
 */
export interface InitPickingResult {
  pickingSystem: PickingSystem | undefined;
  labelLoader: LabelLoader | undefined;
  imageLabelLoader: ImageLabelLoader | undefined;
}

export interface InitPickingPorts {
  sceneManager: SceneManager;
  pickingEvents: EventGroup;
  previous: InitPickingResult;
  getOverlayManager: () => OverlayManager | undefined;
  /** Optional sink for the public `selection` embedder event. */
  onSelection?: (sel: { nodeName: string; elementIndex: number } | null) => void;
  /**
   * Whether an embedder `selection` listener currently exists. Read at
   * init time to provision the picking pipeline even for label-less
   * datasets, and read LIVE inside the shouldPick gate so pick renders
   * only run while someone consumes them.
   */
  hasSelectionConsumer?: () => boolean;
}

/**
 * (Re-)build the GPU picking pipeline for the current scene.
 *
 * 1. Tear down any previous picking session (event listeners, system,
 *    loaders) so a dataset switch never leaks state.
 * 2. Walk the freshly-loaded LuxarScene root looking for
 *    `userData.attrs.has_labels` / `has_image_labels`. Bail out early
 *    when nothing requests labels — keeps the bench-only synthetic
 *    scenes free of picking overhead.
 * 3. Stand up new {@link LabelLoader} / {@link ImageLabelLoader} backed
 *    by the scene loader's zarr store + root location.
 * 4. Construct {@link PickingSystem} with the pick-result handler that
 *    forwards to {@link OverlayManager.updateHoverContent}.
 * 5. Wire DOM + Three.js EventDispatcher listeners (mousemove,
 *    mouseleave, controls change/start/end, window resize,
 *    sceneManager 'camera-changed') through the supplied
 *    {@link EventGroup} so a future `dispose()` removes them in one call.
 *
 * Returns the new system + loaders; the orchestrator stores all three
 * on its own fields. The handler closures capture the locally-created
 * pickingSystem so they always see the current instance (not a stale
 * reference from a previous session).
 */
export async function initPicking(ports: InitPickingPorts): Promise<InitPickingResult> {
  // Re-init: tear down listeners from any previous picking session.
  // We dispose pickingEvents here (before the no-labels early-return) so
  // we don't leak listeners from the previous session if this re-init
  // ends up with no labels in the new scene. Subsequent listener
  // registrations below all funnel through the same `pickingEvents`
  // EventGroup, so a future `dispose()` (or the next initPicking call)
  // removes them in one shot.
  ports.pickingEvents.dispose();
  ports.previous.pickingSystem?.dispose();
  ports.previous.labelLoader?.dispose();
  ports.previous.imageLabelLoader?.dispose();

  // Check if any node has labels or image labels
  const root = ports.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
    | THREE.Group
    | undefined;
  if (!root) {
    return { pickingSystem: undefined, labelLoader: undefined, imageLabelLoader: undefined };
  }

  let hasAnyLabels = false;
  let hasAnyImageLabels = false;
  root.traverse((obj) => {
    if (obj.userData?.attrs?.has_labels) {
      hasAnyLabels = true;
    }
    if (obj.userData?.attrs?.has_image_labels) {
      hasAnyImageLabels = true;
    }
  });
  // Provision picking when the scene declares labels OR an embedder
  // `selection` listener exists at load time. Without either there is no
  // consumer, so skip the pick-mesh/GPU overhead entirely (keeps the
  // bench-only synthetic scenes free of picking cost).
  const wantsSelection = ports.hasSelectionConsumer?.() ?? false;
  if (!hasAnyLabels && !hasAnyImageLabels && !wantsSelection) {
    return { pickingSystem: undefined, labelLoader: undefined, imageLabelLoader: undefined };
  }

  // Get the scene loader for store/rootLoc access
  const sceneLoader = getSceneLoader('default');
  if (!sceneLoader) {
    return { pickingSystem: undefined, labelLoader: undefined, imageLabelLoader: undefined };
  }

  // Create label loaders from the scene loader's zarr store. The loaders
  // (tooltip content) need the store; selection events do not — so a
  // missing store only aborts when labels were the sole reason to pick.
  const store = sceneLoader.zarrStore;
  let labelLoader: LabelLoader | undefined;
  let imageLabelLoader: ImageLabelLoader | undefined;
  if (store) {
    const rootLoc = zarr.root(store);
    labelLoader = hasAnyLabels ? new LabelLoader(store, rootLoc) : undefined;
    imageLabelLoader = hasAnyImageLabels ? new ImageLabelLoader(store, rootLoc) : undefined;
  } else if (!wantsSelection) {
    log.warning(Modules.APP, 'Cannot init picking: zarr store not available');
    return { pickingSystem: undefined, labelLoader: undefined, imageLabelLoader: undefined };
  }

  // Create picking system with result callback. The handler closure
  // lives in `pick-result-handler.ts` so its branch logic (null /
  // label-only / image-only / both / neither / fetch reject /
  // missing loaders) can be unit-tested with stub ports.
  const pickingSystem = new PickingSystem(
    ports.sceneManager.renderer,
    ports.sceneManager.capabilities,
    ports.sceneManager.camera,
    buildPickResultHandler({
      labelLoader,
      imageLabelLoader,
      overlayManager: ports.getOverlayManager(),
      onSelection: ports.onSelection,
    })
  );

  // Wire NodeFactory to create pick nodes for future scene loads
  sceneLoader.nodeFactory.setPickingSystem(pickingSystem);

  // Wire post-processing for lens distortion coordinate correction
  pickingSystem.setPostProcessing(ports.sceneManager.postProcessing);

  // Retroactively register already-loaded nodes (scene loads before picking init)
  sceneLoader.nodeFactory.registerExistingSceneNodes(root);

  // Gate picks on having a consumer: a visible hover overlay (tooltips) OR
  // a live embedder `selection` listener. Read LIVE so unsubscribing stops
  // the pick renders without re-initialising the pipeline.
  pickingSystem.setShouldPick(
    () =>
      (ports.getOverlayManager()?.hasVisibleHoverOverlay() ?? false) ||
      (ports.hasSelectionConsumer?.() ?? false)
  );

  // DOM events go through EventGroup.on(); Three.js EventDispatcher events
  // (controls, sceneManager) use add() with a manual remove closure since
  // their addEventListener/removeEventListener signatures aren't EventTarget.
  // The picking handler never preventDefaults, so register the mousemove
  // listener as `passive: true` (browser-hint optimization).
  const canvas = ports.sceneManager.renderer.domElement;
  const handler = (e: MouseEvent) => pickingSystem.onMouseMove(e);
  ports.pickingEvents.on(canvas, 'mousemove', handler, { passive: true });
  // mouseleave drops the pending cursor so the camera-settle re-pick
  // path doesn't fire when the cursor isn't over the viewer.
  const leaveHandler = (): void => pickingSystem.onMouseLeave();
  ports.pickingEvents.on(canvas, 'mouseleave', leaveHandler, { passive: true });

  const dirtyHandler = () => pickingSystem.markDirty();
  ports.sceneManager.controls.addEventListener('change', dirtyHandler);
  ports.pickingEvents.add(() =>
    ports.sceneManager.controls.removeEventListener('change', dirtyHandler)
  );
  ports.pickingEvents.on(window, 'resize', dirtyHandler);

  // Page scroll / layout shift moves the canvas on screen without changing
  // the 3D view. That invalidates the cached canvas rect used to map cursor
  // coordinates, so bust just the rect (cheap) rather than markDirty (which
  // would needlessly re-render the pick buffer and fade the tooltip).
  // Capture-phase catches scrolls on any ancestor scroll container; passive
  // since we never preventDefault.
  const rectInvalidate = (): void => pickingSystem.invalidateCanvasRect();
  ports.pickingEvents.on(window, 'scroll', rectInvalidate, { capture: true, passive: true });

  // Suppress picking during orbit/pan/zoom — no expensive offscreen renders
  // while the user is navigating, and fade out stale hover labels.
  const controls = ports.sceneManager.controls;
  const interactionStart = () => {
    pickingSystem.suppress(true);
    ports.getOverlayManager()?.updateHoverContent(null);
  };
  const interactionEnd = () => {
    pickingSystem.suppress(false);
  };
  controls.addEventListener('start', interactionStart);
  controls.addEventListener('end', interactionEnd);
  ports.pickingEvents.add(() => controls.removeEventListener('start', interactionStart));
  ports.pickingEvents.add(() => controls.removeEventListener('end', interactionEnd));

  // Update picking camera when perspective ↔ orthographic swap occurs
  const cameraChangedHandler = () => {
    pickingSystem.setCamera(ports.sceneManager.camera);
  };
  ports.sceneManager.addEventListener('camera-changed', cameraChangedHandler);
  ports.pickingEvents.add(() =>
    ports.sceneManager.removeEventListener('camera-changed', cameraChangedHandler)
  );

  log.info(Modules.APP, 'GPU picking system initialized (labels detected)');
  return { pickingSystem, labelLoader, imageLabelLoader };
}
