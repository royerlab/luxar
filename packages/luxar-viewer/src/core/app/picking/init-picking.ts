import * as THREE from 'three';
import * as zarr from '../../../data/zarr';
import { log, Modules } from '../../../utils/log';
import { getSceneLoader } from '../../../data/scene-loader-manager';
import { PickingSystem } from '../../../rendering/picking/picking-system';
import { LabelLoader, ImageLabelLoader } from '../../../data/loaders';
import { buildPickResultHandler } from './pick-result-handler';
import { PickedElementCache } from '../interaction/picked-element-cache';
import { explainLinkRejection } from '../interaction/element-actions';
import { installCanvasActions, type ElementPointerPayload } from '../interaction/canvas-actions';
import type { SceneManager } from '../../../scene/scene-manager';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Result of {@link initPicking}. All four fields are `undefined` when
 * the scene has no picking consumers / no scene loader / no LuxarScene root, in which
 * case picking is intentionally inactive for this session.
 */
export interface InitPickingResult {
  pickingSystem: PickingSystem | undefined;
  labelLoader: LabelLoader | undefined;
  imageLabelLoader: ImageLabelLoader | undefined;
  /**
   * Reads the per-element `keys` CSR (issue #1917). A `LabelLoader` on the
   * `'keys'` channel — same class, different array names — so it caches,
   * coalesces and disposes exactly like the label one.
   */
  keyLoader: LabelLoader | undefined;
}

export interface InitPickingPorts {
  sceneManager: SceneManager;
  pickingEvents: EventGroup;
  previous: InitPickingResult;
  getOverlayManager: () => OverlayManager | undefined;
  /** Optional sink for the public `selection` embedder event. */
  onSelection?: (
    sel: { nodeName: string; elementIndex: number; hitNodeName: string } | null
  ) => void;
  /**
   * Whether an embedder `selection` listener currently exists. Read at init
   * time to provision the picking pipeline even for datasets with no
   * per-element string/image channel, and read LIVE inside the shouldPick gate
   * so pick renders only run while someone consumes them.
   */
  hasSelectionConsumer?: () => boolean;
  /**
   * Whether an embedder `element-click` / `element-contextmenu` listener
   * currently exists (issue #1917).
   *
   * Separate from {@link hasSelectionConsumer} because the two answer
   * different questions and are read at different times, but they are OR'd
   * for both provisioning and gating: a host that subscribes ONLY to
   * `element-click`, on a scene with no per-element string channels and no
   * interaction templates, would otherwise get a viewer that never picks and
   * therefore an event that never fires — with nothing to indicate why.
   */
  hasElementActionConsumer?: () => boolean;
  /**
   * Whether a picked element's `link` may be opened (issue #1917). False
   * suppresses navigation, the two link menu items and the pointer cursor,
   * while leaving `Copy` working — an embedder showing scenes it did not
   * author needs the guarantee that no navigation can originate in data.
   * Defaults to true.
   */
  allowLinks?: boolean;
  /** Sinks for the public `element-click` / `element-contextmenu` events. */
  onElementClick?: (payload: ElementPointerPayload) => void;
  onElementContextMenu?: (payload: ElementPointerPayload) => void;
}

/**
 * Tear down a picking session: event listeners (DOM + Three
 * EventDispatcher, all funneled through the session's EventGroup), the
 * PickingSystem, and the string/image loaders. Idempotent — every step
 * tolerates an already-disposed / absent collaborator.
 *
 * Called from two places:
 * - `loadDataset` disposes the previous session UP-FRONT (alongside
 *   `disposeOverlays`), in lockstep with `clearSceneContent()`: the old
 *   session's nodeMap references geometries the scene clear disposes,
 *   so if the load fails mid-way the stale session (with a live
 *   embedder-selection consumer) would otherwise keep firing picks
 *   against disposed geometries until the next successful load.
 * - `initPicking` below re-runs it defensively before (re)creating the
 *   session, covering callers that reach init without the up-front
 *   dispose (a second call is a cheap no-op).
 */
export function disposePickingSession(ports: {
  pickingEvents: EventGroup;
  previous: InitPickingResult;
}): void {
  ports.pickingEvents.dispose();
  ports.previous.pickingSystem?.dispose();
  ports.previous.labelLoader?.dispose();
  ports.previous.imageLabelLoader?.dispose();
  ports.previous.keyLoader?.dispose();
}

/**
 * (Re-)build the GPU picking pipeline for the current scene.
 *
 * 1. Tear down any previous picking session (event listeners, system,
 *    loaders) so a dataset switch never leaks state.
 * 2. Walk the freshly-loaded LuxarScene root looking for text/image channels
 *    and interaction templates. Bail out early when nothing can consume a pick.
 * 3. Stand up label/key {@link LabelLoader} instances and an
 *    {@link ImageLabelLoader} backed by the scene loader's zarr store.
 * 4. Construct {@link PickingSystem} with the pick-result handler that
 *    forwards to {@link OverlayManager.updateHoverContent}.
 * 5. Wire DOM + Three.js EventDispatcher listeners (mousemove,
 *    mouseleave, sceneManager change, controls start/end, window resize,
 *    sceneManager 'camera-changed') through the supplied
 *    {@link EventGroup} so a future `dispose()` removes them in one call.
 *
 * Returns the new system + loaders; the orchestrator stores all four
 * on its own fields. The handler closures capture the locally-created
 * pickingSystem so they always see the current instance (not a stale
 * reference from a previous session).
 */
export async function initPicking(ports: InitPickingPorts): Promise<InitPickingResult> {
  // Re-init: tear down listeners from any previous picking session.
  // Usually already done up-front by `loadDataset` (see
  // disposePickingSession) — this defensive re-run (idempotent) covers
  // callers that reach init directly, and runs before the no-consumers
  // early-return so we don't leak listeners from the previous session
  // if this re-init ends up with no pick consumers in the new scene. Subsequent
  // listener registrations below all funnel through the same
  // `pickingEvents` EventGroup, so a future `dispose()` (or the next
  // initPicking call) removes them in one shot.
  disposePickingSession(ports);

  // Check if any node has hover channels or interaction templates.
  const root = ports.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
    THREE.Group | undefined;
  if (!root) {
    return {
      pickingSystem: undefined,
      labelLoader: undefined,
      imageLabelLoader: undefined,
      keyLoader: undefined,
    };
  }

  let hasAnyLabels = false;
  let hasAnyLabelIds = false;
  let hasAnyImageLabels = false;
  let hasAnyKeys = false;
  let hasAnyInteraction = false;
  const linkDiagnostics = new Map<string, { nodeName: string; rejection: string | null }>();
  root.traverse((obj) => {
    const attrs = obj.userData?.attrs;
    if (attrs?.has_labels) {
      hasAnyLabels = true;
    }
    if (attrs?.has_label_ids) {
      hasAnyLabelIds = true;
    }
    if (attrs?.has_image_labels) {
      hasAnyImageLabels = true;
    }
    if (attrs?.has_keys) {
      hasAnyKeys = true;
    }
    // A layer can carry a click action WITHOUT labels — a link built purely
    // from `{hover_index}` is perfectly usable — and such a scene auto-injects
    // no hover overlay either. Without this it would fall through the gate
    // below and never pick at all, so the link would silently never fire
    // (#1917).
    if (typeof attrs?.link === 'string' || typeof attrs?.copy === 'string') {
      hasAnyInteraction = true;
    }
    // Inspect each distinct template once. Partition adders copy non-
    // compositing attrs onto every part, so warning directly in this traversal
    // would flood the console with one identical line per leaf.
    // Deliberately only element-INDEPENDENT faults (bad scheme, relative URL,
    // over-length): a per-element miss such as an unlabelled element is normal
    // and must not log per hover. The Python writer refuses these at authoring
    // time, so reaching here means a hand-edited or third-party store.
    if (typeof attrs?.link === 'string' && !linkDiagnostics.has(attrs.link)) {
      linkDiagnostics.set(attrs.link, {
        nodeName: obj.name,
        rejection: explainLinkRejection(attrs.link),
      });
    }
  });
  for (const { nodeName, rejection } of linkDiagnostics.values()) {
    if (rejection) {
      log.warning(Modules.APP, `Invalid link template on node "${nodeName}": ${rejection}`);
    }
  }
  // Provision picking when the scene declares a per-element string/image or
  // categorical channel, declares an interaction template, or an embedder
  // selection / element-action listener exists at load time. Without any of
  // those there is no consumer, so skip the pick-mesh/GPU overhead entirely.
  const wantsSelection =
    (ports.hasSelectionConsumer?.() ?? false) || (ports.hasElementActionConsumer?.() ?? false);
  if (
    !hasAnyLabels &&
    !hasAnyLabelIds &&
    !hasAnyImageLabels &&
    !hasAnyKeys &&
    !hasAnyInteraction &&
    !wantsSelection
  ) {
    return {
      pickingSystem: undefined,
      labelLoader: undefined,
      imageLabelLoader: undefined,
      keyLoader: undefined,
    };
  }

  // Get the scene loader for store/rootLoc access
  const sceneLoader = getSceneLoader('default');
  if (!sceneLoader) {
    return {
      pickingSystem: undefined,
      labelLoader: undefined,
      imageLabelLoader: undefined,
      keyLoader: undefined,
    };
  }

  // Create content loaders from the scene loader's zarr store. The current
  // missing-store gate aborts when declared channels / interaction templates
  // are the only reasons to pick; embedder selection / element-action consumers
  // keep consumer-backed picking active without one.
  const store = sceneLoader.zarrStore;
  let labelLoader: LabelLoader | undefined;
  let imageLabelLoader: ImageLabelLoader | undefined;
  let keyLoader: LabelLoader | undefined;
  if (store) {
    const rootLoc = zarr.root(store);
    labelLoader = hasAnyLabels ? new LabelLoader(store, rootLoc) : undefined;
    imageLabelLoader = hasAnyImageLabels ? new ImageLabelLoader(store, rootLoc) : undefined;
    // Same class, `'keys'` channel — only built when some node declares one,
    // so a scene without keys pays nothing.
    keyLoader = hasAnyKeys ? new LabelLoader(store, rootLoc, 'keys') : undefined;
  } else if (!wantsSelection) {
    log.warning(Modules.APP, 'Cannot init picking: zarr store not available');
    return {
      pickingSystem: undefined,
      labelLoader: undefined,
      imageLabelLoader: undefined,
      keyLoader: undefined,
    };
  }

  // Create picking system with result callback. The handler closure
  // lives in `pick-result-handler.ts` so its branch logic (null / any
  // combination of label/key/image content / no content / fetch reject /
  // missing loaders) can be unit-tested with stub ports.
  // Retains the settled pick so a click can act on it without a fresh
  // (asynchronous, user-activation-spending) GPU readback. Session-scoped:
  // a dataset switch builds a new one alongside the new PickingSystem.
  const pickedElements = new PickedElementCache();
  // Assigned right after the system is constructed; the `onPicked` closure
  // below only runs on a real pick, which cannot happen before then.
  let canvasActions: { refreshCursor(): void } | undefined;

  const pickingSystem = new PickingSystem(
    ports.sceneManager.renderer,
    ports.sceneManager.capabilities,
    ports.sceneManager.camera,
    buildPickResultHandler({
      labelLoader,
      imageLabelLoader,
      keyLoader,
      overlayManager: ports.getOverlayManager(),
      onSelection: ports.onSelection,
      onPicked: (pick) => {
        if (pick) {
          pickedElements.store(pick, pickingSystem);
        } else {
          pickedElements.clear();
        }
        // Keep the pointer affordance in step with the tooltip.
        canvasActions?.refreshCursor();
      },
    })
  );

  // Wire NodeFactory to create pick nodes for future scene loads
  sceneLoader.nodeFactory.setPickingSystem(pickingSystem);

  // Wire post-processing for lens distortion coordinate correction
  pickingSystem.setPostProcessing(ports.sceneManager.postProcessing);

  // Retroactively register already-loaded nodes (scene loads before picking init)
  sceneLoader.nodeFactory.registerExistingSceneNodes(root);

  // Gate picks on having a consumer: a visible hover overlay (tooltips), an
  // element carrying an interaction template (click/menu), OR a live embedder
  // `selection` listener. The overlay and selection sides are read LIVE so
  // unsubscribing stops the pick renders without re-initialising the pipeline;
  // `hasAnyInteraction` is a property of the loaded scene, so it is constant
  // for the session.
  pickingSystem.setShouldPick(
    () =>
      (ports.getOverlayManager()?.hasVisibleHoverOverlay() ?? false) ||
      hasAnyInteraction ||
      (ports.hasSelectionConsumer?.() ?? false) ||
      (ports.hasElementActionConsumer?.() ?? false)
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
  ports.sceneManager.addEventListener('change', dirtyHandler);
  ports.pickingEvents.add(() => ports.sceneManager.removeEventListener('change', dirtyHandler));
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

  // Canvas click / context-menu actions on the picked element. Registered
  // through the same EventGroup, so one `pickingEvents.dispose()` still tears
  // down the whole session.
  canvasActions = installCanvasActions({
    canvas,
    events: ports.pickingEvents,
    cache: pickedElements,
    picking: pickingSystem,
    allowLinks: ports.allowLinks ?? true,
    onElementClick: ports.onElementClick,
    onElementContextMenu: ports.onElementContextMenu,
  });

  log.info(Modules.APP, 'GPU picking system initialized');
  return { pickingSystem, labelLoader, imageLabelLoader, keyLoader };
}
