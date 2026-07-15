import { log, Modules } from '../../../utils/log';
import { cleanupUI } from '../../../ui/ui-cleanup';
import { ThemeManager } from '../../../themes/theme-manager';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { SceneLoaderManager } from '../../../data/scene-loader-manager';
import { disposeWorkerPool } from '../../../workers/worker-pool';
import { disposeDepthSort } from '../../../rendering/depth-sort-coordinator';
import { disposeConsoleInterceptor } from '../../../utils/console-interceptor';
import { clearNotifierBackend } from '../../../utils/cross-layer/notifier';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { resetViewerContainer } from '../../../utils/viewer-container';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { PerformanceMonitor } from '../../../ui/performance-monitor';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import type { ResolutionIndicator } from '../../../ui/resolution-indicator';
import type { InputHandler } from '../../../input/input-handler';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { LayersPanel } from '../../../ui/layers';
import type { ControlRail } from '../../../ui/control-rail';
import type { ScaleBar } from '../../../ui/scale-bar';
import type { ColormapLegend } from '../../../ui/colormap-legend';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { PickingSystem } from '../../../rendering/picking/picking-system';
import type { LabelLoader, ImageLabelLoader } from '../../../data/loaders';
import type { DatasetBrowser } from '../../../ui/dataset-browser';

/**
 * Snapshot of mutable orchestrator state the dispose pipeline reads.
 * Optional-typed because dispose runs idempotently — many fields may
 * be undefined when init() threw partway through.
 */
export interface DisposePipelinePorts {
  events: EventGroup;
  pickingEvents: EventGroup;
  // Subsystems (heavy)
  sceneManager: SceneManager | undefined;
  animationController: AnimationController | undefined;
  performanceMonitor: PerformanceMonitor | undefined;
  adaptiveDPRManager: AdaptiveDPRManager | undefined;
  resolutionIndicator: ResolutionIndicator | undefined;
  inputHandler: InputHandler | undefined;
  renderingControls: RenderingControls | undefined;
  recordingPanel: RecordingPanel | undefined;
  layersPanel: LayersPanel | undefined;
  controlRail: ControlRail | undefined;
  // Overlays + picking (reset between dataset loads)
  scaleBar: ScaleBar | undefined;
  colormapLegend: ColormapLegend | undefined;
  overlayManager: OverlayManager | undefined;
  pickingSystem: PickingSystem | undefined;
  labelLoader: LabelLoader | undefined;
  imageLabelLoader: ImageLabelLoader | undefined;
  datasetBrowser: DatasetBrowser | undefined;
  // Per-field reset callbacks for the things the helper must NOT keep
  // dangling references to. The orchestrator clears its own field via
  // these so the helper never reaches into LuxarApp directly.
  clearScaleBar: () => void;
  clearColormapLegend: () => void;
  clearOverlayManager: () => void;
  clearRecordingPanel: () => void;
  clearLayersPanel: () => void;
  clearControlRail: () => void;
  clearPickingSystem: () => void;
  clearLabelLoader: () => void;
  clearImageLabelLoader: () => void;
  clearDatasetBrowser: () => void;
}

/**
 * Tear down every subsystem an init() pass could have constructed.
 *
 * Component order is the teardown contract: animation first (so the
 * loop stops emitting events before listeners disappear); overlays /
 * picking / browser before input-handler (those routes wire through
 * the input handler); scene-manager + theme-manager + cleanupUI
 * between component teardown and singleton teardown; data-monitor,
 * scene-loader, worker-pool last because earlier components hold
 * loader/worker references the singletons own.
 *
 * Each step is wrapped in safeDispose; a throwing component must NOT
 * skip later cleanup (especially singletons + workers + the manager
 * registry). Errors collect and log without bubbling.
 */
export function runDisposePipeline(ports: DisposePipelinePorts): void {
  const errors: Array<{ label: string; error: unknown }> = [];
  const safeDispose = (label: string, fn: () => void): void => {
    try {
      fn();
    } catch (error) {
      errors.push({ label, error });
      log.warning(Modules.LUXAR, `Error disposing ${label}:`, error);
    }
  };

  // Stop animation first.
  safeDispose('animationController', () => ports.animationController?.dispose());
  // Then the perf readout — it subscribes to the animation loop's
  // frame-start/frame-end bus events, so tear it down right after the loop
  // stops emitting them (otherwise a visible monitor leaks its bus
  // subscription across an embedder's mount/unmount cycle).
  safeDispose('performanceMonitor', () => ports.performanceMonitor?.dispose());
  safeDispose('adaptiveDPRManager', () => ports.adaptiveDPRManager?.dispose());
  safeDispose('resolutionIndicator', () => ports.resolutionIndicator?.dispose());
  safeDispose('scaleBar', () => {
    ports.scaleBar?.dispose();
    ports.clearScaleBar();
  });
  safeDispose('colormapLegend', () => {
    ports.colormapLegend?.dispose();
    ports.clearColormapLegend();
  });
  // pickingEvents is reusable: dispose() leaves it empty for next initPicking().
  // HIGH-12: dispose BEFORE overlayManager. The picking system's mousemove
  // handler closes over the overlay manager; disposing the overlay first
  // would leave a window in which a synchronously-dispatched mousemove can
  // hit a null-deref on the disposed overlay. Releasing the listeners first
  // closes that race.
  safeDispose('pickingEvents', () => ports.pickingEvents.dispose());
  // Clear recording-panel back-reference before overlayManager dispose.
  safeDispose('overlayManager', () => {
    ports.recordingPanel?.setOverlayManager(null);
    ports.overlayManager?.dispose();
    ports.clearOverlayManager();
  });
  safeDispose('recordingPanel', () => {
    ports.recordingPanel?.dispose();
    ports.clearRecordingPanel();
  });
  safeDispose('controlRail', () => {
    ports.controlRail?.dispose();
    ports.clearControlRail();
  });
  safeDispose('layersPanel', () => {
    ports.layersPanel?.dispose();
    ports.clearLayersPanel();
  });
  safeDispose('pickingSystem', () => {
    ports.pickingSystem?.dispose();
    ports.clearPickingSystem();
  });
  safeDispose('labelLoader', () => {
    ports.labelLoader?.dispose();
    ports.clearLabelLoader();
  });
  safeDispose('imageLabelLoader', () => {
    ports.imageLabelLoader?.dispose();
    ports.clearImageLabelLoader();
  });
  // Explicitly close any open DatasetBrowser BEFORE input-handler
  // teardown. The browser is a child panel owned by LuxarApp; without
  // this step its DOM stays attached and the PanelCoordinator close
  // handle stays bound until input-handler disposes its listeners.
  // Embedded re-init scenarios must not start with a stale browser
  // modal from the prior app.
  safeDispose('datasetBrowser', () => {
    ports.datasetBrowser?.close();
    ports.clearDatasetBrowser();
    ports.inputHandler?.setDatasetBrowser(undefined);
  });
  safeDispose('inputHandler', () => ports.inputHandler?.dispose());
  safeDispose('renderingControls', () => ports.renderingControls?.dispose());
  safeDispose('sceneManager', () => ports.sceneManager?.dispose());
  // ThemeManager: disconnects glass-refraction MutationObserver, removes
  // injected SVG filters, clears CSS custom properties.
  safeDispose('themeManager', () => ThemeManager.disposeInstance());
  safeDispose('cleanupUI', () => cleanupUI());
  // App-level event listeners (focus, visibility, beforeunload,
  // open-dataset-browser, picking-system subscriptions).
  safeDispose('events', () => ports.events.dispose());

  // Process-global singletons that hold cross-app state. Resetting them on
  // dispose keeps a serial mount/unmount cycle clean (stale dim listeners,
  // notifier backends bound to torn-down UI, and a patched host console must
  // not survive into the next init). sceneDims listeners + the notifier
  // backend are cleared before the scene/UI owners are fully gone.
  safeDispose('sceneDimsManager', () => sceneDimsManager.reset());
  safeDispose('notifierBackend', () => clearNotifierBackend());
  safeDispose('consoleInterceptor', () => disposeConsoleInterceptor());
  // Restore the viewer container to document.body and undo any
  // containing-block styles applied to a custom container. Runs after UI
  // teardown so the overlays detach from the (still valid) container first.
  safeDispose('viewerContainer', () => resetViewerContainer());

  // Three-tier singleton teardown. Monitor first (factory wiring holds
  // loader refs); loader manager drops loaders + cache stores; worker
  // pool terminates remaining workers last so any in-flight worker
  // call sees the upstream owners gone before being torn down itself.
  safeDispose('dataMonitorManager', () => DataMonitorManager.disposeInstance());
  safeDispose('sceneLoaderManager', () => SceneLoaderManager.disposeInstance());
  safeDispose('workerPool', () => disposeWorkerPool());
  // Depth-sort worker last for the same reason as the pool: any
  // in-flight sort resolves onto already-cleared coordinator state.
  safeDispose('sortWorker', () => disposeDepthSort());

  if (errors.length > 0) {
    log.error(
      Modules.LUXAR,
      `dispose(): ${errors.length} component(s) threw during teardown`,
      errors.map((e) => e.label).join(', ')
    );
  }
}
