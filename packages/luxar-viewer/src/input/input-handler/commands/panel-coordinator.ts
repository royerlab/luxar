/**
 * Panel-coordination concern extracted from `input/input-handler.ts`.
 *
 * Owns the priority-ordered "close everything" flow used by the
 * Escape key. The InputHandler holds a single `PanelCoordinator`
 * instance configured with the optional UI components (rendering
 * controls, dimension sliders, recording panel, debug console,
 * performance stats) plus the always-present hide helpers
 * (help overlay, error toast, data-monitor).
 *
 * Behavior is identical to the inline `closeAllPanels()` /
 * `handleEscapeKey()` originals: the same close order, the same
 * recording-priority short-circuit, the same fullscreen-defer rule.
 *
 * @module input/handlers/panel-coordinator
 */

import { notifier } from '../../../utils/cross-layer/notifier';
import { eventBus } from '../../../utils/cross-layer/event-bus';
import { isDocumentFullscreen } from '../../../utils/fullscreen';
import type {
  CloseableHandle,
  DebugConsoleHandle,
  DimensionSlidersHandle,
  PerformanceStatsHandle,
  RecordingPanelHandle,
  RenderingControlsHandle,
  VisiblyHideableHandle,
} from '../panel-capabilities';

/** Optional / always-present panel handles the coordinator manages. */
export interface PanelRefs {
  /** Always present: the debug console — owned by InputHandler from construction. */
  debugConsole: DebugConsoleHandle;
  /** Always present: the animation controller's performance-stats handle. */
  performanceStats: PerformanceStatsHandle;
  /** Optional: rendering controls panel. */
  renderingControls?: RenderingControlsHandle;
  /** Optional: dimension sliders. */
  dimensionSliders?: DimensionSlidersHandle;
  /** Optional: recording panel. */
  recordingPanel?: RecordingPanelHandle;
  /**
   * Optional: dataset browser, exposed only as a close handle so the
   * Escape path runs the panel's `close()` method (which fires
   * `onClose` to clear the owner's ref) instead of just yanking the
   * DOM element. Without this, Escape leaves `LuxarApp.datasetBrowser`
   * dangling and the `O` shortcut becomes a silent no-op.
   */
  datasetBrowser?: CloseableHandle;
  /**
   * Optional: layers panel. Wired so the close button's advertised
   * `aria-keyshortcuts="escape"` closes the panel through the same
   * coordinator path as other panels.
   */
  layersPanel?: VisiblyHideableHandle;
}

/**
 * Coordinator for "close all panels" and Escape-key behavior.
 *
 * Holds setter-style updaters for the panels that are wired in
 * lazily by the InputHandler (`setRenderingControls`,
 * `setDimensionSliders`, `setRecordingPanel`) — these match the
 * `setRenderingControls()` / `setDimensionSliders()` /
 * `setRecordingPanel()` setters on InputHandler. Always-present
 * panels (debugConsole, performanceStats) are passed at construction
 * time and never replaced.
 */
export class PanelCoordinator {
  private refs: PanelRefs;

  constructor(refs: PanelRefs) {
    this.refs = refs;
  }

  setRenderingControls(rc: RenderingControlsHandle | undefined): void {
    this.refs.renderingControls = rc;
  }

  setDimensionSliders(sliders: DimensionSlidersHandle | undefined): void {
    this.refs.dimensionSliders = sliders;
  }

  setRecordingPanel(panel: RecordingPanelHandle | undefined): void {
    this.refs.recordingPanel = panel;
  }

  /**
   * Set or clear the dataset-browser close handle. Owners populate it
   * when opening the browser and clear it from the browser's own
   * `onClose` callback.
   */
  setDatasetBrowser(browser: CloseableHandle | undefined): void {
    this.refs.datasetBrowser = browser;
  }

  /** Set or clear the layers-panel show/hide handle. */
  setLayersPanel(panel: VisiblyHideableHandle | undefined): void {
    this.refs.layersPanel = panel;
  }

  /**
   * Close all open UI panels and overlays in priority order
   * (topmost first):
   *
   *   1. Help overlay
   *   2. Error toast
   *   3. Dataset browser (DOM lookup by id)
   *   4. Rendering controls
   *   5. Data loading monitor
   *   6. Dimension sliders
   *   7. Debug console
   *   8. Recording panel
   *   9. Performance stats
   *
   * Each step is guarded so already-hidden panels are no-ops; the
   * order matches the inline original byte-for-byte so any visual
   * "topmost wins" expectation users developed survives.
   */
  closeAll(): void {
    // Close help overlay (usually topmost) — `hideHelpOverlay` also
    // cleans up its click-outside listener.
    notifier.hideHelp();

    // Close error messages
    notifier.clearError();

    // Close dataset browser via its own close() method so onClose fires
    // and the owner's reference (LuxarApp.datasetBrowser) is cleared;
    // otherwise the `O` reopen shortcut can see a dangling ref.
    this.refs.datasetBrowser?.close();

    if (this.refs.renderingControls?.isVisible()) {
      this.refs.renderingControls.hide();
    }

    eventBus.emit('panel-hide', { panelId: 'data-monitor' });

    if (this.refs.dimensionSliders?.getIsVisible()) {
      this.refs.dimensionSliders.hide();
    }

    if (this.refs.debugConsole.getIsVisible()) {
      this.refs.debugConsole.hide();
    }

    if (this.refs.recordingPanel?.isVisible()) {
      this.refs.recordingPanel.hide();
    }

    if (this.refs.layersPanel?.isVisible()) {
      this.refs.layersPanel.hide();
    }

    if (this.refs.performanceStats.visible) {
      this.refs.performanceStats.hide();
    }
  }

  /**
   * Escape-key handler with context-aware behavior:
   *
   *   - If the recording panel is currently recording, stop the
   *     recording and return — recording takes priority over panel
   *     close.
   *   - If we are in fullscreen, do nothing — the browser handles
   *     Escape natively to exit fullscreen.
   *   - Otherwise, call `closeAll()`.
   */
  handleEscape(): void {
    if (this.refs.recordingPanel?.isCurrentlyRecording()) {
      this.refs.recordingPanel.stopVideoRecording();
      return;
    }
    // Use the cross-browser check: on Safari <16.4 fullscreen is entered via
    // the webkit API, so `document.fullscreenElement` alone is null and Escape
    // would wrongly closeAll() while also natively exiting fullscreen.
    if (!isDocumentFullscreen()) {
      this.closeAll();
    }
  }
}

// Help overlay open is now exposed via `notifier.showHelp()` from
// `utils/cross-layer/notifier`. Callers should import it from there directly.
