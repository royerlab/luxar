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

import { notifier } from '../../utils/notifier';
import { eventBus } from '../../utils/event-bus';
import type { RenderingControls } from '../../ui/rendering-controls';
import type { RecordingPanel } from '../../ui/recording-panel';
import type { DimensionSliders } from '../../ui/panels/dimension-sliders';
import type { DebugConsole } from '../../ui/panels/debug-console';

// Avoid a hard dependency on Stats.js — only the visible/hide surface we
// touch is captured here.
interface PerformanceStatsHandle {
  readonly visible: boolean;
  hide(): void;
}

/** Optional / always-present panel handles the coordinator manages. */
export interface PanelRefs {
  /** Always present: the debug console — owned by InputHandler from construction. */
  debugConsole: DebugConsole;
  /** Always present: the animation controller's performance-stats handle. */
  performanceStats: PerformanceStatsHandle;
  /** Optional: rendering controls panel. */
  renderingControls?: RenderingControls;
  /** Optional: dimension sliders. */
  dimensionSliders?: DimensionSliders;
  /** Optional: recording panel. */
  recordingPanel?: RecordingPanel;
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

  setRenderingControls(rc: RenderingControls | undefined): void {
    this.refs.renderingControls = rc;
  }

  setDimensionSliders(sliders: DimensionSliders | undefined): void {
    this.refs.dimensionSliders = sliders;
  }

  setRecordingPanel(panel: RecordingPanel | undefined): void {
    this.refs.recordingPanel = panel;
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

    // Close dataset browser (the only panel with no instance handle —
    // looked up by element id).
    const datasetBrowser = document.getElementById('luxar-dataset-browser');
    if (datasetBrowser) {
      datasetBrowser.remove();
    }

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
    if (!document.fullscreenElement) {
      this.closeAll();
    }
  }
}

// Help overlay open is now exposed via `notifier.showHelp()` from
// `utils/notifier`. Callers should import it from there directly.
