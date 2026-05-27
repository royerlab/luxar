/**
 * Unit tests for the PanelCoordinator.
 *
 * The coordinator owns the "close everything in priority order" flow
 * that Escape (and a few other entry points) trigger. Tests exercise:
 *   - the recording-priority short-circuit on Escape;
 *   - the fullscreen-defer rule (no closeAll while fullscreen);
 *   - the priority order of closing (help first, performance stats last);
 *   - the visible-guard on each optional panel (closing a hidden panel
 *     does NOT call hide twice);
 *   - the late-binding setters (setRenderingControls /
 *     setDimensionSliders / setRecordingPanel) so the InputHandler can
 *     wire panels in as they're created.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// PanelCoordinator now goes through `notifier` for help/error UI; mock
// the notifier surface instead of the underlying ui/ helper modules.
// Spies must come from vi.hoisted to be defined when the mock factory runs.
const notifierMocks = vi.hoisted(() => ({
  hideHelp: vi.fn(),
  clearError: vi.fn(),
}));
vi.mock('../../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    hideHelp: notifierMocks.hideHelp,
    clearError: notifierMocks.clearError,
    error: vi.fn(),
    toast: vi.fn(),
    showHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
  },
}));

const hideHelpOverlay = notifierMocks.hideHelp;
const clearError = notifierMocks.clearError;
// panel-coordinator emits 'panel-hide' on the event bus instead of
// calling hideDataMonitor directly. Spy on the bus to observe the emit.
import { eventBus } from '../../../../../utils/cross-layer/event-bus';
import { PanelCoordinator } from '../../../../../input/input-handler/commands/panel-coordinator';
import type { RenderingControls } from '../../../../../ui/rendering-controls';
import type { RecordingPanel } from '../../../../../ui/recording-panel';
import type { DimensionSliders } from '../../../../../ui/dimension-sliders';
import type { DebugConsole } from '../../../../../ui/debug-console';

function makeRenderingControls(initiallyVisible = true): {
  controls: RenderingControls;
  hide: ReturnType<typeof vi.fn>;
  isVisible: ReturnType<typeof vi.fn>;
} {
  const hide = vi.fn();
  const isVisible = vi.fn(() => initiallyVisible);
  const controls = { hide, isVisible } as unknown as RenderingControls;
  return { controls, hide, isVisible };
}

function makeDimensionSliders(initiallyVisible = true): {
  sliders: DimensionSliders;
  hide: ReturnType<typeof vi.fn>;
} {
  const hide = vi.fn();
  const getIsVisible = vi.fn(() => initiallyVisible);
  const sliders = { hide, getIsVisible } as unknown as DimensionSliders;
  return { sliders, hide };
}

function makeDebugConsole(initiallyVisible = true): {
  console: DebugConsole;
  hide: ReturnType<typeof vi.fn>;
} {
  const hide = vi.fn();
  const getIsVisible = vi.fn(() => initiallyVisible);
  const console = { hide, getIsVisible } as unknown as DebugConsole;
  return { console, hide };
}

function makeRecordingPanel(
  opts: {
    isVisible?: boolean;
    isRecording?: boolean;
  } = {}
): {
  panel: RecordingPanel;
  hide: ReturnType<typeof vi.fn>;
  stopVideoRecording: ReturnType<typeof vi.fn>;
} {
  const hide = vi.fn();
  const stopVideoRecording = vi.fn();
  const isVisible = vi.fn(() => opts.isVisible ?? true);
  const isCurrentlyRecording = vi.fn(() => opts.isRecording ?? false);
  const panel = {
    hide,
    isVisible,
    isCurrentlyRecording,
    stopVideoRecording,
  } as unknown as RecordingPanel;
  return { panel, hide, stopVideoRecording };
}

function makePerformanceStats(initiallyVisible = true): {
  stats: { visible: boolean; hide(): void };
  hide: ReturnType<typeof vi.fn>;
} {
  const hide = vi.fn();
  // Cast the mock to a plain `(): void` so it matches PerformanceStatsHandle.
  const stats = { visible: initiallyVisible, hide: hide as unknown as () => void };
  return { stats, hide };
}

describe('PanelCoordinator.closeAll', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    vi.mocked(hideHelpOverlay).mockClear();
    vi.mocked(clearError).mockClear();
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('always calls hideHelpOverlay, clearError, and emits panel-hide for data-monitor', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    const hideListener = vi.fn();
    const off = eventBus.on('panel-hide', hideListener);

    try {
      coord.closeAll();
      expect(hideHelpOverlay).toHaveBeenCalledTimes(1);
      expect(clearError).toHaveBeenCalledTimes(1);
      expect(hideListener).toHaveBeenCalledWith({ panelId: 'data-monitor' });
    } finally {
      off();
    }
  });

  it('calls datasetBrowser.close() when one is registered', () => {
    const close = vi.fn();
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      datasetBrowser: { close },
    }).closeAll();

    expect(close).toHaveBeenCalledTimes(1);
  });

  it('does NOT crash when no datasetBrowser is registered, but still runs the unconditional cleanup', () => {
    // input.md [W1][P2] strengthening: previously `.not.toThrow()` only.
    // closeAll()'s contract is that the unconditional cleanups
    // (hideHelpOverlay + clearError) still fire even when no optional
    // panels (including datasetBrowser) are registered. Pin the
    // unconditional channels — a mutation that gated them behind a
    // datasetBrowser-presence check would otherwise survive.
    vi.mocked(hideHelpOverlay).mockClear();
    vi.mocked(clearError).mockClear();
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    new PanelCoordinator({ debugConsole, performanceStats: stats }).closeAll();
    expect(hideHelpOverlay).toHaveBeenCalled();
    expect(clearError).toHaveBeenCalled();
  });

  it('hides the layers panel when one is registered AND visible', () => {
    const layersHide = vi.fn();
    const layersIsVisible = vi.fn(() => true);
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      layersPanel: { isVisible: layersIsVisible, hide: layersHide },
    }).closeAll();

    expect(layersIsVisible).toHaveBeenCalled();
    expect(layersHide).toHaveBeenCalledTimes(1);
  });

  it('does NOT call layersPanel.hide() when the panel is hidden', () => {
    const layersHide = vi.fn();
    const layersIsVisible = vi.fn(() => false);
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      layersPanel: { isVisible: layersIsVisible, hide: layersHide },
    }).closeAll();

    expect(layersIsVisible).toHaveBeenCalled();
    expect(layersHide).not.toHaveBeenCalled();
  });

  it('setLayersPanel late-binds the visibility handle', () => {
    const layersHide = vi.fn();
    const layersIsVisible = vi.fn(() => true);
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    coord.setLayersPanel({ isVisible: layersIsVisible, hide: layersHide });
    coord.closeAll();
    expect(layersHide).toHaveBeenCalledTimes(1);

    coord.setLayersPanel(undefined);
    layersHide.mockClear();
    coord.closeAll();
    expect(layersHide).not.toHaveBeenCalled();
  });

  it('setDatasetBrowser late-binds and clears the close handle', () => {
    const close = vi.fn();
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    coord.setDatasetBrowser({ close });
    coord.closeAll();
    expect(close).toHaveBeenCalledTimes(1);

    coord.setDatasetBrowser(undefined);
    coord.closeAll();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it('hides only optional panels that are currently visible', () => {
    const rc = makeRenderingControls(false);
    const ds = makeDimensionSliders(false);
    const rp = makeRecordingPanel({ isVisible: false });
    const { console: debugConsole, hide: dcHide } = makeDebugConsole(false);
    const { stats, hide: psHide } = makePerformanceStats(false);

    const coord = new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      renderingControls: rc.controls,
      dimensionSliders: ds.sliders,
      recordingPanel: rp.panel,
    });
    coord.closeAll();

    expect(rc.hide).not.toHaveBeenCalled();
    expect(ds.hide).not.toHaveBeenCalled();
    expect(rp.hide).not.toHaveBeenCalled();
    expect(dcHide).not.toHaveBeenCalled();
    expect(psHide).not.toHaveBeenCalled();
  });

  it('hides every visible panel exactly once', () => {
    const rc = makeRenderingControls(true);
    const ds = makeDimensionSliders(true);
    const rp = makeRecordingPanel({ isVisible: true });
    const { console: debugConsole, hide: dcHide } = makeDebugConsole(true);
    const { stats, hide: psHide } = makePerformanceStats(true);

    const coord = new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      renderingControls: rc.controls,
      dimensionSliders: ds.sliders,
      recordingPanel: rp.panel,
    });
    coord.closeAll();

    expect(rc.hide).toHaveBeenCalledTimes(1);
    expect(ds.hide).toHaveBeenCalledTimes(1);
    expect(rp.hide).toHaveBeenCalledTimes(1);
    expect(dcHide).toHaveBeenCalledTimes(1);
    expect(psHide).toHaveBeenCalledTimes(1);
  });

  it('survives missing optional panels: cleanup still fires for the ones that ARE present', () => {
    // input.md [W1][P2] strengthening: previously `.not.toThrow()` only.
    // The "missing optional panels" contract still requires the
    // unconditional + present-panel paths to run. Verify the debug
    // console / performance stats are queried (they are present), and
    // the optional panels' undefined-guarded paths don't blow up.
    const { console: debugConsole, hide: dcHide } = makeDebugConsole(true);
    const { stats, hide: psHide } = makePerformanceStats(true);
    new PanelCoordinator({ debugConsole, performanceStats: stats }).closeAll();
    // debugConsole + performanceStats are present + visible → both hide.
    expect(dcHide).toHaveBeenCalledTimes(1);
    expect(psHide).toHaveBeenCalledTimes(1);
  });

  it('late-binds optional panels via setRenderingControls / setDimensionSliders / setRecordingPanel', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    const rc = makeRenderingControls(true);
    coord.setRenderingControls(rc.controls);
    coord.closeAll();
    expect(rc.hide).toHaveBeenCalledTimes(1);
  });

  it('clears a previously-bound panel when its setter is called with undefined', () => {
    const rc = makeRenderingControls(true);
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      renderingControls: rc.controls,
    });
    coord.setRenderingControls(undefined);
    coord.closeAll();
    expect(rc.hide).not.toHaveBeenCalled();
  });
});

describe('PanelCoordinator.handleEscape', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    vi.mocked(hideHelpOverlay).mockClear();
    vi.mocked(clearError).mockClear();
    // Default: not in fullscreen.
    Object.defineProperty(document, 'fullscreenElement', {
      configurable: true,
      get: () => null,
    });
  });

  it('calls stopVideoRecording and short-circuits when recording', () => {
    const rp = makeRecordingPanel({ isRecording: true });
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({
      debugConsole,
      performanceStats: stats,
      recordingPanel: rp.panel,
    });

    coord.handleEscape();

    expect(rp.stopVideoRecording).toHaveBeenCalledTimes(1);
    // Recording short-circuit means no closeAll cascade.
    expect(hideHelpOverlay).not.toHaveBeenCalled();
  });

  it('does NOT call closeAll when in fullscreen', () => {
    Object.defineProperty(document, 'fullscreenElement', {
      configurable: true,
      get: () => document.body,
    });
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    coord.handleEscape();
    expect(hideHelpOverlay).not.toHaveBeenCalled();
  });

  it('calls closeAll when not recording and not in fullscreen', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    coord.handleEscape();
    expect(hideHelpOverlay).toHaveBeenCalledTimes(1);
  });
});
