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

vi.mock('../../../../ui/helpers', () => ({
  hideHelpOverlay: vi.fn(),
  clearError: vi.fn(),
  showHelpOverlay: vi.fn(),
}));

vi.mock('../../../../data', () => ({
  hideDataMonitor: vi.fn(),
}));

import { hideHelpOverlay, clearError } from '../../../../ui/helpers';
import { hideDataMonitor } from '../../../../data';
import { PanelCoordinator } from '../../../../input/handlers/panel-coordinator';
import type { RenderingControls } from '../../../../ui/rendering-controls';
import type { RecordingPanel } from '../../../../ui/recording-panel';
import type { DimensionSliders } from '../../../../ui/dimension-sliders';
import type { DebugConsole } from '../../../../ui/debug-console';

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

function makeRecordingPanel(opts: {
  isVisible?: boolean;
  isRecording?: boolean;
} = {}): {
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
    vi.mocked(hideDataMonitor).mockClear();
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('always calls hideHelpOverlay, clearError, hideDataMonitor', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    const coord = new PanelCoordinator({ debugConsole, performanceStats: stats });

    coord.closeAll();

    expect(hideHelpOverlay).toHaveBeenCalledTimes(1);
    expect(clearError).toHaveBeenCalledTimes(1);
    expect(hideDataMonitor).toHaveBeenCalledTimes(1);
  });

  it('removes #luxar-dataset-browser if present', () => {
    const browser = document.createElement('div');
    browser.id = 'luxar-dataset-browser';
    document.body.appendChild(browser);
    expect(document.getElementById('luxar-dataset-browser')).not.toBeNull();

    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    new PanelCoordinator({ debugConsole, performanceStats: stats }).closeAll();

    expect(document.getElementById('luxar-dataset-browser')).toBeNull();
  });

  it('does NOT crash when #luxar-dataset-browser is absent', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    expect(() =>
      new PanelCoordinator({ debugConsole, performanceStats: stats }).closeAll()
    ).not.toThrow();
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

  it('survives missing optional panels (renderingControls / dimensionSliders / recordingPanel = undefined)', () => {
    const { console: debugConsole } = makeDebugConsole(false);
    const { stats } = makePerformanceStats(false);
    expect(() =>
      new PanelCoordinator({ debugConsole, performanceStats: stats }).closeAll()
    ).not.toThrow();
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
    vi.mocked(hideDataMonitor).mockClear();
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
