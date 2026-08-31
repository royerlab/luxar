/**
 * Tests for gui-construction.ts — buildRecordingGUI + captureLabelForMode.
 *
 * buildRecordingGUI had 9.5% function coverage: the controllers' onChange
 * callbacks (the actual panel ↔ options wiring) were never invoked by any
 * test. These tests drive a fake GUI that captures every controller's
 * onChange handler, then fire each one and assert the resulting options
 * mutation / callback dispatch — the behavior the panel depends on.
 *
 * The fake GUI mirrors the chainable lil-gui surface the builder uses
 * (add → name → onChange, addFolder, domElement.closest/querySelector).
 * lil-gui itself is covered by ui/gui/core/*.test.ts; here we test the
 * builder's wiring, not the GUI library.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  buildRecordingGUI,
  captureLabelForMode,
} from '../../../../ui/recording-panel/ui/gui-construction';
import type { RecordingMode, RecordingOptions } from '../../../../ui/recording-panel/types';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  getMaxPixelRatio,
  setMaxPixelRatioCap,
} from '../../../../rendering/pixel-ratio-cap';

interface FakeController {
  _name: string;
  _target: Record<string, unknown>;
  _prop: string;
  _options: unknown;
  _onChange: ((val: unknown) => void) | null;
  domElement: { closest: ReturnType<typeof vi.fn>; querySelector: ReturnType<typeof vi.fn> };
  name(n: string): FakeController;
  onChange(cb: (val: unknown) => void): FakeController;
  show: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  max: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
}

function makeFakeController(
  target: Record<string, unknown>,
  prop: string,
  options: unknown,
  registry: FakeController[]
): FakeController {
  const ctrl: FakeController = {
    _name: '',
    _target: target,
    _prop: prop,
    _options: options,
    _onChange: null,
    domElement: {
      closest: vi.fn(() => ({ setAttribute: vi.fn(), classList: { add: vi.fn() } })),
      querySelector: vi.fn(() => ({ readOnly: false, style: {} as Record<string, string> })),
    },
    name(n: string) {
      this._name = n;
      return this;
    },
    onChange(cb: (val: unknown) => void) {
      this._onChange = cb;
      return this;
    },
    show: vi.fn(),
    hide: vi.fn(),
    max: vi.fn().mockReturnThis(),
    updateDisplay: vi.fn(),
  };
  registry.push(ctrl);
  return ctrl;
}

function makeFakeGui(registry: FakeController[]): any {
  return {
    domElement: {
      closest: vi.fn(() => ({ setAttribute: vi.fn(), classList: { add: vi.fn() } })),
      querySelector: vi.fn(() => null),
    },
    add: vi.fn((target: Record<string, unknown>, prop: string, options?: unknown) =>
      makeFakeController(target, prop, options, registry)
    ),
    addFolder: vi.fn(() => makeFakeGui(registry)),
    close: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    destroy: vi.fn(),
  };
}

function makeOptions(overrides: Partial<RecordingOptions> = {}): RecordingOptions {
  return {
    outputFormat: 'webp',
    imageQuality: 0.92,
    captureDPR: 1,
    transparentBackground: false,
    videoDurationLimit: 60,
    videoFPS: 30,
    videoCodec: 'h265',
    videoQuality: 'high',
    videoResolution: 0,
    syncToSlider: false,
    syncDimensionIndex: -1,
    turntableSpeed: 36,
    frameByFrame: true,
    showPanels: false,
    includeOverlays: true,
    ...overrides,
  };
}

function build(initialMode: RecordingMode = 'image') {
  const registry: FakeController[] = [];
  const gui = makeFakeGui(registry);
  const options = makeOptions();
  const deps = {
    gui,
    options,
    initialMode,
    setMode: vi.fn(),
    updateControlVisibility: vi.fn(),
    getTurntableInfo: vi.fn(() => '10.0s, 300 frames'),
    getNavigableDimensionOptions: vi.fn(() => ({ '(no dimensions)': -1 })),
    captureScreenshot: vi.fn(),
    startVideoRecording: vi.fn(),
  };
  const result = buildRecordingGUI(deps);
  const byName = (n: string) => registry.find((c) => c._name === n);
  return { registry, gui, options, deps, result, byName };
}

describe('captureLabelForMode', () => {
  it('reads "Capture" in image mode and "Record" otherwise', () => {
    expect(captureLabelForMode('image')).toBe('Capture');
    expect(captureLabelForMode('video')).toBe('Record');
    expect(captureLabelForMode('turntable')).toBe('Record');
  });
});

describe('buildRecordingGUI', () => {
  it('returns all the named controllers the panel keeps for visibility toggling', () => {
    const { result } = build();
    expect(result.formatController).toBeDefined();
    expect(result.qualityController).toBeDefined();
    expect(result.transparentController).toBeDefined();
    expect(result.videoCodecController).toBeDefined();
    expect(result.videoQualityController).toBeDefined();
    expect(result.videoDurationController).toBeDefined();
    expect(result.syncToggleController).toBeDefined();
    expect(result.syncDimensionController).toBeDefined();
    expect(result.captureController).toBeDefined();
    // Group arrays are populated (image: quality, transparent, captureDPR;
    // video: quality, resolution, duration, fps, codec, sync, syncDim;
    // turntable: info, speed, smooth).
    expect(result.imageControllers).toHaveLength(3);
    expect(result.videoControllers).toHaveLength(7);
    expect(result.turntableControllers).toHaveLength(3);
  });

  it('creates the Advanced Options folder and starts it collapsed', () => {
    const { gui } = build();
    expect(gui.addFolder).toHaveBeenCalledWith('Advanced Options');
    const folder = gui.addFolder.mock.results[0].value;
    expect(folder.close).toHaveBeenCalledTimes(1);
  });

  it('Mode onChange dispatches setMode and refreshes control visibility', () => {
    const { byName, deps } = build();
    byName('Mode')!._onChange!('video');
    expect(deps.setMode).toHaveBeenCalledWith('video');
    expect(deps.updateControlVisibility).toHaveBeenCalled();
  });

  it('Format onChange writes options.outputFormat and refreshes visibility', () => {
    const { byName, deps, options } = build();
    deps.updateControlVisibility.mockClear();
    byName('Format')!._onChange!('mp4');
    expect(options.outputFormat).toBe('mp4');
    expect(deps.updateControlVisibility).toHaveBeenCalledTimes(1);
  });

  it('Image Quality onChange writes options.imageQuality', () => {
    const { byName, options } = build();
    byName('Image Quality')!._onChange!(0.55);
    expect(options.imageQuality).toBe(0.55);
  });

  it('Transparent BG onChange writes options.transparentBackground', () => {
    const { byName, options } = build();
    byName('Transparent BG')!._onChange!(true);
    expect(options.transparentBackground).toBe(true);
  });

  it('Video Quality / Resolution / Max Duration onChange write their options', () => {
    const { byName, options } = build();
    byName('Video Quality')!._onChange!('max');
    byName('Resolution')!._onChange!(2160);
    byName('Max Duration (s)')!._onChange!(120);
    expect(options.videoQuality).toBe('max');
    expect(options.videoResolution).toBe(2160);
    expect(options.videoDurationLimit).toBe(120);
  });

  it('Frame Rate onChange writes videoFPS and recomputes the turntable info', () => {
    // fpsCtrl.onChange is re-bound after creation so it ALSO refreshes the
    // turntable info display — pin the final binding.
    const { byName, deps, options } = build();
    deps.getTurntableInfo.mockClear();
    byName('Frame Rate')!._onChange!(60);
    expect(options.videoFPS).toBe(60);
    expect(deps.getTurntableInfo).toHaveBeenCalled();
  });

  it('Turn Duration onChange converts to deg/s and refreshes the read-only info', () => {
    // The row is a DURATION (matching every other timing control in the
    // viewer) while `RecordingOptions.turntableSpeed` keeps the degrees per
    // second the capture strategies and saved presets expect: 4 s → 90 °/s.
    const { byName, deps, options } = build();
    deps.getTurntableInfo.mockClear();
    byName('Turn Duration (s)')!._onChange!(4);
    expect(options.turntableSpeed).toBe(90);
    expect(deps.getTurntableInfo).toHaveBeenCalled();
    // updateTurntableInfo() re-renders the 'Output' display controller.
    expect(byName('Output')!.updateDisplay).toHaveBeenCalled();
  });

  it('Smooth (offline) onChange writes options.frameByFrame', () => {
    const { byName, options } = build();
    byName('Smooth (offline)')!._onChange!(false);
    expect(options.frameByFrame).toBe(false);
  });

  it('Advanced controls (Show Panels / Include Overlays / Capture DPR) write their options', () => {
    const { byName, options } = build();
    byName('Show Panels')!._onChange!(true);
    byName('Include Overlays')!._onChange!(false);
    byName('Capture DPR')!._onChange!(2);
    expect(options.showPanels).toBe(true);
    expect(options.includeOverlays).toBe(false);
    expect(options.captureDPR).toBe(2);
  });

  /**
   * WYSIWYG by default: the capture ratio is seeded from the LIVE
   * on-screen ceiling on every panel build, so flipping Allow High DPR in
   * the Performance popover moves the export default with it instead of
   * stranding whatever the ceiling was when the panel was first created.
   */
  it('refreshes an untouched Capture DPR from the live on-screen ceiling', () => {
    setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
    const { options, result, byName } = build();
    options.captureDPR = null;
    setMaxPixelRatioCap(Infinity);

    result.refreshCaptureDPR();

    expect(byName('Capture DPR')!._target.captureDPR).toBe(getMaxPixelRatio());
    expect(options.captureDPR).toBeNull();
  });

  it('Codec / Dimension onChange write their options', () => {
    const { byName, options } = build();
    byName('Codec')!._onChange!('vp9');
    byName('Dimension')!._onChange!(3);
    expect(options.videoCodec).toBe('vp9');
    expect(options.syncDimensionIndex).toBe(3);
  });

  it('Sync to Slider onChange writes the flag and shows/hides the dimension dropdown', () => {
    const { byName, options, result } = build();
    const sync = byName('Sync to Slider')!;
    sync._onChange!(true);
    expect(options.syncToSlider).toBe(true);
    expect(result.syncDimensionController.show).toHaveBeenCalledTimes(1);
    sync._onChange!(false);
    expect(options.syncToSlider).toBe(false);
    expect(result.syncDimensionController.hide).toHaveBeenCalledTimes(1);
  });

  it('capture button is labelled "Capture" and invokes captureScreenshot in image mode', () => {
    const { byName, deps } = build('image');
    const btn = byName('Capture')!;
    // The button binds actions.capture; invoke it the way lil-gui would.
    (btn._target[btn._prop] as () => void)();
    expect(deps.captureScreenshot).toHaveBeenCalledTimes(1);
    expect(deps.startVideoRecording).not.toHaveBeenCalled();
  });

  it('capture button is labelled "Record" and invokes startVideoRecording in non-image mode', () => {
    const { byName, deps } = build('video');
    const btn = byName('Record')!;
    (btn._target[btn._prop] as () => void)();
    expect(deps.startVideoRecording).toHaveBeenCalledTimes(1);
    expect(deps.captureScreenshot).not.toHaveBeenCalled();
  });
});
