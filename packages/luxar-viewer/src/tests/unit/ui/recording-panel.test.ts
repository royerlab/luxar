/**
 * Tests for RecordingPanel — the thin coordinator.
 *
 * Covers: visibility (show/hide/toggle), mode dispatch (which strategy
 * gets called for each mode/format), GUI controller visibility logic,
 * Panel-routed utility wrappers (downloadBlob, generateFilename), and
 * the pure-helper module integrations (computeVideoBitrate,
 * generateFfmpegScript, getSupportedMimeType) that the Panel re-exposes.
 *
 * Per-collaborator tests live alongside their target:
 *   - `recording-panel/session.test.ts`
 *   - `recording-panel/screenshot-strategy.test.ts`
 *   - `recording-panel/video-recording-strategy.test.ts`
 *
 * The GUI is heavily mocked (createMockController/createMockFolder).
 * E2E tests cover real GUI structure and event propagation.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  computeVideoBitrate,
  generateFfmpegScript,
  getSupportedMimeType,
} from '../../../ui/recording-panel/media-utilities';

// jsdom polyfill — required by createMockSceneManager.
if (typeof globalThis.ImageData === 'undefined') {
  (globalThis as any).ImageData = class ImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
    constructor(widthOrData: number | Uint8ClampedArray, heightOrWidth: number, height?: number) {
      if (widthOrData instanceof Uint8ClampedArray) {
        this.data = widthOrData;
        this.width = heightOrWidth;
        this.height = height ?? widthOrData.length / (4 * heightOrWidth);
      } else {
        this.width = widthOrData;
        this.height = heightOrWidth;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}

// Mock the GUI module before importing RecordingPanel
vi.mock('../../../ui/gui', () => {
  function createMockElement(): any {
    return {
      style: {},
      className: '',
      classList: { add: vi.fn(), remove: vi.fn() },
      closest: vi.fn().mockReturnValue({ classList: { add: vi.fn() }, setAttribute: vi.fn() }),
      appendChild: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      querySelector: vi.fn().mockReturnValue({ readOnly: false, style: {}, cursor: '' }),
    };
  }

  function createMockController() {
    return {
      name: vi.fn().mockReturnThis(),
      onChange: vi.fn().mockReturnThis(),
      show: vi.fn().mockReturnThis(),
      hide: vi.fn().mockReturnThis(),
      updateDisplay: vi.fn().mockReturnThis(),
      domElement: createMockElement(),
    };
  }

  function createMockFolder(): any {
    return {
      add: vi.fn().mockImplementation(() => createMockController()),
      addFolder: vi.fn().mockImplementation(() => createMockFolder()),
      close: vi.fn(),
    };
  }

  const MockGUI = vi.fn().mockImplementation(() => ({
    domElement: createMockElement(),
    add: vi.fn().mockImplementation(() => createMockController()),
    addFolder: vi.fn().mockImplementation(() => createMockFolder()),
    show: vi.fn(),
    hide: vi.fn(),
    destroy: vi.fn(),
  }));

  return {
    default: MockGUI,
    GUI: MockGUI,
    Controller: vi.fn(),
  };
});

vi.mock('../../../config', () => ({
  config: {
    ui: {
      zIndex: {
        recordingPanel: 1500,
      },
    },
  },
}));

vi.mock('../../../ui/toast', () => ({
  showToast: vi.fn(),
}));

vi.mock('../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
  Modules: { RECORDING: 'Recording' },
}));

vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn().mockReturnValue({ ndim: 4, displayed: [0, 1, 2] }),
    getDimensionNames: vi.fn().mockReturnValue(['x', 'y', 'z', 'time']),
    getDimensionRanges: vi.fn().mockReturnValue([
      [0, 100],
      [0, 100],
      [0, 100],
      [0, 50],
    ]),
    setDimensionValue: vi.fn(),
    hasNonDisplayedDimensions: vi.fn().mockReturnValue(true),
  },
}));

import { RecordingPanel } from '../../../ui/recording-panel';
import { createMockSceneManager, createMockAnimationController } from './recording-panel/_helpers';

URL.createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
URL.revokeObjectURL = vi.fn();

describe('RecordingPanel', () => {
  let panel: RecordingPanel;
  let mockSceneManager: any;
  let mockAnimController: any;

  beforeEach(() => {
    document.body.innerHTML = '';
    mockSceneManager = createMockSceneManager();
    mockAnimController = createMockAnimationController();
    panel = new RecordingPanel(mockSceneManager, mockAnimController);
  });

  afterEach(() => {
    panel.dispose();
    document.body.innerHTML = '';
  });

  describe('visibility', () => {
    it('starts hidden', () => {
      expect(panel.isVisible()).toBe(false);
    });

    it('toggles visibility via show/hide', () => {
      panel.show();
      expect(panel.isVisible()).toBe(true);
      panel.hide();
      expect(panel.isVisible()).toBe(false);
    });

    it('toggles via toggle()', () => {
      panel.toggle();
      expect(panel.isVisible()).toBe(true);
      panel.toggle();
      expect(panel.isVisible()).toBe(false);
    });
  });

  describe('EXR sequence dispatch', () => {
    it('initializes isEXRSequenceRecording to false', () => {
      expect((panel as any).session.isEXRSequenceRecording).toBe(false);
    });

    it('routes EXR format in any mode to the offline-capture strategy', async () => {
      (panel as any).options.outputFormat = 'exr';
      (panel as any).mode = 'video';

      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(offlineSpy).toHaveBeenCalled();
    });

    it('stopVideoRecording while isEXRSequenceRecording routes to the offline strategy abort', () => {
      (panel as any).session.isRecording = true;
      (panel as any).session.isEXRSequenceRecording = true;
      const abortSpy = vi.spyOn((panel as any).offlineCaptureStrategy, 'abort');

      panel.stopVideoRecording();

      expect(abortSpy).toHaveBeenCalled();
    });
  });

  describe('dependency setters', () => {
    it('routes setAnimationManager into session', () => {
      const mockAnimManager = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        play: vi.fn(),
      };
      panel.setAnimationManager(mockAnimManager as any);
      expect((panel as any).session.animationManager).toBe(mockAnimManager);
    });

    it('routes setAdaptiveDPRManager into session', () => {
      const mockDPR = { isActive: vi.fn() };
      panel.setAdaptiveDPRManager(mockDPR as any);
      expect((panel as any).session.adaptiveDPRManager).toBe(mockDPR);
    });
  });

  describe('panel-routed utility wrappers', () => {
    it('generateFilename produces a timestamped filename', () => {
      const filename = (panel as any).generateFilename('png');
      expect(filename).toMatch(/^luxar-capture-\d{4}-\d{2}-\d{2}-\d{6}\.png$/);
    });

    it('generateFilename respects the extension argument', () => {
      const filename = (panel as any).generateFilename('webm');
      expect(filename).toMatch(/\.webm$/);
    });

    it('downloadBlob routes to the helper which creates an object URL', () => {
      const blob = new Blob(['test']);
      (panel as any).downloadBlob(blob, 'test.png');
      expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    });
  });

  describe('media-utilities re-exports', () => {
    describe('computeVideoBitrate', () => {
      it('low quality at 1080p30', () => {
        const bitrate = computeVideoBitrate(1920, 1080, 30, 'low');
        expect(bitrate).toBe(Math.round(1920 * 1080 * 30 * 0.04));
      });

      it('high quality at 1080p60', () => {
        const bitrate = computeVideoBitrate(1920, 1080, 60, 'high');
        expect(bitrate).toBe(Math.round(1920 * 1080 * 60 * 0.15));
      });

      it('max quality at 4K', () => {
        const bitrate = computeVideoBitrate(3840, 2160, 60, 'max');
        expect(bitrate).toBe(Math.round(3840 * 2160 * 60 * 0.3));
      });
    });

    describe('generateFfmpegScript', () => {
      it('generates valid bash script', () => {
        const script = generateFfmpegScript(30, 300, 'png');
        expect(script.startsWith('#!/bin/bash')).toBe(true);
        expect(script).toContain('set -e');
      });

      it('uses correct frame pattern for PNG', () => {
        const script = generateFfmpegScript(60, 600, 'png');
        expect(script).toContain('frame_%06d.png');
        expect(script).toContain('framerate 60');
      });

      it('includes HDR section for EXR', () => {
        const script = generateFfmpegScript(30, 300, 'exr');
        expect(script).toContain('yuv420p10le');
        expect(script).toContain('bt2020');
      });

      it('excludes HDR section for non-EXR', () => {
        const script = generateFfmpegScript(30, 300, 'jpg');
        expect(script).not.toContain('yuv420p10le');
      });
    });

    describe('getSupportedMimeType', () => {
      beforeEach(() => {
        vi.stubGlobal('MediaRecorder', vi.fn());
      });

      it('returns VP9 mime type when supported', () => {
        (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => type.includes('vp9'));
        const result = getSupportedMimeType();
        expect(result).toBe('video/webm;codecs=vp9');
      });

      it('falls back to VP8', () => {
        (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => type.includes('vp8'));
        const result = getSupportedMimeType();
        expect(result).toBe('video/webm;codecs=vp8');
      });

      it('returns null when nothing supported', () => {
        (MediaRecorder as any).isTypeSupported = vi.fn().mockReturnValue(false);
        const result = getSupportedMimeType();
        expect(result).toBeNull();
      });

      it('returns null when MediaRecorder undefined', () => {
        delete (globalThis as any).MediaRecorder;
        const result = getSupportedMimeType();
        expect(result).toBeNull();
      });
    });
  });

  describe('GUI updateControlVisibility', () => {
    it('Image mode hides video and turntable controls', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'webp';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).videoControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
    });

    it('Video mode hides image controls', () => {
      (panel as any).mode = 'video';
      (panel as any).options.outputFormat = 'webm';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).imageControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
    });

    it('Turntable mode shows video and turntable controls', () => {
      (panel as any).mode = 'turntable';
      (panel as any).options.outputFormat = 'mp4';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).videoControllers) {
        expect(ctrl.show).toHaveBeenCalled();
      }
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.show).toHaveBeenCalled();
      }
    });

    it('Image+EXR hides quality and transparent controls', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'exr';
      (panel as any).updateControlVisibility();

      expect((panel as any).qualityController.hide).toHaveBeenCalled();
      expect((panel as any).transparentController.hide).toHaveBeenCalled();
    });

    it('auto-corrects format when switching to Video mode', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'webp';
      (panel as any).updateControlVisibility();

      // Switch to Video mode — webp is not a valid video format,
      // so updateControlVisibility should reset it to a video default.
      (panel as any).mode = 'video';
      (panel as any).updateControlVisibility();

      // The format controller's updateDisplay() is called when auto-correction fires.
      expect((panel as any).formatController.updateDisplay).toHaveBeenCalled();
    });
  });
});
