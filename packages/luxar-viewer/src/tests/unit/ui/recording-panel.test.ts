// @vitest-environment jsdom
/**
 * Tests for RecordingPanel — the thin coordinator.
 *
 * Covers: visibility (show/hide/toggle), mode dispatch (which strategy
 * gets called for each mode/format), GUI controller visibility logic,
 * Panel-routed utility wrappers (downloadBlob, generateFilename), and
 * the pure-helper module integrations (computeVideoBitrate,
 * getSupportedMimeType) that the Panel re-exposes.
 *
 * Per-collaborator tests live alongside their target:
 *   - `recording-panel/session.test.ts`
 *   - `recording-panel/screenshot-strategy.test.ts`
 *   - `recording-panel/video-recording-strategy.test.ts`
 *
 * RESOLVED (ui.md C1): the previous `vi.mock('../../../ui/gui', ...)` has
 * been dropped — these tests now exercise the real GUI library, which has
 * dedicated coverage under `ui/gui/core/*.test.ts` and runs cleanly in
 * jsdom. A regression in the panel ↔ GUI wiring is now observable here
 * instead of slipping through.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  computeVideoBitrate,
  getSupportedMimeType,
} from '../../../ui/recording-panel/media-utilities';
import { log, Modules } from '../../../utils/log';

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

// ui.md C1 fix: the previous `vi.mock('../../../ui/gui', ...)` has been
// dropped. The real GUI library is constructed and exercised; jsdom
// supports its DOM operations (verified by `ui/gui/core/*.test.ts`).

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
    it('initializes session recording flags to false on construction (no in-flight recording)', () => {
      // [P2] Pin all three session flags together so a mutation that flips
      // one default is caught. (Reading private state remains brittle —
      // OOS: promote to a public observer.)
      const session = (panel as any).session;
      expect(session.isEXRSequenceRecording).toBe(false);
      expect(session.isRecording).toBe(false);
      expect(session.isOfflineCaptureActive).toBe(false);
    });

    it('stopVideoRecording on a fresh panel is a no-op that touches no strategy', () => {
      // [P4] Split from the init test: this is a distinct behavior — the
      // not-recording guard at recording-panel.ts:252 returns before any
      // strategy is reached. A mutation that drops the guard would call
      // abort() on a strategy with no recording in flight.
      const offlineAbort = vi.spyOn((panel as any).offlineCaptureStrategy, 'abort');
      const videoAbort = vi.spyOn((panel as any).videoRecordingStrategy, 'abort');
      expect(() => panel.stopVideoRecording()).not.toThrow();
      expect(offlineAbort).not.toHaveBeenCalled();
      expect(videoAbort).not.toHaveBeenCalled();
    });

    it('routes EXR format in any mode to the offline-capture strategy', async () => {
      (panel as any).options.outputFormat = 'exr';
      (panel as any).mode = 'video';

      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      // [P2] Pin the exact args, not just "was called": the coordinator
      // must hand the strategy its own options, the current mode, and the
      // shared session — a mutation that swaps any of these survives a
      // bare toHaveBeenCalled().
      expect(offlineSpy).toHaveBeenCalledWith(
        (panel as any).options,
        'video',
        (panel as any).session
      );
    });

    it('routes a non-WebM turntable format to the offline strategy even when Smooth is off', async () => {
      // mp4/mkv (and image sequences) can only be produced offline — the
      // real-time path would silently emit WebM. Smooth off must NOT
      // downgrade an mp4 turntable to a webm video.
      (panel as any).mode = 'turntable';
      (panel as any).options.frameByFrame = false;
      (panel as any).options.outputFormat = 'mp4';

      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);
      const realtimeSpy = vi
        .spyOn((panel as any).videoRecordingStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(offlineSpy).toHaveBeenCalledWith(
        (panel as any).options,
        'turntable',
        (panel as any).session
      );
      expect(realtimeSpy).not.toHaveBeenCalled();
    });

    it('routes a PNG-sequence turntable to the offline strategy even when Smooth is off', async () => {
      (panel as any).mode = 'turntable';
      (panel as any).options.frameByFrame = false;
      (panel as any).options.outputFormat = 'png';

      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(offlineSpy).toHaveBeenCalledWith(
        (panel as any).options,
        'turntable',
        (panel as any).session
      );
    });

    it('routes an EXR turntable to the offline strategy regardless of Smooth/frameByFrame', async () => {
      // [P5] EXR always routes offline (recording-panel.ts:237) — the EXR
      // check precedes the turntable branch, so even a turntable that would
      // otherwise be realtime-eligible (frameByFrame off + webm) must NOT
      // reach the real-time strategy when the format is EXR.
      (panel as any).mode = 'turntable';
      (panel as any).options.frameByFrame = false;
      (panel as any).options.outputFormat = 'exr';

      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);
      const realtimeSpy = vi
        .spyOn((panel as any).videoRecordingStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(offlineSpy).toHaveBeenCalledWith(
        (panel as any).options,
        'turntable',
        (panel as any).session
      );
      expect(realtimeSpy).not.toHaveBeenCalled();
    });

    it('routes a WebM turntable with Smooth off to the real-time strategy', async () => {
      (panel as any).mode = 'turntable';
      (panel as any).options.frameByFrame = false;
      (panel as any).options.outputFormat = 'webm';

      const realtimeSpy = vi
        .spyOn((panel as any).videoRecordingStrategy, 'run')
        .mockResolvedValue(undefined);
      const offlineSpy = vi
        .spyOn((panel as any).offlineCaptureStrategy, 'run')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(realtimeSpy).toHaveBeenCalledWith(
        (panel as any).options,
        'turntable',
        (panel as any).session
      );
      expect(offlineSpy).not.toHaveBeenCalled();
    });

    it('stopVideoRecording while isEXRSequenceRecording routes to the offline strategy abort', () => {
      (panel as any).session.isRecording = true;
      (panel as any).session.isEXRSequenceRecording = true;
      const abortSpy = vi.spyOn((panel as any).offlineCaptureStrategy, 'abort');
      const videoAbortSpy = vi.spyOn((panel as any).videoRecordingStrategy, 'abort');

      panel.stopVideoRecording();

      // [P2] Offline/EXR captures abort via the offline strategy ONLY — the
      // real-time strategy must not be touched (recording-panel.ts:254-256).
      expect(abortSpy).toHaveBeenCalled();
      expect(videoAbortSpy).not.toHaveBeenCalled();
    });

    it('stopVideoRecording for a real-time clip aborts the video strategy and logs elapsed time', () => {
      // [P2/C5] Not offline/EXR → the real-time branch at
      // recording-panel.ts:259-261 reads session.recordingStartTime to
      // compute the elapsed seconds. A strategy that forgets to set it, or
      // a broken `/1000`, would surface as NaN in the log message — pin the
      // numeric format so that regression is caught.
      const session = (panel as any).session;
      session.isRecording = true;
      session.isOfflineCaptureActive = false;
      session.isEXRSequenceRecording = false;
      session.recordingStartTime = Date.now() - 2000; // ~2.0s elapsed

      const abortSpy = vi
        .spyOn((panel as any).videoRecordingStrategy, 'abort')
        .mockImplementation(() => {});
      panel.stopVideoRecording();

      expect(abortSpy).toHaveBeenCalled();
      expect(log.info).toHaveBeenCalledWith(
        Modules.RECORDING,
        expect.stringMatching(/Stopping video recording after \d+\.\d+s/)
      );
    });

    it('stopVideoRecording logs "unknown duration" when no start time was stamped', () => {
      // Defensive guard: if isRecording is true but recordingStartTime was
      // never set (still 0), the old code logged `Date.now() - 0` as a
      // ~1.7-billion-second elapsed. The guard now reports a clear message.
      const session = (panel as any).session;
      session.isRecording = true;
      session.isOfflineCaptureActive = false;
      session.isEXRSequenceRecording = false;
      session.recordingStartTime = 0;

      vi.spyOn((panel as any).videoRecordingStrategy, 'abort').mockImplementation(() => {});
      panel.stopVideoRecording();

      expect(log.info).toHaveBeenCalledWith(
        Modules.RECORDING,
        'Stopping video recording after unknown duration...'
      );
    });
  });

  describe('render-suppression accessor', () => {
    // `isOfflineCaptureActive` deliberately stays true across the awaited
    // driver.abort(), so keying the animation loop's render-skip predicate off
    // it would leave the viewport dark through a wedged teardown. The second
    // pair is the negative control that catches exactly that miswiring.
    it('reads isLoopRenderSuppressed, not isOfflineCaptureActive', () => {
      (panel as any).session.isLoopRenderSuppressed = true;
      (panel as any).session.isOfflineCaptureActive = false;
      expect(panel.isLoopRenderSuppressed()).toBe(true);

      (panel as any).session.isLoopRenderSuppressed = false;
      (panel as any).session.isOfflineCaptureActive = true;
      expect(panel.isLoopRenderSuppressed()).toBe(false);
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

      it('with audio, prefers an Opus-capable WebM and falls back to the video-only types', () => {
        (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => type.includes('vp9'));
        expect(getSupportedMimeType(undefined, true)).toBe('video/webm;codecs=vp9,opus');
        (MediaRecorder as any).isTypeSupported = vi.fn(
          (type: string) => type.includes('vp8') && !type.includes('opus')
        );
        expect(getSupportedMimeType(undefined, true)).toBe('video/webm;codecs=vp8');
        (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => type.includes('vp9'));
        expect(getSupportedMimeType(undefined, false)).toBe('video/webm;codecs=vp9');
      });

      it('returns null when MediaRecorder undefined', () => {
        delete (globalThis as any).MediaRecorder;
        const result = getSupportedMimeType();
        expect(result).toBeNull();
      });
    });
  });

  describe('GUI updateControlVisibility', () => {
    // ui.md C1 fix: with the real GUI library, controller.hide() sets
    // `domElement.style.display = 'none'` and the public `isVisible = false`
    // (see ui/gui/controller.ts:189-193). Assert that observable instead
    // of `toHaveBeenCalled()` on a mock-spy that no longer exists.
    it('Image mode hides video and turntable controls', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'webp';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).videoControllers) {
        expect(ctrl.isVisible).toBe(false);
        expect(ctrl.domElement.style.display).toBe('none');
      }
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.isVisible).toBe(false);
        expect(ctrl.domElement.style.display).toBe('none');
      }
    });

    it('Video mode hides image controls', () => {
      (panel as any).mode = 'video';
      (panel as any).options.outputFormat = 'webm';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).imageControllers) {
        expect(ctrl.isVisible).toBe(false);
        expect(ctrl.domElement.style.display).toBe('none');
      }
    });

    it('Turntable mode shows video and turntable controls', () => {
      (panel as any).mode = 'turntable';
      (panel as any).options.outputFormat = 'mp4';
      (panel as any).updateControlVisibility();

      // Turntable mode shows the video group (resolution / fps / codec)
      // and the turntable group, but explicitly hides duration / sync /
      // syncDim (computeControlVisibility: showVideoDuration=false,
      // showSyncToggle=false, showSyncDimension=false for turntable).
      // So at LEAST the core video controls are visible — assert that
      // some are, not that all are.
      const videoVisible = (panel as any).videoControllers.filter(
        (c: { isVisible: boolean }) => c.isVisible
      );
      expect(videoVisible.length).toBeGreaterThan(0);

      // The turntable group is fully shown in turntable mode.
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.isVisible).toBe(true);
        expect(ctrl.domElement.style.display).not.toBe('none');
      }
    });

    it('Image+EXR hides quality and transparent controls', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'exr';
      (panel as any).updateControlVisibility();

      expect((panel as any).qualityController.isVisible).toBe(false);
      expect((panel as any).qualityController.domElement.style.display).toBe('none');
      expect((panel as any).transparentController.isVisible).toBe(false);
      expect((panel as any).transparentController.domElement.style.display).toBe('none');
    });

    it('action button reads "Capture" in image mode and "Record" otherwise', () => {
      (panel as any).mode = 'image';
      (panel as any).updateControlVisibility();
      expect((panel as any).captureController.label).toBe('Capture');

      (panel as any).mode = 'video';
      (panel as any).updateControlVisibility();
      expect((panel as any).captureController.label).toBe('Record');

      (panel as any).mode = 'turntable';
      (panel as any).updateControlVisibility();
      expect((panel as any).captureController.label).toBe('Record');
    });

    it('auto-corrects format when switching to Video mode', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'webp';
      (panel as any).updateControlVisibility();

      // Switch to Video mode — webp is not a valid video format, so
      // updateControlVisibility() should reset outputFormat to a video
      // default. The auto-correction is observable through the public
      // panel.options.outputFormat state.
      (panel as any).mode = 'video';
      (panel as any).updateControlVisibility();

      const correctedFormat = (panel as any).options.outputFormat as string;
      expect(['webm', 'mp4']).toContain(correctedFormat);
    });
  });
});
