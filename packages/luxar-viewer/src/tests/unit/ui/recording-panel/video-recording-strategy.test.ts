/**
 * Tests for VideoRecordingStrategy — the real-time MediaRecorder path.
 *
 * Covers: MediaRecorder lifecycle (start/stop), the disposed-onstop
 * branch (suppresses download + toast, still cleans up tracks), the
 * setup-failure catch path (symmetric undo of every state mutation),
 * duration-timer cleanup, captureStream track cleanup, turntable
 * rotation registration, and Panel.dispose() routing through the
 * strategy.
 *
 * AUDIT NOTE (ui.md C2): some tests below mutate
 * `(panel as any).videoRecordingStrategy.X` (e.g. `mediaRecorder =
 * mockMediaRecorder`, `captureStream = fakeStream`). This pins the
 * internal field shape of VideoRecordingStrategy; a refactor of the
 * strategy's internal state will break these tests even with the public
 * contract preserved. The `disposed-onstop branch` test below now drives
 * the real onstop closure via `panel.startVideoRecording()`; the
 * remaining internal-mutation tests are tracked for the same rewrite.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

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

// ui.md C1 fix: drop the `ui/gui` mock and exercise the real GUI library
// under jsdom.
vi.mock('../../../../config', () => ({ config: { ui: { zIndex: { recordingPanel: 1500 } } } }));
vi.mock('../../../../ui/toast', () => ({ showToast: vi.fn() }));
vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: { RECORDING: 'Recording' },
}));
vi.mock('../../../../scene/scene-dims-manager', () => ({
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

import { RecordingPanel } from '../../../../ui/recording-panel';
import { showToast } from '../../../../ui/toast';
import { createMockSceneManager, createMockAnimationController } from './_helpers';

describe('VideoRecordingStrategy', () => {
  let panel: RecordingPanel;
  let mockSceneManager: any;
  let mockAnimController: any;
  let mockMediaRecorder: any;

  beforeEach(() => {
    document.body.innerHTML = '';
    mockSceneManager = createMockSceneManager();
    mockAnimController = createMockAnimationController();
    panel = new RecordingPanel(mockSceneManager, mockAnimController);

    mockMediaRecorder = {
      start: vi.fn(),
      stop: vi.fn(),
      ondataavailable: null as any,
      onstop: null as any,
    };
    vi.stubGlobal(
      'MediaRecorder',
      vi.fn().mockImplementation(() => mockMediaRecorder)
    );
    (MediaRecorder as any).isTypeSupported = vi.fn().mockReturnValue(true);
  });

  afterEach(() => {
    panel.dispose();
    document.body.innerHTML = '';
  });

  describe('lifecycle', () => {
    it('is not recording initially', () => {
      expect(panel.isCurrentlyRecording()).toBe(false);
    });

    it('toasts and returns when no supported MIME type', async () => {
      (MediaRecorder as any).isTypeSupported = vi.fn().mockReturnValue(false);

      const freshPanel = new RecordingPanel(mockSceneManager, mockAnimController);
      await freshPanel.startVideoRecording();

      expect(showToast).toHaveBeenCalledWith('Video recording not supported in this browser');

      freshPanel.dispose();
    });

    // [ui.md/C2 / Phase F2b] Drives `panel.stopVideoRecording()` from a
    // REAL recording started via `panel.startVideoRecording()` rather than
    // hand-installing `mediaRecorder = mockMediaRecorder` on the strategy.
    // Verifies the same contract — stopVideoRecording → mediaRecorder.stop
    // — but the full call chain is now exercised.
    it('stops the MediaRecorder when stopVideoRecording is called', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      // jsdom has no MediaStream constructor, and _helpers.ts's default
      // captureStream returns `new MediaStream()` — override with a minimal
      // fake stream that has the methods VideoRecordingStrategy reads.
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));

      const recordingPromise = panel.startVideoRecording();
      // Wait one microtask so VideoRecordingStrategy.run() installs the
      // real onstop closure on mockMediaRecorder and sets isRecording=true.
      await new Promise((r) => setTimeout(r, 0));
      expect((panel as any).session.isRecording).toBe(true);

      panel.stopVideoRecording();
      expect(mockMediaRecorder.stop).toHaveBeenCalled();

      // Fire the real onstop closure so startVideoRecording resolves and
      // doesn't leak into the next test.
      mockMediaRecorder.onstop();
      await recordingPromise;
    });

    it('ignores stopVideoRecording when not recording', () => {
      panel.stopVideoRecording();
      expect(mockMediaRecorder.stop).not.toHaveBeenCalled();
    });

    // [ui.md/C2 / Phase F2c] Drives the duration-timer-cleanup path via
    // the real production code: setting `options.videoDurationLimit > 0`
    // causes VideoRecordingStrategy.run() to install a real durationTimer
    // (line ~198 of video-recording-strategy.ts), and stopVideoRecording
    // routes through abort() which clears it. Previously the test
    // hand-installed `durationTimer = setTimeout(...)` and
    // `mediaRecorder = mockMediaRecorder` directly on the strategy's
    // private fields.
    it('clears duration timer on stop', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));
      // Force the production path to install a real durationTimer.
      (panel as any).options.videoDurationLimit = 60;

      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');

      const recordingPromise = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));
      expect((panel as any).videoRecordingStrategy.durationTimer).not.toBeNull();

      panel.stopVideoRecording();

      expect(clearTimeoutSpy).toHaveBeenCalled();
      expect((panel as any).videoRecordingStrategy.durationTimer).toBeNull();

      // Fire the real onstop so startVideoRecording resolves.
      mockMediaRecorder.onstop();
      await recordingPromise;
    });
  });

  describe('captureStream cleanup', () => {
    it('stops every track on the disposed onstop branch', () => {
      const trackA = { stop: vi.fn() };
      const trackB = { stop: vi.fn() };
      const fakeStream = { getTracks: vi.fn().mockReturnValue([trackA, trackB]) };
      (panel as any).videoRecordingStrategy.captureStream = fakeStream;
      (panel as any).session.disposed = true;
      (panel as any).videoRecordingStrategy.mediaRecorder = mockMediaRecorder;

      (panel as any).videoRecordingStrategy.cleanupCaptureStream();

      expect(fakeStream.getTracks).toHaveBeenCalledTimes(1);
      expect(trackA.stop).toHaveBeenCalledTimes(1);
      expect(trackB.stop).toHaveBeenCalledTimes(1);
      expect((panel as any).videoRecordingStrategy.captureStream).toBeNull();
    });

    it('cleanupCaptureStream is idempotent when no stream is active', () => {
      (panel as any).videoRecordingStrategy.captureStream = null;
      expect(() => (panel as any).videoRecordingStrategy.cleanupCaptureStream()).not.toThrow();
      expect((panel as any).videoRecordingStrategy.captureStream).toBeNull();
    });
  });

  describe('disposed-onstop branch', () => {
    // [ui.md/C2] Drives the REAL onstop closure installed by
    // `videoRecordingStrategy.run()` instead of hand-rolling a parallel
    // closure. The previous test re-implemented lines 158-164 of
    // video-recording-strategy.ts inside the test body and so could not
    // catch a regression that drifted the production branch.
    it('stops tracks, suppresses download + toast when onstop fires after dispose', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);

      const trackA = { stop: vi.fn() };
      const trackB = { stop: vi.fn() };
      const fakeStream = { getTracks: vi.fn().mockReturnValue([trackA, trackB]) };
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => fakeStream);

      const downloadBlobSpy = vi.spyOn(panel as any, 'downloadBlob');
      vi.mocked(showToast).mockClear();

      // Kick off the real recording flow — startVideoRecording() awaits
      // an `onstopComplete` promise that resolves only when we invoke
      // mockMediaRecorder.onstop() below. Do NOT await yet.
      const recordingPromise = panel.startVideoRecording();

      // Let the async chain reach `mediaRecorder.start(100)` so the
      // production closure has been installed on mockMediaRecorder.onstop.
      await new Promise((r) => setTimeout(r, 0));
      expect(typeof mockMediaRecorder.onstop).toBe('function');

      // Pre-conditions for the disposed branch: session.disposed=true +
      // recordedChunks populated so we can verify they get cleared.
      (panel as any).session.disposed = true;
      (panel as any).videoRecordingStrategy.recordedChunks = [new Blob(['x'])];

      // Fire the REAL onstop closure.
      mockMediaRecorder.onstop();
      await recordingPromise;

      expect(trackA.stop).toHaveBeenCalledTimes(1);
      expect(trackB.stop).toHaveBeenCalledTimes(1);
      expect((panel as any).videoRecordingStrategy.captureStream).toBeNull();
      expect((panel as any).session.isRecording).toBe(false);
      expect((panel as any).videoRecordingStrategy.recordedChunks).toEqual([]);
      expect(downloadBlobSpy).not.toHaveBeenCalled();
      expect(showToast).not.toHaveBeenCalledWith('Video saved');
    });
  });

  describe('setup-failure catch', () => {
    it('restores state when canvas.captureStream throws', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => {
        throw new Error('captureStream not supported');
      });

      const restoreStateSpy = vi.spyOn((panel as any).session, 'restoreRecordingState');
      const cleanupStreamSpy = vi.spyOn(
        (panel as any).videoRecordingStrategy,
        'cleanupCaptureStream'
      );

      vi.mocked(showToast).mockClear();
      await expect(panel.startVideoRecording()).rejects.toThrow('captureStream not supported');

      expect(restoreStateSpy).toHaveBeenCalled();
      expect(cleanupStreamSpy).toHaveBeenCalled();
      expect((panel as any).session.isRecording).toBe(false);
      expect((panel as any).videoRecordingStrategy.mediaRecorder).toBeNull();
      expect(showToast).toHaveBeenCalledWith('Video recording failed to start');
    });
  });

  describe('Panel.dispose() routing', () => {
    it('stops recording when Panel disposes mid-recording', () => {
      const mockRec = { start: vi.fn(), stop: vi.fn(), onstop: null, ondataavailable: null };
      (panel as any).session.isRecording = true;
      (panel as any).videoRecordingStrategy.mediaRecorder = mockRec;

      panel.dispose();

      expect(mockRec.stop).toHaveBeenCalled();
    });

    it('clears the recording indicator on dispose', () => {
      const indicator = document.createElement('div');
      document.body.appendChild(indicator);
      (panel as any).session.recordingIndicator = indicator;
      (panel as any).session.recordingTimeInterval = setInterval(() => {}, 1000);

      panel.dispose();

      expect(indicator.parentNode).toBeNull();
    });
  });

  describe('turntable rotation', () => {
    it('registers a continuous per-frame callback when started', () => {
      (panel as any).videoRecordingStrategy.startTurntableRotationForTests(
        (panel as any).options,
        (panel as any).session
      );

      expect(mockAnimController.addPerFrameCallback).toHaveBeenCalledWith(
        'recording-turntable',
        expect.any(Function),
        { continuous: true }
      );
    });
  });
});
