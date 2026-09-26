// @vitest-environment jsdom
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
import { eventBus } from '../../../../utils/cross-layer/event-bus';
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
      vi.spyOn((freshPanel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
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

    // [P2/B-C3] The video-mode slider-sync branch (video-recording-strategy.ts:204)
    // was never exercised — a regression that drops the `mode === 'video' &&
    // syncToSlider` guard would go unnoticed. Drive the real run() flow.
    it('starts slider sync when in video mode with syncToSlider enabled', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));

      (panel as any).mode = 'video';
      (panel as any).options.syncToSlider = true;
      (panel as any).options.syncDimensionIndex = 3;
      (panel as any).session.animationManager = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        play: vi.fn(),
      };
      const sliderSyncSpy = vi
        .spyOn((panel as any).session, 'startSliderSync')
        .mockImplementation(() => {});

      const recordingPromise = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));

      // Sync is wired with the configured dimension index and an abort callback.
      expect(sliderSyncSpy).toHaveBeenCalledWith(3, expect.any(Function));

      mockMediaRecorder.onstop();
      await recordingPromise;
    });

    // [P2/W4 + P11/M4] The successful onstop path: blob download, "Video saved"
    // toast, AND removal of BOTH per-frame callbacks. Deleting either
    // removePerFrameCallback line (video-recording-strategy.ts:178-179) would
    // leak stale callbacks into the next recording — pin both here. This does
    // not prove that jsdom's Vitest Blob bridge round-trips the chunk bytes.
    it('on successful onstop: downloads the webm, toasts, and removes both per-frame callbacks', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));
      const downloadBlobSpy = vi.spyOn(panel as any, 'downloadBlob');
      vi.mocked(showToast).mockClear();

      const recordingPromise = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));
      // A captured chunk so the finalized blob has content.
      (panel as any).videoRecordingStrategy.recordedChunks = [new Blob(['frame-data'])];

      panel.stopVideoRecording();
      mockMediaRecorder.onstop();
      await recordingPromise;

      expect(downloadBlobSpy).toHaveBeenCalledWith(
        expect.any(Blob),
        expect.stringMatching(/\.webm$/)
      );
      expect(showToast).toHaveBeenCalledWith('Video saved');
      expect(mockAnimController.removePerFrameCallback).toHaveBeenCalledWith('recording-keepalive');
      expect(mockAnimController.removePerFrameCallback).toHaveBeenCalledWith('recording-turntable');
    });

    it('unwinds panel and renderer state even when delivery throws', async () => {
      // A throw from the delivery half (a huge Blob, a download hook, a
      // toast) used to skip every restore below it, leaving the panel
      // hidden with DPR disabled, resize locked and the per-frame
      // callbacks still registered.
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));
      vi.spyOn(panel as any, 'downloadBlob').mockImplementation(() => {
        throw new Error('download blocked');
      });
      const restoreSpy = vi.spyOn((panel as any).session, 'restoreRecordingState');
      vi.mocked(showToast).mockClear();

      const recordingPromise = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));
      panel.stopVideoRecording();
      mockMediaRecorder.onstop();
      await recordingPromise;

      expect(showToast).toHaveBeenCalledWith('Recording finalize failed');
      expect(restoreSpy).toHaveBeenCalled();
      expect(mockAnimController.removePerFrameCallback).toHaveBeenCalledWith('recording-keepalive');
      expect((panel as any).session.isRecording).toBe(false);
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

  // The real-time path used to hand `captureStream()` the WebGL canvas,
  // which carries no DOM overlays — so "Include Overlays" was silently a
  // no-op for every Video-mode recording. These pin WHICH canvas the
  // stream comes from, because that choice is the entire fix.
  describe('overlay compositing (which canvas is captured)', () => {
    let capturedFrom: HTMLCanvasElement[];
    let origGetContext: typeof HTMLCanvasElement.prototype.getContext;

    /** A 2D context stub rich enough for the real compositeOverlays(). */
    function makeFakeCtx(): unknown {
      return {
        save: vi.fn(),
        restore: vi.fn(),
        clearRect: vi.fn(),
        drawImage: vi.fn(),
        globalAlpha: 1,
        globalCompositeOperation: 'source-over',
      };
    }

    function overlayManagerWith(count: number): any {
      return {
        getVisibleOverlays: vi.fn(() =>
          Array.from({ length: count }, () => ({
            // A bare `.luxar-overlay` matches none of the text/image/html
            // branches, so compositeOverlays only does its save/restore
            // envelope — enough to prove it ran without needing a real
            // text-metrics implementation.
            el: document.createElement('div'),
            config: { position: [0.02, 0.02], opacity: 1 },
          }))
        ),
      };
    }

    beforeEach(() => {
      capturedFrom = [];
      const record = function (this: HTMLCanvasElement) {
        capturedFrom.push(this);
        return { getTracks: () => [] };
      };
      (HTMLCanvasElement.prototype as any).captureStream = vi.fn(record);
      (mockSceneManager.renderer.domElement as any).captureStream = vi.fn(record);
      origGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = vi.fn(() =>
        makeFakeCtx()
      ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
    });

    afterEach(() => {
      HTMLCanvasElement.prototype.getContext = origGetContext;
      delete (HTMLCanvasElement.prototype as any).captureStream;
    });

    /**
     * Start a recording and let `run()` get as far as installing the real
     * onstop closure.
     *
     * The in-flight promise is returned WRAPPED: an async function that
     * returned it bare would adopt it, so the helper would not settle
     * until the recording itself finished — a 15 s timeout, not a test.
     */
    async function startRecording(): Promise<{ done: Promise<void> }> {
      const done = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));
      return { done };
    }

    it("adds the sound layer's tracks, asks for an Opus-capable mime, and releases the tap at the end", async () => {
      panel.setOverlayManager(overlayManagerWith(0));
      const audioTrack = { kind: 'audio', stop: vi.fn() };
      const audioStream = { getAudioTracks: () => [audioTrack], getTracks: () => [audioTrack] };
      const port = {
        acquire: vi.fn(() => audioStream as unknown as MediaStream),
        release: vi.fn(),
      };
      panel.setAudioCapture(port);
      const addTrack = vi.fn();
      const stopVideo = vi.fn();
      const record = function () {
        return { getTracks: () => [{ stop: stopVideo }], addTrack };
      };
      (mockSceneManager.renderer.domElement as any).captureStream = vi.fn(record);
      const asked: string[] = [];
      (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => {
        asked.push(type);
        return true;
      });

      const { done } = await startRecording();
      expect(asked[0]).toBe('video/webm;codecs=vp9,opus');
      expect(port.acquire).toHaveBeenCalledTimes(1);
      expect(addTrack).toHaveBeenCalledWith(audioTrack);
      expect((panel as any).videoRecordingStrategy.audioCaptureStream).toBe(audioStream);

      mockMediaRecorder.onstop();
      await done;
      expect(port.release).toHaveBeenCalledWith(audioStream);
      expect((panel as any).videoRecordingStrategy.audioCaptureStream).toBeNull();
    });

    it('records silently when Include Audio is off or the sound layer has nothing to give', async () => {
      panel.setOverlayManager(overlayManagerWith(0));
      const port = { acquire: vi.fn(() => null), release: vi.fn() };
      panel.setAudioCapture(port);
      const asked: string[] = [];
      (MediaRecorder as any).isTypeSupported = vi.fn((type: string) => {
        asked.push(type);
        return true;
      });

      (panel as any).options.includeAudio = false;
      let { done } = await startRecording();
      expect(port.acquire).not.toHaveBeenCalled();
      expect(asked[0]).toBe('video/webm;codecs=vp9');
      mockMediaRecorder.onstop();
      await done;

      (panel as any).options.includeAudio = true;
      ({ done } = await startRecording());
      expect(port.acquire).toHaveBeenCalledTimes(1);
      expect(asked[1]).toBe('video/webm;codecs=vp9');
      expect((panel as any).videoRecordingStrategy.audioCaptureStream).toBeNull();
      mockMediaRecorder.onstop();
      await done;
      expect(port.release).not.toHaveBeenCalled();
    });

    it('captures a mirror canvas — not the WebGL canvas — when overlays are visible', async () => {
      panel.setOverlayManager(overlayManagerWith(3));
      (panel as any).options.includeOverlays = true;

      const { done } = await startRecording();
      const gl = mockSceneManager.renderer.domElement;

      expect(capturedFrom).toHaveLength(1);
      expect(capturedFrom[0]).not.toBe(gl);
      const compositor = (panel as any).videoRecordingStrategy.liveOverlayCompositor;
      expect(compositor).not.toBeNull();
      expect(capturedFrom[0]).toBe(compositor.canvas);

      mockMediaRecorder.onstop();
      await done;
    });

    it('captures the WebGL canvas directly when there is nothing to composite', async () => {
      panel.setOverlayManager(overlayManagerWith(0));
      (panel as any).options.includeOverlays = true;

      const { done } = await startRecording();

      expect(capturedFrom[0]).toBe(mockSceneManager.renderer.domElement);
      expect((panel as any).videoRecordingStrategy.liveOverlayCompositor).toBeNull();

      mockMediaRecorder.onstop();
      await done;
    });

    it('captures the WebGL canvas directly when Include Overlays is off', async () => {
      panel.setOverlayManager(overlayManagerWith(3));
      (panel as any).options.includeOverlays = false;

      const { done } = await startRecording();

      expect(capturedFrom[0]).toBe(mockSceneManager.renderer.domElement);
      expect((panel as any).videoRecordingStrategy.liveOverlayCompositor).toBeNull();

      mockMediaRecorder.onstop();
      await done;
    });

    it('detaches the per-frame compositor when the recording ends', async () => {
      panel.setOverlayManager(overlayManagerWith(2));
      (panel as any).options.includeOverlays = true;

      const { done } = await startRecording();
      expect((panel as any).videoRecordingStrategy.liveOverlayCompositor).not.toBeNull();

      mockMediaRecorder.onstop();
      await done;

      // A `frame-end` listener that outlived its recording would blit into
      // a dead mirror canvas on every frame for the rest of the session.
      expect((panel as any).videoRecordingStrategy.liveOverlayCompositor).toBeNull();
      expect(eventBus.hasListeners('frame-end')).toBe(false);
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
        { continuous: true, phase: 'camera' }
      );
    });

    // The real-time path has the same wake-up requirement the offline path
    // does: the turntable rotates from a per-frame callback, and those only
    // run while the rAF loop is animating — which it is NOT by the time the
    // user has read the panel and confirmed the dialog (the loop idle-stops
    // after ~2s). Registering the `continuous` keep-alive keeps a RUNNING
    // loop alive but never restarts a stopped one, so without
    // `renderOnce()` the recorded video would hold the opening pose for
    // its whole length. Drive the real run() so both halves are pinned.
    it('wakes the idle-stopped loop, keeps it alive, and actually rotates', async () => {
      vi.spyOn((panel as any).session, 'showConfirmationDialog').mockResolvedValue(true);
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => ({ getTracks: vi.fn(() => []) }));
      const controls = mockSceneManager.controls.getControls();

      // Route to the real-time MediaRecorder path: turntable mode, smooth
      // capture off, WebM output.
      (panel as any).mode = 'turntable';
      (panel as any).options.frameByFrame = false;
      (panel as any).options.outputFormat = 'webm';

      const recordingPromise = panel.startVideoRecording();
      await new Promise((r) => setTimeout(r, 0));

      expect(mockAnimController.renderOnce).toHaveBeenCalled();
      // The keep-alive must be continuous — a plain callback does not stop
      // the idle timer from halting the loop two seconds in.
      expect(mockAnimController.addPerFrameCallback).toHaveBeenCalledWith(
        'recording-keepalive',
        expect.any(Function),
        { continuous: true }
      );
      // The turntable callback ran. No frame ticked here: the shared
      // animation-controller double invokes a callback at REGISTRATION
      // time, and only while the loop is running — so this assertion
      // still fails if the wake-up above is deleted (registration on a
      // stopped loop runs nothing), which is the bug being pinned.
      expect(controls.applyOrbitRotation).toHaveBeenCalled();

      mockMediaRecorder.onstop();
      await recordingPromise;
    });

    // [P2/W1] The previous test only checked the callback was registered.
    // Invoke the registered callback and verify it actually drives the orbit
    // rotation — a no-op callback body would otherwise pass the registration
    // assertion while rotating nothing.
    it('drives applyOrbitRotation on the orbit controls when a frame ticks', () => {
      const session = (panel as any).session;
      const controls = mockSceneManager.controls.getControls();
      (panel as any).videoRecordingStrategy.startTurntableRotationForTests(
        (panel as any).options,
        session
      );

      const registration = mockAnimController.addPerFrameCallback.mock.calls.find(
        (c: unknown[]) => c[0] === 'recording-turntable'
      );
      expect(registration).toBeDefined();
      const frameCallback = registration[1] as () => void;

      frameCallback();
      expect(controls.applyOrbitRotation).toHaveBeenCalled();
    });

    it('drives the dolly from the SAME progress as the rotation when it is on', () => {
      // Absolute phase from `progress` (not an accumulated increment) so a
      // dropped frame cannot leave the oscillation out of step with the turn.
      const session = (panel as any).session;
      const controls = mockSceneManager.controls.getControls();
      controls.autoDolly = true;
      controls.autoDollyPeriod = 1;
      (panel as any).videoRecordingStrategy.startTurntableRotationForTests(
        (panel as any).options,
        session
      );

      const registration = mockAnimController.addPerFrameCallback.mock.calls.find(
        (c: unknown[]) => c[0] === 'recording-turntable'
      );
      const frameCallback = registration[1] as () => void;

      frameCallback();
      expect(controls.applyOrbitDolly).toHaveBeenCalled();
      const phase = controls.applyOrbitDolly.mock.calls[0][0] as number;
      expect(phase).toBeGreaterThanOrEqual(0);
      controls.autoDolly = false;
    });

    it('leaves the dolly untouched when the user has it switched off', () => {
      const session = (panel as any).session;
      const controls = mockSceneManager.controls.getControls();
      (panel as any).videoRecordingStrategy.startTurntableRotationForTests(
        (panel as any).options,
        session
      );

      const registration = mockAnimController.addPerFrameCallback.mock.calls.find(
        (c: unknown[]) => c[0] === 'recording-turntable'
      );
      (registration[1] as () => void)();
      expect(controls.applyOrbitDolly).not.toHaveBeenCalled();
    });
  });
});
