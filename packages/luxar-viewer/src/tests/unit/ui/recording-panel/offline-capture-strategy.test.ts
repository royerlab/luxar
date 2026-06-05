/**
 * Tests for OfflineCaptureStrategy — the frame-by-frame offline capture
 * orchestrator (EXR sequences + smooth turntable).
 *
 * This was the largest coverage gap in the recording-panel package
 * (6.7% statement coverage, no dedicated test file). The orchestrator's
 * job is to drive an OfflineCaptureDriver through a deterministic
 * frame loop while owning the shared scaffolding (state save/restore,
 * modal overlay, per-frame pump, progress, abort/dispose, error
 * tolerance). We therefore mock the three driver modules — the loop's
 * trust boundary, each with its own test file — and assert the
 * orchestration logic itself.
 *
 * The strategy is constructed directly (not via RecordingPanel) with a
 * fake RecordingSession, since the behavior under test is the loop, not
 * the panel wiring.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Shared driver-mock state. `vi.hoisted` so the factory is available to
// the (hoisted) vi.mock calls below.
const mockState = vi.hoisted(() => {
  const driverInstances: Array<Record<string, ReturnType<typeof vi.fn>>> = [];
  const config = { setupOk: true, captureThrows: 0, shouldAbort: false, finalizeThrows: false };
  function makeDriver() {
    let captureCalls = 0;
    const d = {
      setup: vi.fn(async () => config.setupOk),
      captureFrame: vi.fn(async () => {
        captureCalls++;
        if (captureCalls <= config.captureThrows) throw new Error('frame fail');
      }),
      finalize: vi.fn(async () => {
        if (config.finalizeThrows) throw new Error('finalize fail');
      }),
      abort: vi.fn(async () => {}),
      shouldAbort: vi.fn(() => config.shouldAbort),
    };
    driverInstances.push(d);
    return d;
  }
  return { driverInstances, config, makeDriver };
});

vi.mock('../../../../ui/recording-panel/drivers/image-sequence-driver', () => ({
  ImageSequenceDriver: vi.fn(() => mockState.makeDriver()),
}));
vi.mock('../../../../ui/recording-panel/drivers/exr-sequence-driver', () => ({
  ExrSequenceDriver: vi.fn(() => mockState.makeDriver()),
}));
vi.mock('../../../../ui/recording-panel/drivers/video-mode-driver', () => ({
  VideoModeDriver: vi.fn(() => mockState.makeDriver()),
}));

vi.mock('../../../../ui/toast', () => ({ showToast: vi.fn() }));
vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: { RECORDING: 'Recording' },
}));

import { OfflineCaptureStrategy } from '../../../../ui/recording-panel/offline-capture-strategy';
import { ImageSequenceDriver } from '../../../../ui/recording-panel/drivers/image-sequence-driver';
import { ExrSequenceDriver } from '../../../../ui/recording-panel/drivers/exr-sequence-driver';
import { VideoModeDriver } from '../../../../ui/recording-panel/drivers/video-mode-driver';
import { showToast } from '../../../../ui/toast';
import { log } from '../../../../utils/log';
import { LuxarOrbitControls } from '../../../../controls/luxar-orbit-controls';
import type { RecordingOptions } from '../../../../ui/recording-panel/types';

function makeOpts(overrides: Partial<RecordingOptions> = {}): RecordingOptions {
  return {
    outputFormat: 'png',
    imageQuality: 0.9,
    maxDPR: true,
    transparentBackground: false,
    videoDurationLimit: 60,
    videoFPS: 2, // with turntableSpeed 360 → 1s → totalFrames = 2
    videoCodec: 'h265',
    videoQuality: 'high',
    videoResolution: 0,
    syncToSlider: false,
    syncDimensionIndex: -1,
    turntableSpeed: 360,
    frameByFrame: true,
    showPanels: false,
    includeOverlays: true,
    ...overrides,
  };
}

function makeSession(overrides: Record<string, unknown> = {}): any {
  return {
    isRecording: false,
    isOfflineCaptureActive: false,
    isEXRSequenceRecording: false,
    recordingStartTime: 0,
    disposed: false,
    isDisposed(): boolean {
      return this.disposed;
    },
    showConfirmationDialog: vi.fn().mockResolvedValue(true),
    saveRecordingState: vi.fn(),
    restoreRecordingState: vi.fn(),
    pauseAutoRotate: vi.fn(),
    restoreAutoRotate: vi.fn(),
    showRecordingIndicator: vi.fn(),
    hideRecordingIndicator: vi.fn(),
    ...overrides,
  };
}

function makeSceneManager(): {
  sm: any;
  orbitControls: { applyOrbitRotation: ReturnType<typeof vi.fn> };
} {
  const orbitControls = Object.assign(Object.create(LuxarOrbitControls.prototype), {
    applyOrbitRotation: vi.fn(),
  });
  const sm = {
    controls: { getControls: vi.fn(() => orbitControls) },
  };
  return { sm: sm as any, orbitControls };
}

function makeAnimController(): any {
  return {
    // Invoke the registered callback once synchronously to simulate a
    // single rendered frame (this is what drives applyOrbitRotation).
    addPerFrameCallback: vi.fn((_id: string, cb?: () => void) => cb?.()),
    removePerFrameCallback: vi.fn(),
  };
}

function makeHooks(): any {
  return {
    hideAllPanels: vi.fn(),
    renderFrameToCanvas: vi.fn(async () => document.createElement('canvas')),
    downloadBlob: vi.fn(),
    generateFilename: vi.fn((ext: string) => `cap.${ext}`),
  };
}

describe('OfflineCaptureStrategy', () => {
  let rafSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    document.body.innerHTML = '';
    mockState.driverInstances.length = 0;
    Object.assign(mockState.config, {
      setupOk: true,
      captureThrows: 0,
      shouldAbort: false,
      finalizeThrows: false,
    });
    vi.mocked(ImageSequenceDriver).mockClear();
    vi.mocked(ExrSequenceDriver).mockClear();
    vi.mocked(VideoModeDriver).mockClear();
    vi.mocked(showToast).mockClear();
    // requestAnimationFrame resolves synchronously so the loop runs to
    // completion within the awaited run() call.
    rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        cb(0);
        return 0;
      });
  });

  afterEach(() => {
    rafSpy.mockRestore();
  });

  describe('canRun', () => {
    it('is true when idle and false during any active capture', () => {
      const { sm } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());
      expect(
        strat.canRun({
          isRecording: false,
          isOfflineCaptureActive: false,
          isCaptureInProgress: false,
        })
      ).toBe(true);
      expect(
        strat.canRun({
          isRecording: true,
          isOfflineCaptureActive: false,
          isCaptureInProgress: false,
        })
      ).toBe(false);
      expect(
        strat.canRun({
          isRecording: false,
          isOfflineCaptureActive: true,
          isCaptureInProgress: false,
        })
      ).toBe(false);
    });
  });

  describe('early exits', () => {
    it('returns without saving state when the confirmation dialog is cancelled', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession({ showConfirmationDialog: vi.fn().mockResolvedValue(false) });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      expect(session.saveRecordingState).not.toHaveBeenCalled();
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(mockState.driverInstances).toHaveLength(0);
    });

    it('bails (restoring state, no overlay) when disposed during the early rAF window', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession();
      // Dispose right as the first rAF fires — after saveRecordingState but
      // before the overlay/recording flags are brought up.
      let frames = 0;
      rafSpy.mockImplementation((cb: FrameRequestCallback) => {
        frames++;
        if (frames === 1) session.disposed = true;
        cb(0);
        return 0;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      expect(session.saveRecordingState).toHaveBeenCalledTimes(1);
      expect(session.restoreRecordingState).toHaveBeenCalled(); // bailEarly
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();
      expect(mockState.driverInstances).toHaveLength(0);
    });

    it('warns and bails when the controls are not orbit controls', async () => {
      const { sm } = makeSceneManager();
      sm.controls.getControls = vi.fn(() => ({})); // not a LuxarOrbitControls
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      expect(log.warning).toHaveBeenCalledWith('Recording', 'Turntable requires orbit controls');
      expect(session.restoreRecordingState).toHaveBeenCalled();
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();
    });
  });

  describe('driver selection', () => {
    it.each([
      ['png', () => vi.mocked(ImageSequenceDriver)],
      ['webp', () => vi.mocked(ImageSequenceDriver)],
      ['jpeg', () => vi.mocked(ImageSequenceDriver)],
      ['exr', () => vi.mocked(ExrSequenceDriver)],
      ['mp4', () => vi.mocked(VideoModeDriver)],
      ['webm', () => vi.mocked(VideoModeDriver)],
      ['mkv', () => vi.mocked(VideoModeDriver)],
    ] as const)('routes %s mode to the correct driver', async (mode, getCtor) => {
      const { sm } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());
      await strat.run(
        makeOpts({ outputFormat: mode as RecordingOptions['outputFormat'] }),
        'turntable',
        makeSession()
      );
      expect(getCtor()).toHaveBeenCalledTimes(1);
    });
  });

  describe('happy-path frame loop', () => {
    it('builds the overlay, drives the driver for every frame, then tears everything down', async () => {
      const { sm, orbitControls } = makeSceneManager();
      const session = makeSession();
      const anim = makeAnimController();
      const appendSpy = vi.spyOn(document.body, 'appendChild');
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      // videoFPS 2 + turntableSpeed 360 → totalFrames = ceil(1 * 2) = 2.
      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 360 }),
        'turntable',
        session
      );

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.setup).toHaveBeenCalledTimes(1);
      expect(driver.captureFrame).toHaveBeenCalledTimes(2);
      // finalize receives the captured-frame count.
      expect(driver.finalize).toHaveBeenCalledWith(expect.anything(), 2, expect.anything());

      // Frame 0 captures the start view; frame 1 rotates by the full step
      // (2π / (totalFrames - 1) = 2π).
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledTimes(1);
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledWith(2 * Math.PI);

      // Overlay built with modal ARIA + cancel button, and its counter
      // advanced to the final frame before teardown.
      const overlay = appendSpy.mock.calls
        .map((c) => c[0] as HTMLElement)
        .find((n) => n.className === 'luxar-recording-overlay');
      expect(overlay).toBeDefined();
      expect(overlay!.getAttribute('role')).toBe('dialog');
      expect(overlay!.getAttribute('aria-modal')).toBe('true');
      expect(overlay!.querySelector('.luxar-recording-overlay__cancel')).toBeTruthy();
      expect(overlay!.querySelector('.luxar-recording-overlay__counter')?.textContent).toBe('2/2');

      // Lifecycle: indicator shown then hidden; flags cleared; overlay
      // removed; state + auto-rotate restored; abort controller cleared.
      expect(session.showRecordingIndicator).toHaveBeenCalledTimes(1);
      expect(session.hideRecordingIndicator).toHaveBeenCalledTimes(1);
      expect(session.isRecording).toBe(false);
      expect(session.isOfflineCaptureActive).toBe(false);
      expect(session.restoreAutoRotate).toHaveBeenCalled();
      expect(session.restoreRecordingState).toHaveBeenCalled();
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();

      // Per-frame callbacks removed in the finally.
      expect(anim.removePerFrameCallback).toHaveBeenCalledWith(
        OfflineCaptureStrategy.CAPTURE_CALLBACK_ID
      );
      expect(anim.removePerFrameCallback).toHaveBeenCalledWith(
        OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID
      );
    });

    it('sets and then clears isEXRSequenceRecording across an EXR run', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession();
      // Capture the flag value at the moment the driver is set up — it must
      // be true mid-run (set before driver.setup) and false after teardown.
      let flagDuringSetup: boolean | undefined;
      mockState.config.setupOk = true;
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());
      vi.mocked(ExrSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.setup = vi.fn(async () => {
          flagDuringSetup = session.isEXRSequenceRecording;
          return true;
        });
        return d as never;
      });

      await strat.run(makeOpts({ outputFormat: 'exr' }), 'turntable', session);

      expect(flagDuringSetup).toBe(true);
      expect(session.isEXRSequenceRecording).toBe(false);
    });
  });

  describe('error + abort handling', () => {
    it('returns early and never captures when driver.setup fails', async () => {
      mockState.config.setupOk = false;
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.captureFrame).not.toHaveBeenCalled();
      expect(driver.finalize).not.toHaveBeenCalled();
      // Cleanup still runs.
      expect(session.isRecording).toBe(false);
      expect(session.restoreRecordingState).toHaveBeenCalled();
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
    });

    it('calls driver.abort and toasts when finalize throws', async () => {
      mockState.config.finalizeThrows = true;
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.finalize).toHaveBeenCalled();
      // setupCompleted && !finalizeSucceeded → driver.abort in the finally.
      expect(driver.abort).toHaveBeenCalledWith(expect.anything(), 'error');
      expect(showToast).toHaveBeenCalledWith('Recording finalize failed');
      expect(session.isRecording).toBe(false);
    });

    it('aborts capture after 3 consecutive frame failures and toasts', async () => {
      mockState.config.captureThrows = 3; // every frame throws
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      // fps 3 + speed 360 → totalFrames = 3, enough to hit the threshold.
      await strat.run(makeOpts({ videoFPS: 3, turntableSpeed: 360 }), 'turntable', session);

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.captureFrame).toHaveBeenCalledTimes(3);
      expect(showToast).toHaveBeenCalledWith(
        'HDR video encoding failed — try EXR sequence instead'
      );
    });
  });

  describe('abort() and dispose()', () => {
    it('abort() aborts the in-flight session signal with reason "user-stop"', () => {
      const { sm } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());
      const controller = new AbortController();
      strat.sessionAbort = controller;

      strat.abort();

      expect(controller.signal.aborted).toBe(true);
      expect(controller.signal.reason).toBe('user-stop');
    });

    it('dispose() aborts the session, runs overlay cleanup, and removes both per-frame callbacks', () => {
      const { sm } = makeSceneManager();
      const anim = makeAnimController();
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());
      const controller = new AbortController();
      strat.sessionAbort = controller;
      const overlayCleanup = vi.fn();
      strat.overlayCleanup = overlayCleanup;

      strat.dispose();

      expect(controller.signal.aborted).toBe(true);
      expect(controller.signal.reason).toBe('disposed');
      expect(overlayCleanup).toHaveBeenCalledTimes(1);
      expect(anim.removePerFrameCallback).toHaveBeenCalledWith(
        OfflineCaptureStrategy.CAPTURE_CALLBACK_ID
      );
      expect(anim.removePerFrameCallback).toHaveBeenCalledWith(
        OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID
      );
    });
  });
});
