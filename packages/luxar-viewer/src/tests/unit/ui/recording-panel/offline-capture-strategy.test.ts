// @vitest-environment jsdom
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
    isLoopRenderSuppressed: false,
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

/**
 * Animation-controller double that models the real idle behaviour: the
 * rAF loop starts STOPPED (the viewer idle-stops after ~2s of no
 * interaction, which is the normal state by the time a user has read
 * the Recording panel and confirmed the dialog), and per-frame
 * callbacks only fire while it is running. Registering a `continuous`
 * callback does NOT restart a stopped loop — only startAnimation()
 * does. A double that fires callbacks unconditionally cannot see the
 * "turntable captures N identical frames" bug at all.
 *
 * Registration QUEUES a callback; it never runs it. The real loop runs
 * every registered callback once per animation frame, so `__tick()`
 * (driven from this file's requestAnimationFrame spy) is the only thing
 * that invokes them. Firing at registration time instead would hide the
 * mirror-image bug: registering the capture callback AFTER the awaited
 * frame — so it is added and removed with no tick in between — leaves
 * `applyOrbitRotation` uncalled and every captured frame on the opening
 * pose, which a registration-time double reports as a healthy sweep.
 */
/** The controller double the requestAnimationFrame spy ticks (each test
 *  builds exactly one). */
let activeAnim: { __tick(): void } | null = null;

function makeAnimController({ animating = false }: { animating?: boolean } = {}): any {
  let isAnimating = animating;
  const callbacks = new Map<string, () => void>();
  const anim = {
    startAnimation: vi.fn(() => {
      isAnimating = true;
    }),
    // The options argument is recorded by vi.fn() so `{ continuous: true }`
    // stays assertable.
    addPerFrameCallback: vi.fn((id: string, cb?: () => void, _opts?: unknown) => {
      if (cb) callbacks.set(id, cb);
    }),
    removePerFrameCallback: vi.fn((id: string) => callbacks.delete(id)),
    /** Simulate one rendered frame: run every registered callback — but
     *  only while the loop is actually animating. */
    __tick: (): void => {
      if (!isAnimating) return;
      for (const cb of [...callbacks.values()]) cb();
    },
  };
  activeAnim = anim;
  return anim;
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
    activeAnim = null;
    // requestAnimationFrame resolves synchronously so the loop runs to
    // completion within the awaited run() call. Each simulated frame runs
    // the controller's registered per-frame callbacks first, exactly as
    // the real loop does — registration alone never invokes them.
    rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        activeAnim?.__tick();
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
      const anim = makeAnimController();
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      expect(session.saveRecordingState).toHaveBeenCalledTimes(1);
      expect(session.restoreRecordingState).toHaveBeenCalled(); // bailEarly
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();
      expect(mockState.driverInstances).toHaveLength(0);
      // …but the bail must NOT wake the loop on a disposed session: the
      // AnimationController is torn down before the RecordingPanel.
      expect(anim.startAnimation).not.toHaveBeenCalled();
    });

    it('warns and bails when the controls are not orbit controls, repainting the cleared canvas', async () => {
      const { sm } = makeSceneManager();
      sm.controls.getControls = vi.fn(() => ({})); // not a LuxarOrbitControls
      const session = makeSession();
      const anim = makeAnimController();
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      await strat.run(makeOpts(), 'turntable', session);

      expect(log.warning).toHaveBeenCalledWith('Recording', 'Turntable requires orbit controls');
      expect(session.restoreRecordingState).toHaveBeenCalled();
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();
      // Reachable without any dispose (fly controls + smooth turntable):
      // restoreRecordingState resized the render target back, clearing the
      // canvas, so the bail owes the viewer exactly one repaint — otherwise
      // it stays blank until the next mouse move.
      expect(anim.startAnimation).toHaveBeenCalled();
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
    it('wakes the idle-stopped rAF loop so the turntable actually rotates', async () => {
      const { sm, orbitControls } = makeSceneManager();
      // Loop stopped by the idle timer while the user read the panel.
      const anim = makeAnimController({ animating: false });
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      // videoFPS 2 + turntableSpeed 120 → totalFrames = ceil(3 * 2) = 6.
      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 120 }),
        'turntable',
        makeSession()
      );

      // The loop has to be woken explicitly: registering a `continuous`
      // keep-alive callback keeps a RUNNING loop alive but never
      // restarts a stopped one.
      expect(anim.startAnimation).toHaveBeenCalled();

      // …and the keep-alive is the other half, load-bearing past the
      // first two seconds: startAnimation() arms the idle timer and
      // nothing in the capture loop re-arms it, so a `continuous`
      // callback is the ONLY thing that stops the timer halting the
      // loop mid-capture. Dropping the option (or the registration)
      // re-creates the identical-frames bug for every frame after ~2s.
      expect(anim.addPerFrameCallback).toHaveBeenCalledWith(
        OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID,
        expect.any(Function),
        { continuous: true }
      );

      // Frames 1..5 each advance one step — without the wake-up the
      // camera never moves and every captured frame is identical. The step
      // is 2π/6, so the six frames cover [0, 2π) and the last one stops
      // short of the first (a turntable has to loop seamlessly).
      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.captureFrame).toHaveBeenCalledTimes(6);
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledTimes(5);
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledWith((2 * Math.PI) / 6);

      // The full sweep must stop exactly one step short of a whole turn —
      // 5 × 2π/6, never 2π.
      const swept = orbitControls.applyOrbitRotation.mock.calls.reduce(
        (sum: number, call: unknown[]) => sum + (call[0] as number),
        0
      );
      expect(swept).toBeCloseTo(2 * Math.PI * (5 / 6), 12);
    });

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

      // Frame 0 captures the start view; frame 1 rotates by one step
      // (2π / totalFrames = π), landing half a turn away rather than back
      // on the start pose.
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledTimes(1);
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledWith(Math.PI);

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

    it('re-wakes the rAF loop after teardown so the cleared canvas gets repainted', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession();
      const anim = makeAnimController();
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      // Record the call order: the repaint has to come AFTER the state
      // restore, because restoreRecordingState resizes the render target
      // (which clears the canvas) and removes the keep-alive. Waking
      // before it would leave the viewer blank until the next mouse move.
      const order: string[] = [];
      session.restoreRecordingState = vi.fn(() => order.push('restore'));
      anim.startAnimation.mockImplementation(() => order.push('start'));

      await strat.run(makeOpts(), 'turntable', session);

      expect(order.at(-1)).toBe('start');
      expect(order.at(-2)).toBe('restore');
    });

    it('does NOT re-wake the loop when the panel was disposed mid-capture', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession();
      const anim = makeAnimController();
      const strat = new OfflineCaptureStrategy(sm, anim, makeHooks());

      // Simulate RecordingPanel.dispose() landing while the loop is parked
      // on a frame: it marks the session disposed and ABORTS the capture,
      // whose finally then runs a tick later. runDisposePipeline disposes
      // the AnimationController FIRST, so a wake-up here would restart the
      // rAF loop against a disposed post-processing pipeline.
      let wakesBeforeTeardown = 0;
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          session.disposed = true;
          strat.dispose();
          wakesBeforeTeardown = anim.startAnimation.mock.calls.length;
        });
        return d as never;
      });

      await strat.run(makeOpts(), 'turntable', session);

      // The loop's own opening wake-up already happened (that is what
      // `wakesBeforeTeardown` records); the teardown must add nothing.
      expect(wakesBeforeTeardown).toBeGreaterThan(0);
      expect(anim.startAnimation).toHaveBeenCalledTimes(wakesBeforeTeardown);
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

    it('sets and then clears isLoopRenderSuppressed across a run', async () => {
      const { sm } = makeSceneManager();
      const session = makeSession();
      // Asserting only that the flag is false after teardown is equally true
      // of a flag never set at all, so probe it MID-RUN: suppression while the
      // capture owns the pipeline is the whole point of the flag.
      let flagDuringSetup: boolean | undefined;
      mockState.config.setupOk = true;
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.setup = vi.fn(async () => {
          flagDuringSetup = session.isLoopRenderSuppressed;
          return true;
        });
        return d as never;
      });

      await strat.run(makeOpts(), 'turntable', session);

      expect(flagDuringSetup).toBe(true);
      expect(session.isLoopRenderSuppressed).toBe(false);
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

    it('leaves isLoopRenderSuppressed false when the pre-try overlay setup throws', async () => {
      const { sm } = makeSceneManager();
      // showRecordingIndicator() runs in the window BETWEEN the recording
      // flags and the `try` — the one stretch the finally does not cover — so
      // it is exactly where a throw used to strand the suppression flag true.
      // The flag is the loop's global render-skip predicate, so a stuck true
      // blanks the viewport until a page reload. Raising it as the first
      // statement inside the try is what makes this window safe; moving the
      // assignment back above here turns this test red.
      const boom = new Error('indicator fail');
      const session = makeSession({
        showRecordingIndicator: vi.fn(() => {
          throw boom;
        }),
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await expect(strat.run(makeOpts(), 'turntable', session)).rejects.toThrow(boom);

      expect(session.isLoopRenderSuppressed).toBe(false);
    });

    it('clears isLoopRenderSuppressed BEFORE awaiting driver.abort', async () => {
      mockState.config.finalizeThrows = true;
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      // The flag globally suppresses the animation loop's render, so an
      // abort that never settles would otherwise leave the viewer frozen
      // with no recovery but a page reload.
      let flagDuringAbort: boolean | undefined;
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.abort = vi.fn(async () => {
          flagDuringAbort = session.isLoopRenderSuppressed;
        });
        return d as never;
      });

      await strat.run(makeOpts(), 'turntable', session);

      expect(flagDuringAbort).toBe(false);
      expect(session.isLoopRenderSuppressed).toBe(false);
    });

    it('keeps isOfflineCaptureActive TRUE while driver.abort is awaited', async () => {
      mockState.config.finalizeThrows = true;
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      // The mutual-exclusion flag must NOT be dropped early: ScreenshotStrategy
      // gates on it, and a mediabunny finalize / EXR zip abort can take
      // seconds. A screenshot started inside that window overwrites and then
      // nulls the single `savedRecordingState` slot, so this capture's own
      // `restoreRecordingState()` no-ops and the viewer is stuck at capture
      // resolution with resize locked until a page reload.
      let flagDuringAbort: boolean | undefined;
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.abort = vi.fn(async () => {
          flagDuringAbort = session.isOfflineCaptureActive;
        });
        return d as never;
      });

      await strat.run(makeOpts(), 'turntable', session);

      expect(flagDuringAbort).toBe(true);
      // …and it is still cleared by the time the teardown returns.
      expect(session.isOfflineCaptureActive).toBe(false);
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
