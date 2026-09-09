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
import * as THREE from 'three';

// Shared driver-mock state. `vi.hoisted` so the factory is available to
// the (hoisted) vi.mock calls below.
const mockState = vi.hoisted(() => {
  const driverInstances: Array<Record<string, any>> = [];
  const config = { setupOk: true, captureThrows: 0, shouldAbort: false, finalizeThrows: false };
  function makeDriver() {
    let captureCalls = 0;
    const d: Record<string, any> = {
      setup: vi.fn(async () => config.setupOk),
      captureFrame: vi.fn(async () => {
        captureCalls++;
        if (captureCalls <= config.captureThrows) throw new Error('frame fail');
      }),
      finalize: vi.fn(async (ctx: any) => {
        // Keep the context so a test can exercise the hooks the real
        // drivers use (notably generateFfmpegScript).
        d.lastCtx = ctx;
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

/** A fragment unique to each tone-mapping mode's emitted expression. */
const TONE_MAP_SCRIPT_MARKER: Record<string, string> = {
  linear: 'tone mapping: Linear',
  reinhard: 'tone mapping: Reinhard',
  cineon: 'tone mapping: Cineon',
  aces: 'tone mapping: ACES Filmic',
  agx: 'tone mapping: AgX',
  neutral: 'tone mapping: Khronos PBR Neutral',
};

function makeOpts(overrides: Partial<RecordingOptions> = {}): RecordingOptions {
  return {
    outputFormat: 'png',
    imageQuality: 0.9,
    captureDPR: 1,
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
    includeAudio: true,
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
    pauseAutoDolly: vi.fn(),
    restoreAutoDolly: vi.fn(),
    showRecordingIndicator: vi.fn(),
    hideRecordingIndicator: vi.fn(),
    ...overrides,
  };
}

function makeSceneManager(
  nativeSize: { width: number; height: number } = { width: 1280, height: 720 },
  ssaaScale = 1
): {
  sm: any;
  orbitControls: {
    applyOrbitRotation: ReturnType<typeof vi.fn>;
    applyOrbitDolly: ReturnType<typeof vi.fn>;
    autoDolly: boolean;
    autoDollyPeriod: number;
  };
} {
  const orbitControls = Object.assign(Object.create(LuxarOrbitControls.prototype), {
    applyOrbitRotation: vi.fn(),
    applyOrbitDolly: vi.fn(),
    // Off by default, matching the shipped default; the dolly tests below
    // switch it on before running the capture.
    autoDolly: false,
    autoDollyAmplitude: 0.15,
    autoDollyPeriod: 10,
  });
  const sm = {
    controls: { getControls: vi.fn(() => orbitControls) },
    // The strategy reads the DISPLAY size to honour the panel's "Native"
    // resolution option. `renderer.getSize()` reports the SSAA-multiplied
    // size instead, so the two disagree whenever SSAA is on — the double
    // models that rather than returning the same number twice.
    postProcessing: {
      getDisplaySize: vi.fn(() => ({ ...nativeSize })),
    },
    renderer: {
      getSize: vi.fn((target: { set: (x: number, y: number) => unknown }) => {
        target.set(
          Math.round(nativeSize.width * ssaaScale),
          Math.round(nativeSize.height * ssaaScale)
        );
        return target;
      }),
    },
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
    // The log mock is module-scoped, so without this a test asserting that a
    // particular warning was NOT emitted would see an earlier test's.
    vi.mocked(log.warning).mockClear();
    vi.mocked(log.error).mockClear();
    vi.mocked(log.info).mockClear();
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

    it('captures at the canvas height when the resolution is Native (0)', async () => {
      // The panel's Resolution dropdown documents "Native = current canvas
      // size", but the offline loop used to force 1080 — downscaling every
      // Retina/4K capture and rescaling composited overlays with it.
      const { sm } = makeSceneManager({ width: 2560, height: 1440 });
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts({ videoResolution: 0, captureDPR: 1 }), 'turntable', session);

      expect(session.saveRecordingState).toHaveBeenCalledWith(
        expect.objectContaining({ scaleResolution: { targetH: 1440, alignEven: true } })
      );
    });

    it('still converts EXR frames when the renderer cannot report a grade', async () => {
      // No `getGradeSettings` on the post-processing manager → the grade
      // is unknown.
      // The script must still sRGB-encode: skipping the colour chain
      // hands the encoder scene-linear floats, which is the dark,
      // colour-shifted video the chain exists to prevent.
      //
      // It must NOT claim to have written out the viewer's own curve:
      // an absent grade used to be replaced by a fabricated neutral one,
      // whose header said the tone map had been reproduced when the
      // viewer may well have been on ACES. The chain still clamps (a
      // bare `geq`), since leaving over-range floats to be clipped after
      // the RGB→YUV matrix shifts hue.
      const { sm } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts({ outputFormat: 'exr' }), 'turntable', makeSession());

      const ctx = mockState.driverInstances.at(-1)!.setup.mock.calls[0][0] as {
        generateFfmpegScript: (frames: number, ext: string) => string;
      };
      const script = ctx.generateFfmpegScript(10, 'exr');
      expect(script).toContain('t=iec61966-2-1');
      expect(script).toContain('grade could not be read');
      expect(script).toContain("r='clip(max(r(X,Y),0),0,1)'");
      expect(script).not.toContain('tone mapping:');
    });

    it('asks for the DISPLAY height under SSAA, not the multiplied one', async () => {
      // Post-processing hands the renderer `display × multiplier`, so
      // `renderer.getSize()` on a 1512×850 viewport at 2× reports 1700 —
      // and `saveRecordingState` multiplies by the SSAA factor AGAIN.
      // Reading the renderer therefore squared the multiplier: 3400
      // requested, a 12096×6800 render target, ~1.3 GB per EXR readback
      // and a lost context part-way through the capture.
      const { sm } = makeSceneManager({ width: 1512, height: 850 }, 2);
      const session = makeSession({
        adaptiveDPRManager: { getNativeDPR: vi.fn(() => 2) },
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      // captureDPR 2 keeps the arithmetic the same as when this test was
      // written (850 display x 2), so the failure it guards stays legible:
      // reading `renderer.getSize()` instead would ask for 3400.
      await strat.run(makeOpts({ videoResolution: 0, captureDPR: 2 }), 'turntable', session);

      expect(session.saveRecordingState).toHaveBeenCalledWith(
        expect.objectContaining({ scaleResolution: { targetH: 1700, alignEven: true } })
      );
    });

    /**
     * "Native = current canvas size" means the DEVICE pixels the capture
     * renders at, which is `captureDPR` — the on-screen ceiling by
     * default, so the file matches the viewport. It is deliberately NOT
     * the display's own DPR: with high DPR disallowed the viewport is at
     * 1.0, and exporting at 2.0 unasked would not match what the user
     * framed.
     */
    it('multiplies the canvas height by the CAPTURE DPR for Native', async () => {
      const { sm } = makeSceneManager({ width: 1280, height: 720 });
      const session = makeSession({
        adaptiveDPRManager: { getNativeDPR: vi.fn(() => 2) },
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      // WYSIWYG: the viewport is at 1.0, so the export is 720 tall.
      await strat.run(makeOpts({ videoResolution: 0, captureDPR: 1 }), 'turntable', session);
      expect(session.saveRecordingState).toHaveBeenCalledWith(
        expect.objectContaining({ scaleResolution: { targetH: 720, alignEven: true } })
      );

      // Raised explicitly: the export doubles, and the session is told to
      // render it at that ratio.
      await strat.run(makeOpts({ videoResolution: 0, captureDPR: 2 }), 'turntable', session);
      expect(session.saveRecordingState).toHaveBeenLastCalledWith(
        expect.objectContaining({
          captureDPR: 2,
          scaleResolution: { targetH: 1440, alignEven: true },
        })
      );
    });

    it('describes the capture with the mode the panel is in', async () => {
      // A Turntable capture with Smooth off still rotates a full 360° in
      // this loop, so the dialog must be the turntable one (frame count +
      // duration) — deriving the mode from the format called it a "Video"
      // recording and dropped both.
      const { sm } = makeSceneManager();
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts({ outputFormat: 'png', frameByFrame: false }), 'turntable', session);

      expect(session.showConfirmationDialog).toHaveBeenCalledWith(
        expect.objectContaining({ mode: 'turntable' })
      );
    });

    it('honours an explicit resolution over the native canvas height', async () => {
      const { sm } = makeSceneManager({ width: 2560, height: 1440 });
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts({ videoResolution: 1080 }), 'turntable', session);

      expect(session.saveRecordingState).toHaveBeenCalledWith(
        expect.objectContaining({ scaleResolution: { targetH: 1080, alignEven: true } })
      );
    });

    it.each([
      [THREE.LinearToneMapping, 'linear'],
      [THREE.ReinhardToneMapping, 'reinhard'],
      [THREE.CineonToneMapping, 'cineon'],
      [THREE.ACESFilmicToneMapping, 'aces'],
      [THREE.AgXToneMapping, 'agx'],
      [THREE.NeutralToneMapping, 'neutral'],
    ])('passes tone mapping %i to the EXR script as %s', async (toneMapping, expected) => {
      // Every mode `MegaShaderMaterial.getToneMapping()` can return has to
      // reach the script. A missing entry falls back to Neutral, which
      // silently bakes the WRONG curve into the encode — and reads as a
      // plausible result, since Neutral is a gentle curve.
      const { sm } = makeSceneManager();
      sm.postProcessing.getGradeSettings = () => ({
        toneMapping,
        exposure: 0,
        offset: 0,
        gamma: 1,
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts({ outputFormat: 'exr' }), 'turntable', makeSession());

      const driver = mockState.driverInstances.at(-1)!;
      const script: string = driver.lastCtx.generateFfmpegScript(8, 'exr');
      expect(script).toContain(TONE_MAP_SCRIPT_MARKER[expected]);
    });

    it('gives every artifact of one capture the same timestamped stem', async () => {
      // generateFilename() stamps new Date() per call, so the ZIP name,
      // its fallback download name and the encode script's output base
      // used to disagree whenever a capture crossed a second boundary.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      let call = 0;
      hooks.generateFilename = vi.fn((ext: string) => `cap-${++call}.${ext}`);
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts({ outputFormat: 'png' }), 'turntable', makeSession());

      // Exactly one stem is minted; the drivers derive everything else.
      expect(hooks.generateFilename).toHaveBeenCalledTimes(1);

      // And every artifact the drivers ask for shares it: the ZIP name
      // the driver downloads and the base the encode script names its
      // outputs after have to be the same capture, not two timestamps.
      const ctx = mockState.driverInstances.at(-1)!.lastCtx;
      const zipName: string = ctx.generateFilename('zip');
      const script: string = ctx.generateFfmpegScript(2, 'png');
      expect(script).toContain(`"${zipName.replace(/\.zip$/, '')}-turntable.mp4"`);
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

    it('bakes a WHOLE number of dolly cycles into the turn so the clip loops', async () => {
      const { sm, orbitControls } = makeSceneManager();
      orbitControls.autoDolly = true;
      // 3 s turn, 1 s period → exactly 3 cycles, no rounding needed.
      orbitControls.autoDollyPeriod = 1;
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 120 }),
        'turntable',
        makeSession()
      );

      // One call per frame INCLUDING frame 0 — that call drives the dolly to
      // phase 0, undoing any offset left by the interactive oscillation so the
      // capture opens at the true baseline distance.
      const phases = orbitControls.applyOrbitDolly.mock.calls.map((c: unknown[]) => c[0] as number);
      expect(phases).toHaveLength(6);
      expect(phases[0]).toBe(0);
      // Same [0, 2π) convention as the rotation: the last frame stops one step
      // short of closing the loop, so playback wraps without a duplicate frame.
      const cycleSpan = 2 * Math.PI * 3;
      expect(phases[5]).toBeCloseTo(cycleSpan * (5 / 6), 12);
      expect(phases[5]).toBeLessThan(cycleSpan);
      // Strictly increasing, evenly spaced.
      for (let i = 1; i < phases.length; i++) {
        expect(phases[i] - phases[i - 1]).toBeCloseTo(cycleSpan / 6, 12);
      }
    });

    it('rounds a period that does not divide the turn UP to a whole cycle', async () => {
      const { sm, orbitControls } = makeSceneManager();
      orbitControls.autoDolly = true;
      // 3 s turn with a 10 s period would be 0.3 of a cycle — rounded to 0 it
      // would silently disable the dolly, so the floor is one slow breath.
      orbitControls.autoDollyPeriod = 10;
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 120 }),
        'turntable',
        makeSession()
      );

      const phases = orbitControls.applyOrbitDolly.mock.calls.map((c: unknown[]) => c[0] as number);
      expect(phases[5]).toBeCloseTo(2 * Math.PI * (5 / 6), 12);
    });

    it('leaves the dolly alone when the user has it switched off', async () => {
      const { sm, orbitControls } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 120 }),
        'turntable',
        makeSession()
      );

      expect(orbitControls.applyOrbitDolly).not.toHaveBeenCalled();
    });

    it('pauses the interactive dolly for the capture and restores it after', async () => {
      // Wall-clock frames here wait on LOD settling, so the live oscillation
      // would judder AND compound with the baked one.
      const { sm, orbitControls } = makeSceneManager();
      orbitControls.autoDolly = true;
      const session = makeSession();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(
        makeOpts({ outputFormat: 'png', videoFPS: 2, turntableSpeed: 120 }),
        'turntable',
        session
      );

      expect(session.pauseAutoDolly).toHaveBeenCalled();
      expect(session.restoreAutoDolly).toHaveBeenCalled();
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

  describe('LOD settle drain (#1695)', () => {
    it('never drains — not even one extra rAF — when the isLODSettled hook is absent', async () => {
      // The hook is optional; without it the loop must be byte-for-byte the
      // pre-#1695 one. Default opts → totalFrames = 2, so exactly one opening
      // rAF plus one per frame.
      const { sm } = makeSceneManager();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), makeHooks());

      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(rafSpy.mock.calls.length).toBe(1 + 2);
      expect(mockState.driverInstances.at(-1)!.captureFrame).toHaveBeenCalledTimes(2);
      expect(showToast).not.toHaveBeenCalled();
    });

    it('skips the drain entirely — mandatory tick included — when the hook answers null', async () => {
      // `null` is the hook's "this scene has nothing that could ever need
      // waiting for" answer (no scene loader, no LOD registry, or a registry
      // with neither lod_groups nor partitions). The provider is wired
      // unconditionally in production, so a plain points/lines scene reaches
      // this path on every capture and must cost exactly what it did pre-#1695
      // — the mandatory selector-catch-up rAF included. Compare the `() =>
      // true` test below, which spends two extra frames for the same two
      // exported frames.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      hooks.isLODSettled = vi.fn(() => null);
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(rafSpy.mock.calls.length).toBe(1 + 2);
      // Consulted once per frame — the answer is read, not the hook's absence.
      expect(hooks.isLODSettled).toHaveBeenCalledTimes(2);
      expect(mockState.driverInstances.at(-1)!.captureFrame).toHaveBeenCalledTimes(2);
      expect(showToast).not.toHaveBeenCalled();
    });

    it('spends exactly one rAF per frame when the hook is already settled', async () => {
      // Not zero: `AnimationController.animate` runs controls.update(), then
      // the per-frame callbacks in Map insertion order, then the render. The
      // LOD selector is registered at pipeline init while the capture's orbit
      // callback is re-added every iteration (hence always last), and
      // applyOrbitRotation moves the camera synchronously — so within the
      // frame the loop awaits for frame N the selector evaluated pose N−1 and
      // only then did the camera advance to pose N. One more tick is what
      // makes the predicate describe the pose being captured.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      hooks.isLODSettled = vi.fn(() => true);
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(rafSpy.mock.calls.length).toBe(1 + 2 + 2);
      // Twice per frame: the tri-state probe that decides whether this scene
      // drains at all (read before the catch-up tick, so only its NULL-ness is
      // used), then the real poll once the selector is on the captured pose.
      expect(hooks.isLODSettled).toHaveBeenCalledTimes(4);
    });

    it('never DECIDES on the hook before the selector has ticked on the pose being captured', async () => {
      // Pins the mandatory catch-up rAF directly: the read a frame actually
      // acts on must come strictly after at least one rAF beyond the frame's
      // orbit tick. (The frame's FIRST read is the tri-state probe, which
      // deliberately sits at the orbit tick and whose boolean is discarded —
      // hence "decides", not "polls".) Deleting the hoisted first await makes
      // every read of a frame land at the orbit tick and turns this red.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      const rafAtRead: number[] = [];
      hooks.isLODSettled = vi.fn(() => {
        rafAtRead.push(rafSpy.mock.calls.length);
        return true;
      });
      const rafAtCapture: number[] = [];
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          rafAtCapture.push(rafSpy.mock.calls.length);
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(rafAtRead).toHaveLength(4); // probe + deciding poll, per frame
      expect(rafAtCapture).toHaveLength(2);
      // Frame 0: opening rAF (1) + orbit tick (2, probe) + catch-up tick (3,
      // poll). Frame 1: orbit tick (4, probe) + catch-up tick (5, poll).
      rafAtCapture.forEach((atCapture, frame) => {
        // The frame's orbit tick is the rAF right after the previous frame's
        // capture (or the opening rAF for frame 0), so a read that has not
        // spent a further tick sits exactly at it.
        const orbitTick = (frame === 0 ? 1 : rafAtCapture[frame - 1]) + 1;
        const decidingRead = rafAtRead.filter((r) => r <= atCapture).at(-1)!;
        expect(decidingRead).toBeGreaterThan(orbitTick);
      });
    });

    it('captures the whole sequence when the settle hook THROWS, instead of failing the run', async () => {
      // The hook is a diagnostic injected from outside this module, and it is
      // called inside the loop's main try — so an unguarded throw would be
      // caught by the outer handler, reported as "Recording failed" and
      // discard every frame already captured. It degrades to "do not wait"
      // instead (the same treatment `AnimationController.pacingSuspended()`
      // gives its injected predicate).
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      hooks.isLODSettled = vi.fn(() => {
        throw new Error('registry exploded');
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts(), 'turntable', makeSession());

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.captureFrame).toHaveBeenCalledTimes(2);
      expect(driver.finalize).toHaveBeenCalled();
      expect(showToast).not.toHaveBeenCalledWith('Recording failed');
      // A throw is "do not wait", not a timeout: no degraded-sequence report.
      // (Both report toasts name LOD; neither may fire here.)
      expect(showToast).not.toHaveBeenCalledWith(expect.stringContaining('LOD'));
      // …and it is surfaced once rather than silently swallowed.
      expect(log.warning).toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('LOD settle predicate threw')
      );
    });

    it('captures a frame only AFTER the hook reports the LOD levels settled', async () => {
      // The regression: a tile reloading its fine level after re-entering the
      // frustum must not be filmed at its coarse fallback.
      const { sm, orbitControls } = makeSceneManager();
      const trace: string[] = [];
      const hooks = makeHooks();
      let reads = 0;
      hooks.isLODSettled = vi.fn(() => {
        reads++;
        const settled = reads > 3; // the first three reads report "still loading"
        trace.push(settled ? 'settled' : 'draining');
        return settled;
      });
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          trace.push('capture');
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      await strat.run(makeOpts(), 'turntable', makeSession());

      // Each frame opens with the tri-state probe (whose boolean is discarded)
      // and then polls after every drain tick. Frame 0: probe + two more
      // "draining" reads before the fourth settles; frame 1: probe + one
      // settled poll.
      expect(trace).toEqual([
        'draining', // frame 0 probe — non-null, so this scene drains
        'draining',
        'draining',
        'settled',
        'capture',
        'settled', // frame 1 probe
        'settled',
        'capture',
      ]);
      // Every capture is immediately preceded by a settled poll.
      trace.forEach((entry, i) => {
        if (entry === 'capture') expect(trace[i - 1]).toBe('settled');
      });
      // Opening rAF + one orbit tick per frame + each frame's drain. Frame 0
      // spends the mandatory selector-catch-up tick plus two more before its
      // fourth read settles; frame 1 spends only the catch-up tick.
      expect(rafSpy.mock.calls.length).toBe(1 + 2 + (1 + 2) + 1);
      // The camera advances ONCE PER EXPORTED FRAME, never once per drain rAF.
      // This is what pins the drain's position after the orbit callback's
      // removal: hoisting it above that removal leaves the callback registered
      // for the drain's own ticks, so the turntable would keep rotating while
      // waiting and smear the sweep across each exported frame. (Two frames,
      // and frame 0 captures the opening pose without rotating — so one call.)
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledTimes(1);
      // No timeout occurred, so no warning and no toast.
      expect(showToast).not.toHaveBeenCalled();
    });

    it('bounds the wait, still captures every frame, and stops draining after 3 consecutive timeouts', async () => {
      // A scene that can never settle (e.g. resident-byte thrash on an
      // over-budget partition) must not multiply the capture's wall-clock by
      // the timeout for every remaining frame.
      const { sm, orbitControls } = makeSceneManager();
      const hooks = makeHooks();
      hooks.isLODSettled = vi.fn(() => false);
      const rafAtCapture: number[] = [];
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          rafAtCapture.push(rafSpy.mock.calls.length);
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      // fps 4 + speed 360 → totalFrames = 4: three timeouts to trip the
      // consecutive limit, plus one frame that must no longer drain.
      await strat.run(makeOpts({ videoFPS: 4, turntableSpeed: 360 }), 'turntable', makeSession());

      const driver = mockState.driverInstances.at(-1)!;
      expect(driver.captureFrame).toHaveBeenCalledTimes(4); // bounded, never stuck
      // rAFs spent per frame: the first three burn the 120-frame drain cap —
      // the mandatory selector-catch-up tick counts as drain frame 1, so the
      // cap is inclusive of it — plus their own orbit frame. (performance.now
      // barely advances under a synchronous rAF spy, so the frame cap, not the
      // ms deadline, is what terminates them.) The fourth frame has had
      // draining switched off and spends only its orbit frame.
      const perFrame = rafAtCapture.map((n, i) => n - (i === 0 ? 1 : rafAtCapture[i - 1]));
      expect(perFrame).toEqual([121, 121, 121, 1]);
      // One camera step per exported frame (frame 0 captures the opening
      // pose), never one per drain rAF — see the same assertion above. With
      // 120 drain ticks a frame, a drain hoisted above the orbit callback's
      // removal would show up here as hundreds of rotations.
      expect(orbitControls.applyOrbitRotation).toHaveBeenCalledTimes(3);
      // Reported rather than silent.
      expect(log.warning).toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('consecutive LOD settle timeouts')
      );
      // Deliberately NOT "3 of 4 frame(s)". Once draining latches off the
      // counter stops describing the run: it only counts the frames that
      // WAITED and gave up, while every frame captured with the wait off was
      // taken without one (here 1 of 4; on a 600-frame capture, hundreds).
      // The report says that instead of a number it cannot know.
      //
      // The toast says waiting was PAUSED, not that the LOD never settled:
      // this branch is also reached by a run that settled cleanly for hundreds
      // of frames and then hit a stall long enough for three in a row. And it
      // says "may not show the settled level" rather than "coarse", because
      // the predicate is direction-blind — a never-downgrade hold on a FINER
      // level reads as unsettled too.
      expect(showToast).toHaveBeenCalledWith(
        'Paused waiting for LOD — some frames may not show the settled level'
      );
      expect(log.warning).toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('the frames captured while it was off were taken without waiting')
      );
      expect(driver.finalize).toHaveBeenCalled();
    });

    it('re-arms the drain when a scene that timed out three times settles later', async () => {
      // The latch must not be terminal. On the scenes this drain targets — an
      // over-budget `adaptive`/`overview` partition, or a Capture pressed
      // before the initial load has finished — the first frames burn the whole
      // budget and the latch fires at frame 3, so a terminal latch would export
      // every remaining frame with the pre-#1695 behaviour even though the
      // scene settles seconds later. A latched frame therefore still spends a
      // FREE probe (no rAF, no poll) and clears the latch on a settled answer.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      let settles = false;
      hooks.isLODSettled = vi.fn(() => settles);
      const rafAtCapture: number[] = [];
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          rafAtCapture.push(rafSpy.mock.calls.length);
          // The scene comes good right after the third (latching) frame.
          if (rafAtCapture.length === 3) settles = true;
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      // fps 6 + speed 360 → totalFrames = 6: three frames to trip the limit,
      // the latched frame that re-arms, then two normally-drained frames.
      await strat.run(makeOpts({ videoFPS: 6, turntableSpeed: 360 }), 'turntable', makeSession());

      const perFrame = rafAtCapture.map((n, i) => n - (i === 0 ? 1 : rafAtCapture[i - 1]));
      // Frames 0-2 burn the 120-frame drain cap plus their orbit frame. Frame 3
      // is latched: orbit frame only, and its free probe (now settled) re-arms
      // the drain. Frames 4-5 drain again — orbit frame + the mandatory
      // selector-catch-up tick, which is what a healthy drained frame costs.
      // Without the re-arm the last three frames would each read 1.
      expect(perFrame).toEqual([121, 121, 121, 1, 2, 2]);
      expect(mockState.driverInstances.at(-1)!.captureFrame).toHaveBeenCalledTimes(6);
      // The latch still happened, so the report keeps the no-number branch:
      // frame 3 was captured without waiting, which the timeout count (3) does
      // not describe.
      expect(showToast).toHaveBeenCalledWith(
        'Paused waiting for LOD — some frames may not show the settled level'
      );
      // Warned once per run, not once per latch.
      expect(
        vi
          .mocked(log.warning)
          .mock.calls.filter(([, msg]) => String(msg).includes('consecutive LOD settle timeouts'))
      ).toHaveLength(1);
    });

    it('reports an EXACT count when waiting stayed on for the whole run', async () => {
      // The other reporting branch: a single frame times out and the next one
      // settles, so the consecutive counter resets, draining is never switched
      // off, and the count genuinely describes the sequence.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      let capturedSoFar = 0;
      // Frame 0 can never settle; every later frame settles at once.
      hooks.isLODSettled = vi.fn(() => capturedSoFar > 0);
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          capturedSoFar++;
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      // Default opts → totalFrames = 2.
      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(capturedSoFar).toBe(2); // both frames still exported
      // Never latched off (one timeout, then a success resetting the streak).
      expect(log.warning).not.toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('consecutive LOD settle timeouts')
      );
      expect(log.warning).toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('1 of 2 frame(s) were captured before their LOD levels settled')
      );
      expect(showToast).toHaveBeenCalledWith('1 frame(s) captured before LOD settled');
    });

    it('counts the exact report against ATTEMPTED frames, so a timing-out run that also throws cannot read "2 of 0"', async () => {
      // `lodSettleTimeouts` is incremented before `driver.captureFrame`, while
      // `capturedFrames` only advances on success — so a run whose frames both
      // time out and then throw printed a numerator larger than its
      // denominator. Two frames, both timing out (short of the three
      // consecutive needed to latch draining off) and both failing to capture.
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      hooks.isLODSettled = vi.fn(() => false);
      vi.mocked(ImageSequenceDriver).mockImplementationOnce(() => {
        const d = mockState.makeDriver();
        d.captureFrame = vi.fn(async () => {
          throw new Error('encoder blew up');
        });
        return d as never;
      });
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);

      // Default opts → totalFrames = 2, i.e. under MAX_CONSECUTIVE_LOD_TIMEOUTS
      // (3) and under MAX_CONSECUTIVE_ERRORS (3): the exact-count branch.
      await strat.run(makeOpts(), 'turntable', makeSession());

      expect(log.warning).not.toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('consecutive LOD settle timeouts')
      );
      expect(log.warning).toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('2 of 2 frame(s) were captured before their LOD levels settled')
      );
      // The pre-fix denominator was `capturedFrames`, which is 0 here.
      expect(log.warning).not.toHaveBeenCalledWith(
        'Recording',
        expect.stringContaining('2 of 0 frame(s)')
      );
    });

    it('breaks out of the drain cleanly when the session is aborted mid-wait', async () => {
      const { sm } = makeSceneManager();
      const hooks = makeHooks();
      const strat = new OfflineCaptureStrategy(sm, makeAnimController(), hooks);
      let polls = 0;
      hooks.isLODSettled = vi.fn(() => {
        polls++;
        if (polls === 2) strat.abort(); // user hits Stop while we wait
        return false;
      });

      await strat.run(makeOpts(), 'turntable', makeSession());

      const driver = mockState.driverInstances.at(-1)!;
      // The abort is observed by the drain's own guard and then by the frame
      // loop's existing post-drain check — no frame is captured, no finalize.
      expect(driver.captureFrame).not.toHaveBeenCalled();
      expect(driver.finalize).not.toHaveBeenCalled();
      // The wait stopped on the abort, not on a bound: one tri-state probe and
      // one poll, versus the 121 reads the frame cap would have allowed. This
      // is the assertion that pins the drain's abort handling — the loop's
      // `!aborted` guard on the timeout counter is NOT observable from here,
      // because an aborted run returns before the end-of-run report either
      // way, so the absent toast below only records that an aborted capture
      // reports nothing at all.
      expect(polls).toBe(2);
      expect(showToast).not.toHaveBeenCalled();
      // …and the normal teardown still ran.
      expect(document.querySelector('.luxar-recording-overlay')).toBeNull();
      expect(strat.sessionAbort).toBeNull();
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
