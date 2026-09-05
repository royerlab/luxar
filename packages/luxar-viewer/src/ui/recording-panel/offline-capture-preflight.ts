/**
 * Everything the offline capture settles BEFORE the first frame: the
 * confirmation dialog, session ownership, capture resolution, and the
 * turntable's frame-indexed rotation and dolly schedules.
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop` (audit A2-01).
 * All of it either bails or produces a {@link TurntablePlan}, and none of it
 * touches a driver or a frame — so the loop no longer opens with a hundred
 * lines it cannot act on.
 */

import { Modules, log } from '../../utils/log';
import { getMaxPixelRatio } from '../../rendering/pixel-ratio-cap';
import { LuxarOrbitControls } from '../../controls/luxar-orbit-controls';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { SceneManager } from '../../scene/scene-manager';
import type { RecordingSession } from './session';
import type { RecordingMode, RecordingOptions } from './types';

/** The frame schedule the capture loop plays out. */
export interface TurntablePlan {
  /** Owns the run; `abort()` on it stops the loop wherever it is. */
  sessionAbort: AbortController;
  controls: LuxarOrbitControls;
  fps: number;
  totalFrames: number;
  /** Radians to advance per frame. Frame 0 does not rotate. */
  anglePerFrame: number;
  /** Whole in-and-out dolly cycles over the turn; 0 when the dolly is off. */
  dollyCycles: number;
  /** Dolly phase for a frame index, in `[0, 2π·dollyCycles)`. */
  dollyPhaseFor: (frame: number) => number;
}

export interface PreflightDeps {
  session: RecordingSession;
  sceneManager: SceneManager;
  animationController: AnimationController;
  hideAllPanels: () => void;
  /** Record the new controller as the owning session before any mutation. */
  claimSession: (abort: AbortController) => void;
  /** Drop it again if this preflight bails. */
  releaseSession: (abort: AbortController) => void;
}

/**
 * The height the capture renders at, and the DPR it renders with.
 *
 * Dimensions are aligned DOWN to even numbers by the caller — H.264/H.265 with
 * yuv420p need even width and height, and encoders pad internally to
 * their own macroblock size, so nothing here has to.
 *
 * `videoResolution === 0` is the panel's "Native" option, documented
 * in its tooltip as "current canvas size" — so capture at the size
 * the canvas actually has (display size × native DPR) rather than
 * silently forcing 1080. Forcing it downscaled every Retina/4K
 * capture and, because it changed the capture-to-CSS pixel ratio,
 * rescaled the composited overlays with it.
 *
 * The display size comes from post-processing, NOT from
 * `renderer.getSize()`: the renderer is handed the SSAA-multiplied
 * size, so under SSAA it reports `display × multiplier` and asking
 * to render THAT squares the multiplier (2× on a 3024×1700 canvas
 * asked for a 12096×6800 target). `saveRecordingState` re-applies
 * the multiplier itself, so the frames on disk still carry SSAA —
 * which is what the real-time path's canvas backbuffer includes too.
 * `captureDPR`, not the display's native DPR: "Native" here means
 * "the resolution this capture renders at", which by default is what
 * is on screen. Raise Capture DPR in the panel for a bigger export.
 */
export function resolveCaptureResolution(
  opts: RecordingOptions,
  sceneManager: SceneManager
): { captureDPR: number; targetH: number } {
  const displayH = sceneManager.postProcessing.getDisplaySize().height;
  const captureDPR = opts.captureDPR ?? getMaxPixelRatio();
  return {
    captureDPR,
    targetH: opts.videoResolution > 0 ? opts.videoResolution : Math.round(displayH * captureDPR),
  };
}

/**
 * Run the pre-capture sequence.
 *
 * @returns the plan, or `null` when the run must not proceed — the user
 *          cancelled, the panel was disposed, or the scene is not on orbit
 *          controls. Every `null` path has already restored whatever it
 *          changed, so the caller just returns.
 */
export async function runCapturePreflight(
  mode: string,
  opts: RecordingOptions,
  recordingMode: RecordingMode,
  deps: PreflightDeps
): Promise<TurntablePlan | null> {
  const { session, sceneManager, animationController } = deps;

  // The dialog gets the mode the panel is actually in. Deriving it from
  // the format instead described a Turntable + PNG/MP4 capture with
  // Smooth off as a "Video" recording — no 360° line, no frame count —
  // even though this loop always rotates a full turntable.
  const confirmed = await session.showConfirmationDialog({
    mode: recordingMode,
    options: opts,
  });
  if (!confirmed || session.isDisposed()) return null;

  // Establish session ownership BEFORE any state mutation. dispose()
  // reads `sessionAbort` to abort an in-flight session; if we
  // assign it later (after hideAllPanels / saveRecordingState / the
  // first rAF), a dispose during that early window leaves the
  // abort controller null and the function continues to bring up
  // overlay/recording flags on a disposed panel.
  const sessionAbort = new AbortController();
  deps.claimSession(sessionAbort);

  const bailEarly = (): null => {
    session.restoreRecordingState();
    // Same repaint guarantee as the main teardown: restoreRecordingState
    // resizes the render target back, which clears the canvas, and no
    // keep-alive is registered on this path — so a stopped loop would
    // leave the viewer blank until the next mouse move. Skipped when the
    // session is disposed (see the note in the finally).
    if (!session.isDisposed()) {
      animationController.startAnimation();
    }
    deps.releaseSession(sessionAbort);
    return null;
  };

  deps.hideAllPanels();

  const { captureDPR, targetH } = resolveCaptureResolution(opts, sceneManager);
  session.saveRecordingState({
    captureDPR,
    lockResize: true,
    scaleResolution: { targetH, alignEven: true },
  });
  await new Promise((r) => requestAnimationFrame(r));

  if (session.isDisposed() || sessionAbort.signal.aborted) {
    return bailEarly();
  }

  // Compute turntable parameters
  const fps = opts.videoFPS;
  const durationSeconds = 360 / opts.turntableSpeed;
  const totalFrames = Math.ceil(durationSeconds * fps);

  const controls = sceneManager.controls.getControls();
  if (!(controls instanceof LuxarOrbitControls)) {
    log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
    return bailEarly();
  }

  session.pauseAutoRotate();
  // The wall-clock dolly must not compound with the frame-indexed one below.
  // It would also judder: these frames wait on LOD settling, so `deltaTime`
  // here bears no relation to playback time.
  const dollyActive = controls.autoDolly;
  session.pauseAutoDolly();

  // Per-frame rotation step: frame 0 captures the starting view without
  // rotation, then frames 1..N-1 each advance by one step, so the N frames
  // cover [0, 2π) and the LAST frame stops one step short of the first.
  // That step is 2π/N, NOT 2π/(N-1): dividing by N-1 lands the last frame
  // exactly back on the start pose, and a turntable is made to loop — the
  // duplicate shows up as a one-frame hitch at every wrap.
  const anglePerFrame = totalFrames > 0 ? (2 * Math.PI) / totalFrames : 0;

  // Auto-dolly, baked frame-indexed alongside the rotation. The number of
  // in-and-out cycles is ROUNDED to a whole number over the turn so the clip
  // loops: at the configured period a 24 s turn with a 10 s period would
  // otherwise end mid-swing, and the wrap would jump. `max(1, …)` keeps a
  // period longer than the whole turn as one slow breath rather than none.
  const dollyCycles =
    dollyActive && controls.autoDollyPeriod > 0
      ? Math.max(1, Math.round(durationSeconds / controls.autoDollyPeriod))
      : 0;
  // Same [0, 2π) convention as the rotation above: frame 0 sits at phase 0
  // (its own baseline distance) and the last frame stops one step short.
  const dollyPhaseFor = (frame: number): number =>
    totalFrames > 0 ? (2 * Math.PI * dollyCycles * frame) / totalFrames : 0;

  log.info(
    Modules.RECORDING,
    `Starting offline ${mode} capture: ${totalFrames} frames, ${fps} FPS, ${durationSeconds.toFixed(1)}s`
  );

  return {
    sessionAbort,
    controls,
    fps,
    totalFrames,
    anglePerFrame,
    dollyCycles,
    dollyPhaseFor,
  };
}
