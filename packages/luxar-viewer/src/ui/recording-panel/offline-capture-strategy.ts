/**
 * Offline (frame-by-frame) capture strategy.
 *
 * Used for:
 * - EXR sequence recording (ZIP of EXR frames, full float precision)
 * - Turntable smooth mode (renders each frame individually for perfectly
 *   smooth output regardless of GPU FPS)
 *
 * Unlike real-time MediaRecorder capture, this loop is decoupled from
 * the browser's animation frame RATE — but not from the animation loop
 * itself: step 1 runs as a per-frame callback, so the rAF loop has to
 * be running (see the wake-up in `runOfflineCaptureLoop`). Each frame
 * is:
 * 1. Camera orbited by one step (quaternion rotation, same as auto-rotate)
 * 2. One mandatory rAF (so the LOD selector, which runs BEFORE the orbit
 *    callback in the same frame, has evaluated the new pose) plus a
 *    bounded wait for it to settle (`hooks.isLODSettled`) — the rAF loop
 *    is live for the whole sweep, so a tile that left the frustum
 *    mid-orbit can be mid-reload when it swings back, and capturing
 *    immediately bakes a coarse-level pop into the sequence (#1695).
 *    Skipped entirely — that mandatory rAF included — when the hook
 *    answers `null`, i.e. this scene has no lod_group to wait for
 * 3. Scene rendered (full pipeline, into the capture's own target)
 * 4. Pixels read back asynchronously (PBO fence on WebGL2, mapAsync on
 *    WebGPU) — the rAF loop keeps ticking through the await, which is
 *    why the loop's own render is suppressed for the whole capture
 *    (see the render-skip predicate wired in `core/app/init/pipeline`)
 * 5. Frame stored / encoded
 * 6. Brief yield to keep the browser responsive
 *
 * The loop owns shared scaffolding (state save/restore via Session,
 * modal overlay, animation pump, progress display, error tolerance);
 * per-mode capture (PNG/WebP/JPEG sequence, video container, EXR
 * sequence) is delegated to a driver implementing
 * {@link OfflineCaptureDriver}. Each driver runs its own setup,
 * captures one frame at a time, and finalizes (download/save).
 *
 * Critical correctness invariants:
 * 1. AbortController is assigned BEFORE any state mutation — dispose()
 *    during the early state-save / rAF window must abort the session.
 * 2. The body from driver.setup onward is wrapped in try/finally so an
 *    exception from driver.setup, driver.captureFrame, driver.finalize,
 *    or any DOM/state mutation cannot leave the panel with a stuck
 *    overlay, hidden panels, scaled renderer, or stale recording flags.
 *    The earlier window — panel hide / saveRecordingState through the
 *    overlay construction — is NOT covered (the finally closes over
 *    bindings that window creates), which is why
 *    isLoopRenderSuppressed is raised inside the try: a stuck value
 *    there blanks the whole viewport, where the other flags only lock
 *    further captures or leave the panel looking wrong. That window is
 *    synchronous DOM construction with no production-reachable throw —
 *    showConfirmationDialog above it already assigns innerHTML, so an
 *    environment that forbids it fails before any state is mutated.
 * 3. The finally block is idempotent — every removal/restore handles
 *    the "wasn't set" case gracefully.
 * 4. The LOD settle drain (step 2) runs AFTER the orbit callback has
 *    been removed. The camera pose is fixed by then, so the extra
 *    frames only let pending loads land; draining before the removal
 *    would keep orbiting the camera while waiting and smear the sweep.
 */

import * as THREE from 'three';
import { log, Modules } from '../../utils/log';
import { getViewerContainer } from '../../utils/viewer-container';
import { showToast } from '../toast';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { LuxarOrbitControls } from '../../controls/luxar-orbit-controls';
import { computeVideoBitrate as computeVideoBitratePure } from './media-utilities';
import {
  generateFfmpegScript as generateFfmpegScriptPure,
  type GradeSettings,
  type ToneMapName,
} from './ffmpeg-script';
import type { CaptureContext, OfflineCaptureDriver } from './drivers/offline-capture-driver';
import { ImageSequenceDriver } from './drivers/image-sequence-driver';
import { ExrSequenceDriver } from './drivers/exr-sequence-driver';
import { VideoModeDriver } from './drivers/video-mode-driver';
import type { CaptureStrategy, SessionState } from './capture-strategy';
import type { RecordingSession } from './session';
import type { RecordingMode, RecordingOptions } from './types';

export interface OfflineCaptureStrategyHooks {
  hideAllPanels(): void;
  renderFrameToCanvas(): Promise<HTMLCanvasElement>;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
  /**
   * Whether every in-frame LOD group is showing its selected level at final
   * quality (`LODGroupRegistry.isCaptureQuiescent()`). The capture drains on
   * this before grabbing each frame, so an asynchronous fine-level reload —
   * kicked when a tile swings back into the frustum mid-orbit — cannot be
   * filmed at its coarse fallback and pop back a few frames later (#1695).
   *
   * TRI-STATE, and the third state is what keeps a plain points/lines scene
   * free:
   * - `true` / `false` — this scene HAS level-of-detail groups, so the loop
   *   drains. It then always spends at least one extra rAF per frame even when
   *   the answer is already `true`, because the selector runs before the orbit
   *   callback within a frame and so is a pose behind until it ticks once more.
   * - `null` — this scene has no lod_group to wait for (no registry, or a
   *   registry with none in it). The loop skips the drain entirely, INCLUDING
   *   that mandatory tick, which is the pre-#1695 behaviour to the frame. Note
   *   the narrowness: `null` is NOT "nothing here could ever be mid-load". A
   *   `--recipe stream` scene — one leaf with an additive ladder and no
   *   lod_group — answers `null` while its progressive refinement is still
   *   climbing the ladder, so its early frames can be exported at a partial
   *   prefix. Same artifact class, not covered by this drain.
   *
   * Absent ⇒ identical to `null`. The panel wires this unconditionally and the
   * pipeline's provider is the one that answers `null`, so "the hook exists"
   * must never be read as "this scene needs draining".
   */
  isLODSettled?(): boolean | null;
}

type OfflineMode = 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg';

/**
 * Wall-clock ceiling on the per-frame wait for the LOD selector to settle
 * (`hooks.isLODSettled`). A fine level that is genuinely being refetched over
 * the network can take a while; 2 s is generous enough to cover a normal
 * reload and short enough that a scene which cannot settle does not multiply
 * a 600-frame capture's runtime beyond recovery.
 */
const LOD_SETTLE_TIMEOUT_MS = 2000;

/**
 * Frame ceiling on the same wait (≈2 s at 60 fps), inclusive of the mandatory
 * selector-catch-up tick. NOT redundant with the millisecond deadline: under a
 * stubbed clock — unit tests, fake timers — `performance.now()` never advances,
 * so the ms deadline alone would spin forever. Whichever bound trips first ends
 * the drain.
 */
const LOD_SETTLE_MAX_FRAMES = 120;

/**
 * Consecutive per-frame settle timeouts after which the capture PAUSES
 * draining. A scene that can never settle (e.g. resident-byte thrash on a
 * partition that does not fit the budget) would otherwise pay
 * `LOD_SETTLE_TIMEOUT_MS` on EVERY remaining frame, silently multiplying the
 * capture's wall-clock. One successful settle resets the counter.
 *
 * The pause RE-ARMS: a latched frame still spends a free, non-waiting probe of
 * the predicate (no rAF, no poll — so a latched frame costs exactly what it did
 * before the latch fired) and clears the latch the moment that probe reports
 * settled. Without the re-arm the latch was terminal, and on exactly the scenes
 * this drain targets — an over-budget `adaptive` / `overview` partition, or a
 * Capture pressed before the initial load finished — the first three frames
 * would burn their budget, the latch would fire at frame 3, and the remaining
 * hundreds of frames would be exported with the pre-#1695 behaviour even though
 * the scene settles seconds later.
 */
const MAX_CONSECUTIVE_LOD_TIMEOUTS = 3;

/** THREE tone-mapping constants → the names the ffmpeg script knows. */
const TONE_MAP_BY_THREE_CONSTANT: Record<number, ToneMapName> = {
  [THREE.LinearToneMapping]: 'linear',
  [THREE.ReinhardToneMapping]: 'reinhard',
  [THREE.CineonToneMapping]: 'cineon',
  [THREE.ACESFilmicToneMapping]: 'aces',
  [THREE.AgXToneMapping]: 'agx',
  [THREE.NeutralToneMapping]: 'neutral',
};

/**
 * Read the display transform an EXR capture bypasses, so the bundled
 * ffmpeg script can put it back.
 *
 * Returns `undefined` when the renderer doesn't expose the grade (older
 * mocks in tests), which is what the script's `unknown-grade` chain is
 * for: it still converts the transfer — linear floats must never go out
 * untouched — and says in the header that the curve is missing. Handing
 * it a fabricated neutral grade instead produced identical pixels but a
 * header claiming the viewer's own curve had been written out, while the
 * viewer may well have been on ACES.
 */
function readGradeSettings(sceneManager: SceneManager): GradeSettings | undefined {
  const grade = sceneManager.postProcessing?.getGradeSettings?.();
  if (!grade) return undefined;
  return {
    toneMapping: TONE_MAP_BY_THREE_CONSTANT[grade.toneMapping] ?? 'neutral',
    exposure: grade.exposure,
    offset: grade.offset,
    gamma: grade.gamma,
  };
}

export class OfflineCaptureStrategy implements CaptureStrategy {
  readonly kind = 'offline' as const;

  /**
   * AbortController for the offline-capture session. Set immediately
   * after the confirmation check, BEFORE any state mutation, so
   * dispose() during the early state-save / rAF window can abort the
   * in-flight session.
   */
  sessionAbort: AbortController | null = null;
  overlayCleanup: (() => void) | null = null;

  /** Offline-capture callback IDs — static so dispose() can remove them
   *  unconditionally even if the loop is parked on an `await` and
   *  hasn't reached its finally yet. */
  static readonly CAPTURE_CALLBACK_ID = 'recording-offline-capture';
  static readonly KEEPALIVE_CALLBACK_ID = 'recording-offline-keepalive';

  constructor(
    private readonly sceneManager: SceneManager,
    private readonly animationController: AnimationController,
    private readonly hooks: OfflineCaptureStrategyHooks
  ) {}

  canRun(state: SessionState): boolean {
    return !state.isRecording && !state.isOfflineCaptureActive;
  }

  async run(opts: RecordingOptions, mode: RecordingMode, session: RecordingSession): Promise<void> {
    return this.runOfflineCaptureLoop(opts.outputFormat as OfflineMode, opts, session, mode);
  }

  /** Synchronously stop the loop. The loop's await checkpoints
   *  observe `sessionAbort.signal` and break out cleanly. */
  abort(): void {
    this.sessionAbort?.abort('user-stop');
  }

  dispose(): void {
    this.sessionAbort?.abort('disposed');
    this.overlayCleanup?.();
    // The normal loop path removes these in its finally; this covers
    // the dispose-while-awaiting case where the loop hasn't reached
    // its finally yet.
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.CAPTURE_CALLBACK_ID);
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID);
  }

  // ── Test-only access (Panel proxies forward to these) ─────────
  cleanupOfflineOverlayForTests(): void {
    this.overlayCleanup?.();
  }

  private async runOfflineCaptureLoop(
    mode: OfflineMode,
    opts: RecordingOptions,
    session: RecordingSession,
    recordingMode: RecordingMode
  ): Promise<void> {
    // The dialog gets the mode the panel is actually in. Deriving it from
    // the format instead described a Turntable + PNG/MP4 capture with
    // Smooth off as a "Video" recording — no 360° line, no frame count —
    // even though this loop always rotates a full turntable.
    const confirmed = await session.showConfirmationDialog({
      mode: recordingMode,
      options: opts,
    });
    if (!confirmed || session.isDisposed()) return;

    // Establish session ownership BEFORE any state mutation. dispose()
    // reads `sessionAbort` to abort an in-flight session; if we
    // assign it later (after hideAllPanels / saveRecordingState / the
    // first rAF), a dispose during that early window leaves the
    // abort controller null and the function continues to bring up
    // overlay/recording flags on a disposed panel.
    const sessionAbort = new AbortController();
    this.sessionAbort = sessionAbort;

    const bailEarly = (): void => {
      session.restoreRecordingState();
      // Same repaint guarantee as the main teardown: restoreRecordingState
      // resizes the render target back, which clears the canvas, and no
      // keep-alive is registered on this path — so a stopped loop would
      // leave the viewer blank until the next mouse move. Skipped when the
      // session is disposed (see the note in the finally).
      if (!session.isDisposed()) {
        this.animationController.startAnimation();
      }
      if (this.sessionAbort === sessionAbort) {
        this.sessionAbort = null;
      }
    };

    this.hooks.hideAllPanels();

    // Save state, disable DPR, lock resize, scale resolution.
    // Dimensions are aligned DOWN to even numbers — H.264/H.265 with
    // yuv420p need even width and height, and encoders pad internally to
    // their own macroblock size, so nothing here has to.
    //
    // `videoResolution === 0` is the panel's "Native" option, documented
    // in its tooltip as "current canvas size" — so capture at the size
    // the canvas actually has (display size × native DPR) rather than
    // silently forcing 1080. Forcing it downscaled every Retina/4K
    // capture and, because it changed the capture-to-CSS pixel ratio,
    // rescaled the composited overlays with it.
    //
    // The display size comes from post-processing, NOT from
    // `renderer.getSize()`: the renderer is handed the SSAA-multiplied
    // size, so under SSAA it reports `display × multiplier` and asking
    // to render THAT squares the multiplier (2× on a 3024×1700 canvas
    // asked for a 12096×6800 target). `saveRecordingState` re-applies
    // the multiplier itself, so the frames on disk still carry SSAA —
    // which is what the real-time path's canvas backbuffer includes too.
    const displayH = this.sceneManager.postProcessing.getDisplaySize().height;
    const nativeDPR = session.adaptiveDPRManager?.getNativeDPR() ?? window.devicePixelRatio ?? 1;
    const targetH =
      opts.videoResolution > 0 ? opts.videoResolution : Math.round(displayH * nativeDPR);
    session.saveRecordingState({
      disableDPR: true,
      lockResize: true,
      scaleResolution: { targetH, alignEven: true },
    });
    await new Promise((r) => requestAnimationFrame(r));

    if (session.isDisposed() || sessionAbort.signal.aborted) {
      bailEarly();
      return;
    }

    // Compute turntable parameters
    const fps = opts.videoFPS;
    const durationSeconds = 360 / opts.turntableSpeed;
    const totalFrames = Math.ceil(durationSeconds * fps);

    const controls = this.sceneManager.controls.getControls();
    if (!(controls instanceof LuxarOrbitControls)) {
      log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
      bailEarly();
      return;
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

    // Build the per-mode driver. EXR mode flips the session's
    // isEXRSequenceRecording flag in finalize via a callback so the
    // driver doesn't need to know about that field.
    const driver: OfflineCaptureDriver =
      mode === 'png' || mode === 'webp' || mode === 'jpeg'
        ? new ImageSequenceDriver(mode)
        : mode === 'exr'
          ? new ExrSequenceDriver(() => {
              session.isEXRSequenceRecording = false;
            })
          : new VideoModeDriver(mode);

    if (mode === 'exr') {
      session.isEXRSequenceRecording = true;
    }
    session.isRecording = true;
    session.isOfflineCaptureActive = true;
    session.recordingStartTime = Date.now();
    session.showRecordingIndicator();

    // Offline overlay — modal dialog with ARIA semantics, focus trap,
    // and explicit Escape handler that aborts the session.
    const overlay = document.createElement('div');
    overlay.className = 'luxar-recording-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'luxar-recording-overlay-label');
    overlay.setAttribute('aria-describedby', 'luxar-recording-overlay-counter');
    overlay.innerHTML = `
      <div class="luxar-recording-overlay__content">
        <canvas class="luxar-recording-overlay__preview"></canvas>
        <div class="luxar-recording-overlay__progress">
          <span id="luxar-recording-overlay-label" class="luxar-recording-overlay__label">Capturing frames...</span>
          <span id="luxar-recording-overlay-counter" class="luxar-recording-overlay__counter">0/${totalFrames}</span>
        </div>
        <button class="luxar-recording-overlay__cancel">Cancel</button>
      </div>
    `;
    const previewCanvas = overlay.querySelector(
      '.luxar-recording-overlay__preview'
    ) as HTMLCanvasElement;
    const previewCtx = previewCanvas.getContext('2d');
    const cancelButton = overlay.querySelector(
      '.luxar-recording-overlay__cancel'
    ) as HTMLButtonElement | null;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const handleCancel = (): void => {
      session.isRecording = false;
      sessionAbort.abort('user-cancel');
    };
    const handleOverlayKeydown = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        handleCancel();
        return;
      }
      if (e.key === 'Tab') {
        const focusable = Array.from(
          overlay.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => !el.hasAttribute('disabled'));
        if (focusable.length > 0) {
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          const active = document.activeElement as HTMLElement | null;
          if (e.shiftKey && active === first) {
            e.preventDefault();
            last.focus();
          } else if (!e.shiftKey && active === last) {
            e.preventDefault();
            first.focus();
          }
        }
        e.stopPropagation();
        return;
      }
      e.stopPropagation();
    };
    cancelButton?.addEventListener('click', handleCancel);
    overlay.addEventListener('keydown', handleOverlayKeydown, true);

    let overlayCleaned = false;
    const cleanupOfflineOverlay = (): void => {
      if (overlayCleaned) return;
      overlayCleaned = true;
      cancelButton?.removeEventListener('click', handleCancel);
      overlay.removeEventListener('keydown', handleOverlayKeydown, true);
      overlay.remove();
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus();
      }
      if (this.overlayCleanup === cleanupOfflineOverlay) {
        this.overlayCleanup = null;
      }
    };
    this.overlayCleanup = cleanupOfflineOverlay;

    getViewerContainer().appendChild(overlay);
    cancelButton?.focus();

    const counterEl = overlay.querySelector('.luxar-recording-overlay__counter');
    const labelEl = overlay.querySelector('.luxar-recording-overlay__label');

    // One timestamped stem for the whole capture. `generateFilename`
    // stamps `new Date()` on every call, so calling it per artifact —
    // the ZIP's save-dialog name at setup, its fallback download name at
    // finalize, the script's output base — hands out names that disagree
    // whenever a capture crosses a second boundary.
    const captureBase = this.hooks.generateFilename('zip').replace(/\.zip$/, '');

    // Build the dependency context the driver needs.
    const ctx: CaptureContext = {
      sceneManager: this.sceneManager,
      fps,
      renderFrameToCanvas: () => this.hooks.renderFrameToCanvas(),
      generateFilename: (ext) => `${captureBase}.${ext}`,
      generateFfmpegScript: (frames, ext) =>
        generateFfmpegScriptPure({
          fps,
          frameCount: frames,
          frameExt: ext,
          mode: recordingMode === 'turntable' ? 'turntable' : 'video',
          outputBase: captureBase,
          // EXR frames are scene-linear and pre-grade, so the script has
          // to re-apply the viewer's display transform. LDR frames are
          // already graded and ignore this.
          grade: ext === 'exr' ? readGradeSettings(this.sceneManager) : undefined,
        }),
      downloadBlob: (blob, filename) => this.hooks.downloadBlob(blob, filename),
      computeVideoBitrate: (w, h) =>
        computeVideoBitratePure(w, h, opts.videoFPS, opts.videoQuality),
      showToast,
      logWarning: (msg) => log.warning(Modules.RECORDING, msg),
      logError: (msg) => log.error(Modules.RECORDING, msg),
      imageQuality: opts.imageQuality,
      videoCodec: opts.videoCodec,
      env: window as unknown as CaptureContext['env'],
      signal: sessionAbort.signal,
    };

    const progress = {
      setLabel: (text: string): void => {
        if (labelEl) labelEl.textContent = text;
      },
      setPreview: (canvas: HTMLCanvasElement): void => {
        if (!previewCtx) return;
        if (previewCanvas.width !== canvas.width || previewCanvas.height !== canvas.height) {
          previewCanvas.width = canvas.width;
          previewCanvas.height = canvas.height;
        }
        previewCtx.drawImage(canvas, 0, 0);
      },
    };

    const captureCallbackId = OfflineCaptureStrategy.CAPTURE_CALLBACK_ID;
    const keepAliveId = OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID;
    let capturedFrames = 0;
    /**
     * Frames the loop actually tried to capture — incremented before
     * `driver.captureFrame`, so it counts the ones that threw too. The LOD
     * settle report's denominator: `lodSettleTimeouts` is also counted before
     * the capture attempt, so measuring it against `capturedFrames` (successes
     * only) could print "2 of 1".
     */
    let attemptedFrames = 0;
    let setupCompleted = false;
    let finalizeSucceeded = false;

    try {
      // The loop's own render is redundant from here on — the capture
      // renders its own pipeline pass per frame (see the render-skip
      // predicate wired in `core/app/init/pipeline`). Set INSIDE the try so
      // the finally below always clears it: the flag suppresses the loop's
      // render globally, so escaping with it stuck true blanks the viewport
      // until a page reload. Nothing between the recording flags above and
      // this point can paint a frame — the REC indicator, the overlay and the
      // driver context are all built synchronously, and the last yield is the
      // rAF well above them.
      session.isLoopRenderSuppressed = true;
      const setupOk = await driver.setup(ctx);
      if (!setupOk) {
        return;
      }
      setupCompleted = true;
      if (sessionAbort.signal.aborted) return;

      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 3;

      // LOD settle bookkeeping (see the drain inside the frame loop).
      let lodSettleTimeouts = 0;
      let consecutiveLodTimeouts = 0;
      let lodDrainDisabled = false;
      /**
       * Sticky counterpart of `lodDrainDisabled`: true once the latch has fired
       * at any point in the run, and never cleared by a re-arm. The latch
       * itself comes back on, so it cannot answer "were any frames captured
       * without waiting?" at the end of the run — this can. It doubles as the
       * warn-once guard, so a run that latches and re-arms repeatedly logs one
       * line rather than one per latch.
       */
      let lodDrainEverDisabled = false;
      let lodPredicateThrew = false;

      /**
       * Read the injected quiescence predicate, tri-state and throw-safe.
       *
       * `null` means "do not wait": either this scene has no lod_group to wait
       * for (no hook, no registry, or a registry with none in it), or the
       * predicate threw. The guard mirrors
       * `AnimationController.pacingSuspended()` and exists for the same
       * reason — this is a guard on the INJECTION POINT, not on a known
       * thrower. Unguarded, a throw would be caught by the loop's outer
       * handler, reported as "Recording failed" and discard the whole
       * sequence, which is a catastrophic price for a diagnostic predicate.
       *
       * The degradation is FRAME-SCOPED, not run-scoped: the predicate is
       * re-probed on the next frame, so a transient throw costs that one
       * frame's wait and nothing more. Only the WARNING is once-per-run, so a
       * persistently throwing provider does not emit one line per frame.
       *
       * A frame whose opening probe throws is skipped whole: it counts no
       * settle timeout AND does not reset `consecutiveLodTimeouts`. So an
       * alternating throw/timeout run latches `lodDrainDisabled` over more
       * than `MAX_CONSECUTIVE_LOD_TIMEOUTS` frames. That is intended — the
       * streak measures consecutive *timeouts*, not consecutive frames — but
       * it is worth stating rather than rediscovering.
       */
      const readLODSettled = (): boolean | null => {
        try {
          return this.hooks.isLODSettled?.() ?? null;
        } catch (err) {
          if (!lodPredicateThrew) {
            lodPredicateThrew = true;
            log.warning(
              Modules.RECORDING,
              'LOD settle predicate threw — that frame was captured without waiting for LOD ' +
                `(warned once per run; the predicate is retried on the next frame): ${err}`
            );
          }
          return null;
        }
      };

      // Wake the rAF loop, exactly as the real-time strategy does. The
      // turntable's rotation is applied from a per-frame callback, and
      // those only run while the loop is animating — but the loop
      // idle-stops after ~2s of no interaction, which is the normal
      // state by the time the user has read the panel and confirmed the
      // dialog. Registering a `continuous` callback only KEEPS a running
      // loop alive; it never restarts a stopped one. Without this call
      // the camera never rotates and the capture silently emits N
      // identical frames — the capture path renders its own pipeline
      // pass (`renderToImageData`), so frames are still produced, just
      // all from the same pose.
      //
      // Don't "optimize" this away by orbiting the camera inline and
      // leaving the loop stopped: the depth-sort scheduler and the LOD
      // group selector are per-frame callbacks too, so a stopped loop
      // would freeze depth order and LOD level at the opening pose
      // while the camera swings a full turn.
      this.animationController.startAnimation();
      this.animationController.addPerFrameCallback(keepAliveId, () => {}, { continuous: true });

      for (let i = 0; i < totalFrames; i++) {
        if (!session.isRecording) break;
        if (sessionAbort.signal.aborted) break;
        if (driver.shouldAbort?.()) break;

        this.animationController.addPerFrameCallback(captureCallbackId, () => {
          if (i > 0) controls.applyOrbitRotation(anglePerFrame);
          // Called on frame 0 too, deliberately: it drives the dolly to phase
          // 0, undoing any offset left over from a live oscillation so the
          // capture starts at the true baseline distance.
          if (dollyCycles > 0) controls.applyOrbitDolly(dollyPhaseFor(i));
        });

        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

        this.animationController.removePerFrameCallback(captureCallbackId);

        // ── Wait for the LOD selector to settle on this pose (#1695) ──
        // The rAF loop runs for the whole capture, so the auto-LOD selector
        // is live and frustum-aware: a tile whose world bbox left the frustum
        // mid-orbit was demoted to its coarsest ready level (and the
        // resident-byte pass may have released its fine one), and swinging
        // back into view kicks an ASYNCHRONOUS reload. Grabbing the frame
        // immediately bakes that coarse level into the ZIP/MP4 and pops back
        // a few frames later. So spend extra rAF ticks here until every
        // in-frame LOD group shows its selected level at final quality.
        //
        // This MUST stay after the orbit callback's removal above: the camera
        // pose is fixed from that point on, so the extra frames only let
        // pending loads land. Draining before it would keep advancing the
        // turntable while we wait, smearing the sweep.
        //
        // The FIRST rAF below is mandatory, not part of the wait — it is what
        // makes the predicate describe the pose we are about to capture.
        // `AnimationController.animate` runs `controls.update()`, then every
        // per-frame callback in Map insertion order, then the render. The LOD
        // selector (`lod-group-selector`) is registered at pipeline init,
        // while the capture's orbit callback is removed and re-added on every
        // loop iteration and is therefore always LAST in that Map — and
        // `LuxarOrbitControls.applyOrbitRotation` moves the camera
        // synchronously. So inside the single rAF awaited above, the selector
        // evaluated pose N−1 and only THEN did the orbit callback advance the
        // camera to pose N. Polling straight away would read state computed
        // for the previous pose, and on the exact frame a tile re-enters the
        // frustum the selector has not seen the re-entry yet — the predicate
        // would report settled and that one frame would still be filmed
        // coarse, which is precisely the pop this drain exists to remove.
        // One more tick puts the selector on pose N.
        //
        // Cost: one rAF per frame even on a fully settled scene THAT HAS LOD
        // GROUPS. At ~16 ms against a full pipeline render plus an async GPU
        // readback plus an encode for every frame, that is noise — and a scene
        // with no lod_group at all pays nothing, because the predicate answers
        // `null` and the whole block below (that tick included) is skipped.
        //
        // Not force-finest: a capture visits the whole scene, so pinning the
        // finest level across a tiled partition would make peak residency the
        // entire dataset. Waiting costs time, not memory.
        if (!sessionAbort.signal.aborted && session.isRecording) {
          if (lodDrainDisabled) {
            // Latched off after three consecutive timeouts — but the latch
            // re-arms (see MAX_CONSECUTIVE_LOD_TIMEOUTS). This probe is FREE:
            // no rAF, no poll, so a latched frame still costs exactly what it
            // did pre-#1695. Its boolean describes pose N−1 rather than the
            // pose being captured, which is fine for a re-arm signal — this
            // frame is captured undrained either way, and the next frame gets
            // the full, correctly-timed drain.
            if (readLODSettled() === true) {
              lodDrainDisabled = false;
              consecutiveLodTimeouts = 0;
            }
          } else {
            // Tri-state probe, spent BEFORE the mandatory tick so a scene with
            // nothing to wait for does not pay for it. Only the NULL-ness of
            // this read is used: its boolean answer describes pose N−1 (see
            // above) and is deliberately discarded.
            const drainApplies = readLODSettled() !== null;
            if (drainApplies) {
              // `performance.now()` rather than `Date.now()`: monotonic, so an
              // NTP/DST step cannot move the deadline backwards mid-drain. It
              // is also the clock the animation controller measures frames
              // with.
              const deadline = performance.now() + LOD_SETTLE_TIMEOUT_MS;
              // The mandatory selector-catch-up tick (see above). Counts
              // against the frame budget like any other drain frame.
              await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
              let drainFrames = 1;
              // Re-check before polling: a Stop landing during that first frame
              // must break out without being counted as a settle timeout. A
              // `null` here (the predicate threw) reads as settled — "do not
              // wait" — rather than as a timeout to report.
              let settled =
                sessionAbort.signal.aborted || !session.isRecording
                  ? false
                  : readLODSettled() !== false;
              // Bounded by BOTH a wall-clock deadline and a frame count —
              // whichever trips first (see LOD_SETTLE_MAX_FRAMES for why the
              // frame cap is not redundant).
              while (
                !settled &&
                !sessionAbort.signal.aborted &&
                session.isRecording &&
                drainFrames < LOD_SETTLE_MAX_FRAMES &&
                performance.now() < deadline
              ) {
                await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
                drainFrames++;
                settled = readLODSettled() !== false;
              }
              if (settled) {
                consecutiveLodTimeouts = 0;
              } else if (!sessionAbort.signal.aborted && session.isRecording) {
                // A genuine timeout (not an abort/stop breaking out of the
                // wait).
                lodSettleTimeouts++;
                consecutiveLodTimeouts++;
                if (consecutiveLodTimeouts >= MAX_CONSECUTIVE_LOD_TIMEOUTS) {
                  lodDrainDisabled = true;
                  // Warned once per run, not once per latch: the latch re-arms,
                  // so a scene that keeps stalling would otherwise log a line
                  // every three frames.
                  if (!lodDrainEverDisabled) {
                    lodDrainEverDisabled = true;
                    log.warning(
                      Modules.RECORDING,
                      `${MAX_CONSECUTIVE_LOD_TIMEOUTS} consecutive LOD settle timeouts — ` +
                        'pausing the per-frame LOD wait. It resumes on the first frame whose ' +
                        'free probe reports the scene settled; frames captured meanwhile may ' +
                        'not show the level the selector settled on.'
                    );
                  }
                }
              }
            }
          }
        }

        if (sessionAbort.signal.aborted) break;

        attemptedFrames++;
        try {
          await driver.captureFrame(ctx, capturedFrames, progress);
          capturedFrames++;
          consecutiveErrors = 0;
        } catch (err) {
          consecutiveErrors++;
          log.error(Modules.RECORDING, `Frame ${i + 1} capture failed: ${err}`);
          if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            log.error(
              Modules.RECORDING,
              `${MAX_CONSECUTIVE_ERRORS} consecutive failures — aborting capture. ` +
                'The browser may not support 10-bit encoding at this resolution.'
            );
            showToast('HDR video encoding failed — try EXR sequence instead');
            break;
          }
        }

        if (counterEl) counterEl.textContent = `${i + 1}/${totalFrames}`;
      }

      if (sessionAbort.signal.aborted) {
        return;
      }

      // Make a degraded sequence visible rather than a mystery: some frames
      // were captured before their LOD levels finished loading, so they may not
      // show the level the selector had settled on.
      //
      // "May not show the settled level" rather than "may be coarse", in both
      // branches, because the predicate is direction-blind: `displayed !==
      // active` also fires while the never-downgrade gate legitimately holds a
      // FINER previously-displayed level over a coarser aspiration that is
      // still streaming. Those frames look better than the selection, not
      // worse, and telling the user they are coarse would be wrong.
      //
      // The two branches report genuinely different things, and conflating
      // them was actively misleading. Once the latch has fired (sticky
      // `lodDrainEverDisabled`, since the latch itself re-arms),
      // `lodSettleTimeouts` stops describing the run: it counts only the
      // frames that WAITED and gave up, while every frame captured while the
      // wait was off was taken with no wait at all — so on a 600-frame capture
      // a handful of counted timeouts can sit in front of hundreds of undrained
      // frames. (It is NOT pinned at MAX_CONSECUTIVE_LOD_TIMEOUTS: that
      // constant bounds the consecutive STREAK, and a run that alternates
      // timeout/settle can reach any total before three land in a row.) So: an
      // exact count only when waiting stayed on for the whole run; otherwise
      // say what actually happened and claim no number. The exact branch's
      // denominator is `attemptedFrames`, not `capturedFrames` — a timeout is
      // counted before the capture attempt, so a frame that timed out and then
      // threw would otherwise be in the numerator but not the denominator.
      if (lodSettleTimeouts > 0) {
        if (lodDrainEverDisabled) {
          log.warning(
            Modules.RECORDING,
            `LOD settle waiting was paused after ${MAX_CONSECUTIVE_LOD_TIMEOUTS} consecutive ` +
              `timeouts (${LOD_SETTLE_TIMEOUT_MS} ms / ${LOD_SETTLE_MAX_FRAMES} frame limit ` +
              'each), and the frames captured while it was off were taken without waiting. An ' +
              `unknown number of this run's ${capturedFrames} frame(s) may therefore not show ` +
              'the level the selector had settled on.'
          );
          // Says what happened (waiting was paused), NOT "LOD never settled" —
          // a 600-frame run that settled cleanly for 550 frames and then hit a
          // network stall lands here too.
          showToast('Paused waiting for LOD — some frames may not show the settled level');
        } else {
          log.warning(
            Modules.RECORDING,
            `${lodSettleTimeouts} of ${attemptedFrames} frame(s) were captured before their ` +
              `LOD levels settled (${LOD_SETTLE_TIMEOUT_MS} ms / ${LOD_SETTLE_MAX_FRAMES} frame ` +
              'limit). Those frames may not show the level the selector had settled on.'
          );
          showToast(`${lodSettleTimeouts} frame(s) captured before LOD settled`);
        }
      }

      try {
        await driver.finalize(ctx, capturedFrames, progress);
        finalizeSucceeded = true;
      } catch (err) {
        log.error(Modules.RECORDING, `Offline ${mode} finalize failed: ${err}`);
        showToast('Recording finalize failed');
      }
    } catch (err) {
      log.error(Modules.RECORDING, `Offline ${mode} capture failed: ${err}`);
      showToast('Recording failed');
    } finally {
      // Clear the render-skip flag FIRST: it globally suppresses the
      // loop's render, and a driver abort that never settles would
      // otherwise leave the viewer frozen with no recovery but a reload.
      // Only this flag — the mutual-exclusion flags below must survive
      // the abort await.
      session.isLoopRenderSuppressed = false;
      if (setupCompleted && !finalizeSucceeded) {
        try {
          const reason = sessionAbort.signal.aborted
            ? sessionAbort.signal.reason === 'user-cancel'
              ? 'user-cancel'
              : 'disposed'
            : 'error';
          await driver.abort?.(ctx, reason as 'disposed' | 'user-cancel' | 'error');
        } catch (abortErr) {
          log.warning(Modules.RECORDING, `Driver abort during cleanup failed: ${abortErr}`);
        }
      }
      this.animationController.removePerFrameCallback(captureCallbackId);
      this.animationController.removePerFrameCallback(keepAliveId);
      session.hideRecordingIndicator();
      session.isRecording = false;
      // NOT cleared before the abort await above: `ScreenshotStrategy`
      // gates mutual exclusion on this flag, so a screenshot started
      // mid-teardown would overwrite and then null the single
      // `savedRecordingState` slot, leaving `restoreRecordingState()`
      // below a no-op — the viewer stuck at capture resolution with
      // resize locked until a page reload.
      session.isOfflineCaptureActive = false;
      session.isEXRSequenceRecording = false;
      cleanupOfflineOverlay();
      session.restoreAutoRotate();
      session.restoreAutoDolly();
      session.restoreRecordingState();
      // Guarantee exactly one repaint after teardown. restoreRecordingState
      // resizes the render target back, which clears the canvas, and the
      // keep-alive callback is already gone by now — so a loop that is
      // still stopped (or that the idle timer stops in the gap right after
      // the resize) leaves the viewer blank until the next mouse move.
      // The render-skip predicate reads `isLoopRenderSuppressed`, cleared
      // at the top of this finally, so this frame is a real render.
      //
      // Never on a disposed session: dispose() tears the AnimationController
      // down BEFORE the RecordingPanel (see `runDisposePipeline`), and
      // RecordingPanel.dispose() only ABORTS an in-flight capture — a loop
      // parked on an await resumes here a tick later. Waking it then would
      // restart the rAF loop against a disposed PostProcessingManager.
      if (!session.isDisposed()) {
        this.animationController.startAnimation();
      }
      if (this.sessionAbort === sessionAbort) {
        this.sessionAbort = null;
      }
    }
  }
}
