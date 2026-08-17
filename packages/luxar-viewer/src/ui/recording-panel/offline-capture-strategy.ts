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
 * 2. Scene rendered (full pipeline, into the capture's own target)
 * 3. Pixels read back asynchronously (PBO fence on WebGL2, mapAsync on
 *    WebGPU) — the rAF loop keeps ticking through the await, which is
 *    why the loop's own render is suppressed for the whole capture
 *    (see the render-skip predicate wired in `core/app/init/pipeline`)
 * 4. Frame stored / encoded
 * 5. Brief yield to keep the browser responsive
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
 * 2. The post-saveRecordingState body is wrapped in try/finally so an
 *    exception from driver.setup, driver.captureFrame, driver.finalize,
 *    or any DOM/state mutation cannot leave the panel with a stuck
 *    overlay, hidden panels, scaled renderer, or stale recording flags.
 * 3. The finally block is idempotent — every removal/restore handles
 *    the "wasn't set" case gracefully.
 */

import { log, Modules } from '../../utils/log';
import { getViewerContainer } from '../../utils/viewer-container';
import { showToast } from '../toast';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { LuxarOrbitControls } from '../../controls/luxar-orbit-controls';
import {
  computeVideoBitrate as computeVideoBitratePure,
  generateFfmpegScript as generateFfmpegScriptPure,
} from './media-utilities';
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
}

type OfflineMode = 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg';

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

  async run(
    opts: RecordingOptions,
    _mode: RecordingMode,
    session: RecordingSession
  ): Promise<void> {
    return this.runOfflineCaptureLoop(opts.outputFormat as OfflineMode, opts, session);
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
    session: RecordingSession
  ): Promise<void> {
    const confirmed = await session.showConfirmationDialog({
      mode: opts.outputFormat === 'exr' || opts.frameByFrame ? 'turntable' : 'video',
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
    // Dimensions are rounded to a multiple of 16 (macroblock alignment).
    const targetH = opts.videoResolution > 0 ? opts.videoResolution : 1080;
    session.saveRecordingState({
      disableDPR: true,
      lockResize: true,
      scaleResolution: { targetH, align16: true },
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

    // Per-frame rotation step: frame 0 captures the starting view without
    // rotation, then frames 1..N-1 each advance by one step, so the N frames
    // cover [0, 2π) and the LAST frame stops one step short of the first.
    // That step is 2π/N, NOT 2π/(N-1): dividing by N-1 lands the last frame
    // exactly back on the start pose, and a turntable is made to loop — the
    // duplicate shows up as a one-frame hitch at every wrap.
    const anglePerFrame = totalFrames > 0 ? (2 * Math.PI) / totalFrames : 0;

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
    // The loop's own render is redundant from here on — the capture
    // renders its own pipeline pass per frame (see the render-skip
    // predicate wired in `core/app/init/pipeline`).
    session.isLoopRenderSuppressed = true;
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

    // Build the dependency context the driver needs.
    const ctx: CaptureContext = {
      sceneManager: this.sceneManager,
      fps,
      renderFrameToCanvas: () => this.hooks.renderFrameToCanvas(),
      generateFilename: (ext) => this.hooks.generateFilename(ext),
      generateFfmpegScript: (rate, frames, ext) => generateFfmpegScriptPure(rate, frames, ext),
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
    let setupCompleted = false;
    let finalizeSucceeded = false;

    try {
      const setupOk = await driver.setup(ctx);
      if (!setupOk) {
        return;
      }
      setupCompleted = true;
      if (sessionAbort.signal.aborted) return;

      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 3;

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
        });

        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

        this.animationController.removePerFrameCallback(captureCallbackId);

        if (sessionAbort.signal.aborted) break;

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
