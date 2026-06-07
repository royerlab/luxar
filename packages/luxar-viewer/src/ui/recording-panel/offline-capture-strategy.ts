/**
 * Offline (frame-by-frame) capture strategy.
 *
 * Used for:
 * - EXR sequence recording (ZIP of EXR frames, full float precision)
 * - Turntable smooth mode (renders each frame individually for perfectly
 *   smooth output regardless of GPU FPS)
 *
 * Unlike real-time MediaRecorder capture, this loop is fully decoupled
 * from the browser's animation frame rate. Each frame is:
 * 1. Camera orbited by one step (quaternion rotation, same as auto-rotate)
 * 2. Scene rendered (full pipeline)
 * 3. Pixels read back (synchronous GPU stall — intentional)
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

    // Per-frame rotation step: frame 0 captures the starting view without rotation,
    // then frames 1..N-1 each advance by one step to complete exactly 2π total.
    const anglePerFrame = totalFrames > 1 ? (2 * Math.PI) / (totalFrames - 1) : 0;

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
      session.isOfflineCaptureActive = false;
      session.isEXRSequenceRecording = false;
      cleanupOfflineOverlay();
      session.restoreAutoRotate();
      session.restoreRecordingState();
      if (this.sessionAbort === sessionAbort) {
        this.sessionAbort = null;
      }
    }
  }
}
