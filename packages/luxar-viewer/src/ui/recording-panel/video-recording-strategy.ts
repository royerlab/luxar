/**
 * Real-time video recording strategy.
 *
 * Drives the browser's MediaRecorder via canvas.captureStream(). Wall-clock
 * driven (frames are captured as the browser paints), unlike
 * OfflineCaptureStrategy which is frame-index driven.
 *
 * Owns:
 * - `mediaRecorder`, `captureStream`, `recordedChunks` — the encoder
 * - `durationTimer` — automatic-stop timer
 * - `keepAliveCallbackId`, `turntableCallbackId` — per-frame callback IDs
 *
 * Lifecycle:
 * - `run(opts, mode, session)` resolves when the recording is finalized
 *   (i.e., after `mediaRecorder.onstop` fires). Caller `await`s it to
 *   know when the user-initiated capture is complete.
 * - `abort()` requests stop synchronously; the onstop handler does the
 *   actual unwind.
 * - `dispose()` synchronously stops tracks + nulls fields; the async
 *   onstop handler captured Session before disposal and checks
 *   `session.isDisposed()` to short-circuit.
 *
 * Important invariants:
 * 1. The entire setup phase is wrapped in try/catch; the catch performs
 *    a symmetric undo of every state mutation up to the throw point, so
 *    a failure in canvas.captureStream() / new MediaRecorder() does NOT
 *    leave panels hidden, DPR disabled, resize locked, or per-frame
 *    callbacks registered.
 * 2. `mediaRecorder.onstop` always checks `session.isDisposed()` first
 *    and takes the minimal-cleanup early-return branch if so. This
 *    matters because Panel.dispose() sets disposed=true BEFORE calling
 *    `mediaRecorder.stop()`, so the (async) onstop fires post-disposal.
 * 3. captureStream tracks must be explicitly `.stop()`ed —
 *    `mediaRecorder.stop()` does NOT stop them, so without this the
 *    browser's media indicator stays lit forever.
 */

import { log, Modules } from '../../utils/log';
import { showToast } from '../toast';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { LuxarOrbitControls } from '../../controls/luxar-orbit-controls';
import {
  computeVideoBitrate as computeVideoBitratePure,
  getSupportedMimeType as getSupportedMimeTypePure,
} from './media-utilities';
import type { CaptureStrategy, SessionState } from './capture-strategy';
import type { RecordingSession } from './session';
import type { RecordingMode, RecordingOptions } from './types';

export interface VideoRecordingStrategyHooks {
  hideAllPanels(): void;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
}

export class VideoRecordingStrategy implements CaptureStrategy {
  readonly kind = 'video' as const;

  // ── MediaRecorder state (non-private so Panel can proxy
  //    `(panel as any).mediaRecorder` etc. through to here for tests) ──
  mediaRecorder: MediaRecorder | null = null;
  // canvas.captureStream() returns a MediaStream whose tracks live
  // until explicitly stopped. mediaRecorder.stop() does NOT stop the
  // underlying tracks; tracked here so cleanupCaptureStream() can.
  captureStream: MediaStream | null = null;
  recordedChunks: Blob[] = [];
  durationTimer: ReturnType<typeof setTimeout> | null = null;

  // ── Per-frame callback IDs ─────────────────────────────────────
  readonly keepAliveCallbackId = 'recording-keepalive';
  readonly turntableCallbackId = 'recording-turntable';

  constructor(
    private readonly sceneManager: SceneManager,
    private readonly animationController: AnimationController,
    private readonly hooks: VideoRecordingStrategyHooks
  ) {}

  canRun(state: SessionState): boolean {
    return !state.isRecording && !state.isOfflineCaptureActive;
  }

  async run(opts: RecordingOptions, mode: RecordingMode, session: RecordingSession): Promise<void> {
    if (session.isRecording) return;

    const mimeType = getSupportedMimeTypePure();
    if (!mimeType) {
      showToast('Video recording not supported in this browser');
      return;
    }

    const confirmed = await session.showConfirmationDialog({ mode, options: opts });
    if (!confirmed || session.isDisposed()) return;

    // Wrap the entire setup phase. Without this, a throw from
    // canvas.captureStream(), `new MediaRecorder(...)`, or
    // mediaRecorder.start() would leave panels hidden, DPR disabled,
    // resize locked, the keepalive callback registered, and no onstop
    // to unwind any of it.
    try {
      this.hooks.hideAllPanels();

      session.saveRecordingState({
        disableDPR: true,
        lockResize: true,
        scaleResolution:
          opts.videoResolution > 0 ? { targetH: opts.videoResolution } : undefined,
      });
      if (opts.videoResolution > 0) {
        await new Promise((r) => requestAnimationFrame(r));
        if (session.isDisposed()) {
          session.restoreRecordingState();
          return;
        }
      }

      const canvas = this.sceneManager.renderer.domElement;
      const videoBitsPerSecond = computeVideoBitratePure(
        canvas.width,
        canvas.height,
        opts.videoFPS,
        opts.videoQuality
      );

      log.info(
        Modules.RECORDING,
        `Starting video recording (${mimeType}, ${opts.videoFPS} FPS, ` +
          `${Math.round(videoBitsPerSecond / 1_000_000)}Mbps, ${canvas.width}x${canvas.height}, ` +
          `mode: ${mode})`
      );

      this.animationController.startAnimation();
      this.animationController.addPerFrameCallback(this.keepAliveCallbackId, () => {}, {
        continuous: true,
      });

      this.captureStream = canvas.captureStream(opts.videoFPS);
      this.mediaRecorder = new MediaRecorder(this.captureStream, {
        mimeType,
        videoBitsPerSecond,
      });
      this.recordedChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };

      // Return a promise that resolves when onstop unwinds everything.
      // Captured locally so abort() / dispose() can synchronously
      // request stop without awaiting; the await-er sees completion
      // via this promise.
      const onstopComplete = new Promise<void>((resolve) => {
        const recorder = this.mediaRecorder as MediaRecorder;
        recorder.onstop = () => {
          try {
            if (session.isDisposed()) {
              this.recordedChunks = [];
              session.isRecording = false;
              session.hideRecordingIndicator();
              this.cleanupCaptureStream();
              return;
            }

            const blob = new Blob(this.recordedChunks, { type: mimeType });
            const totalElapsed = ((Date.now() - session.recordingStartTime) / 1000).toFixed(1);
            log.info(
              Modules.RECORDING,
              `Recording finalized: ${this.recordedChunks.length} chunks, ` +
                `${(blob.size / (1024 * 1024)).toFixed(1)} MB, ${totalElapsed}s elapsed`
            );
            this.hooks.downloadBlob(blob, this.hooks.generateFilename('webm'));
            this.recordedChunks = [];
            session.isRecording = false;
            session.hideRecordingIndicator();

            this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
            this.animationController.removePerFrameCallback(this.turntableCallbackId);
            session.cleanupSyncListener();
            session.restoreAutoRotate();
            session.restoreRecordingState();
            this.cleanupCaptureStream();
            showToast('Video saved');
          } finally {
            resolve();
          }
        };
      });

      this.mediaRecorder.start(100);
      session.isRecording = true;
      session.recordingStartTime = Date.now();
      session.showRecordingIndicator();

      // Duration limit
      if (opts.videoDurationLimit > 0) {
        this.durationTimer = setTimeout(() => {
          this.abort();
        }, opts.videoDurationLimit * 1000);
      }

      // Start slider sync if enabled
      if (mode === 'video' && opts.syncToSlider && session.animationManager) {
        session.startSliderSync(opts.syncDimensionIndex, () => this.abort());
      }

      // Start turntable rotation if in turntable mode
      if (mode === 'turntable') {
        this.startTurntableRotation(opts, session);
      }

      // Wait for onstop to fire so the caller can await completion.
      await onstopComplete;
    } catch (err) {
      log.error(Modules.RECORDING, `Real-time recording setup failed: ${err}`);
      // Symmetric undo of every state mutation up to this point.
      // mediaRecorder may or may not have been constructed; clearing
      // the field is defensive. captureStream cleanup also runs even
      // if it was never assigned (helper handles null).
      this.cleanupCaptureStream();
      this.mediaRecorder = null;
      this.recordedChunks = [];
      this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
      this.animationController.removePerFrameCallback(this.turntableCallbackId);
      session.cleanupSyncListener();
      session.restoreAutoRotate();
      session.restoreRecordingState();
      session.isRecording = false;
      session.hideRecordingIndicator();
      showToast('Video recording failed to start');
      throw err;
    }
  }

  /** Synchronous stop request. The async onstop handler does the unwind. */
  abort(): void {
    if (this.durationTimer) {
      clearTimeout(this.durationTimer);
      this.durationTimer = null;
    }
    this.mediaRecorder?.stop();
  }

  dispose(): void {
    if (this.durationTimer) {
      clearTimeout(this.durationTimer);
      this.durationTimer = null;
    }
    // Per-frame callbacks may still be registered if dispose fires
    // mid-recording before onstop has unwound them.
    this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
    this.animationController.removePerFrameCallback(this.turntableCallbackId);
    // Defensive: stop captureStream tracks even if mediaRecorder.onstop
    // didn't fire (browser quirks, mid-init dispose).
    this.cleanupCaptureStream();
    this.mediaRecorder = null;
    this.recordedChunks = [];
  }

  // ── Internal helpers ───────────────────────────────────────────

  /**
   * Time-based turntable rotation for the real-time MediaRecorder path.
   * Quaternion-based orbit rotation (same math as auto-rotate), driven
   * by wall-clock time so the rotation completes after the correct
   * duration regardless of GPU FPS.
   *
   * Note: OfflineCaptureStrategy uses its own frame-index-based stepping
   * and does NOT use this method.
   */
  private startTurntableRotation(opts: RecordingOptions, session: RecordingSession): void {
    const controls = this.sceneManager.controls.getControls();
    if (!(controls instanceof LuxarOrbitControls)) {
      log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
      return;
    }

    // Pause auto-rotation so it doesn't compound with the turntable.
    session.pauseAutoRotate();

    const totalDuration = (360 / opts.turntableSpeed) * 1000;
    const startTime = Date.now();

    log.info(
      Modules.RECORDING,
      `Turntable started: speed=${opts.turntableSpeed}°/s, ` +
        `duration=${(totalDuration / 1000).toFixed(1)}s`
    );

    let turntableDone = false;
    let frameCount = 0;
    let lastProgress = 0;
    this.animationController.addPerFrameCallback(
      this.turntableCallbackId,
      () => {
        if (turntableDone) return;
        frameCount++;

        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / totalDuration, 1);
        const deltaAngle = (progress - lastProgress) * Math.PI * 2;
        lastProgress = progress;

        controls.applyOrbitRotation(deltaAngle);

        if (progress >= 1) {
          turntableDone = true;
          log.info(
            Modules.RECORDING,
            `Turntable completed: ${frameCount} rendered frames in ` +
              `${(elapsed / 1000).toFixed(1)}s (${(frameCount / (elapsed / 1000)).toFixed(1)} FPS)`
          );
          this.abort();
        }
      },
      { continuous: true }
    );
  }

  /**
   * Stop every track on the captureStream and drop the reference.
   * `mediaRecorder.stop()` does NOT stop the underlying tracks, so
   * without this call canvas-capture media tracks accumulate across
   * repeated recordings.
   *
   * Non-private so Panel can route `panel.cleanupCaptureStream()` through.
   */
  cleanupCaptureStream(): void {
    this.captureStream?.getTracks().forEach((track) => track.stop());
    this.captureStream = null;
  }

  // ── Test/orchestrator-visible: same reason — Panel proxies.
  startTurntableRotationForTests(opts: RecordingOptions, session: RecordingSession): void {
    this.startTurntableRotation(opts, session);
  }
}
