/**
 * Real-time MediaRecorder video flow — extracted from `recording-panel.ts`.
 *
 * Handles `canvas.captureStream()` + `MediaRecorder` setup, the
 * `ondataavailable` / `onstop` plumbing, the duration timer, and the
 * symmetric error-path teardown. Offline / turntable / EXR branches
 * delegate to their respective modes; this function only owns the
 * real-time MediaRecorder path.
 */

import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { log, Modules } from '../../../utils/log';
import { showToast } from '../../toast';
import type { RecordingMode, RecordingOptions } from '../types';

export interface VideoModeCtx {
  isRecording: boolean;
  readonly disposed: boolean;
  readonly mode: RecordingMode;
  readonly options: RecordingOptions;
  readonly animationManager: DimensionAnimationManager | null;
  readonly sceneManager: SceneManager;
  readonly animationController: AnimationController;
  readonly keepAliveCallbackId: string;
  readonly turntableCallbackId: string;
  captureStream: MediaStream | null;
  mediaRecorder: MediaRecorder | null;
  recordedChunks: Blob[];
  recordingStartTime: number;
  durationTimer: ReturnType<typeof setTimeout> | null;
  startEXRSequenceRecording(): Promise<void>;
  runOfflineCaptureLoop(fmt: 'png' | 'webp' | 'jpeg' | 'mp4' | 'webm' | 'mkv'): Promise<void>;
  getSupportedMimeType(): string | null;
  showConfirmationDialog(): Promise<boolean>;
  hideAllPanels(): void;
  saveRecordingState(opts: {
    disableDPR?: boolean;
    lockResize?: boolean;
    scaleResolution?: { targetH: number };
  }): void;
  restoreRecordingState(): void;
  computeVideoBitrate(w: number, h: number): number;
  downloadBlob(blob: Blob, filename: string): void;
  generateFilename(ext: string): string;
  hideRecordingIndicator(): void;
  showRecordingIndicator(): void;
  cleanupCaptureStream(): void;
  cleanupSyncListener(): void;
  restoreAutoRotate(): void;
  startSliderSync(): void;
  startTurntableRotation(): void;
  stopVideoRecording(): void;
}

export async function startVideoRecording(ctx: VideoModeCtx): Promise<void> {
  if (ctx.isRecording) return;

  // Branch: frame-by-frame capture modes
  const fmt = ctx.options.outputFormat;
  const isImageFormat = fmt === 'png' || fmt === 'webp' || fmt === 'jpeg';
  const isTurntableSmooth = ctx.mode === 'turntable' && ctx.options.frameByFrame;

  // EXR → ZIP of EXR frames (always offline)
  if (fmt === 'exr') {
    return ctx.startEXRSequenceRecording();
  }
  // Turntable smooth mode — all formats use offline capture
  if (isTurntableSmooth) {
    if (isImageFormat) {
      return ctx.runOfflineCaptureLoop(fmt); // → ZIP of images
    }
    return ctx.runOfflineCaptureLoop(fmt as 'mp4' | 'webm' | 'mkv'); // → video file
  }
  // Video mode with image formats falls through to real-time MediaRecorder below

  const mimeType = ctx.getSupportedMimeType();
  if (!mimeType) {
    showToast('Video recording not supported in this browser');
    return;
  }

  const confirmed = await ctx.showConfirmationDialog();
  if (!confirmed || ctx.disposed) return;

  // Wrap the entire setup phase. Without this, a throw from
  // canvas.captureStream(), `new MediaRecorder(...)`, or
  // mediaRecorder.start() would leave panels hidden, DPR disabled,
  // resize locked, the keepalive callback registered, and no onstop
  // to unwind any of it.
  try {
    ctx.hideAllPanels();

    // Disable adaptive DPR during recording — resolution changes mid-capture cause
    // frozen frames, aspect ratio glitches, and partial rotations
    ctx.saveRecordingState({
      disableDPR: true,
      lockResize: true,
      scaleResolution:
        ctx.options.videoResolution > 0 ? { targetH: ctx.options.videoResolution } : undefined,
    });
    if (ctx.options.videoResolution > 0) {
      await new Promise((r) => requestAnimationFrame(r));
      if (ctx.disposed) {
        ctx.restoreRecordingState();
        return;
      }
    }

    const canvas = ctx.sceneManager.renderer.domElement;
    const videoBitsPerSecond = ctx.computeVideoBitrate(canvas.width, canvas.height);

    log.info(
      Modules.RECORDING,
      `Starting video recording (${mimeType}, ${ctx.options.videoFPS} FPS, ` +
        `${Math.round(videoBitsPerSecond / 1_000_000)}Mbps, ${canvas.width}x${canvas.height}, ` +
        `mode: ${ctx.mode})`
    );

    ctx.animationController.startAnimation();
    ctx.animationController.addPerFrameCallback(ctx.keepAliveCallbackId, () => {}, {
      continuous: true,
    });

    ctx.captureStream = canvas.captureStream(ctx.options.videoFPS);
    ctx.mediaRecorder = new MediaRecorder(ctx.captureStream, { mimeType, videoBitsPerSecond });
    ctx.recordedChunks = [];

    ctx.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        ctx.recordedChunks.push(event.data);
      }
    };

    ctx.mediaRecorder.onstop = () => {
      if (ctx.disposed) {
        ctx.recordedChunks = [];
        ctx.isRecording = false;
        ctx.hideRecordingIndicator();
        ctx.cleanupCaptureStream();
        return;
      }

      const blob = new Blob(ctx.recordedChunks, { type: mimeType });
      const totalElapsed = ((Date.now() - ctx.recordingStartTime) / 1000).toFixed(1);
      log.info(
        Modules.RECORDING,
        `Recording finalized: ${ctx.recordedChunks.length} chunks, ` +
          `${(blob.size / (1024 * 1024)).toFixed(1)} MB, ${totalElapsed}s elapsed`
      );
      ctx.downloadBlob(blob, ctx.generateFilename('webm'));
      ctx.recordedChunks = [];
      ctx.isRecording = false;
      ctx.hideRecordingIndicator();

      ctx.animationController.removePerFrameCallback(ctx.keepAliveCallbackId);
      ctx.animationController.removePerFrameCallback(ctx.turntableCallbackId);
      ctx.cleanupSyncListener();
      ctx.restoreAutoRotate();
      ctx.restoreRecordingState();
      ctx.cleanupCaptureStream();
      showToast('Video saved');
    };

    ctx.mediaRecorder.start(100);
    ctx.isRecording = true;
    ctx.recordingStartTime = Date.now();
    ctx.showRecordingIndicator();

    // Duration limit
    if (ctx.options.videoDurationLimit > 0) {
      ctx.durationTimer = setTimeout(() => {
        ctx.stopVideoRecording();
      }, ctx.options.videoDurationLimit * 1000);
    }

    // Start slider sync if enabled
    if (ctx.mode === 'video' && ctx.options.syncToSlider && ctx.animationManager) {
      ctx.startSliderSync();
    }

    // Start turntable rotation if in turntable mode
    if (ctx.mode === 'turntable') {
      ctx.startTurntableRotation();
    }
  } catch (err) {
    log.error(Modules.RECORDING, `Real-time recording setup failed: ${err}`);
    // Symmetric undo of every state mutation up to this point.
    // mediaRecorder may or may not have been constructed; clearing
    // the field is defensive. captureStream cleanup also runs even
    // if it was never assigned (helper handles null).
    ctx.cleanupCaptureStream();
    ctx.mediaRecorder = null;
    ctx.recordedChunks = [];
    ctx.animationController.removePerFrameCallback(ctx.keepAliveCallbackId);
    ctx.animationController.removePerFrameCallback(ctx.turntableCallbackId);
    ctx.cleanupSyncListener();
    ctx.restoreAutoRotate();
    ctx.restoreRecordingState();
    ctx.isRecording = false;
    ctx.hideRecordingIndicator();
    showToast('Video recording failed to start');
    throw err;
  }
}

