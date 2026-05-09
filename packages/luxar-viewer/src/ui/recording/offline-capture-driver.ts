/**
 * Phase 21B: per-mode driver interface for the offline-capture loop.
 *
 * The original `runOfflineCaptureLoop` (~432 LOC) mixed shared
 * scaffolding (state save/restore, overlay UI, animation pump,
 * progress display) with three different per-mode capture
 * pipelines (image-sequence ZIP, video container, EXR-sequence
 * ZIP). This interface lifts the per-mode pieces into named
 * driver classes so the loop body can focus on the shared
 * scaffolding.
 *
 * Each driver owns:
 *   - one-time setup (file picker, encoder construction)
 *   - per-frame capture (read pixels, encode, append)
 *   - finalize (close encoder/zip, download, toast)
 *
 * Drivers receive a `CaptureContext` (panel-injected dependencies
 * like sceneManager + helper callbacks) and a `CaptureProgress`
 * (UI hooks) so they stay decoupled from RecordingPanel internals.
 */

import type { SceneManager } from '../../scene/scene-manager';
import type { VideoCodecOption } from '../recording-panel';

/**
 * Dependencies a driver needs from the panel. Mostly wrapped panel
 * methods that the driver doesn't construct itself, plus the few
 * recording options that affect per-mode behaviour.
 */
export interface CaptureContext {
  sceneManager: SceneManager;
  fps: number;
  /** Read the live framebuffer into a 2D canvas. */
  renderFrameToCanvas: () => HTMLCanvasElement;
  /** Build a recording filename for a given extension. */
  generateFilename: (ext: string) => string;
  /** Build the bundled `encode_video.sh` script. */
  generateFfmpegScript: (fps: number, frames: number, ext: string) => string;
  /** Trigger a browser download. */
  downloadBlob: (blob: Blob, filename: string) => void;
  /** Compute the H.264/H.265/etc. bitrate for the canvas size. */
  computeVideoBitrate: (width: number, height: number) => number;
  /** Show a non-blocking toast notification. */
  showToast: (msg: string) => void;
  /** Logger for warnings + errors (panel pre-bound to its module tag). */
  logWarning: (msg: string) => void;
  logError: (msg: string) => void;
  /** Image quality (0–100) for jpeg/webp; ignored for png. */
  imageQuality: number;
  /** User-selected video codec. */
  videoCodec: VideoCodecOption;
  /** Browser environment for showSaveFilePicker (so tests can stub). */
  env: { showSaveFilePicker?: (opts: unknown) => Promise<FileSystemFileHandle> };
}

export interface CaptureProgress {
  /** Update the overlay's status label (e.g. "Packaging ZIP..."). */
  setLabel(text: string): void;
  /** Hand the just-captured canvas to the live preview. */
  setPreview(canvas: HTMLCanvasElement): void;
}

export interface OfflineCaptureDriver {
  /**
   * One-time setup. Return false to abort the capture (e.g. no codec
   * supports the request). Drivers that fail here must show the user
   * an explanatory toast before returning false.
   */
  setup(ctx: CaptureContext): Promise<boolean>;

  /**
   * Capture one frame. Called once per turntable frame after the
   * animation loop has rendered the rotated camera. May throw on
   * encoder errors — the loop tolerates up to MAX_CONSECUTIVE_ERRORS
   * before aborting.
   */
  captureFrame(
    ctx: CaptureContext,
    frameIndex: number,
    progress: CaptureProgress
  ): Promise<void>;

  /**
   * Finalize after the loop completes (or aborts via cancel). Called
   * with the actual count of captured frames (may be < totalFrames
   * if the user cancelled or errors triggered an abort).
   */
  finalize(
    ctx: CaptureContext,
    capturedFrames: number,
    progress: CaptureProgress
  ): Promise<void>;

  /**
   * Optional: signal whether the driver wants the loop to abort
   * early (e.g. ZIP disk write failed and continuing would be
   * pointless). Polled each iteration.
   */
  shouldAbort?(): boolean;
}
