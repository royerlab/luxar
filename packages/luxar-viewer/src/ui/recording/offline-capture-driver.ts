/**
 * Per-mode driver interface for the offline-capture loop. The loop
 * in `runOfflineCaptureLoop` owns the shared scaffolding (state
 * save/restore, overlay UI, animation pump, progress display); each
 * driver owns its mode's pipeline:
 *   - one-time setup (file picker, encoder construction)
 *   - per-frame capture (read pixels, encode, append)
 *   - finalize (close encoder/zip, download, toast)
 *   - optional abort (release partial resources without delivery)
 *
 * Drivers receive a `CaptureContext` (panel-injected dependencies
 * like sceneManager + helper callbacks) and a `CaptureProgress`
 * (UI hooks) so they stay decoupled from RecordingPanel internals.
 */

import type { SceneManager } from '../../scene/scene-manager';
import type { VideoCodecOption } from './types';

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
  /** Image quality in [0, 1] for jpeg/webp (matches canvas.toBlob); ignored for png. */
  imageQuality: number;
  /** User-selected video codec. */
  videoCodec: VideoCodecOption;
  /** Browser environment for showSaveFilePicker (so tests can stub). */
  env: { showSaveFilePicker?: (opts: unknown) => Promise<FileSystemFileHandle> };
  /**
   * AbortSignal for the offline-capture session. Drivers MAY check
   * `signal.aborted` in long-running setup/finalize work to
   * short-circuit cleanly when the panel is disposed mid-capture or
   * the user cancels. The panel also checks the signal between
   * frames so abort during an `await` propagates within ~1 frame.
   */
  signal: AbortSignal;
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
   *
   * `frameIndex` is the OUTPUT sequence index (number of successful
   * captures so far), not the source loop index. With tolerated
   * frame failures the source loop's index advances while the output
   * index stays contiguous — drivers that use this index for
   * timestamps (VideoModeDriver) get gap-free output timing.
   * Drivers that name files use ZipSequenceCapture's own success
   * counter and can ignore this parameter.
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

  /**
   * Optional partial-resource cleanup hook. Called from the panel's
   * finally block when:
   *   - setup completed but capture/finalize threw, OR
   *   - the offline session was aborted (dispose / user cancel).
   * Drivers should release encoder/zip/file resources without
   * downloading or toasting "saved". Distinct from `finalize`,
   * which is the success path that delivers the artifact.
   */
  abort?(ctx: CaptureContext, reason: 'disposed' | 'user-cancel' | 'error'): Promise<void>;
}
