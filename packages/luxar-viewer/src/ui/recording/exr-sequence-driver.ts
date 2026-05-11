/**
 * EXR sequence driver.
 *
 * Captures each frame as an OpenEXR file directly from the HDR
 * postprocessing buffer (preserving 16-bit half-float precision)
 * and streams them into a ZIP archive. Bundles an `encode_video.sh`
 * with an ffmpeg invocation that converts the EXR sequence to
 * standard video formats.
 */

import type {
  CaptureContext,
  CaptureProgress,
  OfflineCaptureDriver,
} from './offline-capture-driver';
import { ZipSequenceCapture } from './zip-sequence-capture';

export class ExrSequenceDriver implements OfflineCaptureDriver {
  private zip: ZipSequenceCapture | null = null;

  /**
   * Optional callback invoked from finalize() so the panel can flip
   * its `isEXRSequenceRecording` flag back to false. Plumbed as a
   * callback rather than a direct field write so the driver stays
   * decoupled from the panel.
   */
  constructor(private readonly onFinalize?: () => void) {}

  async setup(ctx: CaptureContext): Promise<boolean> {
    this.zip = new ZipSequenceCapture(ctx.env, ctx.logError);
    await this.zip.setup({ suggestedName: ctx.generateFilename('zip') });
    return true;
  }

  async captureFrame(
    ctx: CaptureContext,
    _frameIndex: number,
    _progress: CaptureProgress
  ): Promise<void> {
    if (!this.zip) throw new Error('ExrSequenceDriver: captureFrame before setup');
    const exrData = await ctx.sceneManager.postProcessing.captureHDRAsEXR();
    // ZIP entry naming uses an internal success-counter inside
    // ZipSequenceCapture so tolerated frame failures don't create gaps.
    this.zip.addFrame(exrData, 'exr');
    // EXR mode does not push pixels back through a 2D canvas, so no
    // setPreview update — the panel keeps the previous frame visible.
  }

  async finalize(
    ctx: CaptureContext,
    capturedFrames: number,
    progress: CaptureProgress
  ): Promise<void> {
    this.onFinalize?.();
    if (!this.zip) return;
    await this.zip.finalize({
      capturedFrames,
      frameExt: 'exr',
      label: 'EXR',
      totalBytes: this.zip.getTotalBytes(),
      ffmpegScript: ctx.generateFfmpegScript(ctx.fps, capturedFrames, 'exr'),
      fallbackDownloadName: ctx.generateFilename('zip'),
      showToast: ctx.showToast,
      downloadBlob: ctx.downloadBlob,
      onPackagingStart: () => progress.setLabel('Packaging ZIP...'),
      signal: ctx.signal,
    });
  }

  shouldAbort(): boolean {
    return this.zip?.hasDiskFailed() ?? false;
  }

  /** Tear down a partial ZIP without delivering an artifact. */
  async abort(_ctx: CaptureContext, _reason: 'disposed' | 'user-cancel' | 'error'): Promise<void> {
    if (this.zip) {
      await this.zip.abort();
      this.zip = null;
    }
    // Also flip the EXR flag back so a subsequent recording session
    // doesn't observe stale state. This mirrors what finalize would
    // do on the success path.
    this.onFinalize?.();
  }
}
