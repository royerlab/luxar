/**
 * Phase 21B: PNG / WebP / JPEG sequence driver.
 *
 * Reads each frame as a 2D canvas (via the panel-supplied
 * renderFrameToCanvas), encodes via the canvas-native toBlob, and
 * appends to a streaming ZIP. Quality is ignored for PNG (lossless)
 * and applied for WebP/JPEG.
 */

import type {
  CaptureContext,
  CaptureProgress,
  OfflineCaptureDriver,
} from './offline-capture-driver';
import { ZipSequenceCapture } from './zip-sequence-capture';

type ImageMode = 'png' | 'webp' | 'jpeg';

export class ImageSequenceDriver implements OfflineCaptureDriver {
  private zip: ZipSequenceCapture | null = null;

  constructor(private readonly mode: ImageMode) {}

  async setup(ctx: CaptureContext): Promise<boolean> {
    this.zip = new ZipSequenceCapture(ctx.env, ctx.logError);
    await this.zip.setup({ suggestedName: ctx.generateFilename('zip') });
    return true;
  }

  async captureFrame(
    ctx: CaptureContext,
    _frameIndex: number,
    progress: CaptureProgress
  ): Promise<void> {
    if (!this.zip) throw new Error('ImageSequenceDriver: captureFrame before setup');
    const canvas = ctx.renderFrameToCanvas();
    const mimeType = this.mode === 'jpeg' ? 'image/jpeg' : `image/${this.mode}`;
    const quality = this.mode === 'png' ? undefined : ctx.imageQuality;
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, mimeType, quality)
    );
    if (!blob) {
      throw new Error(`canvas.toBlob returned null for ${this.mode}`);
    }
    const buf = new Uint8Array(await blob.arrayBuffer());
    // ZIP entry naming uses an internal success-counter inside
    // ZipSequenceCapture so tolerated frame failures don't create gaps.
    this.zip.addFrame(buf, this.mode === 'jpeg' ? 'jpg' : this.mode);
    progress.setPreview(canvas);
  }

  async finalize(
    ctx: CaptureContext,
    capturedFrames: number,
    progress: CaptureProgress
  ): Promise<void> {
    if (!this.zip) return;
    const ext = this.mode === 'jpeg' ? 'jpg' : this.mode;
    await this.zip.finalize({
      capturedFrames,
      frameExt: ext,
      label: this.mode.toUpperCase(),
      totalBytes: this.zip.getTotalBytes(),
      ffmpegScript: ctx.generateFfmpegScript(ctx.fps, capturedFrames, ext),
      fallbackDownloadName: ctx.generateFilename('zip'),
      showToast: ctx.showToast,
      downloadBlob: ctx.downloadBlob,
      onPackagingStart: () => progress.setLabel('Packaging ZIP...'),
    });
  }

  shouldAbort(): boolean {
    return this.zip?.hasDiskFailed() ?? false;
  }

  /** r8 §A3: tear down a partial ZIP without delivering an artifact. */
  async abort(_ctx: CaptureContext, _reason: 'disposed' | 'user-cancel' | 'error'): Promise<void> {
    if (this.zip) {
      await this.zip.abort();
      this.zip = null;
    }
  }
}
