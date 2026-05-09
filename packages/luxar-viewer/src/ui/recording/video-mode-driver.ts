/**
 * Phase 21B: WebM / MP4 / MKV video driver.
 *
 * Uses mediabunny instead of MediaRecorder.captureStream because
 * WebGL canvases with `preserveDrawingBuffer: false` don't work
 * reliably with captureStream. Each frame is converted from a 2D
 * canvas into a `VideoSample` with explicit timestamp/duration and
 * pushed into the encoder. On finalize the buffer is materialised,
 * blobbed, and downloaded.
 */

import {
  Output,
  BufferTarget,
  VideoSampleSource,
  VideoSample,
  Mp4OutputFormat,
  WebMOutputFormat,
  MkvOutputFormat,
  canEncodeVideo,
} from 'mediabunny';
import type {
  CaptureContext,
  CaptureProgress,
  OfflineCaptureDriver,
} from './offline-capture-driver';
import { selectVideoCodec } from './video-codec-selection';

type VideoMode = 'webm' | 'mp4' | 'mkv';

export class VideoModeDriver implements OfflineCaptureDriver {
  private output: Output | null = null;
  private target: BufferTarget | null = null;
  private source: VideoSampleSource | null = null;
  private mime = 'video/webm';
  private ext: string;

  constructor(private readonly mode: VideoMode) {
    this.ext = mode;
  }

  async setup(ctx: CaptureContext): Promise<boolean> {
    const canvas = ctx.sceneManager.renderer.domElement;
    const bitrate = ctx.computeVideoBitrate(canvas.width, canvas.height);
    const encOpts = { width: canvas.width, height: canvas.height, bitrate };

    const selection = await selectVideoCodec({
      preferredCodec: ctx.videoCodec,
      containerMode: this.mode,
      encOpts,
      canEncodeVideo,
    });
    if (selection.codec === null) {
      ctx.showToast('No supported video codec at this resolution');
      return false;
    }
    if (!selection.isPreferred && selection.fallbackFrom) {
      ctx.logWarning(
        `${selection.fallbackFrom} not supported, falling back to ${selection.codec}`
      );
    }
    const codec = selection.codec;

    const format =
      this.mode === 'mp4'
        ? new Mp4OutputFormat()
        : this.mode === 'mkv'
          ? new MkvOutputFormat()
          : new WebMOutputFormat();
    this.mime =
      this.mode === 'mp4' ? 'video/mp4' : this.mode === 'mkv' ? 'video/x-matroska' : 'video/webm';
    this.target = new BufferTarget();
    this.source = new VideoSampleSource({
      codec,
      bitrate,
      latencyMode: 'quality',
    });
    this.output = new Output({ format, target: this.target });
    this.output.addVideoTrack(this.source, { frameRate: ctx.fps });
    await this.output.start();

    return true;
  }

  async captureFrame(
    ctx: CaptureContext,
    frameIndex: number,
    progress: CaptureProgress
  ): Promise<void> {
    if (!this.source) throw new Error('VideoModeDriver: captureFrame before setup');
    const canvas = ctx.renderFrameToCanvas();
    const frameDuration = 1 / ctx.fps;
    const sample = new VideoSample(canvas, {
      timestamp: frameIndex * frameDuration,
      duration: frameDuration,
    });
    await this.source.add(sample);
    sample.close();
    progress.setPreview(canvas);
  }

  async finalize(
    ctx: CaptureContext,
    capturedFrames: number,
    progress: CaptureProgress
  ): Promise<void> {
    if (!this.output) {
      ctx.showToast('No frames captured');
      return;
    }
    if (capturedFrames === 0) {
      ctx.showToast('No frames captured');
      return;
    }
    progress.setLabel('Finalizing video...');
    try {
      await this.output.finalize();
      const buffer = this.target?.buffer;
      if (!buffer) {
        ctx.showToast('Video encoding produced no output');
        return;
      }
      const blob = new Blob([buffer], { type: this.mime });
      ctx.downloadBlob(blob, ctx.generateFilename(this.ext));
      ctx.showToast(`Video saved (${capturedFrames} frames)`);
    } catch (err) {
      ctx.logError(`Video finalization failed: ${err}`);
      ctx.showToast('Video encoding failed');
    }
  }
}
