/**
 * HDR 10-bit video encoder using mediabunny (WebCodecs-based muxer).
 *
 * Provides a streaming API: create encoder, feed frames one at a time,
 * finalize when done. This avoids accumulating all raw frames in memory.
 */

import {
  Output,
  Mp4OutputFormat,
  BufferTarget,
  VideoSampleSource,
  VideoSample,
  type VideoSampleInit,
  canEncodeVideo,
} from 'mediabunny';
import { rgbaFloatToI420P10 } from './hdr-color-conversion';
import { log, Modules } from './log';

/** Result of HDR video encoding */
export interface HDRVideoResult {
  blob: Blob;
  frameCount: number;
  duration: number;
}

/**
 * Check if the browser supports 10-bit HDR video encoding.
 * Tests AV1 first (better HDR support), then VP9 as fallback.
 */
export async function isHDRVideoSupported(): Promise<{
  supported: boolean;
  codec: 'av1' | 'vp9' | null;
}> {
  try {
    const testOpts = { width: 1920, height: 1080, bitrate: 10_000_000 };
    if (await canEncodeVideo('av1', testOpts)) {
      return { supported: true, codec: 'av1' };
    }
    if (await canEncodeVideo('vp9', testOpts)) {
      return { supported: true, codec: 'vp9' };
    }
  } catch {
    // WebCodecs not available
  }
  return { supported: false, codec: null };
}

/**
 * Streaming HDR video encoder. Feed frames one at a time to avoid
 * accumulating all raw float data in memory (~50 MB per 1080p frame).
 *
 * Usage:
 *   const encoder = await HDRVideoEncoder.create(width, height, fps);
 *   for each frame:
 *     encoder.addFrame(floatPixels);
 *   const result = await encoder.finalize();
 */
export class HDRVideoEncoder {
  private output: Output;
  private target: BufferTarget;
  private videoSource: VideoSampleSource;
  private encWidth: number;
  private encHeight: number;
  private origWidth: number;
  private origHeight: number;
  private fps: number;
  private frameDuration: number;
  private frameIndex = 0;

  private constructor(
    output: Output,
    target: BufferTarget,
    videoSource: VideoSampleSource,
    origWidth: number,
    origHeight: number,
    encWidth: number,
    encHeight: number,
    fps: number
  ) {
    this.output = output;
    this.target = target;
    this.videoSource = videoSource;
    this.origWidth = origWidth;
    this.origHeight = origHeight;
    this.encWidth = encWidth;
    this.encHeight = encHeight;
    this.fps = fps;
    this.frameDuration = 1 / fps;
  }

  /**
   * Create and initialize the encoder.
   * @throws if HDR encoding is not supported
   */
  static async create(width: number, height: number, fps: number): Promise<HDRVideoEncoder> {
    const { codec } = await isHDRVideoSupported();
    if (!codec) {
      throw new Error('10-bit HDR video encoding not supported in this browser');
    }

    // Ensure even dimensions (required by most video codecs)
    const encWidth = width & ~1;
    const encHeight = height & ~1;

    log.info(
      Modules.RECORDING,
      `HDR encoder created: ${codec}, ${encWidth}x${encHeight}, ${fps} FPS`
    );

    const target = new BufferTarget();
    const format = new Mp4OutputFormat();

    const videoSource = new VideoSampleSource({
      codec,
      bitrate: Math.round(encWidth * encHeight * fps * 0.15),
      latencyMode: 'quality',
    });

    const output = new Output({ format, target });
    output.addVideoTrack(videoSource, { frameRate: fps });
    await output.start();

    return new HDRVideoEncoder(
      output,
      target,
      videoSource,
      width,
      height,
      encWidth,
      encHeight,
      fps
    );
  }

  /**
   * Add a single frame to the video. The float pixel data can be
   * discarded by the caller immediately after this returns.
   */
  async addFrame(pixels: Float32Array): Promise<void> {
    // Crop to even dimensions if needed
    let rgba = pixels;
    if (this.origWidth !== this.encWidth || this.origHeight !== this.encHeight) {
      rgba = new Float32Array(this.encWidth * this.encHeight * 4);
      for (let row = 0; row < this.encHeight; row++) {
        const srcOffset = row * this.origWidth * 4;
        const dstOffset = row * this.encWidth * 4;
        rgba.set(pixels.subarray(srcOffset, srcOffset + this.encWidth * 4), dstOffset);
      }
    }

    const yuv = rgbaFloatToI420P10(rgba, this.encWidth, this.encHeight);

    // mediabunny's VideoSample takes the packed YUV via its
    // AllowSharedBufferSource overload. The BT.2020/PQ color-space members
    // (primaries='bt2020', transfer='pq', matrix='bt2020-ncl') are valid in
    // browsers and the WebCodecs spec, but are not yet in
    // @types/dom-webcodecs / lib.dom.d.ts unions; the directives below
    // flag themselves and can be deleted once the upstream types catch up.
    const init: VideoSampleInit = {
      format: 'I420P10',
      codedWidth: this.encWidth,
      codedHeight: this.encHeight,
      timestamp: this.frameIndex * this.frameDuration,
      duration: this.frameDuration,
      colorSpace: {
        // @ts-expect-error: HDR primaries not in lib.dom.d.ts yet
        primaries: 'bt2020',
        // @ts-expect-error: PQ transfer not in lib.dom.d.ts yet
        transfer: 'pq',
        // @ts-expect-error: BT.2020 matrix not in lib.dom.d.ts yet
        matrix: 'bt2020-ncl',
        fullRange: false,
      },
    };
    const sample = new VideoSample(
      yuv,
      init as Required<Pick<VideoSampleInit, 'format' | 'codedWidth' | 'codedHeight' | 'timestamp'>> &
        VideoSampleInit
    );

    await this.videoSource.add(sample);
    sample.close();
    this.frameIndex++;
  }

  /** Number of frames added so far */
  get frameCount(): number {
    return this.frameIndex;
  }

  /**
   * Finalize the video and return the encoded result.
   * The encoder cannot be used after this.
   */
  async finalize(): Promise<HDRVideoResult> {
    await this.output.finalize();

    const buffer = this.target.buffer;
    if (!buffer) {
      throw new Error('Encoding produced no output');
    }

    const blob = new Blob([buffer], { type: 'video/mp4' });
    const duration = this.frameIndex / this.fps;

    log.info(
      Modules.RECORDING,
      `HDR video encoded: ${(blob.size / (1024 * 1024)).toFixed(1)} MB, ` +
        `${this.frameIndex} frames, ${duration.toFixed(1)}s`
    );

    return { blob, frameCount: this.frameIndex, duration };
  }
}
