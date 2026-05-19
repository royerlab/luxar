/**
 * Tests for the per-mode capture drivers. Each test exercises a
 * driver's public protocol (setup, captureFrame, finalize, abort)
 * with stubbed dependencies, so the recording panel itself doesn't
 * need to be in scope.
 *
 * Video-driver tests stub `mediabunny` so the suite runs headlessly
 * without WebCodecs.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ImageSequenceDriver } from '../../../../ui/recording-panel/image-sequence-driver';
import { ExrSequenceDriver } from '../../../../ui/recording-panel/exr-sequence-driver';
import type { CaptureContext } from '../../../../ui/recording-panel/offline-capture-driver';

function makeProgress(): {
  setLabel: ReturnType<typeof vi.fn> & ((text: string) => void);
  setPreview: ReturnType<typeof vi.fn> & ((canvas: HTMLCanvasElement) => void);
} {
  return {
    setLabel: vi.fn() as ReturnType<typeof vi.fn> & ((text: string) => void),
    setPreview: vi.fn() as ReturnType<typeof vi.fn> & ((canvas: HTMLCanvasElement) => void),
  };
}

function makeCtx(overrides: Partial<CaptureContext> = {}): CaptureContext {
  // Build a minimal canvas stub that supports toBlob. jsdom's Blob
  // doesn't implement .arrayBuffer(), so we use a hand-rolled stub
  // that pretends to be a Blob and returns the underlying bytes.
  const stubBytes = new Uint8Array([0xff, 0x00, 0xff, 0x00]);
  const blobStub = {
    arrayBuffer: () => Promise.resolve(stubBytes.buffer),
    size: stubBytes.length,
    type: 'image/png',
  } as unknown as Blob;
  const fakeCanvas = {
    width: 1920,
    height: 1080,
    toBlob: vi.fn((cb: BlobCallback, _type?: string, _quality?: number) => {
      // Resolve asynchronously to mimic the browser behaviour.
      Promise.resolve().then(() => cb(blobStub));
    }),
  } as unknown as HTMLCanvasElement;

  return {
    sceneManager: {
      renderer: { domElement: { width: 1920, height: 1080 } },
      postProcessing: {
        captureHDRAsEXR: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
      },
    } as never,
    fps: 30,
    renderFrameToCanvas: vi.fn(async () => fakeCanvas),
    generateFilename: (ext: string) => `cap.${ext}`,
    generateFfmpegScript: (rate, frames, ext) =>
      `ffmpeg -framerate ${rate} -i frame_%06d.${ext} -frames:v ${frames} out.mp4`,
    downloadBlob: vi.fn(),
    computeVideoBitrate: () => 8_000_000,
    showToast: vi.fn(),
    logWarning: vi.fn(),
    logError: vi.fn(),
    imageQuality: 92,
    videoCodec: 'h264',
    env: {},
    signal: new AbortController().signal,
    ...overrides,
  };
}

describe('ImageSequenceDriver', () => {
  it('setup → captureFrame → finalize: writes one frame and downloads', async () => {
    const driver = new ImageSequenceDriver('png');
    const progress = makeProgress();
    const ctx = makeCtx();

    expect(await driver.setup(ctx)).toBe(true);
    await driver.captureFrame(ctx, 0, progress);
    await driver.finalize(ctx, 1, progress);

    expect(progress.setPreview).toHaveBeenCalledTimes(1);
    expect(progress.setLabel).toHaveBeenCalledWith('Packaging ZIP...');
    expect(ctx.downloadBlob).toHaveBeenCalledTimes(1);
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('PNG sequence saved'));
  });

  it('jpeg mode passes imageQuality through to canvas.toBlob', async () => {
    const driver = new ImageSequenceDriver('jpeg');
    const progress = makeProgress();
    const ctx = makeCtx({ imageQuality: 0.7 });

    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, progress);

    const canvas = (await (ctx.renderFrameToCanvas as ReturnType<typeof vi.fn>).mock.results[0]
      .value) as HTMLCanvasElement & { toBlob: ReturnType<typeof vi.fn> };
    expect(canvas.toBlob).toHaveBeenCalledWith(expect.any(Function), 'image/jpeg', 0.7);
  });

  it('shouldAbort is false by default and after a normal capture', async () => {
    const driver = new ImageSequenceDriver('png');
    const ctx = makeCtx();
    await driver.setup(ctx);
    expect(driver.shouldAbort?.()).toBe(false);
  });

  it('captureFrame throws if canvas.toBlob returns null', async () => {
    const fakeCanvas = {
      width: 100,
      height: 100,
      toBlob: vi.fn((cb: BlobCallback) => Promise.resolve().then(() => cb(null))),
    } as unknown as HTMLCanvasElement;
    const ctx = makeCtx({ renderFrameToCanvas: async () => fakeCanvas });
    const driver = new ImageSequenceDriver('png');
    await driver.setup(ctx);

    await expect(driver.captureFrame(ctx, 0, makeProgress())).rejects.toThrow(/null/);
  });

  it('finalize on zero frames toasts no-frames and skips download', async () => {
    const driver = new ImageSequenceDriver('png');
    const ctx = makeCtx();
    await driver.setup(ctx);
    await driver.finalize(ctx, 0, makeProgress());
    expect(ctx.showToast).toHaveBeenCalledWith('No frames captured');
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });

  it('abort() tears down the partial ZIP without download', async () => {
    const driver = new ImageSequenceDriver('png');
    const ctx = makeCtx();
    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, makeProgress());

    await driver.abort?.(ctx, 'user-cancel');

    expect(ctx.downloadBlob).not.toHaveBeenCalled();
    // After abort, finalize should be a no-op (zip ref cleared).
    await driver.finalize(ctx, 1, makeProgress());
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });
});

describe('ExrSequenceDriver', () => {
  it('captureFrame pulls EXR bytes from postProcessing.captureHDRAsEXR', async () => {
    const onFinalize = vi.fn();
    const driver = new ExrSequenceDriver(onFinalize);
    const progress = makeProgress();
    const ctx = makeCtx();

    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, progress);

    expect(ctx.sceneManager.postProcessing.captureHDRAsEXR).toHaveBeenCalledTimes(1);
    // EXR mode does not push pixels into the preview canvas:
    expect(progress.setPreview).not.toHaveBeenCalled();
  });

  it('finalize invokes the optional onFinalize callback before zip finalize', async () => {
    const onFinalize = vi.fn();
    const driver = new ExrSequenceDriver(onFinalize);
    const ctx = makeCtx();
    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, makeProgress());
    await driver.finalize(ctx, 1, makeProgress());
    expect(onFinalize).toHaveBeenCalledTimes(1);
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('EXR sequence saved'));
  });

  it('onFinalize is optional', async () => {
    const driver = new ExrSequenceDriver();
    const ctx = makeCtx();
    await driver.setup(ctx);
    await driver.finalize(ctx, 0, makeProgress());
    // Should not throw; toast goes through the zero-frames branch.
    expect(ctx.showToast).toHaveBeenCalledWith('No frames captured');
  });
});

describe('VideoModeDriver', () => {
  // mediabunny imports are stubbed via vi.doMock so this suite stays
  // headless. The video driver's setup() touches `canEncodeVideo` and
  // constructs `Output` / `BufferTarget` / `VideoSampleSource`.

  beforeEach(() => {
    vi.resetModules();
  });

  it('setup returns false and toasts when no codec is supported', async () => {
    vi.doMock('mediabunny', () => ({
      canEncodeVideo: vi.fn().mockResolvedValue(false),
      Output: vi.fn(),
      WebMOutputFormat: vi.fn(),
      Mp4OutputFormat: vi.fn(),
      MkvOutputFormat: vi.fn(),
      BufferTarget: vi.fn(),
      VideoSampleSource: vi.fn(),
      VideoSample: vi.fn(),
    }));
    const { VideoModeDriver } = await import('../../../../ui/recording-panel/video-mode-driver');
    const driver = new VideoModeDriver('webm');
    const ctx = makeCtx();
    expect(await driver.setup(ctx)).toBe(false);
    expect(ctx.showToast).toHaveBeenCalledWith('No supported video codec at this resolution');
  });

  it('setup constructs encoder when canEncodeVideo accepts the preferred codec', async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const addVideoTrack = vi.fn();
    const Output = vi.fn().mockImplementation(() => ({ start, addVideoTrack, finalize: vi.fn() }));
    vi.doMock('mediabunny', () => ({
      canEncodeVideo: vi.fn().mockResolvedValue(true),
      Output,
      WebMOutputFormat: vi.fn(),
      Mp4OutputFormat: vi.fn(),
      MkvOutputFormat: vi.fn(),
      BufferTarget: vi.fn(() => ({ buffer: new ArrayBuffer(8) })),
      VideoSampleSource: vi.fn(() => ({ add: vi.fn() })),
      VideoSample: vi.fn(),
    }));
    const { VideoModeDriver } = await import('../../../../ui/recording-panel/video-mode-driver');
    const driver = new VideoModeDriver('webm');
    const ctx = makeCtx({ videoCodec: 'vp9' });
    expect(await driver.setup(ctx)).toBe(true);
    expect(addVideoTrack).toHaveBeenCalledTimes(1);
    expect(start).toHaveBeenCalledTimes(1);
  });

  it('zero-frame finalize still calls output.finalize() to release encoder', async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const finalize = vi.fn().mockResolvedValue(undefined);
    const addVideoTrack = vi.fn();
    const Output = vi.fn().mockImplementation(() => ({ start, addVideoTrack, finalize }));
    vi.doMock('mediabunny', () => ({
      canEncodeVideo: vi.fn().mockResolvedValue(true),
      Output,
      WebMOutputFormat: vi.fn(),
      Mp4OutputFormat: vi.fn(),
      MkvOutputFormat: vi.fn(),
      BufferTarget: vi.fn(() => ({ buffer: new ArrayBuffer(0) })),
      VideoSampleSource: vi.fn(() => ({ add: vi.fn() })),
      VideoSample: vi.fn(),
    }));
    const { VideoModeDriver } = await import('../../../../ui/recording-panel/video-mode-driver');
    const driver = new VideoModeDriver('webm');
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await driver.finalize(ctx, 0, makeProgress());

    expect(finalize).toHaveBeenCalledTimes(1);
    expect(ctx.showToast).toHaveBeenCalledWith('No frames captured');
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });

  it('abort() releases encoder without delivering an artifact', async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const finalize = vi.fn().mockResolvedValue(undefined);
    const addVideoTrack = vi.fn();
    const Output = vi.fn().mockImplementation(() => ({ start, addVideoTrack, finalize }));
    vi.doMock('mediabunny', () => ({
      canEncodeVideo: vi.fn().mockResolvedValue(true),
      Output,
      WebMOutputFormat: vi.fn(),
      Mp4OutputFormat: vi.fn(),
      MkvOutputFormat: vi.fn(),
      BufferTarget: vi.fn(() => ({ buffer: new ArrayBuffer(8) })),
      VideoSampleSource: vi.fn(() => ({ add: vi.fn() })),
      VideoSample: vi.fn(),
    }));
    const { VideoModeDriver } = await import('../../../../ui/recording-panel/video-mode-driver');
    const driver = new VideoModeDriver('webm');
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await driver.abort?.(ctx, 'user-cancel');

    expect(finalize).toHaveBeenCalledTimes(1);
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });
});
