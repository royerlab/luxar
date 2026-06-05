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
import { ImageSequenceDriver } from '../../../../ui/recording-panel/drivers/image-sequence-driver';
import { ExrSequenceDriver } from '../../../../ui/recording-panel/drivers/exr-sequence-driver';
import type { CaptureContext } from '../../../../ui/recording-panel/drivers/offline-capture-driver';

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
    // [P2/C-C1] Pin the driver-supplied fallback download name, not just
    // "was called once": the in-memory ZIP path downloads with the
    // generateFilename('zip') the driver hands ZipSequenceCapture.
    expect((ctx.downloadBlob as ReturnType<typeof vi.fn>).mock.calls[0][1]).toBe('cap.zip');
    expect(ctx.showToast).toHaveBeenCalledWith(expect.stringContaining('PNG sequence saved'));
  });

  it('webp mode passes image/webp + imageQuality through to canvas.toBlob', async () => {
    // [P8/G6] Symmetry with the jpeg test — webp is a distinct MIME branch.
    const driver = new ImageSequenceDriver('webp');
    const ctx = makeCtx({ imageQuality: 0.9 });
    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, makeProgress());
    const canvas = (await (ctx.renderFrameToCanvas as ReturnType<typeof vi.fn>).mock.results[0]
      .value) as HTMLCanvasElement & { toBlob: ReturnType<typeof vi.fn> };
    expect(canvas.toBlob).toHaveBeenCalledWith(expect.any(Function), 'image/webp', 0.9);
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

  it('shouldAbort returns true when the underlying ZIP reports a disk failure', async () => {
    // [P2/W5] The true branch (zip.hasDiskFailed() === true) was never
    // exercised. Stub the real collaborator's method to drive the branch
    // without mocking the whole ZipSequenceCapture module.
    const driver = new ImageSequenceDriver('png');
    const ctx = makeCtx();
    await driver.setup(ctx);
    (driver as unknown as { zip: { hasDiskFailed: () => boolean } }).zip.hasDiskFailed = () => true;
    expect(driver.shouldAbort()).toBe(true);
  });

  it.each(['png', 'webp', 'jpeg'] as const)(
    'captureFrame throws a mode-specific error when canvas.toBlob returns null (%s)',
    async (mode) => {
      // [P5/C-C3] All three image modes share the null-blob guard; the
      // thrown message names the mode, so cover each.
      const fakeCanvas = {
        width: 100,
        height: 100,
        toBlob: vi.fn((cb: BlobCallback) => Promise.resolve().then(() => cb(null))),
      } as unknown as HTMLCanvasElement;
      const ctx = makeCtx({ renderFrameToCanvas: async () => fakeCanvas });
      const driver = new ImageSequenceDriver(mode);
      await driver.setup(ctx);

      await expect(driver.captureFrame(ctx, 0, makeProgress())).rejects.toThrow(
        `canvas.toBlob returned null for ${mode}`
      );
    }
  );

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

    // [P2/W2] Verify the underlying zip.abort() is actually invoked (once),
    // not merely that no download happened — a regression that dropped the
    // teardown would leak the partial ZIP yet still pass the download check.
    const zipAbortSpy = vi.spyOn(
      (driver as unknown as { zip: { abort: () => Promise<void> } }).zip,
      'abort'
    );

    await driver.abort?.(ctx, 'user-cancel');

    expect(zipAbortSpy).toHaveBeenCalledTimes(1);
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

  it('shouldAbort returns true when the underlying ZIP reports a disk failure', async () => {
    // [P8/G5] Symmetry with ImageSequenceDriver's shouldAbort-true test.
    const driver = new ExrSequenceDriver();
    const ctx = makeCtx();
    await driver.setup(ctx);
    (driver as unknown as { zip: { hasDiskFailed: () => boolean } }).zip.hasDiskFailed = () => true;
    expect(driver.shouldAbort()).toBe(true);
  });

  it('abort() invokes the onFinalize callback (mirrors the success path) and downloads nothing', async () => {
    // [P8/G7] exr-sequence-driver.ts:83 calls onFinalize?.() on abort so the
    // panel clears isEXRSequenceRecording; this was untested.
    const onFinalize = vi.fn();
    const driver = new ExrSequenceDriver(onFinalize);
    const ctx = makeCtx();
    await driver.setup(ctx);
    await driver.captureFrame(ctx, 0, makeProgress());

    await driver.abort?.(ctx, 'user-cancel');

    expect(onFinalize).toHaveBeenCalledTimes(1);
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });
});

describe('VideoModeDriver', () => {
  // mediabunny imports are stubbed via vi.doMock so this suite stays
  // headless. The video driver's setup() touches `canEncodeVideo` and
  // constructs `Output` / `BufferTarget` / `VideoSampleSource`.

  beforeEach(() => {
    vi.resetModules();
  });

  // Builds a mediabunny stub + returns the inner spies so a test can assert
  // on finalize / add / close. `opts` overrides the interesting bits.
  function makeMediabunny(
    opts: {
      canEncode?: boolean;
      finalize?: ReturnType<typeof vi.fn>;
      add?: ReturnType<typeof vi.fn>;
      bufferBytes?: number;
      nullBuffer?: boolean;
      close?: ReturnType<typeof vi.fn>;
    } = {}
  ) {
    const start = vi.fn().mockResolvedValue(undefined);
    const addVideoTrack = vi.fn();
    const finalize = opts.finalize ?? vi.fn().mockResolvedValue(undefined);
    const add = opts.add ?? vi.fn().mockResolvedValue(undefined);
    const close = opts.close ?? vi.fn();
    const VideoSample = vi.fn(() => ({ close }));
    const Output = vi.fn().mockImplementation(() => ({ start, addVideoTrack, finalize }));
    return {
      mb: {
        canEncodeVideo: vi.fn().mockResolvedValue(opts.canEncode ?? true),
        Output,
        WebMOutputFormat: vi.fn(),
        Mp4OutputFormat: vi.fn(),
        MkvOutputFormat: vi.fn(),
        BufferTarget: vi.fn(() => ({
          buffer: opts.nullBuffer ? undefined : new ArrayBuffer(opts.bufferBytes ?? 16),
        })),
        VideoSampleSource: vi.fn(() => ({ add })),
        VideoSample,
      },
      start,
      addVideoTrack,
      finalize,
      add,
      close,
      VideoSample,
    };
  }

  async function loadDriver(mb: Record<string, unknown>, mode: 'webm' | 'mp4' | 'mkv' = 'webm') {
    vi.doMock('mediabunny', () => mb);
    const { VideoModeDriver } =
      await import('../../../../ui/recording-panel/drivers/video-mode-driver');
    return new VideoModeDriver(mode);
  }

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
    const { VideoModeDriver } =
      await import('../../../../ui/recording-panel/drivers/video-mode-driver');
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
    const { VideoModeDriver } =
      await import('../../../../ui/recording-panel/drivers/video-mode-driver');
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
    const { VideoModeDriver } =
      await import('../../../../ui/recording-panel/drivers/video-mode-driver');
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
    const { VideoModeDriver } =
      await import('../../../../ui/recording-panel/drivers/video-mode-driver');
    const driver = new VideoModeDriver('webm');
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await driver.abort?.(ctx, 'user-cancel');

    expect(finalize).toHaveBeenCalledTimes(1);
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });

  // [P5/G1] captureFrame was entirely untested — the critical per-frame path
  // (VideoSample construction, source.add, resource close, preview update).
  it('captureFrame adds a VideoSample with frame-indexed timestamp/duration and closes it', async () => {
    const mb = makeMediabunny();
    const driver = await loadDriver(mb.mb);
    const ctx = makeCtx({ videoCodec: 'vp9', fps: 30 });
    const progress = makeProgress();
    await driver.setup(ctx);

    await driver.captureFrame(ctx, 5, progress);

    // timestamp = frameIndex * (1/fps); duration = 1/fps. Computed the same
    // way the source does, so the float comparison is exact.
    expect(mb.VideoSample).toHaveBeenCalledWith(expect.anything(), {
      timestamp: 5 * (1 / 30),
      duration: 1 / 30,
    });
    expect(mb.add).toHaveBeenCalledTimes(1);
    expect(mb.close).toHaveBeenCalledTimes(1);
    expect(progress.setPreview).toHaveBeenCalledTimes(1);
  });

  it('captureFrame closes the VideoSample even when source.add rejects (no resource leak)', async () => {
    // [P5/G1] The finally-block close() must run on the tolerated-failure path.
    const close = vi.fn();
    const mb = makeMediabunny({ add: vi.fn().mockRejectedValue(new Error('encoder busy')), close });
    const driver = await loadDriver(mb.mb);
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await expect(driver.captureFrame(ctx, 0, makeProgress())).rejects.toThrow('encoder busy');
    expect(close).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['webm', 'video/webm', 'cap.webm'],
    ['mp4', 'video/mp4', 'cap.mp4'],
    ['mkv', 'video/x-matroska', 'cap.mkv'],
  ] as const)(
    'finalize with frames downloads a %s blob (%s) named %s and toasts the frame count',
    async (mode, expectedMime, expectedName) => {
      // [P5/G2 + P11/M3] Success path + per-mode MIME mapping. A mutation to
      // any inline MIME string or the frame-count message is caught here.
      const mb = makeMediabunny();
      const driver = await loadDriver(mb.mb, mode);
      const ctx = makeCtx({ videoCodec: 'vp9' });
      await driver.setup(ctx);

      await driver.finalize(ctx, 12, makeProgress());

      expect(mb.finalize).toHaveBeenCalledTimes(1);
      expect(ctx.downloadBlob).toHaveBeenCalledTimes(1);
      const [blob, name] = (ctx.downloadBlob as ReturnType<typeof vi.fn>).mock.calls[0];
      expect((blob as Blob).type).toBe(expectedMime);
      expect(name).toBe(expectedName);
      expect(ctx.showToast).toHaveBeenCalledWith('Video saved (12 frames)');
    }
  );

  it('finalize surfaces an encoding-failed toast and downloads nothing when finalize throws', async () => {
    // [P5/G3] The catch path (video-mode-driver.ts:152-154) was untested.
    const mb = makeMediabunny({ finalize: vi.fn().mockRejectedValue(new Error('mux failed')) });
    const driver = await loadDriver(mb.mb);
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await driver.finalize(ctx, 5, makeProgress());

    expect(ctx.logError).toHaveBeenCalled();
    expect(ctx.showToast).toHaveBeenCalledWith('Video encoding failed');
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });

  it('finalize surfaces "no output" and downloads nothing if mediabunny hands back an empty buffer', async () => {
    // Regression for the external-boundary guard at video-mode-driver.ts:144-148.
    // A successful finalize() is expected to populate target.buffer, but we
    // don't control mediabunny — if it ever yields a null/empty buffer we must
    // NOT construct + "download" a 0-byte Blob.
    const mb = makeMediabunny({ nullBuffer: true });
    const driver = await loadDriver(mb.mb);
    const ctx = makeCtx({ videoCodec: 'vp9' });
    await driver.setup(ctx);

    await driver.finalize(ctx, 5, makeProgress());

    expect(mb.finalize).toHaveBeenCalledTimes(1);
    expect(ctx.showToast).toHaveBeenCalledWith('Video encoding produced no output');
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
  });

  it('finalize skips the artifact handoff when the signal aborts at the commit point', async () => {
    // [P5/G4] video-mode-driver.ts:141-143 — encoder is flushed but the
    // session was cancelled while finalizing, so nothing is delivered.
    const controller = new AbortController();
    const mb = makeMediabunny({
      finalize: vi.fn().mockImplementation(async () => {
        controller.abort();
      }),
    });
    const driver = await loadDriver(mb.mb);
    const ctx = makeCtx({ videoCodec: 'vp9', signal: controller.signal });
    await driver.setup(ctx);

    await driver.finalize(ctx, 5, makeProgress());

    expect(mb.finalize).toHaveBeenCalledTimes(1);
    expect(ctx.downloadBlob).not.toHaveBeenCalled();
    expect(ctx.showToast).not.toHaveBeenCalledWith(expect.stringContaining('Video saved'));
  });
});
