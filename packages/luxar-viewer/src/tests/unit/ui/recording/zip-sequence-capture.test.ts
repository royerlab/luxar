/**
 * Unit tests for the ZipSequenceCapture helper. The helper owns the
 * streaming-ZIP lifecycle for image and EXR offline-capture drivers
 * — disk-streaming via showSaveFilePicker, in-memory fallback, and
 * abort/finalize ordering — and is testable in isolation here.
 */

import { describe, it, expect, vi } from 'vitest';
import { ZipSequenceCapture } from '../../../../ui/recording/zip-sequence-capture';

function makeWritable(): {
  write: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  abort: ReturnType<typeof vi.fn>;
  handle: { createWritable: ReturnType<typeof vi.fn> };
  picker: ReturnType<typeof vi.fn>;
} {
  const write = vi.fn().mockResolvedValue(undefined);
  const close = vi.fn().mockResolvedValue(undefined);
  const abort = vi.fn().mockResolvedValue(undefined);
  const stream = { write, close, abort } as unknown as FileSystemWritableFileStream;
  const handle = { createWritable: vi.fn().mockResolvedValue(stream) };
  const picker = vi.fn().mockResolvedValue(handle);
  return { write, close, abort, handle, picker };
}

describe('ZipSequenceCapture', () => {
  it('without showSaveFilePicker, falls back to in-memory chunks and downloads on finalize', async () => {
    const logError = vi.fn();
    const showToast = vi.fn();
    const downloadBlob = vi.fn();
    const z = new ZipSequenceCapture({}, logError);
    await z.setup({ suggestedName: 'cap.zip' });

    z.addFrame(new Uint8Array([1, 2, 3]), 'png');
    z.addFrame(new Uint8Array([4, 5, 6]), 'png');

    await z.finalize({
      capturedFrames: 2,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: 'ffmpeg ...',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob,
    });

    expect(downloadBlob).toHaveBeenCalledTimes(1);
    const [blob, name] = downloadBlob.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(name).toBe('cap.zip');
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('PNG sequence saved'));
    expect(logError).not.toHaveBeenCalled();
  });

  it('with showSaveFilePicker available, writes to disk handle and closes on finalize', async () => {
    const env = makeWritable();
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: env.picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      vi.fn()
    );
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([0xff, 0xee]), 'png');

    const showToast = vi.fn();
    const downloadBlob = vi.fn();
    await z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: 'ffmpeg ...',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob,
    });

    expect(env.picker).toHaveBeenCalledTimes(1);
    expect(env.handle.createWritable).toHaveBeenCalledTimes(1);
    expect(env.write).toHaveBeenCalled();
    expect(env.close).toHaveBeenCalledTimes(1);
    expect(downloadBlob).not.toHaveBeenCalled(); // Disk path: no in-memory blob
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('saved to disk'));
  });

  it('user-cancelled file picker falls back to in-memory mode', async () => {
    const picker = vi.fn().mockRejectedValue(new DOMException('User aborted', 'AbortError'));
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      vi.fn()
    );
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([1]), 'png');

    const showToast = vi.fn();
    const downloadBlob = vi.fn();
    await z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: 'ffmpeg ...',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob,
    });

    expect(picker).toHaveBeenCalled();
    expect(downloadBlob).toHaveBeenCalledTimes(1); // Fell back to download path
  });

  it('zero captured frames toasts "No frames captured" and aborts disk handle', async () => {
    const env = makeWritable();
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: env.picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      vi.fn()
    );
    await z.setup({ suggestedName: 'cap.zip' });

    const showToast = vi.fn();
    const downloadBlob = vi.fn();
    await z.finalize({
      capturedFrames: 0,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: 0,
      ffmpegScript: 'ffmpeg ...',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob,
    });

    expect(showToast).toHaveBeenCalledWith('No frames captured');
    expect(env.abort).toHaveBeenCalledTimes(1);
    expect(env.close).not.toHaveBeenCalled();
    expect(downloadBlob).not.toHaveBeenCalled();
  });

  it('disk write failure trips hasDiskFailed() and the finalize toast', async () => {
    const writeError = vi.fn().mockRejectedValue(new Error('quota exceeded'));
    const stream = { write: writeError, close: vi.fn(), abort: vi.fn() };
    const handle = { createWritable: vi.fn().mockResolvedValue(stream) };
    const picker = vi.fn().mockResolvedValue(handle);

    const logError = vi.fn();
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      logError
    );
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([1, 2, 3]), 'png');

    // Yield once to flush the rejected write promise's catch handler.
    await new Promise((r) => setTimeout(r, 0));
    expect(z.hasDiskFailed()).toBe(true);

    const showToast = vi.fn();
    await z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: 'ffmpeg ...',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob: vi.fn(),
    });
    expect(showToast).toHaveBeenCalledWith(
      expect.stringContaining('disk storage quota exceeded')
    );
    expect(logError).toHaveBeenCalled();
  });

  it('addFrame before setup throws', () => {
    const z = new ZipSequenceCapture({}, vi.fn());
    expect(() => z.addFrame(new Uint8Array([1]), 'png')).toThrow(/before setup/);
  });

  it('finalize before setup throws', async () => {
    const z = new ZipSequenceCapture({}, vi.fn());
    await expect(
      z.finalize({
        capturedFrames: 0,
        frameExt: 'png',
        label: 'PNG',
        totalBytes: 0,
        ffmpegScript: '',
        fallbackDownloadName: 'x.zip',
        showToast: vi.fn(),
        downloadBlob: vi.fn(),
      })
    ).rejects.toThrow(/before setup/);
  });

  it('setup called twice throws', async () => {
    const z = new ZipSequenceCapture({}, vi.fn());
    await z.setup({ suggestedName: 'a.zip' });
    await expect(z.setup({ suggestedName: 'b.zip' })).rejects.toThrow(/twice/);
  });

  it('onPackagingStart fires before slow finalize work when frames are captured', async () => {
    const z = new ZipSequenceCapture({}, vi.fn());
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([1]), 'png');

    const onPackagingStart = vi.fn();
    await z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: '',
      fallbackDownloadName: 'cap.zip',
      showToast: vi.fn(),
      downloadBlob: vi.fn(),
      onPackagingStart,
    });
    expect(onPackagingStart).toHaveBeenCalledTimes(1);
  });

  it('frame-name regression: tolerated source-frame failures yield contiguous output names', async () => {
    // Simulates the offline-capture loop's behaviour: source frames
    // 0, 1, 2, 3 are attempted; frame 1 fails (no addFrame call);
    // remaining frames go through. Output names must be 000000,
    // 000001, 000002 (contiguous), not 000000, 000002, 000003.
    const env = makeWritable();
    const writtenChunks: Uint8Array[] = [];
    env.write.mockImplementation(async (chunk: Uint8Array) => {
      writtenChunks.push(chunk);
    });
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: env.picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      vi.fn()
    );
    await z.setup({ suggestedName: 'cap.zip' });

    z.addFrame(new Uint8Array([0]), 'png'); // source frame 0
    // (source frame 1 failed — addFrame not called)
    z.addFrame(new Uint8Array([2]), 'png'); // source frame 2
    z.addFrame(new Uint8Array([3]), 'png'); // source frame 3

    await z.finalize({
      capturedFrames: 3,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: '',
      fallbackDownloadName: 'cap.zip',
      showToast: vi.fn(),
      downloadBlob: vi.fn(),
    });

    // ZIP local-file-headers contain the filename right after the
    // 30-byte header — visible as ASCII in the chunk stream.
    const total = writtenChunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Uint8Array(total);
    let off = 0;
    for (const c of writtenChunks) {
      merged.set(c, off);
      off += c.length;
    }
    const text = new TextDecoder().decode(merged);
    expect(text).toContain('frame_000000.png');
    expect(text).toContain('frame_000001.png'); // CONTIGUOUS — was previously frame_000002
    expect(text).toContain('frame_000002.png');
    expect(text).not.toContain('frame_000003.png');
  });

  it('disk writes are awaited before close', async () => {
    const writeResolvers: Array<() => void> = [];
    const env = makeWritable();
    env.write.mockImplementation(
      () => new Promise<void>((resolve) => writeResolvers.push(resolve))
    );

    const z = new ZipSequenceCapture(
      { showSaveFilePicker: env.picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      vi.fn()
    );
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([1]), 'png');

    const finalizeP = z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: '',
      fallbackDownloadName: 'cap.zip',
      showToast: vi.fn(),
      downloadBlob: vi.fn(),
    });

    // Sleep long enough for the requestAnimationFrame inside finalize
    // to fire and for the subsequent end()/awaitPendingWrites step to
    // start awaiting. close() must NOT have been called yet.
    await new Promise((r) => setTimeout(r, 50));
    expect(env.close).not.toHaveBeenCalled();

    writeResolvers.forEach((r) => r());
    await finalizeP;
    expect(env.close).toHaveBeenCalledTimes(1);
  });

  it('late write rejection is observed via diskFailed and toasts truncation warning', async () => {
    // Collect every (resolve, reject) pair so we can fail one and let
    // the others succeed. The write-callback path in fflate fires a
    // chunk per addFrame plus more on end() (central directory).
    const writeControls: Array<{
      resolve: () => void;
      reject: (err: Error) => void;
    }> = [];
    const env = makeWritable();
    env.write.mockImplementation(
      () =>
        new Promise<void>((resolve, reject) => {
          writeControls.push({ resolve, reject });
        })
    );

    const logError = vi.fn();
    const z = new ZipSequenceCapture(
      { showSaveFilePicker: env.picker as unknown as (opts: unknown) => Promise<FileSystemFileHandle> },
      logError
    );
    await z.setup({ suggestedName: 'cap.zip' });
    z.addFrame(new Uint8Array([1]), 'png');

    const showToast = vi.fn();
    const finalizeP = z.finalize({
      capturedFrames: 1,
      frameExt: 'png',
      label: 'PNG',
      totalBytes: z.getTotalBytes(),
      ffmpegScript: '',
      fallbackDownloadName: 'cap.zip',
      showToast,
      downloadBlob: vi.fn(),
    });

    // Sleep so finalize advances past raf and reaches awaitPendingWrites.
    await new Promise((r) => setTimeout(r, 50));
    expect(writeControls.length).toBeGreaterThan(0);
    // First write rejects (simulates disk quota exceeded mid-stream);
    // the rest resolve normally so awaitPendingWrites can complete.
    writeControls[0].reject(new Error('quota exceeded'));
    for (let i = 1; i < writeControls.length; i++) writeControls[i].resolve();
    await finalizeP;

    // diskFailed routes to abort + truncation toast, not close.
    expect(env.close).not.toHaveBeenCalled();
    expect(env.abort).toHaveBeenCalled();
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('truncated'));
    expect(logError).toHaveBeenCalled();
  });
});
