/**
 * Unit tests for the ZipSequenceCapture helper extracted in Phase 21B
 * (deep). The previous inline implementation lived inside a 432-LOC
 * runOfflineCaptureLoop and couldn't be exercised without spinning up
 * the whole panel.
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

    z.addFrame(new Uint8Array([1, 2, 3]), 'png', 0);
    z.addFrame(new Uint8Array([4, 5, 6]), 'png', 1);

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
    z.addFrame(new Uint8Array([0xff, 0xee]), 'png', 0);

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
    z.addFrame(new Uint8Array([1]), 'png', 0);

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
    z.addFrame(new Uint8Array([1, 2, 3]), 'png', 0);

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
    expect(() => z.addFrame(new Uint8Array([1]), 'png', 0)).toThrow(/before setup/);
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
    z.addFrame(new Uint8Array([1]), 'png', 0);

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
});
