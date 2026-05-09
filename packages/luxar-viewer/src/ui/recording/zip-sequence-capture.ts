/**
 * Phase 21B: streaming-ZIP helper used by the image and EXR offline-
 * capture drivers.
 *
 * The image-sequence and EXR-sequence modes share the same ZIP
 * lifecycle: optionally pick a disk handle (File System Access API),
 * stream frames in via `ZipPassThrough` entries, append a final
 * `encode_video.sh` script with the chosen ffmpeg invocation, and
 * either close the disk handle or download an in-memory blob. This
 * class encapsulates that lifecycle so the per-mode drivers only
 * decide their entry name and frame data.
 *
 * The disk-streaming path uses `showSaveFilePicker` (Chrome/Edge);
 * other browsers fall back to in-memory chunks fed to `new Blob([…])`.
 * The split avoids the OOM trigger on long captures with large frames.
 */

import { Zip, ZipPassThrough } from 'fflate';

export interface ZipSetupOptions {
  /** Suggested filename for the OS save dialog. */
  suggestedName: string;
}

export interface ZipFinalizeOptions {
  /** Frame count actually captured (excludes failures). */
  capturedFrames: number;
  /** Frame extension (e.g. 'png', 'jpg', 'exr') for ffmpeg's input pattern. */
  frameExt: string;
  /** Human-readable label for toasts ("PNG", "EXR"). */
  label: string;
  /** Total bytes streamed (used in toasts when saving to disk). */
  totalBytes: number;
  /** ffmpeg script body to attach as `encode_video.sh`. */
  ffmpegScript: string;
  /** Suggested filename for the in-memory fallback download. */
  fallbackDownloadName: string;
  /** Show a toast (panel-supplied so this module stays UI-agnostic). */
  showToast: (msg: string) => void;
  /** Download a blob (panel-supplied). */
  downloadBlob: (blob: Blob, filename: string) => void;
  /** Optional UI hook called before the (potentially slow) finalize step. */
  onPackagingStart?: () => void;
}

/**
 * Caller-side dependency injection for the showSaveFilePicker flow.
 * Tests stub this; production passes the real `window`.
 */
export interface ZipSequenceEnv {
  showSaveFilePicker?: (opts: unknown) => Promise<FileSystemFileHandle>;
}

export class ZipSequenceCapture {
  private streamingZip: InstanceType<typeof Zip> | null = null;
  private chunks: Uint8Array[] = [];
  private totalBytes = 0;
  private diskFailed = false;
  private writable: FileSystemWritableFileStream | null = null;
  private setupCalled = false;
  /**
   * Output frame sequence counter. Increments after each successful
   * addFrame so output names are contiguous even when source frames
   * fail. See addFrame docstring for why this matters.
   */
  private frameSequenceNumber = 0;
  /**
   * Pending disk write promises (File System Access API path). The
   * ZIP chunk callback fires writes asynchronously; finalize() awaits
   * them before close() to avoid racing close-with-pending-writes
   * and to surface write failures the .catch handler trapped.
   */
  private pendingWrites: Promise<void>[] = [];

  constructor(
    private readonly env: ZipSequenceEnv,
    private readonly logError: (msg: string) => void
  ) {}

  /**
   * Start the ZIP stream. Tries to obtain a disk handle from
   * showSaveFilePicker; on failure or unavailability, falls back to
   * in-memory accumulation. Always succeeds — disk failures are
   * folded into in-memory mode, and a user-cancelled picker also
   * falls back rather than aborting the whole capture.
   */
  async setup(opts: ZipSetupOptions): Promise<void> {
    if (this.setupCalled) {
      throw new Error('ZipSequenceCapture.setup() called twice');
    }
    this.setupCalled = true;

    if (typeof this.env.showSaveFilePicker === 'function') {
      try {
        const handle = await this.env.showSaveFilePicker({
          suggestedName: opts.suggestedName,
          types: [
            { description: 'ZIP archive', accept: { 'application/zip': ['.zip'] } },
          ],
        });
        this.writable = await handle.createWritable();
      } catch {
        // User cancelled or API unavailable — fall back to in-memory.
        this.writable = null;
      }
    }

    this.streamingZip = new Zip((err, chunk, _final) => {
      if (err) {
        if (!this.diskFailed) {
          this.diskFailed = true;
          this.logError(`ZIP stream error: ${err}`);
        }
        return;
      }
      if (!chunk) return;
      if (this.writable && !this.diskFailed) {
        // Track each write promise so finalize() can await them all
        // before close() — otherwise close races outstanding writes
        // and any rejection arrives after we've already toasted
        // success.
        const writeP = this.writable.write(chunk as BlobPart).catch((writeErr) => {
          if (!this.diskFailed) {
            this.diskFailed = true;
            this.logError(
              `Disk write failed: ${writeErr}. ` +
                'Free browser storage or use video format instead of image sequence.'
            );
          }
        });
        this.pendingWrites.push(writeP);
      } else if (!this.diskFailed) {
        this.chunks.push(chunk);
      }
      this.totalBytes += chunk.length;
    });
  }

  /**
   * Append a single frame to the ZIP. Frame names are zero-padded to
   * 6 digits and use a private sequence counter that increments only
   * on successful adds, so the output sequence is always contiguous
   * (frame_000000, frame_000001, …) regardless of which source frame
   * indices the caller skipped or had errors on. The bundled ffmpeg
   * script in `encode_video.sh` assumes a contiguous sequence.
   */
  addFrame(data: Uint8Array, ext: string): void {
    if (!this.streamingZip) {
      throw new Error('ZipSequenceCapture: addFrame() before setup()');
    }
    const padded = String(this.frameSequenceNumber).padStart(6, '0');
    const entry = new ZipPassThrough(`frame_${padded}.${ext}`);
    this.streamingZip.add(entry);
    entry.push(data, true);
    this.frameSequenceNumber++;
  }

  /** Whether a disk write has failed (caller may choose to abort the loop). */
  hasDiskFailed(): boolean {
    return this.diskFailed;
  }

  /** Total bytes streamed so far (used in toasts). */
  getTotalBytes(): number {
    return this.totalBytes;
  }

  /**
   * Finalize the ZIP stream. Behaviour:
   *   - on disk-failed: end the stream, abort the writable, toast the error.
   *   - on zero captured frames: end the stream, abort the writable, toast.
   *   - otherwise: append the ffmpeg script, end the stream, await all
   *     pending writes, then close the writable OR build an in-memory
   *     blob and download it.
   *
   * Writes are awaited via Promise.allSettled so a late rejection
   * sets `diskFailed` and we can take the failure-toast path even if
   * the `.catch` handler hadn't yet flipped the flag at the time
   * end() returned.
   */
  async finalize(opts: ZipFinalizeOptions): Promise<void> {
    if (!this.streamingZip) {
      throw new Error('ZipSequenceCapture: finalize() before setup()');
    }

    // Helper: await all pending disk writes (if any), tolerating
    // rejections — the .catch handler on each write already records
    // the failure into this.diskFailed.
    const awaitPendingWrites = async (): Promise<void> => {
      if (this.pendingWrites.length === 0) return;
      await Promise.allSettled(this.pendingWrites);
      this.pendingWrites = [];
    };

    if (this.diskFailed) {
      this.streamingZip.end();
      await awaitPendingWrites();
      if (this.writable) {
        try {
          await this.writable.abort();
        } catch {
          /* already closed/aborted */
        }
      }
      opts.showToast(
        'Recording failed: disk storage quota exceeded. ' +
          'Free browser storage, reduce duration/resolution, or use video format.'
      );
      return;
    }

    if (opts.capturedFrames > 0) {
      opts.onPackagingStart?.();
      // Yield once so any UI label change paints before the (potentially
      // slow) script-encoding + stream-end.
      await new Promise((r) => requestAnimationFrame(r));
      const encoder = new TextEncoder();
      const scriptEntry = new ZipPassThrough('encode_video.sh');
      this.streamingZip.add(scriptEntry);
      scriptEntry.push(encoder.encode(opts.ffmpegScript), true);
      this.streamingZip.end();
      // Await writes triggered by end() before close(). A late
      // rejection here flips diskFailed and we route to the
      // disk-failed toast.
      await awaitPendingWrites();
      if (this.diskFailed) {
        if (this.writable) {
          try {
            await this.writable.abort();
          } catch {
            /* already aborted */
          }
        }
        opts.showToast(
          'Recording failed mid-finalize: disk write rejected after frames. ' +
            'Output may be truncated.'
        );
        return;
      }
      if (this.writable) {
        await this.writable.close();
        opts.showToast(
          `${opts.label} sequence saved to disk (${opts.capturedFrames} frames, ` +
            `${(opts.totalBytes / (1024 * 1024)).toFixed(1)} MB)`
        );
      } else {
        const blob = new Blob(this.chunks as BlobPart[], { type: 'application/zip' });
        opts.downloadBlob(blob, opts.fallbackDownloadName);
        opts.showToast(`${opts.label} sequence saved (${opts.capturedFrames} frames)`);
      }
    } else {
      this.streamingZip.end();
      await awaitPendingWrites();
      if (this.writable) {
        try {
          await this.writable.abort();
        } catch {
          /* already closed/aborted */
        }
      }
      opts.showToast('No frames captured');
    }
  }

  /**
   * r8 §A3: tear down a partial ZIP without delivering an artifact.
   * Called by the panel when the offline session is aborted
   * (disposed / cancelled / capture errored). Idempotent: safe to
   * call after finalize, after abort, or on a never-set-up instance.
   */
  async abort(): Promise<void> {
    if (!this.streamingZip) return;
    try {
      this.streamingZip.end();
    } catch {
      /* fflate may throw if already ended */
    }
    if (this.pendingWrites.length > 0) {
      await Promise.allSettled(this.pendingWrites);
      this.pendingWrites = [];
    }
    if (this.writable) {
      try {
        await this.writable.abort();
      } catch {
        /* already aborted/closed */
      }
      this.writable = null;
    }
    // Drop in-memory chunks so a follow-up gc pass can release them.
    this.chunks = [];
  }
}
