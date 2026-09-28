/**
 * `?renderAudit` (debug only): prove the render-on-change scheduler skips
 * only frames that would have drawn the same pixels.
 *
 * Under the audit the loop renders EVERY tick. On a tick the scheduler would
 * have skipped it reads the frame back (a 64x64 downsample, drawn from the
 * canvas in the same task as the render so the drawing buffer is still
 * valid) and compares it against the readback of the last frame the
 * scheduler really did render. A difference is a state change nobody
 * reported — a "missed dirty": the counter `render.missedDirty` goes up and
 * the last render requests are logged so the silent source can be found.
 * The mismatching frame then becomes the reference, so one missed change
 * counts once, not on every tick after it.
 *
 * A pixel counts as different when any channel moved by more than
 * {@link CHANNEL_TOLERANCE} (of 255): an identical state renders identical
 * pixels, and the tolerance only absorbs readback rounding, never a real
 * change a user could see.
 *
 * @module scene/animation/render-audit
 */

import { perfCounters } from '../../profiling/perf-counters';
import { log, Modules } from '../../utils/log';

/** Side of the square readback. */
export const AUDIT_SIZE = 64;

/** Per-channel difference (0-255) above which a pixel differs. */
export const CHANNEL_TOLERANCE = 2;

/** Render requests kept for the missed-dirty log. */
const REASON_RING_SIZE = 16;

const S_MISSED_DIRTY = perfCounters.slot('render.missedDirty');
const S_AUDIT_COMPARES = perfCounters.slot('render.auditCompares');

/** Reads the canvas the loop renders into as `AUDIT_SIZE`² RGBA bytes. */
export type AuditReadback = () => Uint8ClampedArray | null;

/**
 * The default readback: `drawImage` of the render canvas into a small 2D
 * canvas. Returns null (the audit then compares nothing) when no 2D context
 * is available, e.g. under jsdom.
 */
export function canvasReadback(getCanvas: () => CanvasImageSource | null): AuditReadback {
  let ctx: CanvasRenderingContext2D | null | undefined;
  return () => {
    const source = getCanvas();
    if (!source) return null;
    if (ctx === undefined) {
      const scratch = document.createElement('canvas');
      scratch.width = AUDIT_SIZE;
      scratch.height = AUDIT_SIZE;
      ctx = scratch.getContext('2d', { willReadFrequently: true });
    }
    if (!ctx) return null;
    ctx.clearRect(0, 0, AUDIT_SIZE, AUDIT_SIZE);
    ctx.drawImage(source, 0, 0, AUDIT_SIZE, AUDIT_SIZE);
    return ctx.getImageData(0, 0, AUDIT_SIZE, AUDIT_SIZE).data;
  };
}

/** Number of pixels whose channels differ by more than {@link CHANNEL_TOLERANCE}. */
export function countDifferingPixels(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  if (a.length !== b.length) return Math.max(a.length, b.length) / 4;
  let differing = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (
      Math.abs(a[i] - b[i]) > CHANNEL_TOLERANCE ||
      Math.abs(a[i + 1] - b[i + 1]) > CHANNEL_TOLERANCE ||
      Math.abs(a[i + 2] - b[i + 2]) > CHANNEL_TOLERANCE ||
      Math.abs(a[i + 3] - b[i + 3]) > CHANNEL_TOLERANCE
    ) {
      differing++;
    }
  }
  return differing;
}

export class RenderAudit {
  private reference: Uint8ClampedArray | null = null;
  private readonly reasons: string[] = [];

  constructor(private readonly readback: AuditReadback) {}

  /** Remember a render request (the log's evidence for a missed dirty). */
  noteRequest(entry: string): void {
    this.reasons.push(entry);
    if (this.reasons.length > REASON_RING_SIZE) this.reasons.shift();
  }

  /** Call right after a frame the scheduler really rendered. */
  captureReference(): void {
    this.reference = this.readback();
  }

  /**
   * Call right after an audit render of a tick the scheduler would have
   * skipped. Returns true (and counts `render.missedDirty`) when its pixels
   * differ from the last real frame's.
   */
  checkSkippedFrame(tick: number): boolean {
    const frame = this.readback();
    if (!frame) return false;
    perfCounters.add(S_AUDIT_COMPARES);
    const reference = this.reference;
    this.reference = frame;
    if (!reference) return false;
    const differing = countDifferingPixels(reference, frame);
    if (differing === 0) return false;
    perfCounters.add(S_MISSED_DIRTY);
    log.warning(
      Modules.ANIMATION,
      `Render audit: tick ${tick} would have been skipped but ${differing}/` +
        `${AUDIT_SIZE * AUDIT_SIZE} pixels changed since the last rendered frame ` +
        `(missed dirty). Last render requests: ${this.reasons.join(', ') || '(none)'}`
    );
    return true;
  }
}
