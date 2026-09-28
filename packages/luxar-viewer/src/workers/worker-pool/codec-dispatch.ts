/**
 * Batched, load-spread dispatch of blosc chunk decodes to the data workers —
 * the {@link BloscDecodeBackend} the pool installs for
 * `data/codecs/worker-blosc.ts`.
 *
 * Decodes requested in the same microtask (zarrita resolves a multi-chunk
 * selection's chunks together; a warm L1 hands many back at once) are queued
 * and flushed as batches of at most {@link MAX_BATCH_CHUNKS} chunks /
 * {@link MAX_BATCH_BYTES} decoded bytes. A flush is split so the chunks spread
 * over the IDLE workers first (one batch per idle worker, sized evenly) rather
 * than piling onto one — every batch still goes through the pool's least-busy
 * selection, whose slot stays busy until the worker settles.
 *
 * Returns `null` (decode locally) while the pool has no usable worker — the
 * first paint must not wait for workers to spawn — and for a chunk below
 * {@link MIN_OFFLOAD_DECODED_BYTES}, whose decode is cheaper than the trip.
 *
 * @module workers/worker-pool/codec-dispatch
 */

import { transfer, type Remote } from 'comlink';

import type { BloscDecodeBackend, BloscDecodeRequest } from '../../data/codecs/worker-blosc';
import type { DataWorkerAPI } from '../data-worker';
import type { BloscBatchResult } from '../data-worker/decode/blosc';

/** Most chunks in one worker message. */
export const MAX_BATCH_CHUNKS = 8;
/** Most DECODED bytes in one worker message (from the blosc headers). */
export const MAX_BATCH_BYTES = 8 * 1024 * 1024;
/**
 * Chunks decoding to fewer bytes stay on the main thread: their blosc (+ delta)
 * costs less than the round trip. Measured (Chrome, M4 Max, 6 s playback of the
 * `tp50` gate store — ~49 chunks/tick, median 208 B compressed): offloading
 * every chunk RAISED main-thread busy from ~20% to ~28%, while the codecs it
 * removed had cost ~3 ms in total. The Luxar writer targets 64 KB chunks, so
 * real payload chunks clear this comfortably.
 */
export const MIN_OFFLOAD_DECODED_BYTES = 16 * 1024;

/** The pool surface the dispatcher needs (a port, so tests can fake it). */
export interface CodecPoolPort {
  /** At least one worker is usable right now. */
  isInitialized(): boolean;
  /** Workers with no task in flight. */
  getIdleWorkerCount(): number;
  /** Run `fn` on the least-busy worker (timeout-guarded, never pool-aborted). */
  runDecode<T>(op: string, fn: (api: Remote<DataWorkerAPI>) => Promise<T>): Promise<T>;
}

interface Pending {
  request: BloscDecodeRequest;
  resolve: (data: Uint8Array) => void;
  reject: (error: unknown) => void;
}

/**
 * Decoded size a blosc frame declares (header bytes 4..7, little-endian
 * `nbytes`), or the compressed size when the header is too short to read.
 */
export function bloscDecodedSize(bytes: Uint8Array): number {
  if (bytes.byteLength < 16) return bytes.byteLength;
  return new DataView(bytes.buffer, bytes.byteOffset, 16).getUint32(4, true);
}

/** Split `items` into batches of at most `perBatch` chunks and {@link MAX_BATCH_BYTES}. */
export function splitBatches(items: Pending[], perBatch: number): Pending[][] {
  const batches: Pending[][] = [];
  let current: Pending[] = [];
  let bytes = 0;
  for (const item of items) {
    const size = bloscDecodedSize(item.request.bytes);
    if (current.length > 0 && (current.length >= perBatch || bytes + size > MAX_BATCH_BYTES)) {
      batches.push(current);
      current = [];
      bytes = 0;
    }
    current.push(item);
    bytes += size;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export class BloscDecodeDispatcher implements BloscDecodeBackend {
  private queue: Pending[] = [];
  private flushScheduled = false;

  constructor(private readonly pool: CodecPoolPort) {}

  decode(request: BloscDecodeRequest): Promise<Uint8Array> | null {
    if (!this.pool.isInitialized()) return null;
    if (bloscDecodedSize(request.bytes) < MIN_OFFLOAD_DECODED_BYTES) return null;
    return new Promise<Uint8Array>((resolve, reject) => {
      this.queue.push({ request, resolve, reject });
      if (!this.flushScheduled) {
        this.flushScheduled = true;
        queueMicrotask(() => this.flush());
      }
    });
  }

  private flush(): void {
    this.flushScheduled = false;
    const items = this.queue;
    this.queue = [];
    if (items.length === 0) return;
    // Spread over the idle workers: one batch each, evenly sized, capped.
    const idle = Math.max(1, this.pool.getIdleWorkerCount());
    const perBatch = Math.min(MAX_BATCH_CHUNKS, Math.ceil(items.length / idle));
    for (const batch of splitBatches(items, perBatch)) this.dispatch(batch);
  }

  private dispatch(batch: Pending[]): void {
    // COPY before transferring: the request's bytes are borrowed from the L1
    // cache (and a transfer would detach them under it). The copy owns its
    // whole buffer, so transferring it moves exactly the chunk.
    const payload = batch.map(({ request }) => ({ ...request, bytes: request.bytes.slice() }));
    const buffers = payload.map((r) => r.bytes.buffer as ArrayBuffer);
    this.pool
      .runDecode('decodeBloscBatch', (api) => api.decodeBloscBatch(transfer(payload, buffers)))
      .then(
        (results: BloscBatchResult[]) => settleBatch(batch, results),
        (error: unknown) => batch.forEach((p) => p.reject(error))
      );
  }
}

function settleBatch(batch: Pending[], results: BloscBatchResult[]): void {
  batch.forEach((pending, i) => {
    const result = results[i];
    if (result && 'data' in result) pending.resolve(result.data);
    else pending.reject(new Error(result?.error ?? 'decodeBloscBatch: missing result'));
  });
}
