/**
 * Blosc chunk decompression (+ an optional fused `luxar_delta_v1` undo),
 * run inside a data worker — the worker half of `data/codecs/worker-blosc.ts`.
 *
 * The main thread batches chunk decodes (see
 * `workers/worker-pool/codec-dispatch.ts`) and transfers each chunk's
 * compressed bytes here; the decoded bytes are transferred back. The codec is
 * the same numcodecs Blosc the main thread would have run (blosc frames are
 * self-describing, so no per-array config is needed), and the fused delta runs
 * the same `LuxarDeltaCodec` decode — which is what makes the result
 * byte-identical to the main-thread pipeline
 * (`tests/unit/data/worker-codec-identity.test.ts`).
 *
 * Errors are reported PER CHUNK so one corrupt chunk falls back alone (the main
 * thread re-decodes it locally, surfacing the real error) instead of failing
 * the whole batch.
 */

import { transfer } from 'comlink';

import { loadNativeBlosc } from '../../../data/zarr';
import { LuxarDeltaCodec, type LuxarDeltaSpec } from '../../../data/codecs/luxar-delta';
import type { NativeBlosc } from '../../../data/codecs/worker-blosc';

/** One chunk to decode. */
export interface BloscBatchItem {
  bytes: Uint8Array;
  delta: LuxarDeltaSpec | null;
}

/** Per-chunk outcome: decoded bytes, or the error message. */
export type BloscBatchResult = { data: Uint8Array } | { error: string };

let codec: Promise<NativeBlosc> | null = null;

/** The worker's (lazily instantiated) native Blosc codec. */
function getCodec(): Promise<NativeBlosc> {
  if (!codec) {
    const loading = loadNativeBlosc().then((Blosc) => Blosc.fromConfig({}));
    // A failed load must not poison the worker for good: let the next call retry.
    loading.catch(() => {
      if (codec === loading) codec = null;
    });
    codec = loading;
  }
  return codec;
}

/** Undo a fused delta on blosc's output; returns the codes' bytes. */
function undoDelta(raw: Uint8Array, delta: LuxarDeltaSpec): Uint8Array {
  // Blosc's output is a fresh copy at offset 0, so a Uint16 view is aligned.
  const codes =
    delta.bits === 8
      ? raw
      : new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / Uint16Array.BYTES_PER_ELEMENT);
  const { data } = new LuxarDeltaCodec(delta.cols, delta.bits).decodeResiduals({
    data: codes,
    shape: [],
    stride: [],
  });
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

async function decodeOne(blosc: NativeBlosc, item: BloscBatchItem): Promise<BloscBatchResult> {
  try {
    const raw = await blosc.decode(item.bytes);
    return { data: item.delta ? undoDelta(raw, item.delta) : raw };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/** Decode a batch of blosc chunks; the decoded buffers are transferred back. */
export async function decodeBloscBatch(items: BloscBatchItem[]): Promise<BloscBatchResult[]> {
  const blosc = await getCodec();
  const results: BloscBatchResult[] = [];
  // Sequential: the blosc WASM is single-threaded, so there is nothing to overlap.
  for (const item of items) results.push(await decodeOne(blosc, item));
  const buffers = results.flatMap((r) => ('data' in r ? [r.data.buffer as ArrayBuffer] : []));
  return transfer(results, buffers);
}

/**
 * Instantiate the blosc WASM now (the first decode would otherwise pay the
 * ~600 KB module's base64 decode + compile). Idempotent.
 */
export async function warmCodecs(): Promise<void> {
  const blosc = await getCodec();
  await blosc.encode(new Uint8Array(16));
}
