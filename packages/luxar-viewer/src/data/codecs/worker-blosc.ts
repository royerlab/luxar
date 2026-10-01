/**
 * Worker-offloaded `blosc` zarr codec (main-thread side).
 *
 * Registered by `../zarr.ts` in place of zarrita's `blosc` / `numcodecs.blosc`
 * entries. When a decode backend is installed (the data-worker pool installs
 * one, see `workers/worker-pool/codec-dispatch.ts`), the backend COPIES the
 * compressed bytes (the L1 cache owns the originals), hands the copy to a data
 * worker and returns the decoded bytes the worker transfers back — so blosc's
 * WASM, and the embind copy of its input, run off the main thread.
 *
 * Measured motivation (real Chrome, cold h2afva playback, ~745k splats/frame):
 * the main thread was 86-93% busy at ~2 fps; per cold tick blosc/zstd cost
 * 53 ms and the `luxar_delta` decode after it 36 ms, while 13 of 14 data
 * workers idled.
 *
 * DELTA FUSION. A `luxar_delta_v1` filter directly beneath blosc in the chain
 * (`[luxar_delta_v1, bytes(little), blosc]`; format 2 omits the `bytes` entry)
 * is run IN THE WORKER right after blosc, so its JS loop leaves the main
 * thread too. The returned buffer is marked (`markDeltaDecoded`), and the
 * main-thread delta codec instance — still in zarrita's chain — passes that
 * one chunk through unchanged, so the delta is never applied twice. Anything
 * else between the two (a big-endian `bytes`, another filter) disables fusion
 * and the delta runs on the main thread as before.
 *
 * FALLBACK. With no backend (unit tests, embeds without workers, the
 * `?mainThreadCodecs` kill switch, a pool that has no usable worker yet), or
 * when the worker path fails for ANY reason, the chunk is decoded here by the
 * native numcodecs codec from the untouched original bytes — the exact
 * pre-offload pipeline. Blosc frames are self-describing (compressor, shuffle
 * and typesize live in the header), so the decode needs no config.
 *
 * Structural types only: `../zarr.ts` stays the sole zarrita import boundary
 * and injects the native codec loader via {@link setNativeBloscLoader}.
 */

import { perfCounters } from '../../profiling/perf-counters';
import { log, Modules } from '../../utils/log';
import { LuxarDeltaCodec, markDeltaDecoded, type LuxarDeltaSpec } from './luxar-delta';

/** One chunk decode handed to a worker. */
export interface BloscDecodeRequest {
  /** Compressed bytes, BORROWED (the L1 cache owns them): copy before transferring. */
  bytes: Uint8Array;
  /** Fused delta to undo after blosc, or `null` for blosc alone. */
  delta: LuxarDeltaSpec | null;
}

/** Where {@link WorkerBloscCodec} sends decodes (the data-worker pool). */
export interface BloscDecodeBackend {
  /**
   * Decode off the main thread, or return `null` when no worker can take the
   * job right now (the codec then decodes locally). A rejection also falls
   * back to the local decode.
   */
  decode(request: BloscDecodeRequest): Promise<Uint8Array> | null;
}

/** The numcodecs Blosc surface this module uses (structural). */
export interface NativeBlosc {
  decode(data: Uint8Array): Promise<Uint8Array> | Uint8Array;
  encode(data: Uint8Array): Promise<Uint8Array> | Uint8Array;
}

/** The numcodecs Blosc class (structural). */
export interface NativeBloscCtor {
  fromConfig(config: Record<string, unknown>): NativeBlosc;
}

/** Array metadata zarrita hands `fromConfig` (the fields read here). */
export interface ChunkMeta {
  dataType?: string;
  shape?: number[];
  codecs?: { name: string; configuration?: Record<string, unknown> }[];
}

const S_WORKER = perfCounters.slot('codec.blosc.worker');
const S_MAIN = perfCounters.slot('codec.blosc.main');
const S_FALLBACK = perfCounters.slot('codec.blosc.fallback');
const S_FUSED_DELTA = perfCounters.slot('codec.delta.fused');

const BLOSC_NAMES = new Set(['blosc', 'numcodecs.blosc']);
const DELTA_NAMES = new Set(['luxar_delta_v1', 'numcodecs.luxar_delta_v1']);

let backend: BloscDecodeBackend | null = null;
let workerCodecsEnabled = true;
let nativeLoader: (() => Promise<NativeBloscCtor>) | null = null;
let warnedFallback = false;

/** Install (or with `null`, remove) the off-main-thread decode backend. */
export function setBloscDecodeBackend(next: BloscDecodeBackend | null): void {
  backend = next;
}

/** The installed backend, if any (tests and the pool's own dispose check). */
export function getBloscDecodeBackend(): BloscDecodeBackend | null {
  return backend;
}

/**
 * Kill switch: `false` keeps every blosc decode on the main thread even when
 * a backend is installed (`?mainThreadCodecs`).
 */
export function setWorkerCodecsEnabled(enabled: boolean): void {
  workerCodecsEnabled = enabled;
}

/** Whether decodes may be offloaded (false under `?mainThreadCodecs`). */
export function areWorkerCodecsEnabled(): boolean {
  return workerCodecsEnabled;
}

/** Inject the native numcodecs Blosc loader (called once by `../zarr.ts`). */
export function setNativeBloscLoader(loader: () => Promise<NativeBloscCtor>): void {
  nativeLoader = loader;
}

/** Load the native numcodecs Blosc class (the main-thread fallback codec). */
export function loadNativeBloscCtor(): Promise<NativeBloscCtor> {
  if (!nativeLoader) return Promise.reject(new Error('worker-blosc: native blosc not configured'));
  return nativeLoader();
}

/**
 * The `luxar_delta_v1` config to fuse into this blosc decode, or `null`.
 * Fusion requires the delta to be the step decoded IMMEDIATELY after blosc,
 * with at most a little-endian `bytes` (a no-op view) in between.
 */
export function resolveFusedDelta(meta: ChunkMeta | undefined): LuxarDeltaSpec | null {
  const delta = deltaBeneathBlosc(meta?.codecs);
  if (!delta) return null;
  try {
    // Same validation the delta codec itself runs; an invalid config is left
    // to that codec to reject with its own error rather than fused.
    return LuxarDeltaCodec.fromConfig(delta.configuration ?? {}, meta ?? {}).spec;
  } catch {
    return null;
  }
}

type CodecEntry = NonNullable<ChunkMeta['codecs']>[number];

/** The delta entry decoded right after the chain's single blosc, if any. */
function deltaBeneathBlosc(codecs: CodecEntry[] | undefined): CodecEntry | null {
  if (!Array.isArray(codecs)) return null;
  const bloscAt = codecs.findIndex((c) => BLOSC_NAMES.has(c.name));
  if (bloscAt < 0 || codecs.filter((c) => BLOSC_NAMES.has(c.name)).length !== 1) return null;
  const at = isLittleEndianBytes(codecs[bloscAt - 1]) ? bloscAt - 2 : bloscAt - 1;
  const delta = codecs[at];
  return delta && DELTA_NAMES.has(delta.name) ? delta : null;
}

function isLittleEndianBytes(codec: CodecEntry | undefined): boolean {
  return codec?.name === 'bytes' && (codec.configuration?.endian ?? 'little') === 'little';
}

export class WorkerBloscCodec {
  /** The fused delta config (tests / diagnostics). */
  readonly delta: LuxarDeltaSpec | null;
  private readonly config: Record<string, unknown>;
  private native: Promise<NativeBlosc> | null = null;

  constructor(config: Record<string, unknown>, delta: LuxarDeltaSpec | null) {
    this.config = config;
    this.delta = delta;
  }

  static fromConfig(
    config: Record<string, unknown> | undefined,
    meta?: ChunkMeta
  ): WorkerBloscCodec {
    return new WorkerBloscCodec(config ?? {}, resolveFusedDelta(meta));
  }

  /** Viewer never writes; kept so the codec is a complete zarrita codec. */
  async encode(data: Uint8Array): Promise<Uint8Array> {
    return (await this.getNative()).encode(data);
  }

  async decode(bytes: Uint8Array): Promise<Uint8Array> {
    const job = workerCodecsEnabled ? backend?.decode({ bytes, delta: this.delta }) : null;
    if (job) {
      try {
        const decoded = await job;
        perfCounters.add(S_WORKER);
        if (this.delta) {
          markDeltaDecoded(decoded.buffer);
          perfCounters.add(S_FUSED_DELTA);
        }
        return decoded;
      } catch (error) {
        perfCounters.add(S_FALLBACK);
        if (!warnedFallback) {
          warnedFallback = true;
          log.warning(
            Modules.WORKER_POOL,
            'Worker blosc decode failed; decoding on the main thread',
            error
          );
        }
      }
    }
    perfCounters.add(S_MAIN);
    return (await this.getNative()).decode(bytes);
  }

  private getNative(): Promise<NativeBlosc> {
    this.native ??= loadNativeBloscCtor().then((Ctor) => Ctor.fromConfig(this.config));
    return this.native;
  }
}
