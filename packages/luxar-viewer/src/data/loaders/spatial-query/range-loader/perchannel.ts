import * as zarr from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import type { PerChannelKind } from '../../../../workers/data-worker/decode/perchannel';
import type { LoadRange } from '../../base-types';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import {
  rangeDestOffsets,
  type RangeNumericArray,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';
import { packRanges, readRanges, scatterDecoded } from './packed-ranges';

export interface PerChannelCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  /** Per-update abort signal forwarded to `readArray()` (see RangeLoader). */
  signal?: AbortSignal | null;
}

/**
 * Map a per-channel encoding NAME to the worker decode kind. Exported so the
 * mapping is unit-testable — a mis-route here (e.g. geolog -> 'log') decodes
 * every HDR color with the wrong inverse compand and no error. Order matters:
 * 'geolog_perchannel*' must not be caught by a 'log_perchannel' prefix test
 * ('geolog...' does NOT start with 'log', but keep the explicit order anyway).
 */
export function perChannelKindFor(name: string): PerChannelKind {
  return name.startsWith('signed_log_perchannel')
    ? 'signed_log'
    : name.startsWith('log_perchannel')
      ? 'log'
      : name.startsWith('geolog_perchannel')
        ? 'geolog'
        : 'linear';
}

/** The subset of encoding metadata the per-channel family carries. */
interface PerChannelEncoding {
  name?: string;
  bits?: number;
  col_lo?: number[];
  col_hi?: number[];
  zero_level?: boolean;
}

/**
 * Fully decode a per-channel quantized array (`log_perchannel_*` /
 * `signed_log_perchannel_*` / `linear_perchannel_*`) into a Float32 `output`.
 *
 * This makes per-channel a first-class self-decoded encoding — like
 * `quantized` / `lut` / `broadcasted` — so the decode layer owns dequantization
 * and consumers receive decoded float32 (no consumer-side dequant). Mirrors the
 * Python `ArrayDecoder._decode_*_perchannel`: raw integer levels are dequantized
 * with the array's own per-column `col_lo`/`col_hi` scales.
 *
 * `elementsPerItem` is the column count C (the per-channel dimension, e.g. ndim
 * for positions/centers, d for the Cholesky diagonal): the flattened output is
 * `[item*C + col]`, so `col = globalIndex % C`.
 *
 * Ranges are fetched CONCURRENTLY (like `loadDirect` / `loadQuantized`), then
 * packed at their precomputed destination offsets and decoded as ONE batch
 * (`./packed-ranges.ts`): the packed index IS the global flattened index, so
 * the column phase stays exact. Above the worker threshold that single decode
 * runs in the worker pool on the WASM per-channel kernels
 * (`decode_*_perchannel_*`) — the log/signed-log Cholesky pair costs an
 * `expm1` per element, which is real main-thread work at millions of splats.
 * Below the threshold (or on worker failure) the main-thread
 * `makePerChannelDequant` closure decodes bit-identically — both paths do the
 * same f64 math on the same f64 scales.
 */
export async function loadPerChannel(
  ctx: PerChannelCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata,
  ranges: LoadRange[],
  output: Float32Array,
  elementsPerItem: number
): Promise<number> {
  const cols = Math.max(1, elementsPerItem);
  // Throws (does not silently zero-fill) on missing/malformed per-column
  // scales — this validation guards BOTH decode paths, and the returned
  // closure is the sub-threshold / worker-failure decoder.
  const dequant = ArrayDecoder.makePerChannelDequant(attrs.encoding, cols);

  const params = perChannelParams((attrs.encoding ?? {}) as PerChannelEncoding);
  const { offsets, counts, total } = rangeDestOffsets(array.shape, ranges);

  const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
  const shouldUseWorkers = useWorkers && total > ctx.config.workerThreshold;

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `PerChannel: decoding ${ranges.length} ranges (${params.name}, kind=${params.kind}, worker=${shouldUseWorkers})`
    );
  }

  if (total === 0) return 0;
  const parts = await readRanges(array, ranges, ctx.signal);
  const pack = packRanges(parts, {
    offsets,
    counts,
    total,
    perElement: 1,
    logModule: ctx.config.logModule,
  });
  const data = pack.packed;

  let decoded = shouldUseWorkers ? await decodeOnWorker(ctx, params, data) : null;
  if (!decoded) {
    decoded = new Float32Array(data.length);
    for (let g = 0; g < data.length; g++) {
      decoded[g] = dequant(Number(data[g]), g % cols);
    }
  }
  scatterDecoded(decoded, output, pack, offsets);
  return total;
}

/** The decode kernel inputs a per-channel encoding carries. */
interface PerChannelParams {
  name: string;
  kind: PerChannelKind;
  bits: 8 | 16;
  zeroLevel: boolean;
  /** f64 scales, straight from the attrs (structured-cloned to the worker). */
  colLo: Float64Array;
  colHi: Float64Array;
}

function perChannelParams(enc: PerChannelEncoding): PerChannelParams {
  const name = enc.name ?? '';
  return {
    name,
    kind: perChannelKindFor(name),
    bits: (enc.bits ?? (name.endsWith('u8') ? 8 : 16)) === 8 ? 8 : 16,
    zeroLevel: enc.zero_level === true,
    colLo: Float64Array.from(enc.col_lo ?? []),
    colHi: Float64Array.from(enc.col_hi ?? []),
  };
}

/**
 * Decode the whole pack in ONE worker call (not one per range): the pack
 * starts at global index 0, so `colOffset` is 0 and each code's column is its
 * packed index modulo `cols` — the same phase as its output position. Returns
 * `null` (decode on the main thread) for a non-integer container or a worker
 * failure; an abort is rethrown.
 */
async function decodeOnWorker(
  ctx: PerChannelCtx,
  params: PerChannelParams,
  data: RangeNumericArray
): Promise<Float32Array | null> {
  if (!(data instanceof Uint8Array || data instanceof Uint16Array)) return null;
  const { kind, colLo, colHi, zeroLevel, bits } = params;
  try {
    return await getWorkerPool().runWithTimeout(
      'decodePerChannel',
      'decode',
      (api) => api.decodePerChannel({ data, kind, colLo, colHi, zeroLevel, colOffset: 0, bits }),
      ctx.signal ?? undefined
    );
  } catch (error) {
    if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
    log.warning(
      ctx.config.logModule,
      'Worker per-channel decoding failed, falling back to main thread:',
      error
    );
    return null;
  }
}
