import * as zarr from '../../../zarr';
import { get, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import type { PerChannelKind } from '../../../../workers/data-worker/decode/perchannel';
import type { LoadRange } from '../../base-types';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import {
  clampRangeData,
  firstAxisRangeSlice,
  rangeDestOffsets,
  type RangeNumericArray,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';

export interface PerChannelCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  /** Per-update abort signal forwarded to `get()` (see RangeLoader). */
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
 * Ranges are fetched CONCURRENTLY (like `loadDirect` / `loadQuantized`):
 * destination offsets are precomputed, each range writes its own disjoint
 * output span, and the global flattened index `offsets[i] + j` keeps the
 * column phase exact regardless of resolution order. Above the worker
 * threshold, decode runs in the worker pool on the WASM per-channel kernels
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

  const enc = (attrs.encoding ?? {}) as PerChannelEncoding;
  const name = enc.name ?? '';
  const kind = perChannelKindFor(name);
  const bits: 8 | 16 = (enc.bits ?? (name.endsWith('u8') ? 8 : 16)) === 8 ? 8 : 16;
  const zeroLevel = enc.zero_level === true;
  // f64 scales, shared (structured-cloned) across all range decodes.
  const colLo = Float64Array.from(enc.col_lo ?? []);
  const colHi = Float64Array.from(enc.col_hi ?? []);

  const shape = array.shape;
  const { offsets, counts, total } = rangeDestOffsets(shape, ranges);

  const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
  const shouldUseWorkers = useWorkers && total > ctx.config.workerThreshold;

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `PerChannel: decoding ${ranges.length} ranges (${name}, kind=${kind}, worker=${shouldUseWorkers})`
    );
  }

  await Promise.all(
    ranges.map(async (range, i) => {
      const sliceSpec = firstAxisRangeSlice(shape, range);
      const chunkData = await get(array, sliceSpec, abortOptions(ctx.signal));
      const data = clampRangeData(
        chunkData.data as RangeNumericArray,
        counts[i],
        i,
        ctx.config.logModule
      );
      const base = offsets[i];

      if (shouldUseWorkers && (data instanceof Uint8Array || data instanceof Uint16Array)) {
        try {
          const decoded = await getWorkerPool().runWithTimeout(
            'decodePerChannel',
            'decode',
            (api) =>
              api.decodePerChannel({
                data,
                kind,
                colLo,
                colHi,
                zeroLevel,
                colOffset: base % cols,
                bits,
              }),
            ctx.signal ?? undefined
          );
          output.set(decoded, base);
          return;
        } catch (error) {
          if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
          log.warning(
            ctx.config.logModule,
            'Worker per-channel decoding failed, falling back to main thread:',
            error
          );
        }
      }

      for (let j = 0; j < data.length; j++) {
        const g = base + j;
        output[g] = dequant(Number(data[j]), g % cols);
      }
    })
  );

  return total;
}
