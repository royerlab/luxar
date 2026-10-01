import * as zarr from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import type { LoadRange } from '../../base-types';
import { rangeDestOffsets, type ResolvedRangeLoaderConfig } from './encoding-types';
import { packRanges, readRanges, scatterDecoded } from './packed-ranges';

export interface QuantizedCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  decoder: ArrayDecoder;
  /** Per-update abort signal forwarded to the worker decode (see RangeLoader). */
  signal?: AbortSignal | null;
}

type QuantMetadata = NonNullable<ReturnType<typeof ArrayDecoder.getQuantizationMetadata>>;

/** Dequantize `data` in the worker pool, on the kernel `quantMetadata` selects. */
function decodeOnWorker(
  ctx: QuantizedCtx,
  quantMetadata: QuantMetadata,
  data: Uint8Array | Uint16Array
): Promise<Float32Array> {
  const pool = getWorkerPool();
  const signal = ctx.signal ?? undefined;
  const { bounds, dtype } = quantMetadata;
  if (quantMetadata.isGeologSpace) {
    return pool.runWithTimeout(
      'decodeGeologScalar',
      'decode',
      (api) => api.decodeGeologScalar({ data, minLog: bounds[0], maxLog: bounds[1], dtype }),
      signal
    );
  }
  if (quantMetadata.isLogSpace) {
    return pool.runWithTimeout(
      'decodeLogScalar',
      'decode',
      (api) => api.decodeLogScalar({ data, maxLog: bounds[1], dtype }),
      signal
    );
  }
  return pool.runWithTimeout(
    'decodeQuantized',
    'decode',
    (api) => api.decodeQuantized({ data, bounds, dtype }),
    signal
  );
}

/** Worker decode with the main-thread fallback (an abort is rethrown, not decoded). */
async function dequantize(
  ctx: QuantizedCtx,
  quantMetadata: QuantMetadata,
  data: Uint8Array | Uint16Array,
  useWorker: boolean
): Promise<Float32Array> {
  if (useWorker) {
    try {
      return await decodeOnWorker(ctx, quantMetadata, data);
    } catch (error) {
      if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
      log.warning(
        ctx.config.logModule,
        'Worker decoding failed, falling back to main thread:',
        error
      );
    }
  }
  return ctx.decoder.dequantizeRange(data, quantMetadata);
}

export async function loadQuantized(
  ctx: QuantizedCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata,
  ranges: LoadRange[],
  output: Float32Array
): Promise<number> {
  const zarrDtype = String(array.dtype);
  const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs, zarrDtype);
  if (!quantMetadata) throw new Error('[RangeLoader] Quantization metadata missing');

  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
  const shouldUseWorkers = useWorkers && totalPoints > ctx.config.workerThreshold;

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `Quantized: ${totalPoints} values (${ArrayDecoder.getEncodingMode(attrs)}, dtype=${quantMetadata.dtype}, worker=${shouldUseWorkers})`
    );
  }

  // Load all ranges CONCURRENTLY (network concurrency is bounded by the global
  // fetch gate), then dequantize them as ONE pack: dequantization is
  // element-wise 1:1, so every range keeps its precomputed destination offset
  // and the whole attribute costs a single worker call (see packed-ranges.ts).
  const { offsets, counts, total } = rangeDestOffsets(array.shape, ranges);
  if (total === 0) return 0;
  const parts = await readRanges(array, ranges, ctx.signal);
  const pack = packRanges(parts, {
    offsets,
    counts,
    total,
    perElement: 1,
    logModule: ctx.config.logModule,
  });
  const quantizedData = pack.packed as Uint8Array | Uint16Array;
  const dequantized = await dequantize(ctx, quantMetadata, quantizedData, shouldUseWorkers);
  scatterDecoded(dequantized, output, pack, offsets);

  return total;
}
