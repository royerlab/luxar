import * as zarr from '../../../zarr';
import { get, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import type { LoadRange } from '../../base-types';
import {
  clampRangeData,
  firstAxisRangeSlice,
  rangeDestOffsets,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';

export interface QuantizedCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  decoder: ArrayDecoder;
  /** Per-update abort signal forwarded to the worker decode (see RangeLoader). */
  signal?: AbortSignal | null;
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

  const shape = array.shape;

  // Load + decode all ranges CONCURRENTLY: dequantization is element-wise
  // 1:1, so destination offsets are precomputed and each range writes into
  // its own disjoint output span regardless of resolution order. Network
  // concurrency is bounded by the global fetch gate, decode concurrency by
  // the worker pool.
  const { offsets, counts, total } = rangeDestOffsets(shape, ranges);

  await Promise.all(
    ranges.map(async (range, i) => {
      const sliceSpec = firstAxisRangeSlice(shape, range);
      const chunkData = await get(array, sliceSpec, abortOptions(ctx.signal));
      const quantizedData = chunkData.data as Uint8Array | Uint16Array;

      let dequantized: Float32Array;
      if (shouldUseWorkers) {
        try {
          if (quantMetadata.isLogSpace) {
            dequantized = await getWorkerPool().runWithTimeout(
              'decodeLogScalar',
              'decode',
              (api) =>
                api.decodeLogScalar({
                  data: quantizedData,
                  maxLog: quantMetadata.bounds[1],
                  dtype: quantMetadata.dtype,
                }),
              ctx.signal ?? undefined
            );
          } else {
            dequantized = await getWorkerPool().runWithTimeout(
              'decodeQuantized',
              'decode',
              (api) =>
                api.decodeQuantized({
                  data: quantizedData,
                  bounds: quantMetadata.bounds,
                  dtype: quantMetadata.dtype,
                }),
              ctx.signal ?? undefined
            );
          }
        } catch (error) {
          if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
          log.warning(
            ctx.config.logModule,
            'Worker decoding failed, falling back to main thread:',
            error
          );
          dequantized = ctx.decoder.dequantizeRange(quantizedData, quantMetadata);
        }
      } else {
        dequantized = ctx.decoder.dequantizeRange(quantizedData, quantMetadata);
      }

      output.set(clampRangeData(dequantized, counts[i], i, ctx.config.logModule), offsets[i]);
    })
  );

  return total;
}
