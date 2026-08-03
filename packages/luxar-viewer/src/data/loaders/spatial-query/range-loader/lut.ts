import * as zarr from '../../../zarr';
import { readArray, abortOptions } from '../../../zarr';
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

export interface LUTCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  decoder: ArrayDecoder;
  /** Per-update abort signal forwarded to the worker decode (see RangeLoader). */
  signal?: AbortSignal | null;
}

export async function loadLUT(
  ctx: LUTCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata,
  ranges: LoadRange[],
  output: Float32Array
): Promise<number> {
  const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
  if (!lutMetadata) throw new Error('[RangeLoader] LUT metadata missing');

  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
  const shouldUseWorkers = useWorkers && totalPoints > ctx.config.workerThreshold;

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `LUT: ${totalPoints} indices, k=${lutMetadata.k}, mode=${lutMetadata.lutMode} (worker=${shouldUseWorkers})`
    );
  }

  const flatLUT: number[] = Array.isArray(lutMetadata.lut[0])
    ? (lutMetadata.lut as number[][]).flat()
    : (lutMetadata.lut as number[]);

  const shape = array.shape;

  // Load + decode all ranges CONCURRENTLY: destination offsets are
  // precomputed (row mode decodes k values per stored index) so each range
  // writes into its own disjoint output span regardless of resolution order.
  // Network concurrency is bounded by the global fetch gate, decode
  // concurrency by the worker pool.
  const perElement = lutMetadata.lutMode === 'row' ? lutMetadata.k : 1;
  const { offsets, counts, total } = rangeDestOffsets(shape, ranges, perElement);

  await Promise.all(
    ranges.map(async (range, i) => {
      const sliceSpec = firstAxisRangeSlice(shape, range);
      const chunkData = await readArray(array, sliceSpec, abortOptions(ctx.signal));
      const indices = chunkData.data as Uint8Array | Uint16Array;

      let decoded: Float32Array;
      if (shouldUseWorkers) {
        try {
          decoded = await getWorkerPool().runWithTimeout(
            'decodeLUT',
            'decode',
            (api) =>
              api.decodeLUT({
                indices,
                lut: flatLUT,
                k: lutMetadata.k,
                lutMode: lutMetadata.lutMode as 'row' | 'scalar',
              }),
            ctx.signal ?? undefined
          );
        } catch (error) {
          if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
          log.warning(
            ctx.config.logModule,
            'Worker LUT decode failed, falling back to main thread:',
            error
          );
          decoded = ctx.decoder.decodeLUTIndices(indices, lutMetadata);
        }
      } else {
        decoded = ctx.decoder.decodeLUTIndices(indices, lutMetadata);
      }

      output.set(clampRangeData(decoded, counts[i], i, ctx.config.logModule), offsets[i]);
    })
  );

  return total;
}
