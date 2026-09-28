import * as zarr from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import type { LoadRange } from '../../base-types';
import { rangeDestOffsets, type ResolvedRangeLoaderConfig } from './encoding-types';
import { packRanges, readRanges, scatterDecoded } from './packed-ranges';

export interface LUTCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  decoder: ArrayDecoder;
  /** Per-update abort signal forwarded to the worker decode (see RangeLoader). */
  signal?: AbortSignal | null;
}

type LUTMetadata = NonNullable<ReturnType<typeof ArrayDecoder.getLUTMetadata>>;

/** Decode LUT `indices` in the worker pool, falling back to the main thread. */
async function decodeIndices(
  ctx: LUTCtx,
  lutMetadata: LUTMetadata,
  indices: Uint8Array | Uint16Array,
  useWorker: boolean
): Promise<Float32Array> {
  if (useWorker) {
    const lut: number[] = Array.isArray(lutMetadata.lut[0])
      ? (lutMetadata.lut as number[][]).flat()
      : (lutMetadata.lut as number[]);
    try {
      return await getWorkerPool().runWithTimeout(
        'decodeLUT',
        'decode',
        (api) =>
          api.decodeLUT({
            indices,
            lut,
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
    }
  }
  return ctx.decoder.decodeLUTIndices(indices, lutMetadata);
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

  // Load all ranges CONCURRENTLY, then decode them as ONE pack (one worker
  // call per attribute, see packed-ranges.ts). Destination offsets are in
  // OUTPUT units — row mode decodes k values per stored index — so the pack
  // places each range's indices at `offsets[i] / k`.
  const perElement = lutMetadata.lutMode === 'row' ? lutMetadata.k : 1;
  const { offsets, counts, total } = rangeDestOffsets(array.shape, ranges, perElement);
  if (total === 0) return 0;
  const parts = await readRanges(array, ranges, ctx.signal);
  const pack = packRanges(parts, {
    offsets,
    counts,
    total,
    perElement,
    logModule: ctx.config.logModule,
  });
  const indices = pack.packed as Uint8Array | Uint16Array;
  const decoded = await decodeIndices(ctx, lutMetadata, indices, shouldUseWorkers);
  scatterDecoded(decoded, output, pack, offsets);

  return total;
}
