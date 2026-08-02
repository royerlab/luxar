import * as zarr from '../../../zarr';
import { readArray, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import { config as appConfig } from '../../../../config';
import { getWorkerPool } from '../../../../workers/worker-pool';
import type { ArrayMetadata } from '../../../array-decoder/decoder';
import type { ResolvedRangeLoaderConfig } from './encoding-types';

export interface BroadcastedCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  /** Per-update abort signal forwarded to the worker decode (see RangeLoader). */
  signal?: AbortSignal | null;
}

export async function loadBroadcasted(
  ctx: BroadcastedCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata,
  output: Float32Array,
  totalElements: number,
  elementsPerItem: number
): Promise<void> {
  const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
  const arrayName = attrs.encoding?.name || 'broadcasted';

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `Broadcasted: replicating to ${totalElements} elements (worker=${useWorkers})`
    );
  }

  const fullData = await readArray(array, undefined, abortOptions(ctx.signal));
  const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;
  const valueAsFloat32 =
    broadcastValue instanceof Float32Array ? broadcastValue : new Float32Array(broadcastValue);

  if (useWorkers && totalElements > ctx.config.workerThreshold) {
    try {
      const decoded = await getWorkerPool().runWithTimeout(
        'decodeBroadcasted',
        'decode',
        (api) =>
          api.decodeBroadcasted({
            value: valueAsFloat32,
            numPoints: totalElements,
            elementsPerPoint: elementsPerItem,
          }),
        ctx.signal ?? undefined
      );
      output.set(decoded);
      return;
    } catch (error) {
      if (error instanceof Error && error.name === 'WorkerAbortError') throw error;
      log.warning(
        ctx.config.logModule,
        `Worker broadcast failed for ${arrayName}, falling back to main thread:`,
        error
      );
    }
  }

  for (let i = 0; i < totalElements; i++) {
    for (let j = 0; j < elementsPerItem; j++) {
      output[i * elementsPerItem + j] = valueAsFloat32[j] ?? valueAsFloat32[0];
    }
  }
}
