import * as zarr from '../../../zarr';
import { get } from '../../../zarr';
import { log } from '../../../../utils/log';
import type { LoadRange } from '../../base-types';
import {
  firstAxisRangeSlice,
  numericArrayToFloat32,
  type RangeNumericArray,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';

export interface DirectCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
}

export async function loadDirect(
  ctx: DirectCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  ranges: LoadRange[],
  output: Float32Array
): Promise<number> {
  if (ctx.verbose) {
    log.info(ctx.config.logModule, `Direct: loading ${ranges.length} ranges`);
  }

  let destOffset = 0;
  const shape = array.shape;

  for (const range of ranges) {
    const sliceSpec = firstAxisRangeSlice(shape, range);
    const chunkData = await get(array, sliceSpec);
    const float32Data = numericArrayToFloat32(chunkData.data as RangeNumericArray);
    output.set(float32Data, destOffset);
    destOffset += float32Data.length;
  }

  return destOffset;
}
