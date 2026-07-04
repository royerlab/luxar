import * as zarr from '../../../zarr';
import { get, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import type { LoadRange } from '../../base-types';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import {
  firstAxisRangeSlice,
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
  // Throws (does not silently zero-fill) on missing/malformed per-column scales.
  const dequant = ArrayDecoder.makePerChannelDequant(attrs.encoding, cols);

  if (ctx.verbose) {
    log.info(
      ctx.config.logModule,
      `PerChannel: decoding ${ranges.length} ranges (${attrs.encoding?.name})`
    );
  }

  let destOffset = 0;
  const shape = array.shape;
  for (const range of ranges) {
    const sliceSpec = firstAxisRangeSlice(shape, range);
    const chunkData = await get(array, sliceSpec, abortOptions(ctx.signal));
    const data = chunkData.data as RangeNumericArray;
    for (let i = 0; i < data.length; i++) {
      const g = destOffset + i;
      output[g] = dequant(Number(data[i]), g % cols);
    }
    destOffset += data.length;
  }

  return destOffset;
}
