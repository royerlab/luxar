import * as zarr from '../../../zarr';
import { get, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
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
 * column phase exact regardless of resolution order. Decode itself stays on
 * the main thread deliberately — it is a single multiply-add per element
 * (no expm1, no worker `decodePerChannel` kernel exists), so the cost is
 * marginal next to the network fetch it overlaps with.
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

  const shape = array.shape;
  const { offsets, counts, total } = rangeDestOffsets(shape, ranges);

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
      for (let j = 0; j < data.length; j++) {
        const g = base + j;
        output[g] = dequant(Number(data[j]), g % cols);
      }
    })
  );

  return total;
}
