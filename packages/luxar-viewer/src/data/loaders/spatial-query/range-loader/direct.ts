import * as zarr from '../../../zarr';
import { readArray, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import type { LoadRange } from '../../base-types';
import {
  clampRangeData,
  copyDirectChunk,
  type DirectOutputBuffer,
  firstAxisRangeSlice,
  rangeDestOffsets,
  type RangeNumericArray,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';

export interface DirectCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  /** Per-update abort signal forwarded to `readArray()` (see RangeLoader). */
  signal?: AbortSignal | null;
}

/**
 * Canonical direct (unencoded) range reader for ALL geometry types and
 * attributes. Reads each range and copies it into `output` preserving the
 * output's native dtype (Float32 / Float16 / Uint8 / Uint16 / Uint32) — so a
 * Float32 `output` widens a Uint8 source, while a Uint8 `output` keeps bytes
 * intact (critical for uint8 colors, which THREE.js normalizes 0–255 → 0–1).
 * This is the single implementation behind both `RangeLoader.loadRanges`'
 * `'direct'` case and `RangeLoader.loadDirectTyped`.
 */
export async function loadDirect(
  ctx: DirectCtx,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  ranges: LoadRange[],
  output: DirectOutputBuffer
): Promise<number> {
  if (ctx.verbose) {
    log.info(ctx.config.logModule, `Direct: loading ${ranges.length} ranges`);
  }

  const shape = array.shape;

  // Load all ranges CONCURRENTLY: destination offsets are precomputed
  // (prefix sum over deterministic range sizes) so each range writes into
  // its own disjoint output span regardless of resolution order. Network
  // concurrency is bounded by the global fetch gate (see
  // utils/fetch-concurrency.ts), not here.
  const { offsets, counts, total } = rangeDestOffsets(shape, ranges);

  await Promise.all(
    ranges.map(async (range, i) => {
      const sliceSpec = firstAxisRangeSlice(shape, range);
      const chunkData = await readArray(array, sliceSpec, abortOptions(ctx.signal));
      const data = clampRangeData(
        chunkData.data as RangeNumericArray,
        counts[i],
        i,
        ctx.config.logModule
      );
      copyDirectChunk(output, data, offsets[i]);
    })
  );

  return total;
}
