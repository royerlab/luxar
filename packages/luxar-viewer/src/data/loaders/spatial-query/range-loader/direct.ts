import * as zarr from '../../../zarr';
import { get, abortOptions } from '../../../zarr';
import { log } from '../../../../utils/log';
import type { LoadRange } from '../../base-types';
import {
  copyDirectChunk,
  type DirectOutputBuffer,
  firstAxisRangeSlice,
  type RangeNumericArray,
  type ResolvedRangeLoaderConfig,
} from './encoding-types';

export interface DirectCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
  /** Per-update abort signal forwarded to `get()` (see RangeLoader). */
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

  let destOffset = 0;
  const shape = array.shape;

  for (const range of ranges) {
    const sliceSpec = firstAxisRangeSlice(shape, range);
    const chunkData = await get(array, sliceSpec, abortOptions(ctx.signal));
    destOffset += copyDirectChunk(output, chunkData.data as RangeNumericArray, destOffset);
  }

  return destOffset;
}
