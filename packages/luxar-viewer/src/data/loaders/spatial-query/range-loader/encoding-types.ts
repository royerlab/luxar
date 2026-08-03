import { log, Modules } from '../../../../utils/log';
import * as zarr from '../../../zarr';
import { slice } from '../../../zarr';
import type { LoadRange } from '../../base-types';

export type EncodingType =
  'broadcasted' | 'quantized' | 'lut' | 'array_ref' | 'perchannel' | 'direct';

export interface RangeLoaderConfig {
  /** Minimum elements before using workers (default: 1000) */
  workerThreshold?: number;
  /** Log module for debug output */
  logModule?: string;
}

export type ResolvedRangeLoaderConfig = Required<RangeLoaderConfig>;

export const DEFAULT_RANGE_LOADER_CONFIG: ResolvedRangeLoaderConfig = {
  workerThreshold: 1000,
  logModule: Modules.SPATIAL_INDEX_LOADER,
};

export type RangeNumericArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigUint64Array
  | BigInt64Array;

/**
 * Output buffer kinds the unified direct reader writes into. Spans every
 * native type a direct (unencoded) attribute is allocated as: positions
 * (Float32), scalars (Float16/Float32), colors (Uint8/Uint16/Float32), and
 * lines segments (Uint32). The reader copies *preserving* type when the
 * source matches, and *converts* otherwise — see `copyDirectChunk`.
 */
export type DirectOutputBuffer =
  Float32Array | Float16Array | Uint8Array | Uint16Array | Uint32Array;

/**
 * Copy one fetched chunk into `output` at `destOffset`, preserving the
 * output's native dtype. Same-kind copies (and any numeric→numeric pair) go
 * through `TypedArray.set` (which converts when kinds differ — e.g. Float32
 * output from a Uint8 source). BigInt sources (int64/uint64 zarr arrays)
 * cannot be `set` into a numeric TypedArray, so they are widened element-wise
 * via `Number()`. Returns the number of elements written.
 */
export function copyDirectChunk(
  output: DirectOutputBuffer,
  data: RangeNumericArray,
  destOffset: number
): number {
  if (
    (typeof BigInt64Array !== 'undefined' && data instanceof BigInt64Array) ||
    (typeof BigUint64Array !== 'undefined' && data instanceof BigUint64Array)
  ) {
    for (let i = 0; i < data.length; i++) {
      (output as unknown as Record<number, number>)[destOffset + i] = Number(data[i]);
    }
    return data.length;
  }
  // Numeric source: `set` preserves dtype on a same-kind copy and converts
  // element-wise otherwise (Float32←Uint8, Uint8←Float32, Float16←…, etc.).
  (output as { set(a: ArrayLike<number>, offset: number): void }).set(
    data as ArrayLike<number>,
    destOffset
  );
  return data.length;
}

export function firstAxisRangeSlice(shape: readonly number[], range: LoadRange): zarr.Slice[] {
  return [slice(range.start, range.end), ...shape.slice(1).map(() => slice(null))];
}

/**
 * Precompute the per-range destination offsets (prefix sum) for loading
 * `ranges` in PARALLEL into one contiguous output buffer.
 *
 * Element counts are deterministic: each range spans `(end - start)` rows of
 * `prod(shape[1:])` elements, times `perElement` for encodings whose decoded
 * output is wider than the stored array (LUT row mode decodes k values per
 * stored index). Writes into `[offsets[i], offsets[i] + count_i)` are
 * disjoint by construction, so ranges may resolve in any order.
 */
export function rangeDestOffsets(
  shape: readonly number[],
  ranges: LoadRange[],
  perElement = 1
): { offsets: number[]; counts: number[]; total: number } {
  let rowSize = perElement;
  for (let i = 1; i < shape.length; i++) {
    rowSize *= shape[i];
  }
  const offsets = new Array<number>(ranges.length);
  const counts = new Array<number>(ranges.length);
  let total = 0;
  for (let i = 0; i < ranges.length; i++) {
    offsets[i] = total;
    counts[i] = (ranges[i].end - ranges[i].start) * rowSize;
    total += counts[i];
  }
  return { offsets, counts, total };
}

/**
 * Guard a parallel range write. With precomputed offsets each range owns a
 * fixed output span, so a decoded chunk LONGER than expected would overrun
 * into its neighbour — truncate it. A SHORTER chunk (malformed source)
 * leaves the tail of its span untouched (zero-filled on fresh buffers);
 * this mirrors the historical graceful-fallback behavior where mismatched
 * data degrades output but never throws. Both cases log a warning.
 */
export function clampRangeData<T extends { length: number; subarray(a: number, b: number): T }>(
  data: T,
  expected: number,
  rangeIndex: number,
  logModule: string
): T {
  if (data.length === expected) return data;
  log.warning(
    logModule,
    `Range ${rangeIndex} decoded ${data.length} elements, expected ${expected} — ` +
      (data.length > expected
        ? 'truncating to the range span'
        : 'leaving the remainder of the span unfilled')
  );
  return data.length > expected ? data.subarray(0, expected) : data;
}
