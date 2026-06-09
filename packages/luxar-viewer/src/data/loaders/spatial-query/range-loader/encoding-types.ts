import { Modules } from '../../../../utils/log';
import * as zarr from '../../../zarr';
import { slice } from '../../../zarr';
import type { LoadRange } from '../../base-types';

export type EncodingType = 'broadcasted' | 'quantized' | 'lut' | 'array_ref' | 'direct';

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
  | Float32Array
  | Float16Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

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
