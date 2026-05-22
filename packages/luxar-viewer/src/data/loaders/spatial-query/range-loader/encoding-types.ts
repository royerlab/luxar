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

export function numericArrayToFloat32(data: RangeNumericArray): Float32Array {
  if (data instanceof Float32Array) return data;
  if (typeof BigUint64Array !== 'undefined' && data instanceof BigUint64Array) {
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = Number(data[i]);
    return result;
  }
  if (typeof BigInt64Array !== 'undefined' && data instanceof BigInt64Array) {
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = Number(data[i]);
    return result;
  }
  return new Float32Array(data as ArrayLike<number>);
}

export function firstAxisRangeSlice(shape: readonly number[], range: LoadRange): zarr.Slice[] {
  return [slice(range.start, range.end), ...shape.slice(1).map(() => slice(null))];
}
