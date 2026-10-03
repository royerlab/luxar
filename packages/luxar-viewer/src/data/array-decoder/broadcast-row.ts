import * as zarr from '../zarr';
import { readArray, abortOptions } from '../zarr';
import type { EncodingMetadata } from './types';

/**
 * The stored row of a `broadcasted` array, as float32.
 *
 * The writer repeats the row in `encoding.value`, which arrives with the
 * consolidated metadata, so it costs no request. That matters because the row
 * is a few bytes but its read is a whole request per array per node (per rung
 * of a ladder), and an all-fill row has no chunk at all — zarr skips writing a
 * chunk equal to the fill value — so the read is a 404. A store without `value`
 * reads the row. Both paths give the same float32 values: a float32 row is
 * exact through JSON, an integer row widens either way.
 */
export async function readBroadcastRow(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  encoding: EncodingMetadata | undefined,
  signal?: AbortSignal | null
): Promise<Float32Array> {
  if (encoding?.value) return Float32Array.from(encoding.value);
  const raw = (await readArray(array, undefined, abortOptions(signal))).data;
  // Changing this cast requires re-deriving the writer's
  // positive_scalar_round_trip_slack chunk-bound allowance.
  return raw instanceof Float32Array ? raw : new Float32Array(raw as ArrayLike<number>);
}
