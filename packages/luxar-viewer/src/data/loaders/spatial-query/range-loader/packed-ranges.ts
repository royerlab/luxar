/**
 * Read every range of one attribute, then pack them into ONE buffer so the
 * decode runs as a single worker call instead of one call per range.
 *
 * Every worker-decoded range encoding (per-channel, quantized, log/geolog
 * scalar, LUT) decodes element-wise: element `j` of the output depends only on
 * stored code `j` (and, for per-channel, on its GLOBAL column `g % cols`). So
 * decoding the concatenation is identical to decoding each range on its own,
 * provided each range lands at its own destination offset — which also keeps
 * the per-channel column phase exact (`g` is the packed index itself).
 *
 * The pack is laid out in OUTPUT order, in STORED units (`offsets / perElement`,
 * since LUT row mode decodes `k` values per stored index). A malformed range
 * that decodes SHORT leaves a gap in the pack; the gap decodes to junk that
 * {@link scatterDecoded} never copies, so its output span stays untouched —
 * the same "remainder unfilled" contract `clampRangeData` gives a per-range
 * write. A LONG range is truncated to its span, again as before.
 *
 * Measured motivation: the per-range dispatch sent ~40k worker messages per
 * 6 s loop on a Points timelapse, most a few hundred elements each.
 *
 * @module data/loaders/spatial-query/range-loader/packed-ranges
 */

import * as zarr from '../../../zarr';
import { readArray, abortOptions } from '../../../zarr';
import type { LoadRange } from '../../base-types';
import { clampRangeData, firstAxisRangeSlice, type RangeNumericArray } from './encoding-types';

/** A range pack: the packed stored codes and each range's filled OUTPUT length. */
export interface PackedRanges {
  /** Stored codes of every range, at `offsets[i] / perElement`. */
  packed: RangeNumericArray;
  /** Output elements each range actually fills (`< counts[i]` for a short range). */
  filled: number[];
  /** True when every range filled its whole span (the pack has no gaps). */
  complete: boolean;
}

/** Destination layout of a pack (from `rangeDestOffsets`). */
export interface PackLayout {
  offsets: number[];
  counts: number[];
  total: number;
  /** Decoded output elements per stored element (LUT row mode: `k`). */
  perElement: number;
  logModule: string;
}

/** Read every range CONCURRENTLY (network concurrency is bounded by the fetch gate). */
export function readRanges(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  ranges: LoadRange[],
  signal: AbortSignal | null | undefined
): Promise<RangeNumericArray[]> {
  const shape = array.shape;
  return Promise.all(
    ranges.map(
      async (range) =>
        (await readArray(array, firstAxisRangeSlice(shape, range), abortOptions(signal)))
          .data as RangeNumericArray
    )
  );
}

/**
 * Pack `parts` into one buffer of the first part's type. A single complete
 * part is returned as-is (no copy).
 */
export function packRanges(parts: RangeNumericArray[], layout: PackLayout): PackedRanges {
  const { offsets, counts, total, perElement, logModule } = layout;
  const clamped = parts.map(
    (part, i) =>
      clampRangeData<{ length: number; subarray(a: number, b: number): RangeNumericArray }>(
        part,
        counts[i] / perElement,
        i,
        logModule
      ) as RangeNumericArray
  );
  const filled = clamped.map((part) => part.length * perElement);
  const complete = filled.every((n, i) => n === counts[i]);
  if (complete && clamped.length === 1) return { packed: clamped[0], filled, complete };

  const Ctor = (parts[0]?.constructor ?? Uint8Array) as new (n: number) => RangeNumericArray;
  const packed = new Ctor(total / perElement);
  const setter = packed as unknown as { set(a: RangeNumericArray, o: number): void };
  clamped.forEach((part, i) => setter.set(part, offsets[i] / perElement));
  return { packed, filled, complete };
}

/**
 * Copy a decoded pack into `output`: the whole pack at once when it has no
 * gaps, else only each range's filled span (a short range's tail stays as-is).
 */
export function scatterDecoded(
  decoded: Float32Array,
  output: Float32Array,
  pack: PackedRanges,
  offsets: number[]
): void {
  if (pack.complete) {
    // Ranges are laid out from 0 with no gaps, exactly as the per-range writes were.
    output.set(decoded);
    return;
  }
  pack.filled.forEach((n, i) => {
    output.set(decoded.subarray(offsets[i], offsets[i] + n), offsets[i]);
  });
}
