/**
 * Shared color-attribute helpers used by the lines and gsplats spatial-index
 * loaders.
 *
 * Color arrays carry extra structure compared to position/scalar arrays:
 *
 *   - Native dtype matters at the GPU boundary. Uint8 colors are normalized
 *     0–255 → 0–1 by THREE.js, Float32 colors are read as-is. Round-tripping
 *     uint8 through float32 silently makes everything 255× too bright, so
 *     the loaders preserve native types where possible.
 *
 *   - Encoded color arrays (quantized / LUT / array_ref) carry an
 *     `original_dtype` attribute that records what shape the GPU expects.
 *     After decoding to Float32 we cast back to that type.
 *
 * These helpers used to be duplicated almost byte-for-byte between
 * `lines-spatial-index-loader.ts` and `gsplats-spatial-index-loader.ts` (only
 * the range parameter type was named differently). Extracting them keeps
 * future fixes — e.g. a new dtype string variant — in one place.
 *
 * @module data/loaders/color-attribute-utils
 */

import * as zarr from 'zarrita';
import { get, slice } from 'zarrita';

/** Minimal shape needed by the helpers — both SplatRange and SegmentRange match. */
export interface ColorRange {
  start: number;
  end: number;
}

/** Concrete typed-array union used by every color path in the codebase. */
export type ColorBuffer = Float32Array | Uint8Array | Uint16Array;

/** Tag used to compare expected vs actual buffer kind without reflection. */
export type ColorBufferKind = 'Float32Array' | 'Uint8Array' | 'Uint16Array';

const UINT8_DTYPES = new Set(['uint8', '|u1', '<u1', '>u1']);
const UINT16_DTYPES = new Set(['uint16', '|u2', '<u2', '>u2']);

/**
 * Allocate a color buffer matching `dtype`. Encoded sources always go
 * through Float32 (decoding produces floats); for unencoded sources we
 * preserve the native uint8/uint16 type so THREE.js's normalization runs
 * correctly downstream.
 */
export function allocateColorBuffer(
  totalElements: number,
  isEncoded: boolean,
  dtype: string
): ColorBuffer {
  if (isEncoded) return new Float32Array(totalElements);
  if (UINT8_DTYPES.has(dtype)) return new Uint8Array(totalElements);
  if (UINT16_DTYPES.has(dtype)) return new Uint16Array(totalElements);
  return new Float32Array(totalElements);
}

/**
 * Map a zarr dtype string to the concrete TypedArray kind used at the GPU
 * boundary. Used by the accumulator-aware paths to decide whether a passed-in
 * target buffer can be reused or whether a fresh one needs allocation.
 */
export function getExpectedColorType(dtype: string): ColorBufferKind {
  if (UINT8_DTYPES.has(dtype)) return 'Uint8Array';
  if (UINT16_DTYPES.has(dtype)) return 'Uint16Array';
  return 'Float32Array';
}

/** True when `buffer` is an instance of the kind named by `expected`. */
export function colorBufferTypeMatches(buffer: ColorBuffer, expected: ColorBufferKind): boolean {
  if (expected === 'Uint8Array') return buffer instanceof Uint8Array;
  if (expected === 'Uint16Array') return buffer instanceof Uint16Array;
  return buffer instanceof Float32Array;
}

/**
 * Stream unencoded color ranges directly into `output`, preserving the input
 * data's typed-array kind whenever it matches the output (no value conversion,
 * no buffer reinterpretation). When the kinds disagree we widen through
 * Float32, which is the correct fallback for the few legacy datasets where the
 * stored dtype no longer matches what the loader allocated.
 *
 * RGB layout is hard-coded (3 channels per item) to match the existing
 * loaders; widen this if a future dataset uses RGBA.
 */
export async function loadDirectColorRanges(
  array: zarr.Array<zarr.DataType, zarr.FetchStore>,
  ranges: ColorRange[],
  output: ColorBuffer
): Promise<void> {
  let destOffset = 0;
  const shape = array.shape;

  for (const range of ranges) {
    const sliceSpec: zarr.Slice[] =
      shape.length === 2
        ? [slice(range.start, range.end), slice(null)]
        : [slice(range.start, range.end)];

    const chunkData = await get(array, sliceSpec);
    const data = chunkData.data;

    if (output instanceof Float32Array && data instanceof Float32Array) {
      output.set(data, destOffset);
    } else if (output instanceof Uint8Array && data instanceof Uint8Array) {
      output.set(data, destOffset);
    } else if (output instanceof Uint16Array && data instanceof Uint16Array) {
      output.set(data, destOffset);
    } else {
      const float32Data =
        data instanceof Float32Array ? data : new Float32Array(data as ArrayLike<number>);
      (output as Float32Array).set(float32Data, destOffset);
    }

    destOffset += (range.end - range.start) * 3;
  }
}

/**
 * Cast a freshly-decoded Float32 color buffer back to its `original_dtype`.
 * Returns the input buffer unchanged when no original dtype was recorded
 * (e.g. natively-float32 colors).
 *
 * Clamps to [0, 255] for uint8 and [0, 65535] for uint16 — the same clamp
 * the inline copies of this function applied. Out-of-range values can occur
 * when an encoded color crosses the dequantization edge by a fractional
 * unit; clamping is the documented behavior.
 */
export function restoreOriginalDtype(
  decoded: Float32Array,
  originalDtype: string | undefined,
  totalElements: number
): ColorBuffer {
  if (!originalDtype) return decoded;

  if (UINT8_DTYPES.has(originalDtype)) {
    const out = new Uint8Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      out[i] = Math.round(Math.max(0, Math.min(255, decoded[i])));
    }
    return out;
  }

  if (UINT16_DTYPES.has(originalDtype)) {
    const out = new Uint16Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      out[i] = Math.round(Math.max(0, Math.min(65535, decoded[i])));
    }
    return out;
  }

  return decoded;
}
