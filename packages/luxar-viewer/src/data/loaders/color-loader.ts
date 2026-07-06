/**
 * Shared color-attribute helpers used by the points, lines, and gsplats
 * spatial-index loaders.
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
 * Shared by all three spatial-index loaders so the encode/decode
 * logic doesn't drift between Points, Lines, and GSplats.
 *
 * @module data/loaders/color-attribute-utils
 */

import * as zarr from '../zarr';
import { ArrayDecoder, type ArrayMetadata } from '../array-decoder/decoder';
import type { RangeLoader } from './spatial-query/range-loader';
import type { LoadRange } from './base-types';
import { clamp } from '../../utils/clamp';

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
      out[i] = Math.round(clamp(decoded[i], 0, 255));
    }
    return out;
  }

  if (UINT16_DTYPES.has(originalDtype)) {
    const out = new Uint16Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      out[i] = Math.round(clamp(decoded[i], 0, 65535));
    }
    return out;
  }

  return decoded;
}

/**
 * End-to-end load for a colors array, encoded or unencoded, with native-type
 * preservation and original_dtype restoration. Composes the helpers above to
 * reproduce the load-color flow the geometry loaders run (points, lines,
 * gsplats).
 *
 * Branches:
 * 1. Direct (unencoded, no array_ref): allocate or reuse a typed buffer of the
 *    right kind, stream raw values in via {@link RangeLoader.loadDirectTyped}.
 * 2. `rgb_uint8` / `rgb_uint16` encoded colors with a target buffer of the
 *    matching kind: skip the decode pipeline and stream raw bytes directly
 *    (downstream renderer normalizes 0-255 → 0-1).
 * 3. Anything else (quantized / LUT / broadcasted / array_ref): decode to
 *    Float32, optionally resolve array_ref against `zarrStore`, then cast
 *    back to `original_dtype` if the encoding records one.
 *
 * RGB layout (3 channels) is hard-coded to match the existing loaders.
 *
 * @param array - The colors zarr array.
 * @param ranges - Item ranges to load (splats / vertices etc).
 * @param rangeLoader - Shared encoding-dispatch loader.
 * @param zarrStore - Store used to resolve array_ref targets.
 * @param logPrefix - Caller tag for the array_ref resolution log line.
 * @param targetBuffer - Optional pre-allocated buffer (zero-allocation path).
 */
export async function loadColorRanges(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  ranges: ColorRange[],
  rangeLoader: RangeLoader,
  zarrStore: zarr.Readable,
  logPrefix: string,
  targetBuffer?: ColorBuffer
): Promise<ColorBuffer> {
  const totalItems = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  const totalElements = totalItems * 3; // RGB

  const attrs = array.attrs as unknown as ArrayMetadata;
  // Canonical encoded-ness check (covers quantized, LUT, broadcasted,
  // array_ref AND the per-channel family). The previous hand-rolled
  // three-way check missed per-channel encodings, so geolog_perchannel_u16
  // HDR colors fell into the "direct" branch and streamed RAW u16 CODES as
  // if they were SDR full-scale colors — every HDR scene rendered with
  // wrong (near-white) colors.
  const isEncoded = ArrayDecoder.isEncoded(attrs);
  const isArrayRef = ArrayDecoder.isArrayRef(attrs);

  // 1. Direct (unencoded) path — preserve native type. The unified
  // RangeLoader.loadDirectTyped reader copies into the natively-typed output
  // buffer (and sources the per-update abort signal internally).
  if (!isEncoded && !isArrayRef) {
    const dtype = String(array.dtype);
    const expectedType = getExpectedColorType(dtype);
    const output =
      targetBuffer && colorBufferTypeMatches(targetBuffer, expectedType)
        ? targetBuffer
        : allocateColorBuffer(totalElements, false, dtype);
    await rangeLoader.loadDirectTyped(array, ranges as LoadRange[], output);
    return output;
  }

  // 2. rgb_uint8 / rgb_uint16 with matching target → skip decode, raw stream.
  const encName = attrs.encoding?.name;
  if (encName === 'rgb_uint8' && targetBuffer instanceof Uint8Array) {
    await rangeLoader.loadDirectTyped(array, ranges as LoadRange[], targetBuffer);
    return targetBuffer;
  }
  if (encName === 'rgb_uint16' && targetBuffer instanceof Uint16Array) {
    await rangeLoader.loadDirectTyped(array, ranges as LoadRange[], targetBuffer);
    return targetBuffer;
  }

  // 3. Encoded or array_ref → decode to Float32 (resolving any ref), then
  //    restore original dtype using the CALLER attrs (not the target's), so
  //    an array_ref'd uint8 colors attribute still ends up uint8 even when
  //    the target storage is float32.
  //    Reuse the caller's Float32 target only when it actually FITS this
  //    load — a smaller buffer (e.g. an accumulator swapped to an exact-size
  //    buffer by an earlier load) would silently truncate the decode: the
  //    range writes past its end are dropped by TypedArray semantics and no
  //    error surfaces.
  const decodedFloat32 =
    targetBuffer instanceof Float32Array && targetBuffer.length >= totalElements
      ? targetBuffer
      : new Float32Array(totalElements);

  const shape = array.shape;
  const actualElementsPerItem = shape.length === 2 ? shape[1] : 1;

  await rangeLoader.loadRangesResolvingRef(
    array,
    attrs,
    ranges as LoadRange[],
    decodedFloat32,
    totalItems,
    actualElementsPerItem,
    zarrStore,
    logPrefix
  );

  return restoreOriginalDtype(decodedFloat32, attrs.encoding?.original_dtype, totalElements);
}
