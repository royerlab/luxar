/**
 * Optional-array convenience: open a sibling zarr array, read its attrs,
 * decode via `ArrayDecoder`, and return the float buffer. Returns `null`
 * when the array is missing — used by loaders that need to load an
 * attribute (radii, sharpness, …) only when it's present.
 *
 * Internal helper — not part of the public API surface of `data/`.
 */

import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import { ArrayDecoder } from './decoder';
import type { ArrayMetadata } from './types';

/**
 * Convenience function: Load and decode an optional array
 *
 * @param location - Zarr location
 * @param arrayName - Name of array (e.g., 'colors', 'radii')
 * @param decoder - ArrayDecoder instance
 * @param expectedElements - Expected total elements
 * @returns Decoded array or null if not present
 *
 * @internal — used by loader internals; not part of the public API.
 */
export async function loadAndDecodeOptionalArray(
  location: zarr.Location<zarr.Readable>,
  arrayName: string,
  decoder: ArrayDecoder,
  expectedElements?: number
): Promise<Float32Array | null> {
  try {
    // Try to open array
    const array = await zarr.open(location.resolve(arrayName), { kind: 'array' });

    // Load attributes
    const attrs = array.attrs as unknown as ArrayMetadata;

    // Decode
    const decoded = await decoder.decode(array, attrs, expectedElements);

    log.success(Modules.ZARR_LOADER, `Loaded ${arrayName}: ${decoded.length} elements`);

    return decoded;
  } catch {
    // Array doesn't exist (this is OK for optional arrays)
    return null;
  }
}
