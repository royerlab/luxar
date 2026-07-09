/**
 * Shared SliceCache helpers for the three progressive loaders (GSplats, Points,
 * Lines). Keeps the per-slice cache key + snapshot cloning identical across
 * geometries so the loaders stay symmetric.
 *
 * @module data/loaders/progressive/slice-cache-helper
 */

/** The query fields that determine which elements a slice loads. */
export interface SliceViewLike {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
}

/**
 * Build the view signature used as the SliceCache key (namespaced per node by
 * the cache itself). Covers exactly the query determinants — displayDims,
 * slicePosition, tolerance — matching the progressive loaders' `viewStatesEqual`.
 * Nothing projection- or render-dependent enters the key: projection re-runs on
 * every hit, and a dataset content-hash change clears the whole cache.
 */
export function buildSliceViewSig(view: SliceViewLike): string {
  return JSON.stringify([view.displayDims, view.slicePosition, view.tolerance]);
}

/** Non-DataView ArrayBufferView (i.e. a TypedArray) with a `.slice()`. */
function isTypedArray(v: unknown): v is { slice(): unknown; byteLength: number } {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/**
 * Deep-copy a per-LOD decoded-data snapshot for storage in the SliceCache, and
 * report the total retained bytes.
 *
 * Cloning is REQUIRED: a loaded sub-LOD's typed arrays are views into the
 * spatial-index loader's REUSED accumulator buffer (see the spatial-index
 * loaders' "subarrays, zero copy" return). A cross-slice cache that aliased them
 * would be corrupted by the next load. Every own typed-array property is copied;
 * scalars / null (e.g. `splatCount`, `ndim`, absent `colors`) pass through.
 *
 * Generic over the geometry payload (LoadedGSplatsData / LoadedPointsData /
 * LoadedLinesData) so all three loaders share one implementation — it copies by
 * property shape, never by field name.
 */
export function cloneLodSnapshot<T extends object>(
  lods: readonly T[]
): {
  clone: T[];
  bytes: number;
} {
  let bytes = 0;
  const clone = lods.map((lod) => {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(lod)) {
      const value = (lod as Record<string, unknown>)[key];
      if (isTypedArray(value)) {
        out[key] = value.slice();
        bytes += value.byteLength;
      } else {
        out[key] = value;
      }
    }
    return out as T;
  });
  return { clone, bytes };
}
