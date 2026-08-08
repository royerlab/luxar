# Picking Label Loaders

Lazy, per-element label fetchers for the picking / hover system. When the
user hovers an element (point, line, splat), these loaders fetch the
associated **text label** or **image label** for that single element from
zarr, on demand, and cache the result.

## Overview

Labels and image labels are stored per node as **CSR-style** pairs of
zarr arrays — an offsets array plus a concatenated bytes blob:

| Label kind | Offsets array         | Bytes array         | `.zattrs` flag     |
| ---------- | --------------------- | ------------------- | ------------------ |
| Text       | `label_offsets`       | `label_bytes`       | `has_labels`       |
| Image      | `image_label_offsets` | `image_label_bytes` | `has_image_labels` |

For element `i`, the payload is `bytes[offsets[i] : offsets[i+1]]`. An
empty entry (`offsets[i] === offsets[i+1]`) means "no label for this
element" and resolves to `null`. Both arrays use `uint64` offsets
(decoded as `BigUint64Array`) of length `N+1` and a `uint8` bytes array.

The two loaders differ in how aggressively they fetch:

- **`LabelLoader`** bulk-loads and UTF-8-decodes **every** label for a
  node on first hover, then serves all subsequent lookups from memory.
  That is usually cheap, but not always: labels are stored per element,
  so a node that tags all 168k of its vertices with one 107-byte tract
  name carries ~18 MB of raw CSR bytes over ~275 chunks. The first hover
  of such a node pays that fetch (a few tens of KB once compressed) and
  one synchronous decode pass; consecutive equal labels are decoded once
  (see Invariants), so the retained strings stay small.
- **`ImageLabelLoader`** is **truly lazy**: it bulk-loads only the
  (small) offsets array per node, then fetches each image's byte range
  on demand via a zarr slice. Image blobs can be hundreds of MB, so
  per-element fetching avoids pulling the whole `image_label_bytes`
  array into memory.

Both loaders coalesce concurrent requests (in-flight promise maps) so
repeated hovers over the same element/node share a single fetch.

## File Structure

```
picking/
├── label-loader.ts         # LabelLoader — bulk CSR text-label decode per node
└── image-label-loader.ts   # ImageLabelLoader — per-element image fetch + blob-URL LRU
```

## Components

### LabelLoader (`label-loader.ts`)

Loads and decodes the full set of UTF-8 text labels for a node on first
access, caching the decoded `string[]` per node path.

```typescript
import { LabelLoader } from './picking/label-loader';

const loader = new LabelLoader(store, rootLoc);

// First call loads + decodes all labels for the node; later calls hit the cache.
const label = await loader.getLabel('/points/cells', elementIndex); // string | null

// Available as a cheap guard from cached node .zattrs — but note that NOTHING
// in the viewer currently calls it: the pick-result handler asks for a label
// unconditionally, and an unlabelled node is kept quiet by the info-demotion in
// the loader's catch (below) rather than by this predicate.
if (loader.hasLabels(nodeAttrs)) {
  /* node has a label_offsets / label_bytes pair */
}

loader.dispose(); // clear caches + in-flight map
```

`getLabel` returns `null` when the node has no labels, the element's
label is empty, or the index is out of range. A failed zarr fetch
resolves to an empty label set rather than throwing. A node that simply
has no `label_offsets` array is the ordinary case — the picker calls
`getLabel` for whatever it hit, and most nodes are unlabelled — so that
is logged at info; everything else (decode failures, corrupt chunks, bad
metadata) still logs a warning (`Modules.SCENE_LOADER`).

### ImageLabelLoader (`image-label-loader.ts`)

Returns a blob URL for a single element's image label. Offsets are
loaded once per node; image bytes are fetched per element via
`zarr.get(bytesArr, [slice(start, end)])` (zarr fetches only the
overlapping chunks).

```typescript
import { ImageLabelLoader } from './picking/image-label-loader';

const loader = new ImageLabelLoader(store, rootLoc); // default 50 MB blob-URL cache
// const loader = new ImageLabelLoader(store, rootLoc, 100 * 1024 * 1024);

const url = await loader.getImageUrl('/points/cells', elementIndex); // string | null
if (url) imgElement.src = url;

if (loader.hasImageLabels(nodeAttrs)) {
  /* node has an image_label_offsets / image_label_bytes pair */
}

loader.dispose(); // revokes every cached blob URL, clears all caches
```

Decoded images are kept in an `LRUCache<CachedImage>` keyed by
`"<nodePath>:<elementIndex>"`, sized by image byte length (default
budget `50 MB`). On eviction the cache's `onEvict` callback calls
`URL.revokeObjectURL` so blob URLs never leak. The MIME type is sniffed
from the image's magic bytes — JPEG (`FF D8`), PNG (`89 50 4E 47`), and
WebP (`RIFF…WEBP`) are detected, with `image/png` as the fallback.

## Invariants

- **Offsets are `BigUint64Array`** (`uint64` zarr dtype). Offsets are
  converted to `Number` before slicing — fine for in-range byte offsets.
- **Empty entry ⇒ `null`.** `offsets[i] === offsets[i+1]` is the
  canonical "no label" sentinel for both loaders.
- **A consecutive run of equal labels is decoded once.** `LabelLoader`
  compares each element's bytes to the previous element's and reuses the
  string on a match, so a broadcast label (one value repeated across a
  whole node) costs one decode and one retained string instead of `N`.
  Only _consecutive_ runs collapse — a general pool would cost more than
  it saves on the common all-distinct case.
- **Path normalization.** A leading `/` on `nodePath` is stripped before
  `rootLoc.resolve(...)`, so both `/points/cells` and `points/cells`
  resolve identically.
- **Blob URLs are owned by the loader.** Callers must not
  `URL.revokeObjectURL` a returned URL — the LRU eviction path and
  `dispose()` own that lifecycle.
- **`dispose()` is terminal.** It clears all caches and in-flight maps;
  for `ImageLabelLoader` it also revokes every outstanding blob URL.

## See Also

- [`../README.md`](../README.md) — unified loader infrastructure
  (`SpatialQueryBuilder`, `RangeLoader`, tolerance).
- [`../../README.md`](../../README.md) — the Luxar Data package overview.
- [`../../../cache/README.md`](../../../cache/README.md) — `LRUCache`
  and the multi-level zarr chunk cache used elsewhere in the data layer.
