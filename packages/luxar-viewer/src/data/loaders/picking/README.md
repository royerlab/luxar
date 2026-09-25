# Picking Label Loaders

Lazy, per-element label fetchers for the picking / hover system. When the
user hovers an element (point, line, splat), these loaders fetch the
associated **text label**, **machine-readable key**, or **image label** for that single element from
zarr, on demand, and cache the result.

## Overview

Labels, keys, and image labels are stored per node as **CSR-style** pairs of
zarr arrays — an offsets array plus a concatenated bytes blob:

| Label kind | Offsets array         | Bytes array         | `.zattrs` flag     |
| ---------- | --------------------- | ------------------- | ------------------ |
| Text       | `label_offsets`       | `label_bytes`       | `has_labels`       |
| Key        | `key_offsets`         | `key_bytes`         | `has_keys`         |
| Image      | `image_label_offsets` | `image_label_bytes` | `has_image_labels` |

For element `i`, the payload is `bytes[offsets[i] : offsets[i+1]]`. An
empty entry (`offsets[i] === offsets[i+1]`) means "no label for this
element" and resolves to `null`. Both arrays use `uint64` offsets
(decoded as `BigUint64Array`) of length `N+1` and a `uint8` bytes array.

The two loaders choose different offset strategies based on their payload/index
ratio:

- **`LabelLoader`** slice-reads two offsets and the exact UTF-8 byte range
  for one hovered element. It retains open array handles per node and keeps
  recently decoded labels in a shared 1 MB LRU, so first-hover work and
  retained text no longer scale with the node's element count.
- **`ImageLabelLoader`** bulk-loads the comparatively small offsets array per
  node, then fetches each image's byte range on demand via a zarr slice. Image
  blobs can be hundreds of MB, so retaining the index avoids repeated offset
  reads without pulling the whole `image_label_bytes` array into memory.

The Python writer in
`packages/luxar/src/luxar/io/_compiler/labels/text_labels.py` chunks text
offsets and bytes at 65,536 entries/bytes. Together with spatial element
ordering, that keeps `LabelLoader` hover reads bounded by nearby chunks rather
than whole-node arrays.

Both loaders coalesce concurrent requests (in-flight promise maps) so
repeated hovers over the same element/node share a single fetch.

## File Structure

```
picking/
├── label-loader.ts         # LabelLoader — per-element CSR text fetch + LRU
└── image-label-loader.ts   # ImageLabelLoader — per-element image fetch + blob-URL LRU
```

## Components

### LabelLoader (`label-loader.ts`)

Loads and decodes one UTF-8 text label at a time. Array handles are cached
per node; decoded labels share a byte-bounded LRU across nodes.

```typescript
import { LabelLoader } from './picking/label-loader';

const loader = new LabelLoader(rootLoc);
const keyLoader = new LabelLoader(rootLoc, 'keys');

// Each call fetches one label; recently decoded labels hit the bounded LRU.
const label = await loader.getLabel('/points/cells', elementIndex); // string | null

// Available as a cheap guard from cached node .zattrs — but note that NOTHING
// in the viewer currently calls it: the pick-result handler asks for a label
// unconditionally, and an unlabelled node is kept quiet by the info-demotion in
// the loader's catch (below) rather than by this predicate.
if (loader.hasLabels(nodeAttrs)) {
  /* node has a label_offsets / label_bytes pair */
}

if (keyLoader.hasLabels(nodeAttrs)) {
  /* node has a key_offsets / key_bytes pair, stamped by has_keys */
}

loader.dispose(); // clear array handles, decoded labels, and in-flight maps
```

`getLabel` returns `null` when the node has no labels, the element's
label is empty, or the index is out of range. A failed zarr fetch
resolves to `null` rather than throwing. An absent channel, including a
missing `label_bytes` array next to present offsets, is remembered for the
session; other open or chunk-read failures are not cached, so a later hover
retries them. A node whose
`label_offsets` array is simply absent is the ordinary case — the picker
calls `getLabel` for whatever it hit, and most nodes are unlabelled — so
**that one open** is logged at info. Everything after it still logs a
warning (`Modules.SCENE_LOADER`): a missing `label_bytes` next to
present offsets is a corrupt store rather than an unlabelled node, and so
are missing chunks, decode failures and bad metadata.

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
- **Repeated labels stay bounded.** `LabelLoader` never materialises a
  whole run: it decodes only hovered elements, and the shared LRU caps
  retained text even when every label is distinct.
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
