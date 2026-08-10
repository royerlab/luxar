# labels

Private compiler support for serializing **per-element labels** — both string
labels and image labels — into a Luxar Zarr archive using CSR-style storage.

## Overview

Points, lines, and gsplats can each carry one optional string label and one
optional image per element. Rather than store these as fixed-width arrays
(wasteful, since most labels are short or absent), this module packs them into
two arrays per label kind:

- an **offsets** array (`uint64`, shape `(N+1,)`) giving the byte offset of each
  element's payload, and
- a **bytes** array (`uint8`) holding the concatenated payloads.

Element `i` is recovered as `bytes[offsets[i]:offsets[i+1]]`. Empty / null
entries collapse to `offsets[i] == offsets[i+1]`, costing nothing in the bytes
array. This is the classic compressed-sparse-row (CSR) layout.

These functions are called by `LuxarZarrCompiler` (see `io/compiler.py`) after
the primary geometry arrays are written, once per data node that supplies
labels. The empty `__init__.py` keeps the package import-free; callers import
the writers by their fully-qualified submodule path.

## File structure

```
labels/
├── __init__.py        # empty — package marker only
├── text_labels.py     # write_labels_csr — UTF-8 string labels
└── image_labels.py    # normalize_image_label + write_image_labels_csr
```

## API

### `text_labels.write_labels_csr(group, labels, n_elements, compressor, sort_order=None)`

Encode a sequence of strings (one per element) as UTF-8 and write the CSR pair:

- `label_offsets` — `uint64` `(N+1,)`, written with the scene's default
  `compressor`.
- `label_bytes` — `uint8`, concatenated UTF-8 strings, also compressed.

Sets `group.attrs["has_labels"] = True`. Raises `ValueError` if
`len(labels) != n_elements`. Both arrays are chunked at 65536 elements.

### `text_labels.write_ladder_union_labels_csr(group, level_labels, level_sort_orders, n_elements, compressor)`

Write ONE CSR pair on the **parent** node of an additive-LOD ladder, describing
the ladder's committed **union**. Concatenates each level's labels in that
level's stored order (applying its `sort_order` when not `None`) and delegates to
`write_labels_csr(group, union, n_elements, compressor, None)` — the union is
already in final order, so no further permutation is applied.

Raises `ValueError` if `level_labels` and `level_sort_orders` differ in length;
a union/total mismatch is caught by `write_labels_csr`'s own length check.

### `text_labels.validate_ladder_labels(levels, positions_key) -> bool`

Pre-write gate for a ladder's labels, returning whether the ladder is labelled at
all. **Pure** (reads only `levels`), so the multi-LOD writers call it BEFORE
`require_group` — a rejected ladder must not leave an empty node behind. Enforces
all-or-nothing presence across levels (the error names the first unlabelled level)
and each level's label count against that level's own element count. The length
check is skipped when any level's element array is not `(N, D)` — that fault
belongs to the per-level writer's positions validator, which names it properly.
`positions_key` is `"positions"` for Points and `"vertices"` for Lines.

### `image_labels.write_image_labels_csr(group, image_labels, n_elements, compressor, sort_order=None)`

Encode per-element images as a CSR pair:

- `image_label_offsets` — `uint64` `(N+1,)`, written with the scene's default
  `compressor` (small, compressible).
- `image_label_bytes` — `uint8`, concatenated encoded image blobs, written with
  **no compressor** (the blobs are already JPEG/WebP/PNG) and chunked at 1 MiB.

Sets `group.attrs["has_image_labels"] = True`. Accepts either a dense sequence
(length must equal `n_elements`) or a sparse `Dict[int, item]` where missing
indices become empty blobs. Out-of-range dict indices raise `ValueError`.

### `image_labels.normalize_image_label(item) -> bytes`

Convert a single heterogeneous image input into encoded image bytes:

| Input type | Handling |
|------------|----------|
| `None` | `b""` (empty) |
| `bytes` / `bytearray` | used as-is (pre-encoded blob) |
| `pathlib.Path` / `str` | file read as raw bytes |
| `PIL.Image.Image` | encoded to WebP (quality 85) |
| `numpy.ndarray` `(H,W)`, `(H,W,3)`, `(H,W,4)` uint8 | via PIL → WebP (quality 85) |

Raises `ImportError` (with an install hint) if Pillow is needed but missing, and
`TypeError` / `ValueError` for unsupported types or array shapes.

## The `sort_order` argument

When a data node is spatially reordered (Morton/Hilbert ordering), its labels
must be permuted to stay aligned with the reordered geometry. Both writers
accept an optional index array and apply `labels[i] for i in sort_order` before
building the CSR arrays. The correct array depends on geometry type:

| Geometry | `sort_order` source |
|----------|---------------------|
| Points   | `ordering_data["sort_order"]` |
| Lines    | `ordering_data["vertex_sort_indices"]` |
| GSplats  | `ordering_data["sort_order"]` |

### Additive-LOD ladders: one union CSR on the parent

An additive ladder (`additive_lod=` on `add_points` / `add_lines`) stores its
geometry in `additive_<i>/` subgroups, but the viewer's loader concatenates the
levels it has loaded into a **single buffer** — no one level's array is what a
pick index addresses. So the label CSR lives on the **parent** ladder node (which
therefore carries `has_labels`) and the `additive_<i>` subgroups carry **no** label
arrays at all.

Index `k` of the parent CSR is the `k`-th element of the concatenation
`additive_0 || additive_1 || …` (coarsest → finest), each level in its own
**stored** (spatially reordered) order — the same on-disk index space a flat
labelled leaf's CSR uses, just spanning the levels.
`write_ladder_union_labels_csr` builds it; the multi-LOD writers recover each
level's permutation via the private `_return_sort_order` forwarding flag on
`write_points` / `write_lines` (the permutation is not persisted on disk).

The viewer commits levels coarsest-first, so a fully-loaded ladder maps straight
through — index `k` is committed slot `k`. The **committed buffer** is not in
general a prefix of this union, though: the per-level loader compacts out elements
culled by the current nD slice and fetches only the chunk ranges a query
intersects, so slots shift. For **Points** the viewer corrects that shift: it
composes each level's own visible-slot → on-disk-index map (the map issue #1421 /
PR #1425 introduced for **flat** nodes) into this union index space, offsetting
level `i` by the preceding levels' on-disk counts, so a labelled Points ladder
resolves exactly under an nD slice too (issue #1439) — falling back to the raw
committed slot only where the levels' own metadata is inconsistent. For **Lines**
no map is composed across the levels, so a laddered lines node still resolves at
the raw committed slot (exact for a fully-loaded 3D scene, shifted otherwise).

Under the `partition=`-outer + `additive_lod=`-inner composition the CSR lands on
each `part_<i>` ladder parent — which is exactly where the viewer looks. Since
#1415 / PR #1420 the label lookup path is the hit LEAF scene node
(`result.mainNode.name`) and a laddered part's scene node IS its ladder parent; the
outermost `kind=partition` wrapper is the *reported* path only. So that composition
resolves too, through the same per-geometry path as an unpartitioned ladder.

For Lines the CSR is per-**vertex**, matching the flat Lines writer, while the
viewer's Lines pick id is a per-**segment** storage slot. Issue #1424 supplied the
missing segment→vertex indirection for **flat** lines nodes — the picked segment's
slot resolves back to that segment's start vertex row in the stored ordering — but
through the same visible-slot → on-disk-index map a **lines** ladder does not publish,
so across a lines ladder the hover only lands on the right string when every element
carries the same one (`labels` has no broadcast form — it is always one entry per
element). #1439 carried that map over the levels for Points only.

Labels are all-or-nothing across a ladder — a partially-labelled ladder cannot
produce a correct union, so `validate_ladder_labels` rejects it.

## Usage

These are internal helpers; in practice you pass `labels=` / `image_labels=` to
the compiler's `write_points` / `write_lines` / `write_gsplats` methods. Direct
use mirrors what the compiler does:

```python
import zarr
from luxar.io._compiler.labels.text_labels import write_labels_csr
from luxar.io._compiler.labels.image_labels import write_image_labels_csr

group = zarr.open_group("scene.luxar.zarr/points/cloud", mode="a")

write_labels_csr(
    group,
    labels=["nucleus", "", "spindle"],   # "" → null label
    n_elements=3,
    compressor=compiler.compressor,
    sort_order=ordering_data["sort_order"],
)

write_image_labels_csr(
    group,
    image_labels={0: "thumb0.png", 2: pil_image},  # sparse; index 1 empty
    n_elements=3,
    compressor=compiler.compressor,
)
```

## Dependencies

**Internal:**
- `luxar.typing_utils.protocols.CompressorProtocol` — type of the compressor argument

**External:**
- `numpy` — CSR array construction
- `zarr` — dataset creation
- `arbol` — `aprint` progress logging
- `Pillow` — *optional*, only for encoding `PIL.Image` / `numpy.ndarray` image labels

## See Also

- [io/README.md](../../README.md) — I/O operations and the compiler
- [core/README.md](../../../core/README.md) — data nodes that carry labels
- [typing_utils/README.md](../../../typing_utils/README.md) — `CompressorProtocol`
