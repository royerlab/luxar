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

## Usage

These are internal helpers; in practice you pass `labels=` / `image_labels=` to
the compiler's `write_points` / `write_lines` / `write_gsplats` methods. Direct
use mirrors what the compiler does:

```python
import zarr
from luxar.io._compiler.labels.text_labels import write_labels_csr
from luxar.io._compiler.labels.image_labels import write_image_labels_csr

group = zarr.open_group("scene.zarr/points/cloud", mode="a")

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
