# luxar.gsplats.io

I/O operations for persisting and loading Gaussian splat data in the `.gsplats.zarr` format (**format v3.3** — a detached scene-node subtree: leaf / kind=lod / kind=partition, nestable freely). v3.3 permits the optional `luxar_delta_v1` filter on quantized code arrays; v3.2 renamed the `kind=lod` selector attrs to `selector: "coverage"` / per-child `coverage_fraction`; v3.1 split the Cholesky factors into `cholesky_factors_diag` + `cholesky_factors_offdiag`. v3.0-v3.2 files remain readable.

## Purpose

This package provides functions to save and load fitted Gaussian splat results with:
- **Spatial ordering** (Morton/Hilbert curves) for better compression
- **Semantic encoding** (quantization, broadcasting, LUT) via `luxar.encoding`
- **Chunk-based spatial indexing** for efficient queries
- **Provenance tracking** for reproducibility
- **Fitter-agnostic metadata** allowing different fitting implementations
- **Node-tree LOD layout** (leaf / kind=lod / kind=partition, freely nestable);
  the trivial single-splat-set case is a bare leaf
- **Legacy migration** (v1.0 / v1.1 / pre-v2.0 substitutive directory / v2.0 matrix,
  plus v3.0/v3.1 stores with pre-v3.2 `pixel_size` lod selector attrs → v3.3)

## Main Functions

### Save/Load Functions

**`save_gsplats()`** - Save splats to .gsplats.zarr

```python
from luxar.gsplats.io import save_gsplats
from luxar.encoding import EncodingMode

save_gsplats(
    path="fitted.gsplats.zarr",
    centers=centers,              # (N, d) float32
    amplitudes=amplitudes,        # (N,) float32
    cholesky_factors=cholesky,    # (N, d*(d+1)//2) float32
    colors=colors,                # (N, 3) float32/uint8 (optional)
    label_ids=label_ids,          # (N,) non-negative int class ids (optional)
    label_vocabulary={0: "bone"}, # id -> name; must name every id present
    ordering="hilbert",           # "morton", "hilbert", or "none"
    encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
    fitting_info={"time_seconds": 45.3, "iterations": 850},
    fitting_config=None,          # Fitter-specific config (optional)
    provenance_info=None,         # Image lineage metadata (optional)
    description="DAPI nuclei fitting",
    compress=None,                # "zip" or "tar.gz" for compressed archive
    compressor=DEFAULT_COMP,      # Width-aware default (zstd l9, shuffle by dtype)
    zip_deflate=False,            # Use DEFLATE compression for outer zip
    truncation_radius=2.75,       # Gaussian truncation radius (std devs)
)
```

**`write_flat_leaf_streaming()`** writes one flat leaf from metadata plus a
single-pass stream of decoded splat sets, using disk-backed staging so only one
decoded leaf (all its additive rungs) plus one set's local ordering copy are
resident.

**`load_gsplats()` / `load_default_gsplats()`** - Load splats from .gsplats.zarr

Transparently handles compressed formats (`.gsplats.zarr.zip`, `.gsplats.zarr.tar.gz`) by extracting to a temporary directory automatically (via the shared, hardened `_archive.extract_compressed_zarr` — it rejects links/devices, validates every member before extracting, and caps member count / total size to guard against path-traversal and archive-bomb attacks). Arrays are decoded from their stored encoding (quantization, broadcasting, etc.) to float32 — `label_ids` excepted, which decodes back to its stored unsigned integer dtype (a class id is exact, never a float).

`load_gsplats()` requires a flat/matrix-shaped tree. `load_default_gsplats()`
also accepts partition and nested trees by materializing their default-rendered
selection into one in-memory `GSplatData`.

The store root is resolved from the extracted tree in three tiers: a top-level `*.gsplats.zarr` directory (what a compressed save writes); else the archive ROOT itself when a zarr group document (`zarr.json` / `.zgroup`) sits at depth 0 — the *flat* shape `zip -r x.gsplats.zarr.zip .` from inside a store produces; else the first top-level directory whatever it is called (a stray depth-0 file beside it — a `README.md`, a `.DS_Store` — must not decide the outcome, and `iterdir()` order is not a decision). The named directory deliberately keeps winning over the flat reading, matching `_zip_is_flat_store`. A flat tree is moved one level down under a store-shaped name so the returned path's parent is still a removable temp directory, which is the contract every caller relies on.

Ranking the flat tier above the directory tier reversed one shape: an archive that carries a depth-0 group document *and* the real store one level down under a non-`*.gsplats.zarr` name (`zip -r x.gsplats.zarr.zip .` from inside a parent zarr group that merely contains the store) used to fall through to the directory and load, and now resolves to the wrapper and fails loudly with `Invalid format_type: None`. The order is nonetheless right: nothing in an archive tells "a store whose root has one child group" apart from "a wrapper around a store", and gating the flat tier on "the sole child is not itself a group" would break the flat partition that is #1628's own repro. Re-archive the inner directory on its own to read such a wrapper.

Its read-only sibling `_archive.read_archive_root_attrs` extracts *nothing*: it scans the archive index (zip central directory / tar headers) for the store root's metadata document — `.zattrs` at zarr format 2, `zarr.json` at format 3 — reads that one member's bytes, and returns the attrs (unwrapping them out of the node document at format 3). Only that one payload is ever *read*, but only a zip has a real index: a gzipped tar's headers are walked lazily and the walk stops at the first top-level `*.gsplats.zarr/` root document. How early that stop comes depends on the on-disk format, because `tarfile.add` walks a directory in sorted order: a format-2 store's `.zattrs` is a dotfile and lands second (measured: member 1 of 24), so that layout really does cost a couple of headers, but a format-3 store's `zarr.json` sorts *after* every array sub-directory and lands last (measured: member 26 of 27) — so on the format Luxar writes by default the stop effectively never fires and the peek pays a full inflate. The other two layouts get no such stop (a later member could still outrank the candidate, and both the flat and the sole-top-level-directory rules need the whole member list), and reaching the end of a gzip stream means inflating it, so *those* shapes cost one decompression pass. The member is budgeted by its document *name*, and the two budgets are far apart on purpose: a `.zattrs` **is** the attributes mapping and keeps a small 4 MiB cap, while a format-3 `zarr.json` at a consolidated root also carries the entire consolidated index of the tree (one entry per *node*: measured ~8-10 KB per part for bare leaves, ~48-54 KB with a 6-step `stream` ladder, ~100-110 KB for an `adaptive`-shaped part) and gets 128 MiB; a member whose bytes actually exceeded 4 MiB — only ever a `zarr.json` read under the raised budget — has the attributes it unwraps to re-capped at 4 MiB, so raising the document budget cannot raise what the peek hands back. Any size refusal emits a `UserWarning` naming the archive, member, measured size and budget — `{}` is indistinguishable from "this dataset authored no appearance", so a *silent* refusal would reach the user only as a rebuild that quietly reset the look. It answers `{}` quietly when there is no root metadata document to read (and for a missing path or a non-archive file), but a *corrupt* archive raises — `BadZipFile`, `tarfile.ReadError`, `json.JSONDecodeError` — and it is `load_gsplats.read_authored_appearance` that absorbs those into `{}` for its best-effort carry. The store root is resolved as the same *kind* of node `extract_compressed_zarr` picks, tier for tier — the top-level `*.gsplats.zarr` directory; else the archive root itself, **only when a zarr group document sits at depth 0 and no top-level `*.gsplats.zarr` directory is present** (the flat shape); else, **only when it is the sole top-level directory**, that directory whatever it is named — which for any archive holding one store (every archive a compressed save writes) is the very node the extractor loads. Those are `_zip_is_flat_store`'s rules exactly, all three, so the peek and the extractor classify the same archives as flat — including the *bare* `x.gsplats.zarr/` directory entry `zip -r` emits for an empty subdirectory, which is tracked separately from the top-level-directory set because the sole-directory tier deliberately ignores it (an empty directory holds no store, so counting it there would manufacture false ambiguity). The group-document condition on the flat tier is what keeps a stray depth-0 `.zattrs` (a name that also sits beside an array) from outranking a real store one directory down, and the sole-directory tier is refused outright once the archive is flat, so a flat store that authored no root attrs carries nothing rather than an array sub-directory's attrs. A store carrying *both* format documents at its root — a half-finished in-place migration — is settled here by archive order, so this index-only peek can answer the stale format-2 view where `open_group` sees format 3; a known limitation rather than a contract, and no Luxar writer produces the state. An archive holding *several* `*.gsplats.zarr` directories is not a single dataset; the extractor picks among them arbitrarily (`iterdir()` order) and the peek may pick another. Either way a child group's attrs is never mistaken for the root's, and no link is ever followed. An archive with several top-level directories and no `*.gsplats.zarr`-named one is ambiguous (the extractor settles it by unpredictable `iterdir()` order) and deliberately carries nothing rather than guessing a sibling's attrs. Used by `load_gsplats.read_authored_appearance`, which carries a source root's authored compositing attrs across a structure-only rebuild (`gsplat lod`) for archive inputs as well as directories.

```python
from luxar.gsplats.io import load_gsplats

result = load_gsplats("fitted.gsplats.zarr", include_stats=True)
print(result.centers.shape)  # (N, d)
print(result.stats["time_seconds"])  # Fitting time

# Compressed formats work transparently
result = load_gsplats("fitted.gsplats.zarr.zip")
result = load_gsplats("fitted.gsplats.zarr.tar.gz")
```

**`inspect_gsplats_zarr()`** - Inspect metadata without loading arrays

```python
from luxar.gsplats.io import inspect_gsplats_zarr, format_gsplats_info

info = inspect_gsplats_zarr("fitted.gsplats.zarr")
print(format_gsplats_info(info))
# Output:
# GSplats: 10,000 splats, 3D
# Ordering: hilbert (bits_per_dim=21)
# Size: 1.2 MB (3.8 MB uncompressed, 3.2x compression)
# Fitting: 45.3s, 850 iterations (converged)
```

`inspect_gsplats_zarr()` walks the node tree to surface the primary
leaf's fields (`n_splats`, `ndim`, `ordering`, `chunk_size`,
`amplitude_range`, `center_bounds`, `has_label_ids` and — when set —
`label_vocabulary`), plus tree-shape metadata
(`n_additive_sublods_default`, `kind`, and `n_substitutive` or `n_parts`) so
multi-LOD and partitioned datasets are visible at a glance. Like
`load_gsplats()` it accepts a `.gsplats.zarr.zip` / `.gsplats.zarr.tar.gz`
archive as well as a directory; `storage_bytes` then measures the archive file
itself, and `compression_ratio` is `None` (not `1.0`) whenever that size cannot
be measured. "Without loading arrays" refers to the array data: a directory
store and a *flat* zip (store at the archive root) are read in place, while a
nested archive — what `save_gsplats(..., compress=…)` writes — is extracted to a
temp directory for the duration of the call, so inspecting one costs its
uncompressed size in temp space.

### Convenience Methods

`GSplatData` has convenience methods that wrap the above functions:

```python
from luxar.gsplats import fit_gaussian_splats, GSplatData
from luxar.encoding import EncodingMode

# Fit and save
result = fit_gaussian_splats(image, n_iters=1000)
result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)

# Load
loaded = GSplatData.load("fitted.gsplats.zarr", include_stats=True)
```

## Spatial Ordering

Spatial ordering arranges splats along space-filling curves to improve compression and enable efficient spatial queries.

**Morton (Z-order) Curve**:
- Simple bit-interleaving algorithm
- Fast to compute
- Good compression

**Hilbert Curve** (recommended):
- Better locality preservation
- ~10% better compression than Morton
- Uses a Numba-compiled kernel; falls back to the `hilbertcurve` package if Numba is unavailable

**No ordering**:
- Preserves original order
- Use when order is already optimized

### Ordering Functions

**`sort_splats_spatial()`** - Main interface

```python
from luxar.gsplats.io import sort_splats_spatial

indices, metadata = sort_splats_spatial(
    centers,
    method="hilbert",  # or "morton"
    resolution=None,   # Ignored (kept for signature stability)
)

# Reorder arrays
sorted_centers = centers[indices]
sorted_amplitudes = amplitudes[indices]
```

**`compute_chunk_bounds_gsplats()`** - Compute spatial index

```python
from luxar.gsplats.io import compute_chunk_bounds_gsplats

chunk_bounds = compute_chunk_bounds_gsplats(
    centers=sorted_centers,
    cholesky_factors=sorted_cholesky,
    chunk_size=2048,
    # Explicit override. This function only ever sees arrays, so its own default
    # is the canonical DEFAULT_TRUNCATION_RADIUS (2.75); the compiler call sites
    # pass the dataset's own truncation_radius here.
    coverage_sigma=3.0,
)
# Shape: (num_chunks, d, 2)
# [..., d, 0] = min bound in dimension d
# [..., d, 1] = max bound in dimension d
```

## Encoding Integration

This package uses `luxar.encoding` for semantic type-aware array encoding:

| Array | Semantic Type | MEMORY Mode Encoding |
|-------|---------------|---------------------|
| `centers` | COORDINATE | `linear_perchannel_u16` per-axis fixed-point (AUTO/MEMORY, with a gridded axis's grid snapped so it is exact; extent rail and sigma rail fall back to `float32`) / `float32` (PRECISION) |
| `amplitudes` | POSITIVE_SCALAR | canonical positive-scalar encoding (may quantize to uint8) |
| `cholesky_factors_diag` | CHOLESKY_DIAG | per-channel log: `log_perchannel_u8` (AUTO — certified, escalates to `u16`; MEMORY) / `float32` (PRECISION) |
| `cholesky_factors_offdiag` | CHOLESKY_OFFDIAG | per-channel signed-log: `signed_log_perchannel_u8` (escalates with the diagonal — one shared tier) / `float32`; absent if d==1 |
| `colors` | COLOR | `rgb_uint8` (SDR) or per-channel geolog (HDR, auto-detected): `geolog_perchannel_u16` (AUTO) / `_u8` (MEMORY) / `float32` (PRECISION) |
| `label_ids` | INDEX | smallest exact unsigned int for the largest id **present** (not the vocabulary's largest — a filtered leaf may keep unused entries) — identical in every mode, never LUT-encoded and never quantized (a class id has no near-miss) |

**COORDINATE centers are uint16 per-axis fixed-point** under AUTO/MEMORY
(`linear_perchannel_u16`, decoded back to float32 on read; a per-axis extent
≥ 2¹⁶ falls back to float32). float16 is never used on coordinates — its
*relative* precision is a footgun for absolute positions, so the writer
disables it (there is no `float16_allowed` knob).

**Grid snap (what keeps a stacked axis exact).** A time or channel axis built with
`combine_as_new_dimension(..., sigma=0.0)` has a tiny extent (so it passes the
extent rail) but essentially no width, so an ordinary uint16 grid step of
thousands of σ knocks every interior frame off its integer coordinate — measured
at 7 320 σ on a 100-frame stack, with only the two endpoints surviving (#1748).
Such an axis is **gridded**, though, so the encoder widens its stored `hi` until
the quantization grid coincides with the data's own spacing and every value
round-trips bit-exactly at uint16. It costs nothing (`lo`/`hi` are stored per axis
regardless — a scale choice, not a dtype change) and needs no action from the
caller, so it reports through arbol rather than warning. See
`luxar/encoding/README.md` for the eligibility test.

**Sigma rail (a geometry-aware backstop).** The rails above only see coordinates.
The gsplat writer also has the Cholesky factors in hand, so it compares **half**
each axis's grid step `(hi - lo) / 65535` — the worst-case round-trip
displacement — against *each splat's own* marginal σ on that axis: a splat is
**unrepresentable** there when that displacement exceeds
`MAX_CENTER_DISPLACEMENT_SIGMAS` (1.0) × its σ, i.e. when quantization can push
the center clear of its own core and out of a slice query that used to match it.
The centers are stored as `float32` (with a `UserWarning`) once more than
`MAX_UNREPRESENTABLE_SPLAT_FRACTION` (0.1%) of the splats are unrepresentable on
some axis **and** the encoder has no exact path of its own for that array —
tripping the population gate is necessary but not sufficient. An axis the snap
covers is **not** an offender: the rail runs the
encoder's own `gridded_axis_step` on any axis that trips the population gate
(lazily — `np.unique` per axis is 0.30 s of a 2.85 s encode on 5M×3 coordinates,
and a tripped axis is rare) and skips it when the encoder will store it exactly.
Nor is a **LUT-eligible** centers array, which the encoder already stores verbatim
(exactly, at ~1 B/value) — the rail asks `ArrayEncoder.encodes_as_lut` before
escalating, since float32 would be 4× the bytes for no gain in fidelity.
So a stacked dataset keeps uint16 centers and stays silent, and what is left for
the rail is a degenerate sub-population on a **non-gridded**, non-LUT axis — a
`sigma=0` track stack merged into a fit whose time axis is continuous, or an axis
with more distinct values than uint16 has levels — where the splats really are
destroyed.

Both numbers are set by harm rather than by jitter. Sub-σ displacement is
invisible — a whole-volume light-sheet fit over an 8192-voxel axis has a
0.125-voxel step, so its worst displacement is 0.0625 voxel, and paying 2× the
centers bytes for that is not worth it. The test is over the population rather
than the minimum because real fits contain a few needle Gaussians (an SPZ import
decodes scales as `exp(u8/16 - 10)`; a random-Cholesky fixture draws σ from
`U(0, 1)`), and one of those must not cost the whole array its uint16 win — but
0.1% rather than a looser 1%, because a degenerate *minority* is just as destroyed
as a degenerate whole: merging a 2,000-splat `sigma=0` track stack into a
300,000-splat fit leaves 0.662% of the splats displaced by up to 1,373 σ. Under
the displacement criterion the benign populations measure 0.03% or less, so
0.1% still clears them by 3× or more. Only the centers escalate — the
Cholesky/amplitude/color tiers keep whatever the mode selected. An escalated
centers array is also written with `deduplicate=False`: the encoder's content
registry is keyed on the centers bytes alone, and would otherwise hand the
escalated node an `array_ref` to a sibling's quantized array. The sigma rail
stands down entirely once an axis reaches `COORDINATE_U16_MAX_EXTENT` (2¹⁶):
the extent rail above already stores that array as float32, so it leaves the
clearer "extent ≥ 2¹⁶" diagnosis in place — and leaves the array its dedup,
which is safe for a verdict that depends only on the centers bytes.

**Encoding modes**:
- `AUTO`: Analyzes data and selects encoding (may quantize)
- `PRECISION`: Full float32, lossless (broadcasting still allowed)
- `MEMORY`: Aggressive quantization for minimum storage

**Broadcasting**: Uniform values stored once with metadata:
```python
# If all amplitudes are 1.0, stored as shape (1,) with:
# encoding = {"name": "broadcasted", "n_elements": 10000}
```

## File Format

### Zarr Structure (format v3.x — node tree)

The file root IS the node. `save_gsplats()` writes the trivial single-leaf
shape; `write_gsplats_tree()` accepts any `GSplatNode` (leaf / kind=lod /
kind=partition, freely nestable).

**Bare leaf** (what `save_gsplats()` writes):

```
fitted.gsplats.zarr/
├── .zattrs          # type: "gsplats", n_splats, ndim, has_colors,
│                    #   has_label_ids, label_vocabulary? (decimal-string
│                    #   id keys; present only with label_ids),
│                    #   ordering, ordering_min/max/bits, chunk_size,
│                    #   amplitude_range, center_bounds, position_bounds,
│                    #   truncation_radius, opacity/absorption/gamma/intensity/offset,
│                    #   blending_mode (only when explicitly set; unset ⇒
│                    #   inherited from nearest ancestor, viewer default
│                    #   "additive"),
│                    #   format_version: "3.3", format_type: "gsplats_zarr",
│                    #   timestamp, luxar_gsplats_version, description?
├── .zmetadata       # Consolidated metadata
├── centers                   # (N, d) float32, spatially ordered
├── amplitudes                # (N,) or (1,) float32
├── cholesky_factors_diag     # (N, d) float32          (diagonal of L)
├── cholesky_factors_offdiag  # (N, d*(d-1)/2) float32  (off-diagonal; absent if d==1)
├── colors           # (N, 3) float32/uint8  (optional)
├── label_ids        # (N,) or broadcast (1,) smallest exact uint  (optional)
├── chunk_bounds     # (num_chunks, d, 2) float32  (when ordering ≠ "none")
├── fitting/         # Optimization info (optional)
│   ├── .zattrs      # time_seconds, iterations, converged, …
│   └── config/.zattrs  # Fitter-specific parameters
└── provenance/      # Image lineage (optional)
    └── .zattrs      # source_file, shape, normalization
```

**Additive ladder** (n_additive_sublods > 1):
arrays live in `additive_<i>/` subgroups; the parent leaf group carries
`n_additive_sublods` + aggregate attrs.

**Substitutive LOD** (`kind=lod`):
`child_<i>/` subgroups, coarsest→finest on disk; each child carries
`coverage_fraction` (a dimensionless, viewport-relative value derived by
SCREEN-OCCUPANCY HALVING; coarsest = 0.0, finest = 0.5 for a whole-object ladder).
A ladder bound to a spatial partition — the `adaptive` recipe's per-tile groups,
the `overview` recipe's coarse-cap/fine-partition pair — is scaled ×2 (in area
units) to anchor its finest at `PARTITION_FINEST_AREA` (1.0) instead, keeping the
fills-screen switch point a tile needs; the writer derives the same anchor from
the topology when a node carries no stamped value. (`MAX_COVERAGE_FRACTION` = 4.0
is a different bound: the ceiling on a LEGACY `selector="coverage"` ladder, i.e.
on authored values, not on anything derived.)

**Spatial partition** (`kind=partition`):
`part_<i>/` subgroups; the viewer renders all parts simultaneously.

See `docs/specs/GSPLATS_ZARR_FORMAT.md` for full ASCII trees of all five shapes.

### Root Attributes (bare leaf)

```json
{
  "format_version": "3.3",
  "format_type": "gsplats_zarr",
  "timestamp": "2026-06-09T10:00:00Z",
  "luxar_gsplats_version": "0.1.0",
  "description": "DAPI nuclei fitting",
  "type": "gsplats",
  "n_splats": 10000,
  "ndim": 3,
  "has_colors": false,
  "has_label_ids": false,
  "position_bounds": {"min": [0,0,0], "max": [256,256,128]}
}
```

`luxar_gsplats_version` reflects the installed package version (`"unknown"`
if it cannot be resolved). `description` is only written when supplied.
For group roots (`kind=lod` / `kind=partition`) the node attrs differ
accordingly (see `docs/specs/GSPLATS_ZARR_FORMAT.md`).

### Leaf Attributes

```json
{
  "type": "gsplats",
  "n_splats": 10000,
  "ndim": 3,
  "has_colors": true,
  "has_label_ids": true,
  "label_vocabulary": {"0": "bone", "1": "lung"},
  "ordering": "hilbert",
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [256.0, 256.0, 128.0],
  "ordering_bits_per_dim": 21,
  "chunk_size": 2048,
  "amplitude_range": {"min": 0.01, "max": 1.5},
  "center_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "position_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "truncation_radius": 2.75
}
```

## Usage Examples

### Basic Save/Load

```python
from luxar.gsplats import fit_gaussian_splats
import numpy as np

# Generate test image
image = np.random.rand(128, 128).astype(np.float32)

# Fit Gaussian splats
result = fit_gaussian_splats(image, n_iters=1000)

# Save with default settings (Hilbert ordering, AUTO encoding)
result.save("fitted.gsplats.zarr")

# Load
loaded = GSplatData.load("fitted.gsplats.zarr")
```

### Memory-Optimized Save

```python
# Save with aggressive compression
result.save(
    "compressed.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,  # Quantize amplitudes/colors/centers/cholesky
)
```

### Lossless Save

```python
# Save with full precision (no quantization)
result.save(
    "lossless.gsplats.zarr",
    encoding_mode=EncodingMode.PRECISION,
    ordering="hilbert",  # Still use ordering for compression
)
```

### Compressed Archive Save

```python
# Save as compressed zip archive
save_gsplats(
    "fitted.gsplats.zarr.zip",
    centers=centers,
    amplitudes=amplitudes,
    cholesky_factors=cholesky,
    compress="zip",               # Creates .zip archive
    zip_deflate=True,             # Use DEFLATE for additional compression
)

# Save as tar.gz archive
save_gsplats(
    "fitted.gsplats.zarr.tar.gz",
    centers=centers,
    amplitudes=amplitudes,
    cholesky_factors=cholesky,
    compress="tar.gz",
)
```

### Save with Provenance

```python
result = fit_gaussian_splats(image, n_iters=1000)

# Add provenance info to stats
result.stats["provenance"] = {
    "source_file": "/data/image.tif",
    "shape": [128, 256, 256],
    "dtype": "uint16",
    "normalization": {
        "method": "percentile",
        "low": 0.1,
        "high": 99.9,
    }
}

# Save with provenance
result.save("fitted.gsplats.zarr", include_provenance=True)
```

### Inspect Without Loading

```python
from luxar.gsplats.io import inspect_gsplats_zarr, format_gsplats_info

# Get metadata
info = inspect_gsplats_zarr("fitted.gsplats.zarr")

# Print summary
print(format_gsplats_info(info))

# Access specific fields
print(f"Splats: {info['n_splats']}")
print(f"Ordering: {info['ordering']}")
# `compression_ratio` is None when the on-disk size could not be measured.
if info["compression_ratio"] is not None:
    print(f"Compression: {info['compression_ratio']}x")
```

### Migrate a Legacy Dataset to v3.3

```python
from luxar.gsplats.io.migrate import migrate_format, detect_legacy_format

# Detect the legacy shape: "v1.0", "v1.1", "substitutive_dir", "v2.0", or
# "v3.0-lod-pixel-size"/"v3.1-lod-pixel-size" (a v3.0/v3.1 store with pre-v3.2 lod selector attrs);
# raises on a current file with nothing to upgrade
print(detect_legacy_format("legacy.gsplats.zarr"))

# Convert to a new v3.3 file (returns the detected source format)
migrate_format("legacy.gsplats.zarr", "v3.gsplats.zarr", overwrite=True)
```

Migration mappings:
- **v1.0** (flat `/splats`) → v3.3 bare leaf
- **v1.1** (`/splats/lod_<i>/`) → v3.3 additive ladder leaf
- **Substitutive directory** (`level_<i>.gsplats.zarr` + `manifest.json`) →
  v3.3 `kind=lod` group
- **v2.0** (`splats/substitutive_<s>/additive_<a>/`) → v3.3 node tree
  (bare leaf, additive ladder, or `kind=lod` group depending on shape)
- **v3.0/v3.1 with `selector: "pixel_size"` / per-child `min_pixel_size`** →
  same tree re-written with `selector: "screen-area"` + freshly derived per-child
  `coverage_fraction` (occupancy halving to finest `0.5`, or that × 2 in area
  units — finest `1.0` — for a partition-bound ladder, i.e. a legacy `adaptive` /
  `overview` store, whose derivation is topology-aware), stamped v3.3

Migrated arrays are written with `ordering="none"` so element order is
preserved (no Morton/Hilbert re-sort), but **encoding follows the current policy**:
under the default `EncodingMode.AUTO`, legacy float32 Cholesky factors are
re-encoded as the split diagonal/off-diagonal arrays with certified uint8
per-column quantization (escalating to uint16 when the encode-time covariance
certificate demands it). Pass `encoding_mode=EncodingMode.PRECISION`
(`--lossless` on the CLI) for an exact float32 archival migration.
Fitting/provenance groups are spliced back onto the output. The CLI entry point
is `luxar gsplat migrate-format`.

## Architecture

### Code Organization

- **`save_gsplats.py`**: `save_gsplats()` (bare leaf) and `write_gsplats_tree()`
  (any `GSplatNode`). Delegates to the shared walker
  `io/_compiler/gsplat_tree.write_gsplat_node`; writes the v3.3 root header.
- **`load_gsplats.py`**: Load function with automatic decoding. Reads the
  node tree via `io/_compiler/gsplat_tree.read_gsplat_node`, returning
  a `GSplatData` bridged from the node tree. Raises on any `format_version`
  not in `SUPPORTED_FORMAT_VERSIONS` (`"3.0"`, `"3.1"`, `"3.2"`, `"3.3"`); the
  current writer emits v3.3, and earlier v3.x files are read transparently.
  Also `read_authored_appearance(path)` — the source root's authored compositing
  attrs (`AUTHORED_APPEARANCE_ATTRS`), for a command that rewrites a dataset to
  hand back to `write_gsplats_tree(root_attrs=…)` / `GSplatData.save(root_attrs=…)`
  so a structure-only rebuild does not silently reset the look. Archive inputs
  are carried too — a `.gsplats.zarr.zip` / `.tar.gz` is peeked in place via
  `_archive.read_archive_root_attrs` (see above), no extraction. Best-effort:
  a missing/unreadable store yields `{}`.
  Its N-input sibling `agreed_authored_appearance(paths, exclude=…)` is what a
  command with several inputs (`gsplat merge`) uses: a key is carried only when
  every input that *has* an opinion agrees, an input with no opinion casts no
  vote, and a disagreement drops the key **with a warning** naming the differing
  values (each with the input it came from) and what lands instead. Same unanimity rule as
  `save_gsplats.agreed_normalization_stats`, deliberately loud rather than
  silent because appearance is hand-authored. Two refinements make the rule
  usable, both on `_appearance_votes`: a value equal to the one the WRITER
  manufactures (`WRITER_STAMPED_APPEARANCE_DEFAULTS`) is silence, not a vote —
  otherwise merging a tuned dataset with a freshly fitted one disagrees on seven
  keys and reverts to the untouched look; and `visible` is the one key where
  ABSENCE votes (`true`), so a unilateral `visible: false` cannot hide the merged
  whole. `colormap` is refused outright when any input declares the `"custom"`
  sentinel (warned once for N inputs), since demoting it to "no opinion" would
  let a sibling's palette repaint those splats. `exclude` names keys the CALLER
  invalidates whatever the inputs say — for `gsplat merge` that is `colormap`
  whenever the merge manufactured per-splat RGB; pass a `{key: reason}` mapping
  to have the reason quoted in the warning. It is typed as a set/mapping so a
  bare `str` cannot be passed by accident. `input_has_colors` identifies a
  colored input with no authored palette as relying on its per-splat RGB, so a
  sibling palette is refused rather than repainting it. `output_has_colors`
  lets a drop warning say whether the merged writer leaves `colormap` unset or
  stamps its colorless `"gray"` default.
- **`inspect_gsplats.py`**: Metadata inspection without loading arrays
  (`inspect_gsplats_zarr`, `format_gsplats_info`).
- **`migrate.py`**: Legacy-format migration (`migrate_format`,
  `detect_legacy_format`). Converts v1.0 / v1.1 / pre-v2.0 substitutive
  directory / v2.0 matrix files — plus v3.0/v3.1 stores whose `kind=lod`
  groups still carry the pre-v3.2 `pixel_size` selector attrs — to v3.3.
  Carries embedded legacy reader logic so the live loader only handles the
  current node-tree format (v3.0-v3.3).
- **`tests/`**: Comprehensive tests

**Note**: Spatial ordering functions are imported from `luxar.io.ordering` and re-exported for convenience. `migrate.py` is not re-exported from the package `__init__`; it backs the `luxar gsplat migrate-format` CLI command.

### Dependencies

- **`luxar.encoding`**: Semantic type-based array encoding (see `../../encoding/README.md`)
- **`luxar.typing_utils`**: Format-contract version constants (`GSPLATS_FORMAT_VERSION`, `SUPPORTED_GSPLATS_VERSIONS`)
- **`hilbertcurve`**: Fallback for Hilbert ordering when Numba is unavailable (Morton needs neither)

### Relationship to luxar.io

This package uses the same core principles as `luxar.io` but for standalone splat files:
- Both use spatial ordering for compression
- Both use `luxar.encoding` for semantic types
- Both compute `chunk_bounds` for spatial queries
- **Difference**: Standalone `.gsplats.zarr` vs embedded in scene graph

## Testing

The package includes comprehensive tests covering:

**Ordering tests**:
- Morton encoding (2D, 3D, nD)
- Hilbert encoding (2D, 3D, nD)
- Coordinate normalization
- Auto-resolution computation
- Chunk bounds calculation

**Save/Load tests**:
- Basic save/load
- Encoding modes (AUTO, PRECISION, MEMORY)
- Spatial ordering (Morton, Hilbert, none)
- Colors (SDR uint8 / HDR geolog_perchannel_u16, auto-detected)
- Fitting metadata
- Round-trip accuracy
- Error validation

**Format compliance tests** (`test_format.py`):
- Root attributes
- Splats group structure
- Array shapes
- Encoding metadata
- Ordering metadata
- Fitting/provenance groups
- Chunk bounds format

**Migration tests** (`test_migrate_format.py`):
- Legacy-format detection (v1.0, v1.1, v2.0, substitutive directory)
- Round-trip migration to v3.3 node-tree layout

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/gsplats/io/tests/ -v
```

## Performance

Typical compression ratios (compared to uncompressed float32):

| Configuration | Compression | Notes |
|---------------|-------------|-------|
| PRECISION + Hilbert | 2-3x | Lossless, ordering helps blosc |
| AUTO + Hilbert | 4-6x | Selective quantization |
| MEMORY + Hilbert + log | 8-12x | Aggressive quantization |

Compression gains from:
1. **Spatial ordering** (~2x from blosc shuffle on ordered data)
2. **Quantization** (2-4x from float32→uint16/uint8)
3. **Broadcasting** (massive savings when values are uniform)
4. **LUT encoding** (up to 75% savings for <256 unique values)
5. **Array deduplication** (via xxhash64 in `luxar.encoding`)

## Related Documentation

- **Format spec (v3.3)**: `../../../../../../docs/specs/GSPLATS_ZARR_FORMAT.md` (node-tree: leaf / kind=lod / kind=partition)
- **Encoding system**: `../../encoding/README.md` (semantic types, quantization)
- **Scene embedding**: `../../core/README.md` (GSplats in scene graph)
- **Parent package**: `../README.md` (Gaussian splatting algorithms)
