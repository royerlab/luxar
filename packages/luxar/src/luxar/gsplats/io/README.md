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

**`load_gsplats()`** - Load splats from .gsplats.zarr

Transparently handles compressed formats (`.gsplats.zarr.zip`, `.gsplats.zarr.tar.gz`) by extracting to a temporary directory automatically (via the shared, hardened `_archive.extract_compressed_zarr` — it rejects links/devices, validates every member before extracting, and caps member count / total size to guard against path-traversal and archive-bomb attacks). Arrays are decoded from their stored encoding (quantization, broadcasting, etc.) to float32.

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
`amplitude_range`, `center_bounds`), plus tree-shape metadata
(`n_additive_sublods_default`, `kind`, and `n_substitutive` or `n_parts`) so
multi-LOD and partitioned datasets are visible at a glance.

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
    coverage_sigma=3.0,  # 3σ coverage (99.7%)
)
# Shape: (num_chunks, d, 2)
# [..., d, 0] = min bound in dimension d
# [..., d, 1] = max bound in dimension d
```

## Encoding Integration

This package uses `luxar.encoding` for semantic type-aware array encoding:

| Array | Semantic Type | MEMORY Mode Encoding |
|-------|---------------|---------------------|
| `centers` | COORDINATE | `linear_perchannel_u16` per-axis fixed-point (AUTO/MEMORY; extent rail falls back to `float32`) / `float32` (PRECISION) |
| `amplitudes` | POSITIVE_SCALAR | canonical positive-scalar encoding (may quantize to uint8) |
| `cholesky_factors_diag` | CHOLESKY_DIAG | per-channel log: `log_perchannel_u8` (AUTO — certified, escalates to `u16`; MEMORY) / `float32` (PRECISION) |
| `cholesky_factors_offdiag` | CHOLESKY_OFFDIAG | per-channel signed-log: `signed_log_perchannel_u8` (escalates with the diagonal — one shared tier) / `float32`; absent if d==1 |
| `colors` | COLOR | `rgb_uint8` (SDR) or per-channel geolog (HDR, auto-detected): `geolog_perchannel_u16` (AUTO) / `_u8` (MEMORY) / `float32` (PRECISION) |

**COORDINATE centers are uint16 per-axis fixed-point** under AUTO/MEMORY
(`linear_perchannel_u16`, decoded back to float32 on read; a per-axis extent
≥ 2¹⁶ falls back to float32). float16 is never used on coordinates — its
*relative* precision is a footgun for absolute positions, so the writer
disables it (there is no `float16_allowed` knob).

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
`coverage_fraction` (a dimensionless, viewport-relative value in `[0, 1]`,
`sqrt(N_i / N_finest)`; coarsest = 0.0, finest = 1.0).

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
  same tree re-written with `selector: "coverage"` + derived per-child
  `coverage_fraction` (`sqrt(N_i/N_finest)`), stamped v3.3

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
