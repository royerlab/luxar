# luxar.gsplats.io

I/O operations for persisting and loading Gaussian splat data in the `.gsplats.zarr` format.

## Purpose

This package provides functions to save and load fitted Gaussian splat results with:
- **Spatial ordering** (Morton/Hilbert curves) for better compression
- **Semantic encoding** (quantization, broadcasting, LUT) via `luxar.encoding`
- **Chunk-based spatial indexing** for efficient queries
- **Provenance tracking** for reproducibility
- **Fitter-agnostic metadata** allowing different fitting implementations

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
    color_mode="sdr",             # Required if colors are float32
    fitting_info={"time_seconds": 45.3, "iterations": 850},
    description="DAPI nuclei fitting",
    compress=None,                # "zip" or "tar.gz" for compressed archive
    zip_deflate=False,            # Use DEFLATE compression for outer zip
    compressor=None,              # Custom Blosc compressor (default: zstd level 3)
)
```

**`load_gsplats()`** - Load splats from .gsplats.zarr

Transparently handles compressed formats (`.gsplats.zarr.zip`, `.gsplats.zarr.tar.gz`) by extracting to a temporary directory automatically. Arrays are decoded from their stored encoding (quantization, broadcasting, etc.) to float32.

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
# Ordering: hilbert (resolution=65536)
# Size: 1.2 MB (compression ratio: 3.2x)
# Fitting time: 45.3s, 850 iterations (converged)
```

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
- Requires `hilbertcurve` package

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
    resolution=None,   # Auto-computed if None
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
| `centers` | COORDINATE | `float16` (half precision) |
| `amplitudes` | POSITIVE_SCALAR | `log_scalar_uint8` (log scale) |
| `cholesky_factors` | CHOLESKY | `float16` (~0.1% error) |
| `colors` | COLOR | `rgb_uint8` (SDR) or `float32` (HDR) |

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

### Zarr Structure

```
fitted.gsplats.zarr/
├── .zattrs                      # Format metadata
├── .zmetadata                   # Consolidated metadata
│
├── splats/                      # Core splat data
│   ├── centers                  # (N, d) spatially ordered
│   ├── amplitudes               # (N,) or (1,) spatially ordered
│   ├── cholesky_factors         # (N, k) spatially ordered
│   ├── colors                   # (N, 3) or (1, 3) (optional)
│   ├── chunk_bounds             # (num_chunks, d, 2)
│   └── .zattrs                  # n_splats, ndim, ordering, ranges
│
├── fitting/                     # Optimization info (optional)
│   ├── .zattrs                  # time_seconds, iterations, converged
│   └── config/.zattrs           # Fitter-specific parameters
│
└── provenance/                  # Image lineage (optional)
    └── .zattrs                  # source_file, shape, normalization
```

### Root Attributes

```json
{
  "format_version": "1.0",
  "format_type": "gsplats_zarr",
  "timestamp": "2025-01-15T14:30:00Z",
  "luxar_gsplats_version": "0.1.0",
  "description": "DAPI nuclei fitting"
}
```

### Splats Group Attributes

```json
{
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
  }
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
    encoding_mode=EncodingMode.MEMORY,  # Quantize everything
    positive_scalar_encoding="log",     # Log-scale amplitudes
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

## Architecture

### Code Organization

- **`save_gsplats.py`**: Save function with validation and encoding
- **`load_gsplats.py`**: Load function with automatic decoding
- **`inspect_gsplats.py`**: Metadata inspection without loading arrays
- **`tests/`**: Comprehensive tests

**Note**: Spatial ordering functions are imported from `luxar.io.ordering` and re-exported for convenience.

### Dependencies

- **`luxar.encoding`**: Semantic type-based array encoding (see `../../encoding/README.md`)
- **`luxar.typing_utils`**: Constants (TARGET_CHUNK_BYTES)
- **`hilbertcurve`**: Required for Hilbert ordering (optional for Morton)

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
- Colors with color_mode
- Fitting metadata
- Round-trip accuracy
- Error validation

**Format compliance tests**:
- Root attributes
- Splats group structure
- Array shapes
- Encoding metadata
- Ordering metadata
- Fitting/provenance groups
- Chunk bounds format

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
2. **Quantization** (2-4x from float32→float16/uint8)
3. **Broadcasting** (massive savings when values are uniform)
4. **LUT encoding** (up to 75% savings for <256 unique values)
5. **Array deduplication** (via xxhash64 in `luxar.encoding`)

## Related Documentation

- **Encoding system**: `../../encoding/README.md` (semantic types, quantization)
- **Scene embedding**: `../../core/README.md` (GSplats in scene graph)
- **Parent package**: `../README.md` (Gaussian splatting algorithms)
