# GSplats Zarr Format Design Document

**Status**: Draft / Discussion

## Overview

This document captures the design discussion for a dedicated zarr format for storing Gaussian splats (`.gsplats.zarr`). The goal is to enable efficient persistence, loading, and potential compression of fitted Gaussian splat results.

**Encoding Integration**: This format uses the `luxar.encoding` package for array transformations (quantization, broadcasting, deduplication). See `luxar/encoding/SPECIFICATIONS.md` for the complete encoding specification. This ensures consistent encoding behavior across all Luxar data types.

## Use Cases

1. **Save/Load fitted results** - Persist expensive fitting results for later use
2. **Lightweight rendering** - Load only splat data for visualization
3. **Provenance tracking** - Record what image/parameters produced the splats
4. **Future: Checkpoint/Resume** - Pause and resume fitting (deferred)
5. **Future: Multiscale storage** - Store hierarchical decompositions (deferred)

## Core Data Structure

Each Gaussian splat is parameterized by:

| Field | Shape | Dtype | Semantic Type | Description |
|-------|-------|-------|---------------|-------------|
| `centers` | (N, d) | float32 | COORDINATE | Splat center positions (not broadcastable) |
| `amplitudes` | (N,) or (1,) | float32 | POSITIVE_SCALAR | Non-negative intensity |
| `cholesky_factors` | (N, d*(d+1)/2) or (1, d*(d+1)/2) | float32 | CHOLESKY | Packed lower-triangular L where Σ = LLᵀ |
| `sharpnesses` | (N,) or (1,) | float32 | BOUNDED_SCALAR | Generalized Gaussian exponent (s=2 is standard, bounds [0, 32]) |

**Note**: Cholesky factors are packed in row-major order. For d=3: `[L00, L10, L11, L20, L21, L22]`

**Semantic Types**: Each field maps to an encoding semantic type (see `luxar.encoding.SemanticType`). This determines valid encodings and quantization options for each array.

### Broadcasting Convention

Broadcasting uses the standard `luxar.encoding` format. When all elements share the same value, the array is stored with shape `(1,)` or `(1, d)` with encoding metadata:

```json
// sharpnesses/.zattrs - all splats have sharpness=2.0
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

**Note**: Semantic type is determined by the array name (e.g., `sharpnesses` → BOUNDED_SCALAR), not stored in metadata.

The `n_splats` attribute in `splats/.zattrs` always reflects the true count (N), regardless of broadcasting.

---

## Proposed Zarr Structure

```
fitted.gsplats.zarr/
├── .zattrs                      # Format metadata (see below)
├── .zmetadata                   # Consolidated metadata for fast loading
│
├── splats/                      # Core splat data
│   ├── centers                  # (N, d) float32
│   ├── amplitudes               # (N,) float32
│   ├── cholesky_factors         # (N, d*(d+1)/2) float32
│   ├── sharpnesses              # (N,) float32
│   └── .zattrs                  # n_splats, ndim, ordering info
│
├── fitting/                     # Optimization info (optional, fitter-specific)
│   ├── .zattrs                  # Common: time_seconds, fitter_name, fitter_version
│   └── config/                  # Fitter-specific parameters (free-form JSON)
│       └── .zattrs              # Each fitter defines its own schema
│
└── provenance/                  # Image lineage (optional)
    └── .zattrs                  # source_file, shape, dtype, normalization
```

### Root Attributes (.zattrs)

```json
{
  "format_version": "1.0",
  "format_type": "gsplats_zarr",
  "creation_timestamp": "2025-01-15T14:30:00Z",  // ISO 8601 format
  "luxar_gsplats_version": "X.Y.Z",
  "description": "Optional user description"
}
```

**Note**: Core splat metadata (`n_splats`, `ndim`) is stored in `splats/.zattrs` (single source of truth).

### Splats Group Attributes (Single Source of Truth)

```json
{
  "n_splats": 10000,
  "ndim": 3,
  "ordering": "morton",           // "morton", "hilbert", or "none"
  "ordering_resolution": 65536,   // Only present when ordering != "none"
  "chunk_size": 8192,             // Chunk size used
  "amplitude_range": {"min": 0.01, "max": 1.5},   // Actual data range for rendering
  "sharpness_bounds": {"min": 0.0, "max": 32.0},  // Model constraints (valid range)
  "center_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  }
}
```

**Bounds clarification**: Group attributes store **model constraints** (valid ranges). If arrays are quantized, the encoding metadata may store **tighter bounds** for better precision within the actual data range.

**Note**: Broadcasting information is stored per-array via encoding metadata (see Broadcasting Convention above), not in the group attributes.

### Fitting Group Attributes (Fitter-Agnostic)

The `fitting/` group is **optional** and designed to be **fitter-agnostic**. Different fitting implementations can store their own parameters while sharing common fields.

**Common fields** (fitting/.zattrs):
```json
{
  "fitter_name": "luxar.gsplats",           // Identifier for the fitter
  "fitter_version": "0.1.0",                // Version of the fitter
  "time_seconds": 45.3,                     // Wall-clock fitting time
  "iterations": 850,                        // Number of iterations
  "converged": true,                        // Did it meet convergence criteria?
  "timestamp": "2025-01-15T14:30:00Z"       // ISO 8601 format
}
```

**Fitter-specific config** (fitting/config/.zattrs):
```json
{
  // Luxar gsplats fitter example:
  "n_iters": 1000,
  "lr": 0.05,
  "loss_type": "l1",
  "asymmetric_penalty": 10.0,
  "init_sigma_vox": 0.5,
  "seed_method": "both",
  "enable_dynamic_ops": true,
  // ... any other fitter-specific parameters
}
```

**Design principle**: Other programs that produce gsplats can write their own `fitter_name` and custom config. Readers should:
1. Always read common fields from `fitting/.zattrs`
2. Only interpret `fitting/config/.zattrs` if they recognize the `fitter_name`

### Provenance Group Attributes (Optional)

The `provenance/` group records information about the source image:

```json
{
  "source_file": "/path/to/image.tif",      // Original image path
  "source_hash": "sha256:abc123...",        // Hash for verification (optional)
  "shape": [128, 256, 256],                 // Image dimensions (ZYX or YX)
  "dtype": "uint16",                        // Original image dtype
  "normalization": {
    "method": "percentile",                 // "minmax", "percentile", "none"
    "low": 0.1,                             // Lower percentile (if applicable)
    "high": 99.9                            // Upper percentile (if applicable)
  }
}
```

---

## Splat Ordering for Compression

### Why Order Matters

Splats are inherently unordered, but storage order significantly affects compression:
- **Nearby splats** have similar centers → excellent delta compression
- **Nearby splats** often have similar covariances (local structure)
- **Chunk access** becomes coherent for spatial queries

### Space-Filling Curves

Two options for spatial ordering:

#### Morton (Z-order)

**Pros**:
- Simple bit-interleaving implementation
- Trivial nD extension
- Fast to compute

**Cons**:
- Occasional "jumps" in locality (at quadrant boundaries)

**Algorithm**:
```python
def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Morton codes via bit interleaving."""
    n_points, n_dims = coords.shape
    morton = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)

    return morton
```

#### Hilbert Curve

**Pros**:
- Better locality preservation (never jumps far)
- ~10% better compression than Morton in practice

**Cons**:
- More complex algorithm
- Harder nD extension (but libraries exist)

**Libraries**:
- [`numpy-hilbert-curve`](https://github.com/PrincetonLIPS/numpy-hilbert-curve) - Princeton LIPS, numpy-native
- [`hilbertcurve`](https://pypi.org/project/hilbertcurve/) - Supports nD, based on Skilling 2004

### Implementation Decision

**Support both Morton and Hilbert** with a parameter:

```python
def sort_splats_spatially(
    centers: np.ndarray,
    method: Literal["morton", "hilbert"] = "hilbert",
    resolution: int = None,  # Auto if None, max 2^16
) -> np.ndarray:
    """Return sort indices for spatial ordering."""
    ...
```

**Resolution auto-calculation**:
```python
# Compute based on data spread, capped at 2^16
spread = centers.max(axis=0) - centers.min(axis=0)
max_spread = spread.max()
resolution = min(2**16, max(256, int(max_spread * 10)))
```

---

## Additional Compression Techniques

Based on [3DGS compression survey](https://arxiv.org/html/2502.19457v1) and related research:

### 1. Quantization

Quantization is handled by `luxar.encoding` based on semantic types:

| Field | Semantic Type | MEMORY Mode Encoding |
|-------|---------------|---------------------|
| `centers` | COORDINATE | `float16` (half precision) |
| `amplitudes` | POSITIVE_SCALAR | `positive_scalar_uint8` or `log_scalar_uint8` |
| `cholesky_factors` | CHOLESKY | `float16` (~0.1% error, see encoding spec Section 4.5) |
| `sharpnesses` | BOUNDED_SCALAR | `bounded_scalar_uint8` (8-bit, bounds [0, 32]) |

**Log-scale amplitudes**: For high dynamic range (HDR) amplitudes, use log encoding:
```python
result.save(
    "hdr_splats.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,
    positive_scalar_encoding="log",  # log1p/expm1 for numerical stability
)
```

See `luxar.encoding.SPECIFICATIONS.md` for complete quantization details and precision guarantees.

**Research note**: [OMG](https://arxiv.org/html/2503.16924) achieves 100-300× compression with quantization while maintaining quality.

### 2. Entropy Coding with Spatial Coherence

From research: "Entropy encoding linearizes 3D Gaussians along a space-filling curve to exploit the spatial coherence of scene parameters."

This validates our space-filling curve approach! After Morton/Hilbert ordering:
- Delta encoding of centers (store differences)
- Run-length encoding for similar values
- Standard compressors (zstd, blosc) work better on ordered data

### 3. Attribute Factorization

**F-3DGS approach**: Instead of storing full attributes per splat, factorize:
```
attributes = basis_vectors @ coefficients
```

For our case, could factorize:
- Cholesky factors → shared basis shapes + per-splat coefficients
- Would require fitting a basis during save (more complex)

**Decision**: Defer to v2.0 - start simple with ordering + standard compression.

### 4. Pruning Before Storage

Not a storage format concern, but worth noting:
- Remove low-amplitude splats before saving
- User can set threshold: `result.prune(min_amplitude=0.01).save(...)`

---

## Compression Configuration

### Blosc Settings

Default zarr compressor configuration:

```python
compressor = Blosc(
    cname='zstd',      # Best compression ratio
    clevel=5,          # Balance speed/ratio (1-9)
    shuffle=Blosc.BITSHUFFLE,  # Good for float arrays
)
```

### Chunk Size

Chunk size is **configurable** with an optimal default for compression.

**Blosc optimal chunk size**: Blosc works best with chunks in the 16KB-1MB range. For float32 splats:
- Bytes per splat: `4*d + 4 + 4*d*(d+1)/2 + 4` = `4*(d + 1 + d*(d+1)/2 + 1)`
- **2D**: 28 bytes/splat (8 centers + 4 amp + 12 cholesky + 4 sharp)
- **3D**: 44 bytes/splat (12 centers + 4 amp + 24 cholesky + 4 sharp)
- **4D**: 64 bytes/splat (16 centers + 4 amp + 40 cholesky + 4 sharp)

For 3D (most common):
- 16KB / 44 bytes ≈ 370 splats (minimum efficient)
- 256KB / 44 bytes ≈ 6,000 splats (sweet spot)
- 1MB / 44 bytes ≈ 24,000 splats (maximum efficient)

**Default**: `chunk_size = 8192` splats (~360KB for 3D) - good balance for compression and random access.

```python
# Configurable chunk size with sensible default
DEFAULT_CHUNK_SIZE = 8192  # Optimal for blosc compression

def get_chunk_size(n_splats: int, chunk_size: int | None = None) -> int:
    """Get chunk size, using default if not specified."""
    if chunk_size is not None:
        return min(n_splats, chunk_size)
    return min(n_splats, DEFAULT_CHUNK_SIZE)

# All arrays chunked the same way for coherent access
centers_chunks = (chunk_size, ndim)
amplitudes_chunks = (chunk_size,)
cholesky_chunks = (chunk_size, ndim * (ndim + 1) // 2)
sharpnesses_chunks = (chunk_size,)
```

**Stored in metadata**: `splats/.zattrs["chunk_size"]` for reproducibility.

---

## API Design

### Saving

```python
from luxar.gsplats import GaussianSplatResult
from luxar.encoding import EncodingMode

result = fit_gaussian_splats(image, n_iters=1000)

# Simple save with defaults (hilbert ordering, zstd compression, auto encoding)
result.save("fitted.gsplats.zarr")

# With options
result.save(
    "fitted.gsplats.zarr",
    ordering="morton",           # or "hilbert", "none"
    encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
    include_fitting_info=True,   # Store stats and config
    include_provenance=True,     # Store image metadata
    description="DAPI nuclei fitting",
)

# Memory-optimized save (quantization enabled)
result.save(
    "compressed.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,  # Enable quantization
)
```

**Encoding modes** (see `luxar.encoding.SPECIFICATIONS.md` Section 9):
- `AUTO`: Analyze data and select encoding (may be lossy for some types, e.g., SDR colors → uint8)
- `PRECISION`: Full float32, lossless only (broadcasting still allowed)
- `MEMORY`: Aggressive quantization for minimum storage
- `CUSTOM`: Explicit encoder selection per array (advanced use)

### Loading

```python
# Load for rendering (splats only)
result = GaussianSplatResult.load("fitted.gsplats.zarr")

# Load with all metadata
result = GaussianSplatResult.load(
    "fitted.gsplats.zarr",
    include_stats=True,
)
print(result.stats['time_seconds'])
```

**Note**: Arrays are automatically decoded based on encoding metadata. Files saved with `EncodingMode.MEMORY` (quantized) are transparently decoded to float32 on load.

### Inspection

```python
from luxar.gsplats import inspect_gsplats_zarr

info = inspect_gsplats_zarr("fitted.gsplats.zarr")
print(info)
# GSplats: 10,000 splats, 3D
# Ordering: hilbert (resolution=65536)
# Size: 1.2 MB (compression ratio: 3.2x)
# Fitting time: 45.3s, 850 iterations
```

---

## Luxar Integration (Future)

Once `.gsplats.zarr` format is stable, integrate with Luxar visualization:

```python
from luxar import LuxarZarrCompiler
from luxar.gsplats import GaussianSplatResult

# Option 1: From in-memory result
result = fit_gaussian_splats(image)
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene()
    scene.add_gsplats("nuclei", result)

# Option 2: From saved .gsplats.zarr file
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene()
    scene.add_gsplats("nuclei", "fitted.gsplats.zarr")  # Path
```

This will be designed after the gsplats I/O module is complete.

---

## Design Decisions (Agreed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File extension | `.gsplats.zarr` | Clear, descriptive |
| Encoding system | `luxar.encoding` package | Shared infrastructure, semantic types |
| Ordering methods | Both Morton and Hilbert | User choice, Hilbert default |
| Resolution | Auto with cap at 2^16 | Safe default, configurable |
| Chunk size | Configurable, default 8192 | Optimal for blosc |
| Broadcasting | Standard encoding metadata | Consistent with other Luxar formats |
| Covariance storage | Cholesky (packed) | Already have it, compresses well |
| Fitting info | Fitter-agnostic design | Allows other programs to use format |
| Checkpoint/Resume | Deferred | Focus on basic I/O first |
| Multiscale | Deferred | Not finalized yet |
| Image embedding | No | Keep format focused on splats |
| Compression | Blosc + BITSHUFFLE + zstd | Standard, well-supported |
| Delta encoding | No | Blosc shuffle sufficient |
| Streaming write | No | Not needed |
| `numpy-hilbert-curve` | Required dependency | Needed for Hilbert ordering |

---

## Open Questions

*None currently - all questions resolved.*

### Resolved Questions

1. **Log-scale amplitudes**: ✅ Resolved - Use `positive_scalar_encoding="log"` parameter with `EncodingMode.MEMORY`. See Quantization section above.

---

## References

- [3DGS Compression Survey](https://arxiv.org/html/2502.19457v1) - Comprehensive overview of compression techniques
- [3DGS.zip Survey](https://arxiv.org/abs/2407.09510) - Another survey on compression methods
- [OMG: Optimized Minimal Gaussians](https://arxiv.org/html/2503.16924) - 100-300× compression
- [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve) - nD Hilbert implementation
- [Skilling 2004](https://doi.org/10.1063/1.1751381) - "Programming the Hilbert Curve" algorithm

---

## Changelog

- **v0.7 (Draft)**: Unified sharpness bounds
  - Changed sharpness bounds from [0.16, 24.4] to [0, 32] to align with core SPECIFICATIONS.md
  - All node types (Points, Lines, GSplats) now use unified bounds [0, 32]

- **v0.6 (Draft)**: Cross-specification consistency
  - Fixed AUTO mode description to match encoding spec (can be lossy)
  - Added CUSTOM mode to encoding modes list
- **v0.5 (Draft)**: Polish and consistency
  - Removed "Last Updated" placeholder (changelog tracks history)
  - Removed version from "Proposed Zarr Structure" heading
  - Made bounds format consistent (all use `{min, max}` objects)
  - Clarified `ordering_resolution` only present when `ordering != "none"`
  - Added bytes-per-splat formula for all dimensions (2D/3D/4D)
  - Fixed byte calculations (44 bytes for 3D, not 40)
- **v0.4 (Draft)**: Documentation improvements
  - Clarified `ordering_resolution` is for reproducibility/debugging only
  - Specified ISO 8601 timestamp format for all timestamps
  - Added provenance group JSON example
  - Added note about automatic decoding on load
  - Clarified distinction: group attrs = model constraints, encoding = quantization params
- **v0.3 (Draft)**: Critical review fixes
  - Moved `n_splats`/`ndim` to splats group only (single source of truth)
  - Removed redundant `max_amplitude` (use `amplitude_range[1]`)
  - Added `sharpness_bounds` based on model constraints (later unified to [0, 32] in v0.7)
  - Fixed CHOLESKY MEMORY mode to use `float16` (not uint16)
  - Fixed PRECISION mode description (broadcasting still allowed)
  - Clarified semantic type is from array name, not stored in metadata
  - Clarified centers cannot be broadcasted
- **v0.2 (Draft)**: Integrated `luxar.encoding` package
  - Added semantic types for all splat arrays (COORDINATE, POSITIVE_SCALAR, CHOLESKY, BOUNDED_SCALAR)
  - Updated broadcasting to use standard encoding metadata format
  - Added `encoding_mode` parameter to save() API
  - Simplified quantization section to reference encoding spec
  - Resolved log-scale amplitudes question with `positive_scalar_encoding="log"`
- **v0.1 (Draft)**: Initial design discussion