# luxar.gsplats.io - Technical Specification

**Version**: 1.1.1
**Last Updated**: 2025-11-28

## Purpose

The `gsplats.io` package provides I/O operations for persisting and loading Gaussian splat data in a dedicated zarr format (`.gsplats.zarr`). This enables efficient storage, compression, and retrieval of fitted Gaussian splat results.

**Related Specifications**:
- `luxar.encoding` - Array encoding and semantic types (see `../../encoding/SPECIFICATIONS.md`)
- `luxar.io` - Spatial ordering algorithms (see `../../io/SPECIFICATIONS.md`)
- `luxar.core` - GSplats node specification (see `../../core/SPECIFICATIONS.md`)
- `luxar.gsplats` - Gaussian splatting algorithms (see `../SPECIFICATIONS.md`)

---

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
| `colors` | (N, 3) or (1, 3) | float32/uint8 | COLOR | RGB colors (optional); uint8 [0-255] for SDR, float32 for HDR; absent if not present |
| `sharpnesses` | (N,) or (1,) | float32 | BOUNDED_SCALAR | Generalized Gaussian exponent (s=2 is standard, bounds [0, 31]); optional |

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

## Zarr Structure

```
fitted.gsplats.zarr/
├── .zattrs                      # Format metadata (see below)
├── .zmetadata                   # Consolidated metadata for fast loading
│
├── splats/                      # Core splat data
│   ├── centers                  # (N, d) float32, spatially ordered
│   ├── amplitudes               # (N,) or (1,) float32, spatially ordered
│   ├── cholesky_factors         # (N, k) or (1, k) float32, spatially ordered
│   ├── colors                   # (N, 3) or (1, 3) float32/uint8, spatially ordered (optional)
│   ├── sharpnesses              # (N,) or (1,) float32, spatially ordered (optional)
│   ├── chunk_bounds             # (num_chunks, d, 2) float32, single chunk
│   └── .zattrs                  # n_splats, ndim, ordering info, spatial index metadata
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
  "timestamp": "2025-01-15T14:30:00Z",  // ISO 8601 format (creation time)
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
  "has_colors": true,
  "has_sharpness": true,
  "ordering": "morton",           // "morton", "hilbert", or "none"
  "morton_min": [0.0, 0.0, 0.0],  // Bounds for Morton normalization (all dimensions)
  "morton_max": [256.0, 256.0, 128.0],
  "morton_bits_per_dim": 21,      // Bits per dimension in Morton code
  "chunk_size": 2048,             // Elements per chunk
  "amplitude_range": {"min": 0.01, "max": 1.5},
  "sharpness_bounds": {"min": 0.0, "max": 31.0},
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
  "lr": 0.01,
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

## Spatial Ordering and Indexing

GSplats arrays are spatially ordered for compression and efficient spatial queries using the same algorithms as Points.

**Algorithm Reference**: See `../../io/SPECIFICATIONS.md` → "Spatial Index Specification" for complete details on:
- Morton/Hilbert ordering algorithms
- Compound ordering (discrete dimensions first, then Morton)
- Chunk bounds calculation

**Implementation Location**: Spatial ordering is implemented in `luxar.io` and reused by `gsplats.io` (single code path).

### GSplats-Specific Details

**Extent Calculation for Chunk Bounds**:

GSplats have ellipsoidal extent (unlike point radii). Chunk bounds include this extent:

```python
# For each dimension d, compute extent from Cholesky factors
# Covariance diagonal: covariance[d,d] = sum(L[start_idx + i]^2 for i in 0..d)
# For packed Cholesky (row-major):
#   2D: [L00, L10, L11] → cov[0,0]=L00², cov[1,1]=L10²+L11²
#   3D: [L00, L10, L11, L20, L21, L22] → cov[2,2]=L20²+L21²+L22²

extent[d] = sqrt(covariance[d, d]) * 3.0  # 3σ coverage (99.7%)

# Chunk bounds include extent
chunk_bounds[i, d, 0] = min(centers[chunk_i, d] - extent[chunk_i, d])
chunk_bounds[i, d, 1] = max(centers[chunk_i, d] + extent[chunk_i, d])
```

**Ordering Metadata** (stored in `splats/.zattrs`):
- `ordering`: "morton", "hilbert", or "none"
- `morton_min`, `morton_max`: Coordinate bounds for normalization
- `morton_bits_per_dim`: Bits allocated per dimension (typically 21 for 3D)

**Spatial Index Array**:
- `chunk_bounds`: (num_chunks, d, 2) float32 array
- Enables efficient spatial queries without loading splat data
- Same query algorithm as Points (AABB intersection test)

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

### Ordering Specification

The system supports both Morton and Hilbert ordering methods:

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
| `sharpnesses` | BOUNDED_SCALAR | `bounded_scalar_uint8` (8-bit, bounds [0, 31]) |

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

**Status**: Deferred to future version - initial implementation uses ordering + standard compression.

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

### Chunk Sizing

**Strategy**: Byte-based target converted to element counts (Zarr chunks by elements, not bytes).

**Constants** (from `../../typing_utils/SPECIFICATIONS.md` - single source of truth):
- `TARGET_CHUNK_BYTES = 65536` (64KB) - target chunk size

**IMPORTANT**: Zarr's chunking system operates on **element counts**, not byte counts. Therefore:
1. The byte target is defined once in `typing_utils`
2. At write time, we convert bytes → elements based on each array's dtype
3. Metadata stores the resulting **element count** (what Zarr needs)

**Bytes-to-elements conversion** (per array):
```python
from luxar.typing_utils import TARGET_CHUNK_BYTES  # 65536 (64KB)

# For each array, compute elements per chunk based on its specific layout:
# - centers (N, d) float32:     bytes_per_row = d * 4
# - amplitudes (N,) float32:    bytes_per_row = 4
# - cholesky_factors (N, k):    bytes_per_row = k * 4  (where k = d*(d+1)/2)
# - colors (N, 3) uint8/float32: bytes_per_row = 3 or 12
# - sharpnesses (N,) float32:   bytes_per_row = 4

chunk_elements = TARGET_CHUNK_BYTES // bytes_per_row
```

**Example: 3D splats with all arrays**:
| Array | Shape per row | Bytes/row | Elements/chunk (64KB) |
|-------|---------------|-----------|----------------------|
| centers | (3,) float32 | 12 | 5,461 |
| amplitudes | () float32 | 4 | 16,384 |
| cholesky_factors | (6,) float32 | 24 | 2,730 |
| colors | (3,) float32 | 12 | 5,461 |
| sharpnesses | () float32 | 4 | 16,384 |

**Note**: Each array has its own optimal chunk size. The `chunk_size` in group metadata is a **reference value** for the primary arrays (centers), not a universal constant.

**Chunk shape specification**:
```python
# 1D arrays (amplitudes, sharpnesses)
chunks = (chunk_elements,)

# 2D arrays (centers, cholesky_factors, colors)
chunks = (chunk_elements, n_cols)  # Keep all columns together
```

**Stored in metadata**: `splats/.zattrs["chunk_size"]` records the **element count** for reference (typically computed from centers array).

---

## Encoding Metadata Preservation

When arrays are written to `.gsplats.zarr`, encoding transformations are applied and metadata is preserved for automatic decoding on load.

### Encoding Metadata Storage

**Format**: Each array stores encoding metadata in its `.zattrs` file:
```json
// Example: sharpnesses/.zattrs
{
  "encoding": {
    "name": "bounded_scalar_uint8",
    "min": 0.0,
    "max": 31.0,
    "bits": 8,
    "original_dtype": "float32"
  }
}
```

**Color Mode Storage**:
For float32 colors, the `color_mode` is stored in encoding metadata:
```json
// colors/.zattrs - SDR colors
{
  "encoding": {
    "name": "rgb_uint8",
    "original_dtype": "float32",
    "color_mode": "sdr"
  }
}

// colors/.zattrs - HDR colors (no quantization)
{
  "encoding": {
    "name": "none",
    "color_mode": "hdr"
  }
}
```

**Broadcasting Metadata** (when all splats share same value):
```json
// amplitudes/.zattrs - all splats have amplitude=1.5
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

### Automatic Decoding on Load

When loading `.gsplats.zarr`:
1. Read array from zarr
2. Check for `encoding` metadata in `.zattrs`
3. Apply appropriate decoder based on `encoding.name`
4. Return decoded float32 array

**Transparency**: Encoding is a storage detail - users always work with float32 arrays. Quantization and broadcasting are transparent.

**Implementation**: Uses `luxar.encoding.ArrayDecoder` (see `../../encoding/SPECIFICATIONS.md` Section 11.4).

---

## API Design

### Saving

```python
from luxar.gsplats import GSplatData
from luxar.encoding import EncodingMode

result = fit_gaussian_splats(image, n_iters=1000)

# Simple save with defaults (hilbert ordering, zstd compression, auto encoding)
result.save("fitted.gsplats.zarr")

# With options
result.save(
    "fitted.gsplats.zarr",
    ordering="morton",           # or "hilbert", "none"
    encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
    color_mode="sdr",            # Required if colors present and float32: "sdr" or "hdr"
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
- `MEMORY`: Aggressive quantization for minimum storage (uses float16 only if float16_allowed=True)
- `CUSTOM`: Explicit encoder selection per array (advanced use)

**Float16 compatibility** (`float16_allowed` parameter in save_gsplats()):
- Default: `False` for TypeScript/WebGL compatibility (no native float16 support)
- When False, MEMORY mode uses float32 instead of float16 for coordinates, colors, cholesky factors
- Set to `True` only if decoder supports float16 natively

### Loading

```python
# Load for rendering (splats only)
result = GSplatData.load("fitted.gsplats.zarr")

# Load with all metadata
result = GSplatData.load(
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

## Standalone vs Embedded Formats

There are two ways to store Gaussian splats, serving different purposes:

### 1. Standalone Format (`.gsplats.zarr`)

**Purpose**: Persist fitted results as independent files

**Structure**: This specification
- Root container with format metadata
- `splats/` group with arrays
- Optional `fitting/` and `provenance/` groups
- Spatially ordered with `chunk_bounds`

**Use cases**:
- Save/load fitted results between sessions
- Share fitted splats with others
- Lightweight rendering without full scene
- Archive expensive computation results

### 2. Embedded Format (Luxar Scene)

**Purpose**: Multi-object visualization with scene graph

**Structure**: See `../../core/SPECIFICATIONS.md` Section 7
- GSplats as node in scene hierarchy
- Same arrays: centers, amplitudes, cholesky_factors, colors, sharpness
- Spatially ordered with `chunk_bounds`
- Inherits scene dimensions, transforms, rendering attributes

**Use cases**:
- Visualize splats alongside other data (points, lines)
- Apply hierarchical transforms
- Multi-layer scenes with groups

### Relationship

**Common Foundation**:
- Both use same spatial ordering (Morton/Hilbert) via `luxar.io`
- Both use same encoding system (`luxar.encoding`)
- Both store `chunk_bounds` for spatial queries
- Both use identical array structure and semantics

**Key Difference**: Container structure
- Standalone: Self-contained with provenance/fitting metadata
- Embedded: Part of larger scene graph with inherited attributes

**Code Reuse**: The `luxar.io` package provides the shared implementation:
- Spatial ordering functions
- Chunk bounds calculation
- Encoding application
- Both formats call the same underlying functions

**Future Integration**: `scene.add_gsplats()` will support loading from `.gsplats.zarr` files directly (planned).

---

## Luxar Integration (Future)

Once `.gsplats.zarr` format is stable, integrate with Luxar visualization:

```python
from luxar import LuxarZarrCompiler, Dimensions
from luxar.gsplats import GSplatData

# Option 1: From in-memory result
result = fit_gaussian_splats(image)
dims = Dimensions.default_3d()
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats("nuclei", result)

# Option 2: From saved .gsplats.zarr file
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats("nuclei", "fitted.gsplats.zarr")  # Path
```

This will be designed after the gsplats I/O module is complete.

---

## Design Decisions

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

## References

- [3DGS Compression Survey](https://arxiv.org/html/2502.19457v1) - Comprehensive overview of compression techniques
- [3DGS.zip Survey](https://arxiv.org/abs/2407.09510) - Another survey on compression methods
- [OMG: Optimized  MinimalGaussians](https://arxiv.org/html/2503.16924) - 100-300× compression
- [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve) - nD Hilbert implementation
- [Skilling 2004](https://doi.org/10.1063/1.1751381) - "Programming the Hilbert Curve" algorithm

---

## Changelog

- **v1.1.1** (2025-11-28): Chunk sizing source of truth
  - Removed local TARGET_CHUNK_BYTES redefinition
  - Now explicitly imports from `typing_utils` (single source of truth)
  - Code example shows `from luxar.typing_utils import TARGET_CHUNK_BYTES`

- **v1.1.0** (2025-11-28): Chunk sizing clarification
  - Clarified that Zarr chunks by elements, not bytes
  - Documented the bytes→elements conversion formula per array
  - Added example table showing different chunk sizes per array type
  - Clarified that `chunk_size` in metadata is an element count (Zarr's requirement)

- **v1.0.0** (2025-11-28): Initial versioned specification
  - Moved from `gsplats/GSPLATS_ZARR_FORMAT.md` to `gsplats/io/SPECIFICATIONS.md`
  - Converted from design document to technical specification format
  - Updated sharpness bounds from [0, 32] to [0, 31] for consistency
  - **Spatial ordering**: Added Morton/Hilbert ordering with chunk_bounds (aligned with embedded format)
  - **Byte-based chunking**: Changed from element-based (8192) to byte-based (64KB target) for Luxar consistency
  - **Colors support**: Added optional colors array to core data structure and zarr schema
  - **Color mode**: Specified color_mode storage in encoding metadata (sdr/hdr)
  - **Encoding metadata**: Documented how encoding metadata is preserved for automatic decoding
  - **Code reuse**: Specified that spatial ordering implementation lives in `luxar.io` (single code path)
  - **Format relationship**: Added section clarifying standalone vs embedded format relationship
  - **Cross-references**: Updated to use proper relative paths (../../)
  - Fitter-agnostic metadata design
  - Integration with `luxar.encoding` for semantic types
