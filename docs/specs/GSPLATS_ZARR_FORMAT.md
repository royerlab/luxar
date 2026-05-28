# luxar.gsplats.io - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2026-03-27

## Purpose

The `gsplats.io` package provides I/O operations for persisting and loading Gaussian splat data in a dedicated zarr format (`.gsplats.zarr`). This enables efficient storage, compression, and retrieval of fitted Gaussian splat results.

**Related Specifications**:
- [Luxar Zarr Format](../guides/user/LUXAR_ZARR_FORMAT.md) — Scene-level format including GSplats nodes
- [GSplats Dimension Mapping](GSPLATS_DIMENSION_MAPPING.md) — Dimension mapping for nD scenes
- [nD Transforms](../guides/specs/ND_TRANSFORMS_SPEC.md) — nD navigation and transforms

> **Scope note.** This format is a Python-side **processing artifact** —
> the output of `luxar gsplat fit`, `lod additive`, `lod substitutive`,
> `lod pyramid`, etc. The viewer never consumes a `.gsplats.zarr`
> directly. When gsplats land in a **scene** (loadable by the viewer),
> the substitutive axis is not carried inline; only the additive ladder
> is. Substitutive levels are dropped at scene-write time (only the
> default substitutive level survives) — see
> ``Group.add_gsplats_from_data``.

---

## Use Cases

1. **Save/Load fitted results** - Persist expensive fitting results for later use
2. **Lightweight rendering** - Load only splat data for visualization
3. **Provenance tracking** - Record what image/parameters produced the splats
4. **2-D LOD storage** - Single self-describing file carrying any of:
   - one splat set (trivial `[1, 1]` case)
   - an additive ladder (`[1, M]` — same N splats sliced into prefix-sum levels)
   - a substitutive pyramid (`[N, 1]` — synthesised representative splats per level)
   - a full 2-D pyramid (`[N, M_i]` — substitutive levels each with their own additive ladder)
5. **Future: Checkpoint/Resume** - Pause and resume fitting (deferred)

## Core Data Structure

Each Gaussian splat is parameterized by:

| Field | Shape | Dtype | Semantic Type | Description |
|-------|-------|-------|---------------|-------------|
| `centers` | (N, d) | float32 | COORDINATE | Splat center positions (not broadcastable) |
| `amplitudes` | (N,) or (1,) | float32 | POSITIVE_SCALAR | Non-negative intensity |
| `cholesky_factors` | (N, d*(d+1)/2) or (1, d*(d+1)/2) | float32 | CHOLESKY | Packed lower-triangular L where Σ = LLᵀ |
| `colors` | (N, 3) or (1, 3) | float32/uint8 | COLOR | RGB colors (optional); uint8 [0-255] for SDR, float32 for HDR; absent if not present |

**Note**: Cholesky factors are packed in row-major order. For d=3: `[L00, L10, L11, L20, L21, L22]`

**Semantic Types**: Each field maps to an encoding semantic type (see `luxar.encoding.SemanticType`). This determines valid encodings and quantization options for each array.

### Broadcasting Convention

Broadcasting uses the standard `luxar.encoding` format. When all elements share the same value, the array is stored with shape `(1,)` or `(1, d)` with encoding metadata:

```json
// amplitudes/.zattrs - all splats have amplitude=1.0
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

The `n_splats` attribute in `splats/.zattrs` always reflects the true count (N), regardless of broadcasting.

---

## Format Versions

The current format is **v2.0**, a 2-D ``substitutive × additive`` layout.
Earlier versions (v1.0 single-LOD; v1.1 additive multi-LOD; pre-v2.0 substitutive
directory + manifest.json) are no longer read by the runtime — convert legacy
files with ``luxar gsplat migrate-format <input> <output.gsplats.zarr>``.

A v2.0 file always carries at least one substitutive level and at least one
additive sub-LOD per level. The four canonical shapes are:

| Shape         | Meaning                                                      |
|---------------|--------------------------------------------------------------|
| `[1, 1]`      | One splat set (the "trivial" case — output of plain `fit`)   |
| `[1, M]`      | One substitutive level with an M-step additive ladder        |
| `[N, 1]`      | N substitutive levels, each a single flat splat set          |
| `[N, M_i]`    | Full pyramid: N substitutive levels, each with its own ladder|

---

## Zarr Structure (v2.0 — 2-D substitutive × additive)

```
fitted.gsplats.zarr/
├── .zattrs                                 # Format metadata: format_version: "2.0",
│                                           # format_type, timestamp, luxar_gsplats_version,
│                                           # description, n_substitutive, default_substitutive
├── .zmetadata                              # Consolidated metadata for fast loading
│
├── splats/                                 # Splat container (always a substitutive_<s> hierarchy)
│   ├── .zattrs                             # type: "gsplats", n_substitutive,
│   │                                       # default_substitutive, truncation_radius,
│   │                                       # n_additive_sublods_default, center_bounds, …
│   │
│   ├── substitutive_0/                     # Finest substitutive level (= the original/finest)
│   │   ├── .zattrs                         # n_additive_sublods, compression_factor=1,
│   │   │                                   # parent_method=null, level_index=0, level_stats?
│   │   ├── additive_0/                     # Coarsest additive sub-LOD
│   │   │   ├── centers                     # (N_{0,0}, d) float32, spatially ordered
│   │   │   ├── amplitudes                  # (N_{0,0},) float32
│   │   │   ├── cholesky_factors            # (N_{0,0}, k) float32
│   │   │   ├── colors                      # (N_{0,0}, 3) float32/uint8, optional
│   │   │   ├── chunk_bounds                # (num_chunks, d, 2) float32
│   │   │   └── .zattrs                     # n_splats, ndim, ordering, lod_stats, …
│   │   ├── additive_1/                     # Only present when M_0 > 1
│   │   └── additive_{M_0-1}/
│   │
│   ├── substitutive_1/                     # K^1 splats (e.g. K=4)
│   │   ├── .zattrs                         # compression_factor=K, parent_method, level_index
│   │   └── additive_0/                     # Typically a single flat set; M_i can vary per level
│   │
│   └── substitutive_{N-1}/                 # K^(N-1) splats (coarsest)
│
├── fitting/                                # Optimization info (optional, unchanged from v1.x)
│   ├── .zattrs                             # time_seconds, iterations, converged, psnr_db, …
│   └── config/.zattrs                      # Fitter hyperparameters
│
└── provenance/                             # Image lineage (optional, unchanged from v1.x)
    └── .zattrs                             # source_file, shape, dtype, normalization
```

### Root Attributes (.zattrs)

```json
{
  "format_version": "2.0",
  "format_type": "gsplats_zarr",
  "timestamp": "2026-05-20T10:00:00Z",
  "luxar_gsplats_version": "X.Y.Z",
  "description": "Optional user description",
  "n_substitutive": 3,
  "default_substitutive": 0
}
```

**Note**: Core per-cell splat metadata (`n_splats`, `ndim`, `ordering`, …) lives
on each `splats/substitutive_<s>/additive_<a>/.zattrs`. The outer
`splats/.zattrs` carries pyramid-wide attributes (`n_substitutive`,
`default_substitutive`, `truncation_radius`, `n_additive_sublods_default`).

### Splats Group Attributes (Single Source of Truth)

```json
{
  "n_splats": 10000,
  "ndim": 3,
  "has_colors": true,
  "truncation_radius": 3.0,      // Gaussian truncation in sigmas (default 3.0 if absent)
  "ordering": "morton",           // "morton", "hilbert", or "none"
  "morton_min": [0.0, 0.0, 0.0],  // Bounds for Morton normalization (all dimensions)
  "morton_max": [256.0, 256.0, 128.0],
  "morton_bits_per_dim": 21,      // Bits per dimension in Morton code
  "chunk_size": 2048,             // Elements per chunk
  "amplitude_range": {"min": 0.01, "max": 1.5},
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
  "asymmetric_penalty": 1.0,
  "init_sigma_vox": 0.5,
  "seed_method": "auto",
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

**Algorithm Reference**: See `packages/luxar/src/luxar/io/README.md` and the implementation under `luxar.io` for complete details on:
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

extent[d] = sqrt(covariance[d, d]) * truncation_radius  # default 3.0 (3σ = 99.7%)

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
def sort_splats_spatial(
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

**Log-scale amplitudes**: For high dynamic range (HDR) amplitudes, use log encoding:
```python
result.save(
    "hdr_splats.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,
    positive_scalar_encoding="log",  # log1p/expm1 for numerical stability
)
```

See `packages/luxar/src/luxar/encoding/README.md` for complete quantization details and precision guarantees.

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

### 4. Culling Before Storage

Not a storage format concern, but worth noting:
- Remove low-amplitude splats before saving
- User can cull before saving: `result.cull(method="cumulative", retention=0.95).save(...)`

---

## Compression Configuration

### Blosc Settings

Default zarr compressor configuration:

```python
compressor = Blosc(
    cname='zstd',      # Best compression ratio
    clevel=3,          # Balance speed/ratio (1-9)
    shuffle=Blosc.BITSHUFFLE,  # Good for float arrays
)
```

### Chunk Sizing

**Strategy**: Byte-based target converted to element counts (Zarr chunks by elements, not bytes).

**Constants** (from `luxar.typing_utils.constants` - single source of truth):
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

chunk_elements = TARGET_CHUNK_BYTES // bytes_per_row
```

**Example: 3D splats with all arrays**:
| Array | Shape per row | Bytes/row | Elements/chunk (64KB) |
|-------|---------------|-----------|----------------------|
| centers | (3,) float32 | 12 | 5,461 |
| amplitudes | () float32 | 4 | 16,384 |
| cholesky_factors | (6,) float32 | 24 | 2,730 |
| colors | (3,) float32 | 12 | 5,461 |

**Note**: Each array has its own optimal chunk size. The `chunk_size` in group metadata is a **reference value** for the primary arrays (centers), not a universal constant.

**Chunk shape specification**:
```python
# 1D arrays (amplitudes)
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
// Example: amplitudes/.zattrs
{
  "encoding": {
    "name": "positive_scalar_uint8",
    "min": 0.0,
    "max": 10.0,
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

**Implementation**: Uses `luxar.encoding.ArrayDecoder` (see `packages/luxar/src/luxar/encoding/README.md`).

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

**Encoding modes** (see `packages/luxar/src/luxar/encoding/README.md`):
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
from luxar.gsplats.io import inspect_gsplats_zarr

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

**Structure**: See `packages/luxar/src/luxar/core/README.md`
- GSplats as node in scene hierarchy
- Same arrays: centers, amplitudes, cholesky_factors, colors
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

**Scene Integration**: `scene.add_gsplats_from_file()` loads `.gsplats.zarr` files directly, preserving LOD structure for multi-LOD data.

---

## Luxar Scene Integration

The `.gsplats.zarr` format integrates with the Luxar scene system.
Single-LOD and multi-LOD data are both supported:

```python
from luxar import LuxarZarrCompiler, Dimensions
from luxar.gsplats import fit_gaussian_splats, fit_progressive_gaussian_splats

dims = Dimensions.default_3d()

# Single-pass fitting → flat gsplats node
result = fit_gaussian_splats(image, n_iters=1000)
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_data("nuclei", result)

# Progressive fitting → multi-LOD gsplats node (per-LOD subgroups)
result = fit_progressive_gaussian_splats(image, max_splats=50000)
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_data("nuclei", result)  # auto-detects multi-LOD

# From saved .gsplats.zarr file (preserves LOD structure)
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_file("nuclei", "fitted.gsplats.zarr")

# Fit-and-add in one step (supports progressive=True)
with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_volume("nuclei", image, progressive=True)
```

Multi-additive-LOD scene nodes use per-sub-LOD subgroups
(``additive_0/``, ``additive_1/``, …) flat under the gsplats node —
no substitutive wrapper. The viewer streams these progressively
(prefix-sum LODs). Substitutive levels carried by a ``GSplatData``
input are dropped at scene-write time; only the default substitutive
level's additive ladder is written into the scene.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File extension | `.gsplats.zarr` | Clear, descriptive |
| Encoding system | `luxar.encoding` package | Shared infrastructure, semantic types |
| Ordering methods | Both Morton and Hilbert | User choice, Hilbert default |
| Resolution | Auto with cap at 2^16 | Safe default, configurable |
| Chunk size | 64KB (TARGET_CHUNK_BYTES = 65536) | Optimal for blosc |
| Broadcasting | Standard encoding metadata | Consistent with other Luxar formats |
| Covariance storage | Cholesky (packed) | Already have it, compresses well |
| Fitting info | Fitter-agnostic design | Allows other programs to use format |
| Checkpoint/Resume | Deferred | Focus on basic I/O first |
| 2-D LOD | v2.0 `substitutive_<s>/additive_<a>/` cells | Substitutive (replacement) and additive (extension) compose as a 2-D matrix in a single file |
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

- **v3.0.0** (2026-05-20): `.gsplats.zarr` format v2.0 — 2-D LOD matrix
  - On-disk format bumped to v2.0; legacy v1.0 / v1.1 / pre-v2.0 substitutive
    directory layouts no longer read at runtime. Convert with
    `luxar gsplat migrate-format <input> <output.gsplats.zarr>`.
  - Splat container is now a `substitutive × additive` matrix laid out at
    `splats/substitutive_<s>/additive_<a>/`. The four canonical pyramid shapes
    `[1, 1]`, `[1, M]`, `[N, 1]`, `[N, M_i]` all live in a single self-describing
    file (the pre-v2.0 substitutive directory + manifest.json layout is retired).
  - Root attrs surface `n_substitutive` and `default_substitutive`; the splats
    group surfaces `n_additive_sublods_default`. Per-substitutive-level attrs
    (`compression_factor`, `parent_method`, `level_index`, `n_additive_sublods`)
    live on the `substitutive_<s>/` groups; per-cell attrs (`n_splats`, `ndim`,
    `ordering`, `lod_stats`, …) live on the leaf `additive_<a>/` groups.
  - Python: `GSplatLOD` renamed to `AdditiveSubLOD`; new `SubstitutiveLevel`
    dataclass; `GSplatData` refactored around `substitutive_levels`. Scene
    embedding writes the same layout (`splats/substitutive_0/additive_<i>/`).
  - CLI: `lod substitutive` now writes a single v2.0 file (no more directory
    + manifest.json); `lod additive` gains `--substitutive-level`; new
    `lod pyramid` builds the full 2-D pyramid in one call.
  - Viewer: TypeScript loader walks `substitutive_<defaultSub>/additive_<i>/`;
    legacy v1.x reading path removed.

- **v2.0.0** (2026-03-27): Multi-LOD format (v1.1) and scene integration
  - Added format v1.1 with per-LOD subgroups (`splats/lod_0/`, `splats/lod_1/`, ...)
  - Produced by `fit_progressive_gaussian_splats()` (iterative residual decomposition)
  - LODs are additive: render LODs 0..L to get cumulative approximation at level L
  - Each LOD has independent spatial ordering, chunk bounds, and per-LOD stats
  - Loader auto-detects v1.0 vs v1.1 from `format_version` in root attrs
  - Scene API now writes multi-LOD nodes with per-LOD subgroups (matching standalone format)
  - Updated Luxar Integration section — `add_gsplats_from_data()`, `add_gsplats_from_volume(progressive=True)`, `add_gsplats_from_file()` all preserve LOD structure
  - Replaced "Future: Multiscale storage (deferred)" with implemented multi-LOD support

- **v1.2.0** (2026-03-18): Removed sharpness from GSplats
  - Removed `sharpnesses` array from core data structure, zarr schema, and all examples
  - GSplats now use fixed standard Gaussian falloff (equivalent to sharpness=2.0)
  - Removed `has_sharpness` and `sharpness_bounds` from splats group attributes

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
  - Relocated from the old `gsplats/GSPLATS_ZARR_FORMAT.md` path
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
