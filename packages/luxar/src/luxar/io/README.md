# luxar.io - I/O Operations

Progressive writing and reading of Luxar Zarr scenes with spatial ordering for memory-efficient processing of massive datasets.

## Quick Start

Write and read a scene in 3 steps:

```python
from luxar.io import LuxarZarrCompiler, LuxarScene
from luxar.core.dimensions import Dimensions
import numpy as np

# 1. Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)
dims = Dimensions.default_3d()

# 2. Write to zarr (progressive - data written immediately)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("cloud", positions, colors, radii=0.1)

# 3. Read it back (memory-efficient lazy loading)
scene = LuxarScene.load("scene.luxar.zarr")
points = scene.get_points("cloud")
print(f"Loaded {points['positions'].shape[0]} points")
```

**Key Benefits**:
- Progressive writing - no intermediate caching, handle TB-scale data
- Spatial ordering - better compression and viewer performance
- Scalar convenience - pass uniform values directly: `radii=0.5` instead of `np.full(N, 0.5)`
- Memory-efficient reading - lazy loading via zarrita

## Purpose

This package provides infrastructure for:
- **Writing**: Progressive writing of Points, Lines, GSplats, and Mesh data to Zarr archives
- **Reading**: Full read-only access to Luxar scenes via `LuxarScene`

Both paths enable processing of TB-scale datasets on GB-scale machines.

## Main Components

### LuxarZarrCompiler

Main entry point for creating Luxar scenes with progressive writing.

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with (
    LuxarZarrCompiler(
        "scene.luxar.zarr",
        encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
        ordering_method="hilbert",  # "hilbert" (default, best locality) or "morton" (fastest)
        enable_spatial_index=True,  # Apply spatial ordering
    ) as compiler
):
    scene = compiler.create_scene(dimensions=dims)

    # Full arrays
    scene.add_points("cloud", positions, colors, radii)

    # Scalar convenience - no intermediate arrays!
    scene.add_points(
        "uniform",
        positions,
        radii=0.5,  # Scalar instead of np.full(N, 0.5)
        colors=(1.0, 0, 0),  # Tuple instead of np.full((N,3), [1,0,0])
        sharpness=0.5,
    )  # Scalar instead of np.full(N, 0.5)
```

**Key Features**:
- Progressive writing (data written immediately, not cached)
- **Scalar convenience**: Pass uniform values directly (no `np.full()` needed)
- Morton/Hilbert spatial ordering for better compression
- Compound ordering for nD data (discrete/barrier dims → spatial curve within)
  - Shared `_compound_sort` core drives **all three geometries** (Points, Lines,
    GSplats): categorical/barrier axes (time, channel) are lexsorted first, then
    a space-filling curve orders spatially *within* each barrier value, so a
    chunk never straddles a category. GSplats derive the barrier authoritatively
    where known — the scene compiler uses the scene's `Dimension.discrete`
    non-displayed axes (exactly like Points/Lines), and the batch merge uses the
    stacked-time axis — falling back to the persisted LOD `coarsen_dims`
    complement, then a conservative value-based auto-detect (`detect_barrier_dims`,
    strict integer + low-cardinality) only for provenance-less standalone files.
- **Dimension-aware spatial indexing**: Optimized for time-series and nD slicing
  - Tight discrete/barrier bounds (categorical axes get exact-value bounds
    padded only by a float-boundary epsilon `_BARRIER_BOUND_EPS`, no σ/radius
    expansion — a splat at time=0 never extends into time=1's chunk bounds;
    the reader's per-dimension query tolerance owns the reach)
  - Smart chunk sizing (smaller chunks for animated data)
  - ~7× performance improvement for time-animated Lines; barrier-aware GSplat
    ordering makes per-timepoint reads hit the ideal ~1/T of chunks (vs a
    chaotic 2.5–3.5× over-fetch when time is smeared across chunks)
- Semantic type-based encoding (via `luxar.encoding`)
- Automatic chunk size calculation
- Metadata consolidation for fast loading

### LuxarScene (Reading)

Read-only access to Luxar zarr scenes with automatic decoding.

```python
from luxar.io import LuxarScene

# Load a scene
scene = LuxarScene.load("scene.luxar.zarr")

# Scene metadata
print(scene.version)  # "0.1" (LUXAR_VERSION_CURRENT)
print(scene.dimensions)  # Dimensions object or None
print(scene.path)  # Path to zarr store

# List nodes by type
print(scene.list_points())  # ['cloud1', 'cloud2']
print(scene.list_gsplats())  # ['splats1']
print(scene.list_lines())  # []
print(scene.list_groups())  # ['group1']

# Check if a node exists
if scene.has_node("cloud1"):
    print(scene.get_node_type("cloud1"))  # 'points'

# Get node metadata (without loading array data)
metadata = scene.get_node_metadata("cloud1")
print(metadata["type"])  # 'points'
print(metadata["n_points"])  # Number of points

# Get full point data with automatic decoding
points = scene.get_points("cloud1")
print(points["positions"].shape)  # (N, 3)
print(points["colors"].shape)  # (N, 3) or None
print(points["radii"].shape)  # (N,) or None
print(points["metadata"]["transform"])  # 4x4 numpy array (if present)

# Similarly for GSplats and Lines
splats = scene.get_gsplats("splats1")
print(splats["centers"].shape)  # (N, 3)
print(splats["cholesky_factors"].shape)  # (N, 6)

# A node whose colormap attr is the sentinel 'custom' carries its palette as a
# LUT dataset (the writer resolves any non-builtin name or array to that pair)
print(scene.get_colormap_lut("cloud1"))  # (256, 3) uint8, or None

# Scene-level viewer hints, parsed like Scene.viewer_config (None if unset).
# A rewriter must carry this across, or the output silently falls back to the
# viewer's ACES default.
print(scene.viewer_config)  # ViewerConfig(...) or None
```

**Return Types** (dataclasses with dict-compatible access):
- `PointsData`: positions, colors, radii, sharpness, chunk_bounds, metadata
- `LinesData`: vertices, widths, colors, sharpness, segments, metadata, indices
- `GSplatsData`: centers, amplitudes, cholesky_factors, colors, chunk_bounds, metadata

**Key Features**:
- Automatic decoding (broadcasting, LUT, quantization, array_ref)
- Full scene introspection (list nodes, check types, get metadata)
- Lazy loading (arrays only loaded when requested)
- Node types as strings ('points', 'gsplats', 'lines', 'group')
- Transform matrices automatically converted from zarr storage format

### Spatial Ordering Module

`luxar.io.ordering` provides Morton and Hilbert curve ordering.

```python
from luxar.io.ordering import sort_points_compound, compute_chunk_bounds_points

# Apply compound ordering to Points
sort_indices, metadata = sort_points_compound(
    positions,
    dimensions,  # List of Dimension objects
    method="hilbert",  # or "morton"
)

# Compute chunk bounds (radii can be per-point array or scalar)
chunk_bounds = compute_chunk_bounds_points(
    sorted_positions,
    sorted_radii,
    chunk_size=2048,
)
```

**Compound Ordering** (for nD Points):
1. **Primary sort**: Discrete dimensions (time, channel) - lexicographic
2. **Secondary sort**: Morton/Hilbert code of spatial dimensions

This ensures:
- Time-slice queries load contiguous chunks
- Spatial locality preserved within each slice
- Works with both Morton and Hilbert curves

**Functions**:
- `sort_points_compound()`: Compound ordering for Points
- `sort_splats_spatial()`: Simple spatial ordering for GSplats
- `sort_segments_compound()`: Compound ordering for Lines segments
- `compute_chunk_bounds_points()`: Chunk bounds with radius extent (`radii=None` ⇒ the renderer's `DEFAULT_POINT_RADIUS`, so the bound is never tighter than the drawn disc)
- `compute_chunk_bounds_gsplats()`: Chunk bounds with ellipsoidal extent
- `morton_encode_nd()`: Morton (Z-order) encoding
- `morton_encode_128bit()`: Morton encoding for 128-bit coordinates
- `hilbert_encode_nd()`: Hilbert curve encoding

### Writer Protocol

`ZarrWriterProtocol` defines the interface for Zarr writers, enabling different implementations while maintaining API consistency.

### Re-chunking an existing store (`optimise.py`)

`luxar.io.optimise` re-chunks a store that is **already on disk**, in one
structure-preserving pass — no refit, no source volume, no GPU. It backs the
`luxar optimise` CLI command and the `luxar info --stats` chunk diagnostic.

Everything but the zarr chunk grid survives verbatim: values bit-for-bit, dtype,
codecs, filters, serializer, `fill_value`, memory order, the on-disk zarr format,
and every group and array attribute **except** the two the pass is contractually
required to move — the root's `content_hash`, which is restamped, and the
`chunk_layout` summary written beside it (see *Cache invalidation* below).

```python
from luxar.io.optimise import optimise_store, plan_optimisation, summarise_chunk_layout

plan = optimise_store("scene.luxar.zarr", "out.luxar.zarr", target_bytes=65_536,
                      verify=True)
print(plan.source_n_chunks, "→", plan.target_n_chunks)
```

- `plan_optimisation(root, target_bytes=…)` → `OptimisePlan` — what would change,
  per array, without writing. Each `ArrayPlan` carries the source/target chunk
  shape, the resolved spatial atom, and a `skip_reason` when the array is left
  alone.
- `optimise_store(src, dst, …)` — does it. `verify=True` re-reads the output and
  compares every array byte for byte.
- `summarise_chunk_layout(root)` → `ChunkLayoutSummary` — average chunk bytes,
  arrays under the 16 KB floor, and the chunk-file count a full load fetches.
  Counts objects, so a shard is one file and a `(0, D)` placeholder is none —
  and an array that fetches nothing is left out of the floor share entirely.
  `summarise_plan(plan)` is the same diagnostic off a plan already walked, which
  is how `luxar info --stats` reports both from a single pass.
- `resolve_target_bytes(target_bytes=…, target_kb=…, profile=…)` — the three
  mutually exclusive size flags, and `CHUNK_PROFILES` (`hosting` 256 KB,
  `local` 64 KB, `archive` 1 MB).

**What must not move.** `chunk_size` / `chunk_bounds` are the viewer's partition
grid: every emitted chunk is a whole multiple of the node's atom (rounded down
from the byte budget, never below one atom), so a row-range read never straddles
a boundary. Lines' two atoms — `vertex_ordering.chunk_size` for the per-vertex
arrays, `segment_ordering.chunk_size` for `segments` — are resolved separately,
and the bounds arrays themselves are never re-chunked. A `chunk_size` attr is
trusted **only** when the matching bounds array exists, because a gsplat leaf
written with `ordering="none"` still gets a vestigial one stamped — and that
vestigial value is a power of two, so "the array's chunk is already a multiple
of it" is arithmetic coincidence rather than proof. (Every Luxar writer omits
the bounds array only for a zero-row node, so a node with rows to re-chunk
always carries its proof.) `array_ref` placeholders (`(0, D)`), `(1,)`/`(1, k)`
broadcasts and sharded arrays are copied verbatim — a sharded array keeps its
**shard** grid, not just its inner chunk shape; nothing is chunked smaller than
it already is; the chunk **key** layout (v2 `dimension_separator` / v3
`chunk_key_encoding`) survives; and the copy walks chunk-aligned slabs rather
than reading an array whole.

**All-or-nothing.** The output is built in a hidden sibling directory and
renamed onto the destination only after the copy (and `verify=True`, when asked
for) succeeds, so a failure leaves neither a half-written store at the
destination nor a damaged previous one. An existing destination is renamed
**aside** and deleted only once the new one is in place (and restored if that
rename fails), because deleting first can lose both copies: an `rmtree` that
fails partway propagates before the artifact is marked consumed, and for a
directory destination the artifact *is* the staging tree. A `.zarr.zip`
destination is compressed out of that directory rather than written into a
`ZipStore`, which appends rather than replaces and would otherwise accumulate
one dead copy of every group document per attr write. `overwrite=True` replaces
an existing **zarr store or empty directory** only, and refuses a destination
that is, contains, or lives inside the source — or that is a **symlink**, since
the rename would replace the link rather than its target (the error says to pass
the target instead).

**Cache invalidation.** Neither hasher can see a chunk shape —
`compute_content_hashes` hashes values + attrs, the `.gsplats.zarr` stamp hashes
`(name, shape, dtype)` + attrs — while the viewer validates its persistent cache
on `content_hash`. So the `chunk_layout` root attr is what MOVES the hash (attrs
are hashed) and the restamp is what propagates it into the stored
`content_hash`; both are load-bearing, and the restamp runs for `--generic` too,
because that flag describes the input rather than the output's cache safety. The
scene restamp is a slab-wise reimplementation of the finalize-time walk — the
same digest, without the whole-array materialisation that would peak at twice a
629 MB array's size.

### Input Volume Loading

`luxar.io.volume` and `luxar.io.ome_zarr` load arbitrary input volumes (the
sources fed to gsplat fitting/calibration), independent of the compiled
`.luxar.zarr` scene format above:

- `volume.load_volume(path, channel=, timepoint=, array_key=, axes=, info=)` —
  reads `.npy` / `.npz` / `.zarr` / `.zarr.zip` / `.tiff` / imageio-supported
  files to a float32 volume, with OME-Zarr-aware positional slicing and an
  explicit `--axes` override. Missing optional readers raise `ImportError` (the
  CLI turns it into a clean exit). Pass an `info` dict to get back
  `source_dtype`, the element type the array was **stored** in — the return is
  always float32, so this is the last point at which it is knowable, and it is
  the honest denominator of any size/compression figure quoted about the result.
- `ome_zarr.discover_ome_zarr_shape(path, ...)` → `OMEZarrInfo` — discovers the
  T/C/Z/Y/X layout, voxel size, unit, and resolution levels from NGFF
  `multiscales` (with custom-`axes` and shape-heuristic fallbacks).

These are domain-layer helpers (no CLI dependency); the gsplat CLI re-exports
them. Dimension inference from a splat bounding box lives in
`luxar.core.dimension_inference.build_dimensions_from_data`.

## Encoding Integration

The I/O layer uses `luxar.encoding` for semantic type-aware array encoding:

```python
# Automatically applied by compiler:
# - positions → COORDINATE (float16 in MEMORY mode)
# - colors → COLOR (rgb_uint8 for SDR, geolog_perchannel_u16 for HDR)
# - radii → POSITIVE_SCALAR (log_scalar_uint8 for wide ranges)
# - sharpness → BOUNDED_SCALAR (bounded_scalar_uint8, bounds [0, 1])
```

**Encoding Modes**:
- `AUTO`: Analyzes data and selects encoding (may quantize)
- `PRECISION`: Full float32, lossless (broadcasting still allowed)
- `MEMORY`: Aggressive quantization for minimum storage

See `luxar.encoding` package for complete encoding system documentation.

## Spatial Ordering

### Morton vs Hilbert

**Morton (Z-order)**:
- Fast bit-interleaving algorithm
- Simple implementation
- Good compression

**Hilbert**:
- Better locality preservation
- ~10% better compression
- **Default** for Points, Lines, and GSplats (`ordering_method="hilbert"`)

Both encoders use a Numba JIT kernel when available (with a vectorized NumPy
fallback for Morton and a pure-Python `hilbertcurve` fallback for Hilbert), so
neither method needs an extra dependency to run.

### Metadata Structure

Morton/Hilbert ordering adds metadata to point/splat groups:

```json
{
  "ordering": "hilbert",
  "slice_dims": [3, 4],           // Discrete dimensions (time, channel)
  "ordering_dims": [0, 1, 2],     // Spatial dimensions (X, Y, Z)
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [100.0, 100.0, 100.0],
  "ordering_bits_per_dim": 21,
  "chunk_size": 2048
}
```

Plus a `chunk_bounds` array: shape `(num_chunks, n_dims, 2)` with min/max bounds per chunk.

## Architecture

### Progressive Writing Flow

```
User Code → Scene API → Compiler → Spatial Ordering → Encoding → Zarr Store
                                        ↓                  ↓
                                   sort_points_compound  ArrayEncoder
                                   compute_chunk_bounds  (semantic types)
```

1. User creates data arrays
2. Scene API validates data
3. Compiler applies spatial ordering (if enabled)
4. Arrays encoded using semantic types
5. Data written immediately to Zarr
6. Only metadata kept in memory

### Reading Flow

```
Zarr Store → LuxarScene → ArrayDecoder → Decoded Arrays → User Code
                  ↓
            Node introspection
            Dimension parsing
            Transform conversion
```

1. Open zarr store read-only
2. Parse scene metadata (version, dimensions)
3. Build node index from zarr groups
4. On `get_*()` call: load arrays with automatic decoding
5. Return decoded data with proper numpy shapes/dtypes

### nD Transforms and Bounds Expansion

When nodes have `nd_transform` attributes (affine scale/offset or categorical permutations on non-displayed dimensions), the compiler stores them in each node's zarr group attrs. During `finalize()`, the compiler:

1. Walks the zarr tree to compose world nd_transforms (root to leaf)
2. Applies `apply_nd_transform_to_bounds()` to each leaf node's local position bounds
3. Computes the union of all world-space bounds
4. Stores the result as `position_bounds` in root attrs

This means scene-level `position_bounds` has:
- **Displayed dims**: local-space bounds (for camera auto-framing)
- **Non-displayed dims**: world-space bounds (for slider auto-ranging)

Per-node `position_bounds` remain in local space. See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for the full specification.

### Memory Management

- **Zero-copy writing**: Data goes directly to disk
- **Chunked storage**: Optimal chunk sizes (64KB target)
- **Spatial ordering**: Morton/Hilbert ordering with compound sorting for nD data

## Examples

### Basic Scene with Spatial Ordering

```python
from luxar.io import LuxarZarrCompiler
from luxar.core import Dimensions, Dimension

# Define nD dimensions
dims = Dimensions(
    [
        Dimension("X", unit="um", display=True),
        Dimension("Y", unit="um", display=True),
        Dimension("Z", unit="um", display=True),
        Dimension("Time", discrete=True, display=False),  # Discrete dimension
    ]
)

with LuxarZarrCompiler("scene.luxar.zarr", ordering_method="hilbert") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Data will be compound-sorted: Time first, then Hilbert(X,Y,Z)
    scene.add_points("cells", positions_4d, colors, radii)
```

### Memory-Optimized Scene

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with LuxarZarrCompiler(
    "compressed.luxar.zarr",
    encoding_mode=EncodingMode.MEMORY,  # Aggressive quantization
    ordering_method="hilbert",
) as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    scene.add_points("cloud", positions, colors, radii)
    # Positions: float16
    # Colors: uint8 (if SDR)
    # Radii: log_scalar_uint8
    # With Hilbert ordering for better compression
```

### Round-Trip (Write and Read)

```python
from luxar.io import LuxarZarrCompiler, LuxarScene
import numpy as np

# Write data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)

with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    scene.add_points("cloud", positions, colors, radii=0.1)

# Read it back
scene = LuxarScene.load("scene.luxar.zarr")
points = scene.get_points("cloud")

# Verify data (accounting for encoding precision)
np.testing.assert_allclose(points["positions"], positions, atol=1e-5)
np.testing.assert_allclose(points["colors"], colors, atol=1e-5)
```

## Performance

Typical compression ratios (vs uncompressed float32):

| Configuration | Compression | Notes |
|---------------|-------------|-------|
| PRECISION + Morton | 2-3x | Lossless, ordering helps blosc |
| AUTO + Hilbert | 4-6x | Selective quantization |
| MEMORY + Hilbert | 8-12x | Aggressive quantization |

Compression gains from:
1. **Spatial ordering** (~2x from blosc on ordered data)
2. **Quantization** (2-4x from float32→float16/uint8)
3. **Broadcasting** (massive savings for uniform values)
4. **LUT encoding** (up to 75% for <256 unique values)
5. **Array deduplication** (via xxhash64)

## Dependencies

**Internal**:
- `luxar.core`: Scene graph nodes and dimensions
- `luxar.encoding`: Semantic type-based encoding
- `luxar.typing_utils`: Constants and type definitions
- `luxar.validation`: Data validation

**External**:
- `zarr>=3.2,<4`: Storage backend. Note the library version and the on-disk
  format are separate axes — Luxar writes zarr **format 3** by default
  (`LUXAR_ZARR_FORMAT=2` still produces format 2) and READS both, so existing
  format-2 stores keep working untouched. Both axes are pinned in
  `luxar._zarr_compat`, the only module that names a zarr format; go through its
  helpers (`open_group`, `create_array`, `consolidate`, ...) rather than calling
  `zarr.*` directly. Code that inspects a store on disk should use its
  bi-format readers (`read_array_meta`, `read_node_attrs`, `is_consolidated`,
  `read_consolidated_attrs`) instead of naming `.zarray` / `.zattrs` /
  `.zmetadata`, which exist in only one of the two formats. Editing a store in
  place likewise goes through `open_group`: re-opening an already-consolidated
  store with plain `zarr.open_group` and re-consolidating writes a nested
  consolidated index holding the pre-edit attributes, which later reads then
  serve in preference to the (correct) documents on disk.
- `numpy>=2.0`: Array operations
- `numcodecs`: Blosc compressor (`DEFAULT_COMP`)
- `arbol>=0.3.5`: Progress logging
- `xxhash>=3.0.0`: Array deduplication hashing
- `hilbertcurve>=2.0.5`: Pure-Python Hilbert fallback when Numba is unavailable
- `numba` (optional): JIT-accelerated Morton/Hilbert encoding kernels

## Related Documentation

- **Encoding system**: `../encoding/README.md` (semantic types, quantization)
- **Core structures**: `../core/README.md` (Scene graph, dimensions)
- **Ordering module**: `ordering.py` (Morton/Hilbert implementations)
