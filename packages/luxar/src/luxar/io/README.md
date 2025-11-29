# luxar.io - I/O Operations

Progressive writing to Zarr format with spatial ordering for memory-efficient processing of massive datasets.

## Purpose

This package provides infrastructure for writing Points and GSplats data progressively to Zarr archives, enabling processing of TB-scale datasets on GB-scale machines.

## Main Components

### LuxarZarrCompiler

Main entry point for creating Luxar scenes with progressive writing.

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with LuxarZarrCompiler(
    'scene.zarr',
    encoding_mode=EncodingMode.AUTO,      # AUTO, PRECISION, or MEMORY
    ordering_method="morton",             # "morton" or "hilbert"
    enable_spatial_index=True,            # Apply spatial ordering
) as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points('cloud', positions, colors, radii)
```

**Key Features**:
- Progressive writing (data written immediately, not cached)
- Morton/Hilbert spatial ordering for better compression
- Compound ordering for nD data (discrete dims → spatial curve)
- Semantic type-based encoding (via `luxar.encoding`)
- Automatic chunk size calculation
- Metadata consolidation for fast loading

### Spatial Ordering Module

**New in v1.3.0**: `luxar.io.ordering` provides Morton and Hilbert curve ordering.

```python
from luxar.io.ordering import sort_points_compound, compute_chunk_bounds_points

# Apply compound ordering to Points
sort_indices, metadata = sort_points_compound(
    positions,
    dimensions,  # List of Dimension objects
    method="hilbert",  # or "morton"
)

# Compute chunk bounds
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
- `compute_chunk_bounds_points()`: Chunk bounds with radius extent
- `compute_chunk_bounds_gsplats()`: Chunk bounds with ellipsoidal extent
- `morton_encode_nd()`: Morton (Z-order) encoding
- `hilbert_encode_nd()`: Hilbert curve encoding

### Writer Protocol

`ZarrWriterProtocol` defines the interface for Zarr writers, enabling different implementations while maintaining API consistency.

## Encoding Integration

The I/O layer uses `luxar.encoding` for semantic type-aware array encoding:

```python
# Automatically applied by compiler:
# - positions → COORDINATE (float16 in MEMORY mode)
# - colors → COLOR (rgb_uint8 for SDR, float32 for HDR)
# - radii → POSITIVE_SCALAR (log_scalar_uint8 for wide ranges)
# - sharpness → BOUNDED_SCALAR (bounded_scalar_uint8, bounds [0, 31])
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
- **Default for Points**

**Hilbert**:
- Better locality preservation
- ~10% better compression
- Requires `hilbertcurve` package
- **Default for GSplats**

### Metadata Structure

Morton/Hilbert ordering adds metadata to point/splat groups:

```json
{
  "ordering": "morton",
  "slice_dims": [3, 4],         // Discrete dimensions (time, channel)
  "morton_dims": [0, 1, 2],     // Spatial dimensions (X, Y, Z)
  "morton_min": [0.0, 0.0, 0.0],
  "morton_max": [100.0, 100.0, 100.0],
  "morton_bits_per_dim": 21,
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
dims = Dimensions([
    Dimension("X", unit="um", display=True),
    Dimension("Y", unit="um", display=True),
    Dimension("Z", unit="um", display=True),
    Dimension("Time", discrete=True, display=False),  // Discrete dimension
])

with LuxarZarrCompiler('scene.zarr', ordering_method="hilbert") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Data will be compound-sorted: Time first, then Hilbert(X,Y,Z)
    scene.add_points('cells', positions_4d, colors, radii)
```

### Memory-Optimized Scene

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with LuxarZarrCompiler(
    'compressed.zarr',
    encoding_mode=EncodingMode.MEMORY,  // Aggressive quantization
    ordering_method="hilbert",
) as compiler:
    scene = compiler.create_scene()
    scene.add_points('cloud', positions, colors, radii)
    # Positions: float16
    // Colors: uint8 (if SDR)
    // Radii: log_scalar_uint8
    // With Hilbert ordering for better compression
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
- `zarr>=2.16`: Storage backend
- `numpy>=2.0`: Array operations
- `arbol>=0.3.5`: Progress logging
- `hilbertcurve>=2.0.5`: Hilbert curve ordering (optional, for Hilbert method)

## Related Documentation

- **Technical specification**: `SPECIFICATIONS.md` (Morton/Hilbert algorithms, chunk sizing)
- **Encoding system**: `../encoding/SPECIFICATIONS.md` (semantic types, quantization)
- **Core structures**: `../core/SPECIFICATIONS.md` (Scene graph, dimensions)
- **Ordering module**: `ordering.py` (Morton/Hilbert implementations)
