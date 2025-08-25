# I/O Package

The `io` package handles all input/output operations for Luxar, focusing on progressive writing to Zarr format for memory-efficient processing of massive datasets.

## Overview

This package provides the infrastructure for writing point cloud data progressively to Zarr archives, enabling processing of TB-scale datasets on GB-scale machines.

## Modules

### `compiler.py`
Main entry point for creating Luxar scenes with progressive writing.

**Key Classes:**
- `LuxarZarrCompiler`: Context manager for progressive scene compilation

**Key Features:**
- Progressive writing to Zarr with automatic chunking
- Intelligent chunk size calculation based on data patterns
- Memory-efficient processing (data written immediately, not cached)
- Context manager protocol for resource safety
- Metadata consolidation for fast loading
- Scene creation with dimension support
- **Automatic data type optimization**: Configurable dtype selection for memory efficiency

**Usage:**
```python
from luxar.io import LuxarZarrCompiler

with LuxarZarrCompiler('output.zarr') as compiler:
    scene = compiler.create_scene()
    # Add data progressively
    scene.add_points('points', positions, colors)
```

### `streaming.py`
Streaming API for datasets larger than RAM.

**Key Classes:**
- `StreamingPoints`: Special node for appending point data in batches

**Key Features:**
- Append data in arbitrary-sized batches
- No loading of existing data into memory
- Automatic shape tracking and validation
- Support for all point attributes (positions, colors, radii, sharpness)

**Usage:**
```python
from luxar.io import StreamingPoints

streaming = StreamingPoints('huge_cloud', compiler)
for batch in data_generator():
    streaming.append_batch(
        positions=batch['positions'],
        colors=batch['colors']
    )
streaming.finalize()
```

### `writer.py`
Protocol and base implementation for Zarr writers.

**Key Classes:**
- `ZarrWriterProtocol`: Protocol defining the writer interface

**Key Features:**
- Protocol-based design for flexibility
- Support for different Zarr backends
- Chunk and compression configuration
- Metadata management

### `reader.py`
Constants and utilities for reading Zarr data.

**Key Constants:**
- `DEFAULT_COMP`: Default Blosc compressor configuration

## Architecture

### Progressive Writing Flow
```
User Code → Scene API → Writer Protocol → Zarr Store
                ↓
          Node Metadata
```

1. User creates data (positions, colors, etc.)
2. Scene API validates and prepares data
3. Writer immediately persists to Zarr
4. Only metadata kept in memory

### Memory Management
- **Zero-copy writing**: Data goes directly to disk
- **Chunked storage**: Optimal chunk sizes for access patterns
- **Streaming support**: Process unlimited data in fixed memory

### Zarr Structure
```
output.zarr/
├── .zmetadata          # Consolidated metadata
├── scene/
│   ├── .zattrs        # Scene attributes
│   └── points_0/
│       ├── .zattrs    # Point cloud attributes
│       ├── positions/ # (N, D) array (float32/float16)
│       ├── colors/    # (N, 3) array (float32 for HDR, uint8/uint16 for SDR)
│       ├── radii/     # (N,) array (float32/float16/uint8)
│       └── sharpness/ # (N,) array (float32/float16/uint8)
```

## Performance Considerations

### Chunk Size Selection
The compiler automatically selects chunk sizes based on:
- Data dimensionality
- Access patterns (full vs. sliced)
- Memory constraints
- Compression efficiency

Default: 32,768 elements per chunk

### Compression
- Default: Blosc with zstd codec, level 3
- Bit-shuffle filter for better compression
- Configurable per dataset

### Best Practices
1. Use context manager to ensure cleanup
2. Process data in batches for large datasets
3. Let the compiler choose chunk sizes
4. Use StreamingPoints for append-only workflows

## Examples

### Basic Scene Creation
```python
from luxar.io import LuxarZarrCompiler
import numpy as np

with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene()
    
    # Add 1M points
    positions = np.random.randn(1_000_000, 3).astype(np.float32)
    colors = np.random.rand(1_000_000, 3).astype(np.float32)
    
    scene.add_points('cloud', positions, colors)
```

### Streaming Large Dataset
```python
from luxar.io import LuxarZarrCompiler, StreamingPoints
import numpy as np

with LuxarZarrCompiler('huge.zarr') as compiler:
    scene = compiler.create_scene()
    streaming = StreamingPoints('trajectory', compiler)
    
    # Stream 1B points in 1M batches
    for i in range(1000):
        batch_positions = np.random.randn(1_000_000, 3).astype(np.float32)
        batch_colors = np.random.rand(1_000_000, 3).astype(np.float32)
        
        streaming.append_batch(
            positions=batch_positions,
            colors=batch_colors
        )
        
        print(f"Processed batch {i+1}/1000")
    
    streaming.finalize()
```

### Custom Compression
```python
from luxar.io import LuxarZarrCompiler
from numcodecs import Blosc

compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

with LuxarZarrCompiler('fast.zarr', compressor=compressor) as compiler:
    scene = compiler.create_scene()
    # Fast compression for real-time data
```

## Data Type Optimization

The compiler supports automatic data type optimization to reduce memory usage:

### DataTypeConfig Modes
- **AUTO** (default): Automatically selects optimal dtype based on data range
- **PRECISION**: Uses float32 for maximum precision
- **MEMORY**: Uses smallest viable dtype (float16/uint8)
- **CUSTOM**: Use explicitly specified dtypes

### Example with Memory Optimization
```python
from luxar.io import LuxarZarrCompiler
from luxar.typing_utils.datatypes import DataTypeConfig, DataTypeMode

# Use memory-efficient dtypes
dtype_config = DataTypeConfig(mode=DataTypeMode.MEMORY)

with LuxarZarrCompiler('efficient.zarr', dtype_config=dtype_config) as compiler:
    scene = compiler.create_scene()
    # SDR colors will be stored as uint8
    # Positions as float16
    # Radii/sharpness as uint8
```

Memory savings can be 30-65% compared to all-float32 storage.

## Dependencies

Internal:
- `core`: Scene graph nodes
- `typing_utils`: Type definitions and data type configuration
- `validation`: Data validation

External:
- `zarr`: Storage backend
- `numcodecs`: Compression codecs
- `numpy`: Array operations
- `arbol`: Progress logging