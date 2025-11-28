# luxar.encoding

The `encoding` package provides semantic type definitions, data type configuration, and array encoding utilities for efficient storage and transmission of visualization data.

## Overview

This package handles the mapping between semantic data types (like "position", "color", "scalar") and concrete NumPy dtypes, with support for automatic optimization based on data characteristics. It enables efficient storage while preserving data fidelity.

## Purpose

The encoding system serves several critical functions:

1. **Semantic Type System**: Map high-level concepts (positions, colors) to appropriate dtypes
2. **Storage Optimization**: Choose optimal dtypes based on data range and precision needs
3. **Format Conversion**: Convert between different numeric representations (float32 ↔ uint8)
4. **Normalization**: Handle value range mapping (e.g., 0-1 float ↔ 0-255 uint8)
5. **HDR Support**: Preserve high dynamic range data where needed

## Key Concepts

### Semantic Types

Rather than directly specifying dtypes everywhere, Luxar uses semantic types:

- **COORDINATE/POSITION**: Spatial coordinates (typically float32 for precision)
- **COLOR**: RGB color values (uint8 for LDR, float32 for HDR)
- **POSITIVE_SCALAR**: Positive values like radii, widths (float32/float16/uint8)
- **BOUNDED_SCALAR**: Values in fixed range like sharpness (uint8/float16/float32)

This abstraction allows the encoding system to choose appropriate dtypes based on:
- Data value range
- Precision requirements
- Memory constraints
- Storage mode (precision vs memory)

### Data Type Modes

The package supports different optimization strategies:

1. **AUTO**: Automatically select dtype based on data analysis (default)
2. **PRECISION**: Prioritize precision (use float32 everywhere)
3. **MEMORY**: Prioritize memory efficiency (use uint8/float16 where safe)
4. **CUSTOM**: Use explicitly specified dtypes

## Key Components

### 1. DataTypeConfig (`datatypes.py`)

Configuration class for controlling dtype selection across all attributes.

**Purpose:**
Centralized control over how data is encoded for storage.

**Usage Example:**
```python
from luxar.encoding import DataTypeConfig, DataTypeMode

# Auto mode (default) - analyzes data to choose dtypes
config = DataTypeConfig(mode=DataTypeMode.AUTO)

# Precision mode - always use float32
config = DataTypeConfig(mode=DataTypeMode.PRECISION)

# Memory mode - use smallest safe dtypes
config = DataTypeConfig(mode=DataTypeMode.MEMORY)

# Custom mode - explicit control
config = DataTypeConfig(
    mode=DataTypeMode.CUSTOM,
    position_dtype='float32',
    color_dtype='uint8',
    radius_dtype='float16',
    sharpness_dtype='uint8'
)

# Query dtypes (may analyze data array if in AUTO mode)
positions = np.random.randn(1000, 3)
dtype = config.get_position_dtype(positions)
```

**Methods:**
- `get_position_dtype(data=None)` - Get dtype for positions
- `get_color_dtype(data=None)` - Get dtype for colors (HDR-aware)
- `get_radius_dtype(data=None)` - Get dtype for radii
- `get_sharpness_dtype(data=None)` - Get dtype for sharpness

**Pre-configured Instances:**
```python
from luxar.encoding import DEFAULT_CONFIG, PRECISION_CONFIG, MEMORY_CONFIG

# Ready-to-use configurations
DEFAULT_CONFIG   # mode=AUTO
PRECISION_CONFIG # mode=PRECISION
MEMORY_CONFIG    # mode=MEMORY
```

### 2. DataTypeMode (`datatypes.py`)

Enum defining dtype selection strategies.

**Values:**
- `AUTO` - Analyze data and choose optimal dtype
- `PRECISION` - Use float32 for everything
- `MEMORY` - Use smallest safe dtype (uint8/float16)
- `CUSTOM` - Use explicitly specified dtypes

**Usage:**
```python
from luxar.encoding import DataTypeMode

config = DataTypeConfig(mode=DataTypeMode.AUTO)
```

### 3. Type Aliases (`datatypes.py`)

Type-safe aliases for supported dtypes.

**Position Types:**
```python
PositionDType = Union[np.float32, np.float16]
PositionDTypeStr = Literal["float32", "float16"]
```

**Color Types:**
```python
ColorDType = Union[np.float32, np.uint8, np.uint16]
ColorDTypeStr = Literal["float32", "uint8", "uint16"]
```

**Scalar Types:**
```python
ScalarDType = Union[np.float32, np.float16, np.uint8]
ScalarDTypeStr = Literal["float32", "float16", "uint8"]
```

### 4. Conversion Functions (`datatypes.py`)

#### `convert_array_dtype()`

Convert arrays between dtypes with optional normalization.

**Purpose:**
Safe dtype conversion with value range mapping.

**Usage Example:**
```python
from luxar.encoding import convert_array_dtype

# Float to uint8 with normalization (0-1 → 0-255)
colors_float = np.array([[1.0, 0.5, 0.0], [0.2, 0.8, 0.6]])
colors_uint8 = convert_array_dtype(
    colors_float,
    target_dtype=np.uint8,
    normalize=True,
    input_range=(0.0, 1.0)
)

# Result: [[255, 127, 0], [51, 204, 153]]

# Uint8 to float with normalization (0-255 → 0-1)
restored = convert_array_dtype(
    colors_uint8,
    target_dtype=np.float32,
    normalize=True,
    input_range=(0.0, 1.0)
)

# HDR colors (no normalization, preserve >1.0 values)
hdr_colors = np.array([[2.5, 1.2, 0.8]])  # HDR values
hdr_stored = convert_array_dtype(
    hdr_colors,
    target_dtype=np.float32,
    normalize=False  # Keep HDR range
)
```

**Parameters:**
- `array` - Input array to convert
- `target_dtype` - Target NumPy dtype
- `normalize` - Whether to normalize during conversion
- `input_range` - Range for normalization (min, max)

**Conversion Types:**
- Float → uint8/uint16 (with/without normalization)
- Uint8/uint16 → float (with/without denormalization)
- Float16 ↔ float32 (precision change)

#### `infer_optimal_dtype()`

Analyze data and infer optimal dtype.

**Purpose:**
Automatic dtype selection based on data characteristics.

**Usage Example:**
```python
from luxar.encoding import infer_optimal_dtype

# Positions - need precision
positions = np.random.randn(1000, 3)
dtype = infer_optimal_dtype(positions, 'position')
# Returns: np.float32 (positions need precision)

# Colors in [0,1] - can use uint8
colors = np.random.rand(1000, 3)
dtype = infer_optimal_dtype(colors, 'color')
# Returns: np.uint8 (normalized colors)

# HDR colors - need float32
hdr_colors = np.random.rand(1000, 3) * 3.0  # Values > 1.0
dtype = infer_optimal_dtype(hdr_colors, 'color')
# Returns: np.float32 (HDR preservation)

# Small radii - can use uint8
radii = np.random.rand(1000) * 0.5  # Range [0, 0.5]
dtype = infer_optimal_dtype(radii, 'radius')
# Returns: np.uint8 (small range, normalized)
```

**Parameters:**
- `array` - Input array to analyze
- `attribute_type` - Semantic type: "position", "color", "radius", "sharpness"

**Returns:**
Optimal NumPy dtype for the array

#### `get_dtype_info()`

Get detailed information about a dtype.

**Purpose:**
Inspect dtype properties for debugging and validation.

**Usage Example:**
```python
from luxar.encoding import get_dtype_info

info = get_dtype_info(np.float32)
# Returns:
# {
#     'name': 'float32',
#     'bytes': 4,
#     'kind': 'f',
#     'range': (-3.4e38, 3.4e38),
#     'normalized': False
# }

info = get_dtype_info(np.uint8)
# Returns:
# {
#     'name': 'uint8',
#     'bytes': 1,
#     'kind': 'u',
#     'range': (0, 255),
#     'normalized': True  # WebGL can normalize to [0,1]
# }
```

#### `validate_dtype_string()`

Validate dtype string for attribute type.

**Purpose:**
Ensure dtype is valid for given semantic type.

**Usage Example:**
```python
from luxar.encoding import validate_dtype_string

# Valid combinations
validate_dtype_string('float32', 'position')  # True
validate_dtype_string('uint8', 'color')       # True
validate_dtype_string('float16', 'radius')    # True

# Invalid combination (raises ValueError)
try:
    validate_dtype_string('int32', 'position')
except ValueError as e:
    print(e)  # "Invalid dtype 'int32' for position. Valid: float32, float16"
```

## Supported Data Types

### Positions/Coordinates
- **float32** - Full precision (default)
- **float16** - Half precision (compact, but be careful with range)

### Colors
- **float32** - HDR colors, unlimited range
- **uint8** - Standard LDR colors (0-255), normalized to [0,1] in viewer
- **uint16** - High-precision LDR (0-65535), normalized to [0,1]

### Scalars (Radii, Widths, Amplitudes)
- **float32** - Full precision and range
- **float16** - Half precision (good for moderate ranges)
- **uint8** - Normalized to [0,1] or specific range

### Sharpness
- **float32** - Full precision
- **float16** - Adequate for typical range
- **uint8** - Mapped to [0, 15] range (sufficient for most cases)

## Encoding Workflow

### 1. Writer Configuration

The LuxarZarrCompiler accepts a DataTypeConfig:

```python
from luxar import LuxarZarrCompiler
from luxar.encoding import DataTypeConfig, DataTypeMode

# Create compiler with encoding config
config = DataTypeConfig(mode=DataTypeMode.MEMORY)
with LuxarZarrCompiler('output.zarr', dtype_config=config) as compiler:
    scene = compiler.create_scene()
    # All data will use memory-optimized dtypes
```

### 2. Automatic Dtype Selection

In AUTO mode, dtypes are selected per-array:

```python
config = DataTypeConfig(mode=DataTypeMode.AUTO)

# Array 1: Standard colors [0,1]
colors1 = np.random.rand(1000, 3)
# → Stored as uint8 (memory efficient)

# Array 2: HDR colors [0,3]
colors2 = np.random.rand(1000, 3) * 3.0
# → Stored as float32 (preserves HDR)

# Array 3: Small radii [0,1]
radii = np.random.rand(1000)
# → Stored as uint8 (normalized)

# Array 4: Large radii [0,100]
large_radii = np.random.rand(1000) * 100
# → Stored as float32 (range too large for uint8)
```

### 3. Manual Dtype Control

For precise control, use CUSTOM mode:

```python
config = DataTypeConfig(
    mode=DataTypeMode.CUSTOM,
    position_dtype='float32',    # High precision positions
    color_dtype='float32',       # HDR colors
    radius_dtype='float16',      # Compact radii
    sharpness_dtype='uint8'      # Very compact sharpness
)
```

## HDR Color Support

The encoding system has special handling for HDR (High Dynamic Range) colors:

### Detection
```python
# Standard colors [0,1] → uint8
colors = np.random.rand(1000, 3)
dtype = infer_optimal_dtype(colors, 'color')
# Returns: np.uint8

# HDR colors [0,∞) → float32
hdr_colors = np.array([[2.5, 1.5, 0.8], [0.2, 3.0, 0.5]])
dtype = infer_optimal_dtype(hdr_colors, 'color')
# Returns: np.float32 (preserves values > 1.0)
```

### Conversion
```python
# HDR-aware conversion
hdr_data = np.array([[1.5, 2.0, 0.5]])

# DO NOT normalize HDR colors
float_colors = convert_array_dtype(
    hdr_data,
    target_dtype=np.float32,
    normalize=False  # Critical: preserve HDR range
)
```

## Performance Considerations

### Memory Savings

Different dtypes have different memory footprints:

```python
# Example: 1M points with colors

# float32 colors: 1M × 3 × 4 bytes = 12 MB
# uint8 colors:   1M × 3 × 1 byte  = 3 MB  (75% savings!)

# float32 radii:  1M × 4 bytes = 4 MB
# float16 radii:  1M × 2 bytes = 2 MB  (50% savings)
# uint8 radii:    1M × 1 byte  = 1 MB  (75% savings)
```

### Precision Trade-offs

```python
# float32 range: ±3.4e38, precision: ~7 decimal digits
# float16 range: ±65504, precision: ~3 decimal digits
# uint8 range: 0-255, precision: 256 discrete values
# uint16 range: 0-65535, precision: 65536 discrete values
```

### Conversion Overhead

```python
# Conversion has minimal overhead
colors_float = np.random.rand(1000000, 3)

# Fast conversion (vectorized NumPy operations)
colors_uint8 = convert_array_dtype(colors_float, np.uint8, normalize=True)
# Typical time: <10ms for 1M points
```

## Best Practices

### 1. Use AUTO Mode by Default
```python
# Let the system analyze and optimize
config = DataTypeConfig(mode=DataTypeMode.AUTO)
```

### 2. Use PRECISION Mode for Critical Data
```python
# When accuracy is paramount
config = DataTypeConfig(mode=DataTypeMode.PRECISION)
```

### 3. Use MEMORY Mode for Large Datasets
```python
# When dataset size is a concern
config = DataTypeConfig(mode=DataTypeMode.MEMORY)
```

### 4. Preserve HDR Colors
```python
# Don't normalize HDR data
hdr_colors = load_hdr_image()  # Values > 1.0
config = DataTypeConfig(mode=DataTypeMode.PRECISION)  # Use float32
```

### 5. Validate Before Conversion
```python
# Check if conversion is safe
from luxar.encoding import validate_dtype_string

validate_dtype_string('uint8', 'color')  # OK
validate_dtype_string('int32', 'color')  # Raises ValueError
```

### 6. Document Encoding Choices
```python
# When using CUSTOM mode, document why
config = DataTypeConfig(
    mode=DataTypeMode.CUSTOM,
    position_dtype='float32',  # Precision needed for large coordinates
    color_dtype='float32',     # HDR emission data
    radius_dtype='float16',    # Range [0,1000], float16 sufficient
    sharpness_dtype='uint8'    # Range [0,15], uint8 optimal
)
```

## Dependencies

**Internal:**
- `luxar.typing_utils` - Type definitions and protocols

**External:**
- `numpy` - Array operations and dtypes
- Standard library only otherwise

## Testing

Tests are located in `encoding/tests/`:
- `test_datatypes.py` - DataTypeConfig and mode tests
- `test_conversion.py` - Array conversion tests
- `test_inference.py` - Optimal dtype inference tests

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/encoding/tests/
```

## Implementation Notes

### WebGL Compatibility

The encoding system considers WebGL capabilities:
- Uint8/uint16 can be auto-normalized to [0,1] in shaders
- Float32 is native WebGL type
- Float16 support varies (emulated if needed)

### Normalization Strategy

For integer types, normalization maps:
- Uint8: [0, 255] ↔ [0.0, 1.0]
- Uint16: [0, 65535] ↔ [0.0, 1.0]

Custom ranges can be specified:
```python
# Map [0,255] to [0.5, 1.5]
convert_array_dtype(
    data,
    np.float32,
    normalize=True,
    input_range=(0.5, 1.5)
)
```

### Precision Loss Warning

When AUTO mode selects a lower-precision dtype, it ensures:
1. Value range fits in target dtype
2. Precision loss is acceptable for semantic type
3. Conversion is reversible (with quantization error)

## See Also

- [core/README.md](../core/README.md) - Core data structures
- [io/README.md](../io/README.md) - I/O operations and writers
- [validation/README.md](../validation/README.md) - Validation utilities
- [typing_utils/README.md](../typing_utils/README.md) - Type system
- [Main README](../../../../README.md) - Project overview
