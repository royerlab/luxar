# luxar.typing_utils - Technical Specification

## Purpose

The `typing_utils` package centralizes all type definitions, constants, protocols, and configuration for Luxar. It provides the type system foundation for the entire codebase.

---

## Type System Organization

### aliases.py - Type Aliases
**Purpose**: Simple type aliases for improved readability

**Array Types**:
- `Float32Array = NDArray[np.float32]`
- `Uint8Array = NDArray[np.uint8]`
- `PositionArray = Float32Array`  # (N, D) shape
- `ColorArray = Float32Array | Uint8Array`  # (N, 3) shape
- `RadiusArray = Float32Array`  # (N,) shape
- `SharpnessArray = Float32Array`  # (N,) shape

**Transform Types**:
- `TransformMatrix = Float32Array`  # 4x4 matrix (NumPy format)
- `TransformList = List[float]`  # 16 elements (storage format)

**Other**:
- `PathLike = str | Path`
- `ChunkSpec = None | bool | int | Tuple[int, ...]`
- `NodePath = str`  # Hierarchical path like "group1/group2/points"

### protocols.py - Protocol Definitions
**Purpose**: Define structural contracts for duck typing

**Protocols**:
- `CompressorProtocol`: Zarr compressor interface (encode/decode)
- `NodeProtocol`: Scene graph node interface
- `PointsProtocol`: Points node interface
- `SceneProtocol`: Scene root interface

**Generic Type Variables**:
- `NodeT = TypeVar('NodeT', bound=NodeProtocol)`
- `NumericT = TypeVar('NumericT', bound=np.generic)`
- `ZarrDataT = TypeVar(np.float32 | np.uint8 | np.int32 | ...)`

**Note**: protocols.py also imports and re-exports validation functions from validation/types.py

### enums.py - Enumeration Types
**Purpose**: Type-safe string constants

**BlendingMode**:
- `NORMAL = "normal"` - Standard alpha blending
- `ADDITIVE = "additive"` - Colors add together (glow effect)

**NodeType**:
- `SCENE = "scene"` - Root node
- `GROUP = "group"` - Container node
- `POINTS = "points"` - Point cloud node

**PhysicalUnit**:
- Metric: nm, um, mm, cm, m, metre, km
- Imperial: inch, foot
- Other: px, au
- Includes .validate() method with normalization (meter/metre, micrometer/micron/μm)

**RenderingLimits** (class with constants):
- OPACITY_MIN/MAX, GAMMA_MIN/MAX, SHARPNESS_MIN/MAX
- COLOR_SDR_MIN/MAX, COLOR_HDR_MAX

**Defaults** (class with constants):
- OPACITY, GAMMA, SHARPNESS, BLENDING_MODE
- CHUNK_SIZE, RADIUS
- COLOR_WHITE, COLOR_BLACK

---

## Constants (constants.py)

### Version Constants
- `LUXAR_VERSION_CURRENT = "0.1"` - Current format version
- `DEFAULT_ZARR_VERSION = "0.1"` - Default when creating new stores

### Rendering Constants
- `OPACITY_MIN/MAX = 0.0, 1.0`
- `GAMMA_MIN/MAX = 0.2, 2.0`
- `SHARPNESS_MIN/MAX = 0.0, 15.0`
- `DEFAULT_OPACITY/GAMMA/BLENDING_MODE = 1.0, 1.0, "additive"`

### Data Constants
- `CHUNK_SIZE_DEFAULT = 32768` - 32KB elements per chunk
- `CHUNK_SIZE_MIN/MAX = 1024, 1048576` - Valid range
- `COMPRESSION_LEVEL_DEFAULT = 3` - Blosc compression level
- `DEFAULT_COMPRESSOR = "blosc"` - Default algorithm

### Spatial Index Constants
- `SPATIAL_INDEX_MAX_CELLS_DISCRETE = 10000` - Max for categorical dims
- `SPATIAL_INDEX_MAX_CELLS_CONTINUOUS = 10` - Max for continuous dims
- `SPATIAL_INDEX_MIN_CELLS = 3` - Minimum cells
- `SPATIAL_INDEX_TARGET_POINTS_PER_CELL = 10000` - Target density
- Plus 5 more for multi-dim grids, single-dim grids, fallback values

### Point Radius Constants
- `MIN_POINT_RADIUS = 0.001` - Minimum visible
- `MAX_POINT_RADIUS = 1000.0` - Maximum practical

### Dimension Constants
- `MAX_DISPLAYED_DIMENSIONS = 3` - Viewer limitation
- `DEFAULT_DIMENSION_STEP_PERCENT = 0.01` - 1% of range

---

## Configuration (config.py)

### DataTypeConfig

**Purpose**: Configure which dtypes to use for storage

**Modes**:
- `AUTO`: Automatically select based on data range (default)
- `PRECISION`: Always use float32 (maximum precision)
- `MEMORY`: Use smallest viable dtype (float16/uint8)
- `CUSTOM`: Use explicitly specified dtypes

**Methods**:
- `get_position_dtype(data)` - Usually float32 (accuracy critical)
- `get_color_dtype(data)` - Auto: float32 if HDR, uint8 if SDR
- `get_radius_dtype(data)` - Auto: based on value range
- `get_sharpness_dtype(data)` - Auto: uint8 if ≤15, else float32

**Optimization Strategy**:
- HDR detection: Check if any color > 1.0
- Range detection: Check min/max for appropriate dtype
- Normalization: uint8 can represent [0,1] range or [0,15] for sharpness

### Other Config

- `DEFAULT_LOG_LEVEL = "INFO"` - For arbol
- `SUPPORTED_VERSIONS = ("0.1", "0.2", "0.3")` - For format compatibility
- `SUPPORTED_COMPRESSION = ("blosc", "zstd", "lz4", "gzip", "bz2", "lzma")`

---

## Data Type Conversion (datatypes.py)

### convert_array_dtype(array, target_dtype, normalize, input_range)

**Purpose**: Convert between dtypes with optional normalization

**Key Conversions**:

**Float → uint8** (with normalization):
```
normalized = (array - input_min) / (input_max - input_min)
result = clip(normalized * 255, 0, 255).astype(uint8)
```

**uint8 → Float** (with denormalization):
```
normalized = array / 255.0
result = normalized * (output_max - output_min) + output_min
```

**Float16 ↔ Float32**: Direct conversion

**Normalization Ranges**:
- Colors: [0, 1] typical input range
- Sharpness: [0, 15] for uint8 mapping
- Radii: [0, max_radius] for uint8 mapping

### infer_optimal_dtype(array, attribute_type)

**Heuristics**:
- Position: Always float32 (accuracy matters)
- Color: float32 if any > 1.0 (HDR), else uint8
- Radius/Sharpness: Based on value range (uint8 if small, float16 if medium, float32 if large)

**Returns**: Optimal numpy dtype for the array

---

## Type Guards and Validators

**All validators now in validation/types.py, re-exported from protocols.py**

**Purpose of Re-export**: Maintain import paths for backward compatibility while centralizing implementation

---

## Constant Usage Patterns

**Where Constants Used**:
1. `io/compiler.py` - Spatial index grid sizing, chunking
2. `io/point_spatial_index.py` - Grid calculation, cell limits
3. `validation/base.py` - Range validation (opacity, gamma, etc.)
4. Throughout codebase - Default values, validation bounds

**Benefits**:
- Self-documenting code (names explain purpose)
- Easy tuning (change in one place)
- Consistent values (no duplicate magic numbers)

---

## Memory Estimation

**Function**: `estimate_memory_usage(n_points, has_colors)`

**Formula**:
```
memory_per_point = 12 bytes (positions: 3 * float32)
if has_colors:
    memory_per_point += 3 bytes (colors: 3 * uint8)

total = n_points * memory_per_point
```

**Warning Thresholds**:
- LARGE_DATASET_WARNING = 1,000,000 points
- MAX_RECOMMENDED_POINTS = 10,000,000 points

---

## This specification provides sufficient detail to re-implement the type system and configuration management.
