# luxar.typing_utils - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2025-11-29

## Purpose

The `typing_utils` package centralizes all type definitions, constants, protocols, and configuration for Luxar. It provides the type system foundation for the entire codebase.

**Related Specifications**:
- `luxar.core` - Uses type definitions from this package (see `core/SPECIFICATIONS.md`)
- `luxar.io` - Uses constants and types (see `io/SPECIFICATIONS.md`)
- `luxar.validation` - Uses constants for bounds validation (see `validation/SPECIFICATIONS.md`)
- `luxar.encoding` - Uses type aliases (see `encoding/SPECIFICATIONS.md`)

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
- `CategoryList = List[str] | None`  # Category labels for categorical dimensions

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

**Note**: Validation functions have been moved to `validation/types.py`. Import from there directly (protocols.py no longer re-exports them).

### enums.py - Enumeration Types
**Purpose**: Type-safe string constants

**BlendingMode**:
- `NORMAL = "normal"` - Standard alpha blending
- `ADDITIVE = "additive"` - Colors add together (glow effect)
- `MAX = "max"` - Maximum of source and destination (brightest wins)
- `OPAQUE = "opaque"` - Solid rendering with depth write (closest object wins)
- `LUMINOUS = "luminous"` - Same visual as additive, but respects depth occlusion

**NodeType**:
- `SCENE = "scene"` - Root node
- `GROUP = "group"` - Container node
- `POINTS = "points"` - Point cloud node
- `LINES = "lines"` - Line primitives node
- `GSPLATS = "gsplats"` - Gaussian splat node

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
- `GAMMA_MIN/MAX = 0.1, 10.0` (symmetric: gamma and 1/gamma have equal range)
- `INTENSITY_MIN/MAX = 0.0, 100.0` (per-node color multiplier)
- `OFFSET_MIN/MAX = -10.0, 10.0` (per-node color offset)
- `SHARPNESS_MIN/MAX = 0.001, 31.0` (SHARPNESS_MIN is practical minimum; values must be > 0)
- `DEFAULT_OPACITY/GAMMA/BLENDING_MODE = 1.0, 1.0, "additive"`

### Data Constants

**Chunk Size Constants** (SINGLE SOURCE OF TRUTH):
- `TARGET_CHUNK_BYTES = 65536` - 64KB target chunk size
- `MIN_CHUNK_BYTES = 16384` - 16KB minimum to amortize HTTP overhead
- `MAX_CHUNK_BYTES = 262144` - 256KB maximum for responsive streaming

**Note**: Zarr chunks by elements, not bytes. Consumers (io, gsplats.io) import these constants and convert to element counts based on each array's dtype. See `io/SPECIFICATIONS.md` for the conversion strategy.

**Compression Constants**:
- `COMPRESSION_LEVEL_DEFAULT = 3` - Blosc compression level
- `DEFAULT_COMPRESSOR = "blosc"` - Default algorithm

### Point Radius Constants
- `MIN_POINT_RADIUS = 0.001` - Minimum visible
- `MAX_POINT_RADIUS = 1000.0` - Maximum practical

### Dimension Constants
- `MAX_DISPLAYED_DIMENSIONS = 3` - Viewer limitation
- `DEFAULT_DIMENSION_STEP_PERCENT = 0.01` - 1% of range
- `MIN_CATEGORIES = 1` - Minimum categories for categorical dimensions
- `CATEGORICAL_STEP = 1.0` - Step size for categorical dimensions (always 1)

### Timestamp Format
- **Format**: ISO 8601 with UTC timezone
- **Pattern**: `datetime.datetime.now(datetime.timezone.utc).isoformat()`
- **Example output**: `"2025-11-28T14:30:00+00:00"`
- **Usage**: Fitting timestamps, provenance, creation dates

**Implementation pattern** (used in gsplats.io.save_gsplats):
```python
import datetime
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
# Result: "2025-11-28T14:30:00+00:00"
```

**Where used**:
- GSplats root metadata (`/.zattrs["timestamp"]`)
- Fitting metadata (`fitting/.zattrs["timestamp"]`)

---

## Configuration (config.py)

**Note**: Data type configuration (dtype selection, AUTO/PRECISION/MEMORY modes) has been moved to `luxar.encoding`. See `encoding/SPECIFICATIONS.md` for the `EncodingMode` enum and `ArrayEncoder` class.

### Configuration Constants

- `DEFAULT_CHUNK_SIZE` - From typing_utils.constants
- `DEFAULT_VERSION = "0.1"` - Current Luxar format version
- `SUPPORTED_VERSIONS = ("0.1", "0.2", "0.3")` - For format compatibility
- `DEFAULT_LOG_LEVEL = "INFO"` - For arbol
- `SUPPORTED_COMPRESSION = ("blosc", "zstd", "lz4", "gzip", "bz2", "lzma")`

### Performance Constants

- `MAX_RECOMMENDED_POINTS = 10_000_000` - Maximum recommended points
- `LARGE_DATASET_WARNING = 1_000_000` - Threshold for performance warning
- `MEMORY_PER_POINT_POSITIONS = 12` bytes (3 × float32)
- `MEMORY_PER_POINT_COLORS = 3` bytes (3 × uint8)

### Validation Constants

- `POSITION_SHAPE_DIMS = 2` - Array must be 2D
- `POSITION_SHAPE_CHANNELS = 3` - Default 3D positions
- `COLOR_SHAPE_CHANNELS = 3` - RGB colors
- `TRANSFORM_MATRIX_SIZE = (4, 4)` - 4×4 transform matrix

### Data Type Defaults

- `POSITION_DTYPE = "float32"` - Default for positions
- `COLOR_DTYPE = "float32"` - Default for colors (supports HDR)
- `TRANSFORM_DTYPE = "float32"` - Always float32 for accuracy
- `SUPPORTED_POSITION_DTYPES = ("float32", "float16")`
- `SUPPORTED_COLOR_DTYPES = ("float32", "uint8", "uint16")`
- `SUPPORTED_SCALAR_DTYPES = ("float32", "float16", "uint8")`

### Configuration Functions

#### validate_chunk_size(chunk_size) → int
Validate chunk size within MIN/MAX bounds. Raises ValueError if invalid.

#### validate_compression_level(level) → int
Validate compression level is 1-9. Raises ValueError if invalid.

#### estimate_memory_usage(n_points, has_colors) → int
Estimate memory in bytes for a points dataset.

#### check_dataset_size_warning(n_points) → Optional[str]
Returns warning message if dataset is large, None otherwise.

---

## Type Guards and Validators

**All validators now in validation/types.py** (import directly from `luxar.validation.types`, not from `protocols.py`)

**Note**: `protocols.py` no longer re-exports validators. The re-export was removed to centralize validation in the `validation` package.

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

---

## Changelog

- **v1.2.0** (2025-11-29): Fix spec to match implementation
  - Added `LINES = "lines"` to NodeType enum (was implemented but missing from spec)
  - Fixed timestamp documentation to match actual implementation (datetime.isoformat())
  - Removed non-existent `format_timestamp()` function specification

- **v1.1.0** (2025-11-30): Remove obsolete DataTypeConfig and datatypes.py documentation
  - **BREAKING**: Removed `DataTypeConfig` class documentation (never implemented)
  - **BREAKING**: Removed `datatypes.py` module documentation (never implemented)
  - These features are now handled by `EncodingMode` in `luxar.encoding`
  - Updated config.py section to reflect actual implementation
  - Added reference to encoding/SPECIFICATIONS.md for dtype selection

- **v1.0.7** (2025-11-28): Chunk size documentation enhancement
  - Added "SINGLE SOURCE OF TRUTH" emphasis for chunk size constants
  - Added note explaining consumers (io, gsplats.io) convert bytes→elements

- **v1.0.6** (2025-11-28): NodeType GSPLATS
  - Added `GSPLATS = "gsplats"` to NodeType enum

- **v1.0.5** (2025-11-28): Categorical dimension support
  - Added `CategoryList = List[str] | None` type alias
  - Added `MIN_CATEGORIES = 1` constant
  - Added `CATEGORICAL_STEP = 1.0` constant

- **v1.0.4** (2025-11-28): Timestamp format standardization
  - Added `TIMESTAMP_FORMAT` constant: ISO 8601 standard
  - Specified format: "YYYY-MM-DDTHH:MM:SSZ" (UTC)
  - Documented usage in fitting, provenance, and format metadata

- **v1.0.3** (2025-11-28): Chunk size constants alignment
  - Changed from element-based to byte-based chunk sizing
  - `CHUNK_SIZE_DEFAULT` → `TARGET_CHUNK_BYTES = 65536` (64KB)
  - Added `MIN_CHUNK_BYTES = 16384` (16KB) and `MAX_CHUNK_BYTES = 262144` (256KB)
  - See io/SPECIFICATIONS.md "Chunk Sizing Strategy" for full rationale

- **v1.0.2** (2025-11-27): Gamma and sharpness range updates
  - Updated GAMMA_MIN/MAX from [0.2, 2.0] to [0.1, 10.0] (symmetric: gamma and 1/gamma have equal range)
  - Updated SHARPNESS_MAX from 32.0 to 31.0 (final value)
  - Removed deprecated SPATIAL_INDEX_* constants (Morton ordering replaces grid-based indexing)

- **v1.0.1** (2025-11-27): Sharpness range fix
  - Fixed SHARPNESS_MAX from 15.0 to 32.0

- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented type aliases and protocols
  - Specified constants and configuration
  - Defined memory estimation formulas
