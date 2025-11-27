# Luxar Encoding Package Specifications

**Status**: Draft
**Version**: 0.3

---

## 1. Purpose

The `luxar.encoding` package provides a unified system for transforming arrays between representations optimized for different purposes:
- **Precision**: Full floating-point accuracy
- **Storage**: Reduced byte size through quantization
- **Compression**: Better compressibility through ordering and normalization

This package serves as a shared foundation for encoding arrays across all Luxar data types (points, Gaussian splats, future primitives), eliminating code duplication and ensuring consistency.

**Language-Agnostic Format**: While this specification shows Python/numpy code examples for clarity, the encoding metadata format is **language-agnostic**. The metadata is stored as JSON in zarr `.zattrs` files and uses standard data types (integers, floats, strings, arrays). Any language that can read zarr and JSON can decode the encoded arrays. The formulas and algorithms specified here can be implemented in JavaScript, Rust, C++, or any other language.

---

## 2. Scope

### In Scope

- Array encoding/decoding (dtype transformations, quantization)
- LUT (lookup table) encoding for arrays with limited unique values
- Metadata specification for encoded arrays
- Auto-selection logic for choosing encoders
- Semantic type definitions (what arrays represent)

### Out of Scope

- Byte-level compression (blosc, zstd) - handled by zarr
- File I/O operations - handled by `luxar.io` and `luxar.gsplats.io`
- Validation of input data - handled by `luxar.validation`
- **Spatial ordering** (Morton, Hilbert curves) - requires coordination across multiple linked arrays (positions, colors, radii must share the same permutation); handled by I/O layer

---

## 3. Definitions

| Term | Definition |
|------|------------|
| **Encoding** | Transformation of array data from one dtype/representation to another |
| **Quantization** | Mapping continuous values to discrete integer values |
| **Semantic Type** | The meaning of array data (e.g., "color", "coordinate"), which informs encoding choices |
| **Encoding Mode** | User preference for precision vs storage trade-off |
| **Broadcasting** | A single value representing a uniform attribute across all elements |
| **LUT Encoding** | Storing indices into a lookup table when unique values are limited |
| **Array Reference** | Pointer to another array for deduplication (same content stored once) |

---

## 4. Semantic Types

Arrays have inherent semantics that constrain valid encodings. The package recognizes the following semantic types.

**IMPORTANT**: Semantic type must be **explicitly specified** by the caller. The encoder does NOT attempt to infer semantic type from array values, as this is ambiguous and error-prone. The caller knows what the data represents (e.g., "these are colors" vs "these are coordinates").

### 4.1 Coordinate

**Definition**: Spatial position values (positions, centers, vertices).

**Characteristics**:
- Can be negative
- Precision directly affects spatial accuracy
- Range varies widely (sub-pixel to astronomical scales)

**Valid encodings**: float32 (default), float16 (with precision loss)

**Constraints**: None (any real value)

### 4.2 Color

**Definition**: RGB or RGBA color values.

**Characteristics**:
- Non-negative
- SDR (Standard Dynamic Range): values in [0, 1]
- HDR (High Dynamic Range): values can exceed 1.0

**Valid encodings**:
- SDR: uint8, uint16, float16, float32
- HDR: float16, float32

**SDR Detection**: SDR vs HDR detection depends on input dtype:
- **Integer input** (uint8, uint16): Always treated as SDR. Integer color data is already in quantized format.
- **Float input** (float16, float32, float64): SDR if `np.all(data <= 1.0)`, otherwise HDR.

Note: negative color values are an error (see Section 13), not HDR.

**Constraints**: values ≥ 0

### 4.3 Bounded Scalar

**Definition**: Scalar values with known minimum and maximum bounds.

**Examples**: sharpness [0, 32], opacity [0, 1]

**Characteristics**:
- Known finite range
- Uniform precision across range is acceptable

**Valid encodings**: uint8, uint16, float16, float32

**Encoding formula**:
```
normalized = (value - min) / (max - min)
encoded = round(normalized * (2^bits - 1))
```

**Decoding formula**:
```
normalized = encoded / (2^bits - 1)
value = normalized * (max - min) + min
```

**Bounds specification**: The `bounds` parameter in `encode()` can be:
- **Explicit**: `bounds=(min, max)` - use provided values
- **Auto-detected**: `bounds=None` - compute from `data.min()`, `data.max()`

Auto-detection is convenient but explicit bounds are preferred when the logical range is known (e.g., opacity is always [0, 1] even if data only contains [0.2, 0.8]).

**Constraints**: min ≤ value ≤ max

### 4.4 Positive Scalar

**Definition**: Non-negative scalar values with potentially wide dynamic range.

**Examples**: radii, amplitudes, distances

**Characteristics**:
- Always ≥ 0
- May span multiple orders of magnitude
- Relative precision often more important than absolute

**Valid encodings**:
- Linear: uint8, uint16, float16, float32
- Logarithmic: uint8, uint16 (for wide dynamic range)

**Log encoding formula** (using log1p for numerical stability):
```
log_val = log1p(value)                    # log(1 + value), handles value=0 gracefully
max_log = log1p(max_value)
normalized = log_val / max_log
encoded = round(normalized * (2^bits - 1))

Decoding:
value = expm1(encoded / (2^bits - 1) * max_log)  # exp(x) - 1
```

**Note**: `max_value` is automatically determined from the data (`data.max()`) during encoding. The computed `max_log` is stored in metadata for decoding.

**Encoding selection**: The `positive_scalar_encoding` parameter controls which encoding is used:
- `"linear"` (default): Linear quantization, good for narrow dynamic range
- `"log"`: Logarithmic quantization using log1p/expm1, good for wide dynamic range (multiple orders of magnitude)

Log encoding is experimental and only applies to non-negative values. Use when data spans >100:1 dynamic range.

**Constraints**: value ≥ 0

### 4.5 Cholesky Factor

**Definition**: Packed lower-triangular Cholesky factors of covariance matrices.

**Examples**: Gaussian splat covariance representation

**Characteristics**:
- Shape: (N, d*(d+1)/2) for d-dimensional covariance, or (1, d*(d+1)/2) if broadcasted
- Diagonal elements: strictly positive
- Off-diagonal elements: any real number
- Mathematical constraint: L @ L.T must be positive definite
- **Broadcasting**: Supported when all elements share the same covariance (e.g., uniform isotropic splats)

**Packing convention** (row-major lower triangle):
```
For d=2: [L00, L10, L11]
For d=3: [L00, L10, L11, L20, L21, L22]
For d=4: [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]

General: d*(d+1)/2 elements
```

**Valid encodings**: float32, float16

**Empirical analysis** (synthetic microscopy-like splats, sigma 0.5-50 pixels):
```
Typical value ranges:
  Diagonal elements:   0.5 to 50 (2 orders of magnitude)
  Off-diagonal:        -50 to +50, centered around 0
  All values fit in:   [-100, 100]

Float16 precision loss:
  Max relative error:  ~0.1%
  Mean relative error: ~0.02%
  99th percentile:     ~0.04%
```

**Recommendation**: Float16 is sufficient for most use cases. Split encoding (log diagonal, linear off-diagonal) adds complexity without meaningful benefit given the narrow value range.

### 4.6 Index

**Definition**: Non-negative integer indices or counts.

**Examples**: cell_ranges, sort_order, point counts

**Characteristics**:
- Non-negative integers
- Maximum value determines required dtype

**Valid encodings**: uint8, uint16, uint32, uint64

**Selection rule**: Use smallest dtype that accommodates max value.

### 4.7 Unit Vector

**Definition**: Normalized vectors with unit length (‖v‖ = 1).

**Examples**: surface normals, directions

**Characteristics**:
- 3 components but only 2 degrees of freedom
- Can use specialized compact representations

**Valid encodings**:
- Standard: float32 (12 bytes), float16 (6 bytes)
- Octahedral: 2×uint16 (4 bytes) - 66% savings

**Note**: Not currently used in Luxar points/splats. Lower priority.

---

## 5. Broadcasting (Uniform Values)

When all elements share the same value, storing N copies is wasteful. Broadcasting allows storing a single value with explicit metadata.

### 5.1 Definition

A **broadcasted array** stores a single value that applies to all N elements:
- Storage: 1 value instead of N values
- Metadata explicitly indicates broadcasting

### 5.2 Storage Format

Broadcasted arrays are stored with shape `(1,)` or `(1, d)` instead of `(N,)` or `(N, d)`.

**Metadata**:
```json
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

The `n_elements` field indicates how many elements the single value represents.

### 5.3 Examples

**Uniform sharpness (all splats have sharpness=2.0)**:
```
Array shape: (1,) with value [2.0]
Metadata: {"encoding": {"name": "broadcasted", "n_elements": 10000}}
```

**Uniform color (all points are red)**:
```
Array shape: (1, 3) with value [[1.0, 0.0, 0.0]]
Metadata: {"encoding": {"name": "broadcasted", "n_elements": 50000}}
```

### 5.4 Dtype Preservation

Broadcasting preserves the original dtype of the data. If the input is float32, the broadcasted value is stored as float32. If the input is uint8, the broadcasted value is stored as uint8. No dtype transformation occurs during broadcasting.

### 5.5 Decoder Behavior

When loading a broadcasted array:
- If consumer needs full array: expand to (N, ...) by repeating
- If consumer supports broadcasting: use single value directly

---

## 6. LUT Encoding (Limited Unique Values)

When an array has a limited number of unique values, storing indices into a lookup table (LUT) can significantly reduce storage.

### 6.1 Definition

**LUT encoding** replaces values with indices into a small lookup table:
- Original: N values of type T (e.g., float32)
- Encoded: N indices (uint8) + K unique values (lookup table)

### 6.2 When to Use

Use LUT encoding when:
- `unique_count ≤ 256` (fits in uint8 indices)
- `array_length >> unique_count` (significant storage savings)

**Rationale for 256 threshold**: Beyond 256 unique values, uint16 indices (2 bytes) offer no advantage over float16 (2 bytes), which preserves more precision.

### 6.3 Storage Format

**Indices array**: dtype `uint8`, shape depends on LUT mode:
- **Row mode**: Shape `(N,)` - one index per row
- **Scalar mode**: Shape matches original array (e.g., `(N,)` for 1D, `(N, d)` for 2D)

**Lookup table in metadata**:
```json
{
  "encoding": {
    "name": "lut_uint8",
    "lut": [0.0, 0.5, 1.0, 2.5, 3.7],
    "original_dtype": "float32"
  }
}
```

**Note**: For 1D arrays, `lut_mode` and `original_shape` are optional and default to `"scalar"` and the indices array shape respectively.

**JSON Precision Note**: LUT values are stored as JSON numbers in zarr `.zattrs`. JSON supports approximately 15-17 significant decimal digits, which is sufficient for float32 values (7 significant digits). For float64 data with more than 15 significant digits, minor precision loss may occur during JSON round-trip. This is rarely a concern in practice since LUT encoding is typically used for discrete value sets.

### 6.4 Storage Savings

```
Original:  N × sizeof(T) bytes
Encoded:   N × 1 byte + K × sizeof(T) bytes

Example (1M float32 values, 100 unique):
  Original:  1,000,000 × 4 = 4.0 MB
  Encoded:   1,000,000 × 1 + 100 × 4 = 1.0 MB
  Savings:   75%
```

### 6.5 Decode Performance

Benchmarks show LUT decoding is approximately **3-4x slower** than direct array copy in JavaScript:

| Elements | LUT Decode | Direct Copy | Slowdown |
|----------|------------|-------------|----------|
| 100K | 0.08 ms | 0.03 ms | 2.8x |
| 1M | 0.7 ms | 0.2 ms | 3.5x |

This overhead is acceptable given the storage benefits. The 256-entry lookup table (1KB) fits entirely in L1 cache.

### 6.6 Decoder Behavior

When loading a LUT-encoded array:
```
decoded[i] = lut[indices[i]]  for all i
```

### 6.7 Auto-Detection

In AUTO mode, LUT encoding is applied when:
1. `unique_count ≤ 256`
2. `array_length ≥ 4 × unique_count` (ensures meaningful savings)

### 6.8 Multi-Dimensional Arrays

LUT encoding handles (N, d) arrays differently based on semantic type:

**Case 1: Colors** - shape `(N, d)` with `d ≤ 4` and `semantic_type == COLOR`:
- Treat each **row** as a single value (color tuple)
- Find unique rows (e.g., unique colors in a palette)
- LUT contains distinct color tuples

```python
# For (N, 3) color array (any dtype)
unique_rows, indices = np.unique(colors, axis=0, return_inverse=True)
# indices: (N,) uint8
# unique_rows: (K, 3) same dtype where K ≤ 256
```

**Case 2: All other arrays**:
- Treat each **element** individually
- Flatten, find unique values, reshape indices
- LUT contains scalar values

```python
# For (N, d) float32 array
unique_vals, indices = np.unique(data.ravel(), return_inverse=True)
indices = indices.reshape(data.shape)
# indices: (N, d) uint8
# unique_vals: (K,) float32 where K ≤ 256
```

**Metadata for row mode** (colors):
```json
{
  "encoding": {
    "name": "lut_uint8",
    "lut": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    "original_dtype": "uint8",
    "lut_mode": "row",
    "original_shape": [10000, 3]
  }
}
```

**Metadata for scalar mode** (2D non-color):
```json
{
  "encoding": {
    "name": "lut_uint8",
    "lut": [0.0, 0.5, 1.0, 2.5],
    "original_dtype": "float32",
    "lut_mode": "scalar",
    "original_shape": [10000, 3]
  }
}
```

| `lut_mode` | LUT Format | Meaning |
|------------|------------|---------|
| `"row"` | Nested lists `[[r,g,b], ...]` | Each row treated as a value (for colors) |
| `"scalar"` | Flat list `[v1, v2, ...]` | Each element treated individually |

**Note**: Scalar mode for 2D arrays is only beneficial when original dtype > 1 byte (e.g., float32 → uint8 indices). For uint8 data, prefer row mode or skip LUT entirely.

---

## 7. Array References (Deduplication)

When the same array content appears in multiple places within a scene, storing it once and referencing it elsewhere can provide massive storage savings.

> **Note on package placement**: Array references involve I/O coordination (registry, path resolution) which could argue for placement in `luxar.io`. However, the core concepts (hashing, metadata format, detection algorithm) are encoding concerns. The `ArrayRefRegistry` API may live in `luxar.io` while the metadata specification lives here. Final placement TBD during implementation.

### 7.1 Definition

An **array reference** (array_ref) stores a pointer to another array instead of duplicating data:
- Original array: stored normally at some path
- Reference: empty array with metadata pointing to the original

### 7.2 When to Use

Array references are valuable when:
- Multiple nodes share identical attribute arrays (e.g., same colors)
- Large arrays are reused across the scene
- Exact byte-for-byte match exists

**Not suitable for:**
- Arrays that are "similar" but not identical
- Cross-file references (same zarr only)

### 7.3 Storage Format

The reference array is stored as empty with metadata preserving original shape info:
- 1D array `(N,)` → stored as `(0,)`
- 2D array `(N, d)` → stored as `(0, d)`

```json
{
  "encoding": {
    "name": "array_ref",
    "target": "../points_1/colors",
    "hash": "xxh64:a1b2c3d4e5f6",
    "original_shape": [10000, 3],
    "original_dtype": "float32"
  }
}
```

| Field | Description |
|-------|-------------|
| `target` | Relative path to the original array |
| `hash` | Full content hash for verification |
| `original_shape` | Shape of the referenced array |
| `original_dtype` | Dtype of the referenced array |

### 7.4 Detection Algorithm

Efficient duplicate detection uses a two-stage approach:

**Stage 1: Quick Check**
```
quick_key = (dtype, shape, xxhash64(first_32KB_of_data))
```
- Fast to compute (only reads first 32KB of array data)
- High discriminating power (different arrays rarely match)
- dtype and shape compared directly (not hashed)
- **Small arrays**: If array ≤ 32KB, skip quick check and use full hash directly

**Stage 2: Full Hash (only if quick key matches)**
```
full_hash = xxhash64(entire_array_data_bytes)
```
- Only computed when quick key suggests a potential match
- Hash covers **data bytes only**, not dtype or shape metadata
- Confirms exact byte-for-byte equality of array contents

**Hash specification**:
- Algorithm: xxhash64 (fast, collision-resistant)
- Input: Array data bytes (`array.tobytes()` in numpy)
- NOT included in hash: dtype, shape, strides (compared separately)
- Format in metadata: `"xxh64:<hex_digest>"`

### 7.5 Registry API

The `luxar.encoding` package provides `ArrayRefRegistry` for use by I/O code:

```python
from luxar.encoding import ArrayRefRegistry, ArrayRefMatch

class ArrayRefRegistry:
    """Track arrays for deduplication via references."""

    def check(self, data: np.ndarray, path: str) -> ArrayRefMatch:
        """
        Check if array matches existing one.
        If new, registers it automatically.

        Returns:
            ArrayRefMatch with is_duplicate, target_path, hash
        """
        ...

    def clear(self) -> None:
        """Reset registry (e.g., between scenes)."""
        ...

@dataclass
class ArrayRefMatch:
    is_duplicate: bool
    target_path: str | None  # Path to original (if duplicate)
    hash: str | None         # Full hash (always provided)
```

**Usage in compiler:**
```python
match = self._array_refs.check(data, path)

if match.is_duplicate:
    self._write_array_ref(path, match.target_path, match.hash)
else:
    self._write_data(path, data)
```

### 7.6 Decoder Behavior

When loading an array_ref:
1. Read the `target` path from metadata
2. Resolve relative path to absolute
3. Load array from target path
4. Optionally verify hash matches

### 7.7 Scope and Lifecycle

- **Scope**: Same zarr archive only (relative paths)
- **Registry lifecycle**: Per compiler instance (cleared between scenes)
- **No circular references**: Target must exist before reference is written

---

## 8. Encoding Priority Order

**Exactly one encoding is applied per array.** Encodings are not chained or composed. The encoder follows a strict priority order, and the first applicable encoding is used:

```
1. Broadcasting    → If all values are identical
2. Array Reference → If exact duplicate exists in registry
3. LUT Encoding    → If ≤256 unique values
4. Dtype Encoding  → Standard encoding based on semantic type and mode
```

### 8.1 Priority Rationale

| Priority | Encoding | Why First? |
|----------|----------|------------|
| 1 | Broadcasting | Maximum space savings (N values → 1 value) |
| 2 | Array Ref | Eliminates entire array, just stores pointer |
| 3 | LUT | Significant savings when applicable |
| 4 | Dtype | Fallback when specialized encodings don't apply |

**Mode interaction**:
- **Broadcasting** and **Array Reference** apply in ALL modes (they're lossless optimizations)
- **LUT encoding** is skipped in PRECISION mode (would lose precision information)
- **Dtype encoding** varies by mode (see Section 9)

**Performance note**: The LUT check (step 3) requires computing `np.unique()` which can be expensive for large arrays. In PRECISION mode, this check is skipped entirely.

### 8.2 Decision Flow

```
┌─────────────────────────────────────────────────┐
│ encode(array, semantic_type, mode)              │
└─────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │ VALIDATE INPUT          │
        │ - Check for NaN/Inf     │
        │ - Check semantic type   │
        │   constraints           │
        │ - Error if invalid      │
        └─────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │ All values identical?   │
        └─────────────────────────┘
           │                │
          YES              NO
           │                │
           ▼                ▼
    ┌────────────┐  ┌────────────────────────┐
    │ Broadcast  │  │ Duplicate in registry? │
    └────────────┘  └────────────────────────┘
                       │                │
                      YES              NO
                       │                │
                       ▼                ▼
                 ┌──────────┐  ┌─────────────────────────────────┐
                 │ ArrayRef │  │ mode != PRECISION AND           │
                 └──────────┘  │ ≤256 unique vals?               │
                               └─────────────────────────────────┘
                                  │                │
                                 YES              NO
                                  │                │
                                  ▼                ▼
                            ┌─────────┐    ┌─────────────┐
                            │   LUT   │    │ Dtype Encode│
                            └─────────┘    └─────────────┘
```

**Note**: In PRECISION mode, the LUT check is skipped entirely (goes directly to Dtype Encode). This avoids the `np.unique()` computation overhead and preserves full precision.

---

## 9. Encoding Modes

Users can specify encoding preference through modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **AUTO** | Analyze data and select appropriate encoding | Default, balanced |
| **PRECISION** | Preserve maximum precision (float32 for all) | Scientific accuracy |
| **MEMORY** | Minimize storage size aggressively | Large datasets, streaming |
| **CUSTOM** | User specifies encoder for each array | Full control |

### 9.1 Mode Behavior by Semantic Type

| Semantic Type | AUTO | PRECISION | MEMORY | AUTO Lossy? |
|---------------|------|-----------|--------|-------------|
| **Coordinate** | float32 | float32 | float16 | No |
| **Color (SDR)** | uint8 | float32 | uint8 | Yes* |
| **Color (HDR)** | float32 | float32 | float16 | No |
| **Bounded Scalar** | uint8 | float32 | uint8 | Yes |
| **Positive Scalar** | analyze range | float32 | uint8 (normalized) | Depends |
| **Cholesky** | float32 | float32 | float16 | No |
| **Index** | smallest uint | smallest uint | smallest uint | No |

*SDR colors: uint8 provides 256 levels per channel, sufficient for display but lossy for float32 input.

### 9.2 AUTO Mode Details

AUTO mode analyzes data to select encoding:

| Semantic Type | Selection Logic |
|---------------|-----------------|
| Coordinate | Always float32 (precision critical) |
| Color | uint8 if all values in [0,1] (maps to [0,255]), else float32 for HDR |
| Bounded Scalar | uint8 if range well-defined |
| Positive Scalar | uint8 if max ≤ 1, float16 if max < 1000, else float32 |
| Cholesky | float32 (conservative, could use float16) |
| Index | Smallest uint type for max value |

### 9.3 MEMORY Mode Trade-offs

MEMORY mode prioritizes storage over precision:

| Trade-off | Impact |
|-----------|--------|
| float16 positions | ~0.1% relative error, may affect sub-pixel accuracy |
| uint8 radii | 256 levels, ~0.4% error |
| float16 Cholesky | ~0.1% error (acceptable per empirical analysis) |

Use MEMORY mode when:
- Dataset is very large (millions of points/splats)
- Streaming to limited bandwidth
- Precision loss is acceptable for visualization

### 9.4 CUSTOM Mode

CUSTOM mode allows explicit control over the encoder used for each array. When `mode=EncodingMode.CUSTOM`, the `custom_encoder` parameter specifies which encoder to use:

```python
encoder.encode(
    data=radii,
    zarr_group=group,
    name="radii",
    semantic_type=SemanticType.POSITIVE_SCALAR,
    mode=EncodingMode.CUSTOM,
    custom_encoder="log_scalar_uint8",  # Explicit encoder choice
)
```

**Valid encoder names**:
- `"float32"`, `"float16"` - Passthrough with dtype conversion
- `"uint8"`, `"uint16"` - For INDEX semantic type
- `"bounded_scalar_uint8"`, `"bounded_scalar_uint16"` - Requires `bounds` parameter
- `"log_scalar_uint8"`, `"log_scalar_uint16"` - For POSITIVE_SCALAR
- `"rgb_uint8"`, `"rgb_uint16"` - For COLOR

**Error handling**: If `mode=CUSTOM` and `custom_encoder` is None, an error is raised.

**Note**: Broadcasting and Array Reference checks still apply in CUSTOM mode (they are lossless optimizations). CUSTOM mode only controls the dtype/quantization encoding.

---

## 10. Metadata Specification

Encoded arrays store metadata in zarr `.zattrs` to enable decoding.

### 10.1 Standard Attribute Name

All encoding metadata stored under the key `"encoding"`:

```json
{
  "encoding": {
    "name": "<encoder_name>",
    "<param1>": "<value1>",
    "<param2>": "<value2>"
  }
}
```

The specific parameters depend on the encoder (see Section 10.3 for examples).

### 10.2 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Encoder identifier (e.g., "bounded_scalar_uint8") |

### 10.3 Encoder-Specific Fields

**BoundedScalarEncoder**:
```json
{
  "encoding": {
    "name": "bounded_scalar_uint8",
    "min": 0.0,
    "max": 15.0,
    "bits": 8,
    "original_dtype": "float32"
  }
}
```

**LogScalarEncoder**:
```json
{
  "encoding": {
    "name": "log_scalar_uint16",
    "max_log": 2.3,
    "bits": 16,
    "original_dtype": "float32"
  }
}
```

**ColorEncoder** (SDR):
```json
{
  "encoding": {
    "name": "rgb_uint8",
    "original_dtype": "float32"
  }
}
```

**ColorEncoder** (SDR, 16-bit):
```json
{
  "encoding": {
    "name": "rgb_uint16",
    "original_dtype": "float64"
  }
}
```

**BroadcastedEncoder**:
```json
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

**LutEncoder**:
```json
{
  "encoding": {
    "name": "lut_uint8",
    "lut": [0.0, 0.5, 1.0, 2.5, 3.7],
    "original_dtype": "float32"
  }
}
```

**ArrayRefEncoder**:
```json
{
  "encoding": {
    "name": "array_ref",
    "target": "../points_1/colors",
    "hash": "xxh64:a1b2c3d4e5f6",
    "original_shape": [10000, 3],
    "original_dtype": "float32"
  }
}
```

### 10.4 No Encoding (Passthrough)

Arrays without encoding transformation have no `"encoding"` attribute, or:
```json
{
  "encoding": {
    "name": "none"
  }
}
```

---

## 11. Encoder Interface

The encoding system uses a unified class-based design with separate encoder and decoder classes.

### 11.1 SemanticType Enum

```python
from enum import Enum

class SemanticType(Enum):
    """Semantic types for array data."""
    COORDINATE = "coordinate"
    COLOR = "color"
    BOUNDED_SCALAR = "bounded_scalar"
    POSITIVE_SCALAR = "positive_scalar"
    CHOLESKY = "cholesky"
    INDEX = "index"
    UNIT_VECTOR = "unit_vector"
```

### 11.2 EncodingMode Enum

```python
class EncodingMode(Enum):
    """Encoding preference modes."""
    AUTO = "auto"           # Analyze data and select
    PRECISION = "precision" # Preserve maximum precision
    MEMORY = "memory"       # Minimize storage size
    CUSTOM = "custom"       # User specifies encoding
```

### 11.3 ArrayEncoder Class

The `ArrayEncoder` is the main entry point for encoding arrays. It maintains an internal registry for deduplication and writes directly to zarr groups.

```python
class ArrayEncoder:
    """Unified encoder with internal registry for deduplication."""

    def __init__(self, broadcast_rtol: float = 0.0, broadcast_atol: float = 0.0):
        """
        Args:
            broadcast_rtol: Relative tolerance for broadcasting check (default: exact equality)
            broadcast_atol: Absolute tolerance for broadcasting check (default: exact equality)
        """
        self._registry = ArrayRefRegistry()
        self._broadcast_rtol = broadcast_rtol
        self._broadcast_atol = broadcast_atol

    def encode(
        self,
        data: np.ndarray,
        zarr_group: zarr.Group,
        name: str,
        semantic_type: SemanticType,  # REQUIRED - must be explicit
        mode: EncodingMode = EncodingMode.AUTO,
        bounds: tuple[float, float] | None = None,  # For BOUNDED_SCALAR
        positive_scalar_encoding: Literal["linear", "log"] = "linear",  # For POSITIVE_SCALAR
        custom_encoder: str | None = None,  # For CUSTOM mode
    ) -> None:
        """
        Encode array and write to zarr group.

        Follows priority order:
        1. Broadcasting (if all values identical within tolerance)
        2. Array reference (if duplicate exists)
        3. LUT encoding (if ≤256 unique values)
        4. Dtype encoding (based on semantic type and mode)
        """
        ...

    def _is_uniform(self, data: np.ndarray) -> bool:
        """Check if all values are identical (within tolerance)."""
        if self._broadcast_rtol == 0.0 and self._broadcast_atol == 0.0:
            # Exact equality (default, safe for all types)
            return np.all(data == data.flat[0])
        else:
            # Approximate equality (use with caution)
            return np.allclose(data, data.flat[0],
                               rtol=self._broadcast_rtol, atol=self._broadcast_atol)

    def reset(self) -> None:
        """Clear registry (call between independent scenes)."""
        self._registry.clear()
```

**Broadcasting tolerance**: By default, exact equality is used (`rtol=0, atol=0`). This is safe for all semantic types including indices and colors. Approximate equality can be enabled for floating-point data where small variations should be treated as uniform, but use with caution.

**Key design decisions**:
- Writes directly to zarr (no intermediate return value)
- Registry is internal (not exposed to caller)
- Semantic type is required (no inference)
- Single entry point for all encoding

### 11.4 ArrayDecoder Class

The `ArrayDecoder` handles reading encoded arrays back to numpy.

```python
class ArrayDecoder:
    """Decode any encoded array from zarr."""

    def decode(
        self,
        zarr_array: zarr.Array,
        zarr_root: zarr.Group | None = None,  # For array_ref resolution
    ) -> np.ndarray:
        """
        Decode array based on encoding metadata.

        Returns numpy array with dtype matching the original input:
        - Quantized encodings use 'original_dtype' from metadata
        - LUT encoding uses 'original_dtype' from metadata
        - Passthrough returns stored dtype
        - Broadcasted returns stored dtype (expanded to full size)
        """
        enc = zarr_array.attrs.get("encoding", {})
        name = enc.get("name", "none")

        # Special encodings
        if name == "broadcasted":
            return self._expand_broadcasted(zarr_array, enc)
        elif name == "array_ref":
            return self._follow_ref(zarr_array, enc, zarr_root)
        elif name == "lut_uint8":
            return self._decode_lut(zarr_array, enc)

        # Quantized encodings (require inverse transformation)
        elif name == "bounded_scalar_uint8":
            return self._decode_bounded_scalar(zarr_array, enc)
        elif name == "bounded_scalar_uint16":
            return self._decode_bounded_scalar(zarr_array, enc)
        elif name == "log_scalar_uint8":
            return self._decode_log_scalar(zarr_array, enc)
        elif name == "log_scalar_uint16":
            return self._decode_log_scalar(zarr_array, enc)
        elif name == "rgb_uint8":
            return self._decode_color(zarr_array, enc)

        # Passthrough (none, float16, float32, etc.)
        else:
            return zarr_array[:]

    def _decode_bounded_scalar(self, arr, enc) -> np.ndarray:
        """Decode bounded scalar: uint -> original dtype using min/max."""
        data = arr[:]
        min_val, max_val = enc["min"], enc["max"]
        bits = enc["bits"]
        original_dtype = enc.get("original_dtype", "float32")
        # Use float64 intermediate for precision, then cast to original dtype
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = normalized * (max_val - min_val) + min_val
        return result.astype(original_dtype)

    def _decode_log_scalar(self, arr, enc) -> np.ndarray:
        """Decode log scalar: uint -> original dtype using expm1."""
        data = arr[:]
        max_log = enc["max_log"]
        bits = enc["bits"]
        original_dtype = enc.get("original_dtype", "float32")
        # Use float64 intermediate for precision, then cast to original dtype
        normalized = data.astype(np.float64) / (2**bits - 1)
        result = np.expm1(normalized * max_log)
        return result.astype(original_dtype)

    def _decode_color(self, arr, enc) -> np.ndarray:
        """Decode color: uint8/uint16 [0,max] -> original dtype [0,1]."""
        data = arr[:]
        original_dtype = enc.get("original_dtype", "float32")
        # Determine max value based on stored dtype
        max_val = 255.0 if data.dtype == np.uint8 else 65535.0
        result = data.astype(np.float64) / max_val
        return result.astype(original_dtype)

    def _decode_lut(self, arr, enc) -> np.ndarray:
        """Decode LUT-encoded array."""
        indices = arr[:]
        lut = np.array(enc["lut"], dtype=enc["original_dtype"])
        lut_mode = enc.get("lut_mode", "scalar")  # Default for 1D

        if lut_mode == "row":
            # Row mode: indices are (N,), lut is (K, d)
            # Result is (N, d)
            return lut[indices]
        else:
            # Scalar mode: indices match original shape
            # lut is (K,), indices may be (N,) or (N, d)
            return lut[indices]

    def _expand_broadcasted(self, arr, enc) -> np.ndarray:
        """Expand broadcasted array to full size."""
        data = arr[:]  # Shape (1,) or (1, d)
        n_elements = enc["n_elements"]
        # Repeat the single value n_elements times
        return np.repeat(data, n_elements, axis=0)

    def _follow_ref(self, arr, enc, zarr_root) -> np.ndarray:
        """Follow array reference and decode target (recursive)."""
        if zarr_root is None:
            raise ValueError("zarr_root required for array_ref decoding")

        target_path = enc["target"]
        # Resolve relative path (e.g., "../points_1/colors")
        target_array = zarr_root[target_path]

        # IMPORTANT: Recursively decode the target array
        # This handles cases where target is itself encoded (e.g., LUT)
        return self.decode(target_array, zarr_root)
```

### 11.5 Error Handling

The encoder raises errors for:
- Missing semantic type (required parameter)
- Semantic type constraint violation (e.g., negative values for COLOR)
- Value out of quantization range (e.g., value > max for BOUNDED_SCALAR)
- NaN or Inf in input data
- array_ref target not found (path resolution failed)
- Hash mismatch when verifying array_ref (data corruption detected)

**Silent fallbacks** (not errors):
- LUT with >256 unique values → falls back to dtype encoding
- Broadcasting not applicable → continues to next priority

See Section 13 for complete error handling specification.

---

## 12. Compression Interaction

### 12.1 Separation of Concerns

Encoding and compression are separate:
- **Encoding**: Semantic transformation (this package)
- **Compression**: Byte-level reduction (zarr/blosc)

Encoders do not perform compression. Compression is applied by the storage layer after encoding.

**Important**: Zarr array settings (compressor, chunk size, filters) are controlled by the caller, not by the encoding package. The `ArrayEncoder` writes data to a zarr group but does not configure compression settings. The caller (e.g., `luxar.io` or `luxar.gsplats.io`) is responsible for setting up the zarr group with appropriate compression before calling `encode()`.

### 12.2 Compression Hints

Encoders may provide hints about effective compression settings:

| Hint | Values | Meaning |
|------|--------|---------|
| `shuffle` | "byte", "bit", "none" | Recommended shuffle filter |

Example:
- uint8 arrays: byte shuffle effective
- float32 arrays: bit shuffle effective
- Already-ordered data: shuffle may help less

These are suggestions, not requirements. The storage layer decides final compression settings.

---

## 13. Error Handling

### 13.1 Encoding Errors

The encoder does NOT silently modify data. Invalid data causes errors:

| Condition | Behavior |
|-----------|----------|
| Semantic type constraint violation | **Error** (e.g., negative COLOR, negative POSITIVE_SCALAR) |
| Value out of quantization range | **Error** (e.g., value > max for BOUNDED_SCALAR) |
| NaN or Inf in input | **Error** (must be handled by caller before encoding) |
| Wrong input dtype | Convert if lossless, else **Error** |
| Wrong array shape | **Error** |

**Rationale**: Silent clamping or modification can hide bugs and produce unexpected results. The caller should validate and handle edge cases before encoding.

### 13.2 Empty Array Handling

Empty arrays (shape `(0,)` or `(0, d)`) are valid input and handled as follows:
- **No encoding applied**: Empty arrays pass through without transformation
- **No metadata written**: The `"encoding"` attribute is omitted (or set to `"none"`)
- **Preserved dtype**: The empty array retains its original dtype

This is a valid edge case that can occur when filtering removes all elements.

### 13.3 Decoding Errors

| Condition | Behavior |
|-----------|----------|
| Missing metadata | Error (cannot decode without parameters) |
| Unknown encoder name | Error |
| Corrupted encoded data | Implementation-dependent |

---

## 14. Quantization Error Analysis

For quantized encodings, understand the precision loss:

### 14.1 Linear Quantization (Bounded Scalar)

```
bits    levels    max_error (fraction of range)
8       256       1/512 ≈ 0.2%
16      65536     1/131072 ≈ 0.0008%
```

### 14.2 Log Quantization

Relative error is approximately uniform across the range:
```
bits    relative_error
8       ~0.4% per value
16      ~0.002% per value
```

### 14.3 Float16 vs Float32

float16 has ~3 significant decimal digits:
- Suitable for values where 0.1% relative error is acceptable
- Not suitable for positions requiring sub-pixel precision over large ranges

---

## 15. Package Dependencies

### Required
- numpy
- xxhash (for array deduplication hashes)

---

## 16. Open Questions

*Note: Several questions have been resolved and documented above. Remaining open questions:*

1. **Real-world Cholesky validation**: The empirical analysis in section 4.5 used synthetic data. Validation on real fitted splats from diverse microscopy data would be valuable.

2. **Encoder versioning**: How to handle encoder version changes while maintaining backward compatibility? (Deferred - too early in development to address)

---

## 17. Future Considerations

### Potential Extensions

- **Delta encoding**: Store differences between consecutive values after spatial ordering
- **Run-length encoding**: For arrays with many repeated values
- **Bit-packing**: Sub-byte storage (e.g., 4-bit sharpness)
- **Vector quantization**: Codebook-based encoding for complex attributes

These are not in scope for v1.0 but the architecture should not preclude them.

---

## 18. References

- [Blosc compression](https://www.blosc.org/)
- [Octahedral normal encoding](https://jcgt.org/published/0003/02/01/)
- [xxhash](https://github.com/Cyan4973/xxHash) - Fast hash algorithm used for array deduplication
- [3DGS compression survey](https://arxiv.org/html/2502.19457v1)

---

## Changelog

- **v0.3**: Unified sharpness bounds
  - Changed sharpness bounds example from [0.16, 24.4] to [0, 32] to align with core SPECIFICATIONS.md
  - All Luxar specs now use unified bounds [0, 32] for sharpness

- **v0.2**: Cross-specification consistency review
  - Removed "Last Updated" placeholder
  - Added broadcasting support note to CHOLESKY semantic type
  - Added "AUTO Lossy?" column to mode behavior table
  - Updated bounded scalar example to use sharpness bounds (later unified to [0, 32] in v0.3)
- **v0.1**: Initial specification draft