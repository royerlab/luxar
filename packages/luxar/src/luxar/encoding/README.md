# luxar.encoding

The `encoding` package provides semantic type definitions, encoding modes, and array encoding/decoding utilities for efficient storage and transmission of visualization data.

## Overview

This package handles the transformation of arrays between representations optimized for different purposes:
- **Precision**: Full floating-point accuracy
- **Storage**: Reduced byte size through quantization
- **Compression**: Better compressibility through ordering and normalization

The encoding system serves as a shared foundation for encoding arrays across all Luxar data types (points, Gaussian splats, future primitives), eliminating code duplication and ensuring consistency.

## Quick Start

Encode arrays for storage in 3 simple steps:

```python
from luxar.encoding import ArrayEncoder, SemanticType, EncodingMode
import zarr
import numpy as np

# 1. Create encoder
encoder = ArrayEncoder()
store = zarr.DirectoryStore("output.zarr")
root = zarr.group(store=store)

# 2. Encode positions (automatic optimization!)
positions = np.random.randn(1000, 3).astype(np.float32)
encoder.encode(
    data=positions,
    zarr_group=root,
    name="positions",
    semantic_type=SemanticType.COORDINATE,
    mode=EncodingMode.AUTO  # Automatically choose best encoding
)

# 3. Encode with scalar convenience (no intermediate arrays needed)
encoder.encode(
    data=(1.0, 0.0, 0.0),  # Uniform red color - stored once!
    n_elements=1000,        # Applied to all 1000 points
    zarr_group=root,
    name="colors",
    semantic_type=SemanticType.COLOR,
    color_mode="sdr"
)

print(f"Encoded {positions.nbytes + 12} bytes → {root.store.getsize('positions')} bytes compressed")
```

**What Just Happened**:
- Positions: Automatically quantized to uint16 (50% size reduction)
- Colors: Broadcasting strategy (stored once, ~99.9% size reduction)
- Both: Compressed with blosc/zstd for additional 2-3× reduction

## Purpose

The encoding system provides:

1. **Semantic Type System**: Define what data represents (positions, colors, radii) to inform encoding choices
2. **Storage Optimization**: Automatically choose optimal dtypes based on data range and precision needs
3. **Format Conversion**: Convert between numeric representations (float32 ↔ uint8)
4. **Multiple Encoding Strategies**: Broadcasting, array references, LUT encoding, dtype encoding
5. **HDR Support**: Preserve high dynamic range data where needed

### Encoding Pipeline Visual

```
Input Array (numpy float32)
    ↓
┌─────────────────────────────────────────────┐
│ Semantic Type Detection                     │ ← What does data represent?
│ • COORDINATE → spatial position data        │
│ • COLOR → RGB/HDR color values              │
│ • POSITIVE_SCALAR → radii, amplitudes       │
│ • BOUNDED_SCALAR → sharpness, opacity       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Encoding Mode Selection                     │ ← Quality vs compression
│ • PRECISION: No quantization (float32)      │
│ • AUTO: Balanced analysis + quantization    │
│ • MEMORY: Aggressive 8/16-bit quantization  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Strategy Selection (Priority Order)         │
│ 1. BROADCAST (scalar/small arrays)          │ ← Store once, apply to all
│ 2. ARRAY_REF (duplicates)                   │ ← Deduplicate shared data
│ 3. LUT (many-to-one mapping)                │ ← Palette for repeated values
│ 4. DTYPE (quantize + compress)              │ ← Standard quantization
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Encoding Execution                          │
│ • Quantize: float32 → uint16/uint8          │
│ • Store metadata: {scale, offset, min, max} │
│ • Compress: blosc/zstd level 3              │
└─────────────────────────────────────────────┘
    ↓
Zarr Array (compressed, deduplicated, optimized)
    │
    ├─ .zarray (dtype, chunks, compressor)
    ├─ .zattrs (encoding metadata)
    └─ chunks/ (compressed binary data)

Decoding reverses: decompress → dequantize → reconstruct float32
```

## Key Components

### 1. ArrayEncoder (`encoder.py`)

**Main entry point for encoding arrays.** Writes directly to zarr groups and follows a strict priority order.

**See Also:**
- `../io/README.md` - How encoding integrates with I/O layer
- `../core/README.md` - Points, Lines, GSplats data structures that use encoding

**Purpose:**
Unified encoding system with automatic deduplication and intelligent encoding selection.

**Usage Example:**
```python
from luxar.encoding import ArrayEncoder, SemanticType, EncodingMode
import zarr
import numpy as np

# Create encoder
encoder = ArrayEncoder()

# Create zarr group
store = zarr.DirectoryStore("output.zarr")
root = zarr.group(store=store)

# Encode positions (COORDINATE semantic type)
positions = np.random.randn(1000, 3).astype(np.float32)
encoder.encode(
    data=positions,
    zarr_group=root,
    name="positions",
    semantic_type=SemanticType.COORDINATE,
    mode=EncodingMode.AUTO
)

# Encode colors (COLOR semantic type, explicit mode required)
colors = np.random.rand(1000, 3).astype(np.float32)  # SDR colors in [0, 1]
encoder.encode(
    data=colors,
    zarr_group=root,
    name="colors",
    semantic_type=SemanticType.COLOR,
    mode=EncodingMode.AUTO,
    color_mode="sdr"  # Required for float colors
)

# Encode radii (POSITIVE_SCALAR semantic type)
radii = np.random.rand(1000).astype(np.float32) * 0.5
encoder.encode(
    data=radii,
    zarr_group=root,
    name="radii",
    semantic_type=SemanticType.POSITIVE_SCALAR,
    mode=EncodingMode.MEMORY  # Aggressive compression
)

# Encode sharpness (BOUNDED_SCALAR semantic type; normalized [0, 1] knob)
sharpness = np.random.rand(1000).astype(np.float32)
encoder.encode(
    data=sharpness,
    zarr_group=root,
    name="sharpness",
    semantic_type=SemanticType.BOUNDED_SCALAR,
    mode=EncodingMode.AUTO,
    bounds=(0.0, 1.0)  # Explicit bounds
)
```

**Scalar Input Support (v0.6.0):**

For uniform attributes, you can pass scalars directly instead of creating full arrays:

```python
# Scalar float - no intermediate array created!
encoder.encode(
    data=0.5,  # Scalar instead of np.full(10000, 0.5)
    n_elements=10000,  # Required: how many elements this represents
    zarr_group=root,
    name="radii",
    semantic_type=SemanticType.POSITIVE_SCALAR,
)

# Color tuple - automatically converted to (1, 3) array
encoder.encode(
    data=(1.0, 0.0, 0.0),  # RGB tuple instead of np.full((5000, 3), [1,0,0])
    n_elements=5000,
    zarr_group=root,
    name="colors",
    semantic_type=SemanticType.COLOR,
    color_mode="sdr",
)

# Result: Stored as (1,) or (1,3) array with metadata {"name": "broadcasted", "n_elements": N}
# Performance: Zero intermediate array allocation!
```

**Methods:**
- `encode(data, zarr_group, name, semantic_type, mode=AUTO, n_elements=None, bounds=None, positive_scalar_encoding="linear", custom_encoder=None, color_mode=None, chunks=None, compressor=None, deduplicate=True)` - Encode and write array or scalar
- `reset()` - Clear internal registry (call between scenes)

**Key keyword arguments:**
- `n_elements` - Broadcast target count. Required for scalar/tuple/list input; optional for arrays (opts into broadcast/uniform validation when given).
- `bounds` - `(min, max)` for BOUNDED_SCALAR (auto-detected if omitted).
- `positive_scalar_encoding` - `"linear"` (default) or `"log"` for POSITIVE_SCALAR.
- `custom_encoder` - Explicit encoder name, required when `mode=CUSTOM`.
- `color_mode` - `"sdr"` or `"hdr"`, required for float COLOR arrays.
- `chunks` / `compressor` - Optional zarr dataset chunk shape and compressor.
- `deduplicate` - When `True` (default), a byte-identical array already written elsewhere is stored as a lightweight `array_ref`. Pass `False` for arrays whose reader cannot resolve refs (e.g. line vertices/segments, read as raw chunked zarr) so they are always materialised.

The constructor accepts `float16_allowed` (see [Compatibility Control](#compatibility-control)).

**Encoding Priority Order:**
1. **Broadcasting** - If scalar input OR all values are identical (stores only 1 value)
2. **Array Reference** - If exact duplicate exists in registry (stores pointer)
3. **LUT Encoding** - If ≤256 unique values (stores indices + lookup table)
4. **Dtype Encoding** - Standard encoding based on semantic type and mode

### 2. ArrayDecoder (`decoder.py`)

**Decodes encoded arrays back to numpy.**

**Purpose:**
Read encoded arrays and apply appropriate inverse transformations.

**Usage Example:**
```python
from luxar.encoding import ArrayDecoder
import zarr

# Open zarr archive
root = zarr.open("output.zarr", mode="r")

# Create decoder
decoder = ArrayDecoder()

# Decode arrays (automatically handles all encoding types)
positions = decoder.decode(root["positions"], zarr_root=root)
colors = decoder.decode(root["colors"], zarr_root=root)
radii = decoder.decode(root["radii"], zarr_root=root)

print(f"Positions: {positions.shape}, {positions.dtype}")
print(f"Colors: {colors.shape}, {colors.dtype}")
print(f"Radii: {radii.shape}, {radii.dtype}")
```

**Methods:**
- `decode(zarr_array, zarr_root)` - Decode array based on metadata

**Supports:**
- Broadcasting (expansion)
- Array references (recursive decoding)
- LUT encoding (lookup table expansion)
- Quantized encodings (inverse transformation)
- Passthrough (direct read)

### 3. SemanticType Enum (`semantic_types.py`)

**Defines what array data represents**, which constrains valid encodings.

**Types:**

| Type | Description | Constraints | Valid Dtypes |
|------|-------------|-------------|--------------|
| **COORDINATE** | Spatial positions/centers | Can be negative | float32, float16 |
| **COLOR** | RGB/RGBA colors | Non-negative, SDR [0,1] or HDR | uint8, uint16, float16, float32 |
| **BOUNDED_SCALAR** | Scalars with known [min, max] | Within bounds | uint8, uint16, float16, float32 |
| **POSITIVE_SCALAR** | Non-negative scalars | ≥ 0 | uint8, uint16, float16, float32 |
| **CHOLESKY** | Packed Cholesky factors | Shape (N, d(d+1)/2) | float32, float16 |
| **INDEX** | Non-negative integer indices | Non-negative integers | uint8, uint16, uint32, uint64 |
| **UNIT_VECTOR** | Normalized vectors (‖v‖=1) | Unit length | float32, float16 |

**Usage Example:**
```python
from luxar.encoding import SemanticType

# Semantic types are explicitly specified by caller
encoder.encode(
    data=positions,
    zarr_group=root,
    name="positions",
    semantic_type=SemanticType.COORDINATE  # Required, no inference
)
```

**IMPORTANT:** Semantic type must be explicitly specified. The encoder does NOT attempt to infer semantic type from array values.

### 4. EncodingMode Enum (`modes.py`)

**Controls precision vs storage trade-off.**

**Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **AUTO** | Analyze data and select optimal encoding | Default, balanced approach |
| **PRECISION** | Preserve maximum precision (float32) | Scientific accuracy |
| **MEMORY** | Minimize storage aggressively | Large datasets, streaming |
| **CUSTOM** | User specifies encoder explicitly | Full control |

**Mode Behavior by Semantic Type:**

| Semantic Type | AUTO | PRECISION | MEMORY |
|---------------|------|-----------|--------|
| COORDINATE | linear_perchannel_u16 | float32 | linear_perchannel_u16 |
| COLOR (SDR) | uint8 | float32 | uint8 |
| COLOR (HDR) | float32 | float32 | float16 |
| BOUNDED_SCALAR | uint8 | float32 | uint8 |
| POSITIVE_SCALAR | Analyze range | float32 | uint8 |
| CHOLESKY | float32 | float32 | float16 |
| CHOLESKY_DIAG | log_perchannel_u8 (certified) | float32 | log_perchannel_u8 |
| CHOLESKY_OFFDIAG | signed_log_perchannel_u8 (certified) | float32 | signed_log_perchannel_u8 |
| INDEX | Smallest uint | Smallest uint | Smallest uint |

`CHOLESKY_DIAG` / `CHOLESKY_OFFDIAG` / `COORDINATE` select the **generic, reusable**
per-channel quantizers — `log_perchannel_*` (non-negative), `signed_log_perchannel_*`
(signed), and `linear_perchannel_*` (identity / fixed-point) — the semantic type is
the policy; the encoding is geometry-agnostic.

**CHOLESKY_DIAG / CHOLESKY_OFFDIAG at AUTO = uint8 with an encode-time
certificate.** The pair is encoded through `ArrayEncoder.encode_cholesky_split`,
which round-trips both halves through the exact quantization transform, rebuilds
Σ = L·Lᵀ, and measures the p95 per-splat relative Frobenius error. u8 is kept when
the error is ≤ `COV_CERT_RELF_P95_MAX` (0.05); otherwise AUTO escalates to u16 (and,
as a practically-unreachable last rung, float32) — so the AUTO error bound is a hard
invariant, and "a reason to go richer" is measured, not guessed. The measured
certificate is written into each array's own `encoding` attrs as provenance
(`{"metric": "cov_relf_p95", "value", "threshold", "tier"}`) — decode never needs it.
Both halves always share one tier. MEMORY is u8 unconditionally (no certificate).
Calibration (2026-07 covariance spike, real light-sheet fit): u8 measured relF p95
~0.02 while rendering 94.5 dB vs the float32 render (~46 dB below the fit-error
floor) at 2.48 B/splat compressed vs 8.25 for u16 (~3.3x).

**COORDINATE** (positions / centers / vertices) uses **uint16 per-axis fixed-point**
(`linear_perchannel_u16`) in both AUTO and MEMORY: each axis is quantized over its own
`[min, max]` to 65536 uniform levels, decoded back to float32 **regardless of the
input dtype** (`original_dtype` is pinned to float32 — the decode contract, matching
PRECISION's float32 cast) — visually lossless
(sub-unit) and ~2× smaller than float32. Coordinates never use uint8 (256 levels is far
too coarse) and never **float16** (its *relative* precision degrades with magnitude — a
footgun for absolute positions). An **array-local extent rail** warns when a per-axis
extent exceeds 2¹² and falls back to float32 at/above 2¹⁶ (where uint16 can't resolve a
unit step).

**Usage Example:**
```python
from luxar.encoding import EncodingMode

# AUTO mode (default) - balanced
encoder.encode(..., mode=EncodingMode.AUTO)

# PRECISION mode - maximum accuracy
encoder.encode(..., mode=EncodingMode.PRECISION)

# MEMORY mode - aggressive compression
encoder.encode(..., mode=EncodingMode.MEMORY)

# CUSTOM mode - explicit control
encoder.encode(
    ...,
    mode=EncodingMode.CUSTOM,
    custom_encoder="bounded_scalar_uint8",
    bounds=(0.0, 1.0)
)
```

### 5. ArrayRefRegistry (`registry.py`)

**Internal registry for deduplication via array references.**

**Purpose:**
Track arrays and detect duplicates using efficient two-stage hashing.

**Usage:**
```python
from luxar.encoding import ArrayRefRegistry, ArrayRefMatch

# Registry is used internally by ArrayEncoder
# Manual usage example:
registry = ArrayRefRegistry()

# Check if array is duplicate
match = registry.check(data, "path/to/array")
if match.is_duplicate:
    print(f"Duplicate found at: {match.target_path}")
    print(f"Hash: {match.hash}")
else:
    print("New array registered")

# Clear registry between scenes
registry.clear()
```

**Detection Algorithm:**
1. **Quick Check** (for arrays > 32KB): Hash first 32KB + dtype + shape
2. **Full Hash** (if quick check matches): xxhash64 of entire array

**Result:**
- `ArrayRefMatch(is_duplicate, target_path, hash)`

## Encoding Strategies

### 1. Broadcasting (Uniform Values)

When all elements share the same value, store only one value with metadata.

**Storage Format:**
- Array shape: `(1,)` or `(1, d)` instead of `(N,)` or `(N, d)`
- Metadata: `{"encoding": {"name": "broadcasted", "n_elements": N}}`

**Example:**
```python
# All points have same color (red)
colors = np.ones((10000, 3), dtype=np.float32) * [1.0, 0.0, 0.0]

# Encoded as: (1, 3) array + metadata
# Savings: 10000x compression!
```

**Decoding:**
```python
# Decoder expands (1, d) to (N, d) by repeating
decoded = np.repeat(broadcasted_value, n_elements, axis=0)
```

### 2. Array References (Deduplication)

When the same array appears multiple times in a scene, store it once and reference it elsewhere.

**Storage Format:**
- Empty array: shape `(0,)` or `(0, d)`
- Metadata:
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

**Example:**
```python
# Multiple point clouds with identical colors
for i in range(10):
    encoder.encode(
        data=same_colors,
        zarr_group=root[f"points_{i}"],
        name="colors",
        semantic_type=SemanticType.COLOR,
        mode=EncodingMode.AUTO
    )
# First instance stored normally
# Subsequent 9 instances stored as references
```

**Opting out:** Pass `deduplicate=False` to force a full materialised write even
when a duplicate exists. This is required for arrays whose reader reads raw
chunked zarr and cannot follow `array_ref` indirection (e.g. line vertices and
segments in the viewer).

**Decoding:**
```python
# Decoder follows reference path and recursively decodes target
target_array = zarr_root[target_path]
decoded = decoder.decode(target_array, zarr_root)
```

### 3. LUT Encoding (Limited Unique Values)

When an array has ≤256 unique values, store indices into a lookup table.

**Storage Format:**
- Indices: uint8 array, shape depends on mode
- Metadata:
```json
{
  "encoding": {
    "name": "lut_uint8",
    "lut": [0.0, 0.5, 1.0, 2.5, 3.7],
    "original_dtype": "float32",
    "lut_mode": "scalar",
    "original_shape": [10000]
  }
}
```

**LUT Modes:**
- **Row mode** (for colors): Each row (color tuple) is a value
  - Indices: `(N,)` uint8
  - LUT: nested list `[[r,g,b], ...]`
- **Scalar mode** (for everything else): Each element is a value
  - Indices: same shape as original
  - LUT: flat list `[v1, v2, ...]`

**Example:**
```python
# Array with few unique values
radii = np.random.choice([0.1, 0.2, 0.3, 0.4], size=10000)

# Encoded as:
# - Indices: 10000 uint8 values (1 byte each)
# - LUT: [0.1, 0.2, 0.3, 0.4] (4 float32 = 16 bytes)
# Savings: 10000×4 = 40KB → 10KB + 16B = 75% reduction
```

**When Used:**
- ≤256 unique values
- Array length ≥ 4× unique count
- Mode != PRECISION

**Decoding:**
```python
# Lookup indices in table
decoded = lut[indices]
```

### 4. Dtype Encoding (Quantization)

Standard encoding based on semantic type and mode. Includes quantization for storage optimization.

**Types:**

#### Bounded Scalar Encoding
```python
# Quantize to uint8: [min, max] → [0, 255]
encoder.encode(
    data=sharpness,
    semantic_type=SemanticType.BOUNDED_SCALAR,
    mode=EncodingMode.MEMORY,
    bounds=(0.0, 1.0)
)
```

**Formula:**
```
normalized = (value - min) / (max - min)
encoded = round(normalized * 255)

# Decode:
normalized = encoded / 255
value = normalized * (max - min) + min
```

#### Log Scalar Encoding (POSITIVE_SCALAR)
```python
# For wide dynamic range (multiple orders of magnitude)
encoder.encode(
    data=radii,
    semantic_type=SemanticType.POSITIVE_SCALAR,
    mode=EncodingMode.MEMORY,
    positive_scalar_encoding="log"
)
```

**Formula:**
```
log_val = log1p(value)  # log(1 + value)
max_log = log1p(max_value)
normalized = log_val / max_log
encoded = round(normalized * 255)

# Decode:
value = expm1(encoded / 255 * max_log)  # exp(x) - 1
```

#### Color Encoding (SDR)
```python
# Quantize float [0, 1] to uint8 [0, 255]
encoder.encode(
    data=colors,
    semantic_type=SemanticType.COLOR,
    mode=EncodingMode.MEMORY,
    color_mode="sdr"
)
```

**Formula:**
```
encoded = clip(value * 255, 0, 255).astype(uint8)

# Decode:
value = encoded / 255.0
```

## HDR Color Support

The encoding system has special handling for HDR (High Dynamic Range) colors.

### SDR vs HDR Determination

**Float color arrays require explicit `color_mode` parameter:**

```python
# SDR colors (values in [0, 1])
encoder.encode(
    data=sdr_colors,
    semantic_type=SemanticType.COLOR,
    color_mode="sdr",  # Required for float colors
    mode=EncodingMode.AUTO
)

# HDR colors (values > 1.0 allowed)
encoder.encode(
    data=hdr_colors,
    semantic_type=SemanticType.COLOR,
    color_mode="hdr",  # Required for float colors
    mode=EncodingMode.AUTO
)
```

**Integer colors (uint8, uint16) are always treated as SDR** (already quantized).

### Mode Behavior

| Color Type | AUTO | PRECISION | MEMORY |
|------------|------|-----------|--------|
| SDR float | uint8 | float32 | uint8 |
| HDR float | float32 | float32 | float16 |
| Integer | Keep as-is | Keep as-is | Keep as-is |

### Why Explicit color_mode?

**IMPORTANT:** Auto-detection (values > 1 = HDR) was rejected because buggy SDR data would silently be treated as HDR instead of raising an error. Explicit `color_mode` prevents silent bugs.

## Usage with Compiler

The encoding system is used by the Luxar compiler:

```python
from luxar import LuxarZarrCompiler, Dimensions
from luxar.encoding import EncodingMode

# Create compiler with encoding mode
dims = Dimensions.default_3d()
with LuxarZarrCompiler(
    "output.luxar.zarr",
    encoding_mode=EncodingMode.MEMORY  # Use aggressive compression
) as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("cloud", positions, colors, radii=radii)
```

**Internal Flow:**
1. Compiler creates `ArrayEncoder` instance
2. For each attribute (positions, colors, radii):
   - Compiler calls `encoder.encode()` with appropriate semantic type
   - Encoder follows priority order (broadcast → ref → LUT → dtype)
   - Data written to zarr with encoding metadata
3. Decoder reads zarr and reconstructs original arrays

## Performance Considerations

### Memory Savings

**Example: 1M points with colors**

```
float32 colors: 1M × 3 × 4 bytes = 12 MB
uint8 colors:   1M × 3 × 1 byte  = 3 MB   (75% savings!)

float32 radii:  1M × 4 bytes = 4 MB
float16 radii:  1M × 2 bytes = 2 MB   (50% savings)
uint8 radii:    1M × 1 byte  = 1 MB   (75% savings)
```

### LUT Decode Performance

Benchmarks (JavaScript):

| Elements | LUT Decode | Direct Copy | Slowdown |
|----------|------------|-------------|----------|
| 100K | 0.08 ms | 0.03 ms | 2.8x |
| 1M | 0.7 ms | 0.2 ms | 3.5x |

**Acceptable overhead** given the storage benefits. 256-entry LUT fits in L1 cache.

### Quantization Error

**Linear quantization (uint8):**
```
Max error: 1/512 ≈ 0.2% of range
```

**Log quantization (uint8):**
```
Relative error: ~0.4% per value
```

**float16 vs float32:**
```
float16: ~3 significant decimal digits, ~0.1% relative error
float32: ~7 significant decimal digits
```

## Best Practices

### 1. Use AUTO Mode by Default
```python
# Let the system analyze and optimize
encoder.encode(..., mode=EncodingMode.AUTO)
```

### 2. Use PRECISION Mode for Critical Data
```python
# When accuracy is paramount
encoder.encode(..., mode=EncodingMode.PRECISION)
```

### 3. Use MEMORY Mode for Large Datasets
```python
# When dataset size is a concern
encoder.encode(..., mode=EncodingMode.MEMORY)
```

### 4. Always Specify Semantic Type
```python
# REQUIRED - no inference
encoder.encode(
    data=positions,
    semantic_type=SemanticType.COORDINATE  # Explicit
)
```

### 5. Explicit color_mode for Float Colors
```python
# REQUIRED for float color arrays
encoder.encode(
    data=colors,
    semantic_type=SemanticType.COLOR,
    color_mode="sdr"  # or "hdr"
)
```

### 6. Provide Bounds for BOUNDED_SCALAR
```python
# Explicit bounds preferred over auto-detection
encoder.encode(
    data=sharpness,
    semantic_type=SemanticType.BOUNDED_SCALAR,
    bounds=(0.0, 1.0)  # Known logical range (normalized sharpness knob)
)
```

### 7. Clear Registry Between Scenes
```python
encoder = ArrayEncoder()

# Encode scene 1
# ... encode arrays ...

# Clear before scene 2
encoder.reset()

# Encode scene 2
# ... encode arrays ...
```

## Error Handling

The encoder raises errors for invalid input:

| Condition | Behavior |
|-----------|----------|
| Missing semantic type | **ValueError** |
| Semantic type constraint violation | **ValueError** (e.g., negative COLOR) |
| NaN or Inf values | **ValueError** |
| Float COLOR without color_mode | **ValueError** |
| CUSTOM mode without custom_encoder | **ValueError** |
| Data outside specified bounds | **ValueError** |

**Philosophy:** Silent clamping or modification can hide bugs. The caller should validate before encoding.

### Empty Arrays

Empty arrays (shape `(0,)` or `(0, d)`) are valid input:
- Pass through without encoding
- No metadata written
- Preserved dtype

## Dependencies

**Internal:**
- `luxar.typing_utils` - Type definitions
- `luxar.validation` - Input validation

**External:**
- `numpy` - Array operations
- `zarr` - Storage backend
- `xxhash` - Fast hashing for deduplication

## Testing

Tests are located in `encoding/tests/`:
- `test_decoder.py` - ArrayDecoder tests (broadcasting, LUT, array references, quantized encodings)
- `test_dynamic_range.py` - Dynamic range-based dtype selection (range computation, dtype mapping)
- `test_edge_cases.py` - Edge case tests (error paths, boundary conditions, unusual inputs)
- `test_encoder.py` - ArrayEncoder tests (semantic types, encoding modes, special encodings)
- `test_registry.py` - ArrayRefRegistry tests (duplicate detection, hashing, lifecycle)
- `test_roundtrip_encoding.py` - Full encode-decode roundtrip tests (numerical tolerance per semantic type)
- `test_scalar_input.py` - Scalar input support in ArrayEncoder (v0.6.0 feature)

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/encoding/tests/
```

## Implementation Notes

### WebGL Compatibility

The encoding system considers WebGL capabilities:
- uint8/uint16 can be auto-normalized to [0,1] in shaders
- float32 is native WebGL type
- float16 support varies (emulated if needed)

### Metadata Format

All encoding metadata stored in zarr `.zattrs` under the `"encoding"` key:

```json
{
  "encoding": {
    "name": "<encoder_name>",
    "<param1>": "<value1>",
    ...
  }
}
```

The metadata format is **language-agnostic** and uses standard JSON types (integers, floats, strings, arrays). Any language that can read zarr and JSON can decode the arrays.

### Precision Loss Warning

When AUTO mode selects a lower-precision dtype, it ensures:
1. Value range fits in target dtype
2. Precision loss is acceptable for semantic type
3. Conversion is reversible (with quantization error)

## See Also

- [core/README.md](../core/README.md) - Core data structures
- [io/README.md](../io/README.md) - I/O operations
- [validation/README.md](../validation/README.md) - Validation utilities
- [typing_utils/README.md](../typing_utils/README.md) - Type system
- [Main README](../../../../../README.md) - Project overview

## Compatibility Control

### float16_allowed Parameter

The `float16_allowed` parameter controls whether float16 encoding is used:

**Default: `False`** - Uses float32 for maximum compatibility (TypeScript/zarrita/web)
**Set to `True`** - Uses float16 for 50% memory savings (Python-only workflows)

**Affected Semantic Types**: COORDINATE, COLOR (HDR), POSITIVE_SCALAR, CHOLESKY, UNIT_VECTOR

**Usage**:
```python
from luxar import LuxarZarrCompiler
from luxar.encoding import EncodingMode

# For web/TypeScript compatibility (default)
compiler = LuxarZarrCompiler("data.luxar.zarr", encoding_mode=EncodingMode.MEMORY)

# For maximum memory efficiency (Python-only)
compiler = LuxarZarrCompiler(
    "data.luxar.zarr",
    encoding_mode=EncodingMode.MEMORY,
    float16_allowed=True
)
```

**Why False by Default?**
JavaScript/TypeScript zarr libraries (zarrita) don't support float16 dtype. Setting `float16_allowed=False` ensures datasets can be loaded by web viewers while maintaining good precision with float32.
