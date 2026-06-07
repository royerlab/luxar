# Validation Package

The `validation` package provides comprehensive validation functions for Luxar data structures with helpful error messages.

## Quick Start

Validate data before writing in 3 steps:

```python
from luxar.validation import validate_positions_for_writing, validate_colors_for_writing
import numpy as np

# 1. Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)

# 2. Validate positions (catches NaN, Inf, shape issues)
n_points, n_dims = validate_positions_for_writing(positions, context="particle_positions")

# 3. Validate colors
validate_colors_for_writing(colors, n_points, context="particle_colors")

print("Data validated and ready for writing")
```

**What Gets Checked**:
- Shape validation - ensures 2D array with correct dimensions
- Type checking - verifies numpy array input
- Range validation - checks for NaN, Inf values
- Helpful errors - messages tell you exactly what's wrong and how to fix it

**When to Use**:
- Before calling `scene.add_points()` or `scene.add_lines()`
- When loading external data (user uploads, file imports)
- In data processing pipelines to catch issues early

## Overview

This package ensures data integrity throughout the Luxar pipeline by validating inputs at critical points, providing clear error messages that help users quickly identify and fix issues.

**See Also:**
- `../core/README.md` - Scene, Points, Lines, GSplats classes that use validation
- `../typing_utils/README.md` - Constants (GAMMA_MIN, SHARPNESS_MAX, etc.) used in validation

## Modules

### `types.py`
Core type validation functions with runtime type checking.

**Purpose:**
Provides basic validation functions used for type guards, property validation, and runtime type checking. These are simpler validators focusing on type conversion and basic checks.

**Key Functions:**
- `validate_positions()`: Validate position arrays (N, D) with optional dimensionality check
- `validate_colors()`: Validate color arrays in HDR float32 format
- `validate_radii()`: Validate radii arrays (positive values)
- `validate_sharpness()`: Validate sharpness arrays (normalized [0, 1] knob)
- `validate_transform()`: Validate 4x4 transformation matrices
- `validate_node_type()`: Validate node type strings
- `validate_physical_unit()`: Validate physical unit strings
- `validate_opacity()`: Validate opacity values (0.0-1.0)
- `validate_gamma()`: Validate gamma values (0.1-10.0)
- `validate_intensity()`: Validate intensity values (INTENSITY_MIN-INTENSITY_MAX)
- `validate_offset()`: Validate offset values (OFFSET_MIN-OFFSET_MAX)
- `validate_layer()`: Coerce a layer flag to bool
- `validate_visible()`: Coerce a visibility flag to bool
- `validate_blending_mode()`: Validate blending mode strings (`normal`, `additive`, `max`, `opaque`, `luminous`)
- `validate_colormap()`: Validate a colormap name (resolved via `colormaps.registry`) or LUT array (N, 3)
- `validate_category_indices()`: Validate category index arrays

Note: of the above, only `validate_blending_mode` and `validate_category_indices`
are re-exported at the package level. The others (`validate_intensity`,
`validate_offset`, `validate_layer`, `validate_visible`, `validate_colormap`)
are imported directly from `luxar.validation.types`.

**Type Guards:**
- `is_position_array()`: Check if object is valid position array
- `is_color_array()`: Check if object is valid color array
- `is_transform_matrix()`: Check if object is valid transform

**Usage Example:**
```python
from luxar.validation.types import validate_positions
from luxar.validation import validate_categories

# Basic validation
positions = np.random.randn(1000, 3)
validated = validate_positions(positions, ndim=3)

# Categorical validation
categories = ['DAPI', 'GFP', 'mCherry']
validate_categories(categories)  # Checks uniqueness, non-empty, etc.
```

### `base.py`
Detailed validation for write operations with comprehensive error messages.

**Purpose:**
Provides detailed validation functions specifically for write-time validation, with helpful error messages and suggestions for users writing data.

**Key Functions:**
- `validate_positions_for_writing()`: Validate positions before Zarr writing
- `validate_colors_for_writing()`: Validate colors with HDR support
- `validate_radii_for_writing()`: Validate point radii arrays
- `validate_sharpness_for_writing()`: Validate sharpness values
- `validate_zarr_attributes()`: Validate zarr group attributes dictionary

**Classes:**
- `ValidationError`: Custom error with optional suggestion message

**Features:**
- Detailed error messages with suggestions
- Shape validation
- Finiteness checks (rejects NaN / ±Inf via `_validate_numeric_finite_values`)
- Range checking with warnings

**Note:** these validators deliberately do NOT dtype-convert. `validate_positions_for_writing`
accepts any numeric dtype and returns `(n_points, n_dims)` — callers needing
float32 storage must convert afterwards (e.g. via `ensure_float32`).

### `nd_transforms.py`
Validation and composition for nD transforms on non-displayed dimensions.

**Purpose:**
Validates and composes per-dimension transforms (affine or categorical permutation) that operate on non-displayed dimensions, separate from the 4x4 spatial transform.

**Key Functions:**
- `validate_nd_transform()`: Validate an nD transform dictionary, optionally checking dimension names and domain compatibility against a `Dimensions` object
- `compose_nd_transforms()`: Compose multiple nD transforms into one (affine: multiply scales, add offsets; categorical: compose permutations)
- `apply_nd_transform_to_bounds()`: Apply an nD transform to bounding-box ranges for each dimension

### `category_validation.py`
Categorical dimension validation.

**Purpose:**
Validates category lists for categorical dimensions. Imported at the package level (`from luxar.validation import validate_categories`).

**Key Functions:**
- `validate_categories()`: Validate category lists for categorical dimensions (uniqueness, non-empty, etc.)

### `nd.py`
Validation for n-dimensional data and dimensional coverage.

**Key Classes:**
- `DimensionalCoverageError`: Exception for dimensional inconsistencies

**Key Functions:**
- `validate_dimensional_coverage()`: Ensure consistent nD coverage
- `broadcast_to_all_slices()`: Broadcast data across dimensions

**Features:**
- nD slicing validation
- Dimensional consistency checks
- Broadcasting helpers for nD data

### `overlays.py`
Validation for overlay parameters (image, text, and HTML overlays).

**Purpose:**
Validates the inputs used by the overlay system: normalized screen positions,
anchors, fonts, blend modes, transitions, text alignment, image formats,
visible ranges, image inputs, and HTML content. Not re-exported at the package
level — import directly from `luxar.validation.overlays`.

**Key Functions:**
- `validate_position()`: Validate a normalized `(x, y)` screen position in `[0, 1]`
- `validate_anchor()`: Validate an anchor against the 3x3 grid (`VALID_ANCHORS`)
- `validate_font()`: Accept a preset (`sans`/`serif`/`mono`) or any CSS font-family string
- `validate_blend_mode()`: Validate against `VALID_BLEND_MODES` (`normal`, `multiply`, `screen`, `overlay`, `additive`, `difference`)
- `validate_transition()`: Validate `none` or `fade` (`VALID_TRANSITIONS`)
- `validate_text_align()`: Validate `left`/`center`/`right`/`justify` (`VALID_TEXT_ALIGNS`)
- `validate_image_format()`: Validate `png`/`jpeg`/`webp` (`VALID_IMAGE_FORMATS`)
- `validate_visible_range()`: Validate a dim-name → value or `(min, max)` dict against scene dimension names
- `validate_image_input()`: Convert path/bytes/numpy/PIL/imageio input to `(encoded_bytes, format)`
- `sanitize_html()`: Strip disallowed tags/attributes to a safe allowlist subset

**Note:** the overlay `blend_mode` set is distinct from the geometry
blending modes validated by `types.validate_blending_mode` — overlays use CSS
compositing modes (`multiply`, `screen`, ...), not the geometry modes
(`additive`, `max`, `opaque`, `luminous`).

## Categorical Dimension Validation

### Overview

Categorical dimensions require special validation to ensure:
1. Category labels are valid strings
2. Labels are unique and non-empty
3. Position values are integer indices into the category list
4. Indices are within valid range

### `validate_categories()`

Validates a list of category labels.

**Purpose:**
Ensures category lists meet all requirements for categorical dimensions.

**Signature:**
```python
def validate_categories(categories: CategoryList) -> CategoryList
```

**Parameters:**
- `categories`: List of category labels, or None for non-categorical dimensions

**Returns:**
- Validated category list (or None if input is None)

**Raises:**
- `TypeError`: If categories is not a list or None
- `ValueError`: If categories is invalid

**Validation Checks:**
1. Must be a list or None
2. Must have at least 1 category (MIN_CATEGORIES=1)
3. Each category must be a string
4. No empty strings allowed
5. No duplicate category names
6. Each label must be ≤ 1024 characters (MAX_CATEGORY_LABEL_LENGTH)

**Usage Example:**
```python
from luxar.validation import validate_categories

# Valid categories
categories = ['DAPI', 'GFP', 'mCherry', 'Cy5']
validate_categories(categories)  # Returns validated list

# None is valid (non-categorical)
validate_categories(None)  # Returns None

# Invalid: duplicate names
try:
    validate_categories(['DAPI', 'GFP', 'DAPI'])
except ValueError as e:
    print(e)  # "duplicate category name: 'DAPI' appears at indices 0 and 2"

# Invalid: empty string
try:
    validate_categories(['DAPI', '', 'GFP'])
except ValueError as e:
    print(e)  # "category at index 1 is empty string"

# Invalid: not a list
try:
    validate_categories('DAPI')
except TypeError as e:
    print(e)  # "categories must be a list or None, got str"
```

**Integration with Dimensions:**
```python
from luxar.core.dimensions import Dimension

# Dimension automatically validates categories
dim = Dimension(
    'channel',
    categories=['DAPI', 'GFP', 'mCherry']  # Validated internally
)

# Invalid categories will raise during dimension creation
try:
    dim = Dimension('channel', categories=['DAPI', 'DAPI'])
except ValueError as e:
    print(e)  # Duplicate category error
```

### `validate_category_indices()`

Validates that array values are valid indices into a category list.

**Purpose:**
Ensures position data for categorical dimensions contains only valid integer indices.

**Signature:**
```python
def validate_category_indices(
    values: np.ndarray,
    categories: List[str],
    context: str = "values"
) -> None
```

**Parameters:**
- `values`: 1D array of values to validate (position data for one dimension)
- `categories`: List of category labels
- `context`: Context string for error messages (e.g., "channel dimension")

**Raises:**
- `ValueError`: If values contain invalid category indices

**Validation Checks:**
1. All values must be integers (or very close to integers)
2. All values must be ≥ 0
3. All values must be ≤ len(categories)-1
4. Provides helpful error messages with position of first invalid value

**Usage Example:**
```python
from luxar.validation import validate_category_indices
import numpy as np

categories = ['DAPI', 'GFP', 'mCherry']  # 3 categories: indices 0, 1, 2

# Valid indices
values = np.array([0, 1, 2, 1, 0, 2])  # All in range [0, 2]
validate_category_indices(values, categories, context='channel')  # OK

# Valid: float values that are exactly integers
values = np.array([0.0, 1.0, 2.0])
validate_category_indices(values, categories, context='channel')  # OK

# Invalid: negative index
try:
    values = np.array([0, -1, 2])
    validate_category_indices(values, categories, context='channel')
except ValueError as e:
    print(e)
    # "negative category index -1 at position 1 in channel"

# Invalid: out of range
try:
    values = np.array([0, 1, 3])  # 3 is out of range [0,2]
    validate_category_indices(values, categories, context='channel')
except ValueError as e:
    print(e)
    # "category index 3 at position 2 is out of range [0, 2] in channel.
    #  Valid categories: ['DAPI', 'GFP', 'mCherry']"

# Invalid: non-integer
try:
    values = np.array([0.0, 1.5, 2.0])  # 1.5 is not an integer
    validate_category_indices(values, categories, context='channel')
except ValueError as e:
    print(e)
    # "non-integer category index 1.5 at position 1 in channel"
```

**Integration with Position Validation:**
```python
from luxar.core.dimensions import Dimension, Dimensions

# Define categorical dimension
dims = Dimensions([
    Dimension('x', display=True),
    Dimension('y', display=True),
    Dimension('z', display=True),
    Dimension('channel', categories=['DAPI', 'GFP', 'mCherry'], display=False)
])

# Position data: [x, y, z, channel_index]
positions = np.array([
    [10.0, 20.0, 5.0, 0.0],  # Channel 0 (DAPI)
    [11.0, 21.0, 5.5, 1.0],  # Channel 1 (GFP)
    [12.0, 22.0, 6.0, 2.0],  # Channel 2 (mCherry)
])

# Validate position values for categorical dimensions
channel_dim = dims.get_dimension('channel')
if channel_dim and channel_dim.is_categorical:
    channel_indices = positions[:, dims.get_index('channel')]
    validate_category_indices(
        channel_indices,
        channel_dim.categories,
        context='channel dimension'
    )
```

## Validation Philosophy

### Early Detection
Validate data as early as possible in the pipeline to catch errors before expensive operations.

### Helpful Messages
Every validation error includes:
1. What went wrong
2. What was expected
3. How to fix it

### Progressive Validation
```python
# Level 1: Type checking
if not isinstance(data, np.ndarray):
    raise ValidationError("Data must be numpy array")

# Level 2: Shape validation
if data.ndim != 2:
    raise ValidationError(f"Expected 2D array, got {data.ndim}D")

# Level 3: Value validation
if np.any(data < 0):
    raise ValidationError("Values must be non-negative")
```

## Error Message Examples

### Clear Problem Description
```python
ValidationError: "Positions shape (100,) is not 2D.
Expected shape (N, D) where N is number of points and D is dimensionality.
Got 1D array - perhaps you meant to reshape it?"
```

### Actionable Suggestions
```python
ValidationError: "Colors shape (100, 4) doesn't match expected (100, 3).
Luxar expects RGB colors. If you have RGBA, use colors[:, :3] to extract RGB."
```

### Sharpness range enforcement

Sharpness is a normalized `[0, 1]` knob (the viewer maps it to the super-Gaussian
falloff exponent `β = 2^(6s − 2)`; `s = 0.5 → β = 2`, a true Gaussian). Both
`validate_sharpness` (`types.py`) and `validate_sharpness_for_writing` (`base.py`)
**hard-raise** (no warning) for any value outside `[0, 1]`.

```python
# Error from either validator:
ValidationError: "Sharpness values exceed maximum allowed value (1.0). Found maximum: 2.000"
```

### Categorical Validation Errors
```python
# Duplicate categories
ValueError: "duplicate category name: 'GFP' appears at indices 1 and 3"

# Invalid index
ValueError: "category index 5 at position 42 is out of range [0, 3] in channel.
Valid categories: ['DAPI', 'GFP', 'mCherry', 'Cy5']"

# Non-integer index
ValueError: "non-integer category index 1.5 at position 10 in time_phase"
```

## Usage Examples

### Basic Validation
```python
from luxar.validation import validate_positions_for_writing

# Validate before writing (returns n_points, n_dims)
positions = np.random.randn(1000, 3)
n_points, n_dims = validate_positions_for_writing(positions, context="trajectory")
```

### Custom Validation
```python
from luxar.validation import ValidationError

def validate_time_series(data, timestamps):
    """Validate time series data."""
    if len(data) != len(timestamps):
        raise ValidationError(
            f"Data length {len(data)} doesn't match "
            f"timestamp length {len(timestamps)}. "
            "Each data point needs a corresponding timestamp."
        )

    if not np.all(np.diff(timestamps) > 0):
        raise ValidationError(
            "Timestamps must be strictly increasing. "
            "Found non-monotonic timestamps - check for duplicates."
        )
```

### nD Coverage Validation
```python
from luxar.validation import validate_dimensional_coverage

# Ensure all point groups cover same dimensions
groups = {
    "neurons": positions_3d,      # (N, 3)
    "synapses": positions_4d,      # (M, 4)
}

validate_dimensional_coverage(groups, dimensions)
# Raises if groups have inconsistent dimensionality
```

### Categorical Dimension Validation
```python
from luxar.validation import validate_categories, validate_category_indices

# Validate category definition
categories = ['Control', 'Treatment_A', 'Treatment_B']
validate_categories(categories)

# Validate experimental group assignments
positions = np.random.randn(100, 4)  # [x, y, z, group]
group_indices = positions[:, 3]
validate_category_indices(group_indices, categories, context='experimental_group')
```

## Validation Patterns

### Shape Validation
```python
def validate_shape(array, expected_shape, name):
    """Validate array shape with detailed error."""
    if array.shape != expected_shape:
        raise ValidationError(
            f"{name} has shape {array.shape}, "
            f"expected {expected_shape}. "
            f"Mismatch in dimension {np.where(array.shape != expected_shape)[0]}"
        )
```

### Type Coercion
```python
def validate_and_coerce(data, dtype, name):
    """Validate and safely coerce to target dtype."""
    try:
        converted = np.asarray(data, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Cannot convert {name} to {dtype}: {e}. "
            f"Input type is {type(data).__name__}"
        )
    return converted
```

### Range Checking
```python
def validate_range(values, min_val, max_val, name):
    """Validate values are in range."""
    out_of_range = np.logical_or(values < min_val, values > max_val)
    if np.any(out_of_range):
        n_bad = np.sum(out_of_range)
        bad_vals = values[out_of_range][:5]  # Show first 5
        raise ValidationError(
            f"{n_bad} {name} values outside [{min_val}, {max_val}]. "
            f"Examples: {bad_vals}. "
            f"Use np.clip() to constrain values."
        )
```

### Categorical Index Validation Pattern
```python
def validate_categorical_positions(positions, dimensions):
    """Validate all categorical dimensions in position array."""
    for i, dim in enumerate(dimensions.dimensions):
        if dim.is_categorical:
            values = positions[:, i]
            validate_category_indices(
                values,
                dim.categories,
                context=f"{dim.name} dimension"
            )
```

## Best Practices

1. **Validate at boundaries**: At API entry points and before expensive operations
2. **Fail fast**: Don't continue with invalid data
3. **Be specific**: Tell users exactly what's wrong
4. **Suggest fixes**: Include remediation steps in error messages
5. **Allow warnings**: Some issues warrant warnings, not errors
6. **Preserve types**: Maintain dtype precision when possible
7. **Context matters**: Provide context in error messages (which dimension, which node, etc.)
8. **Validate categories early**: Check category definitions when creating dimensions
9. **Validate indices during write**: Check that position data has valid category indices

## Performance Considerations

- Validation has overhead - balance safety vs. speed
- Cache validation results when possible
- Use numpy operations for batch validation
- Skip redundant validation in tight loops
- Category validation is O(n) for uniqueness check
- Index validation is O(n) for range check

## Constants

**From `typing_utils.constants`:**
- `MIN_CATEGORIES = 1` - Minimum number of categories
- `MAX_CATEGORY_LABEL_LENGTH = 1024` - Maximum length per label
- `OPACITY_MIN = 0.0`, `OPACITY_MAX = 1.0` - Opacity range
- `GAMMA_MIN = 0.1`, `GAMMA_MAX = 10.0` - Gamma range
- `INTENSITY_MIN`, `INTENSITY_MAX` - Intensity range (used by `validate_intensity`)
- `OFFSET_MIN`, `OFFSET_MAX` - Offset range (used by `validate_offset`)
- `SHARPNESS_MIN = 0.0`, `SHARPNESS_MAX = 1.0` - Normalized sharpness knob range (enforced by `base.validate_sharpness_for_writing`; maps to super-Gaussian exponent beta=2^(6s-2))

## Dependencies

Internal:
- `typing_utils`: Type definitions and constants
- `core.dimensions`: Dimension specifications

External:
- `numpy`: Array operations and validation
- `PIL` (Pillow) / `imageio`: optional, only for `overlays.validate_image_input` /
  `_numpy_to_bytes` (imported lazily; raises a clear error if missing)
- Standard library only otherwise

## Testing

Tests are located in `validation/tests/`:
- `test_types_validation.py` - Type validation tests (includes categorical dimension tests)
- `test_base_validation.py` - Write-time validation tests (`base.py`)
- `test_validation_nd.py` - nD dimensional coverage tests (`nd.py`)
- `test_nd_transforms.py` - nD transform validation and composition tests
- `test_points_validation.py` - Points-specific validation tests
- `test_lines_validation.py` - Lines-specific validation tests
- `test_gsplats_validation.py` - GSplats-specific validation tests
- `test_colormap_validation.py` - Colormap validation tests
- `test_overlay_validation.py` - Overlay parameter validation tests (`overlays.py`)

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/validation/tests/
```

## See Also

- [core/README.md](../core/README.md) - Core data structures (Dimension, Dimensions)
- [typing_utils/README.md](../typing_utils/README.md) - Type system and constants
- [encoding/README.md](../encoding/README.md) - Data encoding and semantic types
- [Main README](../../../../../README.md) - Project overview
