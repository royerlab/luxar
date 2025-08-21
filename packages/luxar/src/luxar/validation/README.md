# Validation Package

The `validation` package provides comprehensive validation functions for Luxar data structures with helpful error messages.

## Overview

This package ensures data integrity throughout the Luxar pipeline by validating inputs at critical points, providing clear error messages that help users quickly identify and fix issues.

## Modules

### `base.py`
Core validation functions for basic data types.

**Key Classes:**
- `ValidationError`: Custom exception with helpful error messages

**Key Functions:**
- `validate_positions_for_writing()`: Validate positions before Zarr writing
- `validate_colors_for_writing()`: Validate colors with HDR support
- `validate_radii_for_writing()`: Validate point radii arrays
- `validate_sharpness_for_writing()`: Validate sharpness values

**Features:**
- Detailed error messages with suggestions
- Shape validation
- Type coercion where safe
- Range checking with warnings

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

### Range Warnings
```python
Warning: "Sharpness values outside typical range [0.5, 10.0] detected.
Values < 0.5 create uniform disks, values > 10 create hard edges."
```

## Usage Examples

### Basic Validation
```python
from luxar.validation import validate_positions_for_writing

# Validate before writing
positions = np.random.randn(1000, 3)
validated = validate_positions_for_writing(
    positions, 
    expected_shape=(1000, 3),
    name="trajectory"
)
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

## Best Practices

1. **Validate at boundaries**: At API entry points and before expensive operations
2. **Fail fast**: Don't continue with invalid data
3. **Be specific**: Tell users exactly what's wrong
4. **Suggest fixes**: Include remediation steps in error messages
5. **Allow warnings**: Some issues warrant warnings, not errors
6. **Preserve types**: Maintain dtype precision when possible

## Performance Considerations

- Validation has overhead - balance safety vs. speed
- Cache validation results when possible
- Use numpy operations for batch validation
- Skip redundant validation in tight loops

## Dependencies

Internal:
- `typing_utils`: Type definitions and constants
- `core.dimensions`: Dimension specifications

External:
- `numpy`: Array operations and validation
- Standard library only otherwise