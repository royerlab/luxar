# luxar.validation - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2025-11-28

## Purpose

The `validation` package provides comprehensive validation for all Luxar data structures with helpful, actionable error messages. It's organized into three modules serving different purposes.

**Related Specifications**:
- `luxar.core` - Data structures being validated (see `core/SPECIFICATIONS.md`)
- `luxar.io` - Uses validation during write operations (see `io/SPECIFICATIONS.md`)
- `luxar.typing_utils` - Constants for validation bounds (see `typing_utils/SPECIFICATIONS.md`)

---

## Module Organization

### types.py - Basic Type Validation
**Purpose**: Fast validation for type guards and property validation
**Returns**: Validated value (typed)
**Errors**: Simple ValueError/TypeError

### base.py - Write-Time Validation
**Purpose**: Detailed validation with helpful error messages for data writing
**Returns**: None or tuple (e.g., n_points, n_dims)
**Errors**: Custom ValidationError with suggestions

### nd.py - Dimensional Validation
**Purpose**: Validate dimensional coverage consistency
**Returns**: None or transformed arrays
**Errors**: Custom DimensionalCoverageError

---

## Validation Specifications

### Position Array Validation

**Basic** (types.py):
```
Input: Any
Requirements:
  - Must be numpy array
  - Must be 2D (shape: N × D)
  - Must have at least 1 dimension (D ≥ 1)
  - If ndim specified, must match exactly
Output: float32 array
```

**For Writing** (base.py):
```
Input: Any
Requirements: Same as basic, plus:
  - N > 0 (cannot write empty)
  - D > 0 (must have dimensions)
  - Warning if D > 10 (high-dimensional)
Output: (n_points, n_dims) tuple
Error Messages: Include shape info, suggestions (e.g., "reshape with .reshape(-1, 1)")
```

### Color Array Validation

**Basic**:
```
Input: Any, n_points: int
Requirements:
  - Must be numpy array
  - Must be shape (N, 3) where N = n_points
  - HDR supported (values > 1.0 allowed)
Output: float32 array
```

**For Writing**:
```
Additional checks:
  - No negative values (raise error with min value)
  - Warning if max > 10.0 (extreme HDR)
Error Messages: Distinguish between wrong count vs wrong channels
                Suggest broadcasting for single color
```

### Radii/Sharpness Validation

**Requirements**:
- Must be 1D array
- Must have N elements (match n_points)
- All values must be positive (> 0)
- Sharpness: must be in range [0, 31]

**Error Handling**:
- Detect zeros vs negatives (different suggestions)
- Suggest np.clip() or replacement strategies

### Transform Validation

**Requirements**:
- Must be numpy array
- Must be 4x4 matrix
- Converted to float32

**Note**: Does not validate that matrix is valid transformation (e.g., orthogonal), only shape/type

### Rendering Attribute Validation

**Opacity**:
- Range: [0.0, 1.0]
- Accepts: int, float, np.number
- Converts to float
- Raises TypeError if not convertible

**Gamma**:
- Range: [0.1, 10.0] (symmetric: gamma and 1/gamma have equal range)
- Accepts: int, float, np.number
- Converts to float
- Raises ValueError if out of range

**Blending Mode**:
- Valid: "normal", "additive"
- Must be string
- Case-sensitive

### Enum Validation

**NodeType**: Must be "scene", "group", "points", or "gsplats"
**PhysicalUnit**: Must be one of supported units (nm, um, mm, cm, m, metre, meter, km, inch, foot, px, au)
- Handles variations (meter/metre, micrometer/micron/μm)
- Case-insensitive matching with normalization map

### Categorical Dimension Validation

**Purpose**: Validate categorical dimension definitions and category index values in point coordinates.

**Category List Validation** (`validate_categories()`):
```
Input: categories (List[str] | None)
Requirements:
  - If None: Valid (non-categorical dimension)
  - If provided:
    - Must be a list (not tuple, set, or other iterable)
    - Must have at least 1 element (MIN_CATEGORIES = 1)
    - All elements must be non-empty strings
    - All category names must be unique (no duplicates)
    - Maximum length per label: 1024 characters (practical limit)
Output: Validated list or None
Error Messages:
  - "categories must have at least 1 element"
  - "category at index {i} is empty string"
  - "duplicate category name: '{name}' appears at indices {i} and {j}"
  - "category at index {i} exceeds maximum length (1024 chars)"
```

**Category Index Validation** (`validate_category_indices()`):
```
Input: values (array), categories (List[str])
Requirements:
  - Values must be integers (or convertible to int without loss)
  - All values must be valid indices: 0 ≤ value < len(categories)
Output: None (raises on error)
Error Messages:
  - "category index {value} at position {i} is out of range [0, {max}]"
  - "negative category index {value} at position {i}"
  - "non-integer category index {value} at position {i}"
```

**Integration with Dimension Validation**:
When validating a `Dimension` object:
1. If `categories` is not None, validate the category list
2. Automatically set `discrete = True` (if not already)
3. Validate that `range` is consistent: should be `(0, len(categories) - 1)`
4. Validate that `step` is 1.0 (categorical dimensions step by one category)

**Point Coordinate Validation** (for categorical dimensions):
When validating point positions against scene dimensions:
1. Identify which dimensions are categorical
2. For categorical dimensions, validate that all point values are valid indices
3. Provide helpful error messages: "point {i} has channel=5, but valid channels are: DAPI (0), GFP (1), mCherry (2)"

---

## Dimensional Coverage Validation

### Purpose
Ensure all point groups in a scene have consistent coverage of non-displayed dimensions.

### Problem Statement
In nD scenes with non-displayed dimensions (like Time and Channel), each point group should either:
1. Have points at ALL time/channel combinations, OR
2. Be explicitly marked for special handling (future: static flag)

### Validation Algorithm

**Input**:
- `scene_dimensions`: Scene dimension specifications
- `point_groups`: Dict mapping group names to position arrays

**Algorithm**:
1. Identify non-displayed dimensions
2. For each group, collect unique values per non-displayed dimension
3. Select reference coverage (from largest group)
4. For each other group:
   - Compare coverage to reference
   - If missing values → raise DimensionalCoverageError with missing values
   - Error includes suggestions for fixing

**Output**: None (raises error if invalid)

### Broadcasting Helper

**Function**: `broadcast_to_all_slices()`

**Purpose**: Replicate points across all combinations of non-displayed dimensions

**Algorithm**:
1. Identify non-displayed dimensions
2. Get all unique values per dimension (from range if discrete, or from data)
3. Calculate total slices = product of dimension sizes
4. Create output arrays: size = n_points * n_slices
5. For each slice combination:
   - Copy original points
   - Set non-displayed dimension values for this slice
   - Replicate colors/radii if present

**Output**: (broadcast_positions, broadcast_colors, broadcast_radii)

---

## Error Message Design

### Principles
1. **Be Specific**: Include actual values that caused the error
2. **Be Helpful**: Provide actionable suggestions for fixing
3. **Show Context**: Include variable names and context info
4. **Format Nicely**: Use structured formatting for readability

### Custom Exception Classes

**ValidationError**:
```
message: Primary error message
suggestion: Optional fix suggestion
Format: "{message}\n💡 Suggestion: {suggestion}"
```

**DimensionalCoverageError**:
```
message: Description of coverage issue
group_name: Name of problematic group
missing_coverage: Dict[dim_name, Set[missing_values]]
Format: Includes breakdown of missing values per dimension
        Plus suggestions (provide all combos, use broadcasting, mark as static)
```

---

## Validation Constants

**From constants.py**:
- `OPACITY_MIN = 0.0`, `OPACITY_MAX = 1.0`
- `GAMMA_MIN = 0.1`, `GAMMA_MAX = 10.0`
- `SHARPNESS_MIN = 0.0`, `SHARPNESS_MAX = 31.0`
- `MIN_POINT_RADIUS = 0.001`, `MAX_POINT_RADIUS = 1000.0`

---

## Validation Flow

### At Object Creation (Node/Points):
```
User provides attributes → Validate via types.py functions → Store if valid
```

### At Write Time (Compiler):
```
User provides arrays → Validate via base.py functions →
If invalid: Raise ValidationError with helpful message
If valid: Write to Zarr
```

### At Scene Level (Optional):
```
Scene built → validate_dimensional_coverage() →
If inconsistent: Raise DimensionalCoverageError with missing values
If consistent: Continue
```

---

## Type Guards

**Purpose**: Enable conditional type narrowing in type checkers

**Pattern**:
```python
def is_position_array(obj: Any) -> bool:
    try:
        validate_positions(obj)
        return True
    except (ValueError, TypeError):
        return False
```

**Available Guards**:
- `is_position_array(obj)`
- `is_color_array(obj, n_points)`
- `is_transform_matrix(obj)`

---

## This specification provides sufficient detail to re-implement the validation system with equivalent behavior and error messages.

---

## Changelog

- **v1.1.0** (2025-11-28): Categorical dimension validation
  - Added `validate_categories()` for category list validation
  - Added `validate_category_indices()` for point coordinate validation
  - Category constraints: non-empty, unique, max 1024 chars per label
  - Integration with Dimension validation (auto-set discrete, validate range/step)
  - Helpful error messages showing valid category names
  - Updated NodeType to include "gsplats"

- **v1.0.2** (2025-11-27): Gamma and sharpness range updates
  - Updated GAMMA_MIN/MAX from [0.2, 2.0] to [0.1, 10.0] (symmetric: gamma and 1/gamma have equal range)
  - Updated SHARPNESS_MAX from 32.0 to 31.0 (final value)
  - Removed sharpness "typical range" warning (was arbitrary and unhelpful)

- **v1.0.1** (2025-11-27): Sharpness range fix
  - Fixed SHARPNESS_MAX from 15.0 to 32.0

- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented three-module organization (types, base, nd)
  - Defined validation functions and error handling
  - Specified type guard patterns
