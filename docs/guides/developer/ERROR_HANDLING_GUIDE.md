# Error Handling Guide for Luxar

**Version**: 1.0.0
**Status**: ✅ Adopted Standard
**Created**: 2025-12-15
**Last Updated**: 2025-12-15

---

## Overview

This document defines the error handling patterns for the Luxar codebase. Consistent error handling improves maintainability, debugging, and user experience.

---

## Core Principles

1. **Be Explicit**: Never fail silently - always log, throw, or return error indicators
2. **Fail Early**: Validate inputs at boundaries (user input, network, file system)
3. **Provide Context**: Include relevant data in error messages (file paths, values, etc.)
4. **Chain Errors**: Use `from e` (Python) or nested errors (TypeScript) to preserve stack traces
5. **Log Appropriately**: Use correct log levels (error, warning, info)

---

## TypeScript Patterns

### Pattern 1: Async Loaders (Return null on failure)

**Use when**: Loading optional resources that may not exist

```typescript
async loadResource(path: string): Promise<Resource | null> {
  try {
    const data = await fetch(path);
    return parseResource(data);
  } catch (error) {
    log.warning(Modules.LOADER, `Failed to load ${path}:`, error);
    return null; // Caller can handle missing resource
  }
}
```

**Key points**:
- Return `null` for optional resources
- Log at `warning` level (expected failure)
- Include context in log message

### Pattern 2: Critical Operations (Throw on failure)

**Use when**: Failure indicates a bug or corrupted state

```typescript
validateMetadata(attrs: Metadata): void {
  if (!attrs.version) {
    const error = new Error('Missing required version field in metadata');
    log.error(Modules.VALIDATOR, error.message, { attrs });
    throw error;
  }
}
```

**Key points**:
- Throw errors for unexpected/critical failures
- Log at `error` level before throwing
- Include diagnostic data in log

### Pattern 3: User Input (Validate and throw with clear messages)

**Use when**: Handling user-provided data

```typescript
parseUserInput(input: string): Config {
  if (!input.trim()) {
    throw new Error('Configuration cannot be empty. Please provide valid JSON.');
  }
  try {
    return JSON.parse(input);
  } catch (error) {
    throw new Error(`Invalid JSON: ${error.message}. Check syntax and try again.`);
  }
}
```

**Key points**:
- Clear, actionable error messages
- Don't expose internal implementation details
- Suggest solutions when possible

### Pattern 4: Resource Cleanup (Try-finally)

**Use when**: Resources must be cleaned up regardless of errors

```typescript
async processWithCleanup(): Promise<void> {
  const resource = await allocate();
  try {
    await resource.process();
  } finally {
    // ALWAYS runs, even if process() throws
    resource.dispose();
  }
}
```

---

## Python Patterns

### Pattern 1: Validation Functions (Raise with context)

**Use when**: Validating data structures

```python
def validate_positions(positions: np.ndarray, n_dims: int) -> None:
    """Validate position array shape and values.

    Raises:
        ValueError: If positions are invalid, with details about the issue
    """
    if positions.shape[1] != n_dims:
        raise ValueError(
            f"Position dimensions mismatch: expected {n_dims} columns, "
            f"got {positions.shape[1]}. Shape: {positions.shape}"
        )
    if np.any(np.isnan(positions)):
        raise ValueError("Positions contain NaN values - check input data")
```

**Key points**:
- Raise `ValueError` for invalid input
- Include expected vs actual values
- Suggest potential causes

### Pattern 2: IO Operations (Chain exceptions with `from e`)

**Use when**: File/network operations

```python
def load_zarr_store(path: str) -> zarr.Group:
    """Load zarr store from path.

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path exists but isn't valid zarr
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zarr store not found: {path}")

    try:
        store = zarr.open(path, mode='r')
        return store
    except Exception as e:
        raise ValueError(
            f"Failed to open '{path}' as zarr store. "
            f"Check if path is a valid zarr directory."
        ) from e
```

**Key points**:
- Chain exceptions with `from e` to preserve stack trace
- Provide context about what was being attempted
- Distinguish between missing files and corrupted files

### Pattern 3: Warnings for Non-Critical Issues

**Use when**: Something is unusual but not fatal

```python
from warnings import warn

def validate_radii(radii: np.ndarray) -> None:
    """Validate radii array."""
    if np.any(radii <= 0):
        raise ValueError("Radii must be positive")

    if np.any(radii > 100):
        # Unusual but not fatal
        warn(
            f"Very large radii detected (max: {radii.max()}). "
            "This may cause rendering issues.",
            UserWarning
        )
```

---

## Common Mistakes to Avoid

### ❌ Silent Failures
```typescript
// BAD: Silently ignores errors
try {
  await loadData();
} catch {
  // Nothing - error is lost!
}

// GOOD: Log and handle
try {
  await loadData();
} catch (error) {
  log.error(Modules.LOADER, 'Failed to load data:', error);
  return null;
}
```

### ❌ Generic Error Messages
```python
# BAD: No context
raise ValueError("Invalid input")

# GOOD: Specific and actionable
raise ValueError(
    f"Expected 3D positions but got {positions.shape[1]}D. "
    "Ensure positions array has shape (N, 3)."
)
```

### ❌ Swallowing Exceptions
```python
# BAD: Loses original error information
try:
    parse_config(data)
except Exception:
    raise ValueError("Bad config")

# GOOD: Chains exceptions
try:
    parse_config(data)
except Exception as e:
    raise ValueError("Failed to parse config") from e
```

### ❌ Inconsistent null/undefined Handling
```typescript
// BAD: Mix of null and undefined
function getData(): Data | null | undefined { ... }

// GOOD: Pick one
function getData(): Data | null { ... }
```

---

## When to Use Each Error Type

### Python

| Error Type | Use Case | Example |
|------------|----------|---------|
| `ValueError` | Invalid input values | Wrong array shape, negative radius |
| `TypeError` | Wrong type | Expected array, got scalar |
| `FileNotFoundError` | Missing file | Zarr store not found |
| `KeyError` | Missing dict key | Required metadata missing |
| `RuntimeError` | Unexpected state | Already initialized |
| `NotImplementedError` | Unimplemented feature | "HDR not yet supported for lines" |

### TypeScript

| Error Type | Use Case | Example |
|------------|----------|---------|
| `Error` | General errors | Default for most cases |
| `TypeError` | Type violations | Expected number, got string |
| `RangeError` | Out of range | Index out of bounds |
| Return `null` | Optional failure | Resource not found |
| Return `undefined` | Uninitialized state | Value not set yet |

---

## Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `error` | Critical failures, bugs | "Corrupted metadata detected" |
| `warning` | Unusual but handled | "Using fallback value for missing attr" |
| `info` | Important events | "Loaded 1M points from zarr" |
| `debug` | Diagnostic details | "Cache hit for chunk [0,0,0]" |
| `custom` | Special emphasis | "✓ Scene validation passed" |

---

## Testing Error Handling

Always test error paths:

```python
def test_validate_positions_with_nan():
    """Test that NaN positions are rejected."""
    positions = np.array([[1, 2, 3], [np.nan, 0, 0]])
    with pytest.raises(ValueError, match="NaN values"):
        validate_positions(positions)

def test_validate_positions_with_wrong_dims():
    """Test that wrong dimensions are rejected."""
    positions = np.array([[1, 2]])  # 2D not 3D
    with pytest.raises(ValueError, match="expected 3 columns"):
        validate_positions(positions, n_dims=3)
```

---

## Updating Existing Code

### Standardizing Existing Code

1. **Identify inconsistent patterns**: Use grep for mixed error handling
   ```bash
   grep -r "catch.*{" --include="*.ts" | grep -v "log\|throw\|return"
   ```

2. **Choose appropriate pattern**: Use decision tree above

3. **Add tests**: Before changing error handling, add tests for error paths

4. **Update gradually**: Change one module at a time, test thoroughly

---

## Summary

| Situation | Action | Example |
|-----------|--------|---------|
| Optional resource | Return `null` + log warning | Loading optional config |
| Critical operation | Throw + log error | Corrupted metadata |
| User input | Throw with clear message | Invalid file path |
| Validation | Raise with details | Wrong array shape |
| IO operation | Chain exceptions (`from e`) | File loading |
| Cleanup needed | Use try-finally | Dispose resources |
| Unusual but OK | Warn, don't fail | Very large values |

**Key Takeaway**: Be consistent within each module, prefer explicit over implicit, and always provide enough context for debugging.
