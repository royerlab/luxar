# luxar.utils.tests

Tests for the Luxar utilities package.

## Test Files

- `test_array.py` - Tests for array utilities (ensure_float32, validate_array_shape)
- `test_builder_helpers.py` - Tests for scene builder helper functions
- `test_deprecation.py` - Tests for the deprecation helpers (message wording, warning category, caller attribution, kwarg forwarding)
- `test_demo_scenes.py` - Tests for demo scene generators (Lorenz, random spheres, time series)
- `test_dtype_support.py` - Tests for data type support across the pipeline
- `test_error_handling.py` - Tests for error handling and edge cases

## Running

```bash
hatch run pytest packages/luxar/src/luxar/utils/tests/
```
