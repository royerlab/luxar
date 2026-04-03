# luxar.validation.tests

Tests for the Luxar validation package.

## Test Files

- `test_types_validation.py` - Type validation functions (positions, colors, radii, sharpness, transforms, category indices)
- `test_validation_nd.py` - nD dimensional coverage validation
- `test_nd_transforms.py` - nD transform validation and composition
- `test_points_validation.py` - Points-specific validation (write-time checks)
- `test_colormap_validation.py` - Colormap validation
- `test_validation_module.py` - Module-level integration tests (exports, cross-module consistency)

## Running

```bash
hatch run pytest packages/luxar/src/luxar/validation/tests/
```
