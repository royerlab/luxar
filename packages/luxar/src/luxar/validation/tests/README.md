# luxar.validation.tests

Tests for the Luxar validation package.

## Test Files

- `test_types_validation.py` - Type validation functions (positions, colors, radii, sharpness, transforms, category indices)
- `test_base_validation.py` - Shared/base validation helpers
- `test_validation_nd.py` - nD dimensional coverage validation
- `test_nd_transforms.py` - nD transform validation and composition
- `test_colormap_validation.py` - Colormap validation
- `test_overlay_validation.py` - Overlay validation
- Per-geometry write-time validators (one per first-class geometry type):
  - `test_points_validation.py` - Points-specific validation
  - `test_lines_validation.py` - Lines-specific validation
  - `test_gsplats_validation.py` - GSplats-specific validation
  - `test_mesh_validation.py` - Mesh-specific validation

## Running

```bash
hatch run pytest packages/luxar/src/luxar/validation/tests/
```
