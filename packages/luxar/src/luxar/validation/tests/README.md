# luxar.validation.tests

Tests for the Luxar validation package.

## Test Files

- `test_base_validation.py` - Shared/base validation helpers
- `test_types_validation.py` - Type validation functions (positions, colors, radii, sharpness, transforms, category indices)
- `test_validation_nd.py` - nD dimensional coverage validation
- `test_nd_transforms.py` - nD transform validation and composition
- `test_colormap_validation.py` - Colormap validation
- `test_overlay_validation.py` - Overlay validation

Geometry-type validators (one per first-class geometry):

- `test_points_validation.py` - Points-specific validation (write-time checks)
- `test_lines_validation.py` - Lines-specific validation (vertices, widths, sharpness)
- `test_gsplats_validation.py` - GSplats-specific validation (centers, amplitudes, cholesky factors)
- `test_mesh_validation.py` - Mesh-specific validation (vertices, faces, normals)

## Running

```bash
hatch run pytest packages/luxar/src/luxar/validation/tests/
```
