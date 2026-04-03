# luxar - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2026-03-31

## Purpose

The root `luxar` package serves as the public API surface for the Luxar Python library. It re-exports all user-facing classes, functions, enumerations, and type aliases from internal submodules so that users can write `import luxar` or `from luxar import ...` for all common operations. It also registers backward-compatibility module aliases in `sys.modules`.

---

## Public API

All symbols listed below are exported via `__all__` in `luxar/__init__.py` unless noted otherwise.

### Core Scene Graph Classes

These classes define the hierarchical scene structure.

| Class | Source Module | Description |
|-------|--------------|-------------|
| `Scene` | `luxar.core.scene` | Root container for an entire visualization scene |
| `Group` | `luxar.core.group` | Named group node for organizing child nodes |
| `Points` | `luxar.core.points` | Point cloud node with positions, colors, radii, sharpness |
| `Lines` | `luxar.core.lines` | Line/polyline node with vertices, widths, colors |
| `GSplats` | `luxar.core.gsplats` | Gaussian splatting node with centers, amplitudes, Cholesky factors |
| `Node` | `luxar.core.node` | Abstract base class for all scene nodes |

### Dimension System

| Class | Source Module | Description |
|-------|--------------|-------------|
| `Dimensions` | `luxar.core.dimensions` | Collection of `Dimension` objects defining the nD coordinate system |
| `Dimension` | `luxar.core.dimensions` | Single dimension specification (name, unit, range, step, display, discrete, spatial, categorical) |

### Viewer Configuration

| Class | Source Module | Description |
|-------|--------------|-------------|
| `CameraConfig` | `luxar.core.viewer_config` | Initial camera position, target, up vector, and FOV hints for the viewer |
| `ViewerConfig` | `luxar.core.viewer_config` | Scene-level viewer configuration (camera, tone mapping, controls, background, etc.) |
| `UIConfig` | `luxar.core.viewer_config` | UI panel visibility configuration |
| `DimensionsConfig` | `luxar.core.viewer_config` | nD dimension navigation state |
| `AnimationConfig` | `luxar.core.viewer_config` | Per-dimension animation state |

### Gaussian Splatting (Optional)

| Symbol | Source Module | Description |
|--------|--------------|-------------|
| `GSplatData` | `luxar.gsplats` | Data class for Gaussian splat fitting results |
| `fit_gaussian_splats` | `luxar.gsplats` | Fit Gaussian splats to a volume |

### nD Transform Utilities

| Function | Source Module | Description |
|----------|--------------|-------------|
| `validate_nd_transform` | `luxar.validation.nd_transforms` | Validate an nD transform dictionary |
| `compose_nd_transforms` | `luxar.validation.nd_transforms` | Compose two nD transforms |
| `apply_nd_transform_to_bounds` | `luxar.validation.nd_transforms` | Apply an nD transform to dimension bounds |

### Version

| Symbol | Description |
|--------|-------------|
| `__version__` | Package version string (e.g., `"2025.08.03"`) |

### I/O Classes

| Class | Source Module | Description |
|-------|--------------|-------------|
| `LuxarZarrCompiler` | `luxar.io.compiler` | Context-manager writer for creating Luxar zarr scenes with progressive writing |
| `LuxarScene` | `luxar.io.reader` | Read-only loader for Luxar zarr scenes with automatic array decoding |

### Enumerations and Constants

| Symbol | Source Module | Description |
|--------|--------------|-------------|
| `BlendingMode` | `luxar.typing_utils.enums` | Enum: `NORMAL`, `ADDITIVE`, `MAX`, `OPAQUE`, `LUMINOUS` |
| `NodeType` | `luxar.typing_utils.enums` | Enum: `SCENE`, `GROUP`, `POINTS`, `LINES`, `GSPLATS` |
| `PhysicalUnit` | `luxar.typing_utils.enums` | Enum of supported physical units: nm, um, mm, cm, m, km, inch, foot, px, au |
| `RenderingLimits` | `luxar.typing_utils.enums` | Class with rendering property ranges (opacity, gamma, sharpness, color) |
| `Defaults` | `luxar.typing_utils.enums` | Class with default values (opacity, gamma, sharpness, blending mode, chunk size, radius, colors) |

### Type Aliases

| Alias | Source Module | Underlying Type |
|-------|--------------|-----------------|
| `PositionArray` | `luxar.typing_utils.aliases` | `NDArray[np.float32]` |
| `ColorArray` | `luxar.typing_utils.aliases` | `Union[NDArray[np.float32], NDArray[np.uint8]]` |
| `TransformMatrix` | `luxar.typing_utils.aliases` | `NDArray[np.float32]` (4x4 matrix) |
| `PathLike` | `luxar.typing_utils.aliases` | `Union[str, Path]` |

### Transform Functions

All transform functions operate on 4x4 homogeneous matrices (`float32`). Imported from `luxar.core.transforms`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `identity()` | `() -> TransformMatrix` | 4x4 identity matrix |
| `translate(x, y, z)` | `(float, float, float) -> TransformMatrix` | Translation matrix |
| `translation(x, y, z)` | Alias for `translate` | Translation matrix (alias) |
| `rotate(degrees, axis)` | `(float, str\|tuple\|ndarray) -> TransformMatrix` | Rotation by degrees around named axis ('x','y','z') or arbitrary axis vector |
| `rotate_x(degrees)` | `(float) -> TransformMatrix` | Rotation around X axis (degrees) |
| `rotate_y(degrees)` | `(float) -> TransformMatrix` | Rotation around Y axis (degrees) |
| `rotate_z(degrees)` | `(float) -> TransformMatrix` | Rotation around Z axis (degrees) |
| `rotation(degrees, axis)` | Alias for `rotate` | Rotation matrix (alias) |
| `scale(x, y, z, uniform)` | `(float, float, float, Optional[float]) -> TransformMatrix` | Non-uniform scaling matrix; `uniform` overrides x/y/z with single value |
| `scaling(x, y, z)` | Alias for `scale` | Scaling matrix (alias) |
| `compose(*transforms)` | `(*TransformMatrix) -> TransformMatrix` | Right-multiply transforms (T1 applied first, last applied last) |
| `inverse(matrix)` | `(TransformMatrix) -> TransformMatrix` | Matrix inverse |
| `look_at(eye, target, up)` | `(tuple, tuple, tuple) -> TransformMatrix` | View matrix from eye position looking at target |
| `to_list(matrix)` | `(TransformMatrix) -> List[float]` | Transpose and flatten 4x4 matrix to 16-element list (column-major for THREE.js) |
| `from_list(flat)` | `(List[float]) -> TransformMatrix` | Reshape 16-element list to 4x4 matrix |
| `prepare_transform_for_zarr(matrix)` | `(TransformMatrix) -> List[float]` | Transpose and flatten for THREE.js column-major storage |
| `read_transform_from_zarr(data)` | `(Any) -> TransformMatrix` | Read and un-transpose transform from zarr |

The `transforms` module object itself is also exported for qualified access (e.g., `luxar.transforms.identity()`).

---

## Submodule Structure

| Submodule | Purpose |
|-----------|---------|
| `luxar.core` | Scene graph classes, dimensions, transforms, viewer config |
| `luxar.io` | Zarr compiler (writer) and reader |
| `luxar.encoding` | Array encoding/decoding (quantization, LUT, broadcasting) |
| `luxar.validation` | Input validation and error formatting |
| `luxar.typing_utils` | Type aliases, enums, constants, protocols |
| `luxar.utils` | Array utilities, path helpers, download helpers, demo scene generators |
| `luxar.demos` | Self-contained demonstration scripts |
| `luxar.gsplats` | Gaussian splatting fitting, seeding, and optimization |

---

## Backward Compatibility Aliases

The `__init__.py` registers module aliases in `sys.modules` so that legacy import paths continue to work:

| Legacy Import Path | Actual Module |
|--------------------|---------------|
| `luxar.array_utils` | `luxar.utils.array` |
| `luxar.dimensions` | `luxar.core.dimensions` |
| `luxar.node` | `luxar.core.node` |
| `luxar.points` | `luxar.core.points` |
| `luxar.scene` | `luxar.core.scene` |
| `luxar.compiler` | `luxar.io.compiler` |
| `luxar.writer` | `luxar.io.writer` |
| `luxar.demos` | `luxar.utils.demos` |
| `luxar.transforms` | `luxar.core.transforms` |
| `luxar.types` | `luxar.typing_utils` |
| `luxar.config` | `luxar.typing_utils.config` |
| `luxar._io` | `luxar.io.reader` |

---

## Basic Usage Example

```python
import luxar
import numpy as np

# Create dimension specification
dims = luxar.Dimensions.default_3d()

# Generate some data
positions = np.random.rand(10000, 3).astype(np.float32) * 100
colors = np.random.rand(10000, 3).astype(np.float32)

# Write to zarr
with luxar.LuxarZarrCompiler('my_scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points('cloud', positions=positions, colors=colors, radii=0.5)

# Read back
reader = luxar.LuxarScene.load('my_scene.zarr')
data = reader.get_points('cloud')
assert np.allclose(data.positions, positions, atol=1e-3)
```

---

## Related Specifications

| Topic | Location |
|-------|----------|
| Scene graph and data models | `core/SPECIFICATIONS.md` |
| Progressive writing and spatial indexing | `io/SPECIFICATIONS.md` |
| Array encoding and semantic types | `encoding/SPECIFICATIONS.md` |
| Input validation | `validation/SPECIFICATIONS.md` |
| Type definitions and constants | `typing_utils/SPECIFICATIONS.md` |
| Utility functions | `utils/SPECIFICATIONS.md` |
| Demonstration scripts | `demos/SPECIFICATIONS.md` |
| Gaussian splatting | `gsplats/SPECIFICATIONS.md` |

---

## Changelog

- **v2.0.0** (2026-02-28): Complete rewrite documenting actual public API
  - Documented all classes exported via `__all__`: Scene, Group, Points, Lines, GSplats, Node, LuxarZarrCompiler, LuxarScene, Dimensions, Dimension
  - Documented viewer config classes (exported via `__all__`): CameraConfig, ViewerConfig, UIConfig, DimensionsConfig, AnimationConfig
  - Documented all enumerations and constant classes: BlendingMode, NodeType, PhysicalUnit, RenderingLimits, Defaults
  - Documented all transform functions: identity, translate, rotate, scale, compose, inverse, look_at, etc.
  - Documented type aliases: PositionArray, ColorArray, TransformMatrix, PathLike
  - Listed all backward compatibility module aliases
  - Added basic usage example
  - Added submodule structure overview

- **v1.0.0** (2026-01-02): Initial specification (placeholder).
