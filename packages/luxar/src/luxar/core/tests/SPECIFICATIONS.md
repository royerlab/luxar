# luxar.core.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Luxar core module: scene graph construction, node types (Points, Lines, GSplats, Group), dimensions, transforms, physical units, HDR colors, viewer configuration, and the `extend_to_all` optimization.

---

## Test Files

| File | Description |
|------|-------------|
| `test_scene_structure.py` | Scene graph structure, node hierarchy, zarr serialization |
| `test_scene_methods.py` | Scene class methods not covered elsewhere |
| `test_scene_advanced.py` | Advanced Scene class tests for coverage of edge cases |
| `test_dimensions.py` | Scene-level dimension definitions and behavior |
| `test_dimension_metadata.py` | Dimension class metadata and current Dimension API |
| `test_spatial_dimensions.py` | Spatial flag implementation for dimensions |
| `test_dim_order.py` | `dim_order` dimension mapping on `add_points`/`add_lines`/`add_gsplats` |
| `test_transforms.py` | Transform utilities (translate, rotate, scale, compose) |
| `test_extend_to_all.py` | `extend_to_all` functionality in `Scene.add_points()` |
| `test_gsplats_extend_to_all.py` | `extend_to_all` functionality in `Scene.add_gsplats()` |
| `test_group.py` | Group class with `add_*` methods |
| `test_node_properties.py` | Node properties and method chaining |
| `test_node_rendering.py` | Rendering attributes for Node class |
| `test_datanode_types.py` | DataNode types: Lines and GSplats |
| `test_hdr_colors.py` | Edge case tests for HDR color support (values > 1.0) |
| `test_physical_units.py` | Physical units support through Dimensions system |
| `test_viewer_config.py` | ViewerConfig and CameraConfig dataclasses |

**Total**: 17 test files

---

## Key Test Patterns

- Scene graph tests verify both in-memory structure and zarr serialization round-trips.
- Dimension tests verify metadata propagation through the scene hierarchy.
- Transform tests verify matrix composition order and NumPy/THREE.js transpose conventions.
- `extend_to_all` tests verify that points/gsplats are extended along non-spatial dimensions.

---

## Critical Gotchas Tested

- Matrix storage: NumPy row-major vs THREE.js column-major (transpose on serialization).
- Constructor initialization order (parent before subclass).
- Transform composition order (`compose(T1, T2, T3)` applies T1 first).

---

## Related Specifications

- `luxar.core` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (17 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
