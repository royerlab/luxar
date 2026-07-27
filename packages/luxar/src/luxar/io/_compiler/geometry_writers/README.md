# luxar.io._compiler.geometry_writers

**Internal write pipelines** for the three geometry types (Points, Lines, GSplats). Each module exports a single `write_*` function that sequences the validation → ordering → array encoding → metadata stamping pipeline for its geometry. These are the bodies of the `LuxarZarrCompiler.write_{points,lines,gsplats}` orchestrator methods; the orchestrator builds a `GeometryWriteCtx` or `GSplatsWriteCtx`, calls the appropriate pipeline here, then records the returned metadata in its cache.

## Purpose

Extract the per-geometry pipeline bodies from the orchestrator so each geometry concern lives in one focused module. The split is behavior-preserving: all zarr writes are unchanged; only the control flow is reorganized.

## Module Ownership

| Module | Geometry | Entry Point | Context Type |
|--------|----------|-------------|--------------|
| **points.py** | Points | `write_points(ctx: GeometryWriteCtx, path, positions, colors=None, radii=None, sharpness=None, scalars=None, labels=None, image_labels=None, **attrs)` | `GeometryWriteCtx` |
| **lines.py** | Lines | `write_lines(ctx: GeometryWriteCtx, path, vertices, widths, colors=None, sharpness=None, scalars=None, indices=None, line_type="polyline", labels=None, image_labels=None, **attrs)` | `GeometryWriteCtx` |
| **gsplats.py** | GSplats | `write_gsplats(ctx: GSplatsWriteCtx, path, centers, amplitudes, cholesky_factors, colors=None, labels=None, image_labels=None, **attrs)` | `GSplatsWriteCtx` |
| **gsplats.py** | GSplats subtree | `write_gsplat_leaf_subtree(ctx: GSplatsWriteCtx, path, leaf, **attrs)` | `GSplatsWriteCtx` |

The pipelines are stateless: they read only the narrow config in the `Ctx` dataclass (encoder, compressor, ordering settings, zarr store) and return a metadata dict for the caller to record.

## Call Flow

### Points Pipeline (`write_points`)

1. **Fail-fast pre-write gate** (runs BEFORE zarr group creation):
   - `validate_render_attrs(attrs, POINTS_RESERVED_ATTRS)` — reserved writer-stamp collisions
   - `validate_node_path(path)` — every segment must be a valid node name
   - `validate_positions_for_writing(positions)` → `(n_points, n_dims)`
   - `validate_colors_for_writing(colors, n_points, channels=(3,4))` — if colors is an array (Points accept RGBA: alpha is per-point opacity)
   - `validate_broadcast_color(colors, "colors")` — if colors is a tuple/list
   - `validate_radii_for_writing(radii, n_points)` — arrays AND broadcast scalars
   - `validate_sharpness_for_writing(sharpness, n_points)` — arrays AND broadcast scalars
   - `validate_scalars_preflight(scalars, n_points)` — length check
   - `validate_labels_for_writing(labels, n_points)` — sequence-of-str type + length
   - `prepare_transform_attrs(attrs, ctx.store)` — transform / nd_transform normalization

2. **Setup**: `ctx.store.require_group(path)`

3. **Spatial ordering** (if `ctx.ordering_ctx.enable_spatial_index`):
   - `build_points_ordering(positions, n_points, n_dims, radii_for_ordering, ctx.ordering_ctx, ctx.store)` → `ordering_data` or `None`
   - Apply `ordering_data["sort_order"]` to `positions` and all non-broadcasted arrays (skip arrays with `shape[0] == 1`)

4. **Write arrays**:
   - `write_positions(positions, group, "positions", ctx.dataset_ctx, ordering_data)`
   - `write_colors(colors, group, "colors", n_points, ctx.dataset_ctx, ordering_data)` — if `colors is not None`
   - `write_radii(radii, group, "radii", n_points, ctx.dataset_ctx, ordering_data)` — if `radii is not None`
   - `write_bounded_scalar(sharpness, group, "sharpness", n_points, bounds=(0, SHARPNESS_MAX), ctx.dataset_ctx, ordering_data)` — if `sharpness is not None`
   - `write_scalars(scalars, group, "scalars", n_points, ctx.dataset_ctx, ordering_data)` — if `scalars is not None`

5. **Write labels** (CSR serialization):
   - `write_labels_csr(labels, group, ordering_data)` — if `labels is not None`
   - `write_image_labels_csr(image_labels, group, ordering_data)` — if `image_labels is not None`

6. **Spatial ordering metadata**:
   - `write_points_ordering_to_zarr(group, ordering_data)` — if `ordering_data is not None`

7. **Compute bounds**:
   - `compute_position_bounds(positions)` → `position_bounds`

8. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs, POINTS_RESERVED_ATTRS)` — fill `opacity=1.0`, `point_size_mode="absolute"`, `point_size_method="radius"`, `pixel_size=1.0`, `gamma=1.0`, `intensity=1.0`, `offset=0.0`, `tone_mapping=None`, `colormap=None` (only if absent); `blending_mode` is deliberately never stamped (no identity value)
   - Stamp `type="points"`, `position_bounds`, `transform` / `nd_transform`, `n_points`, `n_dims`, `colormap`, `tone_mapping`, and all rendering attrs

9. **Return metadata**: `{"type": "points", "n_points": n_points, "position_bounds": position_bounds}`

### Lines Pipeline (`write_lines`)

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, LINES_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_positions_for_writing(vertices)` → `(n_vertices, n_dims)`
   - Validate `line_type` in `("segments", "polyline", "loop", "indexed")`
   - Type-specific vertex count checks (segments: even, polyline: ≥2, loop: ≥3, indexed: requires `indices`)
   - `validate_widths_for_writing(widths, n_vertices)` — arrays AND broadcast scalars
   - `validate_colors_for_writing(colors, n_vertices, channels=(3,4))` — if colors is an array
   - `validate_broadcast_color(colors, "colors")` — if colors is a tuple/list
   - `validate_sharpness_for_writing(sharpness, n_vertices)` — arrays AND broadcast scalars
   - `validate_scalars_preflight(scalars, n_vertices)` — length check
   - `validate_labels_for_writing(labels, n_segments)` — if `labels is not None` (labels are per-segment, not per-vertex)
   - `prepare_transform_attrs(attrs, ctx.store)`

2. **Setup**: `ctx.store.require_group(path)`

3. **Convert to indexed representation**:
   - `convert_to_indexed(vertices, line_type, indices)` → `(vertices_indexed, segments_indexed, n_segments, indices_out)`

4. **Spatial ordering** (dual-indexed: order both vertices AND segments):
   - `build_lines_ordering(vertices, segments, widths, n_segments, n_dims, ctx.ordering_ctx, ctx.store)` → `ordering_data` or `None`
   - Apply `ordering_data["vertex_order"]` to `vertices_indexed` and non-broadcasted vertex-count arrays
   - Apply `ordering_data["segment_order"]` to `segments_indexed` and non-broadcasted segment-count arrays (colors, sharpness, scalars, labels)

5. **Write arrays**:
   - `vertices`: `SemanticType.COORDINATE`, chunks via `calculate_intelligent_chunks`, `deduplicate=False` (raw reader)
   - `segments`: `SemanticType.INDEX`, chunks `(ordering_chunk_size,)` if ordering present else intelligent, `deduplicate=False` (raw reader)
   - `widths`: via `write_positive_scalar` (rejects negative; same default-precision policy as Points radii)
   - `colors`, `sharpness`, `scalars`: same as Points, but per-segment instead of per-vertex
   - `indices`: if `indices_out is not None` (from "indexed" line_type), `SemanticType.INDEX`, `deduplicate=False`

6. **Write labels** (CSR serialization):
   - `write_labels_csr(labels, group, ordering_data)` — if `labels is not None`
   - `write_image_labels_csr(image_labels, group, ordering_data)` — if `image_labels is not None`

7. **Spatial ordering metadata**:
   - `write_lines_ordering_to_zarr(group, ordering_data)` — if `ordering_data is not None`

8. **Compute bounds**:
   - `compute_position_bounds(vertices_indexed)` → `position_bounds`

9. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs, LINES_RESERVED_ATTRS)` — same rendering defaults as Points
   - Stamp `type="lines"`, `line_type`, `position_bounds`, `transform` / `nd_transform`, `n_segments`, `n_dims`, `colormap`, `tone_mapping`, and all rendering attrs

10. **Return metadata**: `{"type": "lines", "n_segments": n_segments, "position_bounds": position_bounds}`

### GSplats Pipeline (`write_gsplats`)

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, GSPLATS_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)` → `(centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, cholesky_is_uniform)`
   - `validate_labels_for_writing(labels, n_splats)` — if `labels is not None`

2. **Setup**: `ctx.store.require_group(path)`

3. **Spatial ordering**:
   - `barrier_dims = scene_barrier_dims(ctx.store, n_dims)` — from scene `Dimensions` metadata (discrete non-display dims), or `None` if no scene dims
   - `apply_gsplat_spatial_ordering(centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, barrier_dims, ctx.ordering_ctx, ctx.store)` → `ordering_data` or `None`

4. **Write arrays**:
   - `write_gsplat_arrays(centers, amplitudes, cholesky_factors, colors, cholesky_is_uniform, n_splats, n_dims, group, ctx.dataset_ctx, ordering_data)` → `array_metadata`

5. **Write labels** (CSR serialization):
   - `write_labels_csr(labels, group, ordering_data)` — if `labels is not None`
   - `write_image_labels_csr(image_labels, group, ordering_data)` — if `image_labels is not None`

6. **Apply rendering defaults** + stamp attrs:
   - `apply_gsplat_group_attrs(group, attrs, n_splats, n_dims, array_metadata, ctx.store, ctx.lut_tone_mapping_warned)` → `lut_tone_mapping_warned_out`
   - This resolves the colormap LUT, prepares/validates `transform` + `nd_transform`, fills rendering defaults (`opacity`, `absorption`, `gamma`, `intensity`, `offset`, `truncation_radius` — `blending_mode` is deliberately never stamped), then stamps authoritative `type="gsplats"` attrs and `position_bounds`

7. **Return metadata**: `{"type": "gsplats", "n_splats": n_splats, "position_bounds": array_metadata["position_bounds"], "lut_tone_mapping_warned": lut_tone_mapping_warned_out}`

### GSplat Subtree Pipeline (`write_gsplat_leaf_subtree`)

Embeds a pre-fitted `.gsplats.zarr` file (output of `luxar.gsplats.fit_gaussian_splats` or `luxar gsplat lod`) as a detached gsplat-node subtree. The heavy lifting is in `gsplat_tree.py`; this is a thin orchestrator-facing wrapper.

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, GSPLATS_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `prepare_transform_attrs(attrs, ctx.store)` — runs BEFORE the subtree copy so a bad transform must not leave a partial tree

2. **Setup**: `ctx.store.require_group(path)`

3. **Copy subtree**:
   - `write_gsplat_node(source_group=gsplats_zarr_root, dest_group=group, ctx=ctx.ordering_ctx, colormap_lut_tone_mapping_warned=ctx.lut_tone_mapping_warned, store=ctx.store)` → `(tree_metadata, lut_tone_mapping_warned_out)`
   - This recursively copies the tree structure, applies rendering attrs at the root, and resolves the colormap LUT

4. **Stamp attrs**:
   - Write `transform` / `nd_transform`, `colormap`, `tone_mapping`, and all rendering attrs to the root group

5. **Return metadata**: `{"type": "gsplats", "n_splats": tree_metadata["n_splats"], "position_bounds": tree_metadata["position_bounds"], "lut_tone_mapping_warned": lut_tone_mapping_warned_out}`

## Context Types

**`GeometryWriteCtx`** (Points / Lines):
- `store: zarr.Group` — open zarr store
- `dataset_ctx: DatasetCtx` — encoder, encoding_mode, compressor
- `ordering_ctx: OrderingCtx` — enable_spatial_index, ordering_method

**`GSplatsWriteCtx`** (GSplats):
- `store: zarr.Group`
- `dataset_ctx: DatasetCtx`
- `ordering_ctx: OrderingCtx`
- `lut_tone_mapping_warned: bool` — threaded by value (at-most-once colormap LUT + ACES tone-mapping warning)

Both are frozen dataclasses built by the orchestrator and passed in by value.

## Key Invariants

1. **Fail-fast pre-write gate**: All validators that do NOT need the zarr store run BEFORE `require_group(path)`, so an invalid input cannot leave a partial node on disk. Validators that need the store (image_labels, custom colormap LUT resolution inside `apply_gsplat_group_attrs`, transform/nd_transform normalization) still run post-write and can leak a partial node on failure (F7 residual — transactional/temp-dir writes are a separate project).

2. **Broadcast detection**: Scalars and tuples/lists are passed through to the encoder without expansion. Arrays with `shape[0] == 1` are treated as broadcasted and are NOT reordered by spatial ordering (the encoding layer handles the broadcast).

3. **Spatial ordering**: Points/Lines reorder ALL per-element arrays (positions, colors, radii, etc.) via fancy indexing. GSplats reorder all arrays inside `apply_gsplat_spatial_ordering`. Broadcasted arrays (scalars, tuples, or `shape[0] == 1`) are skipped.

4. **Reserved attrs**: Each geometry has a set of reserved attr names that the writer stamps authoritatively (`POINTS_RESERVED_ATTRS`, `LINES_RESERVED_ATTRS`, `GSPLATS_RESERVED_ATTRS`). User-provided attrs with these names are rejected at the gate.

5. **Lines dual indexing**: Lines are stored as a dual-indexed representation — `vertices` (D-space positions) and `segments` (2×D-space vertex-pair indices). The `line_type` parameter controls how the input `vertices` are interpreted (`"segments"`, `"polyline"`, `"loop"`, `"indexed"`); `convert_to_indexed` normalizes all types to the canonical indexed form. Both `vertices` and `segments` are written with `deduplicate=False` so the viewer's raw chunked-zarr reader never sees an `array_ref`.

6. **Rendering defaults**: All three geometries stamp the same rendering defaults (`opacity`, `point_size_mode`, `pixel_size`, `gamma`, `intensity`, `offset`, `tone_mapping`, `colormap`) via `apply_default_render_attrs`. `blending_mode` is deliberately never stamped (it has no identity value). GSplats add `absorption` and `truncation_radius` defaults.

## Testing

The geometry writers are NOT unit-tested in isolation (they have no standalone API). The shared test suites in `io/tests/` exercise them through `LuxarZarrCompiler`:

- **test_compiler.py** — End-to-end scene creation with all three geometry types
- **test_ordering_points.py** / `test_ordering_lines.py` / `test_ordering_gsplats.py` — Spatial ordering integration
- **test_write_points.py** / `test_write_lines.py` / `test_write_gsplats.py` — Attribute encoding, broadcasting, edge cases
- **gsplats/tests/test_io_roundtrip.py** — GSplat subtree embedding

## See Also

- `../../compiler.py` — `LuxarZarrCompiler` orchestrator that calls these pipelines
- `../context.py` — `GeometryWriteCtx` / `GSplatsWriteCtx` dataclass definitions
- `../dataset_writers/` — Per-attribute zarr array serializers
- `../spatial_ordering/` — Points/Lines space-filling-curve ordering glue
- `../gsplat_assembly.py` — GSplat-specific validation + ordering + array writing
- `../node_common.py` — Shared node validators and rendering attr defaults
- `../../ordering.py` — Space-filling-curve primitives
