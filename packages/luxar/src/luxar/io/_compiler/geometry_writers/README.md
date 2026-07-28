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
   - `write_positions(group, positions, ordering_data, ctx.dataset_ctx)`
   - `write_colors(group, colors, ordering_data, n_points, ctx.dataset_ctx)` — if `colors is not None`
   - `write_radii(group, radii, ordering_data, n_points, ctx.dataset_ctx)` — if `radii is not None` (returns `max_radius`, which is also stamped as the `max_radius` group attr here)
   - `write_bounded_scalar(group, sharpness, "sharpnesses", (0.0, SHARPNESS_MAX), ordering_data, n_points, ctx.dataset_ctx, "sharpness")` — if `sharpness is not None` (the written dataset name is `"sharpnesses"`)
   - `write_scalars(group, scalars, ordering_data, n_points, ctx.dataset_ctx)` — if `scalars is not None`

5. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs)` — fill `opacity=1.0`, `absorption=1.0`, `gamma=1.0`, `intensity=1.0`, `offset=0.0` (only if absent); `blending_mode` is deliberately never stamped (no identity value)
   - `group.attrs.update(attrs)` then stamp `type="points"`, `n_points`, `has_colors` / `has_radii` / `has_sharpness` / `has_scalars` (no dim-count attr is stamped; user-supplied + default rendering attrs land via the `update(attrs)` call; `max_radius` was already stamped in step 4 when radii are present)

6. **Compute bounds**:
   - `compute_position_bounds(positions)` → `position_bounds`, stamped as the `position_bounds` group attr and forwarded to `ctx.update_scene_bounds(...)`

7. **Spatial ordering metadata**:
   - `write_points_ordering_to_zarr(group, ordering_data, ctx.compressor)` — if `ordering_data is not None` (sets `has_spatial_index`)

8. **Write labels** (CSR serialization; `sort_order` derived from `ordering_data`):
   - `write_labels_csr(group, labels, n_points, ctx.compressor, sort_order)` — if `labels is not None`
   - `write_image_labels_csr(group, image_labels, n_points, ctx.compressor, sort_order)` — if `image_labels is not None`

9. **Return metadata**: `{"n_points", "ndim", "path", "has_colors", "has_radii", "has_sharpness", "position_bounds"}` plus (conditionally) `max_radius`, `has_scalars`, `has_spatial_index`, `has_labels`, `has_image_labels` (no `"type"` key)

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
   - `validate_labels_for_writing(labels, n_vertices)` — if `labels is not None` (labels are per-vertex)
   - `prepare_transform_attrs(attrs, ctx.store)`

2. **Setup**: `ctx.store.require_group(path)`

3. **Convert to indexed representation**:
   - `convert_to_indexed(n_vertices, line_type, indices)` → a single `(S, 2)` uint32 `segments` array (the first arg is the vertex COUNT, not the vertices array); `n_segments = segments.shape[0]`

4. **Spatial ordering** (dual-indexed: order both vertices AND segments):
   - `build_lines_ordering(vertices, segments, widths, n_vertices, n_dims, n_segments, ctx.ordering_ctx, ctx.store)` → `ordering_data` or `None`
   - Replace `vertices` / `segments` with `ordering_data["sorted_vertices"]` / `ordering_data["sorted_segments"]`
   - Permute all non-broadcasted per-vertex arrays (widths, colors, sharpness, scalars) by `ordering_data["vertex_sort_indices"]`; labels are likewise reordered per-vertex via `vertex_sort_indices` at write time. Only `segments` itself is segment-count.

5. **Write arrays**:
   - `vertices`: `SemanticType.COORDINATE`, 2-D chunks via `calculate_intelligent_chunks`, `deduplicate=False`, `allow_lut=False` (raw reader)
   - `segments`: `SemanticType.INDEX`, 2-D chunks `(segment_chunk_size, 2)` (from the segment ordering's `chunk_size` if ordering present, else the constant `2048`), `deduplicate=False` (raw reader)
   - `widths`: via `write_positive_scalar` (rejects negative; same default-precision policy as Points radii)
   - `colors`, `sharpness`, `scalars`: per-vertex, same as Points

   (No `indices` dataset is written — only `vertices` / `segments` / `widths` and the optional per-vertex arrays.)

6. **Spatial ordering metadata**:
   - `write_lines_ordering_to_zarr(group, ordering_data, ctx.compressor)` — if `ordering_data is not None` (sets `has_spatial_index`)

7. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs)` — same rendering defaults as Points
   - `group.attrs.update(attrs)` then stamp `type="lines"`, `n_vertices`, `n_segments`, `ndim`, `original_line_type`, `has_colors` / `has_sharpness` / `has_scalars`, `max_width`, ordering attrs (`ordering` / `vertex_ordering` / `segment_ordering`, or `ordering="none"`) (user + default rendering attrs land via the `update(attrs)` call)

8. **Compute bounds**:
   - `compute_position_bounds(vertices)` → `position_bounds`, stamped as the `position_bounds` group attr and forwarded to `ctx.update_scene_bounds(...)`

9. **Write labels** (CSR serialization; `sort_order` = `ordering_data["vertex_sort_indices"]` when ordered, per-vertex):
   - `write_labels_csr(group, labels, n_vertices, ctx.compressor, sort_order)` — if `labels is not None`
   - `write_image_labels_csr(group, image_labels, n_vertices, ctx.compressor, sort_order)` — if `image_labels is not None`

10. **Return metadata**: `{"n_vertices", "n_segments", "ndim", "original_line_type", "has_colors", "has_sharpness", "max_width"}` plus ordering keys and `position_bounds` (and conditionally `has_spatial_index`, `has_scalars`, `has_labels`, `has_image_labels`) — no `"type"` key

### GSplats Pipeline (`write_gsplats`)

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, GSPLATS_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)` → `(centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, cholesky_is_uniform)`
   - `validate_labels_for_writing(labels, n_splats)` — if `labels is not None`

2. **Setup**: `ctx.store.require_group(path)`

3. **Spatial ordering**:
   - `barrier_dims = scene_barrier_dims(ctx.store, n_dims)` — from scene `Dimensions` metadata (discrete non-display dims), or `None` if no scene dims
   - `apply_gsplat_spatial_ordering(centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, cholesky_is_uniform, ctx.ordering_ctx, truncation_radius, barrier_dims=scene_barrier_dims(ctx.store, n_dims))` → 5-tuple `(centers, amplitudes, cholesky_factors, colors, ordering_data)` (the `truncation_radius` from `attrs` is passed as the `coverage_sigma` arg; `ordering_data` is `None` if ordering was not applied)

4. **Write arrays**:
   - `write_gsplat_arrays(group, centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, cholesky_is_uniform, ordering_data, ctx.dataset_ctx)` → `metadata`

5. **Apply rendering defaults** + stamp attrs (stamped BEFORE labels are written):
   - `ctx.apply_gsplat_group_attrs(group, metadata, attrs)` — a bound orchestrator method returning `None`; it delegates to `apply_gsplat_group_attrs(...)` and stores the warn-once colormap-LUT flag on the orchestrator instance (it is NOT threaded through the ctx)
   - This resolves the colormap LUT, prepares/validates `transform` + `nd_transform`, fills rendering defaults (`opacity`, `absorption`, `gamma`, `intensity`, `offset`, `truncation_radius` — `blending_mode` is deliberately never stamped), stamps authoritative `type="gsplats"` attrs, and adds `position_bounds` into `metadata`
   - Then `ctx.update_scene_bounds(metadata["position_bounds"])` folds the leaf's bounds into the scene extent

6. **Write labels** (CSR serialization; `sort_order` derived from `ordering_data`):
   - `write_labels_csr(group, labels, n_splats, ctx.compressor, sort_order)` — if `labels is not None`
   - `write_image_labels_csr(group, image_labels, n_splats, ctx.compressor, sort_order)` — if `image_labels is not None`

7. **Return metadata**: the `metadata` dict from `write_gsplat_arrays` — `{"n_splats", "ndim", "has_colors", "amplitude_range", "center_bounds"}` plus ordering keys, `position_bounds` (added by `apply_gsplat_group_attrs`), and conditionally `amplitude_data_range` (when `amplitudes` is a non-empty array), `has_labels` / `has_image_labels` (no `"type"` or `"lut_tone_mapping_warned"` key)

### GSplat Subtree Pipeline (`write_gsplat_leaf_subtree`)

Writes an in-memory `GSplatLeaf` (a single splat set or an additive ladder) into the scene. The heavy lifting is in `gsplat_tree.py`; this is a thin orchestrator-facing wrapper that seams onto `write_gsplat_leaf` — the single authoring path also used by the standalone `.gsplats.zarr` writer, so a scene additive ladder is byte-identical to a standalone one.

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, GSPLATS_RESERVED_ATTRS)` — the only validator run here (the compiler entry already validated the path segments; `validate_node_path` and `prepare_transform_attrs` are NOT called in this path)
   - `path = path.lstrip("/")`

2. **Setup**: `ctx.store.require_group(path)`

3. **Write leaf**:
   - `write_gsplat_leaf(group, leaf, dataset_ctx=ctx.dataset_ctx, ordering_ctx=ctx.ordering_ctx, store=ctx.store, attrs=attrs, scene_tone_mapping=ctx.scene_tone_mapping, barrier_dims=scene_barrier_dims(ctx.store, leaf.ndim))` → a single `metadata` dict
   - This writes the leaf's arrays (single set → one leaf; ladder → `additive_<i>/` subgroups), applies rendering attrs from `attrs`, and resolves the colormap LUT

4. **Update scene bounds**: `ctx.update_scene_bounds(metadata["position_bounds"])`

5. **Return metadata**: the `metadata` dict returned straight from `write_gsplat_leaf` (no `"lut_tone_mapping_warned"` key; includes `n_splats` and `position_bounds`, plus `n_additive_sublods` only when the leaf is an additive ladder — a single splat set returns straight from `_write_single_splat_set` without that key)

## Context Types

**`GeometryWriteCtx`** (Points / Lines):
- `store: zarr.Group` — open zarr store
- `dataset_ctx: DatasetCtx` — encoder, encoding_mode, compressor
- `ordering_ctx: OrderingCtx` — enable_spatial_index, ordering_method
- `compressor: CompressorLike` — scene default compressor (used by the ordering + label writers)
- `update_scene_bounds: Callable[[Dict[str, List[float]]], None]` — scene-bounds accumulator hook
- `write_colormap_lut: Callable[[zarr.Group, Dict[str, Any]], None]` — custom-colormap LUT writer hook

**`GSplatsWriteCtx`** (GSplats):
- `store: zarr.Group`
- `dataset_ctx: DatasetCtx`
- `ordering_ctx: OrderingCtx`
- `compressor: CompressorLike`
- `scene_tone_mapping: Optional[str]` — scene tone-mapping value threaded into `write_gsplat_leaf`
- `update_scene_bounds: Callable[[Dict[str, List[float]]], None]`
- `apply_gsplat_group_attrs: Callable[[zarr.Group, Dict[str, Any], Dict[str, Any]], None]` — group-attrs hook that owns the warn-once colormap-LUT flag

Both are frozen dataclasses built by the orchestrator and passed in by value.

## Key Invariants

1. **Fail-fast pre-write gate**: The input validators run BEFORE `require_group(path)`, so an invalid input cannot leave a partial node on disk — this deliberately includes `prepare_transform_attrs` for Points and Lines (it reads `store.attrs["scene_dimensions"]` but is still run in the gate so a bad transform can't leak a partial node). The store-dependent steps that remain post-write, and so can leak a partial node on failure, are `image_labels` writing, custom colormap-LUT resolution, and — for GSplats only — `transform` / `nd_transform` normalization (deferred inside `apply_gsplat_group_attrs`) (F7 residual — transactional/temp-dir writes are a separate project).

2. **Broadcast detection**: Scalars and tuples/lists are passed through to the encoder without expansion. Arrays with `shape[0] == 1` are treated as broadcasted and are NOT reordered by spatial ordering (the encoding layer handles the broadcast).

3. **Spatial ordering**: Points/Lines reorder ALL per-element arrays (positions, colors, radii, etc.) via fancy indexing. GSplats reorder all arrays inside `apply_gsplat_spatial_ordering`. Broadcasted arrays (scalars, tuples, or `shape[0] == 1`) are skipped.

4. **Reserved attrs**: Each geometry has a set of reserved attr names that the writer stamps authoritatively (`POINTS_RESERVED_ATTRS`, `LINES_RESERVED_ATTRS`, `GSPLATS_RESERVED_ATTRS`). User-provided attrs with these names are rejected at the gate.

5. **Lines dual indexing**: Lines are stored as a dual-indexed representation — `vertices` (D-space positions) and `segments` (2×D-space vertex-pair indices). The `line_type` parameter controls how the input `vertices` are interpreted (`"segments"`, `"polyline"`, `"loop"`, `"indexed"`); `convert_to_indexed` normalizes all types to the canonical indexed form. Both `vertices` and `segments` are written with `deduplicate=False` so the viewer's raw chunked-zarr reader never sees an `array_ref`.

6. **Rendering defaults**: Points and Lines stamp the same rendering defaults (`opacity`, `absorption`, `gamma`, `intensity`, `offset`) via `apply_default_render_attrs`. GSplats stamp the same set plus `truncation_radius` via `apply_gsplat_group_attrs` (NOT `apply_default_render_attrs`). `blending_mode` is deliberately never stamped (it has no identity value).

## Testing

The geometry writers are NOT unit-tested in isolation (they have no standalone API). The shared test suites in `io/tests/` exercise them through `LuxarZarrCompiler`:

- **test_compiler_integration.py** / **test_compiler_improvements.py** / **test_compiler_colormap.py** / **test_compiler_nd_bounds.py** — End-to-end scene creation, colormap/LUT handling, and nD bounds
- **test_ordering_points.py** / **test_ordering_lines.py** / **test_ordering_gsplats.py** — Spatial ordering integration
- **gsplats/io/tests/test_save_load.py** and **gsplats/tests/test_gsplat_data_io.py** — GSplat leaf/ladder save-load round trips

## See Also

- `../../compiler.py` — `LuxarZarrCompiler` orchestrator that calls these pipelines
- `../context.py` — `GeometryWriteCtx` / `GSplatsWriteCtx` dataclass definitions
- `../dataset_writers/` — Per-attribute zarr array serializers
- `../spatial_ordering/` — Points/Lines space-filling-curve ordering glue
- `../gsplat_assembly.py` — GSplat-specific validation + ordering + array writing
- `../node_common.py` — Shared node validators and rendering attr defaults
- `../../ordering.py` — Space-filling-curve primitives
