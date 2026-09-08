# luxar.io._compiler.geometry_writers

**Internal write pipelines** for the four geometry types (Points, Lines, GSplats, Mesh) plus the `sound` node. Each module exports a single `write_*` function that sequences the validation → ordering → array encoding → metadata stamping pipeline for its geometry. These are the bodies of the `LuxarZarrCompiler.write_{points,lines,gsplats,mesh}` orchestrator methods; the orchestrator builds a `GeometryWriteCtx` or `GSplatsWriteCtx`, calls the appropriate pipeline here, then records the returned metadata in its cache.

## Purpose

Extract the per-geometry pipeline bodies from the orchestrator so each geometry concern lives in one focused module. The split is behavior-preserving: all zarr writes are unchanged; only the control flow is reorganized.

## Module Ownership

| Module | Geometry | Entry Point | Context Type |
|--------|----------|-------------|--------------|
| **points.py** | Points | `write_points(ctx: GeometryWriteCtx, path, positions, colors=None, radii=None, sharpness=None, scalars=None, labels=None, image_labels=None, keys=None, **attrs)` | `GeometryWriteCtx` |
| **lines.py** | Lines | `write_lines(ctx: GeometryWriteCtx, path, vertices, widths, colors=None, sharpness=None, scalars=None, indices=None, line_type="polyline", labels=None, image_labels=None, keys=None, **attrs)` | `GeometryWriteCtx` |
| **gsplats.py** | GSplats | `write_gsplats(ctx: GSplatsWriteCtx, path, centers, amplitudes, cholesky_factors, colors=None, labels=None, image_labels=None, keys=None, **attrs)` | `GSplatsWriteCtx` |
| **mesh.py** | Mesh | `write_mesh(ctx: GeometryWriteCtx, path, vertices, faces, normals=None, normal_dims=None, colors=None, scalars=None, shading=None, double_sided=True, labels=None, image_labels=None, keys=None, **attrs)` | `GeometryWriteCtx` |
| **gsplats.py** | GSplats subtree | `write_gsplat_leaf_subtree(ctx: GSplatsWriteCtx, path, leaf, **attrs)` | `GSplatsWriteCtx` |
| **sound.py** | Sound | `write_sound(ctx: GeometryWriteCtx, path, payload, fmt, positions, *, sound_attrs, **attrs)` | `GeometryWriteCtx` |

The pipelines are stateless: they read only the narrow config in the `Ctx` dataclass (encoder, compressor, ordering settings, zarr store) and return a metadata dict for the caller to record.

## Call Flow

### Points Pipeline (`write_points`)

1. **Fail-fast pre-write gate** (runs BEFORE zarr group creation):
   - `validate_render_attrs(attrs, POINTS_RESERVED_ATTRS)` — reserved writer-stamp collisions
   - `validate_node_path(path)` — every segment must be a valid node name; the bare root path `overlays` is reserved (screen-space overlay metadata lives under `overlays/<name>`) and rejected here
   - `validate_positions_for_writing(positions)` → `(n_points, n_dims)`
   - `validate_points_channels(n_points, colors=…, radii=…, sharpness=…, scalars=…, labels=…, image_labels=…, keys=…)` — every per-point channel check, in one shared function for the same reason mesh has `validate_mesh_arrays`: a SECOND caller runs **exactly this** and nothing else. `compositing.validate_points_channels_before_split` runs it against the SOURCE point count before a `partition=` / `additive_lod=` / `substitutive_lod=` split, because the per-part slicer passes a wrong-length channel through whole and a part whose own count happens to match then ACCEPTS it (#1437) — where the plain-leaf path refuses the same input. `image_labels` (#1491) is the exception to that mechanism: it has no per-part slicer at all (it rides only the finest `substitutive_lod=` child), so its check closes a narrower, DIFFERENT stranding shape — see the function's own docstring. Sharing the function is what keeps the pre-split gate from drifting from what the child write accepts. It covers, in this order:
     - `validate_colors_for_writing(colors, n_points, channels=(3,4))` — if colors is an array (Points accept RGBA: alpha is per-point opacity). Storage DTYPE included: a COLOR array must be floating point, or integer `uint8`/`uint16`; a wider/signed integer or a `complex` array is refused, because the encoder refuses those rather than converting them — and it refused from inside `write_colors`, one dataset after `positions` (#1489). The check runs last inside that validator so its precedence matches the encoder's (negativity first, empty arrays pass through)
     - `validate_broadcast_color(colors, "colors")` — if colors is a tuple/list
     - `validate_radii_for_writing(radii, n_points)` — arrays AND broadcast scalars
     - `validate_sharpness_for_writing(sharpness, n_points)` — arrays AND broadcast scalars
     - `validate_scalars_preflight(scalars, n_points)` — length check
     - `validate_labels_for_writing(labels, n_points)` — sequence-of-str type + length
     - `validate_labels_for_writing(keys, n_points, context="keys", noun="Keys")` — if `keys is not None` (same validator, same rules — the CSR serializer UTF-8-encodes each entry, so a non-str or a length mismatch must be caught before any array reaches disk)
     - `validate_image_labels_for_writing(image_labels, n_points)` — length (dense) / index bounds (sparse dict) + per-item type (#1491)
   - `prepare_transform_attrs(attrs, ctx.store)` — transform / nd_transform normalization

2. **Setup**: `ctx.store.require_group(path)`

3. **Spatial ordering**:
   - `build_points_ordering(positions, n_points, n_dims, radii_for_ordering, ctx.ordering_ctx, ctx.store, dataset_ctx=ctx.dataset_ctx)` is called unconditionally; it returns `ordering_data` when spatial ordering applies and `None` when ordering is disabled or not applicable
   - Apply `ordering_data["sort_order"]` to `positions` and all non-broadcasted arrays (skip arrays with `shape[0] == 1`)

4. **Write arrays**:
   - `write_positions(group, positions, ordering_data, ctx.dataset_ctx)` (opts into `per_array_bytes=True` internally)
   - `write_colors(group, colors, ordering_data, n_points, ctx.dataset_ctx, per_array_bytes=True)` — if `colors is not None`
   - `write_radii(group, radii, ordering_data, n_points, ctx.dataset_ctx)` — if `radii is not None` (passes `per_array_bytes=True` through to `write_positive_scalar`; returns `max_radius`, which is also stamped as the `max_radius` group attr here)
   - `write_bounded_scalar(group, sharpness, "sharpnesses", (0.0, SHARPNESS_MAX), ordering_data, n_points, ctx.dataset_ctx, "sharpness", per_array_bytes=True)` — if `sharpness is not None` (the written dataset name is `"sharpnesses"`)
   - `write_scalars(group, scalars, ordering_data, n_points, ctx.dataset_ctx, per_array_bytes=True)` — if `scalars is not None`

   Points opt every per-point array into `per_array_bytes=True` (each array's first-axis chunk is sized to its own dtype byte budget, aligned to a multiple of the spatial-index `chunk_size` atom); Lines/GSplats keep the default `False` (plain atom-sized chunks).
   - `ctx.write_colormap_lut(group, attrs)` — writes the `colormap_lut` dataset when `colormap` needs one (a custom array, or a matplotlib/colorcet name, which is then rewritten to `"custom"`); built-in named colormaps write no LUT

5. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs)` — fill `opacity=1.0`, `absorption=1.0`, `gamma=1.0`, `intensity=1.0`, `offset=0.0` (only if absent); `blending_mode` is deliberately never stamped (no identity value)
   - `group.attrs.update(attrs)` then stamp `type="points"`, `n_points`, `ndim`, `has_colors` / `has_radii` / `has_sharpness` / `has_scalars` (user-supplied + default rendering attrs land via the `update(attrs)` call; `max_radius` was already stamped in step 4 when radii are present)

6. **Compute bounds**:
   - `compute_position_bounds(positions)` → `position_bounds`, stamped as the `position_bounds` group attr and forwarded to `ctx.update_scene_bounds(...)` unless the caller set `_skip_scene_bounds` (the multi-LOD parent writer aggregates the global bounds once instead)

7. **Spatial ordering metadata**:
   - `write_points_ordering_to_zarr(group, ordering_data, ctx.compressor)` — if `ordering_data is not None` (sets `has_spatial_index`, and puts the curve name in `metadata["ordering"]`; `"none"` otherwise, so `Points.ordering` reports the writer's real choice)

8. **Write labels** (CSR serialization; `sort_order` derived from `ordering_data`):
   - `write_string_channels_csr(group, labels=…, keys=…, n_elements=n_points, compressor=ctx.compressor, sort_order=sort_order, metadata=metadata)` — writes each present text channel and stamps its `has_*`. One call for both so they cannot receive DIFFERENT permutations, which is the shape of every mis-pairing bug this channel has had (#1917)
   - `write_image_labels_csr(group, image_labels, n_points, ctx.compressor, sort_order)` — if `image_labels is not None`
   - A multi-LOD **parent** instead gets a single union CSR (`write_ladder_union_labels_csr`) and its `additive_<i>` levels get none — the writer is called with `labels=None` plus the private `_return_sort_order=True` flag, which returns this node's `sort_order` in the metadata so the parent can build that union. `keys` rides the identical path (its own union CSR on the parent), which is why the flag is set when the ladder is labelled **or** keyed

9. **Return metadata**: `{"n_points", "ndim", "path", "has_colors", "has_radii", "has_sharpness", "position_bounds", "ordering"}` plus (conditionally) `max_radius`, `has_scalars`, `has_spatial_index`, `has_labels`, `has_image_labels`, `has_keys` (no `"type"` key)

### Lines Pipeline (`write_lines`)

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, LINES_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_positions_for_writing(vertices)` → `(n_vertices, n_dims)`
   - Validate `line_type` in `("segments", "polyline", "loop", "indexed")`
   - Type-specific vertex count checks (segments: even, polyline: ≥2, loop: ≥3, indexed: requires `indices`)
   - `validate_line_indices(indices, n_vertices)` — the indexed edge list, shared with a second caller like the two functions below. Normalizes with `np.asarray` and accepts only a flat even-element `(2E,)` array or an `(E, 2)` pair array, then integer dtype and `[0, n_vertices)` bounds — all before `convert_to_indexed`, which is what makes the bounds check meaningful. Returns the normalized array. `compositing.validate_line_indices_before_split` runs it from each split branch of `add_lines`, ahead of that branch's topology builder: `lod.lines.identify_polylines` checks only dtype and bounds and then reshapes to pairs, so an `(E, 3)` array was reinterpreted as `3E/2` edges the author never wound and written (#1437). Topology before channels, mirroring mesh's faces-first order
   - `validate_lines_channels(n_vertices, widths=…, colors=…, sharpness=…, scalars=…, labels=…, image_labels=…, keys=…)` — every per-vertex channel check in one shared function, the Points sibling (see its entry above for why it is shared). `widths` is required, so it is validated first and unconditionally; all seven channels are per-VERTEX, not per-segment. It covers, in this order:
     - `validate_widths_for_writing(widths, n_vertices)` — arrays AND broadcast scalars
     - `validate_colors_for_writing(colors, n_vertices, channels=(3,4))` — if colors is an array
     - `validate_broadcast_color(colors, "colors")` — if colors is a tuple/list
     - `validate_sharpness_for_writing(sharpness, n_vertices)` — arrays AND broadcast scalars
     - `validate_scalars_preflight(scalars, n_vertices)` — length check
     - `validate_labels_for_writing(labels, n_vertices)` — if `labels is not None` (labels are per-vertex)
     - `validate_labels_for_writing(keys, n_vertices, context="keys", noun="Keys")` — if `keys is not None`
     - `validate_image_labels_for_writing(image_labels, n_vertices)` — length (dense) / index bounds (sparse dict) + per-item type, if `image_labels is not None` (#1491)
   - `prepare_transform_attrs(attrs, ctx.store)`

2. **Setup**: `ctx.store.require_group(path)` and print the named write header. For `segments` input with at least 16 vertices, a warn-only authoring lint fires when more than 90% of consecutive edges form a forward coordinate chain (immediate `(a,b),(b,a)` reversals are excluded). The warning names the logical node and fires once across partition leaves; it recommends shared-index `polyline`/`indexed` authoring for joint continuity.

3. **Convert to indexed representation**:
   - `convert_to_indexed(n_vertices, line_type, indices)` → a single `(S, 2)` uint32 `segments` array (the first arg is the vertex COUNT, not the vertices array); `n_segments = segments.shape[0]`

4. **Spatial ordering** (dual-indexed: order both vertices AND segments):
   - `build_lines_ordering(vertices, segments, widths, n_vertices, n_dims, n_segments, ctx.ordering_ctx, ctx.store, dataset_ctx=ctx.dataset_ctx)` → `ordering_data` or `None`
   - Replace `vertices` / `segments` with `ordering_data["sorted_vertices"]` / `ordering_data["sorted_segments"]`
   - Permute all non-broadcasted per-vertex arrays (widths, colors, sharpness, scalars) by `ordering_data["vertex_sort_indices"]`; labels are likewise reordered per-vertex via `vertex_sort_indices` at write time. Only `segments` itself is segment-count.

5. **Write arrays**:
   - `vertices`: `SemanticType.COORDINATE`, 2-D chunks via `calculate_intelligent_chunks`, `deduplicate=False`, `allow_lut=False` (raw reader)
   - `segments`: `SemanticType.INDEX`, 2-D chunks `(segment_chunk_size, 2)` (from the segment ordering's `chunk_size` if ordering present, else the constant `2048`), `deduplicate=False` (raw reader)
   - `widths`: via `write_positive_scalar` (rejects negative; same default-precision policy as Points radii; `deduplicate=False` keeps the chunk-bound slack tied to this array's encoding)
   - `colors`, `sharpness`, `scalars`: per-vertex, same as Points
   - `ctx.write_colormap_lut(group, attrs)`: writes the `colormap_lut` dataset when `colormap` needs one (a custom array, or a matplotlib/colorcet name, which is then rewritten to `"custom"`); built-in named colormaps write no LUT

   (No `indices` dataset is written — only `vertices` / `segments` / `widths` and the optional per-vertex arrays.)

6. **Spatial ordering metadata**:
   - `write_lines_ordering_to_zarr(group, ordering_data, ctx.compressor)` — if `ordering_data is not None` (sets `has_spatial_index`)

7. **Apply rendering defaults** + stamp attrs:
   - `apply_default_render_attrs(attrs)` — same rendering defaults as Points
   - `group.attrs.update(attrs)` then stamp `type="lines"`, `n_vertices`, `n_segments`, `ndim`, `original_line_type`, `has_colors` / `has_sharpness` / `has_scalars`, `max_width`, ordering attrs (`ordering` / `vertex_ordering` / `segment_ordering`, or `ordering="none"`) (user + default rendering attrs land via the `update(attrs)` call)

8. **Compute bounds**:
   - `compute_position_bounds(vertices)` → `position_bounds`, stamped as the `position_bounds` group attr and forwarded to `ctx.update_scene_bounds(...)` unless the caller set `_skip_scene_bounds` (the multi-LOD parent writer aggregates the global bounds once instead)

9. **Write labels** (CSR serialization; `sort_order` = `ordering_data["vertex_sort_indices"]` when ordered, per-vertex):
   - `write_string_channels_csr(group, labels=…, keys=…, n_elements=n_vertices, compressor=ctx.compressor, sort_order=sort_order, metadata=metadata)` — both text channels through one call, so both get the SAME per-vertex permutation (#1917)
   - `write_image_labels_csr(group, image_labels, n_vertices, ctx.compressor, sort_order)` — if `image_labels is not None`
   - A multi-LOD **parent** instead gets a single per-vertex union CSR (`write_ladder_union_labels_csr`) and its `additive_<i>` levels get none — the writer is called with `labels=None` plus the private `_return_sort_order=True` flag, which returns this node's per-vertex `sort_order` in the metadata so the parent can build that union. `keys` rides the identical path

10. **Return metadata**: `{"n_vertices", "n_segments", "ndim", "original_line_type", "has_colors", "has_sharpness", "max_width"}` plus ordering keys and `position_bounds` (and conditionally `has_spatial_index`, `has_scalars`, `has_labels`, `has_image_labels`, `has_keys`) — no `"type"` key

### GSplats Pipeline (`write_gsplats`)

1. **Fail-fast pre-write gate**:
   - `validate_render_attrs(attrs, GSPLATS_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_gsplat_inputs(centers, amplitudes, cholesky_factors, colors)` → `(centers, amplitudes, cholesky_factors, colors, n_splats, n_dims, cholesky_is_uniform)`
   - `validate_labels_for_writing(labels, n_splats)` — if `labels is not None`
   - `validate_labels_for_writing(keys, n_splats, context="keys", noun="Keys")` — if `keys is not None`
   - `validate_image_labels_for_writing(image_labels, n_splats)` — length (dense) / index bounds (sparse dict) + per-item type, if `image_labels is not None` (#1491). GSplats has no `substitutive_lod=` wrapper of its own (a gsplat leaf IS the coarse-level representation other geometry types lift into), so this check does not need its own pre-split gate the way Points/Lines/Mesh's `substitutive_lod=` wrappers do. The GRAFT door (`_reject_labels_on_a_grafted_wrapper` in `core/group/gsplats_pipeline/from_io.py`) is a pre-WRAPPER gate rather than a pre-split one, but it hoists this same validator too, on its one-flat-leaf exemption (#1505) — so this call is not the only one any more.

2. **Setup**: `ctx.store.require_group(path)`

3. **Spatial ordering**:
   - `barrier_dims = scene_barrier_dims(ctx.store, n_dims)` — from scene `Dimensions` metadata (discrete non-display dims), or `None` if no scene dims
   - `apply_gsplat_spatial_ordering(centers, amplitudes, cholesky_factors, colors, label_ids, n_splats, n_dims, cholesky_is_uniform, ctx.ordering_ctx, truncation_radius, barrier_dims=scene_barrier_dims(ctx.store, n_dims), dataset_ctx=ctx.dataset_ctx)` → 7-tuple `(centers, amplitudes, cholesky_factors, colors, label_ids, ordering_data, centers_encoding_plan)` (the `truncation_radius` from `attrs` is passed as the `coverage_sigma` arg; `ordering_data` is `None` if ordering was not applied)

4. **Write arrays**:
   - `write_gsplat_arrays(group, centers, amplitudes, cholesky_factors, colors, label_ids, label_vocabulary, n_splats, n_dims, cholesky_is_uniform, ordering_data, centers_encoding_plan, ctx.dataset_ctx)` → `metadata`

5. **Apply rendering defaults** + stamp attrs (stamped BEFORE labels are written):
   - `ctx.apply_gsplat_group_attrs(group, metadata, attrs)` — a bound orchestrator method returning `None`; it delegates to `apply_gsplat_group_attrs(...)` and stores the warn-once colormap-LUT flag on the orchestrator instance (it is NOT threaded through the ctx)
   - This resolves the colormap LUT, prepares/validates `transform` + `nd_transform`, fills rendering defaults (`opacity`, `absorption`, `gamma`, `intensity`, `offset`, `truncation_radius` — `blending_mode` is deliberately never stamped), stamps authoritative `type="gsplats"` attrs, and adds `position_bounds` into `metadata`
   - Then `ctx.update_scene_bounds(metadata["position_bounds"])` folds the leaf's bounds into the scene extent

6. **Write annotations** (CSR serialization; `sort_order` derived from `ordering_data`):
   - `_write_element_annotations(group, labels=labels, keys=keys, image_labels=image_labels, n_splats=n_splats, compressor=ctx.compressor, ordering_data=ordering_data, metadata=metadata)` — writes every present text/image channel with the same permutation

7. **Return metadata**: the `metadata` dict from `write_gsplat_arrays` — `{"n_splats", "ndim", "has_colors", "has_label_ids", "amplitude_range", "center_bounds"}` plus ordering keys, `position_bounds` (added by `apply_gsplat_group_attrs`), and — all conditional — `label_vocabulary` (when `has_label_ids`), `amplitude_data_range` (when `amplitudes` is a non-empty array; note that finalize then HARMONIZES that window across the whole gsplat structure — see `finalize/amplitude_window.py`), `has_labels` / `has_image_labels` / `has_keys` (no `"type"` or `"lut_tone_mapping_warned"` key) — plus the mass statistics `amplitude_mass` / `amplitude_mass_weighted_mean`, which are **unconditional** (`compute_amplitude_mass_stats` normalizes every non-finite or mass-less case to `0.0` / `0.0`, so "present and zero" and "absent" stay distinguishable — the harmonization reads absence as a legacy store)

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

### Mesh Pipeline (`write_mesh`)

The shortest of the four, and structurally so: no spatial ordering (the viewer
loads a mesh whole, so a chunk index has nothing to skip), no primary size scalar
(a triangle's extent comes from its own vertices, not a per-element
radius/width/covariance), and no partition. A *substitutive* LOD level is just a
normal mesh leaf, so it comes through here like any other; what mesh has no path
for is an *additive* ladder inside a single leaf.

1. **Fail-fast pre-write gate** (runs BEFORE zarr group creation):
   - `validate_render_attrs(attrs, MESH_RESERVED_ATTRS)`
   - `validate_node_path(path)`
   - `validate_mesh_arrays(vertices, faces, …)` — every array check, in one shared
     function because `add_mesh(substitutive_lod=…)` runs **exactly this** before it
     decimates anything and before `add_lod_group` creates the group. Without that,
     `colors` / `normals` / `shading` (validated per level) were refused only from
     inside whichever child's write hit them first — a `kind=lod` group left with
     no children at all, or missing only its finest one, depending which level
     failed. `labels`, `keys` and `image_labels` (#1491) fail a DIFFERENT way: all
     three are forwarded ONLY to the finest child, written last, so a wrong one used to be
     refused deep inside that child's OWN write — after its other arrays were
     already on disk — leaving every level, finest included, fully written and
     loadable, with only that channel silently missing. Either shape
     is a strand the plain-leaf path's "nothing written" does not have. It covers:
     - `validate_positions_for_writing(vertices, context="vertices")` → `(n_vertices, n_dims)`, then `validate_vertices_for_writing(vertices)` for the `MAX_MESH_VERTICES` (2^27) ceiling. Order matters: the cap reads `shape[0]`, meaningful only once the array is known 2D
     - `validate_faces_for_writing(faces, n_vertices)` — layout `(F,3)` or flat `(3F,)`, integer dtype, `min >= 0`, `max < n_vertices`, `F >= 1`. Runs BEFORE the `uint32` cast, which is what makes the bounds check meaningful
     - `normals` / `normal_dims` enforced as a **pair in both directions** — each is meaningless alone
     - `shading` must be `"smooth"` / `"flat"` / `"none"`; `double_sided` must be a bool
     - `validate_colors_for_writing(..., channels=(3,4))` or `validate_broadcast_color`, `validate_scalars_preflight`, `validate_labels_for_writing` — the last one twice, once for `labels` and once for `keys` (`context="keys", noun="Keys"`)
     - `validate_image_labels_for_writing(image_labels, n_vertices)` — length (dense) / index bounds (sparse dict) + per-item type, if `image_labels is not None` (#1491)
   - `prepare_transform_attrs(attrs, ctx.store)` — not idempotent, so exactly once

2. **Normalize faces** to `(F, 3)` `uint32`. Safe only here: the validator has established an integer dtype and both bounds, and the vertex cap keeps every admitted index far below 2^32, so the cast is value-preserving.

3. **Authoring lint** (warn-only): the unwelded-vertices heuristic — `V == 3F` *and* no shared vertex index means the mesh was authored as independent triangles rather than a welded surface. Routed through `ctx.claim_authoring_warning("mesh", key)` so a partition's leaves would collapse to one message.

4. **Encode arrays**: `vertices` (`COORDINATE`) and `faces` (`INDEX`), both with `deduplicate=False, allow_lut=False` — same reason as `Lines.segments`: the loader reads them as raw chunked zarr without resolving `array_ref`, so dedup would drop geometry for a byte-identical sibling and LUT encoding of grid-snapped values would decode as garbage. Then optional `normals` (`COORDINATE` — per-axis `uint16` over `[-1,1]`, a free 2x over float32, and it correctly blocks broadcasting since a normal is always per-vertex), `colors`, `scalars`.

5. **Stamp attrs** below `attrs.update` so a caller cannot clobber presence truth, plus `ordering="none"` (stamped rather than omitted, so a reader never distinguishes "no ordering" from "attr missing").

6. **Update scene bounds** and write label / key / image-label CSR arrays — the two text channels through one `write_string_channels_csr` call, as Points and Lines do.

### Sound Pipeline (`write_sound`)

The smallest pipeline, because a sound node has no element arrays: the adder
(`core/group/adders/sound.py`) has already sniffed the clip, validated every knob
and resolved the `hidden=` sugar, so the writer only lands what it is handed.

1. `validate_node_path(path)`; `prepare_transform_attrs(attrs, ctx.store)` (once).
2. Optional `positions` `(K, ndim)` as a PLAIN float32 array (`create_array`,
   `compressor=None`) — not the quantizing encoder: K is a handful of rows, the
   uint16 grid is degenerate for one row (every coordinate collapses to code 0),
   and the viewer reads the array raw to run the slab kernel on exact hidden
   coordinates.
3. The clip as a plain store key via `_zarr_compat.write_raw_bytes(group,
   "audio.mp3" | "audio.m4a", payload)`; `attrs["audio_file"]` names it.
4. Attrs: the compositing pass-throughs first, then the writer's own truth on top
   (`type="sound"`, `format`, `audio_file`, `has_positions`, `n_positions`,
   `ndim`, `ordering="none"`, `position_bounds` when spatial, `duration_ms` when
   `mutagen` is importable).

Deliberately **no** `ctx.update_scene_bounds()`: a sound source never stretches
the scene's framing. The bytes reach `content_hash` through
`finalize/hashing.py::PAYLOAD_FILE_ATTRS` (`"audio_file"`).

## Context Types

**`GeometryWriteCtx`** (Points / Lines / Mesh):
- `store: zarr.Group` — open zarr store
- `dataset_ctx: DatasetCtx` — encoder, encoding_mode, compressor
- `ordering_ctx: OrderingCtx` — enable_spatial_index, ordering_method
- `compressor: CompressorLike` — scene default compressor (used by the ordering + label writers)
- `update_scene_bounds: Callable[[Dict[str, List[float]]], None]` — scene-bounds accumulator hook
- `write_colormap_lut: Callable[[zarr.Group, Dict[str, Any]], None]` — custom-colormap LUT writer hook
- `claim_authoring_warning: Callable[[str, str], bool]` — warn-once registry keyed by `(geometry_kind, path)`, so each type's authoring lint fires once per logical node (partition leaves share their parent key) and one type cannot silence another's on the same node

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

1. **Fail-fast pre-write gate**: The input validators run BEFORE `require_group(path)`, so an invalid input cannot leave a partial node on disk — this deliberately includes `prepare_transform_attrs` for Points and Lines (it reads `store.attrs["scene_dimensions"]` but is still run in the gate so a bad transform can't leak a partial node). `image_labels`' LENGTH / sparse-index checks now run in this same gate too (`validate_image_labels_for_writing`, shared with `write_image_labels_csr` — see `labels/README.md`), closing the #1491 stranding class. The store-dependent steps that remain post-write, and so can leak a partial node on failure, are `image_labels`' blob normalization + CSR write (`normalize_image_label` needs PIL / file reads), custom colormap-LUT resolution, and — for GSplats only — `transform` / `nd_transform` normalization (deferred inside `apply_gsplat_group_attrs`) (F7 residual — transactional/temp-dir writes are a separate project).

2. **Broadcast detection**: Scalars and tuples/lists are passed through to the encoder without expansion. Arrays with `shape[0] == 1` are treated as broadcasted and are NOT reordered by spatial ordering (the encoding layer handles the broadcast).

3. **Spatial ordering**: Points/Lines reorder ALL per-element arrays (positions, colors, radii, etc.) via fancy indexing. GSplats reorder all arrays inside `apply_gsplat_spatial_ordering`. Broadcasted arrays (scalars, tuples, or `shape[0] == 1`) are skipped.

4. **Reserved attrs**: Each geometry has a set of reserved attr names that the writer stamps authoritatively (`POINTS_RESERVED_ATTRS`, `LINES_RESERVED_ATTRS`, `GSPLATS_RESERVED_ATTRS`). User-provided attrs with these names are rejected at the gate.

5. **Lines dual indexing**: Lines are stored as a dual-indexed representation — `vertices` (D-space positions) and `segments` (2×D-space vertex-pair indices). The `line_type` parameter controls how the input `vertices` are interpreted (`"segments"`, `"polyline"`, `"loop"`, `"indexed"`); `convert_to_indexed` normalizes all types to the canonical indexed form. Connectivity is index-based: equal endpoint coordinates in distinct vertex rows do not form a joint. Both `vertices` and `segments` are written with `deduplicate=False` so the viewer's raw chunked-zarr reader never sees an `array_ref`.

6. **Rendering defaults**: Points, Lines and Mesh stamp the same rendering defaults (`opacity`, `absorption`, `gamma`, `intensity`, `offset`) via `apply_default_render_attrs`. GSplats stamp the same set plus `truncation_radius` via `apply_gsplat_group_attrs` (NOT `apply_default_render_attrs`). `blending_mode` is deliberately never stamped (it has no identity value).

## Testing

The geometry writers are NOT unit-tested in isolation (they have no standalone API). The shared test suites in `io/tests/` exercise them through `LuxarZarrCompiler`:

- **test_compiler_integration.py** / **test_compiler_improvements.py** / **test_compiler_colormap.py** / **test_compiler_nd_bounds.py** — End-to-end scene creation, colormap/LUT handling, and nD bounds
- **test_ordering_points.py** / **test_ordering_lines.py** / **test_ordering_gsplats.py** — Spatial ordering integration
- **tests/_compiler/test_lines_authoring_lint.py** — indexed-layout boundaries and the warn-only exploded-chain heuristic (including graph false positives and partition warn-once behavior)
- **gsplats/io/tests/test_save_load.py** — GSplat leaf/ladder save-load round trips

## See Also

- `../../compiler.py` — `LuxarZarrCompiler` orchestrator that calls these pipelines
- `../context.py` — `GeometryWriteCtx` / `GSplatsWriteCtx` dataclass definitions
- `../dataset_writers/` — Per-attribute zarr array serializers
- `../spatial_ordering/` — Points/Lines space-filling-curve ordering glue
- `../gsplat_assembly.py` — GSplat-specific validation + ordering + array writing
- `../node_common.py` — Shared node validators and rendering attr defaults
- `../../ordering.py` — Space-filling-curve primitives
