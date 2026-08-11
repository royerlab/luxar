# luxar.core.group.adders

Per-leaf adder implementations for `Group`. Each geometry type (Points,
Lines, GSplats, Mesh) has its own module here holding the body of
`Group.add_<type>` along with its partition-wrapper and multi-LOD-wrapper
helpers (mesh has a partition wrapper and a substitutive-LOD one, but no
multi-LOD/additive wrapper — it still refuses the additive prefix ladder).

## Overview

The orchestrator file `core/group/group.py` keeps the public method
signatures + docstrings and delegates to the free functions in this folder.
Every function takes the calling `group: Group` as its first argument and is
otherwise keyword-only, so `Group.add_points` / `add_lines` / `add_gsplats` /
`add_mesh` are thin one-line delegates over the matching `*_impl` here.

This split keeps `group.py` focused on the public API surface while the
(substantial) add logic — input coercion, `dim_order` application,
validation, auto-partition, additive-LOD decomposition, and the actual
writer calls — lives in dedicated, parallel modules.

## File Structure

```
adders/
├── __init__.py    # Module docstring only (no re-exports)
├── points.py      # add_points_impl + partition / multi-LOD wrappers
├── lines.py       # add_lines_impl + partition / multi-LOD wrappers
├── gsplats.py     # add_gsplats_impl + partition wrapper
└── mesh.py        # add_mesh_impl + partition / substitutive-LOD wrappers
```

## Modules

### `points.py`

- `add_points_impl(group, *, name, positions, ...)` → `Points | Group`
- `add_points_partition_wrapper_impl(group, *, name, pos_arr, parts, ...)` → `Group`
- `add_points_multi_lod_wrapper_impl(group, *, name, pos_arr, levels, ...)` → `Points`

`DEFAULT_POINT_RADIUS = 0.5` is applied when `radii` is not supplied.

### `lines.py`

- `add_lines_impl(group, *, name, vertices, widths, ...)` → `Lines | Group`
- `_collect_partition_vertex_indices(polyline_indices, polyline_parts)` → per-part vertex arrays
- `_bucket_indexed_edge_indices(indices, part_vertex_indices, n_vertices)` → stable edge buckets + shared local map
- `add_lines_partition_wrapper_impl(group, *, name, vert_arr, polyline_indices, polyline_parts, ...)` → `Group`
- `add_lines_multi_lod_wrapper_impl(group, *, name, vert_arr, polyline_levels, ...)` → `Lines`

Lines partition at **polyline granularity** — the BSP runs over per-polyline
centroids and whole polylines / connected components are atomic (each lands
in exactly one part). `indexed` inputs group their original edges by a stable
NumPy part permutation and remap them through one reusable global-to-local
vertex array; this avoids per-edge Python tuples and per-part dictionaries
while preserving exact graph topology and authored edge order. `segments`
re-emit consecutive member pairs; `polyline` / `loop` inputs are already a
single polyline so the BSP yields a single part.

### `gsplats.py`

- `add_gsplats_impl(group, *, name, centers, amplitudes, cholesky_factors, ...)` → `GSplats | Group`
- `add_gsplats_partition_wrapper_impl(group, *, name, ctr_arr, chol_arr, ...)` → `Group`

`dim_order` is applied to **both** `centers` (via `apply_dim_order_positions`)
and `cholesky_factors` (via `apply_dim_order_cholesky`), with `fill_sigma`
controlling the Cholesky embedding of unmapped dimensions. GSplats has no
additive-LOD wrapper here — LOD ladders for splats are built as a post-process
in the `gsplats` package (`luxar gsplat lod`), not at add time.

### `mesh.py`

- `add_mesh_impl(group, *, name, vertices, faces, ...)` → `Mesh | Group`
- `_add_mesh_partition(group, *, name, vert_arr, faces_arr, ...)` → `Group | None`
- `add_mesh_substitutive_lod_wrapper_impl(group, *, name, vert_arr, ...)` → `Group | Mesh`

Mesh partitions at **face granularity** — the BSP runs over face centroids, so
`max_elements` counts faces and no triangle is ever cut. A part cannot be a slice
of the inputs the way the sibling wrappers' are: faces reference a shared vertex
table, so each part gathers and renumbers its own vertices
(`luxar.mesh.split.split_mesh_by_faces`), duplicating those on the cut, and the
per-vertex `normals` / `colors` / `scalars` / `labels` follow that index.

`substitutive_lod=` writes a `kind=lod` group whose coarse children are
progressively DECIMATED copies of the surface (`luxar.mesh.decimate`) and whose
finest child is the original. It cannot be combined with `partition=` — the same
refusal `add_points` / `add_lines` carry, which is why a hand-built
`kind=partition` wrapper is the only route to per-tile mesh ladders; like its
three siblings the wrapper derives its `coverage_fraction` thresholds through
`lod.group.derive_coverage_fractions`, so such a ladder is auto-anchored at
fills-screen (finest `4.0`) instead of the whole-object `1.0`. There is still no
multi-LOD (additive)
wrapper: mesh refuses the additive prefix ladder (and
`blending_mode='volumetric'`) with a per-case explanation.

## Add-Path Anatomy

Each `*_impl` walks the same ordered decision tree:

1. **Coerce + shape-check** the primary array to `(N, D)`.
2. **Apply `dim_order`** (`apply_dim_order_positions`, plus
   `apply_dim_order_cholesky` for gsplats), which may also extend
   `extend_to_all` for unmapped dimensions.
3. **Colormap / colors / scalars mutual-exclusivity gate** — the only place these
   are checked; every branch below therefore rejects an invalid combination
   (the LOD wrappers used to return early and skip them).
4. **Check the scene-dimension count** (`scene._validate_dimension_count`) — the
   hard column-count-vs-scene-dimensions raise (a non-2-D array included), on the
   caller's own array, with the caller's own node name. Placement is load-bearing
   (#1446): above every structural branch below, so a mismatch cannot strand a
   childless wrapper group; after step 2, which is what decides the final column
   count; and below step 3, so a colours fault keeps precedence. Only the count
   half is here — the range `UserWarning` half stays in the single-leaf write at
   step 9, so it still fires once per written leaf rather than once more for the
   source array.
5. **Substitutive-LOD branch** (points/lines, when `substitutive_lod` is set):
   delegate to the substitutive wrapper, whose coarse levels are synthesised
   gsplats under a `kind=lod` group. Fires before (auto-)partition.
6. **Resolve auto-partition** via `resolve_auto_partition(scene, n, partition)`
   — an opt-in compiler heuristic (default off). A user-explicit `partition=`
   always wins. (Lines does not yet wire the auto-partition heuristic; it
   honors only explicit `partition=`.)
7. **Partition branch** (when `partition` is set and `D >= 2`): run a BSP
   (`median` / `midpoint` / `sah`) capped at `max_elements`, and if it yields
   more than one part, delegate to the partition wrapper. A single part falls
   through to the regular write.
8. **Additive-LOD branch** (points/lines, when `additive_lod` is set): build
   prefix-monotone levels and, if more than one level results, delegate to the
   multi-LOD wrapper. Fires after the 1-part-partition fall-through, so a
   single `add_*` call can compose partition-of-additive-LOD.
9. **Single-leaf write**: validate dimensions (the full
   `_validate_data_dimensions`, i.e. the step-4 count check again plus the
   per-dimension range `UserWarning`), resolve `extend_to_all` (this is where
   the `"all"` sentinel becomes a concrete dim-name list for a flat leaf AND
   for every part of a partition, whose recursion re-enters here; the multi-LOD
   wrapper resolves it itself — see below), then call the scene writer
   (`write_points` / `write_lines` / `write_gsplats`) and return the
   constructed `Points` / `Lines` / `GSplats` node.

All `*_impl` entries wrap the body in a `try/except (ValueError, TypeError)`
that re-raises as a `ValueError` with a `Could not add <type> '<name>': ...`
message.

## Wrappers

### Partition wrappers (`kind=partition` group)

Build a wrapper `Group` via `parent.add_partition_group(name, display_type,
max_elements, **wrapper_attrs)` with one leaf child (`part_0`, `part_1`, …)
per BSP part, then persist the wrapper's `position_bounds` from the full input
array. Attrs are split: those in `COMPOSITING_ATTRS` go on the wrapper, the
rest on each leaf. Per-element arrays are sliced into each part via
`slice_optional_array`.

Every wrapper impl opens with its geometry's pre-split gate
(`validate_points_channels_before_split` /
`validate_lines_channels_before_split` /
`validate_gsplats_channels_before_split` from `compositing`), which runs the flat
writer's own step-0 channel sweep against the **source** element count — Points
in the order colors, radii, sharpness, scalars, labels; Lines with `widths`
FIRST, then colors, sharpness, scalars, labels; GSplats as the
amplitudes/Cholesky/colors trio, then labels. The gate does not restate those
rules: it calls the same function the writer calls
(`validate_points_channels` / `validate_lines_channels` /
`validate_gsplat_inputs`, the siblings of mesh's `validate_mesh_arrays`), so a
channel added to a writer's gate is covered here too. Without the gate
`slice_optional_array` passes a wrong-length channel through whole and a part
whose own count happens to match accepts it, so the write succeeds with values on
the wrong elements. Mesh does the same thing in `_validate_partition_sources`;
the gate belongs at the top of the wrapper, never the leaf adder, so the
plain-leaf error order is untouched (a call that also trips the positions/attr
gates therefore reports the channel fault first here). The scene-dimension count
is the one exception: since #1446 it is checked at step 4 of the adder, above the
branch that enters this wrapper, so a wrong column count outranks a wrong-length
channel on both paths alike. Uniform values whose own
length can collide with the element count (an RGB(A) list/tuple, a `(k,)`
Cholesky) are classified before slicing rather than length-tested. Lines
additionally validate `indices` — topology before channels, as mesh validates
`faces` first — via `validate_line_indices_before_split`, called from each split
branch of `add_lines` ahead of that branch's topology builder.

Per-part recursion passes `partition=False` (not `None`) to bypass the
compiler auto-partition heuristic — `None` would re-trigger it on each part
and blow up the leaf count. `dim_order` / `fill` / `fill_sigma` are nulled in
the recursion because they were already applied upstream. `image_labels` is
rejected alongside `partition=`.

### Multi-LOD wrappers (`additive_<i>` subgroups)

Write a single parent node carrying `n_additive_sublods=N` plus a global
`position_bounds`, with one `additive_<i>/` subgroup per LOD level
(`write_points_multi_lod` / `write_lines_multi_lod`). For lines, each subgroup
carries a subset of **whole** polylines with segment indices local to the
subgroup. The returned node is the parent — the user sees one logical node and
the viewer's progressive loader walks the subgroups.

`labels` are written by the writer as ONE union CSR on the **parent** (the
subgroups carry none, because the loader concatenates loaded levels into one
committed buffer), so the wrappers call `scene._notify_labels_added()` exactly as
the flat path does — otherwise a ladder-only scene would get no hover overlay.

Unlike the partition wrapper, this one does not recurse through a leaf adder, so
it resolves `extend_to_all` itself right before the writer call, via the shared
`lod.group.resolve_ladder_extend_to_all` (which leaves `None` unresolved): the
multi-LOD writers stamp that value verbatim onto the parent group AND every
`additive_<i>/` subgroup, so an unresolved `"all"` sentinel would reach disk
where the viewer expects a list of dimension names.

## Dependencies

**Sibling modules** (`core/group/`):
- `auto_partition.resolve_auto_partition`
- `partition` — BSP kernels (`median_bsp_partition`, `midpoint_bsp_partition`,
  `sah_bsp_partition`, `median_bsp_polylines`, `midpoint_bsp_polylines`,
  `DEFAULT_MAX_ELEMENTS`, `warn_if_oversized_single_part`)
- `compositing` — `COMPOSITING_ATTRS`, `position_bounds_from_array`,
  `slice_optional_array`, `is_broadcast_color`,
  `validate_points_channels_before_split`,
  `validate_lines_channels_before_split`,
  `validate_line_indices_before_split`,
  `validate_gsplats_channels_before_split` (labels ride along last — via the
  shared writer sweep for points/lines, via `validate_labels_before_split` for
  gsplats)
- `dim_order` — `apply_dim_order_positions`, `apply_dim_order_cholesky`
- `lod.points`, `lod.lines` — additive-LOD level builders and polyline
  identification
- `lod.group` — `additive_level_stats`, `breakpoints_kind_of`,
  `resolve_ladder_extend_to_all`

**Node types** (`core/`): `Points`, `Lines`, `GSplats`, `Node`, `Group`.

**External**: `numpy`, `arbol` (`aprint`).

## See Also

- [../README.md](../README.md) — `luxar.core` overview (Scene / Group / Node)
- `../group.py` — public `add_points` / `add_lines` / `add_gsplats` delegates
- `../partition.py` — BSP partition kernels
- `../lod/` — additive-LOD level construction
- `../compositing.py` — compositing-attr classification and array slicing
