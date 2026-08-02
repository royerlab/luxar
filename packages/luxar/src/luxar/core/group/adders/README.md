# luxar.core.group.adders

Per-leaf adder implementations for `Group`. Each geometry type (Points,
Lines, GSplats) has its own module here holding the body of
`Group.add_<type>` along with its partition-wrapper and multi-LOD-wrapper
helpers.

## Overview

The orchestrator file `core/group/group.py` keeps the public method
signatures + docstrings and delegates to the free functions in this folder.
Every function takes the calling `group: Group` as its first argument and is
otherwise keyword-only, so `Group.add_points` / `add_lines` / `add_gsplats`
are thin one-line delegates over the matching `*_impl` here.

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
└── gsplats.py     # add_gsplats_impl + partition wrapper
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

## Add-Path Anatomy

Each `*_impl` walks the same ordered decision tree:

1. **Coerce + shape-check** the primary array to `(N, D)`.
2. **Apply `dim_order`** (`apply_dim_order_positions`, plus
   `apply_dim_order_cholesky` for gsplats), which may also extend
   `extend_to_all` for unmapped dimensions.
3. **Resolve auto-partition** via `resolve_auto_partition(scene, n, partition)`
   — an opt-in compiler heuristic (default off). A user-explicit `partition=`
   always wins. (Lines does not yet wire the auto-partition heuristic; it
   honors only explicit `partition=`.)
4. **Partition branch** (when `partition` is set and `D >= 2`): run a BSP
   (`median` / `midpoint` / `sah`) capped at `max_elements`, and if it yields
   more than one part, delegate to the partition wrapper. A single part falls
   through to the regular write.
5. **Additive-LOD branch** (points/lines, when `additive_lod` is set): build
   prefix-monotone levels and, if more than one level results, delegate to the
   multi-LOD wrapper. Fires after the 1-part-partition fall-through, so a
   single `add_*` call can compose partition-of-additive-LOD.
6. **Single-leaf write**: validate dimensions, resolve `extend_to_all`, check
   colormap/colors/scalars mutual-exclusivity, then call the scene writer
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

## Dependencies

**Sibling modules** (`core/group/`):
- `auto_partition.resolve_auto_partition`
- `partition` — BSP kernels (`median_bsp_partition`, `midpoint_bsp_partition`,
  `sah_bsp_partition`, `median_bsp_polylines`, `midpoint_bsp_polylines`,
  `DEFAULT_MAX_ELEMENTS`, `warn_if_oversized_single_part`)
- `compositing` — `COMPOSITING_ATTRS`, `position_bounds_from_array`,
  `slice_optional_array`
- `dim_order` — `apply_dim_order_positions`, `apply_dim_order_cholesky`
- `lod.points`, `lod.lines` — additive-LOD level builders and polyline
  identification

**Node types** (`core/`): `Points`, `Lines`, `GSplats`, `Node`, `Group`.

**External**: `numpy`, `arbol` (`aprint`).

## See Also

- [../README.md](../README.md) — `luxar.core` overview (Scene / Group / Node)
- `../group.py` — public `add_points` / `add_lines` / `add_gsplats` delegates
- `../partition.py` — BSP partition kernels
- `../lod/` — additive-LOD level construction
- `../compositing.py` — compositing-attr classification and array slicing
