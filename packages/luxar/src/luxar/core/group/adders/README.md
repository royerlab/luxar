# luxar.core.group.adders

Per-leaf adder implementations for `Group`. Each geometry type (Points,
Lines, GSplats, Mesh) has its own module here holding the body of
`Group.add_<type>` along with its partition-wrapper and multi-LOD-wrapper
helpers (mesh has all three: a partition wrapper, a substitutive-LOD one, and a
multi-LOD wrapper whose ladder is a REVEAL — see `mesh.py` below).

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
└── mesh.py        # add_mesh_impl + partition / substitutive-LOD / multi-LOD (reveal) wrappers
```

## Modules

### `points.py`

- `add_points_impl(group, *, name, positions, ...)` → `Points | Group`
- `add_points_partition_wrapper_impl(group, *, name, pos_arr, parts, ...)` → `Group`
- `add_points_multi_lod_wrapper_impl(group, *, name, pos_arr, levels, ...)` → `Points`

`DEFAULT_POINT_RADIUS` (`= 0.5`) is applied when `radii` is not supplied. The constant lives in `luxar.typing_utils.constants` and is imported here — it is also the radius the viewer draws a radii-less node with, and the extent the spatial index expands a no-radii chunk's bounds by (`luxar.io.ordering.compute_chunk_bounds_points`), so authoring, bounds and rendering agree by construction.

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
- `add_mesh_multi_lod_wrapper_impl(group, *, name, vert_arr, parts, ...)` → `Mesh`

Mesh partitions at **face granularity** — the BSP runs over face centroids, so
`max_elements` counts faces and no triangle is ever cut. A part cannot be a slice
of the inputs the way the sibling wrappers' are: faces reference a shared vertex
table, so each part gathers and renumbers its own vertices
(`luxar.mesh.split.split_mesh_by_faces`), duplicating those on the cut, and the
per-vertex `normals` / `colors` / `scalars` / `labels` follow that index.

`substitutive_lod=` writes a `kind=lod` group whose coarse children are
progressively DECIMATED copies of the surface (`luxar.mesh.decimate`) and whose
finest child is the original. It cannot be combined with `partition=` — the same
refusal `add_lines` carries. Points instead composes that pair into a global-coarse
overview above partitioned fine detail; Mesh still requires a hand-built
`kind=partition` wrapper for per-tile ladders. Like its three siblings the wrapper
takes BOTH its per-child `coverage_fraction` thresholds and the group-level
`selector` naming their units from `lod.group.resolve_lod_ladder` (which calls
`derive_coverage_fractions` underneath when no explicit
`coverage_fractions=[...]` list was given), so such a ladder is auto-anchored at
fills-screen (finest `PARTITION_FINEST_AREA` = `1.0`) instead of the whole-object
`WHOLE_OBJECT_FINEST_ANCHOR` = `0.5`.

`additive_lod=` writes a REVEAL ladder — `additive_<i>/` levels holding concentric
shells of FACES, innermost first — through `add_mesh_multi_lod_wrapper_impl` and
`write_mesh_multi_lod`. It is the same shape as the Points/Lines multi-LOD wrapper
with three mesh-specific differences, all downstream of "a triangle is three
references, not a row":

* A level is a re-indexing, not a slice. The branch splits FACES
  (`lod.mesh.make_additive_lod_mesh`) and re-indexes each group through
  `luxar.mesh.split.split_mesh_by_faces`, so the wrapper receives `MeshPart`s and
  gathers each per-vertex channel through `part.vertex_index` — a boundary vertex
  is stored once per level that touches it (logged as a duplication factor).
* `labels` / `image_labels` DEGRADE the ladder to a plain leaf with a
  `UserWarning` instead of riding it: there is no union index space for the CSR
  the sibling ladders put on their parent (see `write_mesh_multi_lod`).
* No energy stamps. `additive_level_stats` suppresses them for every reveal
  method, and the wrapper passes all-zero energies plus
  `energy_kind="mesh-reveal-no-energy"` to say the same thing a second way.

`additive_lod=` composes with neither `substitutive_lod=` nor `partition=` yet —
each pairing is refused by name, where Points and Lines compose both.
`blending_mode='volumetric'` and a non-reveal additive `method` stay refused with a
per-case explanation.

## Add-Path Anatomy

Each `*_impl` walks the same ordered decision tree. Above all of it, as the first
statement inside each of the four `try` blocks — above every consumer of `attrs`,
and below only the argument-composition refusals `lines.py` and `mesh.py` raise ahead
of their `try` (which judge `partition` / `substitutive_lod` / `additive_lod`
against each other and never touch these two keys) — sits
`strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_RENDER_ATTRS)`: a
present-but-`None` `colormap` or `coverage_fraction` is deleted so it means
*absent* rather than a value (#1574 — a `None` colormap otherwise survives step 3
and step 5 untouched and is rewritten by `sync_custom_colormap_attr` into a
LUT-less `'custom'` the viewer renders as viridis). Once here rather than at each
consumer, because steps 3, 5 and 10, every structural branch and every
`sync_custom_colormap_attr` call site (two per module, one in `gsplats.py`) read
the same dict — and a structural branch is where the leak actually shipped: with
the strip scoped to the flat write, `add_points(partition=…, colormap=None)`
stamps the LUT-less `'custom'` on *every part*. Only render attrs are
in the set: the structural keys whose `None` also means absent (`colors`,
`labels`, `image_labels`, `partition`) are named parameters of all four `*_impl`
signatures and can never reach `**attrs`.

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
   count; and below step 3, so a colours fault keeps precedence. It is ABOVE the
   kwarg checks that live inside the branches (a malformed `partition=` /
   `additive_lod=` spec, the `image_labels`-with-`partition` ban, the Lines
   `indices` topology check), so a call that also trips one of those is told
   about the width first — on the flat path as well as the split ones, so the two
   still agree. Only the count half is here — the range `UserWarning` half stays
   in the single-leaf write at step 10, so it fires once per written leaf (none
   under an additive ladder, whose writer never validates) rather than once more
   for the source array.
5. **Node-attrs gate** (`validate_render_attrs`, the flat writer's own step
   0a) — since #1529 (Points/Lines) and #1534 (Mesh, GSplats), run once here
   against the caller's un-split `**attrs`, above every structural branch
   below (same placement reasons as step 4): `substitutive_lod=` and
   `partition=` forward the non-compositing remainder of `**attrs` into a
   synthesised child (a gsplat `child_0`, a decimated mesh `child_0`, or a
   `part_i`; a compositing attr is split out and refused earlier, before any
   child is written), and `additive_lod=` (points/lines/mesh) goes straight to
   the multi-LOD writer, which runs this same validator with no reserved-attrs
   set at all — so on the Points/Lines substitutive branch and on the
   Mesh/GSplats `partition=` branch a bad attr used to be caught only from
   inside the first child (by then the wrapper group itself, childless, was
   already on disk). Mesh's OWN `substitutive_lod=` branch is the one
   exception: it already ran this same validator, with the right
   `MESH_RESERVED_ATTRS` set, before #1534 — so it never stranded anything;
   what #1534 fixed there was only the check's placement relative to
   `extend_to_all` (see step 10). On the additive path a genuinely
   reserved key got the wrong verdict (unknown instead of reserved, nothing
   written) while an unreserved-but-clobbering key (`position_bounds=`)
   wasn't refused at all — real ladder data was written and the writer's own
   stamp silently overwritten. Running it here instead means every split path
   refuses byte-identically to the flat path, with nothing written. This
   placement also outranks the #1437 channel gate at the top of the
   partition, substitutive, and multi-LOD wrappers (see "Partition wrappers"
   below) — matching the flat writer's own order, where node attrs are
   validated before channels. The GSplats leaf implementation's only
   structural door is `partition=`; the public array adder resolves its LOD
   controls through `add_gsplats_from_data` before reaching it.
   Before #1534, `add_mesh(..., partition={"max_elements": 40},
   blending="max")` and `add_gsplats(..., partition={"max_elements": 100},
   blending="max")` raised the same message but still left a childless
   `kind=partition` node that survived `finalize()` — the #1529 stranding one
   geometry type over.
6. **Substitutive-LOD branch** (points/lines/mesh, and GSplats through the
   array adder's `add_gsplats_from_data` dispatch): delegate to the
   substitutive wrapper, whose coarse levels are
   synthesised gsplats (points/lines) or decimated meshes (mesh) under a
   `kind=lod` group. Before that wrapper is written, validate an explicit
   `extend_to_all` once against the scene; `None` remains child-only because its
   candidate analysis warns once per written child. Fires before
   (auto-)partition. For Points, an explicit `partition=` is resolved here and
   becomes the finest child of an overview LOD; a one-part result keeps the
   ordinary whole-object ladder.
7. **Resolve auto-partition** via `resolve_auto_partition(scene, n, partition)`
   — an opt-in compiler heuristic (default off). A user-explicit `partition=`
   always wins. (Lines does not yet wire the auto-partition heuristic; it
   honors only explicit `partition=`.)
8. **Partition branch** (when `partition` is set and `D >= 2`, except the Points
   overview composition already handled in step 6): run a BSP
   (`median` / `midpoint` / `sah`) capped at `max_elements`, and if it yields
   more than one part, validate an explicit `extend_to_all` immediately before
   delegating to the partition wrapper. The preflight sits below partition-spec,
   image-label, topology and split resolution so those existing faults keep
   precedence, but still above the wrapper write; `None` remains child-only to
   avoid an extra advisory warning. Mesh is the exception: its shared
   `extend_to_all` resolution already runs above the structural branches in
   `mesh.py:add_mesh_impl`. A single part falls through to the regular write.
9. **Additive-LOD branch** (points/lines/mesh, when `additive_lod` is set —
   GSplats has no `additive_lod=` on `add_gsplats`; its own additive door is
   `additive_lod=` on the separate `add_gsplats_from_data` adder): build
   prefix-monotone levels and, if more than one level results, delegate to the
   multi-LOD wrapper. Fires after the 1-part-partition fall-through, so a
   single `add_*` call can compose partition-of-additive-LOD — except Mesh,
   whose `additive_lod=` is refused outright alongside either of the other two
   structural params (`_reject_additive_lod_compositions`), so a mesh call
   never actually reaches this branch after a real partition split.
10. **Single-leaf write**: validate dimensions (the full
    `_validate_data_dimensions`, i.e. the step-4 count check again plus the
    per-dimension range `UserWarning`), resolve `extend_to_all` (this is where
    the `"all"` sentinel becomes a concrete dim-name list for a flat leaf AND
    for every part of a partition, whose recursion re-enters here; the multi-LOD
    wrapper resolves it itself — see below), then call the scene writer
    (`write_points` / `write_lines` / `write_gsplats` / `write_mesh`) and
    return the constructed `Points` / `Lines` / `GSplats` / `Mesh` node. The
    writer re-runs the same step-5 node-attrs validator on its way in
    (`validate_render_attrs` is idempotent — it only inspects `attrs`), so a
    flat call validates twice; harmless on its own, but since step 5 now runs
    before this step's `extend_to_all` resolve, a multi-fault flat call reports the attrs fault
    where it used to report the `extend_to_all` one — a deliberate consequence
    of validating attrs first, matching the writer's own step 0a, with no
    change to which single-fault calls succeed or fail. This now holds for
    Mesh too: before #1534 its flat-path fallthrough resolved `extend_to_all`
    before ever reaching the flat writer's own attrs gate (the pre-#1529
    order), and its substitutive branch's dispatcher,
    `_maybe_add_mesh_substitutive_lod`, ran its OWN attrs gate after resolving
    `extend_to_all` — the opposite order from Points/Lines. #1534 moved the
    check to the top of `add_mesh_impl`, above step 6 and every later step, so
    every mesh path now agrees with step 5's placement here too.

The pre-write gates above remain the message-quality layer: known scene-level
faults are rejected before a wrapper exists, so errors name the caller's node
rather than a synthesized child. Structural correctness does not depend on
anticipating every future child failure here; each public `Group.add_*` runs in
a writer transaction that removes a new subtree and restores authoring state on
failure, and finalize warns and prunes wrappers created but never populated.

All `*_impl` entries wrap the body in a `try/except (ValueError, TypeError)`
that re-raises as a `ValueError` with a `Could not add <type> '<name>': ...`
message, built by `compositing.funnel_add_error` (#1491) rather than an f-string
directly: on a `partition=` split (or the Mesh `substitutive_lod=` ladder), a
child level is written by calling the SAME adder again for a synthesised child
name (`part_0`, `child_3`), so a failure inside that recursive call already
carries its OWN `Could not add <type> '<child>': ...` funnel prefix before the
outer call catches it. `funnel_add_error` strips that inner prefix ONLY when
the inner geometry word matches this call's own — never for a cross-geometry
inner failure (the Points/Lines `substitutive_lod=` ladder's coarse children are
GSPLATS nodes, so that inner prefix names a real, different geometry's fault and
is left alone) and never for a `Could not create child … group '<x>': ...`
wrapper-creation prefix. The `aprint` line directly above each raise needs the
UN-NESTED inner text alone (not re-prefixed with this call's own type/name, or
the console log would double-print it), so each adder calls the sibling
`compositing.unnest_add_error` for that line and `funnel_add_error` — which is
built on top of `unnest_add_error` — for the raise, rather than re-deriving the
inner text from `funnel_add_error`'s own output by string surgery. The two
calls always agree, since both strip from the same caught exception.

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
in the order colors, radii, sharpness, scalars, labels, image_labels; Lines with
`widths` FIRST, then colors, sharpness, scalars, labels, image_labels; GSplats
as the amplitudes/Cholesky/colors trio, then labels (GSplats has no
`substitutive_lod=` wrapper of its own, so its `image_labels` check lives only
in the flat gate — see `geometry_writers/README.md`). The gate does not restate those
rules: it calls the same function the writer calls
(`validate_points_channels` / `validate_lines_channels` /
`validate_gsplat_inputs`, the siblings of mesh's `validate_mesh_arrays`), so a
channel added to a writer's gate is covered here too. Without the gate
`slice_optional_array` passes a wrong-length channel through whole and a part
whose own count happens to match accepts it, so the write succeeds with values on
the wrong elements. Mesh does the same thing in `_validate_partition_sources`;
the gate belongs at the top of the wrapper, never the leaf adder — that
placement is still correct for the CHANNEL gate on its own. Three checks now
outrank it: the scene-dimension count (#1446) and node-attrs gate
(`validate_render_attrs`, see step 5 of "Add-Path Anatomy") sit at the adder
entry, while the explicit `extend_to_all` preflight sits at each split branch
immediately before the wrapper hand-off. So a call that also trips one of those
reports THAT fault first, not the channel one; only once all three have passed
does a wrong-length channel get reported here. This is not an accident of
hoisting order: it mirrors the flat writer's own step ordering (node attrs at
step 0a, `extend_to_all` before the write, then the channel sweep at steps 0d–0f
for Points / 0e–0h for Lines — see `geometry_writers/points.py` /
`geometry_writers/lines.py`), so the split paths now agree
with the flat path where, before #1529/#1534, they disagreed (a split call
used to report the channel fault even when the flat call on the same input
would have reported the attrs fault first). Uniform values whose own
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
(`write_points_multi_lod` / `write_lines_multi_lod` / `write_mesh_multi_lod`). For
lines, each subgroup
carries a subset of **whole** polylines with segment indices local to the
subgroup. The returned node is the parent — the user sees one logical node and
the viewer's progressive loader walks the subgroups.

`labels` are written by the writer as ONE union CSR on the **parent** (the
subgroups carry none, because the loader concatenates loaded levels into one
committed buffer), so the wrappers call `scene._notify_labels_added()` exactly as
the flat path does — otherwise a ladder-only scene would get no hover overlay.
MESH IS THE EXCEPTION on both counts: a mesh level re-indexes its own vertices, so
the union index space does not exist — `write_mesh_multi_lod` refuses a labelled
level and the adder degrades a labelled mesh to a flat leaf, so there is no
`_notify_labels_added()` call on that path and none is needed.

Unlike the partition wrapper, this one does not recurse through a leaf adder, so
it resolves `extend_to_all` itself right before the writer call, via the shared
`lod.group.resolve_ladder_extend_to_all` (which leaves `None` unresolved): the
multi-LOD writers stamp that value verbatim onto the parent group AND every
`additive_<i>/` subgroup, so an unresolved `"all"` sentinel would reach disk
where the viewer expects a list of dimension names. (Mesh diverges here too: it
dispatches its structural branches BELOW the `extend_to_all` resolution — the
partition branch needs the resolved names — so the value reaching
`add_mesh_multi_lod_wrapper_impl` is already resolved and the shared helper would
only re-emit the advisory.)

## Dependencies

**Sibling modules** (`core/group/`):
- `auto_partition.resolve_auto_partition`
- `partition` — the `partition=` spec validator (`resolve_partition_spec`, one
  spelling of the value vocabulary for all four adders and for the gsplats
  pre-wrapper gates — no adder resolves `DEFAULT_MAX_ELEMENTS` itself any more),
  the production BSP tree builders (`spatial_bsp_tree`,
  `spatial_bsp_polyline_tree`), shared leaf flattening and prune/persist helpers
  (`bsp_leaf_parts`, `persist_pruned_bsp_tree`), and the three advisories
  (`warn_if_partition_needs_more_dims`, which DROPS the request below 2 spatial
  dims, `warn_if_oversized_single_part`, and
  `warn_if_partition_axes_not_displayed`). Each native partition wrapper
  stamps its pruned `bsp_tree`; the flat splitters remain parity-test helpers.
- `compositing` — `COMPOSITING_ATTRS`, `position_bounds_from_array`,
  `slice_optional_array`, `is_broadcast_color`,
  `validate_points_channels_before_split`,
  `validate_lines_channels_before_split`,
  `validate_line_indices_before_split`,
  `validate_gsplats_channels_before_split` (labels, then image_labels, ride
  along last — via the shared writer sweep for points/lines, via
  `validate_labels_before_split` plus the flat gate's own `image_labels` check
  for gsplats), `unnest_add_error` (strips a same-geometry inner child's own
  `Could not add <type> '<child>': ...` funnel prefix from a caught exception's
  message, returning the un-nested inner text alone, #1491), `funnel_add_error`
  (re-prefixes that un-nested text with the CALLER's own type/name — built on
  top of `unnest_add_error` — for the message an adder re-raises)
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
