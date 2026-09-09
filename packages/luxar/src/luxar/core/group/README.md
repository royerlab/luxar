# luxar.core.group

The `Group` scene-graph node and the helper machinery behind its `add_*`
data-adding methods. A `Group` contains child data nodes (`Points`, `Lines`,
`GSplats`) and other `Group`s, forming a hierarchical scene. `Scene` inherits
from `Group`, so every method documented here works on both.

`Group` itself is re-exported from this package, so the historical import paths
keep resolving:

```python
from luxar.core.group import Group   # direct
from luxar import Group              # top-level re-export
```

## Overview

`group.py` holds the public API surface only: each of `add_points`,
`add_lines`, `add_gsplats`, `add_gsplats_from_data`, `add_gsplats_from_file`,
and `add_gsplats_from_volume` is a thin, fully-documented delegate that imports
its `*_impl` from a sibling submodule and forwards keyword args. The substantial
logic — input coercion, `dim_order` remapping, validation, spatial partitioning,
and LOD decomposition — lives in the sibling submodules and subpackages, keeping
`group.py` focused on signatures and docstrings.

Two specialized `Group` *kinds* are produced by these methods when the data
warrants it:

- **`kind=partition`** — a compile-time spatial decomposition of one large
  geometry node into many smaller `part_<i>` children so per-child frustum
  culling / LOD can kick in. Triggered by `partition=` on the leaf adders.
- **`kind=lod`** — a group that selects one of N alternative children at runtime
  based on projected size. Produced by the gsplats LOD pipeline (see the `lod/`
  and `gsplats_pipeline/` subpackages).

To the user both look like a single logical layer of the original geometry type.

Pre-wrapper gates provide the best caller-facing diagnostics, while writer
transactions hold the structural invariant: every public geometry add removes
its new subtree and restores scene/compiler authoring state if a descendant
write fails. Finalize warns and prunes wrappers created in a separate successful
call that still have no children, so caught child refusals remain publishable.

## File structure

```
group/
├── __init__.py          # re-exports Group
├── group.py             # Group class: public add_* API (delegates to adders/ + gsplats_pipeline/)
├── auto_partition.py     # resolve_auto_partition — compiler-level opt-in auto-partition
├── compositing.py        # COMPOSITING_ATTRS, AUTHORED_APPEARANCE_ATTRS, WRITER_STAMPED_APPEARANCE_DEFAULTS, slice_optional_array, is_broadcast_color, validate_*_before_split, position_bounds_from_array, strip_absent_attr_kwargs
├── dim_order.py          # apply_dim_order_positions / apply_dim_order_cholesky
├── partition.py          # BSP splitters + PartitionSpec + validate_partition_group
├── adders/               # per-leaf add_<type> bodies (Points / Lines / GSplats)
├── gsplats_pipeline/     # high-level gsplats write path (from_data / from_file / from_volume)
└── lod/                  # LOD-group machinery + per-geometry axis resolvers
```

## Public API (`group.py`)

`Group` extends `Node` and adds the data-adding methods. The first two,
`_find_scene` and `_require_scene_writer`, walk up the parent chain to reach the
root `Scene` (for dimension validation and writer access) and fail loudly when a
group is detached from a scene or the scene lacks a writer.

| Method | Returns | Purpose |
|--------|---------|---------|
| `add_points(name, positions, ...)` | `Points` or `Group` | Add a point cloud; `partition=` yields a partition, `substitutive_lod=` yields an LOD, and combining them yields an overview LOD whose finest child is partitioned |
| `add_lines(name, vertices, widths, ...)` | `Lines` or `Group` | Add polylines/segments/loops; partition-aware (polylines stay atomic) |
| `add_gsplats(name, centers, amplitudes, cholesky_factors, ...)` | `GSplats` or `Group` | Add Gaussian splats from explicit arrays; partition/substitutive composition partitions every substitutive level |
| `add_gsplats_from_data(name, result, ...)` | `GSplats` or `Group` | Add from a `GSplatData`; resolves substitutive (`substitutive_lod=`, with `lod_group=` as an alias) and additive (`additive_lod=`) LOD axes |
| `add_gsplats_from_file(name, path, ...)` | `GSplats` or `Group` | Load from a `.gsplats.zarr`; `flatten=True` collapses stored structure before applying new partition/LOD controls |
| `add_gsplats_from_volume(name, volume, ...)` | `GSplats` or `Group` | Fit splats to a volume and add in one step (optional progressive multi-pass) |

Cross-cutting keyword arguments shared by the leaf adders:

- `dim_order` — map data columns to scene dimensions *by name* (e.g.
  `["Z", "Y", "X"]` for 3D data in a 4D scene). Unmapped dims are filled via
  `fill` and auto-extended.
- `fill` / `fill_sigma` — fixed coordinate values (and, for gsplats, Cholesky
  embedding sigmas) for unmapped dimensions.
- `extend_to_all` — visibility extension across non-displayed dimensions.
- `partition` — spatial-decomposition control (see below).
- `**attrs` — node attributes such as `layer`, `visible`, `opacity`,
  `absorption`, `intensity`, `gamma`, `blending_mode`, `colormap`.

```python
# A group can add data directly
group = scene.add_group("cells", opacity=0.8, blending_mode="additive")
group.add_points("pts", positions)               # written under cells/pts

# dim_order: lower-dimensional data into a higher-dimensional scene
scene.add_gsplats_from_data(
    "splats", result_3d,
    dim_order=["Z", "Y", "X"],   # data cols → scene dims
    fill={"Time": 0.0},           # fixed value for the unmapped Time dim
    fill_sigma={"Time": 0.5},     # Cholesky sigma for the unmapped dim
)
```

## Helper modules

### `partition.py` — spatial BSP decomposition

A `kind=partition` `Group` is a recursive **binary space partition** of one
large geometry node (10M+ elements) into smaller child nodes. The user passes
`partition=True` or `partition=dict(max_elements=N, rule=...)`; when the
decomposition yields more than one part the leaf adder returns a wrapper `Group`
carrying `display_type` and `max_elements`, with `part_<i>` children.

Three split rules, selected via `partition=dict(rule=...)`:

| Rule | Split position | Trade-off |
|------|----------------|-----------|
| `"median"` (default) | median of the longest axis | balanced part counts in O(n)/level; best for clustered scientific data |
| `"midpoint"` | geometric midpoint of the longest axis | cheapest; axis-aligned tiles, may be uneven on clusters |
| `"sah"` | surface-area-heuristic minimum (binned, 32 candidates) | best on heavily skewed data; O(N·n_candidates·3)/level |

Exports:

- `BSPNode`, `spatial_bsp_tree`, and `spatial_bsp_polyline_tree` — production
  tree builders used by the native geometry adders; scene adders select the
  currently displayed position columns and serialize their original indices.
- `bsp_leaf_parts` — shared left-first leaf flattening used by all four adders.
- `persist_pruned_bsp_tree` — drops unwritten regions, renumbers surviving
  `child_index` labels, stamps the serialized tree on the partition wrapper,
  and warns when its actual split axes are not displayed.
- `median_bsp_partition` / `midpoint_bsp_partition` / `sah_bsp_partition` and
  the median/midpoint polyline wrappers — flat-list conveniences retained as
  parity oracles for the production tree builders.
- `warn_if_oversized_single_part` — surfaces the degenerate
  fully-coincident-input case where the BSP cannot split below `max_elements`.
- `warn_if_partition_axes_not_displayed` — surfaces persisted split axes that
  the current display configuration cannot map for exact viewer traversal.
- `validate_partition_group` — well-formedness check (≥1 child, present
  `display_type`, `max_elements >= 1`, homogeneous child `display_type`).
- `PartitionSpec` — value-vocabulary type alias for `partition=`
  (`None` / `True` / `dict`).
- `resolve_partition_spec` — validator for that vocabulary, returning
  `(max_elements, rule)`. One spelling for all four adders and for the two
  gsplats pre-wrapper gates (`lod_group=` and the graft door), which have to
  judge the same spec one level above the leaf that consumes it (#1550). It
  rejects dict keys other than `max_elements` and `rule`. It also rejects
  `False`, deliberately: normalising the bypass away is the CALLER's job
  (`resolve_auto_partition` for the adders, the gates' own skip), since by the
  time a spec is being resolved into a cap the decision to partition is made.
- `DEFAULT_MAX_ELEMENTS = 1_000_000` — cap used for bare `partition=True`.

### `auto_partition.py` — compiler-level opt-in

`resolve_auto_partition(scene, n_elements, user_partition)` resolves the
effective `partition=` for a leaf-adder call. When
`LuxarZarrCompiler(auto_partition_max_elements=N)` is set and the caller did not
pass `partition=`, any leaf over the threshold is auto-decomposed via a
synthetic `dict(max_elements=N)`. The `partition=False` sentinel is the
recursion guard the partition wrappers use so per-part recursive calls don't
re-trigger auto-partition.

### `compositing.py` — partition/LOD wrapper primitives

Pure data operations (no `Group`/`Node` references) shared by the partition and
LOD wrapper builders:

- `COMPOSITING_ATTRS` — frozenset of attribute names (`transform`, `opacity`,
  `absorption`, `gamma`, `intensity`, `offset`, `blending_mode`, `layer`,
  `visible`, `nd_transform`) that ride on the wrapper `Group` rather than being copied onto
  each child; compositing semantics flow down to children via Group inheritance
  at render time. `truncation_radius` is deliberately excluded — it is
  auto-defaulted per leaf and would otherwise shadow a parent under
  nearest-ancestor-wins. `colormap` is not routed here either, but only as a
  preference: it DOES compose in the viewer since #1600, and copying it onto
  each child is the shape the layers panel's `deriveColormapFromDescendants`
  reads back.
- `AUTHORED_APPEARANCE_ATTRS` — the subset a structure-only rebuild (`gsplat lod`
  and the rest of the rewriting family) carries from the source root to the
  output root, so re-laddering a dataset does not silently reset the look
  (#1600). `COMPOSITING_ATTRS` minus `transform`, plus `colormap`. `transform`
  is excluded because the stored matrix is already column-major and the writer
  would transpose it a second time; `colormap` is included now that the writer
  no longer manufactures a shadowing per-leaf `"gray"` and the viewer composes
  the attr root→leaf (the `'custom'` sentinel is the one value not carried — its
  LUT lives in a sibling array the attrs-only read cannot reach). Read with
  `luxar.gsplats.io.load_gsplats.read_authored_appearance`.
- `WRITER_STAMPED_APPEARANCE_DEFAULTS` (+ `IDENTITY_COMPOSITING_ATTRS`) — the
  value the WRITER manufactures for an appearance attr nobody set, single-sourced
  so the stamp sites (`node_common.apply_default_render_attrs`,
  `gsplat_assembly.apply_gsplat_group_attrs`, and the `layer` stamp in
  `gsplats/io/save_gsplats.py`) and the READER that has to recognise a stamp
  cannot drift. `gsplat merge` uses it to tell "the author chose this" from
  "nobody chose anything": a value equal to the manufactured default casts no
  vote in the agreement rule.
- `mirror_written_colormap(attrs, writer, path)` — copy the colormap the writer
  actually stamped onto the adder's attrs, so the returned node's attr
  write-back cannot put a manufactured `"gray"` on disk that the writer
  declined.
- `slice_optional_array(value, indices, n_elements)` — slice a per-element leaf
  parameter by index; pass scalars / `None` / mis-sized inputs through unchanged.
- `validate_labels_before_split(labels, n_elements)` — its companion guard: reject
  a wrong-length `labels` against the FULL element count before a partition / LOD
  decomposition. Needed precisely because `slice_optional_array` passes a mis-sized
  list through unchanged, which would hand every part / level the same unsliced
  list and write labels into the wrong slots. Two callers reach it directly:
  `validate_gsplats_channels_before_split` (the one geometry whose channel
  validator has no labels channel; Points and Lines get the same check from
  the writer sweep their gates delegate to instead), and
  `gsplats_pipeline.from_io._validate_labelled_leaf_length`, called from the
  graft door's one-flat-leaf label gate (#1505) — a deliberate EXCEPTION: that
  call site is a pre-WRAPPER gate rather than a pre-split one (a bare-leaf
  graft has no wrapper and no split at all) and it DOES change which fault a
  multi-fault call reports, the same trade `validate_points_channels_before_split`
  sanctions below for "a NaN position, an unknown attr". Every other caller's
  check belongs to a wrapper's pre-split gate and never to the top of a leaf
  adder, so the plain-leaf gate order stays exactly as it was. Multi-CHILD gsplats wrappers
  cannot use it at all: a multi-level substitutive `lod_group=` on
  `add_gsplats_from_data`, and any `graft_gsplat_node` subtree holding more than
  one leaf (`kind=lod` / `kind=partition`), REFUSE `labels=` / `image_labels=`
  outright (#1471) — each child holds its own set of splats, so no single list
  has a per-element correspondence to slice, and a list whose length coincides
  with a part's own count was silently written onto every part. The test is leaf
  COUNT, not node type: a one-part `kind=partition` of a flat leaf is non-matrix-
  shaped yet holds every splat, so it still labels. A single LADDERED leaf is
  refused for a different reason (the additive writer has no labels channel) and
  so gets its own message, `labels_on_a_laddered_leaf_reason`; the two multi-leaf
  doors share one template
  (`gsplats_pipeline.from_data.labels_on_wrapper_reason`). An explicit
  `labels=None` is normalised away first (`strip_absent_attr_kwargs`, which since
  #1496 does the same for `image_labels` / `partition` / `colors` /
  `truncation_radius` / `colormap` / `coverage_fraction`) so it means
  "absent" rather than an unknown attr key. Label a single-level node
  (`lod_group=False`) or a single-leaf file, or hand-build the wrapper and give
  each child its own labels.
- `validate_points_channels_before_split(n_points, colors=…, radii=…, sharpness=…,
  scalars=…, labels=…, image_labels=…)`, `validate_lines_channels_before_split(n_vertices, widths=…,
  …)`, `validate_gsplats_channels_before_split(centers, amplitudes,
  cholesky_factors, colors=…, labels=…)` — the same pre-split gate for EVERY
  other per-element channel, not just labels. Each one CALLS its geometry's
  writer-side sweep (`geometry_writers.points.validate_points_channels`,
  `geometry_writers.lines.validate_lines_channels`,
  `gsplat_assembly.validate_gsplat_inputs`) against the SOURCE element count
  rather than restating the rules, so the gate cannot drift from what the child
  write accepts. What that buys is a per-element CHANNEL verdict identical with
  and without `partition=` / `additive_lod=` / `substitutive_lod=` — identical
  exception type and message, which the tests assert byte-for-byte. The gate runs
  ABOVE the positions checks on the split paths, so a call that also trips a bad
  position (e.g. a NaN) reports the channel fault first here and the positions
  fault on the plain-leaf path — both refuse, neither writes. The scene-DIMENSION
  count is one exception: since #1446 the adders check it above their split
  branches, so it precedes this gate on both paths and a mismatched column count
  is reported first either way. The node-attrs check is a second exception:
  since #1529 (Points/Lines) and #1534 (Mesh, GSplats) `validate_render_attrs`
  also runs at every adder's entry, above this gate, so an unknown/reserved
  attr wins there too on all four geometry types now. The explicit
  `extend_to_all` preflight is a third exception on split paths: it runs at the
  wrapper branch after those structural checks but before this channel gate,
  matching the flat writer's order.
  Same placement rule as the labels guard (first statement of the wrapper impl,
  never a leaf adder). Labels, then image labels, come last, in that order, as in
  the flat write: for Points and Lines via `validate_labels_for_writing` then
  `validate_image_labels_for_writing` inside the shared writer sweep (#1491 added
  the latter — it has no per-part slicer at all, since it rides only the finest
  `substitutive_lod=` child, so it closes a narrower and differently-shaped strand
  than the rest of this gate; see `validate_points_channels_before_split`'s own
  docstring), for GSplats via `validate_labels_before_split` (whose validator has
  no labels channel) plus its own `image_labels` check in the flat gate (GSplats
  has no `substitutive_lod=` wrapper to pre-split). The GSplats gate also RETURNS
  the `cholesky_is_uniform` flag its
  validator already computed, so the wrapper does not restate that rule either.
  Every legal broadcast form the flat path accepts passes the GATE and reaches
  disk on every path, `substitutive_lod=` included: the gsplat lift broadcasts a
  uniform `colors` onto the coarse levels, alpha column and all (gsplats carry
  per-splat alpha, and every shader scales intensity by it, so a dropped alpha
  would brighten each coarse level by `1/alpha` at the LOD seam), where it used
  to refuse it (#1444). A per-element `(N, 4)` RGBA is still refused by the lift
  — not a broadcast form, so outside this gate's parity promise.
- `validate_line_indices_before_split(indices, n_vertices, line_type)` — the
  TOPOLOGY half of the Lines gate, and it runs first (mesh validates `faces`
  before any channel for the same reason). Calls the writer's shared
  `validate_line_indices`, because the split paths reach
  `lod.lines.identify_polylines` — dtype and bounds only, then `reshape(-1, 2)` —
  before any writer gate, so an `(E, 3)` array was reinterpreted as `3E/2` edges
  and written. Called from each split branch of `add_lines`, before that
  branch's topology builder (the wrapper impls are too late to beat the reshape).
- `is_broadcast_color(colors)` — classify a uniform RGB(A) list/tuple by
  type/shape, because its OWN length can collide with the element count (3 points
  with an RGB triple, 4 vertices with an RGBA one) and `slice_optional_array`
  would otherwise gather its components as if they were element rows. The uniform
  `(k,)` Cholesky collides the same way (`k = 6` for 3-D, which a 6-splat node
  matches exactly); the GSplats wrapper reads that off the `cholesky_is_uniform`
  flag its gate returns rather than re-testing the shape. On these three
  geometries the parts are disjoint, so the mis-slice REFUSES a legal input
  (sometimes only after a partial node is written) rather than mis-writing
  silently; mesh, whose parts share vertices, is the case where it can be
  silent, and it shares this classifier.
- `position_bounds_from_array(positions)` — per-axis min/max of an `(N, D)`
  array, matching the compiler's per-leaf `position_bounds` shape.
- `strip_absent_attr_kwargs(attrs, keys)` + `ABSENT_WHEN_NONE_RENDER_ATTRS` —
  "an explicit `None` means absent", stated once for every door that can be
  handed one. The tuple is the two render attrs whose `None` no value validator
  catches, so it reaches disk: `colormap` (rewritten to a LUT-less `'custom'`,
  which the viewer renders as viridis) and `coverage_fraction` (a literal null
  selector threshold). Every leaf adder strips it at its entry, above its own
  colours gate, attrs gate and structural branches (#1574); the gsplats pipeline
  passes a wider set that also covers the leaf params it forwards through
  wrappers (#1496, below). The key set is a required argument precisely so
  neither door can inherit the other's — every OTHER render attr refuses a
  `None` loudly on its own, and swallowing those would mask typos.

### `dim_order.py` — dimension remapping

Two free functions backing the `dim_order=` convenience kwarg (no `self`
coupling — they take a `Scene` reference and return pure NumPy arrays):

- `apply_dim_order_positions(...)` — reorder + pad positions/vertices/centers to
  the scene's full dimensionality; auto-derives `extend_to_all` from the unmapped
  dims when the caller didn't set it.
- `apply_dim_order_cholesky(...)` — reorder and embed packed Cholesky factors
  (via `gsplats.utils.trils.embed_cholesky_packed`) so the GSplats writer can
  consume them; validates that `fill_sigma` keys name unmapped scene dims.

## Subpackages

- **[`adders/`](adders/README.md)** — per-leaf `add_<type>_impl` bodies for
  Points, Lines, GSplats, and Mesh, plus their partition- and multi-LOD-wrapper
  helpers (mesh has all three: partition, substitutive-LOD, and a multi-LOD
  wrapper whose ladder is a REVEAL — concentric shells of faces — since an
  arbitrarily ordered prefix of an index buffer is a holed surface).
  `group.py`'s public methods are thin delegates over these.
- **[`gsplats_pipeline/`](gsplats_pipeline/README.md)** — the high-level gsplats
  write path backing `add_gsplats_from_data` / `_from_file` / `_from_volume`;
  resolves the substitutive and additive LOD axes and routes to a flat node, a
  per-sublod multi-LOD node, or a `kind=lod` group.
- **[`lod/`](lod/README.md)** — LOD-group machinery: geometry-agnostic threshold
  derivation, the `kind=lod` validator, the shared display-type resolver, the
  per-geometry axis resolvers, and the spatial-uniform / Poisson-disk ordering
  samplers.

## See Also

- [core/README.md](../README.md) — the `core` scene-graph module (Scene, Node,
  DataNode, Points, Lines, GSplats, Mesh, Dimensions, transforms)
- [gsplats/README.md](../../gsplats/README.md) — Gaussian-splat fitting, LOD, and
  the `GSplatData` container
