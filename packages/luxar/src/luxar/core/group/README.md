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

## File structure

```
group/
├── __init__.py          # re-exports Group
├── group.py             # Group class: public add_* API (delegates to adders/ + gsplats_pipeline/)
├── auto_partition.py     # resolve_auto_partition — compiler-level opt-in auto-partition
├── compositing.py        # COMPOSITING_ATTRS, slice_optional_array, position_bounds_from_array
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
| `add_points(name, positions, ...)` | `Points` or `Group` | Add a point cloud; returns a `kind=partition` wrapper when `partition=` yields >1 part |
| `add_lines(name, vertices, widths, ...)` | `Lines` or `Group` | Add polylines/segments/loops; partition-aware (polylines stay atomic) |
| `add_gsplats(name, centers, amplitudes, cholesky_factors, ...)` | `GSplats` or `Group` | Add Gaussian splats from explicit arrays |
| `add_gsplats_from_data(name, result, ...)` | `GSplats` or `Group` | Add from a `GSplatData`; resolves substitutive (`lod_group=`) and additive (`additive_lod=`) LOD axes |
| `add_gsplats_from_file(name, path, ...)` | `GSplats` or `Group` | Load from a `.gsplats.zarr` file (auto-lowers a stored pyramid into a `kind=lod` group) |
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

- `median_bsp_partition` / `midpoint_bsp_partition` / `sah_bsp_partition` —
  pure-NumPy point/gsplats splitters returning lists of index arrays into the
  original positions (concatenation permutes `range(N)`).
- `median_bsp_polylines` / `midpoint_bsp_polylines` — polyline-atomic variants
  for `add_lines` (every vertex of a polyline lands in one part; accounting is
  by vertex count, splits run over per-polyline centroids).
- `warn_if_oversized_single_part` — surfaces the degenerate
  fully-coincident-input case where the BSP cannot split below `max_elements`.
- `validate_partition_group` — well-formedness check (≥1 child, present
  `display_type`, `max_elements >= 1`, homogeneous child `display_type`).
- `PartitionSpec` — value-vocabulary type alias for `partition=`
  (`None` / `True` / `dict`).
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
  at render time. `colormap` and `truncation_radius` are deliberately excluded —
  they are auto-defaulted per leaf and would otherwise shadow a parent under
  nearest-ancestor-wins.
- `slice_optional_array(value, indices, n_elements)` — slice a per-element leaf
  parameter by index; pass scalars / `None` / mis-sized inputs through unchanged.
- `position_bounds_from_array(positions)` — per-axis min/max of an `(N, D)`
  array, matching the compiler's per-leaf `position_bounds` shape.

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
  Points, Lines, and GSplats, plus their partition- and multi-LOD-wrapper
  helpers. `group.py`'s public methods are thin delegates over these.
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
  DataNode, Points, Lines, GSplats, Dimensions, transforms)
- [gsplats/README.md](../../gsplats/README.md) — Gaussian-splat fitting, LOD, and
  the `GSplatData` container
