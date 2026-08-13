# luxar.core.group.gsplats_pipeline

High-level write path for Gaussian splats. These free functions back the
public `Group.add_gsplats_from_data`, `add_gsplats_from_file`, and
`add_gsplats_from_volume` methods: the orchestrator in
[`core/group/group.py`](../group.py) keeps the public signatures and
docstrings, then delegates the body to the `*_impl` functions here.

## Overview

A fitted (or loaded) [`GSplatData`](../../../gsplats/gsplat_data.py) object
can carry up to two orthogonal LOD axes — a **substitutive** hierarchy
(coarser levels *replace* finer ones) and an **additive** ladder (later
sublods *add to* earlier ones). This package resolves those two axes and
routes the result to one of three write shapes:

| Resolved shape | Write path | On-disk result |
|----------------|------------|----------------|
| multi-substitutive | `add_gsplats_as_lod_group_impl` | a `kind=lod` `Group`, one gsplats child per level |
| single-substitutive + multi-additive | `add_gsplats_multi_lod_impl` | one gsplats leaf with `additive_<i>/` subgroups |
| single-substitutive + single-additive | `Group.add_gsplats` | a flat single-leaf gsplats node |

The substitutive axis is resolved first (it can produce a multi-level
result), then the additive axis (uniform across levels).

## File Structure

```
gsplats_pipeline/
├── __init__.py        # package docstring only (no re-exports)
├── from_data.py       # add_gsplats_from_data_impl — top-level dispatch
├── from_io.py         # add_gsplats_from_file_impl / add_gsplats_from_volume_impl
└── lod_dispatch.py    # add_gsplats_as_lod_group_impl / add_gsplats_multi_lod_impl
```

## Entry points

### `from_data.py` — `add_gsplats_from_data_impl(group, *, name, result, ...)`

The dispatch hub. Validates that `result` is a `GSplatData`, propagates
`truncation_radius` from the data into `attrs` (unless the caller overrode
it), then resolves the two LOD axes via
[`core/group/lod/gsplats.py`](../lod/gsplats.py):

```python
result, explicit_coverage_fractions = \
    resolve_substitutive_axis_gsplats(result, lod_group)
result = resolve_additive_axis_gsplats(result, additive_lod)
```

Branch selection then keys off `result.n_substitutive` and
`result.n_additive_sublods`. Passing `coverage_fraction` while the resolved
result is multi-substitutive raises `ValueError` — thresholds are derived
per-child instead (or set via `lod_group=dict(coverage_fractions=[...])`).

### `from_io.py` — load/fit then delegate

Both functions produce a `GSplatData` and hand it to
`add_gsplats_from_data_impl`:

- `add_gsplats_from_file_impl` — loads a `.gsplats.zarr` via
  `luxar.gsplats.io.load_gsplats` (raises `FileNotFoundError` if the path
  is missing). A matrix-shaped tree goes down the normal data path; a
  genuinely nested one (kind=partition root, or a lod with non-leaf children)
  is GRAFTED node-for-node by `graft_gsplat_node`, which routes attrs exactly
  like `lod_dispatch` does: [`COMPOSITING_ATTRS`](../compositing.py) — including
  `blending_mode` — land on the wrapper **only**, everything else rides onto
  each child. `blending_mode` must NOT be duplicated onto the parts: it is
  nearest-setter-wins, so a part's copy shadows the wrapper and the layer's
  Blend control goes inert. Per-child `coverage_fraction` thresholds ride from
  each node's own `meta`; the **fallback** for a meta-less lod group is
  partition-binding-aware in three ways — the recursion's `_under_partition` flag,
  a `kind=partition` child of the ladder itself (the `overview` shape, the common
  reachable case), and `is_partition_bound(parent or group)` on the SCENE side.
  Note the scope: only a *non*-matrix-shaped subtree is grafted at all, so a
  per-part `add_gsplats_from_file` of an ordinary ladder store never reaches here
  — it is matrix-shaped and anchored by `lod_dispatch`. The scene-side term is a
  defensive fallback for a hand-built / nested lod-of-lods tree, which no library
  producer writes today. Before it grafts, it checks the STORED tree's column count
  against the scene (#1446), below the `dim_order` / `fill` / `fill_sigma` refusal
  the graft path already raises: the chain is built from the on-disk tree before the
  first leaf is added, so without that check a mismatched store refused from inside
  `part_0` / `child_0` and left a childless wrapper behind. One leaf answers for the
  whole subtree — a graft applies no `dim_order`, and both container node types
  reject mixed-`ndim` children at construction. Directly below that (it is the
  first statement of `graft_gsplat_node`, so both pre-graft checks outrank it),
  and for the same fail-before-the-wrapper reason, `_reject_labels_on_a_grafted_wrapper`
  refuses `labels=` / `image_labels=` (#1471) unless the grafted subtree is
  **exactly one leaf with exactly one additive sub-LOD** — the one shape that can
  actually carry them. Leaf COUNT, not node type: a one-part `kind=partition` of
  a flat leaf still has an exact per-element correspondence, so it keeps
  labelling normally **when the list is the right length**; a wrong length is
  now refused right there too, by the same validators the flat writer runs, so a
  mismatched `labels=`/`image_labels=` no longer refuses one level down from
  inside `part_0` with the wrapper already on disk (#1505). But a one-part
  partition of a LADDERED leaf is refused too, for a different reason and with
  a different message
  (`labels_on_a_laddered_leaf_reason`): `write_gsplat_leaf_subtree` has no labels
  channel at all, so exempting it would push the refusal down into `part_0` with
  the wrapper already on disk — and `--recipe tiles` carries a stream ladder by
  DEFAULT, so that is an ordinary call, not a corner case. Multi-leaf
  cannot be sliced here at all — a stored `GSplatPartition` carries no per-part
  index arrays, unlike `add_gsplats(partition=…)`, which slices the BSP `parts`
  it just computed — so the only definable mapping would be implicit
  leaf-concatenation order, and `gsplat flatten` (which emits exactly that order)
  is the remedy the message names. Both doors of the gate share one message
  template, `from_data.labels_on_wrapper_reason(kwarg, structure, remedy)`, with
  the two `*_STRUCTURE` / `*_REMEDY` constants beside it, so the wording cannot
  drift apart. `from_data.strip_absent_label_kwargs` runs first on both and
  DELETES a present-but-`None` key: the leaf adders bind both as named params
  defaulting to `None`, but inside `**attrs` a `None`-valued key is still an
  unknown attr to `validate_render_attrs` (which matches by name, never by
  value), so the idiomatic `labels=maybe_labels` used to strand a wrapper.
- `add_gsplats_from_volume_impl` — fits in one step. With
  `progressive=True` it calls `fit_progressive_gaussian_splats`
  (honoring `max_splats_per_pass`, `psnr_patience`, `max_passes`);
  otherwise `fit_gaussian_splats`. `opacity`, `absorption`, and
  `blending_mode`, when set, are forwarded as scene attrs.

### `lod_dispatch.py` — the two multi-level write paths

**`add_gsplats_as_lod_group_impl`** builds a `kind=lod` `Group` with one
gsplats child per substitutive level, written coarsest→finest and named
`child_<i>`. Substitutive index convention is index 0 = finest,
`n-1` = coarsest, so the level loop iterates in reverse. Per-level splat
counts (summed across each level's additive ladder) feed both logging and
auto-derivation of the per-child `coverage_fraction` thresholds via
[`lod/group.derive_coverage_fractions`](../lod/group.py) when the caller did
not supply explicit ones. That chokepoint picks the anchor from the insertion
point: `coverage_fractions` (finest `1.0`) normally, or
`partitioned_coverage_fractions` (finest `MAX_COVERAGE_FRACTION` = `4.0`) when
`is_partition_bound(parent or group)` — i.e. the ladder is going inside a
hand-built `kind=partition` wrapper, where it switches on one tile rather than
the whole object.

Attribute routing splits on
[`COMPOSITING_ATTRS`](../compositing.py): compositing attrs (opacity,
absorption, gamma, intensity, offset, blending_mode, transform, layer,
visible, nd_transform) land on the wrapper `Group`; everything else (including
`colormap`, deliberately) rides into each child so the user's intent is
not shadowed by the writer's per-leaf `colormap="gray"` default. Each
child is added through `lod_group_node.add_gsplats_from_data(...)` with
both LOD axes explicitly `None`, so the resolvers no-op and the
multi-substitutive branch is never re-entered.

**`add_gsplats_multi_lod_impl`** writes a multi-additive-LOD gsplats leaf.
It builds one `(centers, amplitudes, cholesky_factors, colors)` tuple per
additive sublod — applying `dim_order` per LOD through
[`apply_dim_order_positions` / `apply_dim_order_cholesky`](../dim_order.py)
and validating dimensions against the scene — then calls the shared walker
`write_gsplat_leaf_subtree` (which routes through
`io/_compiler/gsplat_tree.write_gsplat_node`). Specifying both `colors` and
`colormap` raises `ValueError`; a missing colormap defaults to `"gray"` only
when the data has no colors.

## Usage

These are implementation functions; reach them through the public Group
API:

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())

    # Fit a volume and add in one step
    scene.add_gsplats_from_volume("fitted", volume, seeds=8000, n_iters=1000)

    # Load a pre-fitted .gsplats.zarr (v3.3 node tree grafted into the scene)
    scene.add_gsplats_from_file("loaded", "path/to/fitted.gsplats.zarr")
```

## Dependencies

**Internal:**
- `luxar.gsplats` — `GSplatData`, `fit_gaussian_splats`,
  `fit_progressive_gaussian_splats`, `io.load_gsplats`
- `core/group/lod/gsplats.py` — substitutive/additive axis resolvers
- `core/group/lod/group.py` — `derive_coverage_fractions` (and the
  `coverage_fractions` / `partitioned_coverage_fractions` /
  `is_partition_bound` it dispatches between)
- `core/group/compositing.py` — `COMPOSITING_ATTRS`
- `core/group/dim_order.py` — per-LOD dim_order application
- `core/gsplats.py` — the `GSplats` node returned for multi-additive writes

**External:**
- `numpy` — array handling
- `arbol` — structured logging (`aprint`)

## See Also

- [core/README.md](../../README.md) — scene graph and `add_gsplats*` overview
- [core/group/group.py](../group.py) — public method signatures that delegate here
- [gsplats/README.md](../../../gsplats/README.md) — fitting and `GSplatData`
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — the v3.3 node-tree format (leaf / kind=lod / kind=partition)
