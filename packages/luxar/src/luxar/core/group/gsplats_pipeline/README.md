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
  is missing).
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
auto-derivation of `coverage_fractions` via
[`lod/group.coverage_fractions`](../lod/group.py) when the caller did
not supply explicit thresholds.

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

    # Load a pre-fitted .gsplats.zarr (v3.2 node tree grafted into the scene)
    scene.add_gsplats_from_file("loaded", "path/to/fitted.gsplats.zarr")
```

## Dependencies

**Internal:**
- `luxar.gsplats` — `GSplatData`, `fit_gaussian_splats`,
  `fit_progressive_gaussian_splats`, `io.load_gsplats`
- `core/group/lod/gsplats.py` — substitutive/additive axis resolvers
- `core/group/lod/group.py` — `coverage_fractions`
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
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — the v3.2 node-tree format (leaf / kind=lod / kind=partition)
