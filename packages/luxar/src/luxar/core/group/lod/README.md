# luxar.core.group.lod

Helpers for the **LOD-kind `Group`** — a scene-graph group that selects one of
N alternative children at runtime based on a view-driven metric (currently: the
projected bbox diagonal in pixels). The LOD group is geometry-agnostic: its
children can be `points`, `lines`, `gsplats`, or themselves a specialized group
(`kind=lod` / `kind=partition`).

This package splits cleanly into two layers:

- **Geometry-agnostic machinery** (`group.py`) — threshold derivation, the
  `min_pixel_size` monotonicity invariant, the `kind=lod` validator, and the
  shared display-type resolver. Shared by every leaf geometry and by the
  Partition kind.
- **Per-geometry axis resolvers** (`points.py`, `lines.py`, `gsplats.py`) —
  one peer per leaf type, interpreting the `additive_lod=` (and, for gsplats,
  `lod_group=`) convenience kwargs that `add_points` / `add_lines` /
  `add_gsplats_from_data` accept.

The two sampler modules (`spatial_uniform.py`, `poisson_disk.py`) are pure-NumPy
ordering primitives shared by the Points and Lines resolvers.

## File structure

```
lod/
├── group.py            # Geometry-agnostic LOD-group machinery (thresholds, validator, display-type)
├── points.py           # Points additive-LOD ordering + ladder construction + resolver
├── lines.py            # Lines additive-LOD (per-polyline) ordering + ladder + resolver
├── gsplats.py           # GSplats substitutive + additive axis resolvers
├── spatial_uniform.py  # Stratified-grid sampler (default spatial-uniform ordering)
└── poisson_disk.py     # Bridson blue-noise sampler (opt-in alternative)
```

## Two LOD axes

Luxar's gsplat LOD model has two orthogonal axes; the helpers here build both:

- **Substitutive** — coarser levels *replace* finer ones (fewer, larger
  elements). A multi-substitutive dataset becomes a `kind=lod` `Group` whose
  children are the levels in coarsest→finest order. Only gsplats carry a stored
  substitutive pyramid; `gsplats.py::resolve_substitutive_axis_gsplats` resolves
  the `lod_group=` kwarg against it.
- **Additive** — finer levels *add to* coarser ones (a streaming prefix of
  elements). All three geometries support an additive ladder via the
  `additive_lod=` kwarg. For gsplats it is applied per substitutive level
  (`resolve_additive_axis_gsplats`).

See `docs/specs/GSPLATS_ZARR_FORMAT.md` for the 2-D substitutive × additive LOD
matrix the gsplat format stores.

## Geometry-agnostic machinery (`group.py`)

Each LOD-group child carries a `min_pixel_size` attribute, strictly monotonic
increasing in coarsest→finest order (coarsest = `0.0`). The viewer picks the
finest child whose threshold is satisfied by the current view.

| Symbol | Purpose |
|--------|---------|
| `BASE_PIXEL_SIZE = 10.0` | Anchor for auto-derived thresholds; the detail floor below which a finer level isn't worth the cost. |
| `derive_min_pixel_sizes(element_counts, base_pixel_size=None)` | Auto-derive thresholds: child *i* → `base * sqrt(n_i / n_0)`, coarsest = `0.0`. Defensively bumps near-equal levels (×1.1) to keep the list strictly ascending. |
| `validate_lod_group(group)` | Free-function validator for any `Group` with `attrs["kind"] == "lod"`. Raises on no children, out-of-range `default_level`, missing `min_pixel_size`, or non-monotonic thresholds. |
| `resolve_display_type(node)` | The geometry type a node appears as to the user. For `kind in (lod, partition)` returns the recorded `display_type`; else the node's own `type`. Shared with the Partition kind's validator. |
| `compute_lod_display_type(children)` | Derive an LOD group's `display_type` from its finest (last) child, recursing through nested specialized groups. |
| `_assert_strict_ascending(thresholds, source)` | The shared monotonicity guard, applied by both the explicit-`min_pixel_sizes` resolver paths and `derive_min_pixel_sizes`. |

**Heuristic caveat.** Element *count* is only a proxy for screen *coverage* —
a level with 4× the elements does not necessarily resolve 2× the linear detail.
The proxy is weakest for substitutive levels (fewer, larger elements), where a
count-driven threshold can switch a touch early. Override the anchor via
`base_pixel_size` when a particular ladder switches at the wrong zoom.

## Per-geometry resolvers

All three resolvers share a `None | bool | dict` value vocabulary for the
convenience kwargs, but the semantics differ per geometry.

### Points (`points.py`)

`resolve_additive_axis_points(spec)` normalizes the `additive_lod=` kwarg into a
dict (`method` / `n_lods` / `counts` / `seed` / `salience_kind`) or `None`.
The compiler (`_write_points_multi_lod`) then calls
`make_additive_lod_points(...)`, which returns per-LOD-level index arrays.

Ordering methods (`PointsMethodName`):

- `random` — uniform-random permutation (optional `seed`).
- `salience` — sort by radius descending (`salience_kind='size'`, default) or by
  `luminance × radius³` (`salience_kind='energy'`).
- `spatial-uniform` — stratified-grid sampling (see `spatial_uniform.py`).
- `poisson-disk` — Bridson blue-noise sampling.

Breakpoint vocabulary for `counts` / `breakpoints`:

- `None` → equal-count split into `n_lods` levels (`DEFAULT_N_LODS = 4`).
- `List[int]` → cumulative element-count breakpoints, e.g. `[1500, 8000, 40000]`.
- `"energy:0.5,0.9,0.99,1.0"` → cumulative perceptual-energy fractions; element
  energy is `luminance_i × radius_i³`. Requires `colors` or `scalars` for the
  luminance term.

`DEFAULT_METHOD` is `random`; `DEFAULT_N_LODS` is `4`.

### Lines (`lines.py`)

Mirrors Points in shape but operates **per-polyline**: each LOD level carries
whole polylines (vertices + their segments) so segment topology stays valid
during partial loads.

- `identify_polylines(n_vertices, line_type, indices=None)` splits vertices into
  per-polyline index arrays. `segments` → N/2 length-2 polylines; `indexed` →
  Union-Find connected components; `polyline` / `loop` → one polyline spanning
  all vertices (a multi-LOD ladder is then a no-op — a warning is logged and a
  single level emitted).
- `compute_additive_order_lines(...)` orders polylines (not vertices); for
  `spatial-uniform` / `poisson-disk` the representative point is each polyline's
  bbox center.
- `make_additive_lod_lines(...)` returns per-LOD-level lists of per-polyline
  index arrays for `_write_lines_multi_lod` to gather and rewrite with
  subgroup-local segment indices.
- `salience_kind='energy'` uses the tube-volume score
  `mean_luminance × Σ(seg_length × width²)`.
- `resolve_additive_axis_lines(spec)` is the `add_lines(..., additive_lod=...)`
  resolver.

### GSplats (`gsplats.py`)

GSplats are the only geometry with a stored substitutive pyramid, so it has two
resolvers:

- `resolve_substitutive_axis_gsplats(data, spec)` — the `lod_group=` axis.
  Returns `(resolved_data, explicit_min_pixel_sizes_or_None,
  base_pixel_size_or_None)`. `None` auto-keeps a multi-substitutive pyramid
  (routes to the `kind=lod` builder, no work discarded); `False` collapses to
  the finest level (index 0); `True` requires a stored pyramid; `dict(...)`
  reuses a stored pyramid or computes one via
  `gsplats.lod.substitutive.make_substitutive_lod` (canonical default
  `levels=3`), with `recompute=True` forcing recomputation.
- `resolve_additive_axis_gsplats(data, spec)` — the `additive_lod=` axis,
  applied independently per substitutive level. `dict(...)` computes a ladder on
  any level missing one via `gsplats.lod.additive.make_additive_lod` (default
  `n_lods=4`); `False` flattens each level to a single additive sub-LOD.

Convention throughout: the **finest** substitutive level is index 0; LOD-group
children are stored coarsest→finest (finest last).

## Samplers

### `spatial_uniform.py` — `stratified_grid_order(positions, n_lods)`

Assigns elements to LOD levels by which grid resolution first "covers" their
position. Level `i` uses a `2^(i+1)`-per-axis grid (LOD 0 = 2×2×2, LOD 1 =
4×4×4, …); the first unassigned element in each newly-occupied cell joins that
level. Any cumulative prefix of the resulting permutation has approximately
uniform spatial density. Returns `(permutation, per_level_counts)`. Pure NumPy,
`O(N · n_lods)`. Uses only the first 3 spatial dims; the exponent is clamped at
31 to avoid int64 overflow.

### `poisson_disk.py` — `poisson_disk_order(positions, n_lods, seed=0)`

Opt-in blue-noise alternative (Bridson, SIGGRAPH 2007). Runs progressively
finer radii (`r_i = (diag/2) · 0.5^i`); each level keeps the points its radius
selects that no coarser level already took. A cell grid sized at `r/√3` keeps
the rejection test to a local 5×5×5 neighborhood (`O(N)` expected per level).
Same `(permutation, per_level_counts)` contract as the stratified sampler, so
`make_additive_lod_*` stays symmetric across methods; the last level absorbs any
points the finest pass rejected.

## Usage

```python
# Points: 4-level energy-weighted additive ladder
scene.add_points(
    "cloud", positions, colors=colors, radii=radii,
    additive_lod=dict(method="salience", salience_kind="energy", n_lods=4),
)

# Points: explicit cumulative-count breakpoints, spatial-uniform ordering
scene.add_points(
    "cloud", positions,
    additive_lod=dict(method="spatial-uniform", counts=[1500, 8000, 40000]),
)

# GSplats: keep the stored substitutive pyramid as a kind=lod Group,
# and add a 4-level additive ladder per level
scene.add_gsplats_from_data(
    "splats", gsplat_data,
    lod_group=True,            # require/use stored substitutive levels
    additive_lod=dict(n_lods=4),
)
```

## See Also

- [../../README.md](../../README.md) — `luxar.core` overview (Group, Node, Points, Lines, GSplats)
- `../partition.py` — the sibling `kind=partition` group, which shares
  `resolve_display_type` from `group.py`
- `luxar.gsplats.lod` — `make_substitutive_lod` / `make_additive_lod` builders
  the gsplats resolvers delegate to
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — the v2.0 substitutive × additive LOD matrix
