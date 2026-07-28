# Internal GSplatData Domain Mixins

Internal implementation package for `luxar.gsplats.gsplat_data.GSplatData` — splits the monolithic dataclass into focused domain-specific mixins.

## Purpose

`GSplatData` is the single public container for fitted Gaussian splats (`centers`, `amplitudes`, `cholesky_factors`, `colors`, `stats`). As LOD / filtering / culling / rendering capabilities grew, the class became large. This package decomposes it into domain mixins while preserving the single-constructor public API.

**Not a user-facing package**: All exports are internal. Users import `GSplatData` from `luxar.gsplats` and call its methods; they never touch `_data.*` directly.

## Module Map (Implementation Details)

- **`base.py`** — `_GSplatDataOps`: Shared mixin base declaring instance attributes (`colors`, `stats`, `_node`) plus STUBS for cross-mixin method calls (`filter`, `_map_substitutive`, etc.). Inherits from `_SplatArrayMixin` so mixins can call `self.volumes()` / `self.scale()`. The real implementations live on `GSplatData` or sibling mixins; stubs here let mypy resolve cross-mixin `self.` calls.

- **`metrics.py`** — `_SplatArrayMixin`: Shared computed properties for splat array containers. Inherited by BOTH `AdditiveSubLOD` and `GSplatData` (via `_GSplatDataOps`). Reads only `centers` / `amplitudes` / `cholesky_factors` plus `ndim` / `n_splats` / `truncation_radius`. Self-contained (no `GSplatData`-specific dependencies).
  - **Exports**: `n_splats`, `ndim`, `__len__`, `volumes()`, `masses()`, `marginal_sigmas()`, `scale(axes=None)`, `eccentricities(axes=None)`, `principal_radii(anisotropy=True)`, `nearest_neighbor_distances(spatial_axes=None, group_axes=None, k=1)`, `neighbor_counts(radius, spatial_axes=None, group_axes=None)`, internal helpers `_cholesky_diag_elements()`, `_nondegenerate_axes(eps=...)`, `_resolve_axes(axes)`, `_grouped_spatial(...)`, `_cell_size_hint(...)`
  - **Axis auto-detection**: `scale` / `eccentricities` default to `axes=None` → auto-detected non-degenerate (spatial) axes, so they are meaningful on nD timelapses (a zero-variance time axis is dropped; falls back to all axes if that would leave nothing).

- **`render.py`** — `RenderMixin`: `GSplatData.render_to_volume(shape, device=None, truncate=None, ...)` — GPU-accelerated volume rendering adapter. Delegates to `luxar.gsplats.rendering.volume_rendering.render_to_volume`.

- **`io_adapter.py`** — `IOAdapterMixin`: `GSplatData.save(path, ordering="hilbert", ...)` and `GSplatData.load(path)` — .gsplats.zarr I/O adapter. Delegates to `luxar.gsplats.io.save_gsplats` / `load_gsplats`. Sentinel `_USE_DEFAULT_COMPRESSOR` distinguishes "not specified" (→ Blosc) from explicit `compressor=None` (→ uncompressed zarr).

- **`filtering.py`** — `FilteringMixin`: `filter(mask)`, `filter_by(**criteria)`, `slice_by(...)`, and the threshold resolver `_resolve_threshold(val, normalized, dataset_values, percentile)`.
  - `filter(mask)`: Boolean array of shape `(N,)` → new `GSplatData` with only `mask[i]==True` splats. **Multi-substitutive warning**: Drops coarser substitutive levels (mask is sized to the default level); warns loudly and points at `filter_by` / `cull` (which preserve pyramids).
  - `filter_by`: Criteria-based filtering (bbox, volume_min/max, amplitude_min/max, scale_min/max, eccentricity_min/max, mass_min/max, sigma_axis/min/max, isolation_max, min_neighbors, ...). All optional; AND logic. Supports percentile / normalized / absolute thresholds. Preserves multi-substitutive pyramids via `_map_substitutive`.

- **`culling.py`** — `CullingMixin`: `cull(target=None, method="auto", ...)` — unified splat removal. Methods (ordered cheapest → most principled): `"cumulative"` (amplitude sum), `"amplitude_percentile"`, `"combined"` (amplitude+volume heuristic), `"redundancy"` (fractional contribution without target), `"error_budget"` (most principled; requires target volume). `"auto"` selects based on available inputs. Multi-substitutive datasets cull every level via `_map_substitutive` and rebuild the pyramid (preserves LOD structure).

- **`__init__.py`** — Empty docstring: "Internal domain mixins for `GSplatData`" (no exports; package is a private split).

## Ownership

All six domain mixins (`_SplatArrayMixin`, `_GSplatDataOps`, `RenderMixin`, `IOAdapterMixin`, `FilteringMixin`, `CullingMixin`) are inherited by `GSplatData` in `luxar.gsplats.gsplat_data`. The parent class composes them via multiple inheritance, so the mixins' `self.` calls resolve through `_GSplatDataOps`'s stubs.

**Mixin inheritance chain**:
```
GSplatData -> RenderMixin, IOAdapterMixin, FilteringMixin, CullingMixin
              ↓
              _GSplatDataOps -> _SplatArrayMixin
```

**Shared by `AdditiveSubLOD`**: `_SplatArrayMixin` only (the computed metrics that work on any splat array container).

## No Public API (Internal-Only)

Users never import from `_data.*` directly. The mixins expose their methods on the `GSplatData` public API:

```python
# ✅ Correct: import the public container
from luxar.gsplats import GSplatData

data = GSplatData.load("fitted.gsplats.zarr")
volume = data.render_to_volume(shape=(128, 128, 128))
culled = data.cull(method="cumulative", retention=0.95)
filtered = data.filter_by(amplitude_min=0.1, scale_max=90.0, scale_percentile=True)
data.save("output.gsplats.zarr", ordering="hilbert")

# ❌ Wrong: never import mixins directly
# from luxar.gsplats._data.render import RenderMixin  # NO
```

## Invariants & Gotchas

### Cross-Mixin Method Resolution

Mixins call each other's methods (e.g., `CullingMixin.cull` → `FilteringMixin.filter`). `_GSplatDataOps` declares all cross-mixin dependencies as stubs so type-checkers resolve `self.filter()` calls in `CullingMixin` without the real `FilteringMixin` being imported at stub-definition time. The real implementations live on `GSplatData` or the sibling mixins; stubs here only raise `NotImplementedError` (never executed at runtime because `GSplatData` overrides).

### Multi-Substitutive Preservation

- **`filter(mask)`**: Drops coarser substitutive levels (mask is `(N,)` → sized to the default level). Warns loudly: "Use `filter_by(...)` / `cull(...)` to preserve pyramids."
- **`filter_by` / `cull`**: Preserve multi-substitutive pyramids by applying the criteria PER LEVEL via `_map_substitutive` (each level is filtered independently; the pyramid structure is rebuilt).

### Axis Auto-Detection (`_SplatArrayMixin`)

`scale(axes=None)` and `eccentricities(axes=None)` default to the auto-detected **non-degenerate axes** (max marginal sigma > `SPATIAL_SIGMA_EPS` ≈ 1e-6). On a timelapse with a zero-variance time axis (built with `sigma=0` along the categorical/time dim), the time axis is dropped → scale / eccentricity become spatial-by-default. Falls back to all axes if that would leave nothing (e.g., all-degenerate / empty data).

**Practical impact**: `scale_max=90.0, scale_percentile=True` on a 4D timelapse filters by **spatial** size (the geometric mean of XYZ sigmas), ignoring the degenerate T axis — the expected behavior for "remove large diffuse background."

### Threshold Resolution (`_resolve_threshold`)

`filter_by` thresholds accept three modes:
- **Percentile** (`*_percentile=True`): `val` in `[0, 100]` → `np.percentile(dataset_values, val)`. Robust on heavy-tailed attributes (preferred over normalized). Example: `scale_max=90.0, scale_percentile=True` (the CLI equivalent is `--scale-max p90`).
- **Normalized** (`*_normalized=True`): `val` in `[0, 1]` → linear map onto `[min, max]` of `dataset_values`.
- **Absolute** (default): `val` is already in world units.

CLI accepts `"pNN"` / `"NN%"` string notation; Python `filter_by` uses explicit boolean flags.

### Isolation / Density Filtering

`filter_by` supports two density-based criteria (both auto-group by non-spatial axes so timepoints never count as neighbours):
- **`isolation_max`**: Remove spatially-isolated splats (nearest-neighbour distance; higher = more isolated = noise).
- **`min_neighbors` + `neighbor_radius`**: Remove splats with fewer than `min_neighbors` within `neighbor_radius`.

`isolation_max` relies on `_SplatArrayMixin.nearest_neighbor_distances` (k-th nearest-neighbour distance on spatial axes only); `min_neighbors` relies on `_SplatArrayMixin.neighbor_counts` (count within `neighbor_radius`).

### Compressor Sentinel (`io_adapter.py`)

`GSplatData.save(compressor=...)` distinguishes:
- **Not specified** (default) → `_USE_DEFAULT_COMPRESSOR` sentinel → Blosc(zstd) compression.
- **Explicit `compressor=None`** → Uncompressed zarr (e.g., for raw zarrita-readable cross-language fixtures).

A plain `None` default would conflate the two and make uncompressed output impossible (the bug that produced blosc-bitshuffle fixtures zarrita couldn't read).

## Tests

Tests live in the parent package's test suite (`packages/luxar/src/luxar/gsplats/tests/`):

- **`test_gsplat_data.py`**: The core data API — properties (`n_splats`, `ndim`, `__len__`, `repr`, `stats`), construction/validation, and the transforms (`translate`, `center_at_centroid`, `scale_intensity`, affine/normalize/clamp intensity, `transform` incl. covariance correctness).
- **`test_gsplat_data_aggregations.py`**: Computed metrics (`volumes`, `principal_radii`, `masses`, `marginal_sigmas`, `eccentricities`), filtering (`filter`, `filter_by`, `slice_by`), and reshape ops (`concatenate`, `embed_dimension`, `combine_as_new_dimension`).
- **`test_gsip_filters.py`**: Spatial-aware metrics (`scale`/`eccentricity` auto-ignoring a zero-variance time axis), `nearest_neighbor_distances` / `neighbor_counts`, the `isolation_max` filter, percentile thresholds, and the soft (amplitude-reweighting) high/low-pass.
- **`test_gsplat_data_io.py`**: Cull heuristics (`cumulative`, `amplitude_percentile`, `combined`, `auto`), the save whitelist (which `fitting_info` keys the `save()` whitelist includes vs excludes — asserted directly against the whitelist logic, no actual zarr save/load), and multi-dataset channel-color merge.
- **`test_culling.py`**: Contribution-based culling internals — per-splat deletion error, `cull_by_contribution` (`redundancy` / `error_budget`, quality preserved), nD support, the joint-compounding binary search, and `GSplatData.cull` integration.
- **`test_spatial_partition.py`**: `GSplatData.to_spatial_partition` — spatial BSP into a `kind=partition` tree (max-elements/split-rule, splat preservation, `bsp_tree` provenance, on-disk round-trip, scene grafting).
- **`test_spatial_axes.py`**: The shared spatial-axis auto-detection helpers (`spatial_axes_from_max_sigma`, `spatial_only_shift`).

Run via: `hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_gsplat_data.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_aggregations.py packages/luxar/src/luxar/gsplats/tests/test_gsip_filters.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_io.py packages/luxar/src/luxar/gsplats/tests/test_culling.py packages/luxar/src/luxar/gsplats/tests/test_spatial_partition.py packages/luxar/src/luxar/gsplats/tests/test_spatial_axes.py -v`

## See Also

- **`luxar.gsplats.gsplat_data`** — The public `GSplatData` container (composes these mixins)
- **`luxar.gsplats.io`** — The save/load implementation delegated by `IOAdapterMixin`
- **`luxar.gsplats.rendering`** — GPU-accelerated rendering delegated by `RenderMixin`
- **`luxar.gsplats.culling`** — Contribution-based culling algorithms used by `CullingMixin.cull`
- **Parent package** — `luxar.gsplats.README.md` (full public API surface)
