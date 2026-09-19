# Internal GSplatData Domain Mixins

Internal implementation package for `luxar.gsplats.gsplat_data.GSplatData` — splits the monolithic dataclass into focused domain-specific mixins.

## Purpose

`GSplatData` is the single public container for fitted Gaussian splats (`centers`, `amplitudes`, `cholesky_factors`, `colors`, `stats`). As LOD / filtering / culling / rendering capabilities grew, the class became large. This package decomposes it into domain mixins while preserving the single-constructor public API.

**Not a user-facing package**: All exports are internal. Users import `GSplatData` from `luxar.gsplats` and call its methods; they never touch `_data.*` directly.

## Module Map (Implementation Details)

- **`base.py`** — `_GSplatDataOps`: Shared mixin base declaring instance attributes (`colors`, `stats`, `_node`) plus STUBS for cross-mixin method calls (`filter`, `_map_substitutive`, etc.). Inherits from `_SplatArrayMixin` so mixins can call `self.volumes()` / `self.scale()`. The real implementations live on `GSplatData` or sibling mixins; stubs here let mypy resolve cross-mixin `self.` calls. Also home to the shared module-level helpers: `_readonly` / `_readonly_opt` / `_readonly_sublod` (zero-copy non-writable views backing the immutability contract), `_merge_lod_colors` (the None/all/mixed color merge, dtype + RGB→RGBA layout policy) and `_concat_additive_levels` (ladder-wide merge of several views' additive ladders).

- **`metrics.py`** — `_SplatArrayMixin`: Shared computed properties for splat array containers. Inherited by BOTH `AdditiveSubLOD` and `GSplatData` (via `_GSplatDataOps`). Reads only `centers` / `amplitudes` / `cholesky_factors` plus `ndim` / `n_splats` / `truncation_radius`. Self-contained (no `GSplatData`-specific dependencies).
  - **Exports**: `n_splats`, `ndim`, `__len__`, `volumes()`, `masses()`, `marginal_sigmas()`, `scale(axes=None)`, `eccentricities(axes=None)`, `principal_radii(anisotropy=True)`, `nearest_neighbor_distances(spatial_axes=None, group_axes=None, k=1)`, `neighbor_counts(radius, spatial_axes=None, group_axes=None)`, internal helpers `_cholesky_diag_elements()`, `_nondegenerate_axes(eps=...)`, `_resolve_axes(axes)`, `_grouped_spatial(...)`, `_cell_size_hint(...)`
  - **Axis auto-detection**: `scale` / `eccentricities` default to `axes=None` → auto-detected non-degenerate (spatial) axes, so they are meaningful on nD timelapses (a zero-variance time axis is dropped; falls back to all axes if that would leave nothing).

- **`render.py`** — `RenderMixin`: `GSplatData.render_to_volume(shape, device=None, truncate=None, ...)` — GPU-accelerated volume rendering adapter. Delegates to `luxar.gsplats.rendering.volume_rendering.render_to_volume`.

- **`io_adapter.py`** — `IOAdapterMixin`: `GSplatData.save(path, ordering="hilbert", ...)` and `GSplatData.load(path)` — .gsplats.zarr I/O adapter. `save()` delegates to `write_gsplats_tree` (in `luxar.gsplats.io.save_gsplats`) and `load()` to `load_gsplats` (in `luxar.gsplats.io.load_gsplats`). Sentinel `_USE_DEFAULT_COMPRESSOR` distinguishes "not specified" (→ Blosc) from explicit `compressor=None` (→ uncompressed zarr).

- **`filtering.py`** — `FilteringMixin`: `filter(mask)`, `filter_by(**criteria)`, `slice_by(...)`, and the threshold resolver `_resolve_threshold(val, normalized, dataset_values, percentile)`.
  - `filter(mask)`: Boolean array of shape `(N,)` → new `GSplatData` with only `mask[i]==True` splats. **Multi-substitutive warning**: Drops coarser substitutive levels (mask is sized to the default level); warns loudly and points at `filter_by` / `cull` (which preserve pyramids).
  - `filter_by`: Criteria-based filtering (bbox, volume_min/max, amplitude_min/max, scale_min/max, eccentricity_min/max, mass_min/max, sigma_axis/min/max, isolation_max, min_neighbors, ...). All optional; AND logic. Supports percentile / normalized / absolute thresholds. Preserves multi-substitutive pyramids via `_map_substitutive`.
  - **Inherited-`stats` hygiene lives here, in three categories with three predicates** — `_REGION_SCOPED_STATS_KEYS` (what the splats REPRESENT; dropped only by a bbox/slice crop that actually excluded splats — `_is_crop`), `_CONTENT_SCOPED_STATS_KEYS` + `_CONTENT_SCOPED_OP_RECORD_KEYS` (scores MEASURED against the source volume, and the record of the reduction that produced the artifact — `culled` / `culling_method` / `n_original` / `n_culled` / `amplitude_retention`; both dropped by any change to which splats the artifact holds, spatial or not — `_stats_after_content_change`), and `_STRUCTURE_SCOPED_STATS_KEYS` (the artifact's OWN TOPOLOGY — `lod_kind` / `recipe` / `n_substitutive_levels` / `compression_factor` / the `lod_*` ladder summary / the `batch-fit merge` per-part knobs; dropped only when a rewrite changes the STRUCTURE KIND — `stats_after_structure_change`, applied for `flatten` / `partition` / `lod` / `decimate`, the four rewrites whose output CAN be a different kind of thing than their input — the test for a new command is that question, not whether a given run happens to preserve the kind; `flatten`, `partition` and `lod` apply it in the CLI command because the domain methods under them (`flattened()` / `concatenate` / `to_spatial_partition` / the recipe builders) have other callers the record is still true for, while `lod/decimate.py` applies it itself — its return type is one flat leaf whatever it was handed, so the public `luxar.gsplats.lod.decimate` API must not depend on a command to scrub, and only its `target >= n_splats` early return keeps the record, correctly, by handing the input straight back. `lod` scrubs ONCE on the loaded input, which covers both of its write paths (the matrix builders derive `result.stats` from `dict(src.stats)` and then stamp their own record over it; the composed path hands the same dict to `split_fitting_info`), so each recipe publishes only what its own builder stamped: `recipe` alone for `flat` / `tiles` / `overview` / `adaptive`, plus the ladder summary for `stream`, plus the substitutive block for `levels`). The structure rule is a deny-list of key names, never "drop the `pipeline/` group": the normalization block shares that group, and `_STRUCTURE_SCOPE_EXEMPT_KEYS` holds `coarsen_dims`, which `write_gsplats_tree` READS BACK to derive the chunk-ordering barrier (scrubbing it would silently change the output's layout) — exempt from the scrub is not exempt from being TRUE, so a rewrite that coarsens over its own choice of axes re-stamps it instead (`decimate`'s `merge` family does, always as an EXPLICIT dim list — a written null reads to the writer exactly like an absent key, i.e. no provenance → auto-detect, which re-imposes a barrier on the axis just blended whenever the reduction leaves that axis' grid intact and is merely redundant when it does not — measured both ways, so the explicit list is the honest spelling either way; `make_substitutive_lod` and the `batch-fit merge` per-part record resolve the same stamp through the same shared `resolved_merge_coarsen_dims`, so none of the three paths that WRITE this key publishes coarsen-everything as a `null` any more — while `lod --recipe adaptive` / `overview` and `fit --recipe levels` coarsen but publish no stamp at all, which reads the same way and is still open on #1600. The exemption's consequence is that the explicit list is INHERITED: every structure-preserving rewrite of a `levels` store (and `flatten` / `partition` / `additive` / a rebuilt `lod`) now writes `slice_dims: []` where the inherited `null` used to fall back to an auto-detected barrier — safe in direction (a missing barrier over-fetches; a false one can drop splats) and true of those outputs (none of them coarsens anything), at the cost of the finest level, which is the input unreduced and loses a barrier that would have been legitimate. Its `prefix` family blends no axis and keeps the inherited value). The content half reaches both scopes a measured score can live in: the top-level dict and each sub-LOD's (a progressive fit's per-pass `cumulative_psnr_db` / `delta_psnr_db`, persisted as the leaf's `lod_stats`). Every op that stamps a reduction record does so AFTER its `filter()` — the two multi-substitutive branches build their own top-level dict and therefore scrub BEFORE stamping. `filter()` is the chokepoint the count-based half is wired at, so `filter_by` / `slice_by` / every `cull` strategy inherit it; the amplitude-edit half is wired at `intensity.py::_with_new_amplitudes`; `lod_views.py`'s `additive_prefix` / `at_substitutive` drop the INHERITED top-level scores because a reduced VIEW is a reduction (this is what kept `lod --recipe overview`'s merged coarse cap from publishing the input fit's PSNR); and `lod/decimate.py` scrubs explicitly (measured scores because a merge lands on the requested count with every splat replaced, and the topology record because the result is always one flat leaf). Artifact-local energy/count ladder stamps are recomputed after reductions against the rewritten finest level; expensive measured `quality` is dropped and can be restored with `annotate-quality --with-quality`, while source-volume `refine_stats` requires rebuilding with `--refine`. Plain accessors preserve all authored stamps. `drop_content_scoped_stats(stats)` is the dict-level primitive for a caller holding raw `stats` (the CLI's node-tree writer path) and is re-exported publicly from `luxar.gsplats.gsplat_data` alongside `amplitudes_changed`, so `luxar.cli` never imports this private package; `scrub_measured_stats(data)` is the dataset-level one; `measured_stats_snapshot` / `restore_measured_stats` are the deliberate producer-side exemption for the fitters' own closing `cull_retention` trim. `stats_after_structure_change(stats)` is re-exported the same way and returns a scrubbed COPY (its call sites still own their dict). No category subsumes another — see the module docstring and `docs/specs/GSPLATS_ZARR_FORMAT.md`.

  The producer-side closing-trim exemption restores expensive measured scores,
  not `_FINAL_SCALE_POPULATION_STATS_KEYS`: those cheap counts no longer
  describe the artifact when the trim changes its splat set.

  The public region helpers are `scrub_region_scoped_stats(data)` for removing
  inherited region claims and `stamp_region_scoped_stats(stats, ...)` for
  replacing a dict's source-grid record from measurements supplied by its caller.

- **`culling.py`** — `CullingMixin`: `cull(target=None, method="auto", ...)` — unified splat removal. Methods (ordered cheapest → most principled): `"cumulative"` (amplitude sum), `"amplitude_percentile"`, `"combined"` (amplitude+volume heuristic), `"redundancy"` (fractional contribution without target), `"error_budget"` (most principled; requires target volume). `"auto"` selects based on available inputs. Multi-substitutive datasets cull every level via `_map_substitutive` and rebuild the pyramid (preserves LOD structure).

- **`lod_views.py`** — `LODViewsMixin`: The LOD tree structure/accessor family. The derived finest-first matrix views over the ground-truth node (`substitutive_levels`, `additive_sublods`, `n_substitutive`, `n_additive_sublods`, `default_substitutive`, `_finest_leaf()`), per-level access (`additive_sublod(level)`, `at_substitutive(level)`, `_view_of_level(...)`), ladder reshaping (`additive_prefix(level)`, `flattened()`, `lod_psnrs()`), the matrix constructors (`from_additive_sublods`, `from_substitutive_levels`) and the node-tree bridge (`tree`, `from_tree`).

- **`composition.py`** — `CompositionMixin`: Multi-dataset composition. `from_default_selection(node, stats=...)` preserves matrix-shaped trees and materializes every partition part at each LOD group's default child; `concatenate(datasets)` merges per `(substitutive, additive)` cell so pyramids stay pyramids; `combine_as_new_dimension(datasets, values, sigma)`, `merge_with_channel_colors(...)`, `embed_dimension(values, sigma)` (D → D+1 promotion), plus the `kind=partition` builders `to_spatial_partition(max_elements=..., rule=...)` and `partition_from_regions(regions, recipe=...)`.

- **`transforms.py`** — `TransformsMixin`: Geometric transforms — `transform(matrix)` (centers + covariance, diagonal fast path), `translate(offset)`, `center_at_centroid()` (amplitude-weighted, spatial axes only) — and the two structure-preserving rebuild helpers `_map_substitutive(fn)` / `_map_additive(fn)` that every per-level op in the sibling mixins goes through.

- **`intensity.py`** — `IntensityMixin`: Amplitude / color / label-channel edits, ladder-preserving: `affine_intensity`, `normalize_intensity`, `clamp_intensity`, `scale_intensity`, `reweight_amplitude(multiplier)` (per-splat), `soft_scale_filter(highpass=..., lowpass=..., width=...)` (smooth log2-scale band reweighting), `with_colors(colors)`, the categorical-channel pair `with_label_ids(label_ids, label_vocabulary)` / `without_label_ids()` (attach and strip the exact per-splat class channel; substitutive LOD and explicit merge decimation treat attached ids as merge barriers, while automatic decimation stays prefix-based and INRIA export still points callers to `without_label_ids()`; `with_label_ids` refuses an already-built multi-substitutive pyramid because one finest-level array cannot define labels for existing coarse rows) and the shared `_with_new_amplitudes(...)`.

- **`__init__.py`** — Empty docstring: "Internal domain mixins for `GSplatData`" (no exports; package is a private split).

## Ownership

All ten domain mixins are inherited by `GSplatData` in `luxar.gsplats.gsplat_data`: eight (`RenderMixin`, `IOAdapterMixin`, `FilteringMixin`, `CullingMixin`, `LODViewsMixin`, `CompositionMixin`, `TransformsMixin`, `IntensityMixin`) as direct bases and the remaining two (`_GSplatDataOps`, `_SplatArrayMixin`) transitively, as the chain below shows. No two of the eight *direct* mixins define the same name, so their order in the bases list is free of MRO surprises. `_GSplatDataOps` carries thirteen raising stubs — that is how a mixin's `self.` calls type-check against members it does not own — twelve shadowing a name a sibling mixin implements and one (`truncation_radius`) implemented by `GSplatData` itself. Those stubs are safe because `_GSplatDataOps` is the common base of *all eight*, so C3 can only place it once every base that inherits it has been linearized — i.e. dead last. Keep it that way. `_GSplatDataOps` jumps ahead of a mixin only when that mixin both (a) stops inheriting it and (b) sits after the last base that still does — in practice, is the final base. If that mixin is also one that implements a stubbed name (`LODViewsMixin`, `FilteringMixin`, `TransformsMixin`), the stub wins and e.g. `flattened()` starts raising `NotImplementedError`. A mixin that drops `_GSplatDataOps` but still has an inheriting base listed after it keeps its MRO position and is harmless. `GSplatData` itself keeps only construction (`__init__`), `truncation_radius` and `__repr__`.

**Mixin inheritance chain**:
```
GSplatData -> RenderMixin, IOAdapterMixin, FilteringMixin, CullingMixin,
              LODViewsMixin, CompositionMixin, TransformsMixin, IntensityMixin
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

- **`test_gsplat_data.py`**: The core data API — properties (`n_splats`, `ndim`, `__len__`, `repr`, `stats`), construction/validation, the transforms (`translate`, `center_at_centroid`, `scale_intensity`, affine/normalize/clamp intensity, `transform` incl. covariance correctness), and the mixin-composition guard that fails if any `_GSplatDataOps` stub wins the MRO (see Ownership above).
- **`test_gsplat_data_lod.py`**: The LOD matrix API (`LODViewsMixin`) — additive ladders and substitutive levels, `additive_prefix` / `flattened` / `at_substitutive` read-only views, the per-level constructors, and the `_merge_lod_colors` dtype / RGB→RGBA layout policy.
- **`test_gsplat_data_tree_bridge.py`**: The node-tree bridge (`tree` / `from_tree`) round-trips, including the `_readonly_sublod` stats-aliasing regression.
- **`test_gsplat_data_aggregations.py`**: Computed metrics (`volumes`, `principal_radii`, `masses`, `marginal_sigmas`, `eccentricities`), filtering (`filter`, `filter_by`, `slice_by`), and reshape ops (`concatenate`, `embed_dimension`, `combine_as_new_dimension`).
- **`test_gsip_filters.py`**: Spatial-aware metrics (`scale`/`eccentricity` auto-ignoring a zero-variance time axis), `nearest_neighbor_distances` / `neighbor_counts`, the `isolation_max` filter, percentile thresholds, and the soft (amplitude-reweighting) high/low-pass.
- **`test_gsplat_data_io.py`**: Cull heuristics (`cumulative`, `amplitude_percentile`, `combined`, `auto`), the save whitelist (which `fitting_info` keys the `save()` whitelist includes vs excludes — asserted directly against the whitelist logic, no actual zarr save/load), and multi-dataset channel-color merge.
- **`test_culling.py`**: Contribution-based culling internals — per-splat deletion error, `cull_by_contribution` (`redundancy` / `error_budget`, quality preserved), nD support, the joint-compounding binary search, and `GSplatData.cull` integration.
- **`test_spatial_partition.py`**: `GSplatData.to_spatial_partition` — spatial BSP into a `kind=partition` tree (max-elements/split-rule, splat preservation, `bsp_tree` provenance, on-disk round-trip, scene grafting).
- **`test_spatial_axes.py`**: The shared spatial-axis auto-detection helpers (`spatial_axes_from_max_sigma`, `spatial_only_shift`).
- **`test_map_helpers.py`**: `_map_substitutive` / `_map_additive` (`TransformsMixin`) directly — the per-level rebuild pivot every ladder-preserving op in the sibling mixins goes through.
- **`test_filter_per_level.py`**: Per-level rebuild for `FilteringMixin.filter_by` / `slice_by` and `CullingMixin.cull` — reducing a laddered dataset keeps every substitutive level present and consistent, per-level metadata included — plus the complementary raw-`filter(mask)` contract: it collapses to the default level and warns loudly, and must *not* warn on a single-level dataset.
- **`test_content_scoped_metrics.py`**: The two stats categories — a table, one row per operation, stating whether it changes content, and asserting that the MEASURED scores (`psnr_db`, `ssim`, the `foreground_*` trio, the `final_*` residuals, the error-budget bound, the per-sub-LOD ladder PSNRs) and the reduction record go with the splat set while descriptive counters, the region stamps and the Q·e ladder stamps follow their own rule. The table is closed against the real `GSplatData` surface (a new public method returning a `GSplatData` must be classified), and every row also asserts the INPUT is untouched. Includes the deliberate producer-side exemption for the fitters' closing `cull_retention` trim, at the helpers and at a real progressive fit.

Run via: `hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_gsplat_data.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_lod.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_tree_bridge.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_aggregations.py packages/luxar/src/luxar/gsplats/tests/test_gsip_filters.py packages/luxar/src/luxar/gsplats/tests/test_gsplat_data_io.py packages/luxar/src/luxar/gsplats/tests/test_culling.py packages/luxar/src/luxar/gsplats/tests/test_spatial_partition.py packages/luxar/src/luxar/gsplats/tests/test_spatial_axes.py packages/luxar/src/luxar/gsplats/tests/test_map_helpers.py packages/luxar/src/luxar/gsplats/tests/test_filter_per_level.py packages/luxar/src/luxar/gsplats/tests/test_content_scoped_metrics.py -v`

## See Also

- **`luxar.gsplats.gsplat_data`** — The public `GSplatData` container (composes these mixins)
- **`luxar.gsplats.io`** — The save/load implementation delegated by `IOAdapterMixin`
- **`luxar.gsplats.rendering`** — GPU-accelerated rendering delegated by `RenderMixin`
- **`luxar.gsplats.culling`** — Contribution-based culling algorithms used by `CullingMixin.cull`
- **Parent package** — `luxar.gsplats.README.md` (full public API surface)
