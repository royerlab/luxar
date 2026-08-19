# `luxar.gsplats.planner` — content-aware fit planner

Decides **how to decompose a volume into fit regions and how many splats each
region gets**, driven by a cheap content scan and the calibration's transferable
splats-per-feature density (`SplatDensity`). Noise-agnostic — the calibration
(`luxar.gsplats.calibration`) owns the K-selection / noise axis; this package
owns the tiling / scale / budget axis.

This is the implementation behind **`gsplat fit --tiling content`** (and its
cluster sibling **`gsplat batch-fit submit --tiling content`**). It was promoted
from the former standalone `gsplat plan` command.

## Pipeline

```
scan_content(volume)             -> ContentField   # coarse per-cell feature density
plan_volume(field/volume, density) -> FitPlan       # BSP boxes + per-box K budgets
fit_planned(volume, plan)        -> GSplatData|GSplatNode   # fit each box, merge
```

## Key classes / functions

- **`scan_content(volume, ...)` -> `ContentField`** (`content_scan.py`) — a coarse
  feature-density grid (peaks / edges / intensity) used to size boxes.
- **`plan_partition` / `plan_volume`** (`bsp_boxes.py`) — recursive BSP into
  variable-size boxes, each budgeted via `density.predict_k(n_features)`; packs
  more splats where the volume is busy. Returns a `FitPlan`.
- **`FitPlan` / `PlanBox`** (`spec.py`) — the serialisable plan (`to_json` /
  `from_json`): boxes (bounds + budget), `overlap`, `feature_method`, the
  `density` used, `min_leaf` / `max_leaf`. `--plan-box K` fits box `K` of a plan.
- **`fit_planned`** (`fit_planned.py`) — sequential driver: fit each budgeted box
  on a halo-padded crop, keep the core, merge. `partition=True` (default) returns
  a `kind=partition` `GSplatNode` (one part per box); `recipe=`/`recipe_params=`
  give each part its own per-part LOD (`fit --recipe`). `partition=False` (`--flat`)
  concatenates into one flat leaf. Each box is carried through the merge as the
  `GSplatData` it was fitted as (`_fit_one_box` returns the dataset, not bare
  arrays), so the fit's `truncation_radius` — a `truncate:` in the config — survives;
  rebuilding from arrays reset the radius to the default and a content result then
  refused to `GSplatData.concatenate` with a uniform-tiled one fitted the same way
  (#1637). Its per-box stats survive too, on a **bare-leaf** partition: with
  `recipe=` each part's ladder builder writes its own per-sub-LOD stats, and on the
  flat path `concatenate` replaces them with its merge summary. Two of them are
  re-scoped to the part: the whole region-scoped stamp set a `--bbox` crop drops
  (`_REGION_SCOPED_STATS_KEYS` — `source_shape` / `source_voxels` / `source_bytes`
  / `fitted_shape` / `fitted_voxels` / `occupancy` / `voxels_per_splat` /
  `source_declared`) goes whenever the padded crop is larger than the core box or
  the core mask removed splats, and `n_splats` is restamped to the kept count. The
  rest (`psnr_db` / `ssim` / `mse`, the `final_*` losses, the cull provenance
  `n_original` / `n_culled` / `amplitude_retention`) still describes the box's own
  fit — the padded crop, with the pre-mask splat set — which is the same fit
  provenance a uniform tile-part carries (a tile-part keeps its grid stamps too,
  because a tile keeps every splat it fitted; a core-masked box does not).
  Non-finite values are dropped as well: the leaf writer stamps `lod_stats` raw, so
  a signal-free box's `psnr_db = inf` would reach a part's attrs as a bare
  `Infinity` token that a strict JSON parser refuses.
- **`fit_planned_parallel`** (`fit_planned_parallel.py`) — the `-j N` path: fit
  each box in its own subprocess (`fit --plan-box`), then merge identically. A box
  that fits 0 splats writes a sibling `<output>.empty` marker (skipped at merge).
  `_default_worker_cmd_builder` forwards the run's fit configuration (`--preset`,
  `--config`, `--iters`, `--loss`, `--lr`, `--cull-retention`) **verbatim** — an
  absent flag stays absent — so a box worker resolves the same fit config as the
  sequential path (`--seeds` excepted: a content box's budget comes from the plan).
  `truncate:` is settable only through a YAML `--config` (no preset sets it, and
  there is no `--truncate` flag), so without the forwarding `-j N` silently fitted
  at the 2.75 default; and substituting `standard` for an absent `--preset` made a
  box resolve 5000 iterations where `-j 1` resolves 1000 (#1637). An absent
  `--cull-retention` does NOT mean the fitter's 0.95: the worker re-enters the same
  CLI resolution and lands on `CONTENT_CULL_RETENTION` itself, so `-j N` and `-j 1`
  cull identically.
- **`CONTENT_CULL_RETENTION = 0.999`** (`fit_planned.py`) — the near-lossless
  post-fit retention every content box is fitted at, instead of the fitter's own
  0.95 (whose bottom-5% cull would compound across the re-merged boxes).
  `fit_planned` defaults to it directly; `fit_planned_parallel` has no fit kwargs to
  default, so its boxes reach the same value through the CLI content path described
  above, which imports this same constant as its per-command default — the CLI and
  library halves cannot drift (#1729).

Both drivers expect the background floor to arrive as a **concrete level** (or
`"none"`): the CLI resolves `--floor` once against the whole volume and hands the
same number to every box and to the density scan. A spec (`auto`/`pNN`) forwarded
into the per-box fit would be re-estimated against each box CROP, so abutting
core-kept boxes would subtract wildly different pedestals and normalize by different
ranges — visible brightness steps at box boundaries. What is shared is the floor
ARGUMENT, not the input: every box is still handed its own crop, and the
normalization floor inside a box is still clamped up to that crop's own minimum
(`image_min = max(level, min(crop))`), so a box lying entirely above the pedestal
subtracts its own minimum.

## Consumers

- `cli/gsplat_ops/fitting/fit.py::run_fit_volume` (`fit --tiling content`) and
  `cli/gsplat_ops/planner.py::run_content_fit`.
- `cli/gsplat_ops/batch/submit.py::run_batch_submit` (`batch-fit submit --tiling content`)
  builds one shared `FitPlan` and fans its boxes across a Slurm array.
