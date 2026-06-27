# `luxar.gsplats.planner` — content-aware fit planner

Decides **how to decompose a volume into fit regions and how many splats each
region gets**, driven by a cheap content scan and the calibration's transferable
splats-per-feature density (`SplatDensity`). Noise-agnostic — the calibration
(`luxar.gsplats.calibration`) owns the K-selection / noise axis; this package
owns the tiling / scale / budget axis.

This is the implementation behind **`gsplat fit --tiling content`** (and its
cluster sibling **`gsplat slurm-fit submit --tiling content`**). It was promoted
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
  concatenates into one flat leaf.
- **`fit_planned_parallel`** (`fit_planned_parallel.py`) — the `-j N` path: fit
  each box in its own subprocess (`fit --plan-box`), then merge identically. A box
  that fits 0 splats writes a sibling `<output>.empty` marker (skipped at merge).

## Consumers

- `cli/gsplat_ops/fitting.py::fit_volume` (`fit --tiling content`) and
  `cli/gsplat_ops/planner.py::run_content_fit`.
- `cli/gsplat_ops/batch.py::batch_submit` (`slurm-fit submit --tiling content`)
  builds one shared `FitPlan` and fans its boxes across a Slurm array.
