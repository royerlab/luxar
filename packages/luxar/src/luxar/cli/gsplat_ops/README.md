# `luxar.cli.gsplat_ops`

Command implementation package for `luxar gsplat ...`.

The public CLI surface is assembled in `luxar.cli.gsplat_commands`, which imports
and registers command groups from this package.

## Layout

Three subpackages own the three large command families; everything else stays at
the package root. Each subpackage's orchestrator is named `commands.py`. Python cannot host both
`fitting.py` and `fitting/` in one directory — the package shadows the module —
so that one orchestrator could not stay at the root; `batch` and `transforms`
could have, but three uniform `commands.py` beats two-at-the-root-and-one-inside,
and it keeps each family's entry point next to the code it registers. The `__init__.py` files are docstring-only:
importers name the owning module directly, never a re-export root.

### Package root

- `scene_commands.py` — Typer command signatures + registration for convert/migrate-format/reencode
- `inspect_commands.py` — Typer command signatures + registration for inspect commands
- `interchange_commands.py` — Typer command signatures + registration for `import` / `export`
- `benchmark.py` — GPU benchmark helpers
- `recipe_shared.py` — `--recipe` names/methods/validators shared by the CLI and `cli/lod.py`
- `planner.py` — content-plan resolution + `run_content_fit`
- `encoding.py` — `_resolve_encoding_mode` shared by the transform, scene and interchange commands
- `loading.py` — shared matrix-only loading policy and flatten-first guidance

### `fitting/` — `fit` / `cal` / `render` / `denoise`

- `commands.py` — Typer command signatures + registration (`register_fitting_commands`)
- `calibrate.py` — `cal` implementation
- `denoise_render.py` — `denoise` + `render` implementations
- `fit.py` — `fit` command implementation
- `fit_utils.py` — shared fit helpers (tiling/recipe args/output save)
- `recipe_args.py` — fit-time `--recipe` option parsing/validation

### `batch/` — `batch-fit`

- `commands.py` — `batch-fit` app export (`app_batch`) + command wiring
- `submit.py` — `batch-fit submit` implementation
- `submit_pipeline.py` — plan/packing/preemptible pipeline behind `batch-fit submit`
- `plan_configs.py` — `build_plan_configs`: the **single** place flat CLI flags become
  `planning.py`'s four config dataclasses (`FitConfig` / `DenoiseConfig` /
  `ContentKnobs` / `MergeConfig`). Shared by `run` and `submit`, so it belongs to
  neither. `cli/tests/test_batch_config_agreement.py` fails if a second construction
  site appears, if the mapper consumes a flag one command does not expose, or if the
  siblings disagree on a shared default or public spelling — the two used to map the
  flags separately, which is how `--calibration-samples` ended up on `submit` only
  while the local path consumed it
- `submit_slurm.py` — Slurm submission/write orchestration helper for `batch-fit submit`
- `submit_packing.py` — scheduler-aware tasks/job packing heuristics for `batch-fit submit`
- `submit_plan_output.py` — stable human-readable plan summary printer for `batch-fit submit`
- `submit_preemptible.py` — preemptible-partition detection/access checks for `batch-fit submit`
- `run.py` — `batch-fit run` implementation
- `run_orchestration.py` — local `batch-fit run` planning/summary/execute orchestration
  helper. Takes the assembled `PlanConfigs` plus the execution-only flags (which GPUs,
  resume, dry-run), *not* the flat fit/denoise/content/merge flags again
- `merge_command.py` — `batch-fit merge` implementation
- `status_validate_cancel.py` — `batch-fit status` / `validate` / `cancel` implementations
- `denoise_workers.py` — hidden `batch-fit` denoise worker command implementations
- `measurement.py` — tile store size / bytes-per-splat measurement helpers
- `planning.py` — dataset discovery + decomposition + manifest planning (shared by `run` and `submit`)
- `recipe_args.py` — merge-time `--recipe` option parsing/validation
- `validation.py` — structural tile-store validation helpers (leaf arrays, node dirs)

### `transforms/` — edit-style commands on a fitted `.gsplats.zarr`

- `commands.py` — Typer command signatures + registration (`register_transforms_commands`)
- `transform.py` — `transform` implementation
- `merge.py` — `merge` implementation
- `cull.py` — `cull` implementation
- `decimate.py` — `decimate` implementation — reduce to a TARGET SPLAT COUNT,
  returning one flat leaf. Distinct from both neighbours: `cull` removes splats
  by a quality threshold (ratios, no target count) and `lod` builds a
  multi-level structure. Two families — `merge` (cluster neighbours into
  mass-carrying representatives) and `prefix` (keep the first N of an additive
  ordering, discarding the rest) — with `auto` following the measured crossover
  at 50% kept (`luxar.gsplats.lod.decimate.PREFIX_ABOVE_FRACTION`)
- `filter_slice.py` — `filter` + `slice` implementations
- `partition_flatten.py` — `partition` + `flatten` implementations
- `additive.py` — `additive` implementation
- `parsing.py` — shared parsing helpers (bbox, slices, CSV floats, `parse_threshold` for `pNN`/`NN%` percentile syntax)

### `filter` options (GSIP toolbox)

Every min/max threshold accepts a plain number OR a percentile written `pNN` /
`NN%` (resolved against that attribute's distribution — robust on heavy-tailed
data). Criteria AND together.

| Option | Meaning |
|--------|---------|
| `--bbox` | Spatial crop by center position |
| `--amplitude-min/max` | Intensity |
| `--scale-min/max` | Characteristic size = geometric-mean **spatial** sigma; auto-ignores a zero-variance time axis (timelapse-safe). The recommended "remove large diffuse background" knob. |
| `--volume-min/max` | `det(Σ)^(1/d)·truncate` over all dims (legacy size metric) |
| `--eccentricity-min/max` | Spatial isotropy (1.0 = sphere) |
| `--mass-min/max` | amplitude·volume (integrated brightness) |
| `--sigma-axis`+`--sigma-min/max` | Per-axis marginal sigma |
| `--isolation-max` | Remove splats whose nearest-neighbour distance exceeds it (isolated = noise); grouped by the non-spatial axis |
| `--min-neighbors`+`--neighbor-radius` | Remove poorly-supported splats |
| `--soft-highpass`/`--soft-lowpass`+`--soft-width` | SOFT reweighting: attenuate amplitude by a smooth function of scale instead of deleting (count unchanged) |
| `--spatial-dims` | Override the axes used for scale/eccentricity/isolation |
| `--dry-run` | Report impact (splats/mass/amplitude removed) and write nothing |

Reuses `BatchedSpatialHashGrid` (`gsplats/spatial_hash.py`) for the neighbour
queries; percentile/soft resolution lives in `filter_by`
(`gsplats/_data/filtering.py`) and `soft_scale_filter`
(`gsplats/_data/intensity.py`).

### `convert` appearance

`convert` bakes scene appearance: `--colormap` (builtin/matplotlib/colorcet,
validated), `--tone-mapping` (scene `viewer_config`, validated against
`VALID_TONE_MAPPINGS`), `--gamma`, `--intensity`, `--layer/--no-layer`.
Colormap/gamma/intensity/layer flow through `**attrs`; tone-mapping goes through
`ViewerConfig` on `create_scene`. Prefer `--tone-mapping ACES` (the house
default; passing it explicitly records the choice and silences the compiler's
LUT notice, which fires only when nothing was chosen). When a colormap carries
an exact scientific color encoding — ACES shifts hues — reach for `None`, an
exact passthrough, provided the scene stays inside [0, 1]; `Neutral` is not one
(even below its knee it subtracts a channel-minimum offset, so anything but a
fully saturated colour moves, and over range it keeps hue but sheds chroma,
while a `None` clamp flattens everything above 1.0 and can shift hue instead).

## Refactor invariants

When decomposing large files in this package:

1. **No compatibility wrappers.** Callers and tests import every function from
   its canonical module (the registration surface — `<pkg>/commands.py` or a
   root `*_commands.py` — or the concern-named helper that owns it). The former
   thin re-export roots (`batch.py`, `transforms.py`, `inspect.py`, `scene.py`)
   and their ~50 "Back-compat wrapper" functions were removed once all importers
   were repointed — don't reintroduce them when decomposing further, and keep the
   subpackage `__init__.py` files docstring-only for the same reason.
2. **Preserve CLI behavior** (flags, defaults, help text, output order/messages) —
   verify with a recursive `--help` dump diff (walk `typer.main.get_command(app)`
   with `COLUMNS=100` / `NO_COLOR`) before and after.
3. **Move implementation inward**, keeping the registration modules as thin
   signature + registration surfaces.
4. **Prefer concern-named helpers** (planning, parsing, recipe args, validation)
   over adding more unrelated logic to existing god files.

This package is intentionally implementation-oriented; the CLI reference for users
lives in:

- `README.md` (repo root)
- `packages/luxar/src/luxar/gsplats/README.md`
- `CLAUDE.md` / `AGENTS.md`
