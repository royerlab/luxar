# `luxar.cli.gsplat_ops`

Command implementation package for `luxar gsplat ...`.

The public CLI surface is assembled in `luxar.cli.gsplat_commands`, which imports
and registers command groups from this package.

## Ownership map

- `fitting.py` — Typer command signatures + registration (`fit`/`cal`/`render`/`denoise`)
- `fitting_calibrate.py` — `cal` implementation
- `fitting_denoise_render.py` — `denoise` + `render` implementations
- `fitting_fit.py` — `fit` command implementation
- `fitting_fit_utils.py` — shared fit helpers (tiling/recipe args/output save)
- `fitting_recipe_args.py` — fit-time `--recipe` option parsing/validation
- `batch_commands.py` — `batch-fit` app export + command wiring
- `batch_submit.py` — `batch-fit submit` implementation
- `batch_submit_slurm.py` — Slurm submission/write orchestration helper for `batch-fit submit`
- `batch_submit_packing.py` — scheduler-aware tasks/job packing heuristics for `batch-fit submit`
- `batch_submit_plan_output.py` — stable human-readable plan summary printer for `batch-fit submit`
- `batch_submit_preemptible.py` — preemptible-partition detection/access checks for `batch-fit submit`
- `batch_run.py` — `batch-fit run` implementation
- `batch_run_orchestration.py` — local `batch-fit run` planning/summary/execute orchestration helper
- `batch_merge_command.py` — `batch-fit merge` implementation
- `batch_status_validate_cancel.py` — `batch-fit status` / `validate` / `cancel` implementations
- `batch_denoise_workers.py` — hidden `batch-fit` denoise worker command implementations
- `batch_measurement.py` — tile store size / bytes-per-splat measurement helpers
- `batch_recipe_args.py` — merge-time `--recipe` option parsing/validation
- `batch_validation.py` — structural tile-store validation helpers (leaf arrays, node dirs)
- `transforms_commands.py` — Typer command signatures + registration for edit-style commands
- `transforms_transform.py` — `transform` implementation
- `transforms_merge.py` — `merge` implementation
- `transforms_cull.py` — `cull` implementation
- `transforms_filter_slice.py` — `filter` + `slice` implementations
- `transforms_partition_flatten.py` — `partition` + `flatten` implementations
- `transforms_additive.py` — `additive` implementation
- `transforms_parsing.py` — shared parsing helpers (bbox, slices, CSV floats)
- `scene_commands.py` — Typer command signatures + registration for convert/migrate-format/reencode
- `inspect_commands.py` — Typer command signatures + registration for inspect commands
- `benchmark.py` — GPU benchmark helpers
- `batch_planning.py`, `planner.py`, `encoding.py` — shared helpers

## Refactor invariants

When decomposing large files in this package:

1. **No compatibility wrappers.** Callers and tests import every function from
   its canonical module (the `*_commands` registration surface or the
   concern-named helper that owns it). The former thin re-export roots
   (`batch.py`, `transforms.py`, `inspect.py`, `scene.py`) and their ~50
   "Back-compat wrapper" functions were removed once all importers were
   repointed — don't reintroduce them when decomposing further.
2. **Preserve CLI behavior** (flags, defaults, help text, output order/messages) —
   verify with a recursive `--help` dump diff (walk `typer.main.get_command(app)`
   with `COLUMNS=100` / `NO_COLOR`) before and after.
3. **Move implementation inward**, keeping the `*_commands` modules as thin
   signature + registration surfaces.
4. **Prefer concern-named helpers** (planning, parsing, recipe args, validation)
   over adding more unrelated logic to existing god files.

This package is intentionally implementation-oriented; the CLI reference for users
lives in:

- `README.md` (repo root)
- `packages/luxar/src/luxar/gsplats/README.md`
- `CLAUDE.md` / `AGENTS.md`
