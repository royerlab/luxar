# `luxar.cli.gsplat_ops`

Command implementation package for `luxar gsplat ...`.

The public CLI surface is assembled in `luxar.cli.gsplat_commands`, which imports
and registers command groups from this package.

## Ownership map

- `fitting.py` — Typer command signatures + registration wrappers (`fit`/`cal`/`render`/`denoise`)
- `fitting_calibrate.py` — `cal` implementation
- `fitting_denoise_render.py` — `denoise` + `render` implementations
- `fitting_fit.py` — `fit` command implementation
- `fitting_fit_utils.py` — shared fit helpers (tiling/recipe args/output save)
- `batch.py` — `batch-fit` app export + compatibility wrappers
- `batch_commands.py` — `batch-fit` app wiring + command wrappers
- `batch_submit.py` — `batch-fit submit` implementation
- `batch_submit_slurm.py` — Slurm submission/write orchestration helper for `batch-fit submit`
- `batch_run.py` — `batch-fit run` implementation
- `batch_merge_command.py` — `batch-fit merge` implementation
- `batch_status_validate_cancel.py` — `batch-fit status` / `validate` / `cancel` implementations
- `batch_denoise_workers.py` — hidden `batch-fit` denoise worker command implementations
- `transforms.py` — command registration + compatibility wrappers for edit-style commands
- `transforms_commands.py` — Typer command signatures for edit-style commands
- `transforms_transform.py` — `transform` implementation
- `scene.py` — scene command registration + compatibility wrappers
- `scene_commands.py` — Typer command signatures for convert/migrate-format/reencode
- `inspect.py` — inspect command registration + compatibility wrappers
- `inspect_commands.py` — Typer command signatures for inspect commands
- `benchmark.py` — GPU benchmark helpers
- `batch_planning.py`, `planner.py`, `encoding.py` — shared helpers

## Refactor invariants

When decomposing large files in this package:

1. **Keep root modules importable** (`batch.py`, `fitting.py`, `transforms.py`, etc.).
   They are import targets for command registration and tests.
2. **Preserve CLI behavior** (flags, defaults, help text, output order/messages).
3. **Move implementation inward**, leaving root modules as thin wrappers/orchestrators.
4. **Prefer concern-named helpers** (planning, parsing, recipe args, validation)
   over adding more unrelated logic to existing god files.

This package is intentionally implementation-oriented; the CLI reference for users
lives in:

- `README.md` (repo root)
- `packages/luxar/src/luxar/gsplats/README.md`
- `CLAUDE.md` / `AGENTS.md`
