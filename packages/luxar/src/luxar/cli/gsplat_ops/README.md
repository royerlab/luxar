# `luxar.cli.gsplat_ops`

Command implementation package for `luxar gsplat ...`.

The public CLI surface is assembled in `luxar.cli.gsplat_commands`, which imports
and registers command groups from this package.

## Ownership map

- `fitting.py` — fit / cal / render / denoise commands
- `batch.py` — `batch-fit` command group (submit/run/status/merge/validate/cancel)
- `transforms.py` — edit-style commands (cull/filter/partition/flatten/additive/slice/transform/merge)
- `scene.py` — convert / migrate-format / reencode
- `inspect.py` — info / view / compare / annotate-quality / napari
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
