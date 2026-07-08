# Refactor Plan: `packages/luxar/src/luxar/cli/gsplat_ops`

**Audit date**: 2026-07-02  
**Auditor**: Claude (`package-refactor-plan` skill from `royerloic/claude-code-skills`)  
**Audit start SHA**: `92557e082284e40be860f54bd1b195967f2f4c5e`  
**Branch**: `audit/codebase-audit-fixes`  
**Target package**: `packages/luxar/src/luxar/cli/gsplat_ops`  
**Pre-refactor LOC**: 8,479 total; largest orchestrators: `batch.py` 2,306 LOC, `fitting.py` 2,005 LOC, `transforms.py` 1,625 LOC.  
**Shrinkage target**: each command orchestrator ≤ 25% of its current size (`batch.py` ≤ ~575 LOC, `fitting.py` ≤ ~500 LOC, `transforms.py` ≤ ~405 LOC).  
**Gate command** (package-scoped):

```bash
hatch run python -m ruff check packages/luxar/src/luxar/cli/gsplat_ops \
  && hatch run mypy packages/luxar/src/luxar/cli/gsplat_ops \
  && hatch run pytest \
       packages/luxar/src/luxar/cli/tests/test_gsplat_cli_extended.py \
       packages/luxar/src/luxar/cli/tests/test_batch_run.py \
       packages/luxar/src/luxar/cli/tests/test_batch_validate.py \
       packages/luxar/src/luxar/cli/tests/test_batch_merge_validation.py \
       packages/luxar/src/luxar/cli/tests/test_fit_tiled_parallel_cli.py \
       packages/luxar/src/luxar/gsplats/tests/test_batch.py \
       packages/luxar/src/luxar/gsplats/planner/tests/test_planner_cli.py
```

**Sibling packages in flight**: none detected on the current branch; recent history includes `d57b97bb refactor(cli)!: unify gsplat fitting under one verb...`, so preserve all historical import paths from `luxar.cli.gsplat_ops.*`.  
**External imports into this package**: `luxar.cli.gsplat_commands` imports every registration surface; tests import private helpers from `batch.py`, `batch_planning.py`, `planner.py`, and `inspect.py`; docs/skills reference command files.

## Summary

`gsplat_ops` already fixed the worst historical problem — `gsplat_commands.py` is now a thin aggregator — but the package still contains several second-generation god files. `batch.py`, `fitting.py`, and `transforms.py` each combine Typer option declarations, validation, planning, subprocess orchestration, data I/O, progress printing, and helper logic in one file, violating [P3], [P4], and [P6]. The refactor should preserve all public import paths and CLI behavior, but move command bodies into concern-named helper packages so the root command modules become small registration/orchestration surfaces [P1], [P2], [P6].

## Survey Snapshot (Phase 0)

### File inventory

| File | Lines | Public / exported surface | Notes |
| --- | ---: | --- | --- |
| `__init__.py` | 8 | package docstring | OK; no substantive code. |
| `batch.py` | 2,306 | `app_batch`, command funcs, `_validate_tile`, `_build_merge_recipe_params`, `_measure_tiles_bytes_per_splat`, re-export `_select_plan_timepoints` | Biggest god file; owns `batch-fit` app and multiple long command bodies. |
| `batch_planning.py` | 689 | config dataclasses, `plan_batch`, `resolve_merge_recipe_args` | Planning is partially extracted but still has a 329 LOC planner and 150 LOC recipe resolver. |
| `benchmark.py` | 211 | `benchmark_gpu`, `register_benchmark_commands` | Acceptable size; no urgent refactor. |
| `encoding.py` | 17 | `_resolve_encoding_mode` | Small shared helper. |
| `fitting.py` | 2,005 | `fit_volume`, `calibrate_command`, `render_to_file`, `denoise_volume_cmd`, `register_fitting_commands` | Single-file mix of fit/cal/render/denoise plus recipe and output helpers. |
| `inspect.py` | 914 | `info_dataset`, `quick_view`, `compare_quality`, `annotate_quality`, `register_inspect_commands` | Medium god file; less urgent than top three. |
| `planner.py` | 351 | `run_content_fit`, `_resolve_density` | Semi-extracted content-fit helper; coupled to `fitting.py` and `batch_planning.py`. |
| `scene.py` | 353 | `convert_to_scene`, `migrate_format_command`, `reencode_command` | Acceptable, but can eventually split conversion/migration/reencoding. |
| `transforms.py` | 1,625 | cull/filter/partition/flatten/additive/slice/transform/merge commands | Editing toolbox god file; several independent command families. |

### Oversized function inventory

| Function | Lines now | Primary concerns mixed inside |
| --- | ---: | --- |
| `batch.py::batch_submit` | 953 | CLI options, volume discovery, density planning, Slurm script generation, dependency submission, status output. |
| `batch.py::batch_run` | 402 | local GPU selection, dry-run planning, task pool execution, merge orchestration. |
| `batch.py::batch_merge_cmd` | 293 | validation, recipe parsing, per-tile byte measurement, streaming merge, status output. |
| `batch_planning.py::plan_batch` | 329 | input discovery, axis selection, tiling/content planning, manifest assembly. |
| `fitting.py::fit_volume` | 1,065 | input loading, tiling mode resolution, per-tile subprocesses, fit invocation, output save, recipe handling. |
| `fitting.py::calibrate_command` | 510 | input loading, calibration args, report writing, console output. |
| `transforms.py::transform_dataset` | 351 | argument parsing, transform composition, intensity ops, save path handling. |
| `transforms.py::cull_dataset` | 269 | mode selection, target loading, culling execution, save/stat output. |
| `transforms.py::additive_dataset` | 262 | tree traversal semantics, breakpoint parsing, additive rebuild, output encoding. |
| `transforms.py::filter_dataset` | 208 | criteria parsing, mask construction, save/stat output. |
| `inspect.py::info_dataset` | 259 | tree summary, stats computation, histograms, console rendering. |

### Public API / external import map

Root files must remain importable because `gsplat_commands.py` imports them directly [P1, Non-Goal 1]:

- `batch.py`: imported by `gsplat_commands.py`; tests import `_validate_tile`, `_build_merge_recipe_params`, `_measure_tiles_bytes_per_splat`, `app_batch`, `_select_plan_timepoints`.
- `batch_planning.py`: imported by `batch.py`, tests, and gsplats batch/local-runner tests.
- `benchmark.py`: imported by `gsplat_commands.py`.
- `encoding.py`: imported by `gsplat_commands.py`, `scene.py`, `transforms.py`.
- `fitting.py`: imported by `gsplat_commands.py`.
- `inspect.py`: imported by `gsplat_commands.py`; tests patch `inspect.time.sleep`.
- `planner.py`: imported from `fitting.py`, `batch_planning.py`, and planner tests.
- `scene.py`: imported by `gsplat_commands.py`.
- `transforms.py`: imported by `gsplat_commands.py`.

Therefore the root modules must stay as compatibility/orchestration modules, but their internals can delegate to subpackages [P1, P2, P6].

### Existing violations

- [P6] `batch.py`, `fitting.py`, and `transforms.py` are not readable in one sitting and are dominated by command body logic rather than orchestration.
- [P4] `transforms.py` is named as one concern but actually contains at least six concerns: culling, filtering, spatial partitioning, flatten/additive tree normalization, slicing/transforms, and merging.
- [P4] `fitting.py` contains four command families: denoise, fit, render, and calibrate.
- [P4] `batch.py` contains both local execution and Slurm submission plus merge/validate/status/cancel/denoise commands.
- [P1] Some helper functions are public only because they remain at root for test/back-compat imports (`_validate_tile`, `_build_merge_recipe_params`, `_measure_tiles_bytes_per_splat`). Keep root wrappers, but move bodies deeper.
- [P4] There is no `README.md` inside `gsplat_ops/` documenting the package's command-group boundaries.

## Target Tree (Phase 1)

```text
packages/luxar/src/luxar/cli/gsplat_ops/
├── __init__.py
├── README.md                                  # [P3/P4] package map and ownership rules
├── batch.py                                   # [P1/P2/P6] app_batch + command registration/wrappers only
├── batch/                                     # [P3/P4] batch-fit implementation details
│   ├── submit.py                              # [P4] Slurm submission command body
│   ├── run.py                                 # [P4] local multi-GPU command body
│   ├── merge.py                               # [P4] merge command body + byte/splat measurement
│   ├── validate.py                            # [P4] validation command + tile/node validators
│   ├── status.py                              # [P4] status/cancel helpers
│   ├── denoise.py                             # [P4] batch denoise subcommands
│   └── recipe_args.py                         # [P4] merge recipe wrappers kept import-compatible from root
├── batch_planning.py                          # [P1] public planning API remains importable
├── batch_planning/                            # [P3/P4] private planning helpers
│   ├── axes.py                                # [P4] time/channel slice and axis resolution
│   ├── density.py                             # [P4] content knobs and density resolution adapter
│   ├── manifest.py                            # [P4] manifest/config assembly
│   └── fit_args.py                            # [P4] fit CLI arg assembly
├── benchmark.py                               # [P1] small enough to keep
├── encoding.py                                # [P1] small shared helper
├── fitting.py                                 # [P1/P2/P6] fit/cal/render/denoise wrappers only
├── fitting/                                   # [P3/P4] fitting command implementations
│   ├── denoise.py                             # [P4] raw-volume denoise command body
│   ├── fit.py                                 # [P4] `fit_volume` command body
│   ├── fit_output.py                          # [P4] output save/encoding/compression
│   ├── fit_recipe_args.py                     # [P4] fit-time recipe validation
│   ├── tiling_args.py                         # [P4] tiling mode resolution and option rejection
│   ├── render.py                              # [P4] render-to-file command body
│   └── calibration.py                         # [P4] calibration command body/report writing
├── inspect.py                                 # [P1/P2] inspect command wrappers
├── inspect/                                   # [P3/P4] inspection implementation
│   ├── info.py                                # [P4] stats and tree summary
│   ├── histograms.py                          # [P4] ASCII hist/stat helpers
│   ├── compare.py                             # [P4] compare_quality body
│   ├── quick_view.py                          # [P4] view serving/extraction body
│   ├── napari.py                              # [P4] napari viewer body
│   └── quality.py                             # [P4] annotate_quality body
├── planner.py                                 # [P1] content-fit public helper remains importable
├── scene.py                                   # [P1] convert/migrate/reencode wrappers
├── scene/                                     # [P3/P4] scene command implementations
│   ├── convert.py
│   ├── migrate.py
│   └── reencode.py
├── transforms.py                              # [P1/P2/P6] edit command wrappers only
└── transforms/                                # [P3/P4] editing implementation details
    ├── cull.py
    ├── filter.py
    ├── partition.py
    ├── flatten.py
    ├── additive.py
    ├── slice.py
    ├── transform.py
    ├── merge.py
    └── parsing.py                             # bbox/slices/csv parsing shared by edit commands
```

Notes:

- Python permits a `batch.py` module and `batch/` package sibling in the same directory; `import luxar.cli.gsplat_ops.batch` resolves the module, while the module can import helpers from `luxar.cli.gsplat_ops.batch.submit` [P2]. This preserves public imports [Non-Goal 1].
- Do not create stub-only modules. Root modules keep Typer command definitions, public wrappers, imports, and command registration; they are substantive compatibility/orchestration files [P2].
- Keep event/console output at the wrapper boundary when practical; if output moves, preserve exact message order and content [Non-Goal 2/3].

## Orchestrator Shrinkage Plan (Phase 1.4)

### `batch.py`

| Function | Lines now | Moves to | Root wrapper becomes |
| --- | ---: | --- | --- |
| `batch_submit` | 953 | `batch/submit.py::run_batch_submit(ctx)` | Typer signature + `return run_batch_submit(make_submit_ctx(...))` |
| `batch_run` | 402 | `batch/run.py::run_batch_run(ctx)` | Typer signature + delegate |
| `batch_validate_cmd` | 114 | `batch/validate.py::run_batch_validate(ctx)` | signature + delegate |
| `_validate_leaf_arrays`, `_validate_node_dir`, `_validate_tile` | 35/39/34 | `batch/validate.py` | import-and-call compatibility wrappers |
| `_build_merge_recipe_params` | 88 | `batch/recipe_args.py` | import-and-call wrapper |
| `_measure_tiles_bytes_per_splat` | 39 | `batch/merge.py` | import-and-call wrapper |
| `batch_merge_cmd` | 293 | `batch/merge.py::run_batch_merge(ctx)` | signature + delegate |
| `batch_denoise_*` | 55/86 | `batch/denoise.py` | signatures + delegates |

Projected `batch.py`: ~450–575 LOC (Typer app, command signatures, compatibility wrappers, context builders). This meets the 25% target if `batch_submit`/`batch_run` bodies are fully extracted [P6].

### `fitting.py`

| Function | Lines now | Moves to | Root wrapper becomes |
| --- | ---: | --- | --- |
| `denoise_volume_cmd` | 115 | `fitting/denoise.py` | signature + delegate |
| `_resolve_tiling` | 20 | `fitting/tiling_args.py` | compatibility wrapper or direct import |
| `_build_fit_recipe_params` | 128 | `fitting/fit_recipe_args.py` | compatibility wrapper |
| `_save_fit_output` | 26 | `fitting/fit_output.py` | compatibility wrapper |
| `fit_volume` | 1,065 | `fitting/fit.py` with subhelpers in `tiling_args.py`, `fit_output.py` | signature + delegate |
| `render_to_file` | 98 | `fitting/render.py` | signature + delegate |
| `calibrate_command` | 510 | `fitting/calibration.py` | signature + delegate |

Projected `fitting.py`: ~430–500 LOC if Typer signatures remain long. If the projection stays above 500 LOC, move repeated option groups into `Annotated[...]` type aliases in `fitting/options.py` without changing CLI flags [P6].

### `transforms.py`

| Function | Lines now | Moves to | Root wrapper becomes |
| --- | ---: | --- | --- |
| `cull_dataset` | 269 | `transforms/cull.py` | signature + delegate |
| `_parse_bbox` | 12 | `transforms/parsing.py` | compatibility wrapper |
| `filter_dataset` | 208 | `transforms/filter.py` | signature + delegate |
| `partition_dataset` | 111 | `transforms/partition.py` | signature + delegate |
| `flatten_dataset` | 97 | `transforms/flatten.py` | signature + delegate |
| `additive_dataset` | 262 | `transforms/additive.py` | signature + delegate |
| `_parse_slices`, `_parse_csv_floats` | 27/12 | `transforms/parsing.py` | compatibility wrappers |
| `slice_dataset` | 99 | `transforms/slice.py` | signature + delegate |
| `transform_dataset` | 351 | `transforms/transform.py` | signature + delegate |
| `merge_datasets` | 123 | `transforms/merge.py` | signature + delegate |

Projected `transforms.py`: ~350–405 LOC, mostly Typer signatures and `register_transforms_commands()` [P6].

## Execution Plan (Phase 2 — for the implementing agent)

Each step is a single commit gated by the package-scoped gate above.

### Step 1 — Add package README and helper package shells

- Add `gsplat_ops/README.md` describing command-group ownership and the invariant that root modules are public CLI registration surfaces.
- Add empty implementation folders only when immediately populated in the same commit; do not add placeholder files [P2].
- Gate.

### Step 2 — Extract pure parsing and recipe helpers first

- Move `_parse_bbox`, `_parse_slices`, `_parse_csv_floats` to `transforms/parsing.py`; keep wrappers/imports in `transforms.py` for test/back-compat imports.
- Move `_build_fit_recipe_params` to `fitting/fit_recipe_args.py`.
- Move `_build_merge_recipe_params` to `batch/recipe_args.py`.
- Gate.

### Step 3 — Extract validation and measurement helpers

- Move `_validate_leaf_arrays`, `_validate_node_dir`, `_validate_tile` to `batch/validate.py`; preserve root wrappers in `batch.py`.
- Move `_measure_tiles_bytes_per_splat` to `batch/merge.py`; preserve root wrapper.
- Gate.

### Step 4 — Extract `transforms.py` command families

- In this order: `filter`, `slice`, `partition`, `flatten`, `merge`, `cull`, `additive`, `transform`.
- For each command, move the body to the concern file and leave the Typer-decorated root function as a 1–3 line delegate.
- Preserve console output order byte-for-byte.
- Gate after every 1–2 command extractions.

### Step 5 — Extract `fitting.py` command families

- Move `_resolve_tiling` to `fitting/tiling_args.py` and `_save_fit_output` to `fitting/fit_output.py`.
- Extract `denoise_volume_cmd`, `render_to_file`, and `calibrate_command` first; these are less coupled than `fit_volume`.
- Extract `fit_volume` last, using a narrow `FitVolumeCtx` dataclass or `TypedDict` with no `this` back-pointer.
- If Typer signatures keep `fitting.py` above 500 LOC, introduce option aliases in `fitting/options.py` without changing option names/defaults/help.
- Gate.

### Step 6 — Extract `batch.py` command families

- Extract `batch_validate_cmd`, `batch_status_cmd`, `batch_cancel_cmd`, and denoise helpers first.
- Extract `batch_merge_cmd` next.
- Extract `batch_run`.
- Extract `batch_submit` last; split Slurm script generation and dependency chain assembly into helpers under `batch/submit.py` only if their bodies exceed ~300 LOC.
- Gate after every command family; run a focused dry-run CLI test after `batch_submit` extraction.

### Step 7 — Extract `inspect.py` opportunistically

- This is lower priority, but if still in scope, move histogram/stat helpers to `inspect/histograms.py`, tree/info logic to `inspect/info.py`, and command bodies to their concern files.
- Gate.

### Step 8 — Mirror tests only when source moves are stable

- Move tests only if they are currently organized by old file rather than behavior. Do not churn test paths unnecessarily.
- Keep import paths for private helper tests stable through root compatibility wrappers until a follow-up cleanup can update tests intentionally.
- Gate.

## Verification Plan (Phase 3)

### Callsite-invariance check

```bash
git diff 92557e082284e40be860f54bd1b195967f2f4c5e -- \
  packages/luxar/src/luxar \
  ':!packages/luxar/src/luxar/cli/gsplat_ops/**' \
  ':!packages/luxar/src/luxar/cli/tests/**' \
  ':!packages/luxar/src/luxar/gsplats/tests/**' \
  | grep -E "^[+-].*luxar\.cli\.gsplat_ops|^[+-].*\.gsplat_ops\."
```

Expected: empty except test-only import adjustments that intentionally still target root compatibility modules. Any production external import change violates [Non-Goal 1].

### Orchestrator size check

```bash
wc -l packages/luxar/src/luxar/cli/gsplat_ops/{batch.py,fitting.py,transforms.py}
```

Expected:

- `batch.py` ≤ ~575 LOC.
- `fitting.py` ≤ ~500 LOC.
- `transforms.py` ≤ ~405 LOC.

### Helper-tree shape check

```bash
find packages/luxar/src/luxar/cli/gsplat_ops \
  -path '*/__pycache__' -prune -o -type f -name '*.py' -exec wc -l {} + | sort -rn
```

Expected: most helpers ≤ 300 LOC; no new generic `utils.py` or `helpers.py`; no stub re-export files.

### Behaviour-invariance check

- Package-scoped gate passes after every step.
- `hatch run luxar gsplat --help`, `hatch run luxar gsplat fit --help`, and `hatch run luxar gsplat batch-fit --help` show the same command names and option defaults.
- Existing CLI tests for `flatten`, `batch-fit`, `fit --tiling`, `cal`, and `merge` continue to pass.

## Out of Scope / Follow-Up Work

- Do not redesign CLI options, rename commands, or change defaults.
- Do not move algorithmic code out of `luxar.gsplats.*`; this plan only reorganizes the CLI layer.
- Do not remove root private-helper imports in tests during the first refactor; keep compatibility wrappers and clean up tests in a later behavior-preserving pass.
- Do not fold dependency scanning, documentation updates, or bug fixes into refactor commits.

## Open Questions Resolved During Audit

- **Target package**: user selected `packages/luxar/src/luxar/cli/gsplat_ops` as the first `package-refactor-plan` target.

## The Final Test

After execution, `gsplat_ops` should read as a package of command registration surfaces at the root, with implementation details grouped by command concern below each root module. A new contributor should be able to answer: root files wire CLI commands; `fitting/` contains fit/cal/render/denoise implementation; `batch/` contains local/Slurm/merge/validate/status implementation; `transforms/` contains post-fit edit commands. If a root file still contains hundreds of lines of body logic after the extraction, [P6] failed and the plan needs another extraction pass.

Plan saved to `docs/reports/refactor-plan-gsplat-ops.md`. Hand this to a refactoring agent or follow it manually, one commit per step, with the gate command between each.
