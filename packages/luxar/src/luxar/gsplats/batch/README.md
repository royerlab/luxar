# luxar.gsplats.batch

Batch fitting orchestration for large OME-Zarr datasets, fanning a whole nD timelapse (T×C × tiles/boxes) across **either** a multi-GPU local machine (`batch-fit run`) **or** a Slurm cluster (`batch-fit submit`). Generates Slurm array jobs, runs a local subprocess pool, captures execution environments, estimates wall times from GPU profiles, tracks job status, and orchestrates the post-batch streaming merge. Planning (dataset discovery + decomposition + manifest) is shared by both execution backends and lives in `cli/gsplat_ops/batch/planning.py` (CLI layer); this package owns the manifest, the two execution backends, and the merge.

## Key Classes and Functions

- **`BatchManifest`** / **`BatchJob`** — Dataclasses representing the full batch job plan (dataset shape, tiling, Slurm parameters, fit config, denoising settings) and individual fitting tasks. Persisted as `manifest.json`. `floor_level` records the ONE background-floor level the plan resolved for the whole timelapse — the MINIMUM of the levels measured on a bounded set of evenly spaced `(t, c)` slices spanning the store's full extent (up to 4 timepoints, always including `t=0` and `t=T-1` when `T > 1`, x up to 4 channel-like coordinates), so it is a lower bound on every **sampled** slice's pedestal and cannot clip a sampled timepoint/channel to zero (which the tile worker would report as a legitimately empty tile, dropping that slice silently from the merge). Bounded sampling bounds only what it samples: a dimmer non-sampled slice can still be erased that way, and `--floor none` / an explicit numeric `--floor N` is the escape hatch. The level that drives the fits is the concrete number in `fit_args["floor"]`, so every `(t, c)` task subtracts the same pedestal instead of re-estimating its own. Without denoising, `norm_range` records the ONE raw-input range reduced over bounded samples of those same representative slices and forwarded through `fit_args["norm_range"]`, so normalized convergence, seeding, culling, and amplitude limits do not vary by timepoint or channel. Denoising runs leave an automatically sampled range unset so each task resolves on the data it fits: denoise-corrected input for an on-the-fly uniform tile, or the denoised store in `preprocess` mode (including content boxes). An explicit configured range remains an intentional override. Both fields are backward-compatible: an older manifest that omits them keeps its recorded fit arguments and historical behavior.
- **`save_manifest()`** / **`load_manifest()`** — Serialize/deserialize the manifest to/from the output directory.
- **`decode_task_id()`** — Map a flat `SLURM_ARRAY_TASK_ID` back to `(timepoint, channel, tile_index)`.
- **`generate_fit_sbatch()`** — Generate the main sbatch array job script for fitting, with support for sequential or parallel task packing, preemptible requeue, on-the-fly denoising, and preprocessed denoised input. Both packing modes retain per-task failures and make the array element exit non-zero after all assigned tasks have been joined or attempted. Output paths and preset values are quoted as literal arguments; percent signs in the output directory are doubled in the log directives so Slurm's filename-pattern expansion leaves them literal; output directories containing line terminators or NUL are rejected because Slurm directives are line-oriented.
- **`generate_calibrate_sbatch()`** / **`generate_denoise_sbatch()`** / **`generate_merge_sbatch()`** — Generate sbatch scripts for NLM calibration, denoise preprocessing, and post-fit merge jobs. Their log directives use the same literal output-path handling as the fit script.
- **`capture_environment()`** / **`CapturedEnv`** — Snapshot the current conda/venv, loaded modules (including those recorded in `cuda_build_info.json`), CUDA build info, and curated env vars for reproducible Slurm jobs.
- **`generate_env_preamble()`** — Convert a `CapturedEnv` into a shell preamble for sbatch scripts.
- **`get_slurm_scheduler_info()`** / **`is_slurm_mps_available()`** / **`detect_preemptible_gpu_partition()`** / **`validate_partition_access()`** — `scontrol`/`sinfo`-backed cluster introspection used at plan time to tune array packing and preemptible submission.
- **`estimate_tile_wall_seconds()`** / **`estimate_slurm_time_limit()`** — Log-interpolate GPU profile throughput tables to estimate per-tile wall time (with safety margin), then round up to a Slurm `--time` string.
- **`check_batch_status()`** / **`format_status_report()`** — Aggregate job status (`BatchStatus`) from output files and `sacct` queries, and render a human-readable report.
- **`run_batch_local()`** (`local_runner.py`) — The local execution backend: fit every `(t,c,slot)` task of a planned manifest with a multi-GPU subprocess pool (GPU workers pinned via `CUDA_VISIBLE_DEVICES`, host/device quality-memory shares exported per task, automatic concurrency bounded by GPU memory plus shared host RAM/CPU limits, weighted round-robin assignment), promote each task's **per-attempt** staging store (`{tile}.tmp.{host}-{pid}`) to its final path atomically, then call `merge_batch_results()`. Each invocation owns a unique staging token so two concurrent runs on the same `output_dir` never share or delete each other's in-progress store; only complete attempts race to claim the final output. Resumable (skips outputs/`.empty` markers already on disk). The local sibling of the Slurm fit array + merge.
- **`run_task_pool()`** / **`TaskResult`** (`task_pool.py`) — Generic subprocess pool (`ThreadPoolExecutor`) with a per-task `env_builder` hook (for GPU pinning and resource-share metadata) and a `skip_if` resume hook; returns every result, never raises on a worker failure.
- **`build_task_fit_argv()`** / **`iter_fit_arg_flags()`** (`fit_command.py`) — Build the concrete `luxar gsplat fit` argv for one batch task (the pure-python twin of the Slurm bash template); `iter_fit_arg_flags` is the single source for the `fit_args`→flag mapping shared by both.
- **`merge_batch_results()`** — Merge completed tiles into the final `.gsplats.zarr`. Default: a streaming `kind=partition` (one part per spatial tile, ≤1 tile-region resident at a time, preserving spatial structure for per-part frustum culling). Pass `recipe=` (`stream`/`levels`, see `PER_PART_RECIPES`) to give each tile-part its own LOD ladder as it streams — `stream` → the `tiles` topology, `levels` → `adaptive` — the memory-safe way to add level-of-detail to tiled output (the `lod` command rejects a partition outright). `flat=True` uses the legacy single-leaf 3-level fan-in (tiles per (T,C) → timepoints per channel → channels with optional color assignment), which reloads all tiles into memory; mutually exclusive with `recipe`.

## Module Structure

| File | Description |
|------|-------------|
| `manifest.py` | `BatchManifest` and `BatchJob` dataclasses, JSON serialization, task ID encoding/decoding |
| `fit_command.py` | Per-task `luxar gsplat fit` argv builder (`build_task_fit_argv`) + the shared `fit_args`→flag mapping (`iter_fit_arg_flags`), used by both the local runner and the Slurm bash template |
| `task_pool.py` | Generic subprocess task pool with per-task env (GPU pinning) + resume-skip hooks |
| `local_runner.py` | Local (non-Slurm) multi-GPU execution backend (`run_batch_local`) + merge |
| `slurm_gen.py` | Sbatch script generation for fit, calibrate, denoise, and merge jobs |
| `env_capture.py` | Environment detection (conda, venv, modules, env vars), Slurm scheduler queries, env preamble generation; absent optional probes stay quiet, while available probes that break warn once through the CLI's arbol warning bridge and preserve their fallback |
| `time_estimate.py` | Wall-time estimation from GPU benchmark profiles via log-log interpolation |
| `status.py` | Batch status checking via output file existence and `sacct` queries |
| `merge_orchestrator.py` | Post-batch merge: default streaming `kind=partition` (one part per tile); `--flat` for the legacy single-leaf fan-in (tiles -> per-(T,C) -> per-C -> final). The partition's split planes are rebuilt from the manifest's tile grid, which is in VOXELS -- a `voxel_size` in the run's `--config` moves the workers' splats out of that frame (a config `downscale:` does NOT, and is never recorded as one: every task rescales back to the full-resolution frame the planner tiled -- and where a decimating value cannot complete at all, a uniform plan whose workers re-tile their own decimated volume onto a grid that DIFFERS from the planned one, it is refused at PLAN time instead; a factor confined to axes that hold a single tile anyway leaves the grid identical and plans fine -- #1624), so the planes are scaled by the manifest's `grid_scale`, which the PLANNER resolved from `--preset`/`--config` (#1587); an absent one means the frames agree. A recorded frame also refuses `--refine volume` at this front door (each part's crop is taken in voxels from that grid), as the planner already did |

## Usage

Batch fitting is typically driven through the CLI:

```bash
# Plan and submit a Slurm fitting job (submits by default; --dry-run to plan only)
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu

# Check status
luxar gsplat batch-fit status output/

# Merge completed tiles
luxar gsplat batch-fit merge output/
```

Programmatic usage:

```python
from luxar.gsplats.batch.manifest import BatchManifest, save_manifest, load_manifest
from luxar.gsplats.batch.env_capture import capture_environment, generate_env_preamble
from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch
from luxar.gsplats.batch.status import check_batch_status
from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

# Build manifest, capture env, generate sbatch, submit...
env = capture_environment()
preamble = generate_env_preamble(env)
script = generate_fit_sbatch(manifest, preamble)
```
