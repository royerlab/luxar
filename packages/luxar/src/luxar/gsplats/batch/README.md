# luxar.gsplats.batch

HPC batch fitting orchestration for large OME-Zarr datasets. Generates Slurm array jobs, captures execution environments, estimates wall times from GPU profiles, tracks job status, and orchestrates post-batch tile merging.

## Key Classes and Functions

- **`BatchManifest`** / **`BatchJob`** — Dataclasses representing the full batch job plan (dataset shape, tiling, Slurm parameters, fit config, denoising settings) and individual fitting tasks. Persisted as `manifest.json`.
- **`save_manifest()`** / **`load_manifest()`** — Serialize/deserialize the manifest to/from the output directory.
- **`decode_task_id()`** — Map a flat `SLURM_ARRAY_TASK_ID` back to `(timepoint, channel, tile_index)`.
- **`generate_fit_sbatch()`** — Generate the main sbatch array job script for fitting, with support for sequential or parallel task packing, preemptible requeue, on-the-fly denoising, and preprocessed denoised input.
- **`generate_calibrate_sbatch()`** / **`generate_denoise_sbatch()`** / **`generate_merge_sbatch()`** — Generate sbatch scripts for NLM calibration, denoise preprocessing, and post-fit merge jobs.
- **`capture_environment()`** / **`CapturedEnv`** — Snapshot the current conda/venv, loaded modules (including those recorded in `cuda_build_info.json`), CUDA build info, and curated env vars for reproducible Slurm jobs.
- **`generate_env_preamble()`** — Convert a `CapturedEnv` into a shell preamble for sbatch scripts.
- **`get_slurm_scheduler_info()`** / **`is_slurm_mps_available()`** / **`detect_preemptible_gpu_partition()`** / **`validate_partition_access()`** — `scontrol`/`sinfo`-backed cluster introspection used at plan time to tune array packing and preemptible submission.
- **`estimate_tile_wall_seconds()`** / **`estimate_slurm_time_limit()`** — Log-interpolate GPU profile throughput tables to estimate per-tile wall time (with safety margin), then round up to a Slurm `--time` string.
- **`check_batch_status()`** / **`format_status_report()`** — Aggregate job status (`BatchStatus`) from output files and `sacct` queries, and render a human-readable report.
- **`merge_batch_results()`** — Merge completed tiles into the final `.gsplats.zarr`. Default: a streaming `kind=partition` (one part per spatial tile, ≤1 tile-region resident at a time, preserving spatial structure for per-part frustum culling). Pass `recipe=` (`additive`/`substitutive`, see `PER_PART_RECIPES`) to give each tile-part its own LOD ladder as it streams — `additive` → the `partitioned` topology, `substitutive` → `mosaic` — the memory-safe way to add level-of-detail to tiled output (the `lod` command rejects a partition outright). `flat=True` uses the legacy single-leaf 3-level fan-in (tiles per (T,C) → timepoints per channel → channels with optional color assignment), which reloads all tiles into memory; mutually exclusive with `recipe`.

## Module Structure

| File | Description |
|------|-------------|
| `manifest.py` | `BatchManifest` and `BatchJob` dataclasses, JSON serialization, task ID encoding/decoding |
| `slurm_gen.py` | Sbatch script generation for fit, calibrate, denoise, and merge jobs |
| `env_capture.py` | Environment detection (conda, venv, modules, env vars), Slurm scheduler queries, env preamble generation |
| `time_estimate.py` | Wall-time estimation from GPU benchmark profiles via log-log interpolation |
| `status.py` | Batch status checking via output file existence and `sacct` queries |
| `merge_orchestrator.py` | Post-batch merge: default streaming `kind=partition` (one part per tile); `--flat` for the legacy single-leaf fan-in (tiles -> per-(T,C) -> per-C -> final) |

## Usage

Batch fitting is typically driven through the CLI:

```bash
# Plan and submit a Slurm fitting job (submits by default; --dry-run to plan only)
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu

# Check status
luxar gsplat slurm-fit status output/

# Merge completed tiles
luxar gsplat slurm-fit merge output/
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
