# `luxar gsplat batch-fit` + `benchmark` — full option reference

Verified from `cli/gsplat_ops/batch.py`, `benchmark.py`, `gsplats/batch/`. Run
`luxar gsplat batch-fit <sub> --help` for the live list. `submit` = Slurm array;
`run` = local multi-GPU; `status`/`validate`/`merge`/`cancel` are shared.

`batch-fit` requires an **OME-Zarr-style** input (it discovers T/C/spatial structure);
it rejects flat `.npy`/`.tiff`.

---

## Shared tiling options (`submit` and `run`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tiling` | uniform | `uniform` (regular grid) or `content` (content-balanced boxes) |
| `--tile-size` | auto | uniform tile edge (voxels); auto from GPU profile if omitted |
| `--overlap` | 32 | inter-tile overlap voxels (Hann-stitched) |

### Content mode (`--tiling content`) — needs a density
`--cal PATH` (calibration JSON) **or** `--k-star-ref` + `--n-features-ref`. The shared
box plan is scanned from a temporal **max-projection** over up to `--plan-samples` (16)
evenly-spaced timepoints (covers signal at ANY t); `--plan-timepoint N` pins one
timepoint instead. Tuning: `--saturation-exponent` (0.44), `--saturation-cap`,
`--feature-threshold`, `--feature-metric` (peaks/edges/intensity), `--cell` (16),
`--target-features`, `--min-leaf` (256), `--max-leaf` (512).

## Shared fit params
`--preset` (standard), `--config PATH`, `--seeds`, `--iters`/`-n`, `--progressive`,
`--splats-per-pass`, `--psnr-patience`, `--max-passes`, `--cull-retention`
(default 0.95 uniform / 0.999 content; 0 keeps all).

## Shared denoising (NLM)
`--denoise`, `--denoise-h`, `--denoise-2d`, `--denoise-patch-size` (3),
`--denoise-search-distance` (5), `--denoise-backend` (auto).

## Shared dataset selection
| Flag | Meaning |
| --- | --- |
| `--axes` | comma axis names, e.g. `time,camera,channel,z,y,x` (forwarded to every task) |
| `--timepoints` | python slice, e.g. `0:10`, `::10`, `100:200:5` |
| `--channels` | python slice, e.g. `0:2`, `::2` |
| `--array-key` | array within the zarr store, e.g. `h2afva/fused` |

## Shared per-part LOD at merge time (baked into the merge job)
`--merge-recipe` (`stream` → tiles topology; `levels` → adaptive; default
bare-leaf parts) + `--merge-n-lods`, `--merge-additive-method`, `--merge-breakpoints`,
`--merge-compression-factor`, `--merge-levels`, `--merge-substitutive-method`,
`--merge-coarsen-dims` (default spatial only; stacked-timepoint axis stays a barrier).
`--channel-colors "#ff0080,#00ff00"` for per-channel merge. LOD switch thresholds
are auto-derived (`coverage_fraction`, no knob — the `--merge-lod-method` flag
has been removed).

Every subcommand: `--dry-run` shows the plan without submitting/fitting.

---

## `batch-fit submit INPUT OUTPUT` — Slurm-only flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--partition` / `-p` | **REQUIRED** | Slurm partition |
| `--max-concurrent` | none | cap simultaneous array tasks (`--array=0-N%MAX`) |
| `--preemptible` | off | also submit on a preemptible partition (auto-detected); auto-requeue |
| `--preemptible-partition` | auto | explicit preemptible partition |
| `--preemptible-concurrent` | =max-concurrent | concurrency on preemptible partition |
| `--account` / `-A`, `--qos` | none | Slurm account / QoS |
| `--gpus` | 1 | GPUs per task |
| `--cpus` | 4 | CPUs per task |
| `--mem` | 32 | GB per task |
| `--time` | auto | wall time per task (HH:MM:SS); auto-estimated otherwise |
| `--gpu` | auto | GPU name from profile |
| `--gpu-mem` | none | target GPU memory GB (picks closest profile) |
| `--tasks-per-job` | auto | fit tasks packed per Slurm job |
| `--parallel` / `--sequential` | sequential | run packed tasks concurrently on one GPU |
| `--preprocess` / `--no-preprocess` | off | write denoised volumes to zarr before fitting |
| `--calibration-samples` | 5 | timepoints sampled for denoise-h calibration |

## `batch-fit run INPUT OUTPUT` — local-only flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--gpus` | auto | `auto` (cards above a free-VRAM floor) / `all` / `cpu` / `0,1,3` |
| `--jobs-per-gpu` | auto | concurrent workers per GPU (auto sizes from free VRAM) |
| `--no-resume` | off | re-fit every task even if its output exists (default: resume) |

Local runner always denoises on-the-fly per tile (no `--preprocess`).

## `batch-fit status OUTPUT`
`--verbose` / `-v` — per-task status. Reads `manifest.json`, queries sacct/squeue.

## `batch-fit validate OUTPUT`
`--fix` — delete corrupt/incomplete tiles and leftover `.tmp` dirs for re-fitting
(preserves legacy-format tiles → convert with `gsplat migrate-format`). Reports
OK / MISSING / CORRUPT / EMPTY / UNMIGRATED / STALE_TMP. `{tile}.empty` = legitimate
0-splat output (counts OK).

## `batch-fit cancel OUTPUT`
No options. `scancel`s the job IDs recorded in the manifest (calibrate / denoise /
fit array / merge). NOTE: the preemptible array job id is not cancelled by this
command — cancel it manually with `scancel` if you used `--preemptible`.

## `batch-fit merge OUTPUT` — manual/recovery merge
Default = memory-safe `kind=partition` (one part per spatial tile, streamed).
| Flag | Default | Meaning |
| --- | --- | --- |
| `--flat` | off | legacy single flat leaf (loads all tiles); mutually exclusive with `--recipe` |
| `--recipe` | plan default | per-part LOD: `stream` (tiles) / `levels` (adaptive) |
| `--force` | off | re-merge even if outputs exist |
| `--channel-colors` | manifest | override per-channel colors |
| `--n-lods` / `--additive-method` / `--breakpoints` | — | [stream] knobs |
| `-K`/`--compression-factor`, `-L`/`--levels`, `--substitutive-method`, `--coarsen-dims` | — | [levels] knobs |

NOTE: at the `merge` command the knobs are bare (`--n-lods`, `-K`, `-L`); the
`--merge-*`-prefixed forms are only on `submit`/`run`.

---

## `luxar gsplat benchmark` — GPU profiling (feeds auto tile-size)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--list` | off | list profiled GPUs and exit |
| `--force` | off | re-run even if a profile exists (aggregates) |
| `--sweep` / `--no-sweep` | sweep | include splat-count sweep |
| `--shape` | auto | sweep volume shape, e.g. `512,512,512` |
| `--slurm` | off | submit as a one-shot Slurm job |
| `--partition` | none | partition (REQUIRED with `--slurm`) |
| `--verbose` / `--quiet` | verbose | output verbosity |

Writes `~/.luxar/gpu_profiles.yaml` (or `$LUXAR_PROFILES_PATH`). `submit`/`run` read it
for auto tile-size and wall-time estimates; `--tile-size` bypasses the profile.

---

## Output directory layout

```
output/
├── manifest.json          # batch state (BatchManifest); status/validate/merge read this
├── plan.json              # [content mode] shared box plan, reused by every (t,c)
├── *.sbatch               # [submit] fit_array / merge / (calibrate/denoise/preempt)
├── env_snapshot.sh        # [submit] captured env preamble
├── logs/                  # per-task stdout/stderr
├── tiles/                 # per-task .gsplats.zarr  (t00_c0_tile000... / _box000...)
│   └── *.empty / *.tmp    # 0-splat marker / interrupted-worker leftover
└── merged/
    └── final.gsplats.zarr # kind=partition (default) or single leaf (--flat)
```

Both `run` and `submit` are **resumable** — existing tile outputs (and `.empty`
markers) are skipped on re-invoke.
