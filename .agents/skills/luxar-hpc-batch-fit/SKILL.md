---
name: luxar-hpc-batch-fit
description: >-
  Fit Gaussian splats to a whole nD timelapse at scale — across many GPUs on one
  box (local) or a Slurm/HPC cluster array job. Use when a user has a large
  multi-timepoint / multi-channel OME-Zarr volume and wants to fit it end to end:
  plan tiles over T×C, fit every tile, then stream-merge to one kind=partition
  .gsplats.zarr. Covers batch-fit run (local multi-GPU), batch-fit submit (Slurm),
  status/validate/merge/cancel, the gsplat benchmark GPU profile, and running on
  the CZ Biohub "Bruno" cluster.
---

# Luxar HPC / batch Gaussian-splat fitting

`batch-fit` fits a whole nD timelapse by planning tiles once over T×C, fitting each
tile, then running a **memory-safe streaming merge** to one `kind=partition`
`.gsplats.zarr`. Two entrypoints, same plan/merge machinery:

- **`batch-fit run`** — LOCAL multi-GPU (one box, no Slurm). Saturates every GPU.
- **`batch-fit submit`** — Slurm cluster array job.

`status` / `validate` / `merge` / `cancel` are shared. Input must be **OME-Zarr-style**
(it discovers T/C/spatial structure); flat `.npy`/`.tiff` is rejected — for a single
small volume use plain `luxar gsplat fit` instead.

## Decide: local vs cluster

| | `batch-fit run` (local) | `batch-fit submit` (Slurm) |
| --- | --- | --- |
| Where | one machine, N GPUs | HPC cluster array |
| Key flag | `--gpus auto/all/cpu/0,1` | `--partition`/`-p` (REQUIRED) |
| Resumable | yes (`--no-resume` to force) | yes (re-submit skips done tiles) |

## Local multi-GPU (no Slurm)

```bash
# Uniform tiles, every GPU above a free-VRAM floor; resumable.
luxar gsplat batch-fit run vol.zarr out/ --gpus auto --tile-size 256

# Content-balanced boxes (needs a density) + per-part LOD baked into the merge.
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json \
    --gpus auto --merge-recipe stream --merge-n-lods 4

# Subset of timepoints, 2 workers per GPU, explicit cards.
luxar gsplat batch-fit run vol.zarr out/ --gpus 0,1 --jobs-per-gpu 2 --timepoints ::10

luxar gsplat batch-fit run vol.zarr out/ --gpus cpu          # CPU fallback
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --dry-run
```

## Slurm cluster

```bash
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu                 # submits
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --dry-run       # plan only
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --preset draft  # fast preview
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --tile-size 256 # skip GPU profile

# Content-aware cluster fan-out: ONE box plan reused for every (t,c).
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --tiling content --cal cal.json \
    --plan-samples 24                                                   # 24-timepoint max-proj plan

# Pack/parallelize tasks per GPU; cap concurrency; preemptible for throughput.
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --parallel --max-concurrent 32
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu --preemptible

# Per-part LOD baked into the merge job.
luxar gsplat batch-fit submit data.zarr.zip out/ -p gpu \
    --merge-recipe levels --merge-compression-factor 4 --merge-levels 3
```

## Monitor → validate → merge

```bash
luxar gsplat batch-fit status out/ -v          # per-task state (sacct/squeue + disk)
luxar gsplat batch-fit validate out/ --fix     # delete corrupt/stale tiles for re-fit
luxar gsplat batch-fit merge out/              # (re)run merge -> kind=partition
luxar gsplat batch-fit merge out/ --recipe stream --n-lods 6   # + per-part LOD as it streams
luxar gsplat batch-fit cancel out/             # scancel all jobs for this run
```

`merge --recipe` is the memory-safe way to add LOD to tiled output: each spatial
tile-part gets its own ladder as it streams (the plain `lod` command rejects a
partition). `--recipe stream` → `tiles` topology (a per-part prefix-sum
ladder); `--recipe levels` → `adaptive` (per-part coarse↔fine levels). The
stacked-timepoint axis is always a hard coarsening barrier.

## GPU profile (enables auto tile-size)

Auto tile-size and wall-time estimates come from a benchmark profile in
`~/.luxar/gpu_profiles.yaml`. Build it once (or pass `--tile-size` to skip):

```bash
luxar gsplat benchmark --slurm --partition gpu   # profile a cluster GPU
luxar gsplat benchmark --list                    # show profiled GPUs
```

## Running on Bruno (CZ Biohub HPC)

The local Mac cannot reach Bruno directly — relay through `obsidian`
(`Mac → obsidian → Bruno`); obsidian connects non-interactively
(`ssh loic.royer@login.bruno.czbiohub.org`). Typical loop, from a Bruno login node:

```bash
luxar gsplat batch-fit submit /hpc/projects/<grp>/data.zarr out/ -p gpu --tiling content --cal cal.json
luxar gsplat batch-fit status out/ -v       # poll
luxar gsplat batch-fit validate out/ --fix  # clean failures, then re-submit to refill
luxar gsplat batch-fit merge out/ --recipe stream --n-lods 6
```

Build the CUDA extension on a GPU node first if needed: `make build-cuda SLURM=1`.

## Notes

- **Full flag tables** (every `submit`/`run`/`merge`/`benchmark` option, defaults, and
  the output-dir layout) are in `references/batch-fit-options.md`.
- Content mode REQUIRES a density: `--cal` (from `luxar gsplat cal`) or
  `--k-star-ref` + `--n-features-ref`.
- `--axes` must be consistent across plan and tasks; pass it explicitly if
  auto-detection is ambiguous (e.g. `--axes time,camera,channel,z,y,x`).
- For the single-volume / non-timelapse path, calibration, and the LOD recipe
  taxonomy, use the **`luxar-gsplat-pipeline`** skill.
