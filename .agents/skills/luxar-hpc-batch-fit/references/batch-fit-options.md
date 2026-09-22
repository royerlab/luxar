# `luxar gsplat batch-fit` + `benchmark` — full option reference

Verified from `cli/gsplat_ops/batch/`, `benchmark.py`, `gsplats/batch/`. Run
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
`--target-features`, `--min-leaf` (256), `--max-leaf` (512). Content-box workers do
not implement on-the-fly denoising or progressive fitting, so batch planning rejects
those combinations instead of emitting tasks that silently ignore them. `batch-fit
submit --preprocess` is supported: it denoises to a store first, then points every
content-box worker at that store.

## Shared fit params
`--preset` (standard), `--config PATH`, `--seeds`, `--iters`/`-n`, `--progressive`,
`--splats-per-pass`, `--psnr-patience`, `--max-passes`, `--cull-retention`
(default 0.999 in BOTH modes — every task is a `fit --preset <the run's preset>`, and
every preset, `standard` included, sets 0.999; 0 keeps all). The older "0.95 for
uniform tiles, 0.999 for content boxes" split never existed in the batch path.

Output coordinates remain in index space by default. `--physical` opts both
`batch-fit run` and `batch-fit submit` into the selected OME-Zarr NGFF spatial
scale; a config-supplied `voxel_size` takes precedence over discovered spacing.

Under **uniform** tiling an integer `--seeds K` is a **whole-volume budget per
(t, c) volume**: every task is a `--tile k/M` fit, which divides K across that
volume's non-empty tiles with a mass-first adaptive rule instead of fitting K
per tile, so each timepoint/channel tracks K rather than K x the grid size. The
default weight is `Hann voxels × (mean above-floor intensity / ceiling)^saturation_exponent`;
the historical 10%-of-ceiling foreground weights are retained only when every
normalized foreground-weight share is within two percentage points of its mass-weight share
and threshold-empty tiles hold at most 10% of the total intensity mass. One plan uses that
rule only if every selected `(t, c)` slice qualifies. This preserves sparse step-like data
while a dim tile of real structure on hot-spot-normalised data is
proportional rather than falling to a token budget: homogeneous content gets the same
splat density in differently-sized tiles, while the calibrated exponent still
controls relative density between equally-sized tiles. A tile holding at least a
quarter of the equal mass share is never budgeted below a quarter of the equal
seed share. An exponent fitted by `cal --fit-exponent` is measured across region scales
on absolute feature counts; uniform weighting reuses it only for density, keeping
tile size linear at fixed density. Largest-remainder rounding preserves K whenever
K can give every non-empty tile one seed; otherwise each non-empty tile gets one.
When exact plan-time counts cannot be handed to workers, every task derives the
same M from the volume, grid, resolved floor, and Hann-window skip predicate. A
float ratio is scale-free and applied per tile unchanged. Under **content**
tiling `--seeds` is ignored: tasks are emitted as
`--tiling content --plan … --plan-box k` and each box takes its budget from the
shared density plan.

`--floor` (default `auto`, same as `fit`/`cal`): subtract a background floor /
DC-offset (clip at 0) before normalization, so amplitudes are background-relative.
`auto` = histogram-mode estimate (capped at median; no-op on clean data);
`pNN` = subtract that percentile of non-zero voxels; a number = fixed value;
`none` or `0` = disable.

The spec is resolved to **one global level for the whole timelapse** at plan time,
recorded in the manifest (`floor_level`), and handed as a concrete number to every
`(t, c)` task and every tile/box. It is deliberately *not* re-estimated per
timepoint or per tile: that would be a time-varying pedestal, i.e. brightness
flicker across the merged partition.

Without denoising, the plan likewise records one sampled raw-input normalization
range (`norm_range`) and forwards it to every task, keeping normalized optimizer
thresholds consistent across timepoints, channels, and spatial partitions.
Denoising runs leave that sampled range unset so each task resolves on the data it
fits: denoise-corrected input for an on-the-fly uniform tile, or the denoised store in
`preprocess` mode (including content boxes). A deliberately configured range remains
an intentional override.

That one level is the **minimum** of the levels resolved on a bounded set of at
most 16 evenly spaced `(t, c)` slices spanning the store's **full** extent: up to 4
timepoints x up to 4 channel-like coordinates, capped independently so neither axis
starves the other. `T > 1` always samples at least two timepoints **including the
endpoints `t=0` and `t=T-1`** (under a monotone pedestal drift the dimmest pedestal
is at an end, which is what makes the minimum a lower bound for the run), and
`C > 1` at least two channel-like coordinates. A minimum because the two errors are
not symmetric: under `clip(V - level, 0)` a too-LOW level is a recoverable
under-subtraction, while a too-HIGH one destroys signal — a level above some
`(t, c)`'s maximum clips that whole sub-volume to zero, which the tile worker
reports as a legitimate empty tile, so the slice goes silently missing from the
merge while `status` says success. Spanning the full extent (not the selection)
also means `--timepoints 0:50` and `0:100` resolve the *same* level, so a resumed
or extended run matches the tiles already on disk. Only a bounded sample of a
lazily-opened, axis-pinned view of each sampled slice is read (the whole-volume
sample budget divided among them — 2M voxels each at the 16-slice cap), so on the
normal (label-pinnable) path the floor resolution never materializes a timepoint —
not while planning, not on `submit --dry-run`, not on a resume re-plan. (A store whose
discovered labels cannot be pinned falls back to an eager per-slice read,
announced as it happens. The *content* box scan's **materialization** is unchanged:
it still reads up to `--plan-samples` whole timepoints to max-project — only which
timepoints it reads is fixed, since it now pins them by label too.)

**Residual risk:** bounded sampling bounds only the slices it samples. A dimmer
NON-sampled slice — a blank/bleached/bad frame between two samples of a long movie,
or a channel-like coordinate above the cap of 4 — can still clip to all zeros, fit 0
splats and vanish from the merge while `status` reports success. Use `--floor none`,
or an explicit numeric `--floor N` low enough for the dimmest slice, when a
particular slice must be guaranteed to survive.

Two edge cases are announced loudly rather than fudged: a level the "would erase
all signal" guard rejects on any sampled slice disables suppression for the whole
run, and a *negative* resolved level (dark-frame-corrected data) cannot be
forwarded as a concrete `--floor`, so the spec is forwarded and each task resolves
it itself — pedestals may then differ across the run, and the manifest records
`floor_level=None`. A resumed run whose `manifest.json` predates `floor_level`
likewise keeps its recorded spec, so it reproduces what it was planned with.

## Shared denoising (NLM)
`--denoise`, `--denoise-h`, `--denoise-2d`, `--denoise-patch-size` (3),
`--denoise-search-distance` (5), `--denoise-backend` (auto),
`--calibration-samples` (5 timepoints sampled for denoise-h calibration).

## Shared dataset selection
| Flag | Meaning |
| --- | --- |
| `--axes` | comma axis names, e.g. `time,camera,channel,z,y,x` (forwarded to every task) |
| `--timepoints` | python slice, e.g. `0:10`, `::10`, `100:200:5` |
| `--channels` | python slice, e.g. `0:2`, `::2` |
| `--array-key` | array within the zarr store, e.g. `h2afva/fused` |

## Shared per-part LOD at merge time (baked into the merge job)
`--merge-recipe` (`stream` → tiles topology; `levels` → adaptive; default
bare-leaf parts) + `--merge-n-lods`, `--merge-add-method`, `--merge-breakpoints`,
`--merge-compression-factor`, `--merge-levels`, `--merge-subst-method`,
`--merge-coarsen-dims` (default spatial only; stacked-timepoint axis stays a barrier),
`--merge-refine` (`none`/`l2`/`volume`) + `--merge-refine-iters`. `volume` re-opens
THIS input at merge time and re-fits each tile against its own crop (one slice per
stacked timepoint) — the highest-fidelity coarse levels; it needs `--axes` recorded
and a single channel, both checked at PLAN time so a typo costs nothing.
On `batch-fit merge` the same pair is spelled `--refine` / `--refine-iters`.
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
| `--gpus-per-task` | 1 | GPU COUNT per task (distinct from local `run --gpus`, which selects devices) |
| `--cpus` | 4 | CPUs per fit task; parallel requests multiply this by resolved packing |
| `--mem` | 32 | GB per fit task; parallel requests multiply this by resolved packing |
| `--time` | auto | wall time per task (HH:MM:SS); auto-estimated otherwise |
| `--gpu` | auto | GPU name from profile |
| `--gpu-mem` | none | target GPU memory GB (picks closest profile) |
| `--tasks-per-job` | auto | fit tasks packed per Slurm job |
| `--parallel` / `--sequential` | sequential | run packed tasks concurrently, pinned round-robin across allocated GPUs |
| `--preprocess` / `--no-preprocess` | off | write denoised volumes to zarr before fitting |

## `batch-fit run INPUT OUTPUT` — local-only flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--gpus` | auto | `auto` (cards above a free-VRAM floor) / `all` / `cpu` / `0,1,3` |
| `--jobs-per-gpu` | auto | concurrent workers per GPU (auto accounts for GPU memory plus shared host RAM/CPU limits) |
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
| `--n-lods` / `--add-method` / `--breakpoints` | — | [stream] knobs |
| `-K`/`--compression-factor`, `-L`/`--levels`, `--subst-method`, `--coarsen-dims` | — | [levels] knobs |

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
