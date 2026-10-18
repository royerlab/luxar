---
name: luxar-gsplat-pipeline
description: >-
  Fit, calibrate, and build LOD for Gaussian splats with the Luxar CLI. Use when
  a user wants to turn a 3D/nD volume (.zarr, .zarr.zip, .tiff, .npy, .npz) into a
  .gsplats.zarr, choose the splat count K, build level-of-detail topologies for the
  web viewer, run tiled fits on one GPU, or add a fitted gsplat node to a Luxar
  scene from Python. Covers the canonical cal -> fit -> lod pipeline, the full
  gsplat CLI option surface, and the Python fitting API. (For fitting a whole
  multi-timepoint timelapse across many GPUs or a Slurm cluster, use the
  luxar-hpc-batch-fit skill instead.)
---

# Luxar GSplat pipeline

Luxar fits **Gaussian splats** to scientific volumes and serves them to a WebGL
viewer. The standalone artifact is a `.gsplats.zarr` (format v3.4 — a node tree;
older v3.x files are still read transparently).
All commands below are subcommands of `luxar gsplat`.

GSplats need optional deps: `pip install "luxar[gsplats,io]"` (PyTorch, scipy,
numba; `io` reads TIFF, including compressed TIFF). GPU (CUDA or Apple MPS) is
auto-detected; pass `--device cpu` to force CPU. On Linux with a CUDA build of
torch the CUDA kernels compile on first use (NVRTC, cached under
`~/.cache/luxar/cuda`; no `make build-cuda`); on Apple MPS the 3D Metal kernels
compile at runtime (torch >= 2.7, no Xcode); on the CPU, fused numba kernels
compile on first use and run on several threads (`LUXAR_CPU_THREADS`). The fit log
names the kernels that ran; "Metal splatting kernels: none, using PyTorch" or a
`GaussianSplatModel (PyTorch, ...)` line means the slow reference path.
`LUXAR_CUDA_BACKEND` / `LUXAR_METAL_BACKEND` / `LUXAR_CPU_BACKEND` force one.
**A CPU fit is still far slower than a GPU fit**: read "Fitting on a CPU or a Mac"
in `packages/luxar/src/luxar/gsplats/README.md` first (`--downscale 2`, fewer
`--iters`, a short `--k-grid` for `cal`).

## The canonical pipeline: cal -> fit -> lod

Always prefer this order; each step writes a file the next consumes:

1. **`cal`** — pick the splat count `K*` in a principled way (blind-spot
   cross-validation; reports K*, curve type, noise floor, PSNR ceiling).
2. **`fit`** — fit splats to the volume at the chosen `K*`.
3. **`lod`** — build a representation topology (LOD / partition) for the viewer.

```bash
# 1. Calibrate K (Noise2Self blind-spot sweep). Writes cal.json.
luxar gsplat cal volume.tiff cal.json                  # 10 points, bounds from the foreground
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000   # faster

# 2. Fit at the recommended K* (read K* from cal.json, or use a --preset).
luxar gsplat fit volume.tiff splats.gsplats.zarr --seeds <K*>
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --timepoint 0 --channel 0

# 3. Build a topology (--recipe is REQUIRED). See "Choosing a LOD recipe".
luxar gsplat lod splats.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
```

**Check where K\* landed before trusting it.** `cal` sizes its grid from the
foreground voxel count, stops once the foreground held-out curve stops gaining
or falls past its peak, and extends once below a bottom-edge K\*; a K\* still on an edge (or
`signal_limited`) is a bound: widen `--k-min`/`--k-max` and rerun. K\* is a SEEDING
budget: `cal` prints what `fit --seeds K*` will store. A `--region`/`--auto-region`
K\* is a crop budget; `cal` also prints a whole-volume `--seeds` for the simple path.
Details and measurements: [references/calibration-sweep.md](references/calibration-sweep.md).

Presets: `draft` / `standard` / `hifi` / `ultra` / `n2s` (`n2s` = the manuscript's
blind-spot protocol, and `cal`'s default). Override any preset knob, e.g.
`--iters 8000`. Generate a config template with
`luxar gsplat fit --dump-config --preset hifi > config.yaml`, then `--config config.yaml`.

Every preset, fitter and tiling path defaults to `cull_retention=0.999` (keep
99.9% of cumulative amplitude; `cal` defaults to `n2s`, 20,000 iters). Still,
**`--seeds` proposes and the closing cull disposes**: on heavy-tailed sparse data
the culled tail is a lot of splats (cal's `stored` column shows how many). A bigger preset is not free either: `early_stop_patience` grows
with it (200 → 500, tested only every 25th iteration, so the larger one burns AT
LEAST 500 iterations after the last improvement), and it silently moves
`max_eccentricity` (10 → 20 from `draft` to `ultra`; `n2s` stays at 10).

Supported `fit` inputs: `.zarr`, `.zarr.zip`, `.tiff`, `.npy`, `.npz`; nested zarr
group: `--array-key h2afva/fused` (see the `luxar-data-loading` skill).

Anisotropic stack → physically scaled, image-oriented scene:
`luxar gsplat fit stack.ome.tif s.gsplats.zarr --physical` then `luxar gsplat
convert s.gsplats.zarr s.luxar.zarr`. `--physical` reads NGFF / OME-TIFF / ImageJ
spacing (one line says which); `--voxel-size Z,Y,X` (µm, array order) wins.
`--region z0:z1,y0:y1,x0:x1` fits a crop placed at its full-volume position
(`fitting/source_region`); `cal` takes the same three flags.

### Background floor suppression (`--floor`, default `auto`)

A constant pedestal (camera offset, autofluorescence) is the worst case for a
localized-Gaussian basis, so `fit`, `cal`, and `batch-fit` all subtract a
background floor (clip at 0) BEFORE normalization — output amplitudes are
therefore background-relative. `--floor` is ON by default (`auto`):

- `auto` — histogram-mode estimate, capped at the median (a no-op on clean data
  with no pedestal).
- `pNN` — subtract that percentile of non-zero voxels, e.g. `--floor p10`.
- a plain number — subtract a fixed value, e.g. `--floor 110`.
- `none` or `0` — disable (legacy hard-min behaviour; use to reproduce old numbers
  or when `auto` lands inside the specimen on a near-all-zero stack; see below).

`cal` applies the same `--floor` up front so K* is measured on floor-suppressed
data, matching how you fit.

Under any tiling, and across a `batch-fit` timelapse, one floor level is resolved
for the whole volume ([references/background-floor.md](references/background-floor.md)).

**Stay on `auto` unless you have measured otherwise.** A `pNN` floor subtracts a
percentile of non-zero voxels, so on sparse data it lands wherever the sparsity
puts it rather than where the noise ends: on a sparse light-sheet crop `p99`
erased the dim band (foreground 28.2 → 18.9 dB, dim-band mass 40.7% → 0.6%) while
`auto` stayed within 0.25 dB of no floor at all. The exception is a near-all-zero
stack, where `auto`'s mode of the non-zero voxels can land inside the specimen —
test `--floor none` there. A high floor **looks better in a MIP**; judge it on
foreground / dim-band PSNR against unfloored data, never on how the render looks,
and never port a floor choice between datasets without retesting. The measured
table is in [references/background-floor.md](references/background-floor.md).

```bash
luxar gsplat fit volume.tiff out.gsplats.zarr                 # --floor auto (default, recommended)
luxar gsplat fit volume.tiff out.gsplats.zarr --floor p10     # subtract 10th percentile
luxar gsplat cal volume.tiff cal.json --floor none            # legacy (no floor)
```

## Tuning a fit

### Scoring a fit

**`fit` already stamps the quality of the fit it just did** into `result.stats`
(`foreground_psnr_robust_db` = the headline, `foreground_psnr_db`, `foreground_threshold`,
`foreground_fraction`) and prints it, so most arms need no extra run. It is scored against
the input in raw units but on the fit's **background-relative basis**: the level
the fit actually subtracted (`image_min`: the resolved floor when suppression is
on, otherwise the data minimum or percentile endpoint) is removed and clipped,
while fit-time normalization is undone.
Reach for `compare` when scoring against a *different* reference than the volume
that was fitted — a denoised variant's untouched source, another arm's target:

```bash
luxar gsplat compare fitted.gsplats.zarr original.tiff --output-json metrics.json
#   --channel/--timepoint/--axes/--array-key (reference side, as in fit), --mask,
#   --fg-threshold, --at-fit-resolution, --device (--shape must equal the reference)
```

Headline PSNR = robust peak (reference p99.99 − floor; full-range `psnr_db` beside
it), plus SSIM, foreground SSIM, `mse`, `rel_l2`, `max_abs_error` and the
**foreground trio**. **Read the foreground number** (smoothed-Otsu mask on the
target), always with `foreground_fraction`. It is not interchangeable with `cal.json`'s
`held_out_psnr_fg_db`, and neither covers the dim band — details in
[references/foreground-metrics.md](references/foreground-metrics.md).

### Scoring a STACKED or TRANSFORMED archive (read before `gsplat compare`)

Five separate wrong numbers were produced in one day by the same mistake:
**comparing two things held under different conventions**, each a plausible dB.
Before trusting a `compare` on anything but the exact volume you fitted, read
[references/scoring-stacked-archives.md](references/scoring-stacked-archives.md).
The short form: score each timepoint **before** stacking and keep its stamps (a
stacked archive compared whole measured 17.57 dB vs 50.26 dB per timepoint, and
no CLI command slices it back); score **before** any spatial transform (even a
pure scale cost 16 dB); keep the archive's column order separate from the
scene's; never score a preprocessed fit against its own preprocessed input; and
when one fix gives a different wrong number, keep digging. Quality stamps are
the way out, but check they exist: merges (`combine_as_new_dimension`,
`batch-fit` merge) do not carry per-fit quality to the merged root.

### Denoising before a fit (`luxar gsplat denoise`, or your own filter)

Removing noise first is often worth it, but **choosing the filter and its strength
is where this goes wrong**, silently. A smoothing filter MOVES energy, so a metric
keyed on a mask of the unfiltered data scores smearing as removal (measured 12–30x
"reduction" where the real one was ~2x). Judge a filter on a MIP or in 3D, score
against the RAW volume, and match the filter to the noise model: on isolated-voxel
noise a connected-component SIZE filter beat NLM outright (it cannot damage what
it keeps). Set `--h` / `--denoise-h` explicitly: Noise2Self calibration is defeated
by very sparse data and answers at the edge of its grid. Measurements and the
scipy recipe: [references/denoising-before-a-fit.md](references/denoising-before-a-fit.md).

### Symptom → knob

An index into the measured sections, not a substitute for them.

| Symptom | Reach for |
| --- | --- |
| Thin/faint structure missing | start with `--floor auto`, never a higher floor; on a near-all-zero stack also test `--floor none` — "Background floor suppression" above — then raise K |
| `compare` says a known-good archive is mediocre | you compared a stacked 4D store whole, or across a transform — "Scoring a STACKED or TRANSFORMED archive" |
| Noise survives a denoising pass that measured well | you measured displacement, not removal — "Denoising before a fit"; on isolated-voxel noise use a component-size filter |
| Background haze survives | the viewer's display window and opacity, not a higher floor ([references/background-floor.md](references/background-floor.md)); or `filter --soft-highpass p90` (the `luxar-gsplat-edit` skill) |
| Thin filaments render as chains of beads | the six-knob schedule under "The Python API defaults to 1000 iters" below. NOT more seeds (measured *worse*), and NOT a preset: a preset moves `n_iters`, `early_stop_patience` and `max_eccentricity` — but nothing on this schedule, so `enable_dynamic_ops`, `patience` (the plateau LR-decay one, default 15, NOT `early_stop_patience`) and `l1_diag` all stay where they are and you must set them yourself |
| Blobby, over-smoothed detail | more K first, then a preset for iterations (early stopping makes a preset's `n_iters` a ceiling, so raise them when the loss is still falling at the cap) — but on thin structure read the beading row first |
| Elongated streak artifacts | lower `max_eccentricity` — no `fit` flag, set it in a `--config` YAML (`--dump-config` writes a template). It only starts binding once the fit is converged, and there it measured 2.65 dB ([references/cli-options.md](references/cli-options.md)) |
| Result far bigger than needed | `decimate --target N` / `-f 0.1`, or `cull --target vol.npy -p 95` — not a lower `--cull-retention` and a refit |
| Boxy steps at tile boundaries | suspect the SOURCE (mosaic seams, coverage count), not the fit — measure the artefact's period first |
| First paint costs hundreds of requests | you picked a partition recipe on byte size — "First paint cost" under "Choosing a LOD recipe"; prefer `overview`, or raise `--max-elements` |
| Fit is slow, exploring | `--preset draft` (2,000 iters) for the search, one `hifi` run at the end |

`--seed-method` matters on structure the default misses: `peaks` (sparse
point-like maxima) and `decomposition` (blobs, slow) must be asked for, and `auto`
is not the same as `edges,grid` — see "Seed methods" in [references/cli-options.md](references/cli-options.md).

## Traps that cost real time

Each of these has burned a whole fit cycle. Check them before you launch a long run.

- **`--seeds` is a whole-volume budget, but only for the CLI.** `luxar gsplat fit
  --seeds K` divides K across the tiles it makes. The *Python* `fit_tiled_gsplats`
  does NOT — an integer `seeds` there is handed to every tile unchanged, so N tiles
  fit ~N × seeds splats. And `--tiling auto` is the DEFAULT, so a gigavoxel volume
  tiles whether or not you asked (see "Tiled fits" for the whole-volume threshold).
- **`--cal` / `--k-star-ref` / `--feature-*` apply ONLY to `--tiling content`.**
  Under `--tiling none|uniform` they are ignored — the CLI prints a `⚠ … apply only
  to --tiling content` line, but in a long arbol log that scrolls past. The symptom
  is a fit that lands at a few hundred splats when you asked for K* = 128,000,
  because `seeds` silently fell back to `auto`. A `cal --auto-region` K* is
  *region-scoped* anyway: pass `--seeds` the whole-volume budget `cal` prints for
  it, or transfer the density with `--cal` + `--tiling content`.
- **`fit` refuses an existing output** unless you pass `--overwrite` (like `lod`,
  `additive`, `transform`), so a re-run of a chained script stops at the first fit.
- **A/B two fits only at equal splat count.** Splat count dominates every quality
  metric, so comparing a 1.4 M-splat variant against a 2.7 M-splat one measures the
  count, not the variable you changed. Fix the seed budget on both arms.
- **Size VRAM for the per-splat intermediates, not the volume tensor.** A 3.2 Gvoxel
  whole-volume fit asked for 95 GiB after the volume itself came to 12.9 GB. Tile it.
- **Score against the ORIGINAL, on the foreground, per timepoint.** Global PSNR on
  a 97–99%-empty stack is flattered by the empty part; foreground (and a dim band,
  1–10% of max) is where the answer lives. Never score a floored/denoised fit
  against its own preprocessed input, a denoising sweep on a mask of the noise
  voxels (it measures smearing), or a stacked 4D archive WHOLE (17.57 vs 50.26 dB
  per timepoint) — see "Scoring a fit" above.

## Choosing a LOD recipe (`lod --recipe`)

Recipes are scale-ordered — pick by element count `N`, then check "First paint
cost" below, which overrides this ordering when first paint is request-constrained:

| Recipe | What it builds | Use when |
| --- | --- | --- |
| `flat` | single bare leaf, no LOD | tiny N / debug |
| `stream` | one leaf + additive prefix-sum ladder (fast first paint) | small/medium N |
| `levels` | coarse→fine substitutive replacement levels (zoom across scales) | medium/large N |
| `tiles` | spatial BSP parts, each culled + its own stream ladder | large N |
| `overview` | instant coarse substitutive cap + fine tiles branch | huge N |
| `adaptive` | BSP parts, each its own substitutive `levels` group | largest N |

`stream` is *additive* (progressively refines ONE leaf); `levels` is *substitutive*
(swaps a coarse level for a finer one as the object grows on screen). `overview` and
`adaptive` compose the two over spatial tiles.

### Scale-ordered is the wrong first question. Ask about the VIEW.

**Default to `stream`, and make anything costlier justify itself.** Element count
alone is a poor guide: a 128M-splat timelapse wants `stream`, while a 0.7M-splat
galaxy legitimately wants `levels`. Three questions decide it:

- **Is it ONE object, always fully in frame?** Then there is nothing to
  frustum-cull, and parts cost a request per node to bootstrap for no return
  (measured first paint: 63 requests for a single stacked leaf vs 689 for a
  44-part partition).
- **Is it viewed WHOLE, or at range?** At a full-frame view the finest
  substitutive level shows, so coarse levels are bytes nobody fetches; `levels`
  pays off only when the object is genuinely small on screen.
- **Does the RESIDENT set fit?** `MAX_SPLATS_PER_GSPLATS_NODE` is 4,194,304 on a
  4096-class GPU (8.38M at 8192); above it the tail is dropped as a clean-edged
  hole. Only the resident slice counts — a sliced nD node is measured per slice.
  For a STATIC object over the cap, parts are load-bearing.

On a time-stacked 4D node a SPATIAL partition buys nothing (the writer already
orders by time), but a partition per TIMEPOINT can win for stepped playback. And
`stream` on a flat store needs `luxar optimize` re-chunking chosen by the node's
ACCESS PATTERN (played vs keypress-navigated), or scrubbing gets worse. The
measured cases, the per-timepoint partition recipe, and the chunk-profile rules
are in [references/lod-recipe-measurements.md](references/lod-recipe-measurements.md).

```bash
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview -K 8 --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe adaptive --parts 8 -K 4 -L 2
```

`stream` default method is `auto`: `greedy` (optimal at every prefix) at
N ≤ 5000, else the cheap O(N log N) `self_energy` for large N. Override with
`--add-method`/`-m` if you want to force one. For `levels`/`overview`/`adaptive`,
`--coarsen-dims` lists center-column indices coarsening may merge over (the rest
become hard barriers — e.g. a time or channel axis must stay a barrier).

### First paint cost: choose on eager rungs, not on N

On a request-constrained host, recipe choice is driven by the rungs fetched before
first paint, not by element count or total tree nodes:

```
first-pass rungs = eager parts × 1 level × min(3, rungs per level)
converged rungs  = eager parts × 1 level × rungs per level
requests         ≥ eager rungs × arrays per rung   (+ one per extra chunk)
```

A `kind=lod` group fetches only its default level; a `kind=partition` group fetches
every part. A cold stream pass commits two rungs and prefetches a third, but background
refinement drains the rest, so size hard limits against the converged count. The
loader derivation, selector thresholds, measured example, and exact `requestCount`
procedure are in
[references/first-paint-requests.md](references/first-paint-requests.md).

**On a node with a hidden dimension, count elements per SLICE, not per node.** A
ladder's rungs are sized against the whole node while the viewer draws one hidden
coordinate, so an explicit absolute first rung (`-b stream:<c>`) is divided among
the slices and can starve them (the published Nexrad store's `stream:20000`
ladder has p05 = 4 splats per played coordinate). Prefer `--n-lods 3..4`, read the
CLI's logged `--target-ms` slice multiplier, and from Python use
`additive_lod=dict(n_lods=4, slice_dims=[3], recompute=True)` (raw pre-`dim_order`
centre columns) for an equal ABSOLUTE per-slice budget. The measured table and
rules are under "Sliced nodes" in
[references/first-paint-requests.md](references/first-paint-requests.md).

- **`overview` — ~3 eager rungs, independent of part count.** Its root is a
  `kind=lod` whose fine partition defers behind a fills-screen selector. Crossing
  it loads every fine part at once, so this saves opening-view requests, not total
  session requests.
- **`tiles`, `adaptive` — every part is eager**, so ~3 × P rungs on the first pass.
  The part factor buys frustum culling and, for `adaptive`, per-tile detail; pay it
  when requests are not the binding limit.
- **`stream`, `levels` — one eager leaf/level at load**, but `levels` promotes on
  frame 1 for typical whole-object framing, so do not assume its tree depth makes it
  the cheapest opening view.

So when first paint is request-constrained: prefer `overview` over `adaptive`/`tiles`
at huge N, or raise `--max-elements` to cut the part count. At one part,
`adaptive`/`tiles` reduce to the cheap `levels`/`stream` shapes behind a partition
wrapper; use those recipes directly. Large byte size alone is not a reason to choose
`adaptive`.

## Tiled fits (large volumes)

`--tiling auto` (default) picks none/uniform/content: whole volume up to 64 Mvoxel
on CPU, up to 256 M by the GPU's free memory (machine-dependent; faster on a GB10);
`--recipe`/`-j` keep >64 M tiled. A tiled fit emits a `kind=partition`; `--flat`
forces one leaf. Uniform tiles are Hann-blended: `max` blending shows a seam cross.

```bash
# Uniform tiles with Hann stitching (seamless under additive blending).
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
# Parallel tiles on ONE GPU (no Slurm); -j auto accounts for GPU memory plus
# shared host RAM/CPU limits.
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform --tile-size 256 -j 4
# Content-adaptive boxes (more splats where the volume is busy); needs a density.
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json -j 8
# Per-part LOD AT FIT TIME (tiled partition only): stream -> tiles,
# levels -> adaptive. Avoids a separate `lod` pass (which rejects a partition).
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform -j 4 --recipe stream --n-lods 6
```

## Whole-timelapse fitting at scale

A whole multi-timepoint / multi-channel volume across many GPUs or a Slurm cluster
is the **`luxar-hpc-batch-fit`** skill (`batch-fit run` / `submit` / `merge` / ...).

## Inspect, compare, convert, edit

```bash
luxar gsplat info splats.gsplats.zarr                        # statistics
luxar gsplat render splats.gsplats.zarr out.npy --shape 128,128,128
luxar gsplat compare fitted.gsplats.zarr original.tiff        # PSNR / SSIM / MSE
luxar gsplat napari splats.gsplats.zarr                       # inspect in napari
luxar gsplat view splats.gsplats.zarr                         # quick web viewer

luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --center   # -> web scene
luxar gsplat cull in.gsplats.zarr out.gsplats.zarr --target vol.npy  # error-budget cull
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1
luxar gsplat slice in.gsplats.zarr out.gsplats.zarr "0:50, :, 10:90"
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1 --center  # one factor per centre column
luxar gsplat merge a.gsplats.zarr b.gsplats.zarr -o merged.gsplats.zarr
luxar gsplat partition in.gsplats.zarr part.gsplats.zarr --parts 4 --rule sah
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr      # legacy -> v3.4
```

## Python fitting API

For programmatic fitting (e.g. in a demo or notebook), import from `luxar.gsplats`:

```python
import numpy as np
from luxar.gsplats import fit_gaussian_splats, generate_seeds

volume = np.load("volume.npy").astype(np.float32)   # any nD array

# Optional: generate seeds explicitly (else fit_gaussian_splats seeds internally).
seeds = generate_seeds(volume, method="auto", device="auto")

result = fit_gaussian_splats(
    volume,
    seeds=seeds,          # int count | float ratio | ndarray centers | GSplatData | None
    n_iters=5000,         # presets are a CLI concept; set knobs directly here
    lr=0.01,
    loss_type="l1",       # "l1" | "mse" | "poisson"
    max_eccentricity=10.0,
    device="cuda",        # "auto" | "cpu" | "cuda" | "mps"
    cull_retention=0.999,  # post-fit cull; the function default
)
# result is a GSplatData: .centers (N,d), .amplitudes (N,), .cholesky_factors (N, d(d+1)/2), .stats
result.save("fitted.gsplats.zarr", ordering="hilbert")
```

`fit_gaussian_splats` has NO `preset=`: presets expand to `n_iters` /
`early_stop_patience` / `max_eccentricity` / `cull_retention` ([references/cli-options.md](references/cli-options.md)).

### The Python API defaults to 1000 iters — BELOW `draft`. Always set a schedule.

`fit_gaussian_splats`'s `n_iters` defaults to **1000**: below the CLI's lowest preset
(`draft` = 2000) and a fifth of `standard` (5000). The CLI does NOT share it: with no
`--preset`, `load_fit_config` resolves the `standard` preset, so:

```bash
luxar gsplat fit vol.tiff out.gsplats.zarr                   # standard: 5000 iters
luxar gsplat fit vol.tiff out.gsplats.zarr --preset draft    # 2000
```

**From Python, pass `n_iters` (and `early_stop_patience`) on every real fit.**
Calling it without them is not "default quality", and the failure looks
like a rendering problem: **thin filaments render as chains of beads**, because
under-converged splats never leave their isotropic σ = 1-voxel seed shape. Six
defaults must move TOGETHER; any one alone underperforms:

```python
result = fit_gaussian_splats(
    volume,
    n_iters=10_000,               # 1000 leaves splats at their seed shape
    patience=200,                 # 15 decays the shape LR away early
    early_stop_patience=2000,     # 300 stops before shapes settle
    enable_dynamic_ops=False,     # relocation RESETS splats to isotropic mid-fit
    max_eccentricity=None,        # 10.0 caps the axis ratio at sqrt(10)
    l1_diag=0.0,                  # the default penalty pulls toward isotropy
)
```

On a 1-voxel-wide neuron dataset this took foreground PSNR from 22.94 dB (1000
iters) to **25.85 dB**; more seeds makes beading *worse*, and raising `n_iters`
with relocation still on loses ~1 dB. If you cache fits, key the cache on the
schedule too. Measurements and what does NOT fix it:
[references/thin-structure-beading.md](references/thin-structure-beading.md).

## Add a fitted gsplat node to a Luxar scene

A `.gsplats.zarr` is a detached scene subtree; graft it into a scene three ways
(scene authoring API — see the `luxar-visualization` skill for the full scene flow):

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())

    # (a) From a fitted file (auto-lowers multi-level LOD pyramids to a kind=lod group):
    scene.add_gsplats_from_file("embryo", "fitted.gsplats.zarr")

    # (b) From an in-memory fit result (GSplatData from fit_gaussian_splats):
    scene.add_gsplats_from_data("embryo", result)

    # (c) Fit in one step from a volume:
    scene.add_gsplats_from_volume("embryo", volume, seeds=8000, n_iters=5000, device="cuda")

    # (d) From raw arrays — cholesky_factors is the packed lower-triangular factor L
    #     of the COVARIANCE (Σ = L·Lᵀ); 3D = (N,6):
    #     [L00, L10, L11, L20, L21, L22]. Diagonal is scale-like:
    #     isotropic std σ -> [σ,0,σ,0,0,σ]  (NOT 1/σ)
    #     This is the Group.add_gsplats / AdditiveSubLOD docstring contract.
    scene.add_gsplats("trio", centers=c, amplitudes=a, cholesky_factors=L, colors=rgb)
```

The CLI shortcut `luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --center`
does (a) for you and writes a ready-to-serve scene. It keeps the amplitudes in
source data units (labelled `amplitude_unit`, floor in `amplitude_offset`) and
auto-exposes a separate `display_gain`: dimmed (never brightened) so the opening
view's predicted additive p99.9 is `--target-brightness` (2.0; 0.9 under
`--tone-mapping None|Linear`) instead of a saturated blob (side views and other
timepoints capped at 4x); `--no-auto-exposure` = per-splat gain only. The Python
adders keep per-splat normalisation by default; the Python equivalent is
`luxar.gsplats.exposure.data_unit_display_attrs(stats, amplitudes=..., plan=...)`
(plan from `plan_projected_exposure`), passed to `add_gsplats_from_file(...,
normalize_amplitudes=False, **attrs)`.

## Notes

- **Full CLI option tables** (every flag for `fit` / `cal` / `lod`, defaults, and the
  preset table) are in [references/cli-options.md](references/cli-options.md). Load it when a user needs a flag
  not shown above.
- `luxar gsplat <command> --help` is the authoritative live flag list.
- The `lod` command **rejects an existing partition** — to add LOD to tiled output,
  use `fit --recipe` / `batch-fit merge --recipe` (per-part LOD as it streams), or
  `gsplat additive` to ladder every leaf of an existing tree structure-preservingly.
- LOD switch thresholds are auto-derived by SCREEN-AREA occupancy halving and
  stamped `selector="screen-area"`: a whole-object `levels` ladder shows full
  detail while the object occupies at least half the screen and steps one level
  coarser per halving; no threshold knob (`adaptive` and `overview` keep the
  fills-screen anchor). Resolution-independent but not aspect-independent; details
  under "LOD switch tuning" in [references/cli-options.md](references/cli-options.md).
- See `docs/specs/GSPLATS_ZARR_FORMAT.md` for the v3.4 node-tree format.
