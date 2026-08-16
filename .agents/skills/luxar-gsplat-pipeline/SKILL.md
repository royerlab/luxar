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

GSplats need optional deps: `pip install "luxar[gsplats]"` (PyTorch, scipy). GPU
(CUDA or Apple MPS) is auto-detected; pass `--device cpu` to force CPU.

## The canonical pipeline: cal -> fit -> lod

Always prefer this order. Each step writes a file the next step consumes.

1. **`cal`** — pick the splat count `K*` in a principled way (blind-spot
   cross-validation; reports K*, curve type, noise floor, PSNR ceiling).
2. **`fit`** — fit splats to the volume at the chosen `K*`.
3. **`lod`** — build a representation topology (LOD / partition) for the viewer.

```bash
# 1. Calibrate K (Noise2Self blind-spot sweep). Writes cal.json.
luxar gsplat cal volume.tiff cal.json                  # 10-point sweep [1K, 512K]
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000   # faster
luxar gsplat cal volume.tiff cal.json --pdf cal_report.pdf        # PDF report

# 2. Fit at the recommended K* (read K* from cal.json, or use a --preset).
luxar gsplat fit volume.tiff splats.gsplats.zarr --seeds <K*>
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --timepoint 0 --channel 0

# 3. Build a topology (--recipe is REQUIRED). See "Choosing a LOD recipe".
luxar gsplat lod splats.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
```

Presets: `draft` / `standard` / `hifi` / `ultra`. Override any preset knob, e.g.
`--iters 8000`. Generate a config template with
`luxar gsplat fit --dump-config --preset hifi > config.yaml`, then `--config config.yaml`.

Supported `fit` inputs: `.zarr`, `.zarr.zip`, `.tiff`, `.npy`, `.npz`. For a
nested zarr group use `--array-key h2afva/fused`.

### Background floor suppression (`--floor`, default `auto`)

A constant pedestal (camera offset, autofluorescence) is the worst case for a
localized-Gaussian basis, so `fit`, `cal`, and `batch-fit` all subtract a
background floor (clip at 0) BEFORE normalization — output amplitudes are
therefore background-relative. `--floor` is ON by default (`auto`):

- `auto` — histogram-mode estimate, capped at the median (a no-op on clean data
  with no pedestal).
- `pNN` — subtract that percentile, e.g. `--floor p10`.
- a plain number — subtract a fixed value, e.g. `--floor 110`.
- `none` — disable (legacy hard-min behaviour; use to reproduce old numbers).

`cal` applies the same `--floor` up front so K* is measured on floor-suppressed
data, matching how you fit.

Under **any** tiling (`uniform`, `content`, `--tile k/M`, `-j N`) the spec is
resolved against the **whole volume**, never against a tile or box crop — which
would make abutting regions fit against different baselines and show brightness
steps at their boundaries. `uniform` and `content` resolve it once in the parent
and hand every tile/box the concrete level; the uniform `-j N`/`--tile k/M`
workers instead each resolve the spec themselves against that same whole volume,
which agrees because the sampler is deterministic. `batch-fit` extends this
across time: one global level for the whole timelapse, resolved at plan time as
the **minimum** of the levels measured on a bounded set of evenly spaced `(t, c)`
slices spanning the whole store — up to 4 timepoints (always including `t=0` and
`t=T-1` when `T > 1`) x up to 4 channel-like coordinates — and recorded in the
manifest. A minimum cannot clip a **sampled** timepoint/channel to zero and so
lose that slice; a dimmer NON-sampled slice still can, since bounded sampling
bounds only what it samples, so pass `--floor none` or an explicit numeric
`--floor N` when a particular slice must survive.

**Stay on `auto` unless you have measured otherwise.** A `pNN` floor subtracts a
percentile of *all* voxels, so on sparse data it lands wherever the sparsity puts
it rather than where the noise ends. On a 96x640x640 crop of a sparse light-sheet
brain — 1.01% of its voxels foreground (above 10% of max), 12.9% in the dim band
(1–10%, where thin faint neurites live) — `p99` sat at **1.34% of that crop's
max**, squarely inside signal. Note it is a crop figure: over the whole stack the
same percentile is 0.05% of peak. A `pNN` floor moves with whatever you point it
at, which is the whole problem. Every arm at a fixed seed budget, scored against
the **unfloored** original:

| floor | splats | global | foreground | dim-band mass recovered |
|-------|--------|--------|------------|-------------------------|
| none  | 47,172 | 41.90  | 28.49 dB   | 42.0% |
| auto  | 46,020 | 41.76  | 28.24 dB   | 40.7% |
| p95   | 39,859 | 40.33  | 27.04 dB   | 23.0% |
| p99   | 15,483 | 35.81  | **18.86 dB** | **0.6%** |

`auto` is within 0.25 dB of no floor at all, so pedestal removal is essentially
free; all the damage comes from raising the floor. `p99` also produced a third of
the splats from the same seeds — the structure was clipped to zero before fitting
began.

A high floor **looks better in a MIP** (the haze is gone and the render is
crisper than its own source). That is the trap: judge a floor on foreground /
dim-band PSNR against unfloored data, never on how the render looks. Handle
residual haze with the viewer's display window and opacity, not by destroying
data at fit time — and never port a floor choice between datasets without
retesting it there.

```bash
luxar gsplat fit volume.tiff out.gsplats.zarr                 # --floor auto (default, recommended)
luxar gsplat fit volume.tiff out.gsplats.zarr --floor p10     # subtract 10th percentile
luxar gsplat cal volume.tiff cal.json --floor none            # legacy (no floor)
```

## Traps that cost real time

Each of these has burned a whole fit cycle. Check them before you launch a long run.

- **`--seeds` is a whole-volume budget, but only for the CLI.** `luxar gsplat fit
  --seeds K` divides K across the tiles it makes. The *Python* `fit_tiled_gsplats`
  does NOT — an integer `seeds` there is handed to every tile unchanged, so N tiles
  fit ~N × seeds splats. And `--tiling auto` is the DEFAULT, so a gigavoxel volume
  tiles whether or not you asked (the whole-volume threshold is 64 Mvoxel).
- **`--cal` / `--k-star-ref` / `--feature-*` apply ONLY to `--tiling content`.**
  Under `--tiling none|uniform` they are ignored — the CLI prints a `⚠ … apply only
  to --tiling content` line, but in a long arbol log that scrolls past. The symptom
  is a fit that lands at a few hundred splats when you asked for K* = 128,000,
  because `seeds` silently fell back to `auto`. A `cal --auto-region` K* is
  *region-scoped* anyway: transfer it with `--cal` + `--tiling content`, never by
  passing it to `--seeds`.
- **`fit` has no `--overwrite`** (`lod`, `additive`, `transform` etc. do). `rm -rf`
  the output first, or a chained script dies mid-run on a stale store.
- **A/B two fits only at equal splat count.** Splat count dominates every quality
  metric, so comparing a 1.4 M-splat variant against a 2.7 M-splat one measures the
  count, not the variable you changed. Fix the seed budget on both arms.
- **Size VRAM for the per-splat intermediates, not the volume tensor.** A 3.2 Gvoxel
  whole-volume fit asked for 95 GiB after the volume itself came to 12.9 GB. Tile it.
- **Score against the ORIGINAL, and on the foreground.** Global PSNR on a
  97–99%-empty stack is flattered by the empty part and barely moves; foreground
  (say, above 10% of max) and a dim band (1–10%) are where the answer lives. Never
  score a floored/denoised fit against its own preprocessed input.

## Choosing a LOD recipe (`lod --recipe`)

Recipes are scale-ordered — pick by element count `N`:

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

```bash
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview -K 8 --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe adaptive --parts 8 -K 4 -L 2
```

`stream` default method is `auto`: `greedy` (optimal at every prefix) at
N ≤ 5000, else the cheap O(N log N) `self_energy` for large N. Override with
`--method` if you want to force one. For `levels`/`overview`/`adaptive`,
`--coarsen-dims` lists center-column indices coarsening may merge over (the rest
become hard barriers — e.g. a time or channel axis must stay a barrier).

## Tiled fits (large volumes)

`--tiling auto` (default) picks none/uniform/content. A tiled fit emits a
`kind=partition` (one part per tile) for viewer frustum culling; `--flat` forces a
single leaf.

```bash
# Uniform tiles with seamless Hann stitching.
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
# Parallel tiles on ONE GPU (no Slurm); -j auto sizes from free VRAM.
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform --tile-size 256 -j 4
# Content-adaptive boxes (more splats where the volume is busy); needs a density.
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json -j 8
# Per-part LOD AT FIT TIME (tiled partition only): stream -> tiles,
# levels -> adaptive. Avoids a separate `lod` pass (which rejects a partition).
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform -j 4 --recipe stream --n-lods 6
```

## Whole-timelapse fitting at scale

To fit a whole multi-timepoint / multi-channel volume across many GPUs (local) or a
Slurm cluster, use the dedicated **`luxar-hpc-batch-fit`** skill (`batch-fit run` /
`submit` / `status` / `validate` / `merge` / `cancel`, the GPU benchmark profile, and
running on Bruno). This skill covers the single-volume fit; that one covers the fan-out.

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
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center
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
    cull_retention=0.95,  # post-fit cumulative-amplitude cull
)
# result is a GSplatData: .centers (N,d), .amplitudes (N,), .cholesky_factors (N, d(d+1)/2), .stats
result.save("fitted.gsplats.zarr", ordering="hilbert")
```

Note: `fit_gaussian_splats` has NO `preset=` argument — the CLI presets just expand
to `n_iters` / `early_stop_patience` / `max_eccentricity` / `cull_retention` (see
the preset table in `references/cli-options.md`). Set those knobs directly in Python.

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

    # (d) From raw arrays — cholesky_factors packed lower-triangular; 3D = (N,6):
    #     [L00, L10, L11, L20, L21, L22]; isotropic std σ -> [1/σ,0,1/σ,0,0,1/σ]
    scene.add_gsplats("trio", centers=c, amplitudes=a, cholesky_factors=L, colors=rgb)
```

The CLI shortcut `luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --center`
does (a) for you and writes a ready-to-serve scene.

## Notes

- **Full CLI option tables** (every flag for `fit` / `cal` / `lod`, defaults, and the
  preset table) are in `references/cli-options.md`. Load it when a user needs a flag
  not shown above.
- Run `luxar gsplat <command> --help` for the authoritative live flag list — this
  skill summarizes the common paths; the CLI is the source of truth.
- The `lod` command **rejects an existing partition** — to add LOD to tiled output,
  use `fit --recipe` / `batch-fit merge --recipe` (per-part LOD as it streams), or
  `gsplat additive` to ladder every leaf of an existing tree structure-preservingly.
- LOD switch thresholds are auto-derived by SCREEN-AREA occupancy halving and
  stamped `selector="screen-area"`: each `coverage_fraction` is a literal screen-area
  fraction (projected bbox rect area / viewport area), so a whole-object `levels`
  ladder shows full detail while the object occupies at least half the screen and
  steps one level coarser per halving. No threshold knob, and RESOLUTION-independent
  (an NDC-area fraction, so the same framing reads the same on any monitor size).
  It is NOT aspect-independent: `fov` is vertical, so widening the viewport widens
  the visible world and lowers the area fraction — resizing square→ultrawide does
  move the switch points.
  (`adaptive` and `overview` are partition-bound and keep the fills-screen anchor.)
  Legacy stores and explicit `coverage_fractions=[...]` lists keep the older
  `selector="coverage"` diagonal metric; the viewer reads both.
- See `docs/specs/GSPLATS_ZARR_FORMAT.md` for the v3.4 node-tree format.
