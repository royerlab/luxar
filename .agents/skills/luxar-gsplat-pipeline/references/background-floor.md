# Background floor: tiling, timelapses, and the measured case against raising it

Loaded from the `luxar-gsplat-pipeline` skill's "Background floor suppression"
section, which carries the knob list and the decision rule. This file holds how a
floor is resolved under tiling and `batch-fit`, the measurement behind
"stay on `auto`", and how seeds start relative to the floor.

## How the floor is resolved under tiling and batch-fit

Under **any** tiling (`uniform`, `content`, `--tile k/M`, `-j N`) the spec is
resolved against the **whole volume**, never against a tile or box crop — which
would make abutting regions fit against different baselines and show brightness
steps at their boundaries. `uniform` and `content` resolve it once in the parent
and hand every tile/box the concrete level; a uniform `-j N` parent that resolves
the level while weighing its tiles forwards that concrete level and the raw-input
range to its workers, while a hand-run `--tile k/M` worker resolves the same spec
against the same whole volume. A user numeric floor on that hand-run worker is
guarded against the whole-volume maximum just as it is without tiling; a level
forwarded by a parent is marked resolved and applied verbatim. `batch-fit` extends
this across time: one global level for the whole timelapse, resolved at plan time as
the **minimum** of the levels measured on a bounded set of evenly spaced `(t, c)`
slices spanning the whole store — up to 4 timepoints (always including `t=0` and
`t=T-1` when `T > 1`) x up to 4 channel-like coordinates — and recorded in the
manifest. A minimum cannot clip a **sampled** timepoint/channel to zero and so
lose that slice; a dimmer NON-sampled slice still can, since bounded sampling
bounds only what it samples, so pass `--floor none` or an explicit numeric
`--floor N` when a particular slice must survive.

## Why to stay on `auto`: the measured light-sheet case

**Stay on `auto` unless you have measured otherwise.** A `pNN` floor subtracts a
percentile of non-zero voxels, so on sparse data it lands wherever the sparsity
puts it rather than where the noise ends. On a 96x640x640 crop of a sparse light-sheet
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

On this light-sheet crop, `auto` is within 0.25 dB of no floor at all, so pedestal
removal is essentially free; all the damage comes from raising the floor. `p99`
also produced a third of the splats from the same seeds — the structure was
clipped to zero before fitting began. The exception is a near-all-zero stack:
`auto` estimates the histogram mode of the **non-zero** voxels, so when that
population is signal the floor lands inside it. On the sparse confocal timelapse,
turning it off gained up to 10.7 dB foreground PSNR; see [denoising-before-a-fit.md](denoising-before-a-fit.md).

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

## Seeds start above the pedestal

Seeding samples 90% of the voxel value (`SEED_AMPLITUDE_SCALE`) so overlapping
seeds start below their target. That factor now applies to the signal above the
floor; it used to apply to the raw value, so a seed 10 counts above a 405-count
pedestal sampled 0.9 x 415 = 373, below the floor, and started at amplitude 0. Such
seeds never recovered, and the speck prune and cull removed them. On an Allen
hiPSC DNA stack (pedestal 405, robust peak 119) `fit --seeds 16000` stored 1,834
splats where `cal` (which subtracts the floor first) predicted about 13,100; with
the fix it stores 12,394 and foreground PSNR rises 18.5 → 19.5 dB. On EASI-FISH
(a 209-237 count pedestal) most seeds had started dead, which was most of the
"~90% of seeds culled" and of the puncta recall loss first blamed on the prune.
Data without a pedestal is unchanged within GPU run-to-run jitter. Refitting a
pedestal store made before the fix gives a different, larger result; quality
figures measured on such data before the fix are not comparable with new ones.
