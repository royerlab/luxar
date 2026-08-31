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

Presets: `draft` / `standard` / `hifi` / `ultra` / `n2s` (`n2s` = the manuscript's
blind-spot protocol, and `cal`'s default). Override any preset knob, e.g.
`--iters 8000`. Generate a config template with
`luxar gsplat fit --dump-config --preset hifi > config.yaml`, then `--config config.yaml`.

Every preset sets `cull_retention=0.999`. The `0.95` you get without one never came
from a preset — it is the *fitter's* own default falling through, and it drops 5% of
amplitude after every fit. Chasing a false `signal_limited` curve out of `cal`, treat
BOTH as suspects — that retention *and* too few iterations at high K, which is the
reason `cal` itself defaults to the `n2s` preset (20,000 iters, retention 0.999).
A bare `fit` (no preset) still carries that `0.95` — on the CLI path, that is; and not
under `--tiling content`, whose boxes default to `0.999` with or without a preset.
Content only: `--tiling uniform` is not special-cased and a bare one still culls each
tile at `0.95`.
A direct Python `fit_progressive_gaussian_splats` call defaults to `0.98`, which the
CLI overrides — so **`--seeds` proposes and `cull_retention` disposes**: a post-fit
cumulative-amplitude cull discards the tail, and on heavy-tailed sparse data that
tail is a lot of splats — the final count is not `--seeds`. A bigger preset is not
free either: `early_stop_patience` grows with it (200 → 500), and since the counter
resets only on an improvement and the stop is tested only on eval iterations (every
25th), the larger one burns AT LEAST 500 iterations after the last improvement —
somewhat more, never fewer. The preset silently moves `max_eccentricity` too
(10 → 20 from `draft` to `ultra`; `n2s` stays at 10).

Supported `fit` inputs: `.zarr`, `.zarr.zip`, `.tiff`, `.npy`, `.npz`. For a
nested zarr group use `--array-key h2afva/fused`.

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
turning it off gained up to 10.7 dB foreground PSNR; see "Denoising before a fit".

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

## Tuning a fit

### Scoring a fit

**`fit` already stamps the quality of the fit it just did** into `result.stats`
(`foreground_psnr_db`, `foreground_threshold`, `foreground_fraction`) and prints
it, so most arms need no extra run. That stamped number is already scored against
the input in raw units but on the fit's **background-relative basis**: the level
the fit actually subtracted (`image_min`: the resolved floor when suppression is
on, otherwise the data minimum or percentile endpoint) is removed and clipped,
while fit-time normalization is undone.
Reach for `compare` when scoring against a *different* reference than the volume
that was fitted — a denoised variant's untouched source, another arm's target:

```bash
luxar gsplat compare fitted.gsplats.zarr original.tiff --output-json metrics.json
#   --channel/--timepoint (reference side only), --device, --truncate
#   (--shape is not an override: it must equal the reference shape. Omit it.)
```

`compare` reports `mse`, `psnr_db`, `ssim`, `rel_l2`, `max_abs_error` **and the
same foreground trio**. **Read the foreground number** — and see "Traps that cost
real time" below for what to score it *against*. Three things about that trio
worth knowing:

- The **threshold defaults to Otsu** on the target, not a fixed fraction of max.
- Foreground is defined on the **target**, never on the prediction — a fit that
  hallucinates signal is scored against where the signal actually is.
- `data_range` comes from the **whole** target, not from the foreground subset, so
  the figure is not inflated by a shrunken reference. That convention is all it
  shares with `cal.json`'s `held_out_psnr_fg_db`, which scores only the held-out ∩
  foreground voxels of a fit of the *masked* volume — the two dB numbers are not
  interchangeable. Always report `foreground_fraction` beside the dB: a
  PSNR over 0.01% of a volume means something very different from one over 40%.

Neither the global nor the foreground PSNR covers the **dim band** (1–10% of max,
above) — mask it yourself with `gsplats.rendering.render_to_volume_tensor` plus
`metrics.compute_psnr`
(both torch; `render_to_volume` returns NumPy, which `compute_psnr` rejects).

### Scoring a STACKED or TRANSFORMED archive (read before `gsplat compare`)

Five separate wrong numbers were produced in one day by the same mistake:
**comparing two things held under different conventions.** None looked like an
error — each produced a plausible dB that a reader would simply believe. Check
all six failure modes before you trust a `compare`; the measured cases and full
controls are in [references/scoring-convention-failures.md](references/scoring-convention-failures.md).

**1. Score each timepoint before stacking, and retain its stamps.** A stacked
axis is the LAST centre column in a `.gsplats.zarr`; the source movie is usually
time-FIRST, so comparing both whole silently misaligns them. One archive measured
17.57 dB foreground whole versus 50.26 dB per timepoint. There is currently no
CLI route to recover that per-timepoint score from the stacked store:
`compare --timepoint` slices only the reference, and `gsplat slice` preserves the
archive rank. Newly built stacks may carry `fitting/part_provenance`; follow the
format spec's nested part → channel → timepoint levels to find per-timepoint
coordinates, and quote dB only when every `fit_reference.kind` is `acquisition`.
Missing, `preprocessed`, and `synthetic` references are deliberately not quotable.
Do not present either CLI command as an archive slicer.

**2. Keep the input archive's column order separate from the compiled scene's.**
In an `add_gsplats`/`add_points` authoring call, `dim_order[i]` names input data
column `i`; the compiler permutes those columns into scene `Dimensions` order
before storing them. `dim_order` is not persisted. Therefore the upstream
`.gsplats.zarr` follows the authoring argument, while the compiled scene node
follows `Dimensions`. Also check what the stacked column HOLDS: it is often a
physical value (minutes, µm) rather than an index, so `frame = round(value / step)`.

**3. Score before any spatial transform. "It was only a scale" is not a defence.**
Fitted centres are usually voxel indices; a shipped archive has often been scaled
to physical units and recentred. Comparing the transformed archive against an
untransformed reference cost **16 dB** even for a pure diagonal scale. Score at
fit time. If you must score later, invert the transform explicitly and prove the
result against an independent control; do not assume a scale is harmless.

**4. Fixing ONE convention mismatch and getting a different wrong number is not
progress.** One radar comparison successively exposed a mismatched grid, a −999
no-data fill that poisoned Otsu, and centres in km against a voxel-indexed
reference. Fixing the first two still left 6.70 dB foreground. **Carry a prior for
what the number should be, and keep digging while it is violated** — a
wrong-but-less-wrong number is the most expensive state to stop in.

**5. Never assign an axis ROLE from a data-dependent property.** A levelling
rotation derived in the plane of "the two widest axes" is correct only until the
tilt passes ~45°, at which point the second axis becomes the wider one, the roles
swap, and the angle comes out 90° off: a synthetic 70° tilt was measured as −20°
and "levelling" made the cloud LESS elongated (in-plane ratio 2.30 → 1.15). It
passed ruff, mypy, the complexity ratchet and the existing tests. Fix roles by
the convention the consumer uses, and exercise any such derivation against
synthetic inputs whose answer you already know.

**6. Never compare a preprocessed fit against its own preprocessed input.** It is
excellent by construction. Score against the original — see "Denoising before a
fit" for the subtler version, where the mask itself is the thing that lies.

Quality stamps are the way out of most of this, but **check whether they exist
before relying on them**: they are written per fit by `fit`, and coverage is
uneven downstream. Both `combine_as_new_dimension` and the `batch-fit` merge load
or concatenate inputs without carrying their per-fit quality to the merged root.
The component fits and current per-tile stores do carry the numbers, so inspect
and preserve them before merging. Older physical-coordinate fits may lack them
because metrics were skipped before #1668. Absent stamps mean re-measurement, and
re-measurement means every trap above. For post-fit command context, also see the
`luxar-gsplat-edit` skill.

### Denoising before a fit (`luxar gsplat denoise`, or your own filter)

Removing noise before fitting is often worth it: a fit spends its splat budget on
whatever is in the volume, and on sparse data a surprising share of the energy can
be shot noise. But **choosing the filter and its strength is where this goes
wrong**, and the failure is silent.

**A metric keyed on a mask of the UNFILTERED data scores smearing as removal.**
NLM (and any smoothing filter) is a neighbourhood average: it SPREADS a spike
rather than deleting it. If you score "share of the noise voxels' energy still
present", energy that moved one voxel out has left the mask you are watching and
counts as removed — while remaining plainly visible as a softer, wider blob. On a
sparse confocal timelapse this reported a 12–30x noise reduction where the real
out-of-cell reduction was ~2x.

The obvious repair — energy outside a *dilated* signal mask — fails the other way:
the filter's own halo around real structure crosses the boundary and is counted as
residual noise (measured 1.049, "worse than no filter", where it was signal).

> **Rule: when a filter MOVES things, no mask fixed on the unfiltered data can
> separate movement from removal.** Prefer an operation that moves nothing, or
> judge by looking at a MIP / 3D view. A single Z slice will not show it — this
> class of noise lives across Z, and raw vs filtered look nearly identical slice
> by slice while differing obviously in projection.

**Match the filter to the noise model.** If the noise is isolated voxels — check
this, do not assume: on the zebrafish demo the MEDIAN object in a frame was ONE
voxel, p90 was 1–2 — then a connected-component SIZE filter beats a smoothing one
outright, because it cannot damage what it keeps:

| arm | frame energy removed (t=0) | signal energy | mean signal peak |
|-----|---------------------------|---------------|------------------|
| NLM h=0.05 | ~0.32 | 0.953 | **0.783** |
| drop components < 2 vox | 0.2959 | 1.0000 | 1.0000 |
| **drop components < 4 vox** | **0.3208** | **1.0000** | **1.0000** |
| drop components < 12 vox | 0.3225 | 1.0000 | 1.0000 |

There is no trade-off to tune: the signal columns are exactly 1.0000 at every
threshold *by construction*, while NLM dimmed peaks by 6–22%. Only the noise
column moves, and it plateaus quickly — so take the knee. Downstream the size
filter gave equal or better in-signal PSNR almost everywhere — NLM edged it
19.66 vs 19.51 at t=0 and 19.04 vs 18.77 at t=150 — while using ~1/10 the
splats at early timepoints and about half over the whole archive. The fitted
result reproduced 1–11% of out-of-signal energy against NLM's 22–50%.
Scipy, run per timepoint/channel rather than on the stacked array:
`ndimage.label(v > background, structure=generate_binary_structure(v.ndim, 1))`
then zero the labels whose `bincount` is below the threshold. Set `background`
above any pedestal (`0` is valid only for an exact-zero background), and set
connectivity EXPLICITLY — it changes what counts as one object.

Two habits that make any such comparison trustworthy:

- **Score against the RAW volume**, the one reference no arm touched. Never score
  a filtered fit against its own filtered input — that is excellent by construction.
- **Define the signal reference STRICTER than any threshold under test** (e.g.
  components >= 27 voxels while testing thresholds of 2–12) so no arm is being
  judged against its own definition.

Set `--h` on `denoise` or `--denoise-h` on `fit --denoise` explicitly: without it,
both commands auto-calibrate a selected 3D volume on its central Z slice. Beware
`calibrate_nlm_h` (Noise2Self) on very sparse data: it is defeated by the same
sparsity that makes an `auto` floor land inside the specimen on a near-all-zero
stack. Where a volume is ~99% exact zeros, a held-out voxel is best predicted by
predicting zero, so maximal smoothing wins its cross-validation. With the grid
widened, it answered 0.055–0.225 on one stack depending on which slice it was
pointed at, every value at or past the point where the filter ate signal — and
its default `h_range` stops at 0.08, so a default call pins at the ceiling wherever
the answer is above it.
**An estimator that answers at the edge of its own grid has told you nothing.**

### Symptom → knob

An index into the measured sections, not a substitute for them.

| Symptom | Reach for |
| --- | --- |
| Thin/faint structure missing | start with `--floor auto`, never a higher floor; on a near-all-zero stack also test `--floor none` — "Background floor suppression" above — then raise K |
| `compare` says a known-good archive is mediocre | you compared a stacked 4D store whole, or across a transform — "Scoring a STACKED or TRANSFORMED archive" |
| Noise survives a denoising pass that measured well | you measured displacement, not removal — "Denoising before a fit"; on isolated-voxel noise use a component-size filter |
| Background haze survives | the viewer's display window and opacity — same section; or `filter --soft-highpass p90` (the `luxar-gsplat-edit` skill) |
| Thin filaments render as chains of beads | the six-knob schedule under "BOTH entry points default to 1000 iters" below. NOT more seeds (measured *worse*), and NOT a preset: a preset moves `n_iters`, `early_stop_patience` and `max_eccentricity` — but nothing on this schedule, so `enable_dynamic_ops`, `patience` (the plateau LR-decay one, default 15, NOT `early_stop_patience`) and `l1_diag` all stay where they are and you must set them yourself |
| Blobby, over-smoothed detail | more K first, then a preset for iterations (early stopping makes a preset's `n_iters` a ceiling, so raise them when the loss is still falling at the cap) — but on thin structure read the beading row first |
| Elongated streak artifacts | lower `max_eccentricity` — no `fit` flag, set it in a `--config` YAML (`--dump-config` writes a template). It only starts binding once the fit is converged, and there it measured 2.65 dB (`references/cli-options.md`) |
| Result far bigger than needed | `decimate --target N` / `-f 0.1`, or `cull --target vol.npy -p 95` — not a lower `--cull-retention` and a refit |
| Boxy steps at tile boundaries | suspect the SOURCE (mosaic seams, coverage count), not the fit — measure the artefact's period first |
| First paint costs hundreds of requests | you picked a partition recipe on byte size — "First paint cost" under "Choosing a LOD recipe"; prefer `overview`, or raise `--max-elements` |
| Fit is slow, exploring | `--preset draft` (2,000 iters) for the search, one `hifi` run at the end |

`--seed-method` matters on structure the default misses. `auto` draws on the same
two methods a comma list would name — `edges` (boundaries) and `grid` (uniform
coverage) — but it is NOT `edges,grid`: `auto` splits one seed budget 60/40 between
them and derives the grid spacing from its own share, while a comma list runs each
method on its own defaults. Both dedup the union, so that is not the difference —
the mix is. Either way the fit pipeline then subsamples or tops up to `--seeds`, so
the budget is never ignored; only what reaches it changes. `auto`
cannot itself appear inside a comma list. `peaks` (sparse point-like maxima) and
`decomposition` (blobs, slow) have to be asked for explicitly.

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
- **A denoising sweep scored on the noise voxels measures smearing, not removal.**
  Any smoothing filter moves energy; a mask fixed on the unfiltered volume cannot
  tell "deleted" from "moved one voxel". Judge a filter on a MIP or in 3D (a single
  slice hides it), and prefer a filter that moves nothing. See "Denoising before a
  fit".
- **A stacked 4D archive compared WHOLE is mis-sliced, not badly fitted.** The
  stacked axis is the last centre column; the source movie is time-first. Measured
  cost on one archive: foreground 17.57 dB compared whole vs 50.26 dB per
  timepoint. Score and retain each fit before stacking; the CLI cannot slice the
  stacked archive per timepoint. See "Scoring a STACKED or TRANSFORMED archive".
- **Score against the ORIGINAL, and on the foreground.** Global PSNR on a
  97–99%-empty stack is flattered by the empty part and barely moves; foreground
  (say, above 10% of max) and a dim band (1–10%) are where the answer lives. Never
  score a floored/denoised fit against its own preprocessed input.

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
galaxy legitimately wants `levels`. Three questions decide it, and all three
matter (measured 2026-08-26):

**Is it ONE object, always fully in frame?** Then there is nothing to
frustum-cull and parts cost a request per node to bootstrap for no return.
Measured first paint: **63 requests** for a single stacked leaf (8 nodes) vs
**689** for a `kind=partition` of 44 parts x 4 levels x 4 rungs (704 nodes) —
~15-18x, like-for-like on chunking. The scaling is SUBLINEAR (88x nodes, ~16x
requests), so halving a node count does not halve the cost.

**Is it viewed WHOLE, or at range?** The LOD selector is screen-occupancy based,
so at a full-frame view the *finest* substitutive level shows and coarse levels
are bytes nobody fetches. They pay off only when the object is genuinely small
on screen. `cryoem_virus` (a compact particle, always full-frame) moved
`levels -> stream` for **-28%** when regenerated (16.03 -> 11.50 MB, 16 -> 4
nodes; publishing the regenerated archive remains #1879);
`milky_way_dust` keeps `levels` because the galaxy really is orbited at range,
and pays +39% on purpose.

**Does the RESIDENT set fit?** `MAX_SPLATS_PER_GSPLATS_NODE` is 4,194,304 on a
4096-class GPU (8.38M at 8192). Above it the viewer reports the clamp at load
time and drops the tail; since storage is Hilbert-ordered that tail is one
contiguous lobe — a clean-edged hole, not noise. **Only the resident slice
counts**: an nD node sliced on a hidden axis is measured per-slice, so a
500-timepoint node holding 128M splats at ~256k/frame is fine. The compiler also
warns on the node total rather than the resident slice, so that warning is
expected for a sliced nD node that satisfies the runtime limit. For a STATIC
object over the cap, parts are load-bearing.

**On a time-stacked 4D node, a partition buys nothing at all.** The writer
lexsorts by the time barrier, so a timepoint's splats are already contiguous and
chunk-local without any partition. Measured: partition-per-timepoint cost +9%
bytes for zero benefit; substitutive levels cost +49%.

**One catch that comes with `stream`:** a flat store needs re-chunking or
scrubbing gets WORSE. Measured per timepoint step on a 4D leaf: **173 requests
as-built, 2 after `luxar optimise --profile archive`** — the as-built figure is
worse than a partitioned store's re-chunked 12. Additive-only and re-chunking
are a package, not alternatives.

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
coordinate, so an absolute first rung (`--target-ms`, `-b stream:<c>`) arrives
divided by the slice count: on a 500-timepoint leaf a 20,833-splat rung 0 is 42
splats on screen, and playback rendered an empty frame. `--n-lods L` is immune —
it is exactly `1/L` of the frame at any slice count. Prefer `--n-lods 3..4` there;
the "Sliced nodes" section of that reference carries the measured table.

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
    cull_retention=0.95,  # post-fit cull; the function default (presets: 0.999)
)
# result is a GSplatData: .centers (N,d), .amplitudes (N,), .cholesky_factors (N, d(d+1)/2), .stats
result.save("fitted.gsplats.zarr", ordering="hilbert")
```

Note: `fit_gaussian_splats` has NO `preset=` argument — the CLI presets just expand
to `n_iters` / `early_stop_patience` / `max_eccentricity` / `cull_retention` (see
the preset table in `references/cli-options.md`). Set those knobs directly in Python.

### BOTH entry points default to 1000 iters — BELOW `draft`. Always set a schedule.

`n_iters` defaults to **1000**: below the CLI's lowest preset (`draft` = 2000) and a
fifth of `standard` (5000). This is NOT only a Python-API quirk — `--preset` has no
default either, and `load_fit_config` layers one only `if preset is not None`, so:

```bash
luxar gsplat fit vol.tiff out.gsplats.zarr                   # 1000 iters (!)
luxar gsplat fit vol.tiff out.gsplats.zarr --preset standard # 5000
```

**Pass `--preset` (or `--iters`) on every real fit; a bare `fit` is a preview.**
Calling either path without it is not "default quality", and the failure is not
obvious — it looks like a rendering or splat-count problem:

**Symptom: thin filaments (axons, vessels, fibres) render as chains of beads.**
Under-converged splats never leave their seed shape. Edge seeding initialises them
*isotropic at σ = 1.0 voxel*, so a 1-voxel-wide filament is rebuilt from 1-voxel
spheres spaced ~2.5σ apart, which beads by construction. Diagnose by measuring the
fitted shapes — a median axis ratio near 1.0-1.3 with σ ≈ 1 voxel means the
optimizer never moved them, not that the basis cannot represent the structure.

Six defaults must move TOGETHER; any one alone underperforms:

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

Measured on a 1-voxel-wide neuron dataset (skeleton points dipping below 25% of
their local ridge / foreground PSNR): 1000 iters 22.2% / 22.94 dB → 10000 iters
with the above 16.5% / **25.85 dB**, against 11.9% for the raw data itself. Raising
`n_iters` while leaving relocation on LOSES ~1 dB, because it resets the shapes the
extra iterations just bought. 20000 iterations adds only +0.2 dB — beading converges
by ~5000, fidelity by ~10000.

**What does NOT fix beading** (all measured, so don't retry them): more seeds makes
it *worse* (finer subdivision shrinks σ faster than the gaps, so spacing went 2.45σ →
3.22σ); fewer seeds gives tighter spacing and still beads more; and widening the
render truncation radius barely moves it — the splats genuinely do not reach each
other.

Cost is fit time only (~2x), and it is one-time if you cache. **If you cache fits,
key the cache on the schedule too** — it changes splat SHAPES while leaving the
count identical, so a cache keyed only on seeds/floor/retention silently returns the
old fit and makes a retune look like a no-op.

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
