# Denoising before a fit (`luxar gsplat denoise`, or your own filter)

Loaded from the `luxar-gsplat-pipeline` skill. The measured cases behind "a
filter that moves energy cannot be scored on a mask of the unfiltered data", and
the component-size filter that beat NLM on isolated-voxel noise.

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
