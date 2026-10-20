# `gsplat cal` sweep defaults: what they do and the measurements behind them

`luxar gsplat cal` fits the blind-spot target once per grid point, so its cost is
the sum of the per-point fits, and the top of a geometric grid is most of it. Four
defaults keep that cost on the points that decide K\*:

| Default | What it does | Opt out |
| --- | --- | --- |
| Data-derived bounds | With no `--k-grid/--k-min/--k-max`: `k_max` = foreground voxels / 2 (clamped to [4000, 512000]), `k_min` = `k_max` / 512, still `--n-grid` (10) log-spaced points | give any bound or `--k-grid` |
| Early stop | Ends the sweep once the K\*-metric gained ≤ 0.1 dB for 2 consecutive points, or one point fell ≥ 0.2 dB below the best; the skipped K are in `cal.json` `sweep.k_values_skipped` | `--no-early-stop` |
| Low-edge extension | K\* on the smallest point: up to 2 cheaper points below it (the grid's own ratio), once | `--no-extend-below` |
| Foreground K\*-metric | K\* from the foreground held-out curve (`psnr_foreground`); the whole-volume `psnr_minmax` K\* is still reported | `--k-star-metric psnr_minmax` |

The foreground count is the lightly smoothed Otsu mask of the floor-subtracted
volume (the mask `cal` already scores `psnr_fg_weighted` on); the derivation is in
`cal.json` under `sweep.auto_grid`.

Every point prints the elapsed time and an ETA (a least-squares line `t = a + b·K`
over the measured points, warm-up point dropped). Treat it as rough: while the
fits still cost a flat ~10 s it read 0.1x of the remaining time, and past
K = 32000 0.4-1.8x, because an `n2s` fit's time follows its convergence as much
as K (cells3d: 27 s at 4000, 15 s at 8000, 175 s at 64000).
`cal.json` is rewritten after every point (`status: partial`), so a crash keeps
the measured points; Ctrl-C writes them (`status: interrupted`), prints the
best-so-far K\* and exits 130.

## Measurements (GB10, `n2s` preset, `--floor auto`, dense 13-point grid 125…512000)

Each sweep fits every point, so any grid, stop rule or metric can be replayed on
the same fits.

| Volume | Voxels | Foreground | Full sweep | Whole-volume (`psnr_minmax`) K\* over grid windows | Foreground K\* over grid windows |
| --- | --- | --- | --- | --- | --- |
| CTC Fluo-C3DH-A549 t000 (noisy, sparse) | 3.05 M | 3.3 % | 16 min (512K alone: 9.5 min) | 250 / 500 / 1000 / 2000 (moves with `k_min`) | 1000 on every window starting ≤ 1000 |
| skimage `cells3d` nuclei | 3.93 M | 16 % | 25 min (512K: 10 min) | 32000 | 16000 |
| Synthetic dense blob field + Poisson noise | 37.7 M | 7.2 % | 4000…128000 only: 94 min (128K alone: 63 min) | 32000 (plateau) | 64000 (peak) |

A549's whole-volume held-out curve spans 0.37 dB from K = 250 to 512000 (it
falls, then rises again past 128K), so its K\* is set by where the grid starts:
the "K\* moves 1000 → 250 when the grid is widened" report. The foreground curve
rises 1.5 dB to a peak at 1000, falls 0.5 dB after it, and gives 1000 whatever
the window. On `cells3d` both
curves peak cleanly and stay put; they differ by one doubling.

The synthetic sweep ran 4000…128000 (the 125-point fit alone took 17 min, the
128000-point 63 min); it stands in for the tester's 37 Mvoxel stack.

Early stop + extension, replayed on the same fits (selected metric = foreground):

| Volume | Grid | Points fitted | Fit time | K\* |
| --- | --- | --- | --- | --- |
| A549 | historical 1000…512000 | 5 (incl. 2 below) | 44 s of 924 s | 1000 (peak) |
| A549 | 125…512000 | 6 | 58 s of 956 s | 1000 (peak) |
| cells3d | historical 1000…512000 | 7 | 308 s of 1449 s | 16000 (peak) |
| synthetic 37.7 M | 4000…512000 | 6 (stops at 128000: −0.25 dB) | 94 min; 256000/512000 never fitted | 64000 (peak) |

The early stop needs the grid to reach two points past the peak. A cap far below
K\* hides it: every metric reads `signal_limited` at 16000 on a cells3d grid
capped there. The measured K\* sat at 20-40 foreground voxels per splat
(cells3d), 42-85 (synthetic) and 100 (A549); a cap at 2 foreground voxels per
splat leaves more than three doublings above the densest. The single-point
decline rule exists for the large stacks: there the points past the peak are
the costliest of the sweep, and waiting for a second non-improving point on the
synthetic stack would have fitted 256000 (~2 h more).

## K\* is a seeding budget

`--seeds` proposes and the closing reductions dispose. The table's `kept` column
is after the cumulative cull (the scored model), `stored` after the default `fit`
closing reductions (speck pruning at `min_extent`, then the cull), and the summary
prints `fit --seeds K* will store about N splats` (`k_star_stored`). The estimate
comes from cal's own fits of the blind-spot target, which carry fewer noise specks
than a fit of the raw volume: on A549 at K = 16000 (Mac, MPS), `cal` estimated 15078 and an
`n2s` fit of the raw volume stored 12148 (`standard`: 12150); at 128000 the raw
fits stored 97326 (`n2s`) and 71964 (`standard`).

## PSNR convention

`cal` prints PSNR on the robust peak (p99.99 − floor), as `fit` and `compare`
headline it; `cal.json` keeps both (`held_out_psnr_robust_db` beside the
full-range `held_out_psnr_db`). The peak is one constant per volume, so the two
differ by a constant `20·log10(robust/full)` and every K\* rule, which compares dB
differences, picks the same K. The SSIM column uses the same robust peak
(`full_ssim_robust`; the full-range `full_ssim` stays in `cal.json`). A held-out
score is still not a fit's in-sample score: expect `cal` numbers below the fit
summary's.

## Crop calibrations

`cal --region` / `--auto-region` measure a CROP budget. `cal` restates it for the
whole input (`whole_volume_seeds`): the input is cut into crop-sized boxes, each
gets the content planner's density transfer `K*·(n_box/n_crop)^alpha` (K* as the
density's post-cull reference count, features counted at the calibrated level,
capped at the saturation count), and the sum
is printed as `fit --seeds N`. `fit --tiling content --cal cal.json` applies the
same transfer per content box instead. Under `--region` the whole input is read
once for this, only when its header says ≤ 512 Mi elements; above that (or for an
unprobeable format) the suggestion is skipped with a message. The complete
`cal.json` is written before this step, so it cannot leave a partial file.

## Early stop and the saturation cap

The density's `saturation_cap` is the effective count at the TOP of the planned
grid. When early stop skipped that top it is estimated from the largest measured
point's kept/requested ratio and `splat_density.saturation_cap_measured` is
`false` (A549: the measured top would have been 2972, the full grid's 48371).
With `--k-star-metric psnr_foreground`, `held_out_peak` (the min-max K\*) comes
from the truncated curve; the summary says "on the early-stopped curve" and
`sweep.minmax_curve_truncated` is `true` (A549: 392 vs 1566 on the full grid).
A `partial` or `interrupted` `cal.json` handed to `--tiling content --cal` prints
a warning.

The stop also waits until the measured curve can be decided on its own: it spans
at least the 0.3 dB knee margin, does not read `signal_limited`, K\* is not its
top point, and it reads a `peak` / `low_k_optimal`, or a `plateau` whose latest
point is still within 0.1 dB of the best. A plateau already falling is a peak the
truncation has not resolved: a fresh A549 run (MPS) had 783 = 19.27, 1566 = 19.26,
3130 = 19.14 dB, which read as a plateau with K\* = 392; the rule fits one more
point and lands on the peak. The low-edge extension runs only when the curve's
best value is its smallest K (a creep whose 0.3 dB knee is the first point gains
nothing below).

Two limits. On a CREEPING curve (no knee: < 0.1 dB per doubling, or a steep
rise into a creep) the early-stopped K\* is tied to where the sweep stopped — and
the full-grid K\* is equally tied to `k_max`, since the 0.3 dB knee follows the
grid's top. Use `--no-early-stop` and an explicit `--k-max` when you need the
whole curve. A curve that never spans 0.3 dB never stops early: it runs the full
grid. And the single-point decline rule can cut a noisy peak: a 0.3 dB dip
before the real maximum ends the sweep at the dip.

## Sparse puncta (FISH spots)

Held-out PSNR averages over every masked voxel; on sparse, diffraction-limited
puncta the spot centres are a tiny share of them, so the curve is flat and K\*
lands far below what the spots need. `cal` detects spots on the floor-subtracted
volume and records, per K, the median rendered/reference peak at them
(`spot_peak_ratio` in `cal.json`). When a larger K rendered the spots better
and K\* reached under 95% of the sweep's best, it warns and names the first grid
K within 98% of that best (comparing renders with renders cancels the noise bias
of the reference peak). Under `--region`/`--auto-region` it restates that K as a
whole-volume `--seeds`. EASI-FISH level-0 crops: 27-93% of the best at K\* on
sparse crops, 99% on a dense one; an Allen DNA crop reached 91%, so textured
nuclei can trigger it too. K\* itself is unchanged.

Never fit real data with `--seeds auto`: it sizes 100-10,000 seeds from the
volume's linear extent, not its content.

## Relative early stop inside each fit

The sweep's early stop above ends the GRID; this one ends each FIT. Every preset
carries `early_stop_rel_tol: 0.001`: a fit stops once its best loss improved by
less than 0.1% over the last `early_stop_patience` iterations (500 for `n2s`), as
well as after a window with no improvement at all. Before it, the plain rule
accepted any improvement, and the top grid points crept for thousands of
iterations: on an Allen hiPSC K = 64000 point relL2 sat at 0.4860 for ~9,600
iterations. `early_stop_rel_tol: null` in a `--config` restores the plain rule.
The paper's Noise2Self tables use the Python API and are unchanged; to reproduce
the CLI recipe timings in the consumer-GPU and tiled-fitting supplementary
documents, set `early_stop_rel_tol: null`.

Measured before making it the default (RTX 3070, `n2s`, `--n-grid 5`, same grid
per volume; "sweep" is the summed fit time; fg = foreground held-out PSNR at K\*,
robust peak):

| Volume | Plain rule: sweep, K\*, fg | rel 1e-3 | rel 3e-4 | rel 1e-4 |
| --- | --- | --- | --- | --- |
| Allen hiPSC DNA, 65x624x924 | 2487 s, 512000, 19.32 | 315 s, 512000, 19.31 | 432 s, 512000, 19.31 | 1740 s, 512000, 19.32 |
| Tribolium light-sheet crop, 64x256x256 | 263 s, 38055, 43.87 | 96 s, 38055, 43.91 | 101 s, 38055, 43.86 | not run |
| FlyLight neuron crop, 96x256x256 (sparse) | 204 s, 11314, 21.51 | 57 s, 11314, 21.45 | 60 s, 11314, 21.42 | not run |
| IDR 5514375, 31x256x256 (flat curve) | 77 s, 290, 14.75 | 91 s, 290, 14.75 | 92 s, 31000, 15.08 | 77 s, 31000, 15.06 |

K\* under the default `psnr_foreground` criterion was unchanged on Allen,
Tribolium and FlyLight; on Allen's flat curve the `psnr_minmax` K\* moved from
22627 to 512000 at rel 1e-3. Losses were at most 0.10 dB on every held-out and
foreground point (FlyLight, K = 1000); the one larger move, 0.14 dB on a
Tribolium foreground point, was a gain. IDR's K\* flips between 290
(plateau) and 31000 (signal_limited) in two plain-rule runs with identical
settings too: its curve moves 0.18 dB between K = 290 and 31000 and sits at the
noise ceiling, so the flip is run-to-run noise, not the stopping rule. IDR's fits are
short and gain nothing. Standard-preset `fit`s (5000 iterations, patience 300) at
the top K: Allen K = 64000 175 s → 54 s, FlyLight K = 32000 29-39 s → 15 s,
Tribolium K = 64000 33-49 s → 21 s, IDR K = 31000 unchanged (it never stalls
within 5000 iterations). Foreground PSNR of the saved fits stayed within the
0.1-0.2 dB spread of two plain-rule replicates on every volume.

## On a CPU

A CPU fit costs roughly a fixed time per iteration at a given volume (the render
kernels are ~80% of it), so a sweep's CPU time follows its total iterations. With
the relative stop on, the quicker recipe is

    luxar gsplat cal vol.tiff cal.json --device cpu --preset standard --n-grid 5

Total iterations, read from the kept fits of each sweep (device-independent; the
sweeps ran on an RTX 3070 because the CPU host was saturated by other work):

| Volume | Default `cal` (n2s, 10 points) | `--preset standard --n-grid 5` | Ratio | K\*: default / recipe |
| --- | --- | --- | --- | --- |
| Allen hiPSC DNA | 16875 (8 fitted, sweep stopped at 128000) | 8025 | 0.48 | 16000 plateau / 512000 signal_limited (knee 16000 / 22627) |
| Tribolium crop (`--k-max 128000`) | 15475 | 9650 | 0.62 | 25398 / 38055 (one coarse step) |
| FlyLight crop (`--k-max 128000`) | 8400 | 5125 | 0.61 | 8640 / 11314 (one coarse step) |
| IDR 5514375 | 13850 | 11300 | 0.82 | 1944 / 290 (flat curve at the noise ceiling) |

Measured CPU cost per iteration (`taskset` 8 cores of an i9-9900K, 4-8 torch
threads): 31x256x256 at K = 6529 about 0.1 s, 64x256x256 at K = 16000 about
0.3 s; multiply by the totals above for a CPU estimate. `standard` caps each fit at
5000 iterations, which matters on the points that fit noise and never stall (IDR's
K = 31000 point ran 11,400 iterations under `n2s`). The 5-point grid steps 3.4-4.8x
instead of ~1.7-2x, so on a peaked curve K\* lands one coarse step from the default's;
on a creeping curve the 0.1 dB sweep stop never fires and K\* reads
`signal_limited` at the top (Allen), so read the knee (operating point) there.
`--preset draft` is not safe: it under-trains the top point and flipped K\* on IDR
5514375 (31,000 to 1,375). Re-run the default `cal` before a production fit if the K\* matters to
within a factor of two.
