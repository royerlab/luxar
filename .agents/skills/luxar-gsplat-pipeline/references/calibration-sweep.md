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
