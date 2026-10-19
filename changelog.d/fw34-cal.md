#### `fit` no longer starts most seeds dead on a background pedestal; `cal` quotes an honest noise ceiling

Seeding samples 90% of the voxel value so overlapping seeds start below their
target. That factor was applied to the raw value, pedestal included, and the
floor was subtracted afterwards, so on data with a camera offset or background
pedestal most seeds began at exactly zero amplitude, never recovered, and were
removed by the post-fit cull. On an Allen Institute hiPSC DNA stack
(pedestal 405 counts, robust peak 119) `fit --seeds 16000` stored 1,829 splats
while `gsplat cal` promised about 13,100, because `cal` subtracts the floor
before fitting and never hit the problem. The factor now applies to the signal
above the pedestal, so `fit` and `cal` start their seeds the same way. The
same holds for the explicit-seeding workflow,
`fit_gaussian_splats(V, seeds=generate_seeds(V))`; a warm start from a previous
fit's output is unchanged. Fits of data with a background pedestal (the default
`--floor auto` finds one on most microscopy) now store more splats and fit
better, so refitting an existing store gives a different result.

The `gsplat cal` summary used to quote a noise-floor PSNR ceiling from a
median-voxel estimate on the full intensity range, next to a table on the
robust peak. Under shot noise the median reads the quiet background, so the
ceiling sat 4 to 31 dB above the best held-out score. The summary now reports
the RMS noise of the floor-subtracted volume (the intensity-binned estimator
`compare` uses) and the held-out PSNR ceiling it implies on the table's
robust-peak basis. On the two tester volumes the best held-out score sits
within about 0.1 dB of it (one curve is still climbing; on the other the
estimator's own scale check flags fine structure, and the summary says the
ceiling may read low). The median estimate stays in `cal.json`
as `noise_floor`, and the new `noise_sigma_rms`, `noise_scale_ratio` and
`psnr_ceiling_robust_db` fields sit beside it.

The `--pdf` report now plots every PSNR on the same robust peak as the
terminal and labels its axes with the basis. `--pdf` without `--keep-fits`
keeps the per-K fits in a temporary directory beside `cal.json` so the slice
page renders, and says so before the sweep; the slice page's target is floor-subtracted like the
fits. A long grid point prints an in-fit progress line at most every 30
seconds instead of staying silent until it ends.
