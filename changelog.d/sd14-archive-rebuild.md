#### Time-lapse demo archives rebuilt without post-fit culls, with gate-clean ladders

The three 4D Gaussian-splat demos (Drosophila embryogenesis, zebrafish h2afva
51-timepoint variant, neuromast two-channel) shipped archives that scored far below
fresh fits of the same frames. The Drosophila archive was the fit filtered at a fixed
amplitude cutoff (65 % of the splats removed, 5-8 dB below the fit); the neuromast
archive was redundancy-culled at 0.20, which removed bright nuclear splats as
"redundant" with the background beneath them. Both culls are gone: the uncelled
merge is the recipe, and the recorded splat counts follow (128,000,000 and
6,400,000 per channel).

The neuromast recipe fits the raw assembled channels with `--floor none`; the
measured camera pedestals stay in `CHANNELS` as provenance. Fitting the exactly
pedestal-subtracted, zero-clipped frames keeps almost none of the low-contrast
background modulation (very-low band correlation 0.16 against the frame, a third
to two thirds of the seeds dead) while the raw fit keeps it (0.55), and the
amplitude sparsity weight is not the mechanism.

The sliced archives are laddered with four equal-count rungs (`--merge-n-lods 4`,
`gsplat lod --n-lods 4`) instead of a 200 ms target-ms budget: sized against the
whole node that budget starved rung 0 to a handful of splats per timepoint, and
sized per slice it left rung 0 under 10 % of the node, so `check-demo-ladders`
refused both. Neuromast pins its recorded eight rungs explicitly, because
`batch-fit run`'s merge defaults to four.
