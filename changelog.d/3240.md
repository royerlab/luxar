#### Fits stop once the loss stops improving meaningfully; `cal` says when a sweep sits at the noise ceiling

A fit used to stop early only after a full `early_stop_patience` window with no
new best loss, so a fit whose loss was still creeping down by a few parts per
million ran out its whole iteration budget. On an Allen Institute hiPSC DNA
stack one `gsplat cal` point held relL2 at 0.4860 for about 9,600 iterations.
`fit_gaussian_splats` now takes `early_stop_rel_tol`: the fit also stops once
the best loss improved by less than that fraction over the patience window.
Every CLI preset sets it to 0.001, so `fit`, `cal` and the tiled and batch
workers use it; the Python default stays off. On four volumes (Allen hiPSC,
IDR 5514375, a Tribolium light-sheet crop, a FlyLight neuron crop) `cal` got
up to 7.3x faster (Allen: 2510 s to 344 s) and `fit` up to 3.2x. K\* under
the default `psnr_foreground` criterion was unchanged beyond run-to-run noise
(on Allen's flat curve the `psnr_minmax` K\* moved), and foreground PSNR losses
were at most 0.10 dB. `early_stop_rel_tol: null` in a `--config` restores the
old rule. The paper's Noise2Self tables use the Python API and are unchanged;
to reproduce the CLI recipe timings in the consumer-GPU and tiled-fitting
supplementary documents, set `early_stop_rel_tol: null`. A `batch-fit` run
planned before this change and resumed after it fits its remaining tiles under
the new rule, because the manifest stores only the preset name; the quality
difference stays within the spread of plain-rule replicates, but its stored time
estimates now run high. A tolerance written as
`1e-3` in YAML is read as text; the error now says to write `0.001`.
Each fit records why it stopped (`early_stop_reason`: plain, relative or
converged) and the tolerance (`early_stop_rel_tol`) in its stats.

When the noise estimate passes its scale check and the best held-out score is
within 0.3 dB of the RMS noise ceiling, the `cal` summary adds a line saying
that more K mostly fits noise on the held-out voxels, and that foreground
detail can still improve.
