#### The zebrafish timelapse is denoised before it is fitted

A third of the early frames' energy in this confocal recording is shot noise —
at t=0, 16,200 of its non-zero voxels are isolated single voxels against 6,503
in real cells — and the fit was spending its budget on it, reproducing 67% of
that noise energy splat by splat. Each timepoint is now non-local-means
filtered first, at a strength of 0.05.

That strength is measured, not calibrated at runtime. Scored against the raw
frames and split into what filtering risks (fidelity inside real connected
components) and what it buys (the share of shot-noise energy the fit still
reproduces), 0.05 is the knee: cell fidelity is break-even to +0.5 dB against
no filtering at all while the reproduced noise collapses by 12-30x. By 0.08 the
filter has reached the cells at the later timepoints.

The library's own Noise2Self routine answers between 0.055 and 0.225 here
depending on which slice it is pointed at, and every one of those is at or past
that knee. It is defeated by the same sparsity that inverts the `auto`
background floor on this dataset: on a volume that is 98.7% exact zeros a
held-out voxel is best predicted by predicting zero, so maximal smoothing
always wins its cross-validation.

The seed budget is unchanged — 32,000 is still the plateau on denoised data —
but the same budget now delivers far more splats (6,926 to 23,514 on frame 0),
because the 0.9999 amplitude retention has to be met out of cells once the
spikes carrying that amplitude are gone.

The demo's module docstring records the method behind all of this, not just the
numbers: what to score on sparse data, why a preprocessed fit must never be
scored against its own preprocessed input, and how to recognise an automatic
calibration that has pinned at the edge of its own grid.
