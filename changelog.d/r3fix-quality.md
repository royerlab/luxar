#### `gsplat compare` scores a crop fit on its region and says where the noise floor is

A `--region` crop fit compared against its full source volume is now scored on the
region it fitted. Its foreground mask used to be an Otsu threshold of the WHOLE
volume, so the "same" foreground PSNR moved by up to 6 dB between the fit-time
stamp and `compare` (31.5 vs 25.5 dB on one organoid membrane fit), and the splats
filling only their region set off a false "crammed into a corner?" warning.
Outside the region the fit saw nothing, so scoring there counts every structure it
was never asked to model as error; `--whole-volume` still does that, and the corner
heuristic now measures the splats against the fitted region. A full-volume `--mask`
is cropped with the reference, and the JSON records `scored_region`.

`compare` also answers "is ~20 dB foreground PSNR good?". Beside the foreground
PSNR it prints the score a noise-free model would get against the same noisy
reference, `20 log10(peak / sigma)`, with sigma the RMS noise measured in the
foreground (shot noise is larger there than in the background), and a verdict:
at the noise floor (within 1 dB below), near it, or signal left to fit. A score
above the ceiling is never called definitively: the model may be reproducing
noise, or the reference may have sub-voxel structure the estimate counts as
noise, so 1-3 dB above reads "ambiguous" and more than 3 dB "undetermined". A
scale check reports the ceiling as unreliable, with no verdict, when the finest
band is not white (voxel-scale structure or correlated noise); by design that
withholds the verdict on many clean, high-SNR volumes. On the Cell Tracking Challenge A549 `t000` the
ceiling is 20.9 dB, against 21.0 dB for the raw data scored against a lightly
blurred copy of itself; a K=1000 fit scores 20.0 dB (at the floor) and a K=10000
fit 22.9 dB, 2.0 dB above (ambiguous). The JSON carries `noise_sigma`,
`noise_ceiling_reliable`, `foreground_psnr_noise_ceiling_db`,
`model_error_over_noise_sigma` and `noise_verdict`. The compression ratio of a
crop fit is now taken against its region, matching the fit's stamp.

#### `gsplat lod --recipe stream` no longer takes minutes on a few thousand splats

The additive `auto` method orders up to 5,000 splats with the provably
(1-1/e)-optimal greedy. Its cost was never the greedy pass but the sparse Gram it
needs, whose pairs went through a Python loop at ~20 us each: 2.1 minutes for
4,786 densely overlapping splats on one tester's CPU. The pairs are now scored in
vectorized batches with the same closed form, giving the same matrix (to ~1e-16)
and the same order: 4,800 splats with ~100 neighbours each went from 14.8 s to
0.6 s on a laptop CPU, and ~1.3 s at ~440 neighbours. The 5,000 threshold is
unchanged; the lod README records the measurements.

#### `auto_exposure_attrs`: convert's auto exposure for scenes built in Python

`luxar gsplat convert` dims a dense fit so it does not open as a saturated white
blob under `additive` blending; the Python adders normalise each splat but do not
account for splats summing along a ray, so a multichannel scene built with
`add_gsplats_from_file` per channel opened white. The new
`luxar.gsplats.exposure.auto_exposure_attrs(path, dim_order=..., colormap=...)`
runs convert's prediction and returns the adder kwargs convert would use
(`normalize_amplitudes=False`, the dimmed `display_gain`, the unit labels; pass
the store path, which also reads the unit labels' stats and takes tiled fits), so
`scene.add_gsplats_from_file(name, path, ..., **auto_exposure_attrs(path, ...))`
opens the way `convert` does. The adders' defaults are unchanged.
