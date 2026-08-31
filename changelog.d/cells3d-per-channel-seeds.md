#### The cells3d demo declares the seed budget its archives were fitted at

`demo_gsplats_3d_cells3d_multichannel` fitted both channels from one shared
`MAX_SPLATS = 25000`, but the archives it ships were not produced that way.
`scripts/demo_archive_characteristics.json` records 41,975 splats for ch0 and
57,946 for ch1, measured on the `hosted_sha256` the demo pins — roughly twice and
four times what a shared 25,000 yields. Anyone running `--recompute` got about
half the splats of the data everyone else sees, and the file gave no hint of it.

The budget is per channel now, 50,000 and 100,000, which is what those archives
were fitted at. Nothing about the default path changes: the shipped archives are
already these fits, a local refit still lands in the demo's own local-fit
namespace, and no pin moves.

The two numbers differ because a blind-spot calibration sweep put ch0's held-out
foreground PSNR on a plateau at roughly twice the old budget while ch1 was still
climbing at four times it. The comment above `CHANNELS` now carries what the bump
measured on the archive — tenths of a dB, not the 1.5–1.7 the sweep predicted —
because that gap is the point: held-out PSNR over 5% masked voxels is a noisier
estimator than it looks and mispredicted this in both directions. The bump is
kept on matched-zoom renders, not on the metric.
