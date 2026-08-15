#### A fit records the volume its splats represent

A fitted `.gsplats.zarr` recorded its own size but nothing about what it was a
representation *of*, which made the obvious question — how much did this compress?
— unanswerable from the artifact. It was not answerable from the producing script
either: the fitted grid is derived at run time from downscale factors and from
isotropic resampling of the voxel spacing, so it is not a constant anyone can read
off. Across the whole bundled demo corpus, not one dataset could state its own
compression ratio.

`gsplat fit` now stamps the source grid into the `fitting/` group: `source_shape`,
`source_dtype`, `source_voxels`, `source_bytes`, `fitted_shape`, `fitted_voxels`,
`occupancy` and `voxels_per_splat`.

Two details are what make the numbers honest rather than merely present.

The source dtype is captured **before** the volume is cast to float32. A uint16
acquisition doubles in size under that cast, so quoting the cast size would
overstate compression by exactly 2x. There are two casts to get in front of, not
one: the fitter's own, and the earlier one in the CLI's volume loader — which
returns float32 whatever the file holds, and so is the only place the stored
element type still exists. `luxar gsplat fit` reads it there and hands it down,
so a fit of a 16-bit stack records 16-bit source bytes on every path.

The source grid and the fitted grid are kept as separate fields. When a demo
downscales before fitting, they differ, and collapsing them would overstate
compression by the downscale factor cubed. `fitted_*` is what the optimiser saw;
`source_*` is what was handed in.

`occupancy` is the fraction of fitted voxels above the subtracted background floor.
Sparse microscopy volumes are routinely more than 99% empty, and a compression
ratio means something quite different at 0.03% occupancy than at 50%, so the two
belong together.
