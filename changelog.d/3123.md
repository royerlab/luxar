#### `gsplat compare` and `render` score and render on the right grid

`luxar gsplat compare` and `luxar gsplat render` used to compare physical-unit
splats against a voxel-index grid. A fit given a `voxel_size` multiplies its
centers and Cholesky rows by that spacing, so its splats landed in a corner of the
reference and scored as a failed fit. Fits now record the spacing they applied as
`fitting/voxel_size` (single, progressive and tiled fits). `compare` and `render`
divide it back out, and `--voxel-size` covers stores fitted before the stamp.
Both also honour a `physical_coordinates: false` stamp and a `--region`
crop fit's `source_region` (rendering and scoring on the crop's grid).
A timelapse batch-fit merge carries the spatial stamp, and readers pad its
stacked column. `compare` and `render` warn when most splat centers miss the
grid, or are crammed into a corner of it. A scale or rotation, from the CLI
`gsplat transform` or from `GSplatData.transform`, drops the stamp; a
translation keeps it. Without `--shape`,
`render` now uses the fit's own source grid (`fitting/source_shape`), so its
output lines up voxel for voxel with the fitted volume. Stacked stores, whose
`source_shape` lists time first, are excluded. Other stores render on
the bounding box of their centers. That box used to start at index 0 and clipped
any store whose splats did not start there; it now starts at the lowest center,
and the origin is printed.

The headline PSNR now uses a robust peak: the reference's 99.99th percentile
minus the floor, the minimum of the fit-basis reference. One hot pixel used to
set the classical `max - min` peak, and with it the score. The full-range figure
is still reported and labelled, in `compare` and in the fit summary. Stored fits
keep `psnr_db` as the full-range figure and add `psnr_robust_db` and
`foreground_psnr_robust_db`. The `compare --output-json` record carries both
values with explicit keys (`psnr_headline` names the headline) and both peaks.
`compare` also reports a foreground SSIM. The foreground, in `compare` and in
the scores a fit stores, is now one definition: a smoothed Otsu mask, as
calibration uses, with the histogram capped at the reference's 99.99th
percentile. Uncapped, one hot pixel pushed every real structure below the
threshold and reported a 0.00% foreground. In `compare`, `--mask <file>` and
`--fg-threshold <value>` (in the reference's raw units) override it.
`--at-fit-resolution` scores a `--downscale` fit on the grid it was fitted on: the
reference is decimated the same way the fit decimated its input.

`compare` gains `--axes` and `--array-key`, matching `fit`. A `--channel` or
`--timepoint` given for a TIFF or NumPy input without `--axes` is now an error in
`fit`, `cal` and `compare`. Before, the index was ignored without a word and every
channel was fitted as extra z-planes. `compare` streams its full-volume reductions
and tiles SSIM on every device. A 256x512x512 float32 comparison on CPU peaked at
3.2 GB max RSS in our measurement on macOS. `GSplatData.load()` without `include_stats` keeps the
normalization basis (`image_min`, `floor`, ...), so a store loaded, edited and
saved in Python is still scored against the right basis.
