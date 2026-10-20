#### `cal` flags sparse puncta it under-seeds; auto worker sizing understands unified memory; batch tiles are kept as they finish

On sparse, diffraction-limited puncta (FISH spots) the held-out PSNR that picks
K* is set by the background, because the spot centres are a tiny share of the
held-out voxels. The curve comes out flat and K* lands well below what the
spots need. `gsplat cal` now detects spots on the floor-subtracted volume and
records, per K, the median ratio of the rendered peak to the reference peak at
those spots (`spot_count` and `spot_peak_ratio` in `cal.json`). The reference
peak carries its own noise, so even a perfect render of a dim spot scores below
1; every grid point shares that bias, so `cal` compares K* with the best point
of the same sweep. When a larger K rendered the spots better and K* reached
less than 95% of that best, `cal` says so and names the first grid K within 98%
of it. Under `--region` or `--auto-region` the suggestion is restated as a
whole-volume `--seeds` with the same density transfer as K*, or as the
`fit --tiling content` density flags when no whole-volume figure exists. On
EASI-FISH level-0 crops K* reached 27%, 78% and 93% of the best on the three
sparse crops and 99% on a dense one. On an Allen hiPSC DNA crop the chromatin
maxima reached 91%, so the warning can also fire on textured nuclei. K* itself
is chosen exactly as before.

`batch-fit run --jobs-per-gpu auto` and `fit -j auto` checked GPU memory and
host RAM as two separate pools. On a CUDA integrated GPU that shares host memory,
such as a GB10 or a Jetson, they are one pool, and the device's free-memory
figure leaves out page cache, so the worker count swung with cache state; on
the EASI-FISH run it picked 8 workers and left 3 GB free. On such a device the
workers are now sized from `MemAvailable` at about 3 GB plus 64 bytes per task
voxel each, measured on a GB10 at 2.8 GB plus 62 bytes per voxel. Discrete
GPUs and Apple MPS are sized as before.

`batch-fit run` used to promote every finished tile to its final name only
after the last task returned, so an interrupted run kept nothing and
`batch-fit status` read zero while tiles sat finished in staging. Each tile is
now promoted as soon as its worker exits.

`fit` now warns when `--seeds` is left on `auto`, which sizes 100 to 10,000
seeds from the volume's linear extent rather than its content. The `--min-extent`
documentation records what speck pruning costs on puncta: with seeds starting
alive on a pedestal, it kept spot recall within 0.01 of an unpruned fit.
