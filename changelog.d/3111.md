#### `gsplat cal`: grid-edge warnings, bounded memory, and faster seeding

`gsplat cal` now says when its sweep did not bracket the optimum. When K* is the
smallest or largest grid point with a finite result, or when the held-out metric moves less than 1 dB
across the whole grid, a ⚠ line names the flag to extend (`--k-min`, `--k-max`,
`--k-grid`). The line prints even under `--quiet`, and the text is recorded in
`cal.json` as `held_out_peak.warnings`. When the table already says the curve
has not converged at K_max, it skips the duplicate largest-grid-point line. A curve that is best at the first K and
then falls by at least 0.3 dB is now labelled `low_k_optimal`. It used to be
called `plateau`, but a falling curve has no plateau. Older `cal.json` files still
load, with an empty warning list.

Per-K scoring no longer copies the volume around. The render is overwritten in
place with its squared error, only the 5 % held-out set is gathered, and the
train MSE is the total error minus the held-out part. Before, a boolean gather
built `train_pred`/`train_true`, about twice the volume. SSIM now filters with
a separable Gaussian, one 1-D pass per axis, on every device; it matches the
former n-D kernel to within 4e-7. The former CPU `conv3d` lowered the 11³
kernel to im2col, about 5 KB of temporaries per voxel: 73 GB for a 16.8 Mvox
volume, and the reason a 64 Mvox `cal` died on a 16 GB machine. A 200³ SSIM on
CPU now peaks at 0.24 GB in 0.23 s; the reported untiled cost was 37 GB and
79 s. On CPU and MPS, SSIM also tiles once its untiled temporaries would pass
256 MiB; before, it tiled only on CUDA. The whole scoring step on a 65 Mvox
volume now peaks at 0.34 GB on CPU and takes 2.3 s single-threaded. The old
code was OOM-killed there on a 128 GB Mac. On MPS, SSIM on that volume takes
0.8 s untiled or 1.5 s tiled, against 21–25 s with `conv3d`.

Each grid point also prints one progress line with K, held-out and train PSNR,
the fit time and the sweep's running total.

Seed generation drops three superlinear paths. The selected seeds do not change,
and tests check them against the old algorithms. The grid top-up filtered its
grid with a k-nearest-neighbour query. That query fell back to brute force over
every seed for each grid point far from all of them, which is the normal case
when the seeds sit in a few clusters inside an empty volume. It took 29 s for a
32k target on a 9 Mvox volume and grew as queries times seeds. It is now a
bounded KD-tree range test: 0.07 s at that size and 0.37 s for 128k on 37 Mvox.
Greedy minimum-distance deduplication and edge Poisson-disk sampling now resolve
blocks of candidates with KD-tree range tests instead of looping in Python per
candidate, 3 to 12 times faster. Before each block they estimate how densely its candidates conflict from a
small sample, then cap how many candidates one pass resolves. Past about 512
conflicts per candidate (a large minimum distance on dense candidates) they
hand over to the per-point loop. On a 64³ lattice at `min_distance` 8 or 16, an
unshuffled or intensity-ordered input now runs at the old loop's speed (4 s,
under 25 MB on Linux). A fixed block size had needed 4.5 GB there. The public
`greedy_min_distance_select(..., max_count=0)` returns no points; the old loop
always kept the first one. Farthest-first subsampling for targets under
10k kept recomputing the full remaining-by-selected distance matrix, which cost
O(N·T²). It now keeps a running nearest distance, O(N·T): 140 s became 0.46 s for
5,000 of 10,000 candidates. It now always runs this NumPy CPU path. CUDA hosts
used to run a `torch.cdist` branch for targets over 1,000, so their seed order
changes and near-ties can resolve differently.
