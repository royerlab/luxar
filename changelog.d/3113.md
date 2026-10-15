#### Faster, quieter-failing and resumable Gaussian-splat fits on MPS and CUDA

A content-tiled fit on Apple Silicon could look hung: it was slow, silent, and
several steps scaled badly on MPS. `otsu_threshold` no longer runs a full-volume
`torch.histc` on the fit device (on MPS that allocated about numel x bins x 4 bytes,
9.4 GB for an 8 Mvox crop, and failed outright above ~16 Mvox). It now counts a
CPU `bincount` over at most 5 M fixed-seed random samples, with bin edges still
taken from the full range. The calibration Otsu helper used a strided subsample
that could alias with the fastest axis, so a structured volume could read a
threshold near zero; it now uses the same random sample. The post-fit quality render on MPS uses the Metal splatting kernels
the fit trains with, instead of the pure-PyTorch renderer. The PyTorch renderer's
chunk sizing now uses a measured 76 B per (splat, voxel) element (it assumed 48 B)
and a budget taken from the device's reported free memory. It also splits large
groups of same-sized splats, so one chunk can no longer jump to 131,072 voxels.

The fit's best-state snapshot is now taken before `optimizer.step()`, so the
restored "best" parameters are the ones that produced the best loss. They used
to be one step ahead of it. A warm start (`seeds=` an existing fit) whose steps
only made things worse therefore returned the post-first-step parameters: a
synthetic refit at `lr=0.05` came back at 24.1 dB PSNR instead of the seed's
59.7 dB, and an A549 refit lost 8 dB of foreground PSNR. Cold fits that keep
improving are unaffected. The parameters left by the final step are now scored
once after the loop and compete with the best snapshot; before, they were never
considered. On a convergence stop they are returned outright, so
`converged=True` now always means the returned splats meet the threshold.

The fit loop no longer forces a GPU-to-host sync every iteration. The best loss and
best-state snapshot are tracked on the device and read back on eval iterations.
`ReduceLROnPlateau` receives every iteration's loss from a device-side buffer,
replayed every few iterations (5 at the default patience), so a learning-rate
reduction can take effect up to four iterations later than it used to.
Tiled residual peak finding searches all tiles in one batched pass instead of
about 2,500 blocking syncs per relocation step, and selects the same peaks.
With `num_tiles_per_dim=None`, a 3D fit now follows the documented auto rule
(6 tiles per axis, 216 tiles instead of 512), which was measured to keep fit
quality. Other dimensionalities keep the historical 8 tiles per axis inside the
fit loop.

A failed post-fit quality measurement now prints one warning line even when the
fit is not verbose. The reason is stored as `quality_error` in the store's
`fitting/` group, so a missing PSNR is never silent.

Content-tiled fits (`gsplat fit --tiling content`, sequential and `-j N`) now
checkpoint every finished box to `<output>.boxes/`. Rerunning the same command
reuses each box whose marker matches the run's plan, fit configuration and input,
and refits the rest. The `-j N` staging directory is no longer wiped at start.
A marked box whose store no longer reads is refitted. SIGTERM now unwinds the fit
like Ctrl-C, so `kill` leaves finished boxes on disk. On an interrupt, `-j N` also
terminates its running worker processes before releasing the directory. The
directory is removed only after the merged output is saved. A second live fit to
the same output uses a private directory instead of sharing boxes; the message
names the lock file, and `LUXAR_CHECKPOINT_FORCE_UNLOCK=1` breaks a dead lock.
Library callers of `fit_planned_parallel` that pass no `fingerprint` still get
a cleared staging directory.
