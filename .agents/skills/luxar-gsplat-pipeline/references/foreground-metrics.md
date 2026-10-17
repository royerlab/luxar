# What `fit`'s and `compare`'s foreground metrics mean

Loaded from the `luxar-gsplat-pipeline` skill ("Scoring a fit").

`compare` reports `mse`, `psnr_db`, `ssim`, `rel_l2`, `max_abs_error` **and the
same foreground trio**. **Read the foreground number** — and see "Traps that cost
real time" in the skill for what to score it *against*. Three things about that trio
worth knowing:

- The **threshold defaults to Otsu** on the target, not a fixed fraction of max.
- Foreground is defined on the **target**, never on the prediction — a fit that
  hallucinates signal is scored against where the signal actually is.
- `data_range` comes from the **whole** target, not from the foreground subset, so
  the figure is not inflated by a shrunken reference. That convention is all it
  shares with `cal.json`'s `held_out_psnr_fg_db`, which scores only the held-out ∩
  foreground voxels of a fit of the *masked* volume — the two dB numbers are not
  interchangeable. Always report `foreground_fraction` beside the dB: a
  PSNR over 0.01% of a volume means something very different from one over 40%.

Neither the global nor the foreground PSNR covers the **dim band** (1–10% of
max) — mask it yourself with `gsplats.rendering.render_to_volume_tensor` plus
`metrics.compute_psnr` (both torch; `render_to_volume` returns NumPy, which `compute_psnr` rejects).
