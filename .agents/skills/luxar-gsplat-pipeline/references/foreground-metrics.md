# What `fit`'s and `compare`'s foreground metrics mean

Loaded from the `luxar-gsplat-pipeline` skill ("Scoring a fit").

`compare` reports `mse`, `psnr_db`, `psnr_robust_db`, `ssim`, `ssim_robust`,
`foreground_ssim`, `foreground_ssim_robust`, `rel_l2`, `max_abs_error` **and the same foreground trio**
(`foreground_psnr_db`/`foreground_psnr_robust_db`, `foreground_threshold`,
`foreground_fraction`). **Read the foreground number** — and see "Traps that cost
real time" in the skill for what to score it *against*. Four things about these
numbers worth knowing:

- The **headline PSNR uses a robust peak**: the reference's 99.99th percentile
  minus its minimum (on the fit's basis, the floor), so one hot pixel cannot set
  the score. The classical full-range (`max - min`) figure is reported beside it
  and stamped as `psnr_db` (stores without `psnr_robust_db` carry only this one);
  `--output-json` names the headline in `psnr_headline`.
- The **headline SSIM uses the same robust peak** (`ssim_robust`,
  `foreground_ssim_robust`; `ssim_headline` names it). `ssim` / `foreground_ssim`
  stay on the full range. SSIM's constants scale with the range, so a full range
  set by one hot pixel pushes it towards 1 for almost any fit: on one stack with
  max 2002 and p99.99 524, whole-volume SSIM read 0.95 on the full range and
  about 0.20 on the robust range.
- The **foreground defaults to a smoothed Otsu mask** on the target: one light
  blur, then Otsu over a histogram capped at the robust peak (the definition
  `fit` stamps too; the blurred field is the one `cal` scores with). `--mask <file>` or `--fg-threshold <value>` (raw
  reference units) override it.
- Foreground is defined on the **target**, never on the prediction — a fit that
  hallucinates signal is scored against where the signal actually is.
- Both peaks come from the **whole** target, not from the foreground subset, so
  the figure is not inflated by a shrunken reference. That convention is all it
  shares with `cal.json`'s `held_out_psnr_fg_db`, which scores only the held-out ∩
  foreground voxels of a fit of the *masked* volume — the two dB numbers are not
  interchangeable. Always report `foreground_fraction` beside the dB: a
  PSNR over 0.01% of a volume means something very different from one over 40%.

Neither the global nor the foreground PSNR covers the **dim band** (1–10% of
max) — mask it yourself with `gsplats.rendering.render_to_volume_tensor` plus
`metrics.compute_psnr` (both torch; `render_to_volume` returns NumPy, which `compute_psnr` rejects).

## Reading `compare`'s noise verdict

The verdict reports the gap between the foreground PSNR and an *estimated* noise
ceiling, and what that gap may mean; it is not a pass mark.

- "at" / "near the noise floor by this estimate": assumes the finest in-plane
  band is noise. Real texture at that scale (chromatin, fine processes) counts
  as noise too and lowers the ceiling, so a fit that smoothed it away can still
  read as close. Confirm on slices.
- "undetermined: N dB above the estimated ceiling": the residual is smaller than
  the estimated noise by more than the estimate's own error explains (over 3 dB;
  1-3 dB above reads "ambiguous"). Either the model is reproducing noise (too
  many splats; `cal` tests this on held-out voxels) or the reference has fine
  structure the estimate counts as noise.
- "undetermined: the noise estimate failed its scale check": no ceiling is trusted
  (voxel-scale structure, or correlated noise from a deconvolved, upsampled or
  downscaled reference).

`cal`'s `Noise floor` / `PSNR ceiling` is a different, whole-volume estimate on the
full-range peak and can differ from `compare`'s ceiling by tens of dB.
