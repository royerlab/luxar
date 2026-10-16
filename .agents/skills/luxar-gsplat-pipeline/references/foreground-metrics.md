# What `fit`'s and `compare`'s foreground metrics mean

Loaded from the `luxar-gsplat-pipeline` skill ("Scoring a fit").

`compare` reports `mse`, `psnr_db`, `psnr_robust_db`, `ssim`, `foreground_ssim`,
`rel_l2`, `max_abs_error` **and the same foreground trio**
(`foreground_psnr_db`/`foreground_psnr_robust_db`, `foreground_threshold`,
`foreground_fraction`). **Read the foreground number** — and see "Traps that cost
real time" in the skill for what to score it *against*. Four things about these
numbers worth knowing:

- The **headline PSNR uses a robust peak**: the reference's 99.99th percentile
  minus its minimum (on the fit's basis, the floor), so one hot pixel cannot set
  the score. The classical full-range (`max - min`) figure is reported beside it
  and stamped as `psnr_db` (stores without `psnr_robust_db` carry only this one);
  `--output-json` names the headline in `psnr_headline`.
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
