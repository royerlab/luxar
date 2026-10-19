#### SSIM is headlined on the robust peak, and the quality verdicts say what they measured

`gsplat compare`, the fit summary, the merged-tile summary and `cal` computed SSIM
on the full `max - floor` range, while PSNR already headlined a robust peak
(reference p99.99 minus the floor). SSIM needs the robust peak more than PSNR
does: its stabilising constants scale with the range, so a full range set by one
hot pixel swamps the real local variance and pushes SSIM towards 1 for almost any
reconstruction. On a nuclear-stain stack with maximum 2002 and p99.99 524, one fit
read whole-volume SSIM 0.95 on the full range and about 0.20 on the robust range. The
headline SSIM, whole and foreground, now uses the same robust peak as the PSNR
headline, and the full-range figure is printed beside it, labelled.

The JSON and stored keys are additive. `compare --output-json` gains
`ssim_robust` and `foreground_ssim_robust`, and `ssim_headline` names the
headline as `psnr_headline` already did. `ssim` and `foreground_ssim` keep their
full-range meaning. Fitted stores stamp `ssim_robust` beside `ssim`, and `cal.json`
gains `full_ssim_robust` beside `full_ssim`. The `cal` table's SSIM column now shows
the robust figure (an older `cal.json` falls back to `full_ssim`). Computing both
costs a second SSIM pass whenever the two peaks differ.

`compare`'s noise verdict now states what was measured and what it may mean. "At"
and "near the noise floor" carry "by this estimate" and a caveat: the noise sigma
comes from the finest in-plane band, so real texture at that scale counts as
noise and can make a smoothed fit look close to the ceiling. A score more than
3 dB above the ceiling is "undetermined" with the reason spelled out. The
residual is then smaller than the estimated noise, which means either the model
is reproducing noise or the reference has fine structure the estimate counts as
noise. The verdict no longer calls the second reading the likely one. An
unreliable ceiling now says that the noise estimate failed its scale check. The
CLI README and the gsplat skills explain how to read the verdict, and note that
`cal`'s noise floor is a different, whole-volume estimate.

`fit --denoise` scores its summary against the denoised input and now says so,
pointing to `gsplat compare` for a score against the raw data. On the tiled
paths it says that a tile's own line is scored against its denoised tile and the
merged line against the raw input. `gsplat render` prints the level the fit
subtracted (`image_min`), which the background-relative output leaves out and
which must be added back before comparing with the raw volume.
