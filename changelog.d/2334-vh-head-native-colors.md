#### Visible Human head: the colours now live inside the fit

The hosted `vh_head` archive carries its per-splat colours as a native gsplat
attribute instead of leaning on a positional `.npz` sidecar. That is not a
tidy-up: the sidecar had no positions, so it was only ever correct while its row
order happened to match the store's, and it fell out of order twice — once when
it was sampled before the writer's spatial reorder (#1670), and again when the
archive was restructured underneath it (#2334), which left the shipped pair
agreeing on 0.7% of splats. A native attribute is permuted in lockstep with the
centers, so the two cannot separate, and `_colors_match_fit` has nothing left to
check on that path — the guard needs two arrays and there is now one.

Getting there needed the sampling order to stay as it was, for a reason that has
nothing to do with row order. `sample_colors` rounds to the nearest voxel and the
writer quantizes centers, so a splat sitting within a quantization step of a
rounding boundary changes voxel across a save. Sampling at the in-memory centers
and attaching in a single pass mislabels exactly those splats — 2 of 6000 on the
unit fixture, every one within 6.2e-05 of a boundary against a 1.2e-04
quantization step. Invisible on screen, but it drops a fresh pair from agreement
1.000 to 0.9993, and `MIN_COLOR_AGREEMENT` is calibrated against an exact 1.000
baseline. So a refit still saves, samples at the stored centers, attaches, and
saves again; the colours travel with the splats through the second save.

The sidecar does not disappear yet. The in-repo Git-LFS payload is a different,
colourless generation of this fit (1,911,192 splats against the hosted 1,908,888)
and is still what a checkout resolves first, so the demo prefers native colours
and falls back to the sidecar when a fit has none. Both retire together with
those payloads (#2354). The hosted sidecar was re-exported from the new
archive's own colours, which is what let the positional pair move atomically as
the manifest requires — and makes it aligned by construction rather than by a
measurement that happened to pass.
