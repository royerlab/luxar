#### Three embryo demos built on data we own outright

Four of the existing demo datasets cannot be redistributed — two carry no data
license at all, one has a licensing conflict with its origin, and one is
NonCommercial — so they must eventually ship as fetch-the-raw-source-and-fit
demos. These three are the opposite case: the source data is ours (or the
collaborators' with permission), so the fitted products can be hosted and
downloaded directly.

**Drosophila gastrulation** — one SiMView light-sheet stack (Royer/Keller,
`His2Av::mRFP1`) caught at gastrulation, with the cephalic furrow and posterior
midgut invagination both visible. 200,155 splats at 39.9 dB / 0.911 SSIM,
calibrated by a Noise2Self K* sweep and fitted at the sweep's own
diminishing-returns point.

The stack carries no axial calibration, so the z-step was recovered from the
embryo's geometry — a prolate embryo's mid-length cross-section must be circular,
which puts the anisotropy at 4.76x — and corroborated two independent ways: the
instrument (Nikon 16x/0.8 detection on a 6.5 um sCMOS = 0.40625 um laterally,
hence z = 1.93 um, the standard ~2 um sampling) and the result (521 x 190 um,
against a textbook 500 x 180 um embryo). The demo is in physical microns.

**h2afva zebrafish stack** — one timepoint of the zebrahub recording as 1.65M
splats across 41 content-tiled parts, each with its own streaming ladder. A
second, toggleable layer draws one coloured wireframe box per part: these are the
content bounds the viewer actually frustum-culls against. They are disjoint — 0
of the 820 part pairs intersect — because the fit's overlap margin is a
fitting-time apodisation and each part's splats are cropped back to its core;
they only read as interleaved because 41 nested 3D slabs project to 2D.

**Decimation study** — the same embryo at four measured detail levels (100% /
25% / 10% / 5%) on a categorical axis, so picking an entry swaps the level in
place at a fixed camera, which is the only way to perceive a 3 dB difference.
Each entry is labelled with its splat count and its measured FOREGROUND PSNR, and
each is a single flat leaf, so what renders is exactly what the label says.

#### Four demo datasets we may not redistribute are no longer shipped

`gsplats_tribolium`, `gsplats_acto3d_heart`, `gsplats_tng_cosmic_web` and
`milky_way_gaia_3m` are marked `local-compute` in the data manifest — their
licenses do not permit redistributing even the derived product — so their
in-repo Git-LFS copies have been removed. The demos that used them now take
their own fetch-the-raw-source-and-refit path instead of pointing you at a
`git lfs pull` for a file that is no longer there, and their catalog entries say
so (a GPU is needed for the refit; the Gaia catalog has no build path yet, so it
reads a hand-placed file from `~/.cache/luxar/` until #1575 lands). A cached
copy from an earlier run is still used untouched.
