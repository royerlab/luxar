# Several fluorescence channels in one scene: exposure is per layer and authored

Loaded from the `luxar-visualization` skill ("Appearance is authored, not
defaulted"). Measured on two bundled demos:
`demo_gsplats_3d_easifish_hypothalamus.py` (DAPI plus three mRNA channels, a
~250 µm slab) and `demo_gsplats_4d_organoid_budding.py` (a membrane and a nuclear layer,
an organoid that thickens over the time-lapse).

## Per-node exposure is not composite-aware

Every default exposes each layer on its own: the Python adders scale a robust
p99.9 to 1.0, and `gsplat convert`'s auto `display_gain` targets one node's
opening view. Neither knows the other layers are adding light along the same
rays. Four EASI-FISH layers summed through the slab to a beige-white in which the
genes could not be told apart, under both defaults.

Author each layer's brightness instead. EASI-FISH passes a per-channel target to
`normalize_amplitudes=` (radiance is linear in amplitude and sums along the ray,
so this is the exposure knob), at `opacity=1.0` under ACES:

| layer | brightness (p99.9 lands at) | window, in units of that p99.9 |
|---|---|---|
| DAPI (dense, many voxels thick) | 0.025 | 0 – 2.5 |
| Slc17a6 / Gpr83 / Bdnf puncta | 3.5 / 5.0 / 2.5 | 0.05 – 0.6 / 0.05 – 0.6 / 0.2 – 0.7 |

A dense, thick channel needs far less light than sparse puncta: DAPI sits about
200x below Gpr83, as a dim grey context under saturated, distinct gene hues. The
values came from rendering the scene, not from statistics; expect to do the same.

The window only picks the LUT colour. A raised low end renders a diffuse floor
black, so puncta stay crisp dots (Bdnf carries cytoplasmic haze under its
puncta, hence 0.2); a wide window keeps a context channel low on the ramp (DAPI's
p99.9 at LUT 0.4).

## Thickness changes the exposure over a time-lapse

Additive light grows with path length, so a growing specimen brightens frame by
frame. The organoid sets its scene exposure for the late, budding frames: at 0
or −1 stops they washed out to white; at −2.5 the bud and the lineage overlay
read and the single founder cell is still visible (two layers at `opacity=0.15`,
ACES). A wider framing changes it again: the gallery still, framed at 0.6 fill,
let auto exposure brighten +0.93 stops and washed the core white, so its tile
pins −1.0 stops. Check the exposure on the thickest frame and at every framing
you ship.

## What appearance cannot fix

- **Mixed channels.** If one channel of an export contains another (the organoid
  export's red is the max of membranes and nuclei), no colormap, window or
  exposure separates them; every nucleus is drawn twice, as a wash over the
  membranes. Unmix before fitting, with the inverse of the blend the authors
  used: see `references/fitting-real-data-lessons.md` in the
  `luxar-gsplat-pipeline` skill.
- **Mixing `volumetric` and `additive` layers.** In the organoid review renders,
  volumetric nuclei over additive membranes turned the organoid nearly black with
  the membranes gone, and all-volumetric (`absorption=1`) gave an opaque blue
  mass that hid the membranes, so the demo keeps both layers `additive`. That is
  one dataset and one review round (with no tuning of the volumetric layer's
  absorption or opacity), not a rule: render the combination before committing
  to it.
