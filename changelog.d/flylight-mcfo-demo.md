#### MCFO fly brain neurons in a volume-rendered neuropil

`demo_gsplats_3d_flylight_mcfo_neurons.py` fits labelled *Drosophila* neurons
from the FISBe benchmark — long, thin, widely branching — threaded through the
counterstained brain they live in, as one volume-rendered Gaussian splat cloud.

It is the first gsplat demo whose *microscopy* is filamentous rather than
blobby, and the first where the dense medium is the point rather than something
to see past: under volumetric (emission-absorption) compositing the neuropil
occludes front-to-back, so the brain reads as a solid body and neurites
genuinely pass behind it and are dimmed, instead of glowing through as they
would under additive.

The neurons come from FISBe's Zenodo archive, range-extracted over HTTP so a
first run transfers ~415 MB rather than the whole 7.1 GB. (Zenodo's `HEAD`
returns 200 with no `Accept-Ranges`, so the demo probes with a one-byte ranged
`GET` and reads the size from `Content-Range`.) The neuropil is not in FISBe at
all: the sample's `channel_spec` is `sssr` — three signal channels plus a
reference — and FISBe distributes only the signal. The reference channel is
fetched from the FlyLight H5J on S3 and decoded from its HEVC streams, which
needs `ffmpeg`; without it the demo degrades to a neurons-only scene rather
than failing.

Five things about this data are easy to get wrong, and are recorded in the
demo's docstring because they generalise to other sparse filamentous
microscopy.

`gsplat cal` cannot calibrate K here. The blind-spot protocol needs image noise
to find a held-out peak, and FISBe's raw is effectively noise-free — Janelia
Workstation stitching and distortion correction scrub it even though values stay
12-bit. Sweeps report `signal_limited` / `still_climbing` with sigma_hat ~1e-10,
so the reported "K*" is just the top of the supplied grid.

Global PSNR is the wrong number to steer by. Only ~1.5% of the raw volume's
energy sits inside the annotated neurons — 41% once the floor is suppressed,
measured against the same mask — so at K=30,000 a fit reads 42 dB
globally while dropping *half* the neurite brightness, and the neurites visibly
break into disconnected beads on a MIP. Scoring against the ground-truth
instance masks exposes a ~17 dB gap.

The floor is the most consequential knob, and a percentile floor is brutal.
`--floor p99` subtracts 0.0337 here: it zeroes 99% of voxels and removes
**97.5% of the image's energy before a single splat is placed**, most of the
faint neurites with it. Measured against the raw data on the annotated neuron,
that configuration reproduced 0.57x of its energy and lost 87.8% of its
faintest fifth. `auto` (the CLI default) subtracts 0.0029, removes 35% of
energy, and takes the dim-band dropout to **zero** at 787K splats — beating
even a zero floor on the foreground while using 30% fewer splats. The demo now
uses `auto`.

The trap that hid it: scoring against the floor-*suppressed* target rather than
the raw data. Against that reference the p99 fit reads a respectable 28.10 dB,
because the reference has had the same signal deleted from it.

Background suppression moved to render time, where it belongs. Each neuron
splat's alpha now ramps with its own amplitude, so haze goes optically thin
while neurites stay opaque — and unlike a floor it is reversible, because the
faint splats are still in the scene for the display range to recover. The ramp
bottoms out at a small non-zero alpha for exactly that reason: alpha 0 is folded
into a splat's contribution before the shader's discard, so it would delete the
splat as thoroughly as the floor did.

The initial camera is measured, not defaulted. FISBe ships the *unaligned*
FlyLight stack — the specimen as mounted, ~52 degrees off axis for this sample.
Rather than rotate the splats (which would desynchronise them from FISBe's
annotations and require rotating every covariance), the demo measures the
specimen's principal axis and rolls the camera to match, framed close. The
angle is per-specimen, measured at authoring time.

Seeds are not the splat count — `cull_retention` is. The optimiser works a fixed
pool of `seeds` splats and then culls by cumulative amplitude mass (default
0.95), which on floor-suppressed data discards nearly everything: 1.2 M seeds
settle at ~19 K splats. The trap is fixing that by lowering seeds. Measured,
128,000 seeds at retention 1.0 scored *worse* than 18,623 splats from 1.2 M
seeds (fg 25.97 vs 27.60) with a visibly more beaded axon — seeds buy search,
retention buys count. The demo keeps seeds at 1.2 M and retention at 0.9999,
which with the `auto` floor gives ~787 K neuron splats (~1.14 M in the scene
with the neuropil).

Neurons and neuropil share one node. Two nodes covering the same volume have no
correct draw order, so they are merged and depth-sorted together, distinguished
by per-splat RGBA. Alpha there is *optical depth* and it accumulates: across
~356 K neuropil splats spanning ~350 µm, alpha 0.12 renders the brain
effectively opaque — the intended look — while ~0.01 gives a translucent haze
with every neurite visible. The cost is that the Layers panel can no longer fade
the neuropil independently.

Finally, the three MCFO channels must not be fitted separately. MCFO is a
stochastic label, so a neuron's colour is a ratio of the channels at the same
voxels; independent fits do not co-locate and each axon renders candy-striped.
The demo fits the per-voxel channel maximum once and reads each splat's colour
from the channels at its own centre, balancing channel gains first — unbalanced,
the brightest channel wins 96% of splats and the whole brain reads red.
