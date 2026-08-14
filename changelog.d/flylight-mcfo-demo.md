#### A 3-colour MCFO fly brain demo, and three things it taught us about thin filaments

`demo_gsplats_3d_flylight_mcfo_neurons.py` fits a Janelia FlyLight MCFO
volume from the FISBe benchmark — long, thin, widely branching *Drosophila*
neurons — as anisotropic Gaussian splats. It is the first demo whose content
is filamentous rather than blobby, and every default that works for nuclei
and tissue turned out to be wrong for it.

The sample is range-extracted from FISBe's 7.1 GB Zenodo archive over HTTP
byte ranges, so a first run transfers ~415 MB rather than the whole archive.
(Zenodo's `HEAD` returns 200 with no `Accept-Ranges`, so the demo probes with
a one-byte ranged `GET` and reads the total size from `Content-Range`.)

Three findings are recorded in the demo's docstring because they generalise
to other sparse filamentous microscopy:

`gsplat cal` cannot calibrate K on this data. The blind-spot protocol needs
image noise to find a held-out peak, and FISBe's raw is effectively
noise-free — Janelia Workstation stitching and distortion correction scrub
the pixel noise even though values stay 12-bit. Sweeps report
`signal_limited` / `still_climbing` with sigma_hat ~1e-10, so the reported
"K*" is just the top of the supplied grid.

Global PSNR is the wrong number to steer by. Only ~0.2% of this volume's
energy sits inside the annotated neurons, so global PSNR mostly scores empty
space: at K=30,000 a fit reads 42 dB globally while dropping *half* the
neurite brightness, and on a MIP the neurites visibly break into
disconnected beads. Scoring against the ground-truth instance masks exposes
a ~17 dB gap and picks a very different operating point.

Fitting the three MCFO channels separately breaks the colour. MCFO is a
stochastic label, so a neuron's colour is a ratio of the three channels at
the same voxels; three independent fits do not co-locate and each axon
renders candy-striped. The demo instead fits the per-voxel channel maximum
once and reads each splat's colour from the channels at its own centre,
balancing the channels by their own robust maxima first — unbalanced, the
brightest channel wins 96% of splats and the whole brain reads red.

Aggressive floor suppression (`floor="p99"`) is what makes the scene
legible at all: the default `auto` floor correctly removes a pedestal but
leaves the neuropil autofluorescence, which fills the brain silhouette and
renders as a saturated blob under every blending mode. With the haze gone,
1.2 M seeds settle at ~19 K splats — the neurons and little else, at
~1800:1 compression over the source voxels.
