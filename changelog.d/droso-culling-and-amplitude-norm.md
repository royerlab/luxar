#### Splat amplitudes are normalised when they enter a scene

A fitted `.gsplats.zarr` stores amplitudes in raw source units — the fitter
multiplies its `[0, 1]` working copy back out by the volume's own intensity
range, so a fit from a uint16 detector stack carries detector counts, in the
hundreds or thousands. Those units could not reach the screen intact and no
viewer control could rescue them, which is why so much shipped appearance had
become a hand-tuned magic constant.

The stored amplitude does two jobs and the display window only reaches one. It
picks the colormap LUT index — `t = clamp((A - min) * scale, 0, 1)`, clamped, so
it selects a colour — and it sets emitted radiance and, under `volumetric`,
optical depth, neither of which is windowed by anything. An amplitude of 800
therefore emits a thousand times the radiance of 0.8 and drives `1 - exp(-tau)`
straight to an opaque shell. The only lever left was `opacity`, which then had to
carry a factor of about 1/500 on a control that runs from zero to one.

`add_gsplats_from_data`, `add_gsplats_from_file` and the graft path now scale
amplitudes so a robust upper reference (the 99.9th percentile, the same statistic
`amplitude_data_range` already uses) lands at 1.0. The reference is robust rather
than the maximum so that one saturated splat cannot compress the scene toward
black or move the factor on every refit, and the target leaves about a thousandth
of the splats above 1.0 for the authored window to clip. Data already in range is
left alone, so a fit of an already-normalised volume that is then dimmed by hand
is not double-brightened. `normalize_amplitudes=False` opts out, a number sets an
explicit target, and the factor applied is recorded as
`amplitude_normalization_factor`.

One factor covers the whole structure. That is the part worth stating, because
getting it wrong is invisible until it is on screen: a coarser substitutive level
holds merged representatives carrying combined mass, so its own percentile is
higher, and a per-level factor scales it down relative to its siblings and pops
the brightness at every LOD switch. The same applies across partition parts,
where the tile holding the brightest region would otherwise be scaled differently
from its neighbours and step at every seam. Both were live: the substitutive case
was caught by an existing harmonization test whose expectation had moved, and the
partition case has no `GSplatData` to normalise at all — a `kind=partition` tree
is grafted node for node, so until now the largest and rawest datasets were the
ones with no normalisation path whatsoever.

Five places in the repo asserted the opposite — that scaling amplitudes "does
nothing" because "the viewer normalises by the stored maximum". The skill, the
two pathology demos that ship raw-unit amplitudes because of it, and the format
spec now say what the shader actually does.

#### A demo about what culling removes

`gsplats_3d_culling_study` takes the Drosophila gastrulation fit that the
single-frame demo already ships and stacks ten cumulative-amplitude cull levels
of it on one selector, so picking an entry swaps the level in place at a fixed
camera. Side by side the eye has to carry a memory across a gap and differences
this subtle disappear; stacked, stepping the selector is a flicker test, which is
the only way to actually see them.

Cumulative culling keeps the brightest splats accounting for a fraction of the
fit's total amplitude, so `retention=0.99` does not mean keeping 99% of the
splats — it means keeping however many account for 99% of the light. On this fit
that is 93% of them, and giving up a quarter of the amplitude halves the file
while removing splats spread across the dimmest background rather than anything
you are looking at. Each level's size is measured by writing it, not estimated
from a bytes-per-splat constant, because the encoder picks its quantisation from
each level's own range.

The demo adds no hosted bytes: it reuses the existing
`gsplats_3d_drosophila_gastrulation` archive. Its ratios are quoted against the
decoded acquisition and say so — the same timepoint inside the compressed source
is about a quarter of that, and a compression figure that does not name its
denominator is how these numbers get inflated.
