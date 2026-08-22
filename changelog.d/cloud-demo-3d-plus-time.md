#### The cloud demo is now an evolving cumulus, and its noise actually works

`demo_volumetric_cloud` is lifted from a static puff to 3D + time: a convective
cumulus lived through its whole life cycle on a hidden `time` axis, from a low
fragment at the condensation level, through a cauliflower turret and a mature
top leaning downwind, to the whole body pulling in as the thermals feeding it
die. Sixty timepoints, ~1.3M points, about fifteen seconds to generate.

The motion is Lagrangian. 600k air parcels are advected with midpoint steps
through an analytic velocity field built to be exactly divergence-free — an
axisymmetric convection roll written through a Stokes stream function, a wind
shear that leans the column downwind, and a slow swirl — so a point is a parcel
of air that keeps its identity from frame to frame. That property is
load-bearing rather than ornamental: the parcels are a Monte Carlo sample of a
uniform density, and only a solenoidal field keeps that sample uniform as it
deforms. Measured over the run, the parcel count in the cloud core holds to 2%.

The condensate is 4D fractal noise sampled in *material* coordinates — each
parcel's fixed label — times an envelope evaluated at its current *world*
position. Material coordinates weld the texture to the fluid, so it stretches
and folds the way a real cloud's structure does instead of boiling in place;
the envelope is what makes it a cumulus rather than a blob, because liquid
water only exists above the lifting condensation level, which is why cumulus
have famously flat bases. The time axis is a small stack of static 3D
fields quintic-interpolated between, which is exactly 4D value noise evaluated
for the price of a lerp, with the temporal frequency growing as `2^(2k/3)`
rather than `2^k` — small eddies do turn over faster, but at the naive rate the
fine octaves read as shimmer. Points are emissive, so the light is baked in:
condensate is splatted onto a coarse grid and cumulatively summed toward the
light into an optical depth, giving a sunlit crown over a cool shadowed base.

Four defects surfaced along the way, and all four were silent — each produced a
plausible-looking cloud rather than an error.

The positional hash was the oldest and the worst. `(xi*C1 + yi*C2 + zi*C3) % M`
leaves the result correlated with the magnitude of the coordinates, so near the
lattice origin every cube corner comes back close to the same value. The demo
sampled its noise across less than one lattice cell, which put it squarely in
that regime, so the advertised seven octaves of fractal detail were in fact a
smooth ramp — and the scene it shipped was **790 points out of 800,000
candidates**, a barely visible smudge that had been in the tree since the demo
was written. The hash now ends in the standard 32-bit avalanche, and the base
frequency puts several lattice cells across the cloud. Measured, the field's
correlation with distance from the axis fell from −0.35 to 0.01.

Blending two independent keyframes with weights that sum to one gives a variance
of `(1-u)² + u²`, which is half its keyframe value halfway between. Across space
that is invisible, because neighbouring samples sit at different lattice phases;
along *time* every parcel shares one weight, so the entire cloud breathed in and
out of focus in step with the keyframe grid. Normalizing by the weight vector's
norm holds it flat — per-octave standard deviation now varies by under 2% where
it swung 0.28 to 0.40.

Thresholding needed the statistics measured per frame *and* over the right
population. Measuring once is not enough: the coarsest octave has only a few
lattice cells across the domain, so its spatial mean is itself a random variable
that the time axis walks around, and the fraction of the field above a fixed
level wandered from 0.37 to 0.64 — a global wander, since every parcel shares
one `tau`. Measuring per frame over the whole domain is not enough either,
because the flow steadily replaces the air inside the envelope, so the cloud's
bulk water follows a random walk driven by whatever material happens to be
passing through. The statistics are now taken over parcels weighted by how deep
inside the envelope they sit, which leaves the life cycle as the only thing that
moves the point count.

Finally, thresholded noise applied uniformly punches daylight straight through
the core, which is the one thing a cumulus never has — it is optically thick
within a few metres of its surface. The noise now erodes the body inward from
the surface, keeping full authority at the ragged fringe where a real cloud's
structure lives. (That the core was solid and still rendered hollow turned out
to be a second, separate defect — see below.)

Two smaller things fell out of looking at the result in the viewer rather than
at the arithmetic. The optical depth folded the grid geometry and the parcel
count into one magic constant, so `--parcels` or an edit to the shading grid
would silently relight the whole cloud; it now divides both out and `EXTINCTION`
is a real extinction coefficient per unit of water column. And flooring the
turret profile at a small positive width — the obvious way to keep a division
safe — left a thin chimney of cloud running up the axis forever, because above
the crown the profile is meant to be exactly zero. (That profile has since been
replaced entirely; see the last section.)

#### The renderer has to be in the same optical regime as the shading

Everything above was true of a build that, looked at in the actual viewer from
more than one angle, was plainly wrong: the cloud rendered as a glowing archway
with a hole through the middle, and from overhead as a C-shaped ring.

The cause was a regime mismatch, and it is worth stating plainly because
nothing about it shows up in a test. A cumulus is optically THICK — τ ≫ 1
within metres, so you see its surface and essentially nothing behind it.
`additive` blending models the exact opposite: an optically thin emissive
medium, brightness = ∫ρ ds, depth ignored entirely (the enum says so). Shading
each parcel by the water column above it and then compositing with a mode that
cannot occlude is incoherent — the darkened interior is not hidden behind the
lit shell, it is added straight through it. Measured on that build, the core
carried the HIGHEST point density (346 vs 51 per unit area at the rim) and the
LOWEST brightness (0.46 vs 0.85). The hole was in the light, not the geometry.

The node is now `volumetric` — emission composited against absorption, back to
front — with per-point RGBA alpha carrying optical depth, so denser air hides
more of what is behind it. The baked sunlight becomes ordinary single
scattering, and the cloud reads as a body.

Three things followed from fixing that, none of which was visible while the
cloud was a glow.

The sun had to come off the vertical. A light directly overhead illuminates
every surface by its depth alone, so two lobes at the same altitude are lit
identically however they face and the relief that makes a cumulus legible
disappears. `optical_depth` now integrates along an arbitrary direction — the
parcels are rotated into a light-aligned frame, accumulated, and rotated back —
and the shading is composed of three terms rather than one: direct sun along
the beam, blue skylight along the vertical, and a weak ground bounce. Skylight
being blue is why a real cumulus underside reads cool grey rather than black.

Exposure had to leave headroom. Under emission-absorption the accumulated
radiance of a thick medium tends to emission/extinction — that is, to the
parcel colour itself — so a fully lit parcel IS the brightest pixel the cloud
can produce. Three light terms that each looked reasonable summed to 1.4, every
lit face clipped to flat white, and all of this shading work became invisible.
They now sum to ~0.4, because the cinematic preset's bloom threshold is 0.01
and the whole cloud therefore blooms onto itself rather than just its
highlights.

And dissipation had to move to the silhouette. Decay used to be expressed by
raising the noise threshold, which erodes the interior — invisible once the
interior cannot be seen. The turret now sags and the body pulls in.

#### The shape is the union of thermals, not a surface of revolution

The silhouette came from one lathe profile, and it looked like one: a smooth
vase with no lobes for the light to catch. Perturbing its radius with noise did
not help, because cauliflower is not a perturbation of a shape, it IS the
shape. A cumulus is not a shape at all but a process — a succession of buoyant
bubbles punching up through the condensation level, each overshooting slightly
less than the last. The envelope is now the p-norm union of ~42 such thermals,
each rising from the base to its own ceiling and swelling then shrinking, over
a slab of cloud sitting on the LCL that keeps the famous flat bottom and roots
the tower to it.

Two failures on the way there are pinned by tests. Thermals whose late fade was
too gentle arrived at their ceiling still 60% of full size, clear of the crowd
below with nothing to merge into, and rendered as spheres floating above the
cloud like balloons. And a root slab that was wide, hard-rimmed and too shallow
separated from the tower above it and read as a second, unrelated cloud sitting
underneath.

The scene opens on the mature cloud rather than on frame 0, framed by a camera
composed for the cinematic 63° lens from the data's own extent, with the
dimension slider panel open. It does not autoplay: the zarr `animation` block is
capture-only today, so the viewer restores `current_step` and nothing else, and
starting the sequence is `K`.
