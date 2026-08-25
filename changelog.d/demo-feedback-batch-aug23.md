#### Cells3D multichannel demo opens at a brightness you can actually read

The two-channel `cells3d` demo shipped far too dim: to see the membranes and
nuclei at all you had to push the viewer's exposure up by 2.35 stops by hand
every time you opened it. Two thin volumetric channels over a 60-slice stack
integrate to very little radiance, so the scene arrived near the bottom of the
tone curve. That measured +2.35 EV is now baked into the scene's viewer config,
which triples the mean frame luminance (4.8 → 14.4 on an A/B of the same scene)
while leaving highlight clipping unchanged at 0.01% of the frame.

The scene uses viewer exposure rather than rewriting the fitted amplitudes. A
writer-derived display window would compensate the LUT lookup after an amplitude
rescale, but the stored amplitude also scales emitted radiance and volumetric
optical depth, so that rewrite would still change the render.

#### Drosophila gastrulation demo is no longer one saturated pink shell

The embryo rendered as a featureless, fully clipped magenta silhouette — no
cephalic furrow, no midgut invagination, no individual nuclei. This fit stores
raw detector-count amplitudes from 5 to 798, while the scene authored the scalar
display window `[0, 1]`, so every splat also landed at the top of the LUT.

The amplitudes are now **robustly normalised at authoring time**, with the pooled
99.9th percentile across the whole additive ladder mapped to 1.0. One shared
factor preserves the relative brightness of every streaming prefix; a per-rung
factor would make each upgrade render at a different exposure. Values above the
reference percentile remain above 1.0, so a single hot splat cannot darken the
whole scene. Constant-amplitude data maps to 1.0 instead of being left in raw
detector counts. The loaded arrays are rewritten in place so the authored
per-rung metadata used by LOD upgrades survives, and nested partition/LOD trees
are grafted with the same structure as `add_gsplats_from_file`.

That normalisation is the primary exposure change: the stored amplitude scales
both emitted radiance and volumetric optical depth. The display window only
compensates the LUT lookup. The scene authors **`[0, 1.153]`** on the robust
normalised scale (`intensity≈0.8675`, `offset=0`), which preserves the previous
fit's measured colour mapping after moving the reference from max 798 to p99.9
512. Opacity is a further trim at **0.262**, compensated by the same scale change
so the current fit keeps its accumulated radiance and optical depth. Absorption
then controls how quickly that depth builds and remains **0.57**. Blending itself
goes from `normal` (alpha-over) to `volumetric`, which is what a single
fluorescence channel wants, and scene exposure stays neutral.

Individual nuclei now read as discrete blobs across the whole shell, and the
cephalic furrow and posterior pit are both legible. Pixels using magma's warm
upper body go from **0.00%** of the frame to a few percent — the old render
contained no amber at all, despite the comment claiming it spread the nuclei
"across magma's violet-to-amber body".

The scene also authors its **opening camera and auto-rotation** rather than taking
the default whole-scene fit, which left the embryo small in a lot of black. The
embryo's long axis is centre column 1 and so already maps to world Y, i.e.
screen-vertical; the camera backs off along +world Z until that long axis
subtends 0.85 of the half-frame, solved from the data's own bounding box so a
refit reframes itself, and pulled in for the cinematic lens. Because
auto-rotation orbits the up axis, the spin runs about the embryo's own length.

#### Rainbow sphere opens filling the frame, already turning

The sphere used to open as a small ball in the middle of a lot of black, sitting
still. It is a single decorative object, so the scene now authors its own
opening camera instead of taking the default fit: the orbit distance is solved
from the sphere radius so the ball subtends 0.72 of the field of view and spills
past the edges of the canvas at the cinematic preset's 63 degree reference FOV,
while the viewer FOV remains configurable. Auto-rotation is on from the first
frame at the viewer's own presentation speed.

Filling the frame means many more hues sum into every pixel, which washed the
middle of the ball out to pastel grey, so the authored additive gain is halved
to keep the rainbow saturated. Nothing clips at either value.

#### Cubic Array opens on the lattice instead of on the star field

The demo started with the cube as a speck in the middle of the frame. The star
field deliberately spans ±500 units while the lattice is only ~50 across, so the
default whole-scene camera fit was backing off far enough to contain the stars —
framing the backdrop rather than the subject.

The scene now authors its own opening camera, with the orbit distance solved
from the lattice's own bounding-sphere radius (so it tracks `grid_size` and
`spacing` rather than being a magic number) and the orbit pivot bound to the
`CubicArray` node's bbox centre. The view is a three-quarter one, so three faces
are visible and the grid reads as a cube. The distance is solved at the
cinematic preset's 63 degree reference FOV while the viewer FOV remains
configurable. Auto-rotation is on from the first frame, and the scene opens 1.5
stops down.

#### FlyLight whole-brain demo gains a reference cage

MCFO labels only a handful of neurons out of a whole brain, so this scene is
mostly empty and gave no cue for how big the specimen is, where the edges of the
imaged stack are, or how much of the black is "no label" rather than "outside
the data". It now draws a faint box around the stack bounds, ruled at 100 µm
intervals over all six faces — 60 segments in total, at absolute round
coordinates rather than at fractions of the extent, so the spacing means a fixed
physical distance.

The cage is drawn additive at low opacity, which is what lets it cross the
specimen without hiding any of it: additive never occludes the neurons behind
it. Box edges are twice the width and a few times the radiance of the interior
rules, so the bounds stay legible as the outer shape while the grid stays
scaffolding. It is its own layer, so it can be switched off in the Layers panel.

#### The "organoid" demos are a mouse blastocyst, and the second channel is Lamin B1

The demo shipped with no title, no explanatory panel and no channel legend — a
bottom-right caption was the only overlay — and its unnamed first channel was
labelled just "Channel 0". It now carries the same overlay set as its sibling
gsplat demos: title top-left, explanatory panel below it, per-channel legend
bottom-left tinted to each layer's own colormap, and the data-source caption
bottom-right.

The unnamed channel is **Lamin B1**, immunostained with ab16048. The OME-Zarr's
`omero.channels` metadata labels the two channels `LaminB1` and `Dapi` in that
order, and IDR's protocol annotation for the image records the antibody and
dilution. Lamin B1 is a nuclear *lamina* protein, so that channel draws the
envelope around each nucleus while DAPI fills it — which is the whole point of
the source dataset: it is a benchmark volume for Nessys, a segmentation method
that works from the envelope precisely because densely packed nuclei merge in a
DNA channel but stay individually outlined in a lamina one.

Two other facts were wrong and are corrected. The caption said "Light-sheet
microscopy"; IDR records a Leica SP8 point-scanning confocal with an HC PL APO
40×/1.30 Oil objective. And the specimen is not an organoid at all — an organoid
is a stem-cell-derived culture, this is a **mouse blastocyst at E3.5**, a
pre-implantation embryo. IDR files image 6001240 (`B1_C1.tif`) under the dataset
"Blastocysts", its growth protocol reads "Mouse blastocysts (E3.5)", and the file
that shipped it already said so in its own docstring while its metadata said
"mouse intestinal-organoid" two lines below. Nothing anywhere in idr0062 mentions
intestine.

Both demos on that image are renamed accordingly —
`gsplats_3d_organoid_multichannel` → `gsplats_3d_blastocyst_multichannel` and
`gsplats_3d_organoid_dapi_nuclei` → `gsplats_3d_blastocyst_dapi_nuclei` — module
filenames, keys, output names, titles, scene attributes, gallery manifest and
media, calibration artefacts, helper-script slugs and the pinned data-artifact
filenames included. Only names changed here, not bytes — the record's re-encoded
copy is now recorded separately as `hosted_sha256` (#1734). The legend swatches
are read out of the LUT the renderer actually uses, so they cannot drift from the
layers they label.

#### The two "garden" demos were showing the same picture; the INRIA one moves to bonsai

`gsplats_interop_inria_garden` and `gsplats_interop_mipnerf_garden` rendered the
same scene from the same trained model. Measured on the two cached stores: both
5,834,784 splats, the same bounding box to three decimals, bit-identical
amplitudes, and 80% bit-identical centres. The antimatter15 `garden.splat` the
Mip-NeRF demo downloads was produced from the very INRIA checkpoint the other
one range-extracts, so it could not have been otherwise.

Nor was the INRIA demo earning its 1.45 GB on its other claims:
`gsplats_interop_macro_clusterfly` already covers both the INRIA PLY dialect and
zip-member range extraction, for 68 MB. Its one genuinely exclusive piece of
coverage is the **Zip64** central-directory path, which only an archive above
4 GiB reaches — and that is a property of the archive, not of the member.

So the demo now pulls the **bonsai** member of the same `models.zip` instead:
309 MB rather than 1450 MB, an even better extraction ratio (2% of the archive
rather than 10%), the same Zip64 path, the same reference float32 checkpoint,
and a different picture. Renamed throughout to
`demo_gsplats_interop_inria_bonsai`.

The opening camera is derived from the data rather than hand-placed. Bonsai is
an inside-out capture whose floaters push the bounding box to ±27, so
auto-framing shows the room from outside as a ball of white spikes, and even the
median of all 1.24 M centres lands in the middle of the rug. The blossoms,
though, are the only strongly pink thing in the room: selecting on hue isolates
8,223 splats whose trimmed median is (0.50, −0.80, −1.32) with an extent of
1.1 × 0.8 × 1.1 units. That is the tree, and that is where the camera looks. A
regression test now fails if the demo ever returns to the garden member.

#### The nD transform bench explains its channel captions

The help card now clarifies that a marker sits at its own local index rather
than under the cursor, and the reasoning behind the existing world-to-local
caption wording is recorded next to the channel-row definitions.

#### The earthquake globe stops looking like a dot screen

The Global Earthquakes demo drew its Earth from 120,000 points on a bare
Fibonacci lattice with a point radius pinned at a constant 0.003. Points on a
unit sphere sit a mean `sqrt(4π/n)` apart, so at 120k the spacing was 0.0102 —
more than three times that radius. The globe rendered as a stipple with
stair-stepped coastlines, and the undithered lattice beat against the
equirectangular Blue Marble into visible moiré.

Its sibling `demo_ocean_currents_earth` had already solved all of this — 8M
dithered points and a vectorised texture lookup — and had even left a comment
noting that the earthquake demo samples one point per Python-loop iteration and
"does not scale to millions". The two demos had simply drifted.

So the three primitives now live in one place, `demos/_globe_common.py`, and
both demos use them: the dithered Fibonacci sphere, the geographic-to-Cartesian
mapping, and the vectorised equirectangular sampler. The earthquake globe moves
to 2M dithered points with the point radius **derived** from the spacing rather
than pinned — a constant is tuned once at one point count and then silently
stipples the globe when the count changes, which is exactly what had happened.
The per-point Python sampling loop is gone, which is what made the higher count
affordable at all.

The globe's longitude convention was checked rather than assumed while moving
it: sampling the Blue Marble the way this demo does puts land at the Sahara,
the Amazon and Tibet and ocean at the mid-Pacific, mid-Atlantic and mid-Indian
Ocean, six for six, and it agrees with where the quake markers are placed. The
Ring of Fire lines up along the Andes and the Aleutians in the render.

#### The 5D spiral galaxy becomes a Galaxy Simulation

`demo_5d_spiral_galaxy` scattered points along four fixed logarithmic curves and
rotated the whole picture rigidly. That is the material-arm model, and the
winding problem rules it out: a real disc rotates differentially, so any arm
made of a fixed set of stars shears into a rag within two or three turns.

`demo_galaxy_simulation` computes the galaxy instead. A Hernquist bulge, a
Miyamoto–Nagai disc and an NFW halo add in quadrature to a rotation curve that
rises to 260 km/s by 5 kpc and stays flat past 30 — the halo is what makes it
flat, and nothing about the shape is imposed. `Ω = v_c/R` and the epicyclic
frequency `κ² = (2Ω/R) d(R²Ω)/dR` are differentiated off that curve numerically,
so changing the mass model cannot leave a stale expression behind. With the
pattern speed at 23 km/s/kpc they put the inner Lindblad resonance at 1.7 kpc,
corotation at 11.4 and the outer Lindblad resonance at 19.4.

Every star then rides a forced epicycle: free radial oscillation at `κ`,
vertical at `ν`, a tangential excursion of exactly `2Ω/κ` times the radial one,
and on top of that the spiral's forced response with its resonant denominator
`κ² − m²(Ω − Ω_p)²` — which is why the arms are strong across the mid-disc and
fade toward both resonances, as a consequence rather than as a taper anyone
painted on. Because each star's guiding centre turns at its own `Ω(R)` while the
pattern turns rigidly at `Ω_p`, stepping the time axis shows stars **overtaking**
the arms inside corotation and lagging outside it, and after 480 Myr — about two
and a half turns at 8 kpc — the arms have not wound up. That is the whole
argument for density waves, and the old demo could not show any of it.

Ages carry the rest. `σ_R ∝ age^0.33` heats older stars onto bigger epicycles, so
they leave the arms: the observation that spiral arms are a young population
falls out rather than being drawn in. Colour is the blackbody colour of each
population's effective temperature, integrated through the CIE 1931
colour-matching functions and the sRGB primaries — which is why the arms are
genuinely blue-white and the bulge genuinely amber. Dust lanes sit on the
upstream edge of each arm and redden as well as dim; HII regions mark the shock
in Hα.

The five stellar-age bins are **layers**, not a navigable dimension. As a
dimension the viewer opens on one bin and the galaxy shows up as the few percent
of stars that happen to be young; as layers the default view is the whole
galaxy and the Layers panel still isolates any bin. The frame count is likewise
odd on purpose: the viewer opens a non-displayed dimension at the midpoint of
its range and a point only matches a slice it sits exactly on, so an even count
puts the midpoint between two frames and the whole time-resolved scene renders
empty — which it did, at 21,911 of 6.5M points, until the grid was fixed.

#### 4D fractals: four times the resolution, and no more haze

The fractals were fuzzy for two compounding reasons, and the grid was stuck at
50 because of a third.

The generator built four `grid_size**4` meshgrids up front — 100 MB at grid 50,
26 GB at grid 200 — so a finer lattice was simply impossible. It now evaluates
one w-plane at a time, which makes the working set `grid_size**3`, and the
lattice moves to **200**: four times the linear detail. Only every fourth plane
is materialised, so the slider still has the same 50 stops it always had, each
now four times finer.

Second, the demo drew the fractal **solid**. These sets fill up to half their
bounding volume, so every ray summed dozens of hidden interior points into a
haze that buried the surface in front of it. Only boundary voxels are kept now.
That is both the right thing for a solid and what makes the finer grid
affordable: at grid 200 it removes 85% of the XOR set, 86% of the
hypercheckerboard and 82% of the diamond, while leaving the already-thin
Sierpinski and Cantor sets essentially untouched.

Third, the point radius was pinned at a constant tuned for the old grid — 62% of
the lattice step at grid 50, and two and a half times oversized at grid 200. It
is derived from the step now.

The budget is also per-plane rather than global. The old global cap divided a
fixed total across every plane, so raising the grid made each individual slice
*sparser* — exactly backwards. At 150k per plane the flat faces read as
continuous surfaces; at 80k they were still stippled. 44.5M points, 89 MB on
disk (lattice coordinates compress well).

Appearance comes from a live Layers-panel session: `volumetric` blending with
absorption 1.23 and opacity 0.43, gamma 1.0, and an authored display window of
`[0, 1.971]` rather than the `[0, 16]` the old `intensity=0.0625` was stating.

#### Smaller things

The Cubic Array's auto-rotation goes to 0.5, twice the viewer's presentation
default: the lattice's moiré interference is the thing worth watching, and it
only resolves as the view angle sweeps.
