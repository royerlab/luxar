#### A cull that got quietly harsher the more data you gave it

`gsplat cull --method cumulative` keeps the brightest splats accounting for a
given fraction of the total amplitude. It accumulated that total in float32.
The amplitudes are sorted descending, so the running sum grows while the values
being added shrink, and once the sum passes about a billion a single float32
step exceeds the amplitudes still arriving. The sum stops growing. The total is
understated along with it, so the requested fraction of a too-small total is
reached far too early and the cull discards far more than it was asked to.

The error scales with the data. On one Drosophila fit, where the correct answer
is 55.3% of splats kept at every size, float32 kept 53.5% at five million
splats, 37.7% at twenty-five million, and 11.6% at a hundred and twenty-eight
million. A 500-timepoint archive built at retention 0.960 came out five times
over-culled, and nothing anywhere reported a problem: the output is simply a
smaller file that still renders.

Small fixtures cannot catch this. Saturation needs a typical value to fall below
two-to-the-minus-twenty-fourth of the running total, so at a few thousand splats
the error is under a hundredth of a percent and every test passes. What exposed
it was a scale-invariance check — a twenty-frame subset and the full
five-hundred-frame merge have provably identical amplitude distributions, and
culled to 65% and 12%.

The accumulator, the two statistics reductions beside it, and the matching
amplitude CDF in `luxar gsplat info` are now float64. The regression tests assert
the kept fraction against a float64 ground truth computed in the test rather than
a hard-coded number, and assert directly that tiling the same distribution eight
times does not move the fraction, which is the property the bug broke.

#### Choosing a LOD recipe by the view rather than by the element count

The recipe table was ordered by scale, which turns out to be the wrong first
question: a 128-million-splat timelapse wants `stream`, while a 674-thousand-splat
galaxy legitimately wants `levels`. What decides it is how the thing is looked at.

One object always fully in frame has nothing to frustum-cull, so parts cost a
request per node to bootstrap and return nothing — measured at 63 requests to
first paint for a single stacked leaf against 689 for a partition of 44 parts,
each with four substitutive levels and a four-step ladder. An object viewed whole
never selects a coarse substitutive level, because the selector is
screen-occupancy based, so those levels are bytes nobody fetches. And the
per-node splat cap applies to the resident slice rather than the node total, so
an nD node sliced on a hidden axis is measured per-slice.

`cryoem_virus` is a compact particle always seen full-frame and moves to
`stream`; a regenerated archive is 28% smaller with a quarter of the nodes,
while publishing that archive remains #1879. `milky_way_dust` keeps
`levels` and pays 39% for them on purpose, because a galaxy really is orbited at
range. The two pathology slides keep `adaptive`, which is what parts are for.

One catch travels with `stream`: a flat store needs re-chunking or scrubbing gets
worse rather than better — 173 requests per timepoint step as built, two after
`luxar optimise --profile archive`. Additive-only and re-chunking are a package.

#### Adding the 4D Drosophila embryogenesis demo

The demo catalogue now includes a 500-timepoint SiMView recording of Drosophila
embryogenesis as one time-sliced 4D Gaussian-splat node. The scene validates and
normalises the stored time coordinate to the acquisition's 30-second interval,
keeps one appearance scale across every frame and ladder rung, and documents the
fit, cull, physical scaling, and streaming-optimisation pipeline.

The 824 MB archive is pinned to Zenodo record `22118695`; the record remains
unpublished until it is published by hand.
