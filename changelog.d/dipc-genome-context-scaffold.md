#### Dip-C keeps the whole nucleus on screen while you isolate one haplotype

Scrubbing the `haplotype` dimension culled the off-slice copy outright, so
isolating the maternal genome deleted half the nucleus from view. That reads as
"the other half is not there", which is the wrong claim about this data. The two
haplotypes are not alternative readings of one object: they are distinct
physical molecules, folded independently, sharing one nucleus, and their union
is all the chromatin Dip-C reconstructed for the cell — chr1-22 plus X, both
copies, 46 chromosomes.

The same beads are now written twice. A new `all DNA (context)` layer carries
both copies as a thin, faint, neutral-grey scaffold and is pinned visible at
every haplotype slice with `extend_to_all=["haplotype"]`; the chromosome-coloured
`genome` layer still carries its own haplotype coordinate and is still culled
per-scrub. The selected copy is therefore read against the whole nucleus rather
than against empty space, and the scaffold is a normal layer, so it toggles off
for anyone who wants the old isolated view. It stays emissive rather than
volumetric on purpose: it exists to be seen through, and a volumetric scaffold
would absorb the strand it is meant to be framing.

The `genome` layer itself moves from `luminous` to `volumetric` with
`absorption=1.15`. Chromosome territories are volumes, and letting near strands
occlude far ones is what separates them; summing straight through the nucleus
washed the territories into one another. Opacity drops to 0.32 to suit
compositing that accumulates alpha along the ray.

Intensity drops from 0.6 to 0.364, the authored form of the requested
display-range change. At 0.6 the viewer opens the top of the range at
`1 / 0.6 = 1.67`; dragging that top to 2.75 applies a further 1.65x dim. The
equivalent authored gain is therefore `1 / 2.747 = 0.364`, where it survives a
rebuild instead of having to be re-dragged.
