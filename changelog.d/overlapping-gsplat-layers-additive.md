#### Overlapping gsplat layers now composite additively

Splats are depth-sorted within a gsplat node, but the viewer does not sort
across sibling nodes. A demo that stacks several gsplat *layers* over the same
specimen and asks for a depth-sorted mode — `needsDepthSort` is exactly `normal`
and `volumetric` — therefore composites those layers in an arbitrary order, and
the render is not correct. Additive compositing is order-independent, so it is
the only sound choice for that class of scene.

Seven demos authored overlapping layers in `volumetric` and now author
`additive`: the Acto3D mouse embryo heart, cells3d, CT TotalSegmentator, both
kidney multichannel demos, OpenCell MAP4, and the 2-channel neuromast. The
neuromast had been working around the problem with a near-zero absorption
coefficient, which hid the arbitrary ordering rather than removing it; that
knob is gone. A new `test_overlapping_gsplat_layers_are_additive` pins the rule
across the audited set and records why the near misses — side-by-side variants,
per-timepoint nodes, and `kind=partition`/`kind=lod` children of a single node —
are not violations.

Demos with a single gsplat layer are unaffected: with nothing to sort against,
`volumetric` remains the right choice there.
