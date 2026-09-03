#### Overlapping gsplat layers now composite additively

The viewer assigns each depth-sorted gsplat node one global back-to-front order
slot. That cannot represent interleaved volumes, and co-located layers have a
degenerate centroid-ordering key that can flip as the camera moves. A demo that
stacks several gsplat *layers* over the same specimen therefore cannot use
`normal` or `volumetric` correctly. Additive compositing is order-independent,
so it is the sound choice for that class of scene.

Six demos authored overlapping layers in `volumetric` and now author
`additive`: cells3d, CT TotalSegmentator, both kidney multichannel demos,
OpenCell MAP4, and the 2-channel neuromast. The
neuromast had been working around the problem with a near-zero absorption
coefficient, which hid the cross-layer ordering problem rather than removing
it; that knob is gone. This supersedes the appearance tuning from #880 because
the volumetric mode it tuned is not sound for stacked layers. The rebuilt heart
uses channel-specific opacities (nuclei 0.12, vasculature 0.30, cardiac tissue
0.16) so the dense nuclear stain does not wash out the other two channels. A new
`test_overlapping_gsplat_layers_declare_order` pins the rule across the audited
set and records why the near misses — side-by-side variants, view- or
time-exclusive nodes, and `kind=partition`/`kind=lod` children of a single node
— are not violations.

Demos with a single gsplat layer are unaffected: with nothing to sort against,
`volumetric` remains the right choice there.
