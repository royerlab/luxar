#### Overlapping gsplat layers now composite additively

The viewer assigns each depth-sorted gsplat node one global back-to-front order
slot. That cannot represent interleaved volumes, and co-located layers have a
degenerate centroid-ordering key that can flip as the camera moves. A demo that
stacks several gsplat *layers* over the same specimen therefore cannot use
`normal` or `volumetric` correctly. Additive compositing is order-independent,
so it is the sound choice for that class of scene.

Six demos authored overlapping layers in `volumetric` and moved to `additive`:
cells3d, CT TotalSegmentator, both kidney multichannel demos, OpenCell MAP4, and
the 2-channel neuromast. Subsequent appearance tuning returned the neuromast to
`volumetric`: both layers now author a stable `layer_order`, so the
arbitrary-ordering argument above no longer applies to this scene, and a
deliberately small absorption coefficient preserves depth cues without either
layer hiding the other. The rebuilt heart uses channel-specific opacities
(nuclei 0.12, vasculature 0.30, cardiac tissue 0.16) so the dense nuclear stain
does not wash out the other two channels. A new
`test_overlapping_gsplat_layers_declare_order` pins the rule across the audited
set and records why the near misses — side-by-side variants, view- or
time-exclusive nodes, and `kind=partition`/`kind=lod` children of a single node
— are not violations.

Demos with a single gsplat layer are unaffected: with nothing to sort against,
`volumetric` remains the right choice there.
