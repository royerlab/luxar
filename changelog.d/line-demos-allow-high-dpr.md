#### Line-dominant demos opt back into high-DPI rendering

High-DPI rendering became opt-in, on the grounds that a 2x display costs four
times the fragment work and soft-edged emissive geometry barely rewards it.
Measuring that claim on a Retina panel showed it holds — with one clear
exception.

Against DPR 2, mean luminance, lit coverage and p99 luminance all hold to within
2.5% on every geometry type: the minimum-drawn-size widening and the shader's
energy compensation very nearly cancel, so the widely-repeated "sub-pixel dots
come out wider and dimmer" is real per-element but invisible in aggregate. The
entire visible effect of the cap is a 15-35% loss of high-frequency detail.

On points and Gaussian splats that reads as slight softening. On dense thin
lines it reads as mush: the strands of a river network, a tractogram or a wiring
diagram stop being separable, which loses information rather than polish. Those
same scenes are the least fill-bound, so they gain least from the cap in the
first place — a trajectory scene measured 1.06-1.17x faster at DPR 1, against
2.6-2.7x for a point cloud. They are simultaneously where the cap hurts most and
where switching it off is cheapest.

So the fourteen line-dominant demos now author `allow_high_dpr=True`: ocean
currents, Cosmicflows, dMRI tractography, global rivers, both particle-collision
scenes, Zebrahub velocity streamlines, the PPI flow field, the bioluminescent
ocean, the FlyWire connectome, the CAIDA AS topology, the Di-PC 3D genome, the
Hilbert curve and the HuRI interactome.

The reasoning is recorded where an author will meet it rather than only in this
entry: on `ViewerConfig.allow_high_dpr`, in the viewer-authoring skill, in the
viewer guide's configuration section and troubleshooting list, on the viewer's
own default, and in the Performance panel's tooltip. Mesh is flagged as the one
geometry worth checking by eye — it is shaded with hard silhouette edges, so the
"soft-edged emissive" argument for the default does not cover it.
