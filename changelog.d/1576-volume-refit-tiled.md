#### Volume re-fit works on tiled and stacked data, at every entry point

`--refine volume` was the highest-fidelity rung of the coarse-LOD ladder but was
reachable only when nothing was partitioned and nothing was a barrier — which
excluded every timelapse, since a stacked time axis must be a barrier or
coarsening would blend timepoints. Two guards refused those combinations, both
because a re-fit was handed the whole volume when its seed only explained part of
it.

Both now hand each re-fit the sub-volume it is actually responsible for. That is
one mechanism, not two: a barrier group owns one index along the barrier axes, a
BSP tile owns a box of the spatial axes, and they compose — which is exactly what
a tiled timelapse needs.

The barrier axis is sliced out of the fit rather than held still, because it
cannot be held still: the fitting stack has no freeze mechanism, so a 4D re-fit
would drag splats off their timepoint and widen them along time whatever penalty
it was given. Slicing makes that unrepresentable; the barrier coordinate and
covariance rows come back verbatim from the seed, recombined in factor space so
positive-definiteness is structural. A per-tile re-fit that moves a centre more
than one splat sigma out of its own cell is discarded in favour of the merge,
since the never-worse MSE guard is blind to a splat leaving its tile while the
viewer frustum-culls by part bounds.

Available from all three entry points: `gsplat lod` (with `--target-axes` for a
stacked target), `gsplat fit --recipe levels --refine volume` (no `--target` — the
volume being fitted is already in hand), and `batch-fit merge --recipe levels
--refine volume`, which re-opens the source the manifest recorded and crops it per
tile as that tile streams. The volume is only ever sliced, never read whole, so a
lazy zarr target stays lazy: a 253-timepoint 407x2048x2048 uint16 timelapse is
431 GB while one timepoint is 3.4 GB.

A re-opened batch source is the FULL array, so it still carries the axes the fit
selected a single index of — a channel, and the time axis itself when only one
timepoint was fitted. Those are pinned lazily to the index the fit used, so the
canonical `(t, c, z, y, x)` OME-Zarr shape works rather than having to be reduced
to `(t, z, y, x)` first. What cannot be mapped — several selected channels, a
channel index folded over more than one axis, a missing `--axes` — is refused at
PLAN time, before any tile has been fitted.

Measured +7.7 dB mean on coarse levels of a stacked 3-timepoint fit that
previously could not be refined at all.
