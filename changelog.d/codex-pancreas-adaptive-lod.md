#### CODEX pancreas ships spatial tiles instead of twelve flat leaves

The 2D CODEX pancreas demo built its scene with twelve bare `add_gsplats(...)`
calls over loose `centers=` / `amplitudes=` arrays, and cached each channel with
a bare `.save()`. Both halves of that discard structure: the fit's topology never
reached the artifact, and whatever the artifact held would have been flattened
again on the way into the scene. What shipped was twelve single-leaf nodes over a
25,816 x 18,440 plane — no spatial decomposition, no substitutive levels, and no
streaming ladder, so panning anywhere in a 476-megapixel slide committed every
splat of every visible channel at full detail.

The fit now goes through `luxar.demos._lod_policy.save_with_lod` with the
`adaptive` recipe, and the scene grafts each channel with
`add_gsplats_from_file`. That is the pairing `_lod_policy` documents: `adaptive`
writes a `kind=partition` tree, which has no flat matrix form, so only the graft
can carry it into the scene intact. Each channel now lands as spatial tiles, each
tile choosing its own substitutive level, each level carrying a four-rung
additive ladder underneath — the topology a slide that is panned and zoomed
rather than orbited actually uses. The demo was the first entry on
`test_lod_policy._NOT_YET_ROUTED`, the list that exists to shrink to empty; it is
off it.

Two per-channel transforms went with the arrays, both provably no-ops rather than
losses. The shared-centroid subtraction re-centred all twelve channels by one
common offset "so channels stay aligned", but they are aligned by construction —
twelve fits of the same pixel grid — so a common offset never changed their
relative position, and the viewer frames on the bounding box rather than the
origin. The per-channel `scale_intensity(0.1)` cancels because a colormapped
gsplat layer is windowed by its own `amplitude_data_range`. Dropping the
re-centring also makes the declared `unit="um"` read as true slide coordinates.

The `dim_order=["y", "x"]` on the old adder could not come along: grafting a
multi-part subtree refuses `dim_order`, there being no single matrix left to
remap. Unlike the CMU-1 slide next door, that mapping was load-bearing here and
not the identity — `fit_tiled` returns centers in array-index order, so dropping
it would have transposed the slide and swapped its physical axes. The fit
therefore runs on the transposed plane, which puts the centers in the scene's own
`(x, y)` column order at the source. Doing it there rather than permuting the
finished fit keeps centers, Cholesky factors and the recorded source grid in one
frame; a covariance permutation would have to re-decompose every splat and would
leave the provenance shape describing the other orientation. The pixel grid is
isotropic at 0.325 um, so the transpose costs no metric fidelity.

Resume survives, which matters for a twelve-channel GPU fit behind a 5.9 GB
download. A cached channel is reused by PATH, since the artifact no longer reads
back as a matrix; the in-memory fit is kept only for `--show-roundtrip`, which
now compares in the fitted frame and skips channels it did not fit this run
rather than reporting against splats it does not have. The compression summary
scales its pixel side to the channels actually fitted, so a partially-resumed run
no longer divides this run's splats by all twelve channels' pixels. Current
transposed fits use `.v2` cache names, so older unversioned channel caches are
not reused and may be deleted after the replacement fit completes.
