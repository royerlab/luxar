#### `lod --recipe levels` publishes the barrier it earned, not a `null`

`make_substitutive_lod` stamped `coarsen_dims: null` whenever the reduction
coarsened every dimension — which is the default, and also what a request naming
every dim normalises to. The writer reads that key back to place the
chunk-ordering barrier (the barrier is its COMPLEMENT), and
`_barrier_from_coarsen_dims` cannot tell a written `null` from an absent key: both
mean "no provenance" and fall through to `detect_barrier_dims` auto-detection.
Every `lod --recipe levels` build that coarsened all dimensions therefore withheld
its own decision from the store and had it re-guessed — the latent fallback #1600
had already removed from `decimate`, left in place there because fixing it changes
chunk layout. This settles that compatibility question the way the project settles
them: spell the full dimension list.

The stamp is now the explicit `[0, …, d-1]`, resolved through the same
`resolved_merge_coarsen_dims` `decimate`'s `merge` family already used — lifted
next to `_normalise_coarsen_dims` (the collapse it compensates for) so the two
producers of this key cannot spell the same choice two ways again. The
`batch-fit merge` per-part record gets the same treatment: it publishes the dims
its parts will actually coarsen, taking the part width from the manifest's
`spatial_shape` — the same field the explicit stacked-time barrier is already
derived from — and still writes `null` on a manifest that never recorded one,
because inventing a width would publish a barrier over columns that may not exist.

**New `levels` builds get a different chunk layout.** On data whose stacked axis
the merge leaves gridded — timepoints far enough apart that no cluster spans two
of them — auto-detection used to re-impose a barrier on the very axis every level
had just been blended over; that barrier is now correctly absent, and chunks are
ordered purely spatially instead of grouping by timepoint. Where the merge does
average the grid away, the fallback was not wrong so much as inconsistent: it
answers each level independently and keeps the barrier on the finest one (which is
the input unreduced), so a single ladder came out with two layouts. Measured on a
200-splat 4D fixture over three timepoints, `-K 4 -L 2`: the widely-spaced grid
went from a mixture of `[]` and `[3]` across the store's 12 splat-holding groups to
a uniform `[]`, and the step-1 grid from the same mixture to the same uniform `[]`.
A build naming a proper subset of dims was always honest and is unchanged.

Existing stores are not touched and still read correctly — a `null` there means
what it always meant. Only newly written stores move, and they move with a fresh
`content_hash`, so republish under a new URL prefix rather than over a URL a warm
viewer cache may still be validating against.
