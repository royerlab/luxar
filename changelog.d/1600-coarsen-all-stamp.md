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
next to `_normalise_coarsen_dims` (the collapse it compensates for) so the
producers of this key cannot spell the same choice two ways again. The
`batch-fit merge` per-part record gets the same treatment: it publishes the dims
its parts will actually coarsen, taking the part width from the manifest's
`spatial_shape` — the same field the explicit stacked-time barrier is already
derived from — and still writes `null` on a manifest that never recorded one,
because inventing a width would publish a barrier over columns that may not exist.

Three producers, and only three. `lod --recipe adaptive`, `lod --recipe
overview` and `fit --recipe levels` run a substitutive reduction too and still
publish **no** `coarsen_dims` at all, which the reader treats exactly as the
`null` this entry removes. The cause is upstream of the resolver and unchanged
here: those recipes build their `pipeline/` group from their *input's* stats
rather than from the recipe they just ran, so the choice never reaches the
store — including a `--coarsen-dims` the user typed. `adaptive` accordingly
still reproduces the split-ladder defect described below (measured on the same
fixture: 16 groups at `[]`, 8 at `[3]`, with or without `--coarsen-dims
0,1,2`), and `overview` still gets `[3]` on all 12 groups after its coarse level
blended that axis. Plumbing a composed recipe's own parameters into its record
is a separate change and stays on #1600. (`flat`, `stream` and `tiles` publish
nothing either, but honestly — they coarsen nothing.)

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

**And the new layout travels.** `coarsen_dims` is exempt from the structure
scrub, so every downstream rewrite of a `levels` store inherits the explicit list
and writes the same barrier-free layout: measured on the same fixture, `gsplat
flatten`, `additive`, `partition`, `cull`, `filter`, `slice`, `transform`,
`reencode`, and a `lod --recipe tiles|stream` rebuilt off the `levels` store all
carry `[0, 1, 2, 3]` and write `slice_dims: []` where they previously wrote
`[3]`. That is the exemption doing what it is for, not a hole in it: the
companion entry *"`additive` and `decimate` re-stamp what they rewrote"* keeps
the key alive precisely so a rewrite cannot silently relocate the barrier, and
*"`flatten` / `partition` / `lod` / `decimate` no longer advertise a topology
they don't have"* names dropping it — and smearing every chunk across
timepoints — as the harm a scrub would do. Here the barrier moves *because the
provenance says so*, and it moves in the direction that cannot lose data: a
missing barrier only over-fetches, whereas a false one gives a spatial axis tight
chunk bounds and can drop splats from a query (the asymmetry `detect_barrier_dims`
is written around). None of those commands coarsens anything, so the inherited
list stays true of their output.

The honest edge is the finest level. A ladder's finest level is the input
unreduced, so its stacked axis is still gridded and a barrier there would have
been defensible — `gsplat flatten` of a 4D coarsen-all `levels` store returns
exactly those 200 original splats and now orders them with no barrier at all.
That is the cost of one layout per ladder instead of a heuristic answering each
level separately. A `>3D` build that means to keep a time or channel axis should
say so with `--coarsen-dims`, which `gsplat lod` already warns about when the
flag is absent — the warning now spells out that the finest level loses its
per-timepoint chunk locality too.

Existing stores are not touched and still read correctly — a `null` there means
what it always meant. Only newly written stores move, and they move with a fresh
`content_hash`, so republish under a new URL prefix rather than over a URL a warm
viewer cache may still be validating against.
