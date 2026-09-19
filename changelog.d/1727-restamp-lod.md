#### `luxar restamp-lod` migrates a shipped store's LOD thresholds in place

Every LOD ladder in the bundled demo corpus is on the legacy `coverage` selector
— the diagonal metric — because its thresholds were derived before the
`screen-area` one existed. The values are on the wrong scale for the metric the
viewer now prefers, and re-generating a dataset to fix a handful of numbers is
absurd when nothing about the data has changed: only a `selector` attr and one
`coverage_fraction` per ladder child are wrong. The new `luxar restamp-lod`
command rewrites exactly those, in place, without opening a single array (the
`content_hash` restamp that follows a real change is the one exception — see
below).

It walks the store, and for each `kind=lod` group still on `coverage` (or
carrying no `selector`, which historically meant the same) re-derives the
per-child thresholds by screen-occupancy halving and stamps the group
`screen-area`. The anchor follows the tree writers' own rule rather than a
guess, both clauses of it: a ladder gets the fills-screen (per-tile) anchor when
a real multi-part `kind=partition` encloses it, **and** when one of its own
ladder children is a `kind=partition` — the `overview` recipe's coarse cap,
pinned at fills-screen on purpose so the opening framing shows the cap rather
than pulling in the whole dataset. Everything else — including a ladder under a
ONE-part partition, whose single part's bbox is the whole object — keeps the
whole-object anchor. The resolved binding is threaded down to a nested ladder
just as the writers thread it. Children are ordered coarsest→finest by
`child_index`, so a ten-plus-level ladder does not get its finest threshold
handed to `child_10`. A per-recipe round trip (`levels`, `tiles`, `overview`,
`adaptive`: write, age to legacy, restamp) pins the whole point of the command —
a restamped store is indistinguishable from a freshly written one.

The command is a deliberate opt-in and can never be triggered by anything else.
An authored `coverage_fractions=[...]` list and a legacy derived one are
indistinguishable on disk — the reason the compiler's one-part-partition anchor
check only ever warns — so this rewrite may be overriding a choice somebody made
on purpose. It therefore prints the old→new ladder and the anchor for every
group it touches, offers `--dry-run` and a repeatable `--group` to restrict the
pass, and skips (loudly, with a non-zero exit) anything it cannot convert
honestly: a `selector` outside the vocabulary, which wants `luxar gsplat
migrate-format` first, a finest level whose element count the store does not
record, where fabricating a positive would defeat the derivation's own
"finest LOD level is empty" guard, a stored ladder that descends in the
resolved coarsest→finest child order, where the two disagree about which level
is finest and re-deriving would invert the ladder, or a child that carries a
`coverage_fraction` but no scene-node `type` attr, where re-deriving over the
children that do resolve would leave a partial, non-monotonic ladder with that
rung stranded on its legacy threshold.

The exit code covers one further case, in which the rewrite itself succeeded: a
store carrying neither a scene `type` nor a `.gsplats.zarr` `content_hash` — a
bare `kind=partition` root, say — has no digest to restamp, so nothing tells a
warm viewer cache the ladder moved. At zarr format 2, which is what the legacy
corpus this command targets is written in, even the viewer's `zattrs-hash`
fallback digests the root `.zattrs` bytes, and editing a child's ladder does not
move those. The ladders are still written and the run says so plainly, but it
exits non-zero: the store has to be republished under a new URL prefix or the
migration is invisible to every client that already has it.

It is a sibling of `luxar optimize`, not a flag on it: that pass documents that
every attribute is preserved and refuses same-path work, while this one changes
only attributes and works in place. A group already on `screen-area` is skipped,
so a second run changes nothing at all — including the `content_hash`, which is
restamped and re-consolidated only when something really moved, since an
attrs-only edit still has to invalidate a warm viewer cache. That restamp is the
one part of a run that is not free, and the docs now say so: a compiled scene's
digest covers array VALUES, so moving it streams every array in the store once
(linear in the store's size, however few attributes changed), while a standalone
`.gsplats.zarr` gets a metadata-only stamp. The result is then read back and
verified through both the per-node documents and the consolidated index the
viewer fetches, so a consolidation mistake cannot pass silently.

Writes are all-or-nothing. The pass classifies every group in a read-only
planning walk first, and if a write then fails it restores each attribute it had
already rewritten — an absent `coverage_fraction` back to absent — and re-raises
the original error with a note saying what was rolled back. Left unwound, a
mid-walk failure published a torn ladder: screen-area thresholds under
`selector="coverage"`, the two disagreeing about their units with nothing
downstream able to notice.

The recovery restores the store's `content_hash` digests rather than recomputing
them: each is recorded before the hash pass overwrites it, so a legacy store
hashed by an older walk, a hand-edited one, or a scene whose inner groups carry
no per-group digest is not quietly re-digested by a run that FAILED — and the
failure path stays metadata-only instead of streaming every array in the store a
second time. The metadata is re-consolidated only when the run had rewritten the
ROOT document, which at zarr format 3 is the write that destroys the index: a
failure at the first attribute write needs no root write at all, and
re-consolidating there could turn a fully recoverable failure into a published
store with no index, which the viewer loads as an empty scene. On the success
side an index is likewise rebuilt but never introduced — a store handed over
unconsolidated leaves that way, since `is_consolidated` is how `batch-fit` tells
a finished tile from an interrupted one, and the report now carries
`was_consolidated` so an operator can see why.
