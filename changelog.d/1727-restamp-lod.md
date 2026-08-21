#### `luxar restamp-lod` migrates a shipped store's LOD thresholds in place

Every LOD ladder in the bundled demo corpus is on the legacy `coverage` selector
— the diagonal metric — because its thresholds were derived before the
`screen-area` one existed. The values are on the wrong scale for the metric the
viewer now prefers, and re-generating a dataset to fix a handful of numbers is
absurd when nothing about the data has changed: only a `selector` attr and one
`coverage_fraction` per ladder child are wrong. The new `luxar restamp-lod`
command rewrites exactly those, in place, without opening a single array.

It walks the store, and for each `kind=lod` group still on `coverage` (or
carrying no `selector`, which historically meant the same) re-derives the
per-child thresholds by screen-occupancy halving and stamps the group
`screen-area`. The anchor follows the tree writers' own rule rather than a
guess: a ladder bound to a real multi-part `kind=partition` gets the
fills-screen (per-tile) anchor, and everything else — including a ladder under a
ONE-part partition, whose single part's bbox is the whole object — keeps the
whole-object anchor. Children are ordered coarsest→finest by `child_index`, so a
ten-plus-level ladder does not get its finest threshold handed to `child_10`.

The command is a deliberate opt-in and can never be triggered by anything else.
An authored `coverage_fractions=[...]` list and a legacy derived one are
indistinguishable on disk — the reason the compiler's one-part-partition anchor
check only ever warns — so this rewrite may be overriding a choice somebody made
on purpose. It therefore prints the old→new ladder and the anchor for every
group it touches, offers `--dry-run` and a repeatable `--group` to restrict the
pass, and skips (loudly, with a non-zero exit) anything it cannot convert
honestly: a `selector` outside the vocabulary, which wants `luxar gsplat
migrate-format` first, or a finest level whose element count the store does not
record, where fabricating a positive would defeat the derivation's own
"finest LOD level is empty" guard.

It is a sibling of `luxar optimise`, not a flag on it: that pass documents that
every attribute is preserved and refuses same-path work, while this one changes
only attributes and works in place. A group already on `screen-area` is skipped,
so a second run changes nothing at all — including the `content_hash`, which is
restamped and re-consolidated only when something really moved, since an
attrs-only edit still has to invalidate a warm viewer cache. The result is then
read back and verified through both the per-node documents and the consolidated
index the viewer fetches, so a consolidation mistake cannot pass silently.
