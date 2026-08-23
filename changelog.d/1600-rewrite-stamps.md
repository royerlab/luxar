#### `additive` and `decimate` re-stamp what they rewrote

Two rewrites preserved the structure KIND of the store they were handed and then
published a stamp that was no longer true of what they had just written — the
stale-value half of #1600, which the topology scrub cannot reach because there is
nothing false about the kind.

`gsplat additive` rebuilds every leaf's ladder, and each leaf's own stats were
already refreshed, but the root `pipeline/` block rode through from the input: a
store re-laddered from three rungs to six went on advertising three, under the
method and breakpoints kind of the ladder that no longer existed. That block is
now rebuilt from the tree actually written — counts and cutpoints from the
ladder, `lod_method` and `lod_breakpoints_kind` read back off the rebuilt leaf so
an `auto` request publishes the method it resolved to. It summarises ONE ladder,
following the rule `cull` already uses: the leaf itself for a flat store, and for
a `kind=lod` group the level `lod_substitutive_level` names. Where no single leaf
can be that summary — a `kind=partition`, whose parts hold different counts and
therefore different rung counts — the block is dropped rather than filled from an
arbitrary part, which is also what the `tiles` / `overview` / `adaptive` builders
publish at the root. Present keys only: a store that never carried a summary does
not acquire one.

`gsplat decimate` published its input's `coarsen_dims` whatever it had actually
coarsened over. That key is deliberately exempt from the structure scrub because
the writer reads it back to place the chunk-ordering barrier, so the stale value
did not merely misdescribe the output — a `decimate --coarsen-dims 1,2,3` over an
input stamped `[0, 1, 2]` put the barrier on an axis the merge had just blended,
and a run over an input with no stamp dropped the user's own `--coarsen-dims` on
the floor and fell back to auto-detection. The `merge` family now stamps the set
it resolved, spelled the way `make_substitutive_lod` spells it (the sorted list
for a proper subset, `None` for coarsen-everything — the default, and what a
request naming every dim normalises to), so the same choice publishes the same
value whichever command made it. The `prefix` family stamps nothing and keeps the
inherited value: it merges no axis, every survivor is one of the input's splats at
its own coordinates, so whichever axes were hard barriers still are —
`--coarsen-dims` is a merge-only knob and publishing it there would be the lie.
The `target >= n_splats` early return is untouched, as before: nothing changed, so
nothing is re-stamped.
