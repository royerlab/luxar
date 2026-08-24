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
an `auto` request publishes the method it resolved to — and each is deleted outright
when the rebuilt leaf does not publish it, rather than left carrying the inherited
value, which is the one thing the refresh exists to prevent. It summarises ONE
ladder,
following the rule `cull` already uses: the leaf itself for a flat store, and for
a `kind=lod` group the level `lod_substitutive_level` names. Where no single leaf
can be that summary — a `kind=partition`, whose parts hold different counts and
therefore different rung counts — the block is dropped rather than filled from an
arbitrary part, which is also what the `tiles` / `overview` / `adaptive` builders
publish at the root. Present keys only: a store that never carried a summary does
not acquire one.

The same ladder has a second spelling in the same group, and it is handled the
same way: the un-prefixed `n_lods` / `method` / `breakpoints` that `batch-fit
merge --recipe stream` stamps for its per-part recipe. `gsplat additive` over a
batch-fit partition is an advertised use case, and it was dropping `lod_n_lods`
(no single part can back a root rung count) while leaving `n_lods: 6` three keys
away asserting exactly that number under `per_part: True`. `n_lods` is now
refreshed from the ladder and `method` read back off the rebuilt leaf, while
`breakpoints` is dropped rather than refreshed — it holds the build SPEC
(`stream:14000`, `counts:5,15,40`) in a different vocabulary from the leaf's
resolved `lod_breakpoints_kind`, and a spec cannot be read back off the ladder it
produced. `per_part` stays: a per-leaf re-ladder leaves a per-part ladder
per-part. The `levels` branch of the same producer stamps a `method` too, but it
is the substitutive merge method, which a re-ladder does not touch — `lod_kind`
tells the two apart.

The prune family reaches the same key from the other side, and did not handle it
either: `cull` / `filter` (and any rewrite that empties a rung) refresh
`lod_n_lods` / `lod_cutpoints` from the surviving ladder, so a batch-merge store
culled from six rungs to three self-healed the prefixed half while `n_lods: 6`
rode through beside it. The shared refresh now covers both spellings of the
count. Its two siblings stay put here: `method` is the additive *ordering*, which
dropping an emptied rung does not change, and `breakpoints` is the build spec
that was requested — unlike a re-ladder, this rewrite built no new ladder from a
different spec, it pruned the one that spec produced (its prefixed twin
`lod_breakpoints_kind` survives the same rewrite untouched for the same reason).

`gsplat decimate` published its input's `coarsen_dims` whatever it had actually
coarsened over. That key is deliberately exempt from the structure scrub because
the writer reads it back to place the chunk-ordering barrier, so the stale value
did not merely misdescribe the output — a `decimate --coarsen-dims 1,2,3` over an
input stamped `[0, 1, 2]` put the barrier on an axis the merge had just blended,
and a run over an input with no stamp dropped the user's own `--coarsen-dims` on
the floor and fell back to auto-detection. The `merge` family now stamps the set
it resolved, always as an EXPLICIT sorted list: the requested dims for a proper
subset, and the full `[0, …, d-1]` for coarsen-everything (the default, and what
a request naming every dim normalises to). Spelling that case `None` would not
do — `_barrier_from_coarsen_dims` cannot tell a written null from an absent key,
so both read as "no provenance" and land back on `detect_barrier_dims`, which is
a guess about the result's coordinates rather than a "no barrier". What the
guess costs was measured, not assumed: it re-imposes the very barrier the merge
blended over exactly when the reduction leaves the stacked axis' grid INTACT —
timepoints far enough apart that no cluster ever spans two of them — and is
merely redundant when it does not, because a fine integer grid gets averaged
away by the merge and auto-detection then finds nothing on the result. The
explicit list asserts the empty complement on either grid, which is why it is
the honest spelling regardless of which case the data happens to be in.
`make_substitutive_lod` was left writing `None` there for the moment, because
changing it moves the chunk layout of every `lod --recipe levels` build and that
needed a deliberate compatibility decision rather than being folded in here. It
is settled in its own entry, *"`lod --recipe levels` publishes the barrier it
earned, not a `null`"*, which makes the explicit list universal.

The `prefix` family stamps nothing and keeps the inherited value: it merges no
axis, every survivor is one of the input's splats at its own coordinates, so
whichever axes were hard barriers still are — `--coarsen-dims` is a merge-only
knob and publishing it there would be the lie. It now says so — a `UserWarning`,
the same house convention `GSplatData.filter` uses for "your argument had a
surprising effect", so a programmatic `verbose=False` call is not written to
stdout unbidden and the CLI still displays it through `install_arbol_warnings` —
rather than dropping the argument silently, which mattered most under the default
`method="auto"`: the family (and with it whether `--coarsen-dims` is honoured at
all, and therefore the output's chunk layout) flips at the 50 %-kept crossover,
so the same flag was obeyed at `-f 0.4` and ignored at `-f 0.5` with nothing
said. The request is also range-validated for BOTH families before the family is
chosen, so an out-of-range index is no longer a hard error on the merge path and
a silent no-op on the prefix one. The `target >= n_splats` early return is
untouched, as before: nothing changed, so nothing is re-stamped.

One generic post-condition now backs the whole re-ladder rule at the command
level: for every rewriting `gsplat` command — enumerated from the registry, not
from a hand-written list — a published `lod_n_lods` / `lod_cutpoints` / `n_lods`
must match the ladder actually on disk, or be absent. Its fixture publishes BOTH
spellings, taken from the batch producer itself rather than a literal, which is
how the `cull` / `filter` case above was found: with only the `lod_*` half on
disk the un-prefixed key was exercised by nothing but the one hand-written
`additive` test — the "somebody remembered to list it" failure the generic guard
exists to replace. A `kind=partition` root
under `per_part: True` is checked against EVERY leaf rather than rejected
outright: that shape is exactly what `batch-fit merge --recipe stream --n-lods 6`
writes on purpose and it can be perfectly honest, so a blanket rejection would
have gone red on correct output the moment a row reached it.
