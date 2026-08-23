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
so both read as "no provenance" and land back on `detect_barrier_dims`, which on
an integer-gridded stacked axis re-imposes the very barrier the merge blended
over. `make_substitutive_lod` still writes `None` there and carries the same
latent fallback; it is left alone on purpose, because changing it would move the
chunk layout of every `lod --recipe levels` build, and the divergence is recorded
on #1600 instead.

The `prefix` family stamps nothing and keeps the inherited value: it merges no
axis, every survivor is one of the input's splats at its own coordinates, so
whichever axes were hard barriers still are — `--coarsen-dims` is a merge-only
knob and publishing it there would be the lie. It now says so on the console
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
must match the ladder actually on disk, or be absent.
