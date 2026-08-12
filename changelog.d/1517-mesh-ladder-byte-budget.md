#### A mesh reveal ladder's byte budget no longer multiplies per level (#1517)

`preflightMesh` refuses a mesh node whose declared arrays would account for more
than `MESH_DECODE_BUDGET_BYTES` (512 MiB, stored bytes + decoded bytes + the
largest single chunk buffer), before a single chunk is fetched. The mesh reveal
ladder (#1515) builds one `MeshWholeNodeLoader` per `additive_<i>` subgroup and
runs that same preflight independently for each level — so a ladder's real
ceiling had quietly become `n_levels x budget` rather than the budget itself. A
surface written as four levels of ~300 MiB each passes every level's individual
gate while accounting for roughly 1.2 GiB in aggregate and reliably kills the
tab; the same surface written as a plain leaf is refused up front with an
actionable message.

`MeshProgressiveLoader` now charges the ladder's aggregate byte budget on its
own first load — the same point at which a leaf's own budget is enforced
(inside `MeshWholeNodeLoader.fetch()`), and still strictly before any level's
chunks are fetched. It awaits every level's metadata-only preflight in
parallel, sums the levels' accounted bytes, and refuses the ladder as a whole
when the total exceeds the same per-node ceiling. A mesh ladder concatenates
its levels into one committed buffer and keeps all of them resident, so
charging the sum once against one budget is the right quantity — and it is
exactly what turns N budgets back into one.

The first load is also the only place the check can live without losing
something. Run at ladder-construction time there is no node yet to attach a
failure record to, so a level's own transient preflight failure has nowhere
safe to go: propagating it loses the whole node with no retry entry, and
swallowing it lets that level's bytes go uncounted — defeating the check it
sits inside. At first `updateView` the refusal lands in the same contained
path a leaf's fetch already uses, so a level's rejection is recorded, retried
and reported like any other load failure, and the ladder is never latched as
admitted until every level has actually been accounted for.

An aggregate over-budget refusal is now sticky: unlike a level's own transient
preflight failure, the sum comparison is deterministic once every level's
metadata has opened, so a stale re-check on every later `updateView` was
recomputing the same verdict forever and kept `hasMoreLODs` reporting `true` —
enough to keep a dead node in the slice-scrub refinement loop indefinitely.
The refusal is now cached and `hasMoreLODs` reads it directly, so an
over-budget ladder is refused once and then left alone. Gating on every
level's metadata before level 0 paints also has a cost on a store without
consolidated metadata (a serialized round of `.zarray`/`.zattrs` opens per
level); for a Luxar-written store, which always carries `.zmetadata`, that
cost is close to zero.
