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

The aggregate refusal is latched rather than re-derived, and that matters
because the new check is itself the hazard: a ladder it refuses never loads a
level, so its loaded-level count never grows, `hasMoreLODs` keeps reporting
`true`, and every slice scrub re-fires refinement on a node that can never
succeed — burning its failure budget and toasting "showing a partial surface"
for a surface of which not one triangle was ever committed. The sum is
deterministic once every level's metadata has opened, since nothing is left to
vary, so it is computed once, cached, rethrown as the same error object with no
further preflight calls, and read directly by `hasMoreLODs` so the node leaves
the refinement loop instead of being re-failed forever.

A level's own preflight rejection is not latched: it propagates as itself, and a
later attempt genuinely re-preflights, because the ladder is never marked
admitted until every level has been accounted for and a failed metadata open
caches nothing. A `dispose()` racing the gate — which reaches it as a rejection
from the torn-down sub-loaders — resolves quietly with the ladder's empty
payload instead of being recorded as a load failure. Gating on every level's
metadata before level 0 paints also has a cost on a store without consolidated
metadata (a serialized round of `.zarray`/`.zattrs` opens per level); for a
Luxar-written store, which always carries `.zmetadata`, that cost is close to
zero.
