#### Progressive LOD refinement now stops at a memory ceiling instead of an out-of-memory crash

An additive LOD ladder is prefix accumulation, so its terminal state is 100% of
the node — on every node, always. That bounds first paint, but it bounded
nothing at rest: refinement drove every loader to its last rung with no notion
of what the scene could afford. The Cosmicflows/Laniakea demo (10 line nodes,
11.4 million segments) climbed its ladders until the tab ran out of memory.

The viewer already measured this pressure when _loading_ a scene, but that gate
releases as soon as each node's first paint completes, and refinement is
scheduled afterwards — so by the time the ladders climbed, nothing held a
budget at all.

Refinement now shares one residency ceiling across the sweep-registered
progressive leaves of all four geometry types, sized from the same device-heap
signal the load-time gate uses. Lazy `lod_group` levels are not registered in
those sweeps and remain outside this accounting. The gate is a refusal, never an
eviction: tracked leaves at the ceiling stop adding detail and keep everything
already drawn, so the scene settles at a legible partial view instead of dying.
Where the ceiling is reached, the console says so once, naming the budget and
the tracked measured footprint, without repeating on later view changes.

Sharing one budget across the four types is deliberate — Laniakea's ten line
nodes are each individually affordable and only collectively fatal, so per-type
budgets would have admitted all of them. Browsers without a heap-limit signal,
including Firefox and Safari, use the device-class fallback pool and therefore
receive a ceiling too. The shared working-set calculation is capped at 512 MiB,
so sufficiently large desktop heaps resolve to the same ceiling rather than
scaling without bound.

Superseded or otherwise uncommitted refinement passes now release the previous
cumulative payload they used for prefix-append detection instead of retaining
that abort-path lineage until another pass happens to replace it.

The Scene-Graph monitor now reports the number of ladder rungs actually
committed on screen, rather than the loader cursor. Failed or superseded commits
therefore no longer make a stalled node appear fully refined.
