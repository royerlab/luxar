#### Progressive LOD refinement now stops at a memory ceiling instead of an out-of-memory crash

An additive LOD ladder is prefix accumulation, so its terminal state is 100% of
the node — on every node, always. That bounds first paint, but it bounded
nothing at rest: refinement drove every loader to its last rung with no notion
of what the scene could afford. The Cosmicflows/Laniakea demo (10 line nodes,
11.4 million segments) climbed its ladders until the tab ran out of memory.

The viewer already measured this pressure when *loading* a scene, but that gate
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
the tracked measured footprint.

Sharing one budget across the four types is deliberate — Laniakea's ten line
nodes are each individually affordable and only collectively fatal, so per-type
budgets would have admitted all of them. Where no memory signal is available
(Firefox and Safari expose none), refinement is unbounded exactly as before,
rather than being throttled on a measurement that does not exist.
