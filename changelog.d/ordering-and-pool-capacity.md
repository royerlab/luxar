#### The unsorted frame after every nD re-slice is gone, and a huge scene stops over-allocating

Two unrelated failures on the public gallery turned out to have one thing in common:
both were invisible to every probe pointed at them.

`cosmicflows_laniakea_full` lost six of its nine Lines nodes to "Array buffer
allocation failed" and reported success — a tile rendering a third of its streamlines,
which the site audit read as healthy because it asserted `totalElements > 0`. The
scene's source arrays come to ~473 MiB against a 2000 MiB budget, so the failure looked
impossible until the render side was counted instead: a Lines geometry is 96 B/segment
of RGBA32F element texture plus 8 B/segment of ordering pair, and `chooseCapacity` was
adding 1.5x on top unconditionally. The nine nodes wanted 1702 MiB of pool — 85% of
budget before a single source array — and the byte evictor could not help, because it
disposes only *pooled* buffers and an eager load holds all nine *active*. Capacity
headroom is now capped at 262,144 elements: it exists to absorb a per-slice count
wobble of a few thousand, not a fixed share of however large a node is. That returns
333 MiB. (Complementary to the eager-load admission gate, which bounds how many large
siblings may allocate at once rather than how much each one over-reserves.) The three data accumulators had the same shape of bug for a different reason —
their `while (cap < needed) cap *= 1.5` loop landed on a term of the growth *sequence*
rather than on the count the loader already knew, so six of the nine nodes grew to the
identical 1,594,323 vertices and 27.4% of the reserved slots were never used.

The second was reported as LOD thrashing during timelapse playback: coarse levels
appearing and being replaced once per timepoint. The store has no LOD ladder at all.
What actually happened is that every commit of an order-dependent node wrote a
storage-order fallback and waited ~5 ms for the SortWorker, so at least one frame
rendered unsorted — invisible on a one-off load, and continuous during playback, where
a commit lands at every timepoint. The guard meant to preserve the previous ordering
required an unchanged element count, and an nD re-slice changes the resident count at
almost every step (this demo walks 27834 -> 27643 -> 27416 -> ...), so it never fired.
Commits now hand the adapter the previous count so the existing permutation is
*rebuilt* over the new population instead of discarded, and nodes at or below 250,000
elements are sorted synchronously inside the commit, so the first frame is already
correct. Measured as the fraction of element pairs composited in correct back-to-front
order across a timepoint step: 0.617 before, 0.858 with the rebuild, 1.000 with the
synchronous sort.
