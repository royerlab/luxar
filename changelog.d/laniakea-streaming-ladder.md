#### Laniakea's basin streamlines paint progressively

The nine `Basin N streamlines` nodes had no LOD structure of any kind — no
substitutive levels, no partition, and no additive ladder. At the `full` preset a
single basin runs to ~617,685 vertices, three times the 200,000 at which
`scripts/check_demo_ladders.py` requires a ladder, so each basin arrived
all-at-once.

They could not be laddered before: `line_type="indexed"` was refused outright,
because the additive multi-LOD writer carries no edge list and rebuilds one by
chaining each connected component in ascending vertex order. Now that the writer's
assumption is *verified* rather than assumed, this layout qualifies —
`index_map[basin_valid] = np.arange(n_vertices)` fills row-major, so each
streamline's vertices are contiguous and time-ordered, and streamlines share no
vertices.

The interior-hole caveat that blocked this turns out not to apply, and the reason
is worth recording rather than relying on the integrator: even if a row had an
interior invalid timestep, `segment_mask = (start >= 0) & (end >= 0)` drops the
crossing segment, so the two index-contiguous runs become two distinct **connected
components** and the writer chains each separately. No edge is invented across the
gap. (It cannot arise anyway — `valid[0] = True` and a failed streamline is
retired permanently, so every row is a contiguous prefix.) Both facts are pinned
by tests that drive `build_basin_line_data` itself, including a deliberately holed
mask, so a change to the integrator surfaces as a readable test failure rather
than a raised build.

This is **not** a fix for the `RangeError: Array buffer allocation failed` that
currently loses six of nine basin nodes on the live site. That is a separate
investigation whose cause is not established — the scene's decoded working set
measures 473 MiB against a 2,000 MiB budget — and a progressive ladder should not
be read as addressing it.
