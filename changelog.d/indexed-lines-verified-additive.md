#### Indexed lines can stream when their topology permits, and no longer corrupt edges when it does not

`add_lines(line_type="indexed", additive_lod=…)` had **no guard at all** on the
direct path. The additive multi-LOD writer carries no edge list: it rebuilds one
by chaining each connected component's members in ascending vertex order
(`add_lines_multi_lod_wrapper_impl`). For a component that is not already an
ascending simple path that fabricates edges and drops real ones — and it did so
silently, with no warning and no error, producing a scene whose connectivity was
quietly wrong.

The substitutive path did guard, but with the opposite error: it refused
`line_type="indexed"` wholesale, so data whose topology the ladder *would* have
preserved lost its ladder for nothing.

Both now go through `indexed_components_are_chains`, which tests the actual edge
set rather than trusting or distrusting a `line_type`: each component's edges
must be exactly its consecutive-vertex pairs. Qualifying data — the
polyline-grid layout `dmri_tractography` and `cosmicflows_laniakea` both build —
gets its ladder. Non-qualifying data (a fork, a cycle, a path whose order is not
ascending, a chord) is refused: a `UserWarning` under a substitutive wrapper
where a flat level is a reasonable fallback, and a `ValueError` on the direct
path where there is no level to fall back to.

One case is worth naming because it looks like corruption and is not: a gap in a
producer's vertex numbering. Two index-contiguous but unconnected runs are two
distinct connected components, and the writer chains each separately, so no edge
is invented across the gap. The check accepts it. What it rejects is a vertex of
degree 3 — and note that joining one run's last vertex to the next run's first is
also accepted, because contiguous numbering makes that a legitimate longer chain.
