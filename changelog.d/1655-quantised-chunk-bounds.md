#### Points and lines chunk bounds now contain the DECODED coordinates, not just the authored ones

A `chunk_bounds` interval is a promise to the reader: if a slice query does not
intersect it, that chunk is never fetched. The points and lines bound builders
computed the interval from the coordinates the writer was handed, but under the
default AUTO encoding those coordinates are then stored as per-axis uint16 fixed
point — so what the reader actually compares its query against is a coordinate
that can have moved half a quantum, `extent/131070`. On a 1000-unit axis that is
7.6e-3, some 125x the float32 ULP the outward-rounding store was added to
preserve, and far more than the `_BARRIER_BOUND_EPS = 1e-3` a categorical axis
gets. A chunk whose own extremum drifted outward is simply not fetched at that
edge, and nothing reports it: the geometry is missing, not wrong. Measured on
real compiles at an axis extent of 1000, 6 of 6 points chunks, 15 of 15 vertex
chunks and 5 of 5 segment chunks stored a bound their own decoded coordinates
escaped.

The encoder now answers the question directly.
`ArrayEncoder.coordinate_round_trip_slack(data, mode)` returns the per-axis
distance a COORDINATE write can move a value — `None` when the write is exact on
every axis — by replaying `_encode_coordinate`'s own exits in its own order,
without writing anything and without repeating its warnings. PRECISION, an empty
or non-2D array, an extent at or above the 2^16 float32 fallback and a
LUT-eligible array are all exact; per axis, a constant axis is exact and so is a
gridded one (the grid snap makes it round-trip bit-exactly, and
`gridded_axis_step` proves it by replaying the encode and the decode); anything
else gets the conservative half-quantum. Keeping the predicate immediately next
to the encoder it mirrors is the point — the two cannot drift apart in separate
files.

`compute_chunk_bounds_points`, `compute_vertex_chunk_bounds` and
`compute_segment_chunk_bounds` take a new keyword-only `coord_slack` vector and
add it outward on EVERY dimension, in float64 before the outward float32 store:
on top of the radius or width footprint on a spatial axis, and on top of
`_BARRIER_BOUND_EPS` on a barrier axis, because the epsilon is float-boundary
safety while the slack is a real displacement and the two add. The compiler fills
it in from the encoder for both geometries — for lines, one vector shared by both
bound sets, since they are both in D-space over the same vertices array. It
defaults to `None` (zero everywhere), so a direct caller of a builder still gets
authored-coordinate bounds byte for byte. A shared `_normalise_coord_slack`
validator rejects a wrong-length, negative or non-finite pad, the first because
the entries are positional per-axis quantities and the second because a negative
pad tightens the very bound it was meant to widen.

Because the slack is per-axis, an ordinary stacked integer time or channel axis
is untouched: it is grid-snapped, so it round-trips exactly, gets slack zero, and
keeps the tight barrier bound that stops a single-timepoint query pulling in its
neighbour. Only axes the encoder genuinely cannot store exactly pay anything. One
consequence worth knowing: a rebuilt points or lines store generally gets
different `chunk_bounds` bytes and therefore a fresh `content_hash`, so warm
viewer caches invalidate on their own. Existing stores are not rewritten.

GSplats are unaffected and unchanged — they already close the same gap from the
other side, escalating an offending centers axis to float32 rather than padding
the bound.
