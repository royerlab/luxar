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
`ArrayEncoder.coordinate_round_trip_slack(data, mode, *, allow_lut=True)` returns
the per-axis distance a COORDINATE write can move a value — `None` when the write
is exact on every axis — by replaying `_encode_coordinate`'s own exits without
writing anything and without repeating its warnings. PRECISION, an empty array
and an extent at or above the 2^16 float32 fallback are exact; per axis, a
constant axis is exact and so is a gridded one (the grid snap makes it round-trip
bit-exactly, and `gridded_axis_step` proves it by replaying the encode and the
decode); anything else gets the conservative half-quantum. A non-2D array
answers `None` because it is not the `(N, d)` shape the per-axis answer is
defined for — NOT because it is exact (a 1-D COORDINATE array is quantised like
any other), and the bound builders all require `(N, d)`. Non-finite input
answers `None` rather than a NaN entry that the bound validator would reject
with a message pointing at the wrong culprit. Keeping the predicate immediately
next to the encoder it mirrors is the point — the two cannot drift apart in
separate files.

`allow_lut` is the one exit the CALLER has to declare, and it mirrors
`ArrayEncoder.encode`'s parameter of the same name. A LUT stores values verbatim,
so a LUT-eligible array is exact — but only where the write is allowed to reach
for a LUT. The lines writer encodes `vertices` with `allow_lut=False` (the
spatial-index loader reads that array as raw chunked zarr), so a lines vertices
array with ≤256 distinct values is quantised on disk while looking eligible; the
lines glue passes `allow_lut=False` to match, and the points glue keeps the
default because `write_positions` does not block LUT and such a points node
genuinely stores `lut_uint8`. Measured on a 4D lines compile with a 250-value
irregular palette (60,000 segments / 120,000 vertices): treating eligibility as
exactness left 30 of 44 vertex chunks (8,203 rows) and 7 of 15 segment chunks
(487 endpoints) outside their own bound; declaring `allow_lut=False` takes all
four counts to zero. Those counts are that fixture's — the segment half in
particular depends on how the polylines are wired, and a connectivity that keeps
every endpoint interior shows the vertex half alone. The zero after the fix is
what holds regardless.

Because the predicate is asked on every compile, its cost is part of the fix.
The LUT probe is a whole-array `np.unique` and the dominant term whenever it
runs, so it is now gated twice. It is asked LAST — after the cheap exits and the
per-axis grid loop, and only when some axis came out with nonzero slack, since
otherwise the answer is `None` regardless — and it is asked only when no single
AXIS already holds more distinct values than a scalar-mode LUT can address
(`LUT_SCALAR_MAX_DISTINCT`, 256; a COORDINATE array is never the ≤4-channel 2-D
COLOR shape that reaches the uint16 ROW-mode tier). One column above that cap
proves the whole array cannot LUT-encode, because a column's distinct values are
a subset of the array's — and the count is free, since the grid loop's own
`np.unique` supplies it. On 1M×3 float32 continuous coordinates — the common
points path, which previously paid the probe in full — the predicate drops from
467 ms to 120 ms, and at 5M×3 from 3.41 s to 0.79 s, i.e. down to what the
`allow_lut=False` lines path always cost. An all-gridded array is 64 ms. The
irreducible case is an array with ≤256 distinct values per axis but more than
256 overall; only a whole-array pass can settle that one.

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

Keeping the two models in step is now a mechanism rather than a comment. The
writer carries a SECOND, independent model of what the encoder does — four
replayed exits — and this change's own history shows that is easy to get wrong
(the LUT exit was wrong for lines on the first attempt). A parametrised corpus
in `encoding/tests/test_coordinate_quant.py` therefore asks the predicate AND
performs a real encode/decode of the same array, for every mode × `allow_lut`,
and requires per-AXIS agreement in both directions: an axis the predicate calls
exact must be bit-exact on disk, and an axis it pads must not move further than
the pad (plus half a float32 ULP, the decode contract). Dropping the `allow_lut`
gate, dropping the grid snap, or giving `_encode_coordinate` a lossier tier each
turn it red.

GSplats get no `coord_slack` here — they close most of the same gap from the
other side, escalating an offending centers axis to float32 rather than padding
the bound. Note that this is *most*, not all: the σ argument that justifies
skipping the pad holds on a SPATIAL axis, where `truncation_radius · σ` absorbs
the residual displacement, and not on a BARRIER axis, which by design gets only
`_BARRIER_BOUND_EPS`. A gsplats barrier axis that is neither gridded nor
LUT-encoded, on splats whose σ is large enough to keep the escalation rail
quiet, can still decode outside its own chunk bound (reproduced: 2 of 12,000
centers, worst 2.6e-3). That is out of scope here and is tracked separately; the
format guide and `compute_chunk_bounds_gsplats` now say so instead of implying
full coverage. The same is true of the SCALAR half of a points/lines footprint:
`radii` and `widths` are quantised too, and the pad is built from the authored
value, so a decoded radius/width can still escape by up to half its own quantum
(measured 9.6e-3 on `radii ~ U(0.1, 5.0)`). Both docstrings now carry that
caveat rather than claiming a guarantee they do not have.

The gsplats bound builder does change in this release, just not here — see the
companion entry on outward float32 rounding, which reworks
`compute_chunk_bounds_gsplats` (and both lines builders) so a pad is never lost
to a round-to-nearest float32 store. Read the two together: "no `coord_slack`
for gsplats" is not "gsplats unchanged".
