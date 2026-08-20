#### GSplats and lines chunk bounds no longer lose their extent at large coordinates

`chunk_bounds` is a float32 array while every pad the geometry builders add to it
is a small ABSOLUTE quantity — a gsplat's `coverage_sigma · σ`, a line segment's
full endpoint width, and the `_BARRIER_BOUND_EPS = 1e-3` float-boundary pad on a
categorical axis. Past `|x| ~ 2**23` a float32 ULP exceeds 1, so accumulating
`center ± extent` in float32 and storing it round-to-nearest lands straight back
on the unpadded coordinate: the stored bound comes out TIGHTER than the ellipsoid
or ribbon the renderer draws, and a query at an element's own edge simply does not
fetch its chunk. Nothing reports it — the element is missing, not wrong.

Points were fixed this way in #1658; the same treatment now covers the three
sibling builders. `compute_chunk_bounds_gsplats`, `compute_vertex_chunk_bounds`
and `compute_segment_chunk_bounds` accumulate their intervals in float64 and
narrow them to the float32 store with OUTWARD rounding — a bound is stepped one
ULP away from the interval, but only when the cast moved it the wrong way, so a
padless spatial vertex dimension still stores exactly its own coordinates. The
`_store_outward_f32` helper moved from `points.py` into the shared
`io/_ordering/bounds.py` alongside a vectorised `(d,)` form, and the four builders
now share one `slice_dims` sanitiser that raises on an index outside `[0, ndim)`
rather than each ignoring or crashing on it in its own way (an unresolvable
barrier index would silently cost that categorical axis its tight bounds).

Two consequences worth knowing. First, the effective barrier pad is now
`max(1e-3, one float32 ULP at |x|)`, so a unit-step categorical axis with large
absolute values — a millisecond timestamp, an acquisition index offset into an
experiment — over-fetches a whole neighbouring category above `|x| ≈ 2**23`. That
is the deliberate trade (over-fetching a neighbour beats dropping the chunk at its
own category value); re-base such an axis near the origin if the extra traffic
matters. Second, this changes the stored BYTES of every gsplats and lines spatial
index, so a rebuilt store gets a fresh `content_hash` and warm viewer caches
invalidate on their own. Existing stores are not rewritten and keep their old,
occasionally-too-tight bounds.

The write side is also where the reader-side documentation had drifted: the
gsplats tolerance computer's "coordinates far from the origin" limit described the
expansion being rounded away as a live hazard. It is now the stored coordinate
alone — a genuinely zero-variance dimension still yields a zero-width bound — and
the note says so.
