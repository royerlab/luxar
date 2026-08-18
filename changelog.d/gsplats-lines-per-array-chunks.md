#### Size scene GSplats and Lines arrays per-dtype, like Points already were

`#808` / `#1142` gave every Points array its own dtype byte budget for the
first-axis zarr chunk — a multiple of the spatial-index `chunk_size` atom rather
than exactly one atom — and cut the request count several-fold on large scenes.
GSplats and Lines were deliberately left on the atom in that pass to keep the
diff byte-identical, and the reason was diff minimality, not correctness: the
same commit spells out that "correctness depends only on the
`chunk_bounds`/`chunk_size` partition grid, which is unchanged."

Which left the demo that motivated the work still unfixed. DESI is a GSplats
scene, so its `centers` / `cholesky_factors_*` / `amplitudes` / `colors` stayed
pinned at a 1,170-row atom — chunks of 2.3–6.9 KB against a 64 KB target, one
HTTP request each. A cold load of the hosted `desi_galaxies` demo issued **8,522
requests for 32.8 MB**, and the per-request cost is what a hosted gallery pays
for, twice: in latency and in per-operation billing.

Scene GSplats and Lines arrays now opt into the same per-array sizing. On a
300 K-splat synthetic scene the store goes from **774 logical chunks to 152**
(4.8x fewer: 784 -> 162 HTTP requests over a real cold load), with 2.4% *fewer*
bytes — larger chunks compress slightly better — and a pixel-identical render
(same splat count, same lit-pixel count).

The atom grid still subdivides the zarr chunk grid, so a row-range read never
straddles a chunk boundary; that is the property the viewer depends on, and
`_atom_aligned_rows` guarantees it by rounding DOWN to an atom multiple. What is
no longer true is that a chunk-index range maps to *exactly* one zarr chunk — it
now falls *inside* one, which is what the Points arrays have done since #1142.
Three chunk-alignment assertions that encoded the stricter equality have been
relaxed to the subdivision invariant they were really guarding.

The *standalone* `.gsplats.zarr` tree writer keeps its existing one-atom-per-
chunk layout: `write_gsplat_arrays` is shared between it and the scene compiler,
so the opt-in is threaded as a parameter that only the scene path sets. That
format's `§8` "one formula" invariant — and the `gsplat lod` / CUDA / detached
gsplat-node consumers that read it — are untouched and still gated by their own
test.
