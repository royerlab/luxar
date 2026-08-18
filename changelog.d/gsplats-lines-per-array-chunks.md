#### Size GSplats and Lines arrays per-dtype, like Points already were

`#808` / `#1142` gave every Points array its own dtype byte budget for the
first-axis zarr chunk — a multiple of the spatial-index `chunk_size` atom rather
than exactly one atom — and cut the request count several-fold. GSplats and Lines
were left on the atom in that pass to keep the diff byte-identical, and the reason
was diff minimality, not correctness: the same commit spells out that
"correctness depends only on the `chunk_bounds`/`chunk_size` partition grid, which
is unchanged."

Which left the demo that motivated the work unfixed. DESI is a GSplats scene, so
its `centers` / `cholesky_factors_*` / `amplitudes` / `colors` stayed pinned at a
1,170-row atom — chunks of 2.3–6.9 KB against a 64 KB target, one HTTP request
each. A cold load of the hosted `desi_galaxies` demo issued **8,522 requests for
32.8 MB**, and per-request cost is what a hosted gallery pays for twice: in
latency and in per-operation billing.

Both GSplats write paths and Lines now size each array to its own dtype budget.
Measured cold loads:

| scene | before | after |
|---|---|---|
| 300 K-splat flat leaf | 784 requests | 162 (4.8x) |
| 250 K-splat additive ladder | 675 requests | 144 (4.7x) |

Bytes go slightly DOWN in both cases (larger chunks compress a little better).

This deliberately covers the **standalone `.gsplats.zarr`** writer too, rather
than only the scene compiler. `write_gsplat_arrays` is shared between them, and
`tests/test_scene_leaf_parity.py` exists to keep a scene leaf and a standalone
leaf byte-identical — so sizing only one side would have created exactly the "§8
dual divergent writers" drift that file guards against. Sizing both keeps them
equal. It also means laddered and partitioned scene nodes benefit: those route
through `gsplat_tree.write_gsplat_leaf`, the shared authoring path, so scoping the
change to the plain-leaf writer would have missed every LOD topology — which is
most of the large demos.

Note this brings the standalone writer INTO conformance with its own spec rather
than away from it: GSPLATS_ZARR_FORMAT.md §8 already documented per-array chunk
sizes ("Each array has its own optimal chunk size … not a universal constant") and
tabulates 5,461 elements for `centers` against 16,384 for `amplitudes`. The
implementation was pinning every array to one atom regardless; the §8 *test* was
stricter than the §8 *text*.

Both halves of the split Cholesky get their own width's budget too, rather than
one row count derived from the packed `(N, k)` shape — which had been giving each
half roughly half its byte target (2,340 rows for 3D where a 3-column half
affords 4,680). They are read as two independent row-range slices in the Python
reader and the viewer loader alike, so a shared row count buys nothing. On a
200 K-splat 3D leaf the pair drops from 172 chunks to 86, a third of the leaf's
total requests.

What is no longer true is that a chunk-index range maps to *exactly* one zarr
chunk — it now falls *inside* one, which is what Points has done since #1142. The
atom grid still subdivides the chunk grid (`_atom_aligned_rows` rounds DOWN to an
atom multiple), so a row-range read never straddles a chunk boundary, and that is
the only property the viewer's readers depend on: all three loaders resolve
matched partitions to `{start, end}` row ranges and take a zarr slice per array.
Four assertions that encoded the stricter equality now assert the subdivision
invariant they were really guarding.

Three test-quality fixes came out of verifying this:

* `test_standalone_leaf_matches_scene_leaf_with_colors_and_ordering` compared
  chunk shapes at n=128, where the atom equals the row count and every array gets
  one full-array chunk on both paths — so its chunk-parity assertions could not
  fail. Raised to n=20,000, where a real divergence is visible (confirmed: the
  intermediate scene-only version of this change made 4 of 5 arrays diverge, and
  the old test still passed).
* Lines' `scalars` was the one per-vertex array NOT on the atom grid — 16,384 rows
  against a 3,276 atom (`16384 % 3276 == 4`). `write_lines` passed the whole
  nested `ordering_data` to `write_scalars` instead of unwrapping
  `ordering_data["vertex_ordering"]` the way every sibling array does, so the
  chunk calculator never found a `chunk_size` and fell back to a pure byte budget.
  Pre-existing, and invisible because no test wrote lines with scalars and checked
  alignment.
* `test_gsplats_all_arrays_aligned` ran at n=2,500, where three of its five arrays
  fit in a single full-array chunk that the alignment check skips as trivially
  aligned. Raised to n=20,000 (every array spans several atoms) and it now asserts
  that none of them collapsed back to one chunk, so it cannot go vacuous again.

Verified neutral on two independent axes: every stored array value is
bit-identical across both layouts (flat leaf and both ladder rungs), and rendered
frames are pixel-identical under pinned resolution (`?dpr=1`). At *default*
settings the before/after screenshots differ on ~13% of pixels, which looks like a
regression and is not — it is adaptive DPR reacting to different load timing. With
DPR pinned, two runs of the same store are byte-equal and so are before vs after.
Any future visual A/B of a layout change needs `?dpr=1`.
