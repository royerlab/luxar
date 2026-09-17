# luxar.io._compiler.finalize

Finalize-time tree passes for the Luxar zarr compiler. Each function in this
package walks the fully-written zarr scene tree once (post-order) and either
**back-fills** missing aggregate metadata, **validates** authored metadata
against the data that was actually written, or **stamps** a content hash. They
run during `LuxarZarrCompiler.finalize()` — after every node's arrays and
attrs are on disk, before the consolidated metadata is written.

These are deliberately thin, store-only helpers: the compiler exposes each one
as a private `_…` method that supplies the extra context (e.g. scene bounds)
and the bodies live here so `compiler.py` stays a thin orchestration layer.

## File Structure

```
finalize/
├── __init__.py          (empty — functions imported directly by module)
├── amplitude_window.py  harmonize_gsplat_amplitude_windows()
├── blending_warnings.py warn_overlapping_blending()
├── hashing.py           compute_content_hashes()
├── lod_backfill.py      finalize_lod_position_bounds(), finalize_lod_display_types(),
│                        warn_one_part_partition_anchors()
└── validation.py        prune_childless_wrappers(), validate_discrete_dimension_ranges()
```

## API

### `hashing.compute_content_hashes(store) -> str`

Post-order xxhash64 over the whole zarr tree. For each group it hashes, in a
deterministic order:

1. for every array (`array_keys()` sorted), its **storage identity** as sorted
   JSON, then its **decoded values** (`dataset[:].tobytes()`). The identity is
   `name`/`shape`/`chunks`/`dtype`/shard shape, the array's own `attrs`, and its
   codec **ids** (`codec_ids`, derived at either on-disk format) — plus the full
   codec pipeline (`codecs`) when the array is sharded. `_storage_identity` is
   where the reasoning lives: why layout and codec identity count as identity,
   why the per-array `encoding` attrs do, and why codec settings deliberately do
   not.
2. the group's attrs as sorted JSON, **excluding** `HASH_EXCLUDED_ATTRS`: the
   existing `content_hash` (self-reference) and the root's
   `luxar_software_version` stamp (provenance, not content — two Luxar releases
   compiling the same scene must agree on the digest, and an `optimize`
   restamp under a newer release must not churn every viewer's cache). The
   streaming twin in `io/optimize.py` imports the same set, and
   `packages/luxar/src/luxar/io/tests/test_hash_reproducibility.py` fails if either hasher stops
   honouring it.
3. the bytes of any plain **payload file** those attrs name (see below).
4. each child group's **name** (`group_keys()` sorted) together with its
   recursively-computed hash. The name is hashed because a node's own digest does
   not carry it, so digests alone left a renamed child invisible to every
   ancestor.

The resulting hex digest is written back into the group's `content_hash`
attribute, and the root digest is returned. xxhash64 is chosen for speed over
cryptographic strength.

Payload files are non-zarr blobs written straight into a group's directory —
an overlay image (`overlays/<name>/image.png`, named by the group's `image_file`
attr) and a sound node's clip (`<node>/audio.mp3` or `audio.m4a`, named by its
`audio_file` attr). They have no chunk grid and no zarr metadata, so
`array_keys()` and `group_keys()` are both blind to them: before step (3)
existed, two scenes compiled from the same script and differing **only** in
their overlay image bytes got the *same* root hash. `content_hash` is advertised
as a content fingerprint and consumed as one (the viewer's
`scene-identity-watchdog`, cache validation), so a rebuild whose only change was
the logo looked exactly like no change at all. Two walks in the repo hash a
store this way: this compile-time one, and `luxar optimize`'s slab-wise
re-chunk walk (`luxar.io.optimize._compute_content_hashes_streaming`), which
imports `_payload_terms` and folds the same bytes over a store that is already
finished. Nothing else restamps one — swapping the PNG inside an
already-finalized `.luxar.zarr` changes no hash until the scene is recompiled or
re-chunked.

The step is driven off `PAYLOAD_FILE_ATTRS` — the attr keys whose value is a
payload filename — **not** off a directory listing: a listing means enumerating
a group's raw keys and filtering zarr's own documents back out, and
`supports_listing` is not guaranteed by the store ABC, while the attrs already
name the file. There is one targeted exception. A name that differs from a zarr
metadata document only by case (`Zarr.json`, `.ZATTRS`, ...) is checked against
the group's case-exact immediate-child listing before it is read. On a
case-insensitive filesystem an open-by-name would otherwise resolve a dangling
payload onto the real metadata document; because that document carries the
previous `content_hash`, each rewrite would stamp a digest whose input changes
on the next pass. The listing is used only to disambiguate this collision class,
never to discover payloads. If the store cannot answer that targeted probe, the
payload folds the deterministic `unreadable:` sentinel rather than risking the
metadata read.

Payload bytes are read through `_zarr_compat.read_raw_bytes`, which drives the
async `StorePath.get()` (zarr 3.3's public `get_sync()` is opt-in per store and
`ZipStore` does not implement it) — so the READ is store-agnostic, answering the
same way for a local, memory, zip or fsspec-backed store. That is a property of
the read, not of the writer: `write_overlay` writes the image through a
filesystem `Path`, so a payload file only ever exists in a directory store
today.

Three ways a payload can fail to contribute bytes, each folding a distinct
sentinel so they cannot hash alike:

- **absent** — the attrs name a file the store does not hold (distinct from a
  zero-byte one).
- **unsafe** — the name is refused for a *semantic* reason, folded in by name and
  never read. Either it is not a single path component (zarr's `normalize_path`
  rewrites `\` to `/` and raises on a `.`/`..` segment, so such a name addresses
  something outside the group's own directory or nothing at all), or it names one
  of zarr's own metadata documents — those carry the `content_hash` this walk
  stamps, so reading one would make the digest non-convergent.
- **unreadable** — the store raised while reading (`OSError`/`ValueError`, the
  latter covering the `UnicodeEncodeError` a lone surrogate in the name
  produces), the targeted case-exact listing raised (`NotImplementedError`/
  `OSError`/`ValueError`), or the store advertises no listing support for that
  probe. Consequently, the same keys and bytes can hash differently on a
  listing store and a non-listing store for this narrow metadata-name collision
  class. Readability is the store's verdict, and a name heuristic in its place
  would be wrong in **both** directions: a `LocalStore` refuses an over-long
  component that a `MemoryStore` or `ZipStore` reads back fine, or an embedded
  NUL that a `MemoryStore` reads fine, while a short name still fails once the
  group's directory pushes the whole path past `PATH_MAX`. Why these failures
  degrade to a term instead of aborting the compile, and what that costs: see
  the listing and read `except` blocks in `hashing.py`.

Only the metadata-document collision class gets the case-exact listing gate,
because those are the names whose resolved bytes can carry the hash being
stamped. Ordinary payload names keep store-native lookup semantics: for example,
`image_file = "Logo.PNG"` against a stored `logo.png` can still hash as absent on
Linux and as the file's bytes on a case-insensitive macOS or Windows store.

The walk must be **total** over whatever attrs a store on disk actually carries —
including one edited by hand or written by another tool — so any `str` filename
maps to bytes and no name can raise out of the step. Every variable-length term —
key, name, payload — is preceded by its byte length, so the payload block is
injective on its own rather than by relying on step (2) having just folded the
filename in via the attrs JSON. A group naming no payload file adds no terms at
all — a payload-free tree's digest is unchanged from before the step existed.

### `lod_backfill.finalize_lod_position_bounds(store) -> None`

Back-fills a missing `position_bounds` on every `kind == "lod"` group by taking
the **union** of its children's bounds (recursing through nested
`kind="lod"` / `kind="partition"` wrappers and plain groups down to leaves
that carry their own bounds). The convenience-builder path
(`_add_gsplats_as_lod_group`) leaves the LOD parent without bounds because each
leaf carries its own; `kind="partition"` wrappers already persist their union
at write time, so only `kind="lod"` wrappers needed this aggregate. Without it
the viewer's `loadLodGroupNode` saw empty bounds for a nested LOD-of-LOD
construction and skipped that level in projection.

- **Never overwrites** an authored `position_bounds` — fills missing values only.
- Children with empty or mismatched-dimensionality bounds are skipped in the
  union, mirroring the viewer registry's defensive fallback.

### `lod_backfill.finalize_lod_display_types(store) -> None`

Back-fills a missing `display_type` on every `kind == "lod"` group by resolving
it from the **finest** child's own type (recursing through nested
`kind="lod"` / `kind="partition"` groups). LOD children are stored
coarsest→finest, so the finest is the last sorted child. The convenience
builder already sets `display_type` explicitly; this pass covers
explicit-builder constructions where the user called `add_lod_group(...)` with
a mix of leaf types and never set the parent's `display_type`.

- **Never overwrites** an authored `display_type` — fills missing values only.

### `lod_backfill.warn_one_part_partition_anchors(store) -> None`

The one pass here that **reports without writing anything**. A per-TILE
(fills-screen) LOD ladder — one whose `coverage_fraction` thresholds reach the
anchor for the group's own `selector` units (`_tile_anchor`:
`PARTITION_FINEST_AREA` = 1.0 for `selector="screen-area"`, the literal
screen-area fraction; `MAX_COVERAGE_FRACTION` = 4.0 for the legacy
`selector="coverage"` diagonal metric) — is correct only under a real tiling of
**two or more** parts, because a tile's projected bbox is intrinsically a
fraction of the whole object's. Under a **one-part** `kind=partition` that part's
bbox _is_ the whole object, so the ladder holds its finest level back until the
object overfills the viewport.

Every producer that can see the final sibling count already excludes that shape
(`gsplats/lod/recipes.py::build_adaptive` and both gsplat tree writers). The
scene adders cannot: `core/group/lod/group.py::derive_coverage_fractions` is
handed only the insertion point, and part 0's ladder is derived before part 1 has
been added. Finalize is the first moment the count exists.

- One `aprint` warning per offending `kind=lod` group. A ladder is only
  reported when a one-part partition encloses it **and no partition further up
  is a real tiling** — the same rule the writers thread down (`under_partition
  or len(children) > 1`), so neither `partition(2) → partition(1) → lod` nor a
  genuine multi-part partition nested inside a one-part wrapper is blamed. When
  it does fire, the offender named is the **nearest** enclosing partition.
- Never raises and never re-anchors: an authored
  `coverage_fractions=[0, …, 4.0]` list is indistinguishable on disk from a
  derived one, so a silent rewrite would override a deliberate choice.

### `amplitude_window.harmonize_gsplat_amplitude_windows(store) -> None`

Puts every node of a gsplat structure on **one** colormap window. Each writer
derives `amplitude_data_range` per node as `[min(a), p99.9(a)]` of that node's
own amplitudes, which is right for a lone flat leaf and wrong for anything
bigger: on a `kind=lod` ladder a coarse level's merged representatives carry the
same total mass in far fewer splats, so its p99.9 lands ~4.5x above the finest
level's while its typical amplitude grows only ~1.1x — the object re-tones AND
pops ~2x in brightness at every LOD switch. `kind=partition` compounds it
(measured adjacent tiles of one object windowed at `[0.40, 4.04]` vs
`[0.0036, 0.0355]`, ~100x across a single seam).

For every **maximal gsplat structure root** — a `kind in {"lod", "partition"}`
group whose subtree holds at least one `type == "gsplats"` leaf, not descended
past in search of more roots — one reference window is taken and handed down:

- **LOD levels** are scaled by the mass-weighted mean amplitude ratio
  `child / reference`. That estimator measures exactly what a substitutive
  reduction changes; measured on a rasterized proxy of the real colormap render,
  luminance vs the finest level came out 1.04 / 1.01 / 1.00 (chroma L1
  0.048 / 0.028 / 0.028) against 0.49 / 0.51 / 0.71 (0.187 / 0.183 / 0.100) for
  the per-level windows. The p99.9 the window itself uses does not track it at
  all. The ratio is **clamped into `[1/10, 10]`** (`_SCALE_BOUND`, mirroring
  `gsplats/lod/substitutive.py::_MASS_SCALE_BOUND`): the real ratios are
  ~1.1-1.2, `mwma` is a second-moment ratio and so not robust on heavy-tailed
  amplitudes, and a 10x rescale would be a bigger switch pop than the ~2x defect
  this fixes. Out of bounds the ratio is clamped, **not discarded** — at a
  genuine ratio of 0.02, scale 1.0 leaves the level windowed 50x too wide (it
  renders black) while clamping to 0.1 caps the error at 5x. Scale 1.0 is kept
  only for the genuinely unusable cases: missing statistics on either side, a
  non-positive `mwma`, a non-finite quotient.
- **Partition parts** share the window **verbatim**. Parts are disjoint pieces
  of one object with no representation change between them — a dim tile really
  is dim (measured 1.00 / chroma 0.0001 shared, vs 0.62 / 0.123 per-part). Their
  pooled reference is `min(part lows)` (exact) and the **count-weighted mean of
  the part tops** (weights = each part's `n_splats`). Both candidate rules are
  biased and neither recovers the union's true p99.9: `max` over part tops
  drifts UPWARD without bound in the part count (1.05x at 2 parts, 1.58x at 256,
  3.40x at 5000, ~1500x for 60 dim tiles plus one small bright one — the whole
  object renders black), while the weighted mean is biased slightly LOW, since a
  part's own p99.9 already under-estimates the union's. The mean is chosen
  because its bias does not grow with the part count (the streaming merge
  routinely writes thousands) and because its failure mode — clipping
  outlier-bright content — is what a p99.9 window does by design. It is neither
  unbiased nor consistent; do not read it as either.
- **Reference child**: the finest LOD child that actually carries a **usable**
  window (`hi > lo`), walking finest→coarsest — not necessarily the literal
  finest. `add_points/add_lines(substitutive_lod=…)` puts a Points/Lines leaf
  there, a scalar-amplitude level writes no window at all, and a
  constant-amplitude level (every imported classical splat file, via
  `gsplats/interop`) writes a degenerate `[x, x]` one. Any of the three used to
  leave the whole structure un-harmonized, silently. The donor supplies both the
  window and the reference `mwma`. A degenerate part top is likewise excluded
  from the partition pool (its `lo` still counts toward the union minimum).
- **Child enumeration** is name-agnostic. `Node.add_lod_group()` /
  `add_partition_group()` are public, so the child names may be the author's
  (`examples/partition_of_lod_example.py` uses `lod_coarse` / `lod_fine`): levels
  and parts are every child group whose `type` is one of
  `group`/`gsplats`/`points`/`lines`/`mesh` (the type filter keeps a `labels` or
  other auxiliary subgroup from being mistaken for the finest level), minus the
  reserved root buckets `fitting` / `provenance` / `pipeline` (excluded by name
  too, because `pipeline_info` is an open passthrough of caller keys and a stray
  `type` in it would rank `pipeline` last, i.e. "finest"). LOD children are
  ordered coarsest→finest by `child_index` when every candidate has one, else by
  a `child_<i>` numeric suffix, else by sorted name. That last rule assumes
  alphabetically-last is finest — the same convention `lod_backfill.py` uses —
  and is unreachable from any Python producer (`core/node/node.py` always stamps
  `child_index`); it exists for a hand-edited or third-party store. An
  `additive_<i>` sub-LOD keeps the prefix+digit rule — those names are
  writer-owned.
- **Legacy fallback**: a level missing the two mass statistics
  (`amplitude_mass`, `amplitude_mass_weighted_mean`, stamped per leaf by
  `gsplat_assembly.write_gsplat_arrays`) shares the reference window verbatim
  rather than guessing. "Missing" means absent or non-finite — a mass-less leaf
  is stamped `0.0` / `0.0` and does not count as missing, so it no longer drops
  the enclosing structure to scale 1.0. Sharing is not provably better than the
  self-consistent window a legacy sibling already had (it can be clipped by the
  shared one); it is the honest answer when there is no ratio to scale by, and
  it puts the structure on ONE window, which is what the partition arm does
  anyway.

- **Only ever overwrites** an existing `amplitude_data_range`; never creates one
  where the writers left none (a scalar-amplitude leaf, a group wrapper), so
  this is strictly a value correction. It also refuses to write anything that is
  not a finite `lo < hi` (`[x, x]` reads as identity in the viewer, `lo > hi`
  inverts the colormap), and skips the write entirely when the stored value
  already equals the new one — so a re-run, and the reference level itself, cost
  no `zarr.json` rewrite.
- Never raises: a malformed or hand-edited subtree is skipped with a warning and
  the rest of the tree is still processed. It is **not** transactional, though —
  the assignment walk writes incrementally, so a failure partway through one
  structure leaves that structure PARTIALLY rewritten (the warning reports how
  many nodes had already been written).
- Touches nothing but gsplats — points/lines/mesh `scalar_data_range` is left
  alone, and a plain gsplats leaf on its own is a no-op.
- One `aprint` line per harmonized structure, emitted only when at least one
  window actually changed (so a re-run over an already-harmonized store is
  silent), plus at most ONE rolled-up warning per structure reporting how many
  levels were clamped to the bound and the most extreme ratio seen.

A cross-recipe consequence worth knowing: a `levels` structure's reference top
is one level's real p99.9, while an `overview` / `adaptive` one is a pooled
estimate over parts, so the same splats can tone slightly differently depending
on the topology they were written in (measured 222.34 vs 160.50 on one dataset).

### `blending_warnings.warn_overlapping_blending(store) -> None`

Read-only authoring diagnostics over transform-expanded leaf bounds. Effective
`blending_mode` uses nearest-setter-wins ancestor→leaf composition; effective
opacity multiplies down the same chain. The scene root is a carrier and does not
contribute rendering attrs, matching the viewer. Per-type defaults mirror the
viewer (`additive` for Points/Lines/GSplats, `opaque` for Mesh), and source-lock
tests fail if those factories or blend-state predicates drift.

Two hazards are reported: any positive spatial overlap between a node whose
`additive` mode came from its per-type default and a depth-writing node, and
containment between two internally order-dependent, depth-testing nodes that do
not write depth. An explicit `additive` anywhere on the node's ancestry is
treated as intentional X-ray rendering and suppresses the first warning. The
second rule intentionally uses containment rather than every partial AABB
intersection: an empirical scan found pairwise partial overlap too noisy for a
warning authors would keep reading. The final rule set produced one rolled-up
order warning in `multiple_objects_example.luxar.zarr` across 56 materialized
demo/example stores. Non-displayed dimensions use inclusive interval overlap,
while displayed dimensions require positive extent. Every physical leaf resolves
to its author-facing owner: the first geometry node or `kind=lod` /
`kind=partition` wrapper above it. Leaves sharing an owner are never compared,
and warning names and deduplication use that owner so implementation-level LOD
chunks and partition parts report once as the node the author wrote. Candidate
pairs are sweep-pruned on a displayed axis, and repeated pairwise hits are rolled
up into connected overlap clusters. Each diagnosis lists at most five owners,
orders containers first and marks them, then reports how many participants were
omitted. One shared advice block follows all cluster diagnoses instead of
repeating invariant remedies per cluster. Same-geometry, same-mode, same-opacity
clusters can merge their geometry and use `partition={"max_elements": N}` while
preserving placement and appearance. That remedy is available to Points, Lines,
Mesh, and Gaussian Splats. Clusters with differing geometry, blend modes, or
effective opacities instead say that merging cannot preserve the authored
material and fall back to the mode/bounds choices. Additive blending remains an
order-independent option for an emissive medium, but changes surface appearance.

Default-additive/depth-writing warnings print as their pairs are scanned;
order-dependent cluster warnings flush after the scan, so scenes with both see
the additive diagnostics first and the rolled-up clusters second.

### `validation.prune_childless_wrappers(store) -> None`

Post-order cleanup of empty `kind=partition` and `kind=lod` wrapper chains.
Each removal emits a warning naming the path, keeping caught child-add refusals
recoverable without publishing a structurally empty wrapper.

### `validation.validate_discrete_dimension_ranges(store, scene_bounds) -> None`

Emits `UserWarning`s when a discrete, non-displayed dimension's declared
`range` falls outside the actual data extent (read from `scene_bounds`), or
when the data itself sits off the viewer's navigation grid. Catches three
authoring mistakes: a range that starts before any data exists (e.g. range
starts at frame 0 but data starts at frame 1), a range that extends beyond the
data — either of which lets the viewer initialise or navigate to a slice with
nothing in it — and discrete data more than a quarter-step off the
`range[0] + k·step` grid (the viewer snaps navigation to that grid and its
chunk query reaches only a quarter-step around it, so off-grid data can
silently never display).
Comparisons use a quarter-step tolerance (`dim.step / 4`, default `0.25`),
mirroring the viewer's `DISCRETE_TOLERANCE_FRACTION`. A dimension without a
declared step is checked against the unit grid anchored at `range[0]` (the
viewer defaults a missing step to `1.0`). The on-grid check inspects the data min/max only —
interior off-grid values on an otherwise on-grid extent are not scanned.
No-ops when `scene_bounds` is `None` or the store has no `scene_dimensions`
attr.

## How the compiler wires these

`LuxarZarrCompiler.finalize()` calls each pass against the open store. After the
discrete-range check, `prune_childless_wrappers` removes empty wrapper chains
before the LOD back-fills can aggregate over them. The display-type pass then
runs before the position-bounds pass (LOD-of-LOD constructions need a resolved
type before bounds aggregation), the amplitude-window harmonization after
those, the one-part-anchor warning after all three (it only reads), and
the overlapping-blending warning beside it. `compute_content_hashes` runs last
so the stamped hashes cover the back-filled and corrected attrs.

```python
# packages/luxar/src/luxar/io/compiler.py (finalize-time)
from ._compiler.finalize.amplitude_window import harmonize_gsplat_amplitude_windows
from ._compiler.finalize.blending_warnings import warn_overlapping_blending
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
    warn_one_part_partition_anchors,
)
from ._compiler.finalize.validation import (
    prune_childless_wrappers,
    validate_discrete_dimension_ranges,
)
```

`harmonize_gsplat_amplitude_windows` is the one pass here that also runs
**outside** the scene compiler: the two standalone `.gsplats.zarr` writers
(`luxar.gsplats.io.save_gsplats.write_gsplats_tree` and
`write_partition_streaming`) call it just before they stamp their root
`content_hash`, so `luxar gsplat lod/fit/convert` and the batch-fit streaming
merge get the same correction.

## Dependencies

**Internal:**
- `luxar.core.dimensions.Dimensions` — imported lazily in `validation.py` to
  rebuild dimensions from the store's `scene_dimensions` attr.
- `luxar.core.group.lod.group.MAX_COVERAGE_FRACTION` — the single definition of
  the fills-screen anchor, read by `warn_one_part_partition_anchors` so the
  warning cannot drift from what the producers derive.

**External:**
- `zarr` — tree traversal and attribute storage
- `xxhash` — fast non-cryptographic content hashing
- `arbol` — structured `aprint` logging

## See Also

- [io/README.md](../../README.md) — I/O operations, writers, and the compiler
- [core/README.md](../../../core/README.md) — Scene graph, Dimensions, and `position_bounds`
