# luxar.io - I/O Operations

Progressive writing and reading of Luxar Zarr scenes with spatial ordering for memory-efficient processing of massive datasets.

## Quick Start

Write and read a scene in 3 steps:

```python
from luxar.io import LuxarZarrCompiler, LuxarScene
from luxar.core.dimensions import Dimensions
import numpy as np

# 1. Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)
dims = Dimensions.default_3d()

# 2. Write to zarr (progressive - data written immediately)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("cloud", positions, colors, radii=0.1)

# 3. Read it back (memory-efficient lazy loading)
scene = LuxarScene.load("scene.luxar.zarr")
points = scene.get_points("cloud")
print(f"Loaded {points['positions'].shape[0]} points")
```

**Key Benefits**:
- Progressive writing - no intermediate caching, handle TB-scale data
- Spatial ordering - better compression and viewer performance
- Scalar convenience - pass uniform values directly: `radii=0.5` instead of `np.full(N, 0.5)`
- Memory-efficient reading - lazy loading via zarrita

## Purpose

This package provides infrastructure for:
- **Writing**: Progressive writing of Points, Lines, GSplats, and Mesh data to Zarr archives
- **Reading**: Full read-only access to Luxar scenes via `LuxarScene`

Both paths enable processing of TB-scale datasets on GB-scale machines.

## Main Components

### LuxarZarrCompiler

Main entry point for creating Luxar scenes with progressive writing.

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with (
    LuxarZarrCompiler(
        "scene.luxar.zarr",
        encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
        ordering_method="hilbert",  # "hilbert" (default, best locality) or "morton" (fastest)
        enable_spatial_index=True,  # Apply spatial ordering
    ) as compiler
):
    scene = compiler.create_scene(dimensions=dims)

    # Full arrays
    scene.add_points("cloud", positions, colors, radii)

    # Scalar convenience - no intermediate arrays!
    scene.add_points(
        "uniform",
        positions,
        radii=0.5,  # Scalar instead of np.full(N, 0.5)
        colors=(1.0, 0, 0),  # Tuple instead of np.full((N,3), [1,0,0])
        sharpness=0.5,
    )  # Scalar instead of np.full(N, 0.5)
```

**Key Features**:
- Progressive writing (data written immediately, not cached)
- Filterable oversized-node diagnostics via
  `luxar.io.ElementCapacityWarning` (for example, promote them with
  `warnings.simplefilter("error", ElementCapacityWarning)` in build checks).
  Install that filter in-process; Python startup `-W` / `PYTHONWARNINGS`
  filters cannot resolve this package warning category before imports run.
  A broad `warnings.simplefilter("error")` around compilation also promotes
  these diagnostics and will fail if any node exceeds the conservative cap.
- Single-file `.zip` output with a normalized inner store name, published after successful finalization
- **Scalar convenience**: Pass uniform values directly (no `np.full()` needed)
- Morton/Hilbert spatial ordering for better compression
- Compound ordering for nD data (discrete/barrier dims → spatial curve within)
  - Shared `_compound_sort` core drives **all three geometries** (Points, Lines,
    GSplats): categorical/barrier axes (time, channel) are lexsorted first, then
    a space-filling curve orders spatially *within* each barrier value, so a
    chunk never straddles a category. GSplats derive the barrier authoritatively
    where known — the scene compiler uses the scene's `Dimension.discrete`
    non-displayed axes (exactly like Points/Lines), and the batch merge uses the
    stacked-time axis — falling back to the persisted LOD `coarsen_dims`
    complement, then a conservative value-based auto-detect (`detect_barrier_dims`,
    strict integer + low-cardinality) only for provenance-less standalone files.
- **Dimension-aware spatial indexing**: Optimized for time-series and nD slicing
  - Tight discrete/barrier bounds (categorical axes get exact-value bounds
    padded only by a float-boundary epsilon `_BARRIER_BOUND_EPS`, no σ/radius
    expansion — a splat at time=0 never extends into time=1's chunk bounds;
    the reader's per-dimension query tolerance owns the reach)
  - Smart chunk sizing (smaller chunks for animated data)
  - ~7× performance improvement for time-animated Lines; barrier-aware GSplat
    ordering makes per-timepoint reads hit the ideal ~1/T of chunks (vs a
    chaotic 2.5–3.5× over-fetch when time is smeared across chunks)
- Semantic type-based encoding (via `luxar.encoding`)
- Automatic chunk size calculation
- Metadata consolidation for fast loading

### LuxarScene (Reading)

Read-only access to Luxar zarr scenes with automatic decoding.

```python
from luxar.io import LuxarScene

# Load a scene
scene = LuxarScene.load("scene.luxar.zarr")

# Scene metadata
print(scene.version)  # "0.1" (LUXAR_VERSION_CURRENT)
print(scene.dimensions)  # Dimensions object or None
print(scene.path)  # Path to zarr store

# List nodes by type
print(scene.list_points())  # ['cloud1', 'cloud2']
print(scene.list_gsplats())  # ['splats1']
print(scene.list_lines())  # []
print(scene.list_groups())  # ['group1']

# Check if a node exists
if scene.has_node("cloud1"):
    print(scene.get_node_type("cloud1"))  # 'points'

# Get node metadata (without loading array data)
metadata = scene.get_node_metadata("cloud1")
print(metadata["type"])  # 'points'
print(metadata["n_points"])  # Number of points

# Get full point data with automatic decoding
points = scene.get_points("cloud1")
print(points["positions"].shape)  # (N, 3)
print(points["colors"].shape)  # (N, 3) or None
print(points["radii"].shape)  # (N,) or None
print(points["metadata"]["transform"])  # 4x4 numpy array (if present)

# Reconstruct the finest cloud from substitutive LOD, partition, and additive
# increments. Read one field when a large array_ref should not be materialized.
finest = scene.get_points("structured_cloud", flatten=True)
colors = scene.get_point_array("structured_cloud", "colors", flatten=True)

# Similarly for GSplats and Lines
splats = scene.get_gsplats("splats1")
print(splats["centers"].shape)  # (N, 3)
print(splats["cholesky_factors"].shape)  # (N, 6)

# A node whose colormap attr is the sentinel 'custom' carries its palette as a
# LUT dataset (the writer resolves any non-builtin name or array to that pair)
print(scene.get_colormap_lut("cloud1"))  # (256, 3) uint8, or None

# Scene-level viewer hints, parsed like Scene.viewer_config (None if unset).
# A rewriter must carry this across, or the output silently falls back to the
# viewer's ACES default.
print(scene.viewer_config)  # ViewerConfig(...) or None
```

**Return Types** (dataclasses with dict-compatible access):
- `PointsData`: positions, colors, radii, sharpness, chunk_bounds, metadata
- `LinesData`: vertices, widths, colors, sharpness, segments, metadata, indices
- `GSplatsData`: centers, amplitudes, cholesky_factors, colors, chunk_bounds, metadata

**Key Features**:
- Automatic decoding (broadcasting, LUT, quantization, array_ref)
- Full scene introspection (list nodes, check types, get metadata)
- Lazy loading (arrays only loaded when requested)
- Node types as strings ('points', 'gsplats', 'lines', 'group')
- Transform matrices automatically converted from zarr storage format

### Spatial Ordering Module

`luxar.io.ordering` provides Morton and Hilbert curve ordering.

```python
from luxar.io.ordering import sort_points_compound, compute_chunk_bounds_points

# Apply compound ordering to Points
sort_indices, metadata = sort_points_compound(
    positions,
    dimensions,  # List of Dimension objects
    method="hilbert",  # or "morton"
)

# Compute chunk bounds (radii can be per-point array or scalar)
chunk_bounds = compute_chunk_bounds_points(
    sorted_positions,
    sorted_radii,
    chunk_size=2048,
)
```

**Compound Ordering** (for nD Points):
1. **Primary sort**: Discrete dimensions (time, channel) - lexicographic
2. **Secondary sort**: Morton/Hilbert code of spatial dimensions

This ensures:
- Time-slice queries load contiguous chunks
- Spatial locality preserved within each slice
- Works with both Morton and Hilbert curves

**Functions**:
- `sort_points_compound()`: Compound ordering for Points
- `sort_splats_spatial()`: Simple spatial ordering for GSplats
- `sort_segments_compound()`: Compound ordering for Lines segments
- `compute_chunk_bounds_points()`: Chunk bounds with radius extent (`radii=None` ⇒ the renderer's `DEFAULT_POINT_RADIUS`, so the bound is never tighter than the drawn disc)
- `compute_chunk_bounds_gsplats()`: Chunk bounds with ellipsoidal extent
- `morton_encode_nd()`: Morton (Z-order) encoding
- `morton_encode_128bit()`: Morton encoding for 128-bit coordinates
- `hilbert_encode_nd()`: Hilbert curve encoding

### Writer Protocol

`ZarrWriterProtocol` defines the interface for Zarr writers, enabling different
implementations while maintaining API consistency.

Its one implementation, `LuxarZarrCompiler`, is genuinely type-checked against
it. That was not always so: every geometry write method carried
`# type: ignore[override]`, hiding both an annotation split (this module used to
redefine `PositionArray` / `ColorArray` *wider* than `typing_utils.aliases`
spells the same names — see the comment there) and a real signature bug, where
the protocol's `write_mesh` omitted the eleven texture parameters the compiler
takes between `scalars` and `shading`.

`tests/test_writer_protocol_agreement.py` keeps it honest: parameter names,
order and defaults for every declared method, plus a probe that writes a
real store per declared input dtype (float16 positions, uint8/uint16 colors,
float16/uint8 scalar attributes) so the aliases are pinned to what the write
path accepts rather than to what either signature claims.

### Re-chunking an existing store (`optimise.py`)

`luxar.io.optimise` re-chunks a store that is **already on disk**, in one
structure-preserving pass — no refit, no source volume, no GPU. It backs the
`luxar optimise` CLI command and the `luxar info --stats` chunk diagnostic.

Everything but the zarr chunk grid survives verbatim: values bit-for-bit, dtype,
codecs, filters, serializer, `fill_value`, memory order, the on-disk zarr format,
the plain non-zarr payload files a group's attrs name (an overlay image, which
no array or group API reaches), and every group and array attribute **except**
the two the pass is contractually required to move — the root's `content_hash`,
which is restamped, and the `chunk_layout` summary written beside it (see *Cache
invalidation* below).

```python
from luxar.io.optimise import optimise_store, plan_optimisation, summarise_chunk_layout

plan = optimise_store(
    "scene.luxar.zarr", "out.luxar.zarr", target_bytes=65_536, verify=True
)
print(plan.source_n_chunks, "→", plan.target_n_chunks)
```

- `plan_optimisation(root, target_bytes=…)` → `OptimisePlan` — what would change,
  per array, without writing. Each `ArrayPlan` carries the source/target chunk
  shape, the resolved spatial atom, and a `skip_reason` when the array is left
  alone.
- `optimise_store(src, dst, …)` — does it. `verify=True` re-reads the output and
  compares every array — and every payload file it copied — byte for byte,
  reporting both counts.
- `summarise_chunk_layout(root)` → `ChunkLayoutSummary` — average chunk bytes,
  arrays under the 16 KB floor, and the chunk-file count a full load fetches.
  Counts objects, so a shard is one file and a `(0, D)` placeholder is none —
  and an array that fetches nothing is left out of the floor share entirely.
  `summarise_plan(plan)` is the same diagnostic off a plan already walked, which
  is how `luxar info --stats` reports both from a single pass.
- `resolve_target_bytes(target_bytes=…, target_kb=…, profile=…)` — the three
  mutually exclusive size flags, and `CHUNK_PROFILES` (`hosting` 256 KB,
  `local` 64 KB, `archive` 1 MB).

**What must not move.** `chunk_size` / `chunk_bounds` are the viewer's partition
grid: every emitted chunk is a whole multiple of the node's atom (rounded down
from the byte budget, never below one atom), so a row-range read never straddles
a boundary. Lines' two atoms — `vertex_ordering.chunk_size` for the per-vertex
arrays, `segment_ordering.chunk_size` for `segments` — are resolved separately,
and the bounds arrays themselves are never re-chunked. A `chunk_size` attr is
trusted **only** when the matching bounds array exists, because a gsplat leaf
written with `ordering="none"` still gets a vestigial one stamped — and that
vestigial value is a power of two, so "the array's chunk is already a multiple
of it" is arithmetic coincidence rather than proof. (Every Luxar writer omits
the bounds array only for a zero-row node, so a node with rows to re-chunk
always carries its proof.) `array_ref` placeholders (`(0, D)`), `(1,)`/`(1, k)`
broadcasts and sharded arrays are copied verbatim — a sharded array keeps its
**shard** grid, not just its inner chunk shape; nothing is chunked smaller than
it already is; the chunk **key** layout (v2 `dimension_separator` / v3
`chunk_key_encoding`) survives; and the copy walks chunk-aligned slabs rather
than reading an array whole.

**All-or-nothing.** The output is built in a hidden sibling directory and
renamed onto the destination only after the copy (and `verify=True`, when asked
for) succeeds, so a failure leaves neither a half-written store at the
destination nor a damaged previous one. An existing destination is renamed
**aside** and deleted only once the new one is in place (and restored if that
rename fails), because deleting first can lose both copies: an `rmtree` that
fails partway propagates before the artifact is marked consumed, and for a
directory destination the artifact *is* the staging tree. A `.zarr.zip`
destination is compressed out of that directory rather than written into a
`ZipStore`, which appends rather than replaces and would otherwise accumulate
one dead copy of every group document per attr write. `overwrite=True` replaces
an existing **zarr store or empty directory** only, and refuses a destination
that is, contains, or lives inside the source — or that is a **symlink**, since
the rename would replace the link rather than its target (the error says to pass
the target instead).

**Cache invalidation.** The viewer validates its persistent cache on
`content_hash`, and that cache holds encoded chunks keyed by chunk index — so a
re-chunk that left the hash where it was would serve bytes that no longer mean
what their keys say. Both hashers now fold layout in themselves:
`compute_content_hashes` hashes each array's storage identity (name, shape,
dtype, chunks, shards, codec ids, own attrs) before its values, and the
`.gsplats.zarr` stamp folds the same identity terms over metadata alone. So
recomputing over the re-chunked output lands on a different digest by
construction, and the RESTAMP is what makes that reach the viewer — nothing else
rewrites the stored `content_hash`. It runs for `--generic` too, because that
flag describes the input rather than the output's cache safety. The
`chunk_layout` root attr is folded in as well (attrs are hashed): belt-and-braces
against a layout-aware hasher, and still the whole guard for a store that carries
no `content_hash` to restamp, where the viewer falls back to a digest of the raw
root document bytes. The scene restamp is a slab-wise reimplementation of the
finalize-time walk — the same digest, without the whole-array materialisation
that would peak at twice a 629 MB array's size.

### Re-deriving LOD thresholds in an existing store (`lod_restamp.py`)

`luxar.io.lod_restamp` rewrites the LOD switch thresholds of a store that is
**already on disk**, in place. It backs the `luxar restamp-lod` CLI command. The
sibling of `optimise.py`, deliberately not a flag on it: that pass preserves
every attribute and refuses same-path work, this one changes **only** attributes
and moves no chunk.

```python
from luxar.io.lod_restamp import restamp_lod_store

report = restamp_lod_store("scene.luxar.zarr", dry_run=True)
for group in report.restamped:
    print(group.path, group.anchor, group.old_thresholds, "→", group.new_thresholds)
```

Every `kind=lod` group still on the legacy `coverage` diagonal metric (or
carrying no `selector` at all, which means the same) has its per-child
`coverage_fraction` re-derived by screen-occupancy halving —
`partitioned_coverage_fractions` when the group is TILE-BOUND,
`coverage_fractions` otherwise — and its group stamped `screen-area`. Children
are ordered coarsest→finest by `child_index`, and a group already on
`screen-area` is skipped, so a second run is a no-op down to the `content_hash`.

Tile-binding is both gsplat tree writers' full rule, `under_partition or
any(isinstance(c, GSplatPartition) for c in on_disk)`, read off the store — and
it has two clauses, not one:

- **ancestry** — an enclosing `kind=partition` that is a REAL tiling (>1 part;
  a one-part partition's single part IS the whole object);
- **own children** — one of this lod group's own ladder children is a
  `kind=partition`. That is the `overview` recipe's `[coarse_leaf,
  fine_partition]` cap, which `partitioned_coverage_fractions` documents as a
  deliberate product contract. Miss this clause and an `overview` cap comes back
  at the whole-object anchor, i.e. the viewer loads the entire dataset at the
  opening framing — the one cost that recipe exists to avoid.

The binding a lod group resolves is threaded down to its own descendants, as the
writers thread `under_partition=partition_bound`.

- `restamp_lod_store(path, *, dry_run=False, groups=None)` → `RestampReport` —
  the groups restamped, skipped-as-current, skipped-as-unsupported and
  skipped-as-unresolved, plus the new `content_hash` (with a
  `content_hash_status` of `unchanged` / `restamped` / `unstampable`, since a
  `None` hash alone cannot distinguish "nothing changed" from "this store
  carries no digest to move") and any re-verification residual. `report.clean`
  is False when anything was left alone for a reason the caller must act on —
  and also on `unstampable`, which can only happen after a real rewrite: the
  ladders landed but no digest moved, so a warm viewer cache goes on serving the
  old ones (at zarr format 2 the `zattrs-hash` fallback digests the root
  `.zattrs`, which a child's ladder edit does not touch either) until the store
  is republished under a new URL prefix. The CLI keys its exit code on it.
  `report.was_consolidated` says whether the store carried a consolidated index
  when the run started — and therefore whether it has one now.

**Never automatic.** An authored `coverage_fractions=[...]` list and a legacy
derived one are indistinguishable on disk — the point
`_compiler/finalize/lod_backfill.py::warn_one_part_partition_anchors` makes
normatively, which is why that check only warns. Calling this IS the opt-in, and
the per-group old→new ladder is printed as the audit trail for a rewrite that may
be overriding a deliberate choice.

**What is refused, and what is skipped.** A compressed store is refused (an
archive is read through a temp directory, so in-place is impossible), as is a
store that is not a Luxar scene or `.gsplats.zarr` tree. A `selector` outside the
vocabulary (`pixel_size`, the pre-v3.2 gsplats spelling) is REPORTED and left
alone rather than converted, and so is a ladder whose finest child records no
element count — the derivation's "finest LOD level is empty" guard reads that
count, and fabricating one would defeat it on exactly the store that needs it. A
coarser level's missing count is harmless (only the ladder's length and the
finest entry are consumed) and is reported as `None` rather than invented. A
ladder whose stored thresholds DESCEND in the resolved child order is refused
too: the order and the thresholds disagree about which level is finest, so
writing an ascending ladder onto that order would silently invert it. So is a
group with a child that carries a `coverage_fraction` but no scene-node `type`
attr — the node filter drops it, and re-deriving over the rest would write a
PARTIAL ladder, leaving that rung stranded on its legacy threshold (possibly
above the screen-area ceiling of 1.0, where nothing can ever select it).

**Cache invalidation, and what it costs.** When something changed,
`_restamp_content_hash` runs and the metadata is re-consolidated, in that
order — an attrs-only edit must still invalidate a warm viewer cache. This is
the only part of the pass that is not free: a compiled SCENE's digest is over
array VALUES, so the restamp streams every array in the store once (linear in
total store size); a standalone `.gsplats.zarr` takes the metadata-only branch.
A dry run, and a run that changes nothing, hash nothing. Then the store is read
back and verified through BOTH readers, because they can disagree and the
disagreement is the failure worth catching: `open_group` reports the per-node
documents, while `read_consolidated_attrs` reports the root index, which is the
only thing the viewer fetches. An index is REBUILT, never introduced — a store
that arrives unconsolidated leaves that way (`is_consolidated` is `batch-fit`'s
finished-tile sentinel, so writing one would mark an interrupted tile complete),
and the verifier then expects no index rather than reporting its absence.

**All-or-nothing writes.** Every group is classified in a read-only planning
walk before anything is written; if a write then fails, each attr already
rewritten is restored (an absent `coverage_fraction` back to absent) and the
original error is re-raised with a note saying what was rolled back. Without
that, a mid-walk failure leaves a TORN ladder — a screen-area threshold under
`selector="coverage"` — which is the silent, unrecoverable disagreement
`resolve_lod_ladder` warns about.

The recovery is a pure RESTORE, digests included: every `content_hash` is read
into the same undo ledger before the hash pass overwrites it, so a store whose
stored digest is not what a fresh recompute yields — a legacy one, a
hand-edited one, a scene whose inner groups carry none — comes back carrying
exactly what it came in with, and the failure path stays metadata-only instead
of streaming every array again. The index is re-consolidated only when the run
had rewritten the ROOT document (the write that destroys a format-3 index): a
failure at attr write #1 needs no root write at all, and re-consolidating there
would turn a recoverable failure into a store with no index — which the viewer
loads as an empty scene.

### Screening a store's ladders against the opening shot (`lod_screening.py`)

`lod_restamp.py` above can re-derive any legacy ladder. `luxar.io.lod_screening`
answers the question that decides whether doing so **buys anything on a given
store**: per `kind=lod` group, at the framing the viewer actually opens with,
does the re-derived `screen-area` ladder pick a COARSER level than the stored
one? It is read-only and it is a report — no verdict it produces is a failure.
It backs `scripts/check_demo_ladders.py --screen` / `--screen-only`.

```bash
hatch run check-demo-ladders --screen-only datasets/examples/*.luxar.zarr
hatch run check-demo-ladders --screen-only --screen-verdict win --screen-render-fov 63  datasets/demos/*.luxar.zarr
```

```python
from luxar.io.lod_screening import screen_stores, print_screen_report

report = screen_stores(["scene.luxar.zarr"])
print_screen_report(report, verdicts=["win", "fragile"])
```

Each group lands in exactly one bucket:

| verdict | meaning |
|---|---|
| `win` | strictly coarser under the re-derived ladder at EVERY tested aspect |
| `no-op` | no tested aspect picks a coarser level |
| `fragile` | coarser at some aspects and not others; the answer depends on the window |
| `off-screen` | the world box misses the frustum, so no metric is ever taken |
| `already-current` | already `screen-area` — all `restamp-lod` looks at before skipping |
| `skipped` | undecidable; `GroupScreening.reason` says why |

A `[FINER]` row means the re-derived ladder would open on a more expensive
level than today's ladder. It remains in `no-op` because that bucket means only
that no tested aspect gets coarser; it does not promise an unchanged opening
cost.

Five things the screen is careful about, each of which a cruder measurement gets
wrong:

- **The cut is the group's own anchor.** A whole-object ladder's finest rung is
  `WHOLE_OBJECT_FINEST_ANCHOR` = 0.5, not 1.0, so a group opening anywhere in
  `[0.5, 1.0)` still shows full detail and re-deriving it changes nothing. The
  anchor (and the clause of the two-clause rule that decided it) is reported per
  group, resolved through `lod_restamp`'s own `_is_partition_bound`.
- **Every refusal is `restamp-lod`'s refusal.** The screen predicts what that
  command would do, so it must decline exactly the groups the command declines
  or it reports a rewrite that never happens. `already-current` is decided on
  the SELECTOR ALONE (`_plan_lod` returns before reading a threshold — a
  `screen-area` group with an odd ladder is still left alone, and the screen
  prints the stored-vs-derived diff as evidence without calling it a win). For
  a legacy group, a descending stored ladder or an orphan `coverage_fraction`
  child is `skipped` through `lod_restamp`'s own predicates; an already-current
  group keeps that same hygiene finding as detail without changing buckets.
- **The two selectors are in different units.** `today` is scored under the
  STAMPED selector — the legacy `coverage` metric is an UNCLIPPED pixel diagonal
  over `FILL_FACTOR × min(W, H)`, range ~`[0, 4]` — while the re-derived side is
  always the clipped area fraction, range `[0, 1]`. Feeding one into the other's
  ladder inverts the answer; there is a test that does exactly that.
- **No inherited baseline.** What the store does TODAY is measured, never
  assumed. Against an assumed baseline a genuine win and a no-op look identical.
- **Aspect ratio is an input.** Every group is measured at 1:1, 16:9 and 21:9.
  The legacy metric happens to be aspect-invariant under the fitted framing; the
  area metric is not, so a lone object filling the shot reads 0.56 at 1:1 and
  0.24 at 21:9 — two different levels. Those groups are `fragile`, not wins.

The camera pose is the viewer's own: the scene root's `position_bounds` projected
onto the displayed axes and fitted face-on down `-Z` by a transcription of
`bounds-math.ts::calculateCameraDistance`. The fitted DISTANCE always uses the
default FOV (47) because the cinematic preset overrides the FOV only after the
fit, so `fit_fov` and `render_fov` are separate parameters. A scene that authors
`viewer_config.camera.position` / `target` / `target_node` / `up` opens somewhere
else and is skipped by name rather than screened against a framing nobody sees.
So is a scene whose FOV comes from `viewer_config.cinematic_mode` or
`camera.fov_preset`: those name an entry in the viewer's own TypeScript preset
table, and a second unverified copy of it here would be worse than asking for
`--screen-render-fov 63`. That flag supplies a fallback only where the store
does not author numeric `camera.fov`; an authored FOV always wins. A scene
displaying fewer than two dimensions is skipped too —
`lod-group-registry.ts::evaluatePerFrame` bails there before it evaluates any
group.
Dynamic near/far clipping is deliberately NOT modelled — the near-plane hazard
that matters is the homogeneous-`w` straddle inside `project_box_ndc_rect`,
which never reads `camera.near`.

Every metric primitive is exported and unit-tested against hand-derived values
(`project_box_ndc_rect`, `project_box_area_fraction`, `project_box_diagonal_px`,
`legacy_coverage_metric`, `pick_child_with_hysteresis`, `frustum_planes`,
`calculate_camera_distance`, `transform_box`,
`project_bounds_to_display_dims`), so a divergence from the TypeScript twin each
one cites shows up as a failing test rather than as a plausible wrong number.

### Input Volume Loading

`luxar.io.volume` and `luxar.io.ome_zarr` load arbitrary input volumes (the
sources fed to gsplat fitting/calibration), independent of the compiled
`.luxar.zarr` scene format above:

- `volume.load_volume(path, channel=, timepoint=, array_key=, axes=, info=)` —
  reads `.npy` / `.npz` / `.zarr` / `.zarr.zip` / `.tiff` / imageio-supported
  files to a float32 volume, with OME-Zarr-aware positional slicing and an
  explicit `--axes` override. Missing optional readers raise `ImportError` (the
  CLI turns it into a clean exit). Pass an `info` dict to get back
  `source_dtype`, the element type the array was **stored** in — the return is
  always float32, so this is the last point at which it is knowable, and it is
  the honest denominator of any size/compression figure quoted about the result.
- `ome_zarr.discover_ome_zarr_shape(path, ...)` → `OMEZarrInfo` — discovers the
  T/C/Z/Y/X layout, voxel size, unit, and resolution levels from NGFF
  `multiscales` (with custom-`axes` and shape-heuristic fallbacks). Both
  OME-Zarr layouts parse: 0.4's top-level block and 0.5's block nested under an
  `ome` key — resolved by the exported `ome_zarr.resolve_ngff_attrs(attrs)`,
  which every reader of NGFF attributes should go through (the layout can not be
  inferred from the store's zarr format version, and it decides which array gets
  SELECTED as well as how it is described). The nested block wins only when it
  carries a NON-EMPTY `multiscales` (or when the top level declares none at all),
  so neither an `ome` block holding just `omero` rendering metadata nor an empty
  `ome.multiscales` displaces a top-level 0.4 pyramid. A `multiscales` block
  whose `axes` count disagrees with the SELECTED array's ndim is not metadata
  about that array (a 5D image beside its 3D `labels/…`) and is skipped rather
  than parsed.
- **Whose block describes the selected array.** The root's is what is used,
  **unless** the group that OWNS the array declares that very array as one of its
  own `multiscales` levels — then that block wins: a bioformats2raw store puts
  the block on the image group and leaves only `bioformats2raw.layout` at the
  root, so reading the root alone would silently fall through to the shape
  heuristic. The override is gated on **evidence**, not on the block merely
  existing: one of the owner's `datasets[*].path` entries has to resolve to the
  selected array. A same-length but permuted axis list is not evidence, and
  adopting it would rewrite a T/C decomposition the root already had right — a
  silently wrong `batch-fit` fan-out rather than an error. Evidence is the *only*
  gate — the parser degrades honestly on a malformed `datasets`, so a second
  shape check could only discard a block that does describe the selected array.
  An owner block that fails the evidence gate is reported as such in the give-up
  notice ("declares a `multiscales` block that does not name it"), because the
  store plainly declared something. An owner block that wins the evidence gate
  but is then unusable (wrong axis count, no `axes` list) does not hide the
  root's; the root's is retried. Dataset paths are matched exactly relative to
  whichever group won — exactly after normalising surrounding slashes and one
  leading `./`, since the reader accepts `"0"`, `"/0"` and `"./0"` as spellings of
  the same child and NGFF writers do emit the explicitly relative form — so a
  same-named root pyramid level cannot be mistaken for a nested array. Resolving
  a declared level to an actual array normalises the same way, so a pyramid
  spelled `["./0", "./1"]` selects and describes the same arrays as one spelled
  `["0", "1"]`. Although NGFF requires strings, real numeric scalar paths are
  coerced for compatibility while other non-string values name nothing; lookup
  and metadata matching apply that same rule. The custom (non-NGFF) bare `axes`
  attribute goes the other way round, **root first**, because such a list names
  nothing and so no evidence about it is obtainable; an owner's `axes` is
  consulted only when the root has no usable list of its own.
- Voxel size composes any multiscales-level `coordinateTransformations` on top;
  where that match or that composition cannot be made honestly (an `array_key`
  matching no entry of a multi-level pyramid, or two scale vectors of different
  lengths) it reports no spacing rather than a plausible wrong one. Malformed
  metadata degrades to a fallback throughout, never a traceback — a `multiscales`
  whose `axes` is not a list (`{"axes": null}`) or whose `datasets` is not a list
  of mappings, an axis record with no `name` or a `null` `type`, a `scale`
  carrying a `null` or a non-numeric string. Falling through to the shape
  heuristic on a ≥4D store guesses the T/C roles and recovers no voxel size, so it
  says so on the console — stating whether nothing was declared or something was
  declared but unusable — and points at `axes_override` / `--axes`.
- The returned `OMEZarrInfo` publishes the decomposition it used, not just its
  results: `time_axis`, `channel_indices` and `spatial_indices` are indices into
  `shape`. **Read those rather than re-classifying `info.axes`** — NGFF is
  classified by the axis `type` field, so a name-driven rule disagrees in both
  directions (a channel axis named `stain`; an axis typed `view`, which discovery
  treats as spatial), and two vocabularies deciding the same question is how a
  consumer silently plans against a layout discovery never reported.
- `ome_zarr.ngff_scale_transform(transforms)` → the `scale` vector of a NGFF
  `coordinateTransformations` list, or `None`. The list is SEARCHED for the
  `type == "scale"` entry rather than indexed at `[0]`, which breaks on any store
  whose first transform is a `translation`.

Which array gets read out of a group is ONE rule, `volume._select_zarr_array`,
shared by `load_volume`, `open_volume_lazy` and `discover_ome_zarr_shape` — they
have to agree, because a re-fit re-opens a store whose shape another command
already read, and a different choice would silently target a downsampled level.
In order: an explicit `array_key` (which may be nested, `h2afva/fused`, and may
name a *group* — then the rule descends into it; blank counts as absent); else
the OME-NGFF resolution level `"0"`; else the largest array found recursively,
with a size tie broken on the lowest key path so two processes reading the same
store cannot disagree. When `"0"` (or the key) is a **group** rather than an
array — the bioformats2raw layout, whose pyramid levels are `0/0`, `0/1`, … —
that is resolved, not refused, and scoped to that image group, so a store holding
several series (`0`, `1`, …) plus an `OME` metadata group still resolves to full
resolution of the *first* image. Within the image group the candidates are the
levels its own `multiscales` block declares, else its direct array children, else
(skipping `labels/` at any depth) whatever is nested below: NGFF puts an image's
segmentation masks at `<image>/labels/<name>/<level>`, and inside an image group
a mask as big as level 0 must never be selected as the image. That guarantee is
**terminal**: an image group that resolves to no array of its own is a
`ValueError`, never a fall-through to the whole-store sweep — an image group
holding only `0/labels/seg/0` would otherwise select the *mask*, and an empty one
a *different series*, both silently — and on the very same store an explicit
`--array-key 0` used to crash outright (`AttributeError: 'Group' object has no
attribute 'shape'`), so one store answered three different ways depending on how
(or whether) the key was spelled. All three now raise the same clear `ValueError`.
The whole-store fallback is therefore reached only when there is no `"0"` key at
all (the `h2afva/fused` layout), and it sweeps every group recursively, `labels/`
included, as it always has. A store with no array
anywhere is still a clear `ValueError`, naming what led there — the key, or the
OME-NGFF `"0"` convention when no key was passed, since blaming a key the caller
never typed sends them hunting their own command line — what the store does hold,
and the array keys that *would* work. Whichever spelling reaches an array — no
key at all (the whole-store sweep included, which reports the level's immediate
parent), the image group, an intermediate group, or the level itself — its
**owner** is resolved by one rule (the nearest ancestor whose `multiscales`
declares it), so all of them describe the store the same way. Leaving any single
route out of that is not a rule: the one left out gives the same array a
different owner, hence different axes and a different voxel size, decided by
nothing but the spelling.

These are domain-layer helpers (no CLI dependency); the gsplat CLI re-exports
them. Dimension inference from a splat bounding box lives in
`luxar.core.dimension_inference.build_dimensions_from_data`.

## Encoding Integration

The I/O layer uses `luxar.encoding` for semantic type-aware array encoding:

```python
# Automatically applied by compiler:
# - positions → COORDINATE (float16 in MEMORY mode)
# - colors → COLOR (rgb_uint8 for SDR, geolog_perchannel_u16 for HDR)
# - radii → POSITIVE_SCALAR (log_scalar_uint8 for wide ranges)
# - sharpness → BOUNDED_SCALAR (bounded_scalar_uint8, bounds [0, 1])
```

**Encoding Modes**:
- `AUTO`: Analyzes data and selects encoding (may quantize)
- `PRECISION`: Full float32, lossless (broadcasting still allowed)
- `MEMORY`: Aggressive quantization for minimum storage

See `luxar.encoding` package for complete encoding system documentation.

## Spatial Ordering

### Morton vs Hilbert

**Morton (Z-order)**:
- Fast bit-interleaving algorithm
- Simple implementation
- Good compression

**Hilbert**:
- Better locality preservation
- ~10% better compression
- **Default** for Points, Lines, and GSplats (`ordering_method="hilbert"`)

Both encoders use a Numba JIT kernel when available (with a vectorized NumPy
fallback for Morton and a pure-Python `hilbertcurve` fallback for Hilbert), so
neither method needs an extra dependency to run.

### Metadata Structure

Morton/Hilbert ordering adds metadata to point/splat groups:

```json
{
  "ordering": "hilbert",
  "slice_dims": [3, 4],           // Discrete dimensions (time, channel)
  "ordering_dims": [0, 1, 2],     // Spatial dimensions (X, Y, Z)
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [100.0, 100.0, 100.0],
  "ordering_bits_per_dim": 21,
  "chunk_size": 2048
}
```

Plus a `chunk_bounds` array: shape `(num_chunks, n_dims, 2)` with min/max bounds per chunk.

## Architecture

### Progressive Writing Flow

```
User Code → Scene API → Compiler → Spatial Ordering → Encoding → Zarr Store
                                        ↓                  ↓
                                   sort_points_compound  ArrayEncoder
                                   compute_chunk_bounds  (semantic types)
```

1. User creates data arrays
2. Scene API validates data
3. Compiler applies spatial ordering (if enabled)
4. Arrays encoded using semantic types
5. Data written immediately to Zarr
6. Only metadata kept in memory

### Reading Flow

```
Zarr Store → LuxarScene → ArrayDecoder → Decoded Arrays → User Code
                  ↓
            Node introspection
            Dimension parsing
            Transform conversion
```

1. Open zarr store read-only
2. Parse scene metadata (version, dimensions)
3. Build node index from zarr groups
4. On `get_*()` call: load arrays with automatic decoding
5. Return decoded data with proper numpy shapes/dtypes

### nD Transforms and Bounds Expansion

When nodes have `nd_transform` attributes (affine scale/offset or categorical permutations on non-displayed dimensions), the compiler stores them in each node's zarr group attrs. During `finalize()`, the compiler:

1. Walks the zarr tree to compose world nd_transforms (root to leaf)
2. Applies `apply_nd_transform_to_bounds()` to each leaf node's local position bounds
3. Computes the union of all world-space bounds
4. Stores the result as `position_bounds` in root attrs

This means scene-level `position_bounds` has:
- **Displayed dims**: local-space bounds (for camera auto-framing)
- **Non-displayed dims**: world-space bounds (for slider auto-ranging)

Per-node `position_bounds` remain in local space. See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for the full specification.

### Memory Management

- **Zero-copy writing**: Data goes directly to disk
- **Chunked storage**: Optimal chunk sizes (64KB target)
- **Spatial ordering**: Morton/Hilbert ordering with compound sorting for nD data

## Examples

### Basic Scene with Spatial Ordering

```python
from luxar.io import LuxarZarrCompiler
from luxar.core import Dimensions, Dimension

# Define nD dimensions
dims = Dimensions(
    [
        Dimension("X", unit="um", display=True),
        Dimension("Y", unit="um", display=True),
        Dimension("Z", unit="um", display=True),
        Dimension("Time", discrete=True, display=False),  # Discrete dimension
    ]
)

with LuxarZarrCompiler("scene.luxar.zarr", ordering_method="hilbert") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Data will be compound-sorted: Time first, then Hilbert(X,Y,Z)
    scene.add_points("cells", positions_4d, colors, radii)
```

### Memory-Optimized Scene

```python
from luxar.io import LuxarZarrCompiler
from luxar.encoding import EncodingMode

with LuxarZarrCompiler(
    "compressed.luxar.zarr",
    encoding_mode=EncodingMode.MEMORY,  # Aggressive quantization
    ordering_method="hilbert",
) as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    scene.add_points("cloud", positions, colors, radii)
    # Positions: float16
    # Colors: uint8 (if SDR)
    # Radii: log_scalar_uint8
    # With Hilbert ordering for better compression
```

### Round-Trip (Write and Read)

```python
from luxar.io import LuxarZarrCompiler, LuxarScene
import numpy as np

# Write data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)

with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    scene.add_points("cloud", positions, colors, radii=0.1)

# Read it back
scene = LuxarScene.load("scene.luxar.zarr")
points = scene.get_points("cloud")

# Verify data (accounting for encoding precision)
np.testing.assert_allclose(points["positions"], positions, atol=1e-5)
np.testing.assert_allclose(points["colors"], colors, atol=1e-5)
```

## Performance

Typical compression ratios (vs uncompressed float32):

| Configuration | Compression | Notes |
|---------------|-------------|-------|
| PRECISION + Morton | 2-3x | Lossless, ordering helps blosc |
| AUTO + Hilbert | 4-6x | Selective quantization |
| MEMORY + Hilbert | 8-12x | Aggressive quantization |

Compression gains from:
1. **Spatial ordering** (~2x from blosc on ordered data)
2. **Quantization** (2-4x from float32→float16/uint8)
3. **Broadcasting** (massive savings for uniform values)
4. **LUT encoding** (up to 75% for <256 unique values)
5. **Array deduplication** (via xxhash64)

## Dependencies

**Internal**:
- `luxar.core`: Scene graph nodes and dimensions
- `luxar.encoding`: Semantic type-based encoding
- `luxar.typing_utils`: Constants and type definitions
- `luxar.validation`: Data validation

**External**:
- `zarr>=3.2,<4`: Storage backend. Note the library version and the on-disk
  format are separate axes — Luxar writes zarr **format 3** by default
  (`LUXAR_ZARR_FORMAT=2` still produces format 2) and READS both, so existing
  format-2 stores keep working untouched. Both axes are pinned in
  `luxar._zarr_compat`, the only module that names a zarr format; go through its
  helpers (`open_group`, `create_array`, `consolidate`, ...) rather than calling
  `zarr.*` directly. Code that inspects a store on disk should use its
  bi-format readers (`read_array_meta`, `read_node_attrs`, `is_consolidated`,
  `read_consolidated_attrs`) instead of naming `.zarray` / `.zattrs` /
  `.zmetadata`, which exist in only one of the two formats. Editing a store in
  place likewise goes through `open_group`: re-opening an already-consolidated
  store with plain `zarr.open_group` and re-consolidating writes a nested
  consolidated index holding the pre-edit attributes, which later reads then
  serve in preference to the (correct) documents on disk.
- `numpy>=2.0`: Array operations
- `numcodecs`: Blosc compressor (`DEFAULT_COMP`)
- `arbol>=0.3.5`: Progress logging
- `xxhash>=3.0.0`: Array deduplication hashing
- `hilbertcurve>=2.0.5`: Pure-Python Hilbert fallback when Numba is unavailable
- `numba` (optional): JIT-accelerated Morton/Hilbert encoding kernels

## Related Documentation

- **Encoding system**: `../encoding/README.md` (semantic types, quantization)
- **Core structures**: `../core/README.md` (Scene graph, dimensions)
- **Ordering module**: `ordering.py` (Morton/Hilbert implementations)
