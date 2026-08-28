# Luxar CLI Reference

This is the **authoritative catalog of every public `luxar` command**. It lists
each command group and links out to the workflow guide that covers it, so option
lists live in exactly one place and cannot drift.

For the exhaustive, always-current list of options for any command, run the tool
itself:

```bash
luxar --help                 # top-level commands
luxar gsplat --help          # a command group
luxar gsplat fit --help      # a single command and all its options
```

`--help` is the source of truth for options; this page is the source of truth for
*which commands exist* and *where to read about the workflow they belong to*.

## Staying in sync

The list of commands on this page is checked against the live Typer application by
an automated test —
`packages/luxar/src/luxar/cli/tests/test_docs_command_coverage.py`. One check walks
the registered command tree, extracts every `luxar ...` invocation from the code
blocks on this page, and compares the two sets in **both directions**: a newly
added command must be documented here, and a documented command that was removed,
renamed, or hidden must be pruned — either way the test fails.
(Hidden/internal commands and groups are excluded.) A second check in the same file
walks every per-command section heading and verifies each option spelling its prose
names in inline code against the live command's declared options — a heading that names a
command GROUP is checked against the union of its subcommands' options instead,
since a group's prose legitimately names its subcommands' flags — so a stale or
invented flag fails too, not just a stale command path.

## Top-level commands

```bash
luxar info      # Dataset structure, dimensions, and compression statistics (--stats, --format json)
luxar optimise  # Re-chunk an existing store for streaming; values stay bit-identical
luxar restamp-lod  # Re-derive legacy LOD switch thresholds in place (attrs only)
luxar serve     # Serve a .luxar.zarr over HTTP, optionally with the viewer (--viewer, --open)
luxar viewer    # Serve the Luxar viewer, optionally with a dataset (--data)
luxar export    # Export a scene + viewer as a standalone offline folder (or --native bundle)
luxar profiles  # List the network-simulation profiles usable via --profile
```

- Serving and viewing → [Viewer guide](./VIEWER_GUIDE.md)
- Offline / native export (and the `make build-launchers` prerequisite) →
  [Build system guide](../developer/BUILD_SYSTEM_SPEC.md)
- Network profiles → [Network simulation spec](../developer/NETWORK_SIMULATION_SPEC.md)

## `luxar serve`

Serve a scene directory directly with `luxar serve scene.luxar.zarr --viewer`,
or serve a containing directory with `luxar serve /path/to/scenes --viewer`
and choose a scene from the viewer's dataset browser.

For a `.luxar.zarr.zip`, serve its containing directory rather than passing the
archive itself, which fails with `Data mount root must be a directory`. The
server advertises and honours the HTTP byte ranges the viewer uses to read
members without extracting the archive. Zipped scenes are read-only. The viewer
has no browser local-file or drag-and-drop opening path for any scene format.

`luxar export` also requires a directory store; passing an archive fails with
`Invalid zarr store: Path is not a directory`. Its exported preview server does
not add byte-range support for archives. See the
[viewer guide](./VIEWER_GUIDE.md#opening-a-zipped-scene) for the direct-URL form
and the full limitations.

## `luxar optimise`

Re-chunk a store that already exists so it streams well, in one
structure-preserving pass. No refit, no source volume, no GPU: only zarr chunk
shapes change, and array values stay bit-identical.

```bash
luxar optimise                    # Re-chunk SOURCE into OUTPUT at the 64 KB default
luxar optimise --dry-run          # Report the plan and write nothing (omit OUTPUT)
luxar optimise --target-kb 128    # Set the chunk budget directly
luxar optimise --profile hosting  # Preset budget: hosting / local / archive
luxar optimise --verify           # Re-read the output; compare arrays and payload files
luxar optimise --overwrite        # Replace an existing OUTPUT store
luxar optimise --generic          # Allow a plain (non-Luxar) zarr store
```

It takes a source store and, unless `--dry-run` is given, a destination store —
a compiled `.luxar.zarr` scene, a standalone `.gsplats.zarr` tree, or (with
`--generic`) any zarr store at all.

Most already-generated datasets are chunked far below the 64 KB target — the
bundled demo corpus averages 5.1 KB per file, with 97% of files under 16 KB —
and a cold load over object storage is dominated by round trips, not bytes.
Re-chunking one demo to 64 KB cut a 245 s / 9,390-request cold load to 51 s /
2,348 requests.

Pick the budget with exactly one of `--target-kb`, `--target-bytes` or
`--profile`. The profiles are **hosting** (256 KB — fewest round trips over
object storage), **local** (64 KB — the authoring default) and **archive**
(1 MB — not for streaming; minimises file count). `--dry-run` reports the plan
and writes nothing, so the output argument must be omitted. `--verify` re-reads
the written store and compares every array — and every plain payload file the
pass copied, such as an overlay image — byte for byte, reporting how many of
each it checked. `--generic` allows a
plain zarr store that is not a Luxar scene or a `.gsplats.zarr` tree.

One boundary is worth stating for `--generic`, because it is a silent no-op
rather than an error: this pass merges **rows**, i.e. it only ever grows the
chunk along axis 0. An array chunked on its trailing axes instead — an OME-Zarr
`(1, 1, Z, Y, X)` level chunked `(1, 1, 8, 32, 32)`, say — is therefore left
alone and reported as `rows already in one chunk`, even though it may hold
hundreds of small chunk files. Luxar's own arrays are all row-chunked, so this
only affects foreign stores; use a dedicated rechunker (`rechunker`,
`ome-zarr-py`) for those.

The larger profiles trade **partial-query** bytes for **full-load** requests, so
"object storage → `hosting`" is not unconditional. A Points or GSplats node is
not loaded whole: the viewer turns the visible spatial-index chunks into element
ranges of `chunk_size` atoms, and one atom-hit costs one zarr chunk whatever its
size. Measured on a real store (atom 2340, uint16 `(N, 3)`): `local` fetches
4 atoms / 54.8 KB per partial hit, `hosting` 18 atoms / 246.8 KB (4.5x) and
`archive` 74 atoms / 1014.6 KB (18x). Pick `hosting`/`archive` when the access
pattern is "load the whole node" (a gallery still, a small scene, an archive
upload); stay on `local` when the viewer will be slicing into a large one.

dtype, codecs, filters, `fill_value`, memory order, the chunk key layout, the
on-disk zarr format version and every attribute except the two the pass must
move — the root's `content_hash` and the `chunk_layout` summary written beside
it — are all preserved; a sharded array keeps its shard grid; and the
spatial-index grid is never moved, since each new chunk is a whole multiple of
its node's `chunk_size` atom. Nothing is chunked smaller than it already is, so
the chunk grid is a **fixed point**: a second run re-chunks nothing. It still
rewrites the store, and it still moves the hash — the `chunk_layout` attr now
records the counts that changed (22 → 13 becomes 13 → 13). From the third run
on, both the grid and the hash are fixed: the same layout over the same values
hashes the same.

The output gets a fresh `content_hash` and a `chunk_layout` root attribute,
because chunk keys now cover different rows and a warm viewer cache validating
on an unchanged hash would serve stale chunks. For the same reason, replacing an
existing output requires `--overwrite` and rewriting in place is refused —
republishing under a new URL prefix is the safe move. `--overwrite` replaces an
existing zarr store or an empty directory and nothing else; a destination that
contains the source (or sits inside it) is rejected outright, and so is one that
is a symlink, since moving the new store into place would replace the link
rather than what it points at (pass the target path instead). The whole output
is staged beside the destination and moved into place last — and an existing
store is renamed aside and deleted only once the replacement is in place — so an
interrupted or failed run leaves no partial store and never costs you both
copies. Run `luxar info` with its
detailed-statistics flag to see a store's chunk layout before and after.

## `luxar restamp-lod`

Re-derive the LOD switch thresholds of a store that already exists, in place.
An **attributes-only** pass: the ladder rewrite moves no chunk data and opens no
array.

```bash
luxar restamp-lod                          # Re-derive every legacy ladder in STORE
luxar restamp-lod --dry-run                # Report the old→new ladders; write nothing
luxar restamp-lod --group tiled/part_0     # Restrict to one ladder (repeatable)
luxar restamp-lod --group /                # The store ROOT's own ladder
```

Every `kind=lod` group carries per-child `coverage_fraction` thresholds plus a
group-level `selector` naming the units they are in. Ladders written before the
screen-area metric existed sit on the legacy `coverage` diagonal one (or carry
no `selector` at all, which means the same). This command re-derives those
thresholds by screen-occupancy halving — the whole-object anchor for a plain
ladder, the fills-screen anchor for a **tile-bound** one — and stamps the group
`screen-area`. A group already on `screen-area` is skipped, so a second run
changes nothing at all, down to the `content_hash`.

Tile-bound is the tree writers' own two-clause rule, so a restamped store
matches a freshly written one: a ladder is tile-anchored when a REAL multi-part
`kind=partition` encloses it (the `tiles` and `adaptive` per-tile ladders), **or**
when one of its own ladder children is itself a `kind=partition` — the `overview`
recipe's `[coarse cap, fine partition]` pair, which is pinned at the fills-screen
anchor deliberately so the opening framing shows the coarse cap instead of
loading the whole dataset. A one-part partition is not a tiling (its single part
IS the whole object) and does not bind.

**It is an explicit opt-in, and it may override a deliberate choice.** An
authored `coverage_fractions=[...]` list and a legacy derived ladder are
indistinguishable on disk, which is exactly why nothing does this automatically
and why the compiler's one-part-partition check only ever warns. The per-group
old→new ladder and the anchor used are printed for that reason — run `--dry-run`
first, and use `--group` (repeatable; an unmatched path is an error, not a
silent no-op) to restrict the pass to the ladders you meant. Group paths are
spelled as the store spells them (`tiled/part_0`), and the store root is `/` —
the only way to name the ladder of a `.gsplats.zarr` whose root IS the
`kind=lod` group.

Sibling of `luxar optimise`, not a flag on it: that pass preserves every
attribute and refuses same-path work, this one changes only attributes and works
in place. Give it an uncompressed `.luxar.zarr` or `.gsplats.zarr`
**directory** — a `.zarr.zip` is refused, since an archive is read through a
temp directory and there is nothing to write back to.

When anything changes, the store's `content_hash` is restamped and the metadata
re-consolidated, in that order: an attrs-only edit must still invalidate a warm
viewer cache. **That restamp is the one step that is not instant.** A compiled
scene's `content_hash` is a digest of array VALUES, so restamping it reads every
array in the store exactly once — linear in the store's total size, so expect
minutes on a very large scene even though only two attributes changed. A
standalone `.gsplats.zarr` takes the other branch, a metadata-only stamp, and
stays instant. A `--dry-run`, and a run with nothing to change, read nothing at
all. The result is then read back — from both the per-node documents and the
consolidated index the viewer fetches — and verified. An index is **rebuilt,
never introduced**: a store handed over without consolidated metadata leaves
without it, because `is_consolidated` is how `batch-fit` tells a finished tile
from an interrupted one and creating one here would report an unfinished tile as
complete.

If a write fails part-way, every attribute already rewritten is restored —
including each `content_hash`, put back exactly as the store carried it rather
than recomputed, so a legacy or hand-edited digest is not silently rewritten by
a run that failed — and the error is reported. The metadata is re-consolidated
only when the failed run had rewritten the store ROOT, the write that
invalidates the index; a failure that never touched it leaves the valid index
alone rather than risking a second one. The store is left as it was found rather
than carrying a half-migrated ladder whose thresholds and `selector` disagree
about their units — a disagreement nothing downstream can detect.

The command exits 1 if any ladder was left alone for a reason worth acting on: a
`selector` outside the vocabulary (migrate the store with `luxar gsplat
migrate-format` first), a finest level whose element count the store does not
record, a stored ladder that descends in the resolved coarsest→finest child
order (its thresholds and its child order disagree, so re-deriving would invert
it), a child group that carries a `coverage_fraction` but cannot be classified as
a ladder level (re-deriving over the rest would leave a partial, non-monotonic
ladder), or a re-verification residual. Ladders that *were* restamped are still
written in that case — nothing is silently ignored, and nothing is silently
half-done.

It exits 1 for one more case, where the rewrite fully succeeded: a store that
carries neither a scene `type` nor a `.gsplats.zarr` `content_hash` has no digest
to restamp, so nothing tells a warm viewer cache that the ladder moved. At zarr
format 2 — the legacy corpus this command exists for — even the viewer's
`zattrs-hash` fallback digests the root `.zattrs` bytes, which an edit to a child
ladder does not touch, so a warm cache would serve the old ladder indefinitely.
The run says so with a `⚠️` and exits non-zero; republish under a new URL prefix.

## `luxar demo`

Browse, run, and manage the bundled demos.

Most of these are also published as live, interactive scenes at [demos.luxarviewer.dev](https://demos.luxarviewer.dev), so you can see what a demo looks like before spending the time to build it — several need a GPU or a large download.

```bash
luxar demo list          # List all demos (key, needs, status)
luxar demo info          # Full details for one demo
luxar demo run           # Generate a demo and open it in the viewer (forwards -- args)
luxar demo run-all       # Generate every demo (batch)
luxar demo stop          # Stop running demos and free their ports (--dry-run to list)
luxar demo deps          # Optional dependencies (--install, --extra, --only MODULE)
luxar demo cache list    # Inventory the ~/.cache/luxar demo caches
luxar demo cache clear   # Clear demo caches (--dry-run to preview)
```

Report-only `demo deps` exits 1 for any missing or outdated row. Generic
`--install` manages Luxar extras; use `--only MODULE --install` to install one
exact constrained requirement, including a dependency outside every extra.

`demo stop` clears demos left running in forgotten terminals — the usual cause
of a "port busy" warning and a browser tab that still shows an older
scene. It finds runs via the registry `demo run` maintains (plus a
process-table sweep for strays), lists them, asks for confirmation (`-y` to
skip), and tears each one down with the same SIGINT → SIGTERM → SIGKILL
escalation Ctrl-C uses. `luxar demo stop <key>` stops just one demo;
`--dry-run` only lists. On platforms without POSIX process groups it lists the
recorded runs and prints the command to stop each by hand rather than signalling
a pid it cannot first verify still belongs to the demo; once that process is
gone the record drops itself from the next listing.

## `luxar gsplat`

The Gaussian-splat toolbox. The canonical end-to-end pipeline is
`cal → fit → lod` (then `convert` to embed the result in a web-viewer scene); see the format and LOD notes in
[Formats & migration](./FORMAT_AND_MIGRATION.md) and the full data-format spec in
[GSplats zarr format](../../specs/GSPLATS_ZARR_FORMAT.md).

### Fitting & calibration

```bash
luxar gsplat cal        # Calibrate the splat count K via blind-spot cross-validation (K*)
luxar gsplat fit        # Fit Gaussian splats to a volume (presets, tiling, per-part LOD recipes)
luxar gsplat render     # Render a fitted dataset back to a volume for comparison
luxar gsplat denoise    # Denoise a volume with Non-Local Means (auto-calibrates h)
luxar gsplat benchmark  # Benchmark GPU performance to pick optimal tile sizes
```

### Level-of-detail

```bash
luxar gsplat lod        # Build a representation topology (--recipe flat|stream|levels|tiles|overview|adaptive)
luxar gsplat additive   # Give every leaf of an existing tree an additive (streaming) ladder
luxar gsplat flatten    # Collapse any tree (leaf/lod/partition/nested) into one flat leaf
```

### Scene & interchange

```bash
luxar gsplat convert         # Convert a .gsplats.zarr into a Luxar scene for the web viewer
luxar gsplat import          # Import a classical splat file (INRIA PLY / .splat / .spz) → .gsplats.zarr
luxar gsplat export          # Export a .gsplats.zarr → classical INRIA PLY
luxar gsplat migrate-format  # Upgrade a legacy .gsplats.zarr layout to the current v3.3 format
luxar gsplat reencode        # Re-quantize a current-format dataset's Cholesky encoding (structure-preserving)
```

Format versioning and the `migrate-format` vs `reencode` distinction are explained
in [Formats & migration](./FORMAT_AND_MIGRATION.md).

### Inspection

```bash
luxar gsplat info              # Dataset statistics (splat count, dimensions, bounds, LOD structure)
luxar gsplat doctor           # Diagnose standalone gsplats or scene partitions (--fix repairs in place)
luxar gsplat napari           # Open a dataset in napari for visual inspection
luxar gsplat view             # Open a .gsplats.zarr directly in the web viewer
luxar gsplat compare          # Compare reconstruction quality vs a reference volume (PSNR/SSIM/MSE)
luxar gsplat annotate-quality # Retrofit Q·e quality stamps onto an existing dataset, in place
```

`doctor` is for the problems you cannot see: a dataset written by an older Luxar
loads and renders fine while missing something a later version learned to record,
or carrying metadata that went stale under an edit. Given a standalone gsplat path
it prints the `info` report (suppress with `--no-info`); for a scene it notes that
the gsplat report does not apply. It then prints a diagnosis and exits non-zero
while a problem is still standing — so it can gate a pipeline. Pass `--fix` to
repair an uncompressed `.gsplats.zarr` or `.luxar.zarr` directory in place (unpack
a `.zip` or `.tar.gz` first), `--full-provenance` to include the complete nested
fitting record in the info report, or `--json` to write the findings out for a
machine.

It currently diagnoses a `kind=partition` whose split planes (`bsp_tree`) are
missing, or are present but disagree with where the parts actually sit. Without
them the viewer orders parts by centroid, which is not a valid painter's order and
pops at the seams under `normal`/`volumetric` blending; where the parts are
disjoint the planes are recovered exactly from the part boxes. For overlapping
uniform-tiled parts and centroid-split lines/mesh partitions, doctor uses the
largest measured interpenetration on each axis as an overlap-tolerance floor,
but still catches planes outside those bands. Disjoint points and splats must
still separate exactly, even when stale cuts remain between the child centers.
It can recover the band-bounded cuts, and reports a
coordinate-frame scale only when repeated planes support each changed axis, or
when a single plane agrees with a factor proven on another axis. What cannot be
repaired is reported with a remedy rather than guessed at.

### Editing & selection

```bash
luxar gsplat transform  # Apply spatial / intensity transforms (scale, rotate, translate, center)
luxar gsplat merge      # Merge datasets (concatenation, new dimension, or channel colors)
luxar gsplat cull       # Remove low-contribution splats while preserving visual quality
luxar gsplat decimate   # Reduce to a TARGET SPLAT COUNT (one flat leaf): merge or prefix
luxar gsplat filter     # Filter splats by multiple criteria (AND logic; percentile thresholds)
luxar gsplat slice      # Slice splats by coordinate ranges (numpy-style syntax)
luxar gsplat partition  # Partition into a single kind=partition file via spatial BSP
```

## `luxar gsplat batch-fit`

Fit a whole nD dataset at scale — locally across GPUs (`run`) or on a Slurm cluster
(`submit`). Both plan the decomposition once and stream-merge to a single
`kind=partition`.

```bash
luxar gsplat batch-fit run       # Fit a whole timelapse locally across multiple GPUs, then merge
luxar gsplat batch-fit submit    # Plan and submit Slurm array jobs for a large dataset (--dry-run to plan only)
luxar gsplat batch-fit status    # Check the status of a batch fitting run
luxar gsplat batch-fit merge     # Merge completed tiles into a single dataset (per-part LOD via --recipe)
luxar gsplat batch-fit validate  # Validate tile integrity (--fix deletes corrupt/stale tiles)
luxar gsplat batch-fit cancel    # Cancel all Slurm jobs for a batch run
```

## `luxar mesh`

Bring classical triangle-surface files into Luxar, and coarsen them. Both are NumPy +
stdlib only, so they work on a bare `pip install luxar` with no extras.

```bash
luxar mesh import            # Import one mesh file (PLY / OBJ / STL / VTP / glTF / GLB), or a T-indexed directory
luxar mesh lod               # Build a LOD ladder for a mesh scene (levels, or a reveal)
```

Vertices are welded and polygons fan-triangulated on the way in, because STL is always
a triangle soup and glTF sometimes is; an unwelded surface defeats per-vertex normals
and makes picking report a different vertex per triangle for the same corner. Welding
merges two vertices only when their position *and* their normals and colours agree, so
a hard edge — which every modelling package authors as coincident positions with
different normals — survives the import instead of being flattened. Pass `--no-weld` to
keep the reader-produced vertex list. The default welded path also removes vertices no
surviving triangle references, so non-surface points cannot inflate the scene bounds or
picking ordinal range.

`.vtp` is VTK XML PolyData — what ParaView, VTK and PyVista write for a surface,
and the usual output of a marching-cubes isosurface. Every encoding the format allows is
read: `ascii`, inline base64 (`format="binary"`), and an appended section in either `raw`
or `base64` form, each optionally `vtkZLibDataCompressor`-compressed, with a `UInt32` or
`UInt64` block header and either byte order. `Polys` are fan-triangulated and `Strips`
triangulated with the alternate winding a strip requires; `Verts` and `Lines` carry no
surface and are dropped. `PointData` normals and colours come across. An LZ4- or
LZMA-compressed file is refused by name — those codecs are outside the standard library
and this importer takes no new dependency — as is a `.vtu` (UnstructuredGrid), which is a
volume mesh: run ParaView's *Extract Surface* on it first.

Draco- and meshopt-compressed glTF is refused by name rather than decoded — run the
file through `gltf-transform` first.

A directory import defaults to `--pattern '*.vtp'`. Each filename must contain an
uppercase `T<number>` token; if every filename also contains `Ch<number>`, channel is
appended as a second hidden discrete dimension. Numeric values are preserved, so
missing timepoints remain gaps instead of renumbering later files. For a hidden dimension
with multiple distinct integer values, the step is the GCD of their sorted coordinate
differences; because navigation is anchored at the range minimum, aligned strided exports
advance directly between populated slices. A single-valued or non-integer dimension keeps
a step of 1. Mixed channel naming and duplicate time/channel coordinates are refused rather
than guessed. Pass the
directory and output scene as the two positional arguments, e.g.
`luxar mesh import 000_deconv.ome.zarr/meshes/cells cells.luxar.zarr`. Use `--pattern`
for another mesh format.

For other naming schemes, pass `--index-regex` with a required named `t` capture and an
optional named `c` capture. The regex searches each filename stem (without the final
extension) and must match exactly once. Captures must be integers exactly representable
as float32 coordinates. For example:

```bash
luxar mesh import --pattern '*.ply' frames out.luxar.zarr
luxar mesh import --index-regex 'frame_(?P<t>\d+)' frames out.luxar.zarr
luxar mesh import --index-regex 't=(?P<t>\d+)-c=(?P<c>\d+)' surfaces out.luxar.zarr
```

The resulting mesh node has no spatial index: the viewer downloads the entire stacked
directory even when it draws only one timepoint. Use this path for stacks that fit
comfortably in memory, not as a streaming representation for very large timelapses.

### `luxar mesh lod`

Builds one of **two** ladders, selected by `--recipe`:

- **`--recipe levels`** (the default) writes a `kind=lod` group whose coarse children are
  progressively **decimated** copies of the surface and whose finest child is the
  original. The viewer shows exactly one at a time, chosen by how much of the screen the
  object covers.
- **`--recipe reveal`** writes an **additive** ladder *inside* the leaf — `additive_<i>`
  subgroups holding disjoint groups of faces that the viewer concatenates as they arrive,
  so a partial load is a partial surface that grows rather than a coarse one.

They are separate recipes rather than composable flags because a mesh has no coarse
prefix: a prefix of an arbitrary index buffer is a surface with holes, not a simpler
surface. `add_mesh` refuses the two ladders together, so one `--recipe` selects, and a
knob aimed at the other recipe is refused by name rather than silently dropped.

Takes an input scene and an output scene, plus `--node` and `--overwrite`, and then the
knobs of the chosen recipe — `-L/--levels` (default 3), `-K/--compression-factor`
(default 4 — level *i* targets `V / K**i` vertices) and `--subst-method` for `levels`;
`-m/--add-method`, `--n-lods`, `--counts`, `--reveal-centre` and `--spatial-dims` for
`reveal`. The output path is normalized to the
canonical `<stem>.luxar.zarr`, so an output argument of `out` writes `out.luxar.zarr`; that
normalized path is what `--overwrite` replaces and what the same-path guard
compares against.

Unlike `luxar gsplat lod` this reads and writes a **scene**, not a standalone store —
there is no standalone mesh format, so the only sink for a mesh is a `.luxar.zarr`. The
node is optional when the scene holds exactly one mesh (what `luxar mesh import`
produces); with several, naming one is required rather than guessed at.

`--subst-method` shares its **name** with `luxar gsplat lod` — on both commands it selects
the substitutive, level-replacing reduction — but **not its values**: this one takes
`auto`, `qem`, or `cluster`, because a mesh is decimated where a gsplat level reduces a
Gaussian mixture, which a surface is not. QEM is Garland-Heckbert edge collapse with a
link-condition veto, so it preserves manifold topology; clustering is the vectorized
large-mesh tier. `auto` uses QEM through 10,000 source vertices and clustering above that
measured worst-case open-surface envelope. A QEM ladder builds one collapse sequence and
snapshots every level from it. On an open near-planar surface QEM's
orientation veto can stop well above the requested count and shorten the ladder; use
`cluster` when closely hitting the count matters more than topology preservation. QEM
also requires at least three coarsening dimensions;
`auto` falls back to clustering for a one- or two-dimensional coarsening, while explicit
`qem` is refused. The selected tier is printed with the reason.

The flag was called `--method` before August 2026, and `-m` was its short form. `--method`
is gone, and `-m` has since been **claimed** by `--add-method` — the additive ordering it
already names on `gsplat lod`, which is what it was reserved for. Either old spelling still
exits with a pointer naming `--subst-method` and carrying your value, rather than silently
doing something else.

#### The reveal knobs

`-m/--add-method` takes `radial` and nothing else. That is the restriction the whole
recipe rests on: only an ordering whose **every prefix is one connected patch** is
admitted, which is what makes a partial load a growing surface rather than lace.

`--n-lods` (default 4) asks for N equal-count levels; `--counts` gives the boundaries
explicitly as **cumulative** face counts (`--counts 500,2000,10000` → four levels of 500,
1500, 8000 and the remainder), or a streaming ladder as `stream:<c>`. Pass one or the
other, not both.

`--reveal-centre` and `--spatial-dims` are the same flags, with the same meanings and the
same shared parser, as on `luxar gsplat lod`. The centre defaults to the mesh's own
bounding-box centre, so a surface far from the origin still grows from its middle. The
**order** of `--spatial-dims` is significant: it pairs one-for-one with the centre's
coordinates, which is why it is not sorted the way a coarsen-dims barrier set is (there
the order carries nothing, so sorting is free). Use it to keep a
stacked time or channel column out of the distance, so shells do not expand through time.

So, given an input and an output scene: `--recipe reveal --n-lods 4` for an
equal-count ladder, `--recipe reveal --counts 500,2000,10000` for explicit boundaries,
and `--recipe reveal --reveal-centre 0,0 --spatial-dims 0,1` to grow the shells from a
chosen point on a chosen pair of axes.

(Written as prose rather than a fenced block on purpose: `test_docs_command_coverage`
reads every `luxar …` line in a fenced block as a command path, stopping at the first
flag, so an example carrying positional arguments would register `mesh lod
in.luxar.zarr out.luxar.zarr` as a command that does not exist.)

Levels that cannot reduce the surface are dropped, so a small mesh may come back with
fewer than `--levels`; one that cannot be reduced at all comes back as a plain leaf
rather than a one-child group.

The source node's placement and appearance come across with it, as does the scene's
`viewer_config` (a source that set none gets an explicit `tone_mapping='ACES'`, matching
`luxar mesh import`). The compositing attrs — `transform`, `nd_transform`, `opacity`,
`blending_mode` and the rest — land on the `kind=lod` wrapper group, which is the layer
the viewer inherits them from; `colormap` lands on every child, as a LUT when the palette
is not one of the builtin names. Per-vertex colours and scalars are averaged per cluster
on every coarse level, and every level stamps the source field's `scalar_data_range`, so
the colormap maps the same value to the same colour at every level rather than only at the
finest. Per-vertex **labels and image labels are not carried** — the reader does not
surface them, so the round trip cannot see them.

The output is a brand-new scene containing ONLY the picked mesh's ladder. **Every other
node in the source scene is left out** — other points/lines/gsplats/mesh nodes, other
groups, and any user-authored overlays (`add_text`/`add_html`/`add_image`) — since
there is nowhere else for them to go. Also left behind: any placement/compositing
(`transform`, `opacity`, `blending_mode`, …) an ancestor group genuinely CHANGES — only
the mesh's OWN attrs are forwarded. "Genuinely changes" is narrower than "is set at
all": a key sitting at its neutral value (`opacity`/`gamma`/`intensity`/`absorption` at
`1.0`, `offset` at `0.0`) composes as a no-op regardless of which layer sets it, an
identity `transform`/`nd_transform` moves nothing, a `layer`/`visible` at its own
default (`false`/`true`) is likewise a no-op, `blending_mode` is nearest-setter-wins so
it only matters for the nearest group that sets it and only when the picked mesh does
not set it itself, and `join` is skipped outright (it is lines-only — `add_mesh` refuses
it, so a mesh leaf can never carry it and an ancestor's `join` can never affect a mesh
ladder) — so re-laddering a level of an existing ladder (`--node surf/child_0`) or a
partition tile (`--node surf/part_0`) reports nothing here, even though the wrapper
groups those workflows nest under do carry a few of these keys at their neutral values
(this command's own ladders re-forward the picked mesh's stamped defaults onto the
wrapper it writes). After every check that can still abort the command and before
anything is written, it warns about each node that will not be carried across
(collapsed into one line per parent + kind when more than three share both, worded by
node kind rather than as "parts" — that term is reserved for actual `kind=partition`
children — so a large group of siblings does not flood the console), each ancestor
group that does lose something (naming exactly which keys), and any per-vertex
labels/image labels on the picked mesh, by name.
