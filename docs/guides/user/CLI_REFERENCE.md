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
luxar serve     # Serve a .luxar.zarr over HTTP, optionally with the viewer (--viewer, --open)
luxar viewer    # Serve the Luxar viewer, optionally with a dataset (--data)
luxar export    # Export a scene + viewer as a standalone offline folder (or --native bundle)
luxar profiles  # List the network-simulation profiles usable via --profile
```

- Serving and viewing → [Viewer guide](./VIEWER_GUIDE.md)
- Offline / native export (and the `make build-launchers` prerequisite) →
  [Build system guide](../developer/BUILD_SYSTEM_SPEC.md)
- Network profiles → [Network simulation spec](../developer/NETWORK_SIMULATION_SPEC.md)

## `luxar demo`

Browse, run, and manage the bundled demos.

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
luxar gsplat napari           # Open a dataset in napari for visual inspection
luxar gsplat view             # Open a .gsplats.zarr directly in the web viewer
luxar gsplat compare          # Compare reconstruction quality vs a reference volume (PSNR/SSIM/MSE)
luxar gsplat annotate-quality # Retrofit Q·e quality stamps onto an existing dataset, in place
```

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
luxar mesh import            # Import a classical mesh file (PLY / OBJ / STL / glTF / GLB) → a .luxar.zarr scene
luxar mesh lod               # Build a substitutive LOD ladder for a mesh scene
```

Vertices are welded and polygons fan-triangulated on the way in, because STL is always
a triangle soup and glTF sometimes is; an unwelded surface defeats per-vertex normals
and makes picking report a different vertex per triangle for the same corner. Welding
merges two vertices only when their position *and* their normals and colours agree, so
a hard edge — which every modelling package authors as coincident positions with
different normals — survives the import instead of being flattened. Pass `--no-weld` to
keep the file's exact vertex list.

Draco- and meshopt-compressed glTF is refused by name rather than decoded — run the
file through `gltf-transform` first.

### `luxar mesh lod`

Writes a `kind=lod` group whose coarse children are progressively **decimated** copies
of the surface and whose finest child is the original. The viewer shows exactly one at a
time, chosen by how much of the screen the object covers.

Takes an input scene and an output scene, plus `-L/--levels` (default 3),
`-K/--compression-factor` (default 4 — level *i* targets `V / K**i` vertices),
`--node`, `--subst-method` and `--overwrite`. The output path is normalized to the
canonical `<stem>.luxar.zarr`, so an output argument of `out` writes `out.luxar.zarr`; that
normalized path is what `--overwrite` replaces and what the same-path guard
compares against.

Unlike `luxar gsplat lod` this reads and writes a **scene**, not a standalone store —
there is no standalone mesh format, so the only sink for a mesh is a `.luxar.zarr`. The
node is optional when the scene holds exactly one mesh (what `luxar mesh import`
produces); with several, naming one is required rather than guessed at.

`--subst-method` shares its **name** with `luxar gsplat lod` — on both commands it selects
the substitutive, level-replacing reduction — but **not its values**: this one takes `auto`
or `cluster`, because a mesh is decimated where a gsplat level reduces a Gaussian mixture,
which a surface is not. `auto` resolves to `cluster` today.

The flag was called `--method` before August 2026, and `-m` was its short form. Both are
gone: `-m` is reserved for the additive ordering it already names on `gsplat lod`. Either
old spelling exits with a pointer naming the replacement and carrying your value, rather
than silently doing something else.

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
