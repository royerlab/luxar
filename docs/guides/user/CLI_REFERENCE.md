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
`packages/luxar/src/luxar/cli/tests/test_docs_command_coverage.py`. The test walks
the registered command tree, extracts every `luxar ...` invocation from the code
blocks on this page, and compares the two sets in **both directions**: a newly
added command must be documented here, and a documented command that was removed,
renamed, or hidden must be pruned — either way the test fails.
(Hidden/internal commands and groups are excluded.)

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
luxar demo deps          # Optional dependencies (--install, --extra, --only MODULE)
luxar demo cache list    # Inventory the ~/.cache/luxar demo caches
luxar demo cache clear   # Clear demo caches (--dry-run to preview)
```

Report-only `demo deps` exits 1 for any missing or outdated row. Generic
`--install` manages Luxar extras; use `--only MODULE --install` to install one
exact constrained requirement, including a dependency outside every extra.

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

Bring classical triangle-surface files into Luxar. The reader is NumPy + stdlib only,
so this works on a bare `pip install luxar` with no extras.

```bash
luxar mesh import            # Import a classical mesh file (PLY / OBJ / STL / glTF / GLB) → a .luxar.zarr scene
```

Vertices are welded and polygons fan-triangulated on the way in, because STL is always
a triangle soup and glTF often is; an unwelded surface defeats per-vertex normals and
makes picking report a different vertex per triangle for the same corner. Pass
`--no-weld` to keep the file's exact vertex list.

Draco- and meshopt-compressed glTF is refused by name rather than decoded — run the
file through `gltf-transform` first.
