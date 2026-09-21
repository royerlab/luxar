# Luxar CLI Package

This package contains the command-line interface (CLI) for Luxar, providing tools for serving, viewing, and inspecting Luxar zarr datasets.

## Quick Start

Essential CLI commands in 3 steps:

```bash
# 1. List the bundled demos, then run one (fastest way to see Luxar)
luxar demo
luxar demo run lorenz

# 2. Serve your own dataset with the viewer
luxar serve my_data.luxar.zarr --viewer

# 3. Get dataset information and statistics
luxar info my_data.luxar.zarr --stats
```

The same commands are also available programmatically: `luxar.cli` exports the
Typer `app`, which you can invoke directly (e.g. from a script or a test):

```python
from typer.testing import CliRunner

from luxar.cli import app

runner = CliRunner()
result = runner.invoke(app, ["info", "my_data.luxar.zarr", "--stats"])
print(result.stdout)
```

**What Each Does**:
- `luxar demo` - Lists the bundled demos; `luxar demo run <key>` generates one and opens it in the viewer
- `luxar serve --viewer` - Serves your data via HTTP alongside the viewer (add `--open` to launch the browser)
- `luxar info --stats` - Shows dataset structure, dimensions, and compression stats

**Pro Tips**:
- Browser launch: `viewer` and `gsplat view` open the browser by default (add `--no-open` to skip); `demo run` opens a viewer too but is stopped with a forwarded `--no-serve` (it has no `--no-open`); `serve` and `export` do not open by default (add `--open` to launch it)
- Use `luxar profiles` to list network simulation profiles
- Use `luxar serve --help` for all serving options
- Use `luxar export --native macos` (or `linux-amd64`/`linux-arm64`) for double-clickable native bundles backed by an embedded Go launcher (requires `make build-launchers` first)
- For complete working scripts that generate datasets for these commands, see `packages/luxar/examples/` and repository-level `examples/` when present.

## Module Structure

- `__init__.py` - Package initialization, exports the main app
- `main.py` - Main CLI application with the top-level commands
- `_traceback.py` - Shared quiet-error reporting with the `LUXAR_TRACEBACK` opt-in escape hatch
- `serving.py` - HTTP serving internals (`create_server_app`, data/viewer servers; re-exported by `main.py`)
- `info_command.py` - The `luxar info` command implementation
- `optimize_command.py` - The `luxar optimize` command (a thin Typer layer over `luxar.io.optimize`)
- `restamp_lod_command.py` - The `luxar restamp-lod` command (a thin Typer layer over `luxar.io.lod_restamp`)
- `gsplat_commands.py` - Thin registration hub (~56 lines) that assembles the `gsplat` sub-app: fit, cal, render, denoise, lod, convert, migrate-format, reencode, info, napari, view, compare, annotate-quality, transform, merge, cull, filter, slice, partition, flatten, additive, benchmark; the `batch-fit` group: run/submit/status/validate/cancel/merge/denoise-calibrate/denoise-preprocess
- `gsplat_ops/` - The gsplat subcommand implementations: 8 root modules (scene/inspect/interchange registration, `benchmark`, `recipe_shared`, `planner`, `encoding`, `loading`) plus three subpackages — `fitting/` (fit/cal/render/denoise), `batch/` (`batch-fit`), `transforms/` (edit-style commands) — 31 modules across them. Each subpackage's registration surface is its `commands.py`; the `__init__.py` files are docstring-only. See `gsplat_ops/README.md`.
- `lod.py` - the unified `lod --recipe {flat,stream,levels,tiles,overview,adaptive}` command (thin wrapper over `gsplats/lod/recipes.py`; registered onto the `gsplat` app)
- `gsplat_config.py` - Config system: presets, YAML loading, volume loaders, helpers
- `demo_commands.py` - The `luxar demo` sub-app: list/info/run/run-all/stop/deps/cache, driven entirely by the `luxar.demos` DEMO_META registry
- `demo_runs.py` - Stdlib-only discovery and process-group teardown engine behind `luxar demo stop`; tracks live runs under `~/.cache/luxar/running/` and sweeps the process table for unregistered demos
- `demo_render.py` - Presentation for all four `luxar demo` listings: the catalogue, one demo's detail record, the dependency report, and the cache inventory. The one place in the Python CLI that renders through `rich` and writes to plain stdout instead of `aprint` — a five-column catalogue with one row per bundled demo needs real column layout, and arbol's `├` tree prefix belongs on nested progress output, not on rows meant to be scanned and copy-pasted. The split is by what the output *is*: a standalone inventory you read renders here; anything interleaved with an action (progress, confirmations, install advice, the running-demo list `demo stop` prints before killing them) stays on arbol. Every column width is measured in terminal **cells**, not code points.
- `../_process.py` - Shared package-root child-process lifecycle primitive; see `../README.md`
- `utils.py` - Utility functions for CLI operations
- `export.py` - Standalone scene export (viewer + data + serve script)
- `_export_serve_template.py` - The `serve.py` that `luxar export` writes into an export folder, copied with two values substituted. A real module rather than a string inside `export.py` so ruff, mypy and the test suite see it — it hosts the touch-panel relay behind `--control`, and a WebSocket relay hidden in an f-string (where every brace has to be doubled) is unreviewable. **Stdlib only**, because an export folder gets zipped and handed to someone who has never installed Luxar. It is the third implementation of the relay in `control_hub.py`, so it carries a copy of the generated wire contract that `tests/test_export_control_relay.py` pins to `_control_contract.py`.
- `native_app.py` - Native bundle producers (macOS `.app`, Linux portable folder) for `luxar export --native`
- `network_simulation.py` - Network simulation middleware and profile definitions
- `_launchers/` - Go-compiled launcher binaries (populated by `make build-launchers`; ride along in wheel builds)
- `_launcher_assets/` - Bundle icons (`luxar-logo.png`, `AppIcon.icns`) used by `native_app.py`

### `demo_runs.py`

Discovery and teardown engine behind `luxar demo stop`: find every running demo,
including one forgotten in another terminal, and free its ports.

**Key Functions:**
- `register_run()` / `unregister_run()`: Maintain one JSON pidfile per launch
  under `~/.cache/luxar/running/`; the registry only names processes that may
  have survived their owner
- `discover_runs()`: Combine live registry entries with a process-table sweep
  for stray `python -m luxar.demos.demo_*` group leaders; prune dead or hijacked
  entries and never return the caller's own process group. Linux falls back to
  `_process.proc_table()` when `ps` is unavailable, preserving the identity
  check that protects a recycled process-group ID. Off POSIX, `tasklist` still
  prunes records left by rebooted or hard-killed owners
- `stop_run()`: Revalidate and terminate one run's process group; returns False
  without signalling off POSIX, where a recorded PID cannot be safely checked
  before a hard kill
- `describe_port_holder()`: Best-effort `lsof` hint for `pick_port()`'s
  busy-port warning when a requested demo port looks Luxar-owned

## Available Commands

### `luxar demo`
Browse, run, and manage Luxar's bundled demos. Bare `luxar demo` prints the
demo table; run one by key or index (extra args are forwarded to the demo).
```bash
luxar demo                        # List all demos (key, needs, status)
luxar demo list --category astronomy   # Filter the table
luxar demo info lorenz            # Full details for one demo
luxar demo run lorenz             # Generate + open the viewer
luxar demo run 3                  # Run by table index
luxar demo run lorenz -- --no-serve --points=100000  # Forward args to the demo
luxar demo stop                   # Stop running demos and free their ports
luxar demo stop lorenz --dry-run  # List one demo's live runs, stop nothing
luxar demo cache list             # Inventory ~/.cache/luxar demo caches
luxar demo cache clear lorenz --dry-run   # Preview a cache clear
luxar demo deps                         # Which optional demo deps are missing?
luxar demo deps --install               # Install the extras that provide them
luxar demo deps --extra io              # Restrict to one extra
luxar demo deps --only scipy            # Restrict to one import module
luxar demo deps --only scipy --install  # Install its exact constrained spec
```
In report-only mode, `deps` exits 1 when anything is missing or out of date, so
it doubles as a CI/setup gate. It reports the *constrained* requirement from
`luxar.demos.INSTALL_SPECS` — the same table the runtime `require_module` gate
uses — and surveys with `find_spec`, so it never imports `torch` just to tell you
`torch` is present. A package that imports but whose version is below its pinned
floor is flagged `OUTDATED` rather than `ok` (the version check is best-effort:
it needs `packaging`, and gives the benefit of the doubt when it cannot decide).

Generic `--install` installs Luxar extras and judges only what that pip command
attempted. An unmet row outside every extra is reported but does not make an
otherwise successful extras install fail; if it is the only row, the command is
a successful no-op. Use `--only MODULE --install` to install and verify that
row's exact tabled requirement directly (including an orphan such as `gdown`).
`--only` and `--extra` are mutually exclusive.

### `luxar serve`
Serve zarr datasets or directories via HTTP.
```bash
luxar serve data.luxar.zarr         # Serve data
luxar serve data.luxar.zarr --viewer # Serve with viewer
luxar serve --viewer-only      # Serve only viewer
```

Security defaults are optimized for local development: CORS allows local
browser origins (`localhost`, `127.0.0.1`, `[::1]`; `0.0.0.0` is deliberately
excluded) by default, directory listing
requests cannot escape the served root, and obvious system paths such as `/`,
`/etc`, `/proc`, `/sys`, and `/dev` are refused unless you pass
`--allow-sensitive-path`. Use `--cors-origin '*'` only when you intentionally
want any website to read the served data; wildcard mode disables credentials.

### `luxar viewer`
Serve the Luxar viewer with optional data.
```bash
luxar viewer                  # Serve viewer
luxar viewer --data data.luxar.zarr # Serve viewer with data
luxar viewer --no-open        # Don't open browser
```

### `luxar info`
Display detailed information about zarr datasets.
```bash
luxar info data.luxar.zarr          # Basic info with tree view
luxar info data.luxar.zarr --stats  # Include detailed statistics
luxar info data.luxar.zarr --format json # JSON output
```

`--stats` also reports the store's **chunk layout** — average chunk KB, the
share of arrays under the 16 KB floor, and the projected request count for a
full load — computed off the same helper `luxar optimize` plans from, so a
store that is badly chunked for streaming is visible without hosting it first.
When the mean chunk is under 32 KB it also warns that the load will be
round-trip bound on an HTTP/1.1 host (which `luxar serve` is): measured 10.6 s vs
4.0 s over HTTP/2 for the same 1.5 M-point example at 25 Mbps / 30 ms RTT.

### `luxar optimize`
Re-chunk an existing store for streaming. One structure-preserving pass: only
zarr chunk shapes change, values stay bit-identical, and no refit / source
volume / GPU is involved.
```bash
luxar optimize scene.luxar.zarr optimized.luxar.zarr
luxar optimize scene.luxar.zarr --dry-run                    # report only
luxar optimize scene.luxar.zarr out.luxar.zarr --target-kb 128
luxar optimize scene.luxar.zarr out.luxar.zarr --profile hosting  # 256 KB
luxar optimize scene.luxar.zarr out.luxar.zarr --profile local    # 64 KB
luxar optimize scene.luxar.zarr out.luxar.zarr --profile archive  # 1 MB
luxar optimize scene.luxar.zarr out.luxar.zarr --verify      # re-read + compare every array
luxar optimize fit.gsplats.zarr fit_opt.gsplats.zarr         # standalone gsplat trees
luxar optimize arbitrary.zarr out.zarr --generic             # plain zarr
```

dtype, codecs, filters, `fill_value`, memory order, the chunk key layout, the
zarr format version and every attribute **except** the two the pass is required
to move — the root's `content_hash` and the `chunk_layout` summary beside it —
are preserved; a sharded array keeps its shard grid; and the spatial-index grid
(`chunk_size` / `chunk_bounds`) is never moved, so every new chunk is a whole
multiple of its node's atom. Nothing is ever chunked *smaller* than it already
is. Those two attrs are the cache guard: without them a warm viewer cache would
keep serving chunks whose keys now cover different rows, and the restamp runs
for `--generic` too, since the flag describes the input. Overwriting an existing
output needs `--overwrite` and only replaces a zarr store or an empty directory;
a destination that contains the source is refused, as is rewriting in place, as
is a destination that is a symlink (the rename would replace the link, not its
target — pass the target instead). The output is staged beside the destination
and moved into place last, and a previous store is renamed aside rather than
deleted first, so a failure leaves nothing partial behind and never costs both
copies. Larger profiles trade partial-query bytes for
full-load requests — see the CLI reference before reaching for `--profile
hosting` on a store the viewer will slice into. The logic lives in
`luxar.io.optimize`.

### `luxar restamp-lod`
Re-derive a store's LOD switch thresholds in place. An attributes-only pass: the
ladder rewrite moves no chunk data and opens no array.
```bash
luxar restamp-lod scene.luxar.zarr                      # every legacy ladder
luxar restamp-lod scene.luxar.zarr --dry-run            # report only
luxar restamp-lod scene.luxar.zarr --group tiled/part_0 # one ladder (repeatable)
luxar restamp-lod scene.luxar.zarr --anchor 0.25        # re-anchor whole-object ladders
luxar restamp-lod fit.gsplats.zarr --group /            # the gsplats root ladder
```

Every `kind=lod` group still on the legacy `coverage` diagonal metric (or
carrying no `selector`, which means the same) gets its per-child
`coverage_fraction` thresholds re-derived by screen-occupancy halving — the
whole-object anchor normally, the fills-screen one when the ladder is
tile-bound — and its group stamped `screen-area`. Tile-bound is the tree
writers' full rule: a real multi-part `kind=partition` above the ladder, OR a
`kind=partition` among the ladder's own children (the `overview` recipe's coarse
cap, which is pinned at fills-screen on purpose). A group already on
`screen-area` is skipped by default, so a second run changes nothing,
`content_hash` included. `--anchor` sets the requested finest area for every
whole-object ladder the pass processes, both legacy ladders being migrated and
already-`screen-area` groups rebuilt from their stored level count.
Partition-bound ladders keep their fills-screen `1.0` anchor. A ladder already
matching the requested anchor remains a no-op, including its `content_hash`.

It is never automatic: an authored `coverage_fractions=[...]` list and a derived
one are indistinguishable on disk, including a hand-authored ladder already
stamped `screen-area`, so running the command IS the opt-in and the per-group
old→new ladder is printed as the audit trail. Sibling of
`luxar optimize` rather than a flag on it — that pass preserves every attribute
and refuses same-path work; this one changes only attributes and works in place.
A `.zarr.zip` is refused (nothing to write back to). When anything changes the
`content_hash` is restamped and the metadata re-consolidated, then read back and
verified from both the per-node documents and the consolidated index — and that
restamp is the one costly step, since a SCENE's digest covers array values and
therefore reads the whole store once (a standalone `.gsplats.zarr` gets a
metadata-only stamp). An index is rebuilt, never introduced: a store that
arrives unconsolidated leaves that way, since `is_consolidated` is how
`batch-fit` tells a finished tile from an interrupted one. A failed write is
rolled back rather than left half applied — digests restored as the store had
them rather than recomputed, and the index re-consolidated only if the run had
rewritten the root. Exit code 1 when any ladder was left alone for a reason worth acting
on, and also when ladders were rewritten in a store that carries no
`content_hash` to restamp — the rewrite landed, but nothing invalidates a warm
viewer cache until the store is republished under a new URL prefix. The logic
lives in `luxar.io.lod_restamp`.

### `luxar profiles`
List available network simulation profiles for testing.
```bash
luxar profiles                # Display all network profiles with descriptions
```

**Available profiles:** 3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested

Use these profiles with `serve`, `viewer`, or `demo` commands via the `--profile` option to simulate various network conditions for testing.


### `luxar export`
Export a zarr scene and the Luxar viewer as a standalone offline folder. The output is self-contained: anyone can view the scene with just Python 3 and a browser by running `serve.py`.
```bash
luxar export my_scene.luxar.zarr -o my_export/              # Export scene + viewer
luxar export my_scene.luxar.zarr -o my_export/ --overwrite   # Overwrite existing export
luxar export my_scene.luxar.zarr -o my_export/ --open        # Export and serve in browser
luxar export my_scene.luxar.zarr -o my_export/ --open --port 9000  # Custom port
```

**Options**: `--output/-o` (required), `--overwrite`, `--open` (serve and launch browser), `--port/-p` (port for local server, default 8000).

#### `luxar export --native` (double-clickable bundles)
Produce a double-clickable native bundle instead of the Python `serve.py` folder. The bundle wraps the viewer + zarr around a Go-compiled launcher binary that opens an embedded WebView (WKWebView on macOS, WebKitGTK on Linux).

```bash
luxar export my_scene.luxar.zarr -o out/ --native macos                       # macOS .app
luxar export my_scene.luxar.zarr -o out/ --native linux-amd64                  # Linux folder (x86_64)
luxar export my_scene.luxar.zarr -o out/ --native macos,linux-amd64,linux-arm64 \
                                          --name MyScene                # All three at once
```

**Options** (in addition to the parent command's): `--native PLATFORMS` (comma-separated; choices: `macos`, `linux-amd64`, `linux-arm64`), `--name NAME` (defaults to the zarr stem), `--zip/--no-zip` (when `--native macos`, also produces a sibling `<name>.app.zip` — via `ditto` on Darwin to preserve resource forks and extended attributes, or `zipfile` on cross-build hosts with executable-bit preservation; on by default, pass `--no-zip` to skip).

**Prerequisites**: run `make build-launchers` first to populate `cli/_launchers/` with the host-platform binary. `--native` produces `macos`, `linux-amd64`, and `linux-arm64` bundles only. CGO blocks pure cross-compilation, so each platform's binary must be built on a host of the matching OS (typically via CI).

**Runtime fallback**: setting `LUXAR_LAUNCHER_NO_WEBVIEW=1` makes the launcher open the user's default browser instead of an embedded WebView — useful for headless smoke tests. It does not let the prebuilt Linux binary run without `libwebkit2gtk`: WebKit is linked at build time, so the launcher needs the `webkit2gtk-4.1` runtime to start regardless.

See `packages/luxar-launcher/README.md` for the launcher source itself.


### GSplat Processing Commands

#### `luxar gsplat info`
Print dataset statistics for a `.gsplats.zarr` (splat count, dimensions, bounds, LOD structure).
```bash
luxar gsplat info splats.gsplats.zarr
```

#### `luxar gsplat view`
Open a `.gsplats.zarr` in the web viewer. The standalone `.gsplats.zarr` is a v3.4 node subtree — exactly what the viewer renders inside a scene — so it is served **directly** via `?src=` with no scene-compile round-trip (works for a single leaf, an additive ladder, a `kind=lod` hierarchy, a `kind=partition` split, and arbitrary nestings; archives are extracted first).
```bash
luxar gsplat view splats.gsplats.zarr
```

#### `luxar gsplat fit`
Fit Gaussian splats to a volume with preset or YAML config.
```bash
luxar gsplat fit volume.npy splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit volume.tiff splats.gsplats.zarr --config params.yaml
luxar gsplat fit --dump-config --preset hifi > config.yaml  # Generate config template
```

**Presets:** `draft` (fast preview), `standard` (balanced), `hifi` (high quality), `ultra` (max quality)

`--floor` (default `auto`) subtracts a background pedestal (clip at 0) before normalization, so output amplitudes are background-relative. `auto` = histogram-mode estimate (a no-op on clean data); `pNN` subtracts that percentile of non-zero voxels; a plain number is a fixed level; `none` and `0` both disable suppression (legacy hard-min). Both volume-derived forms exclude exact-zero voxels so masked/out-of-FOV padding does not move their population. Negative user levels are rejected, while a negative level produced internally from dark-frame-corrected data is retained. With a supplied `norm_range`, the erase-all guard and high endpoint use that whole-volume range unchanged so every tile keeps the same physical scale. Uniform and content fits resolve that raw-input range once for the whole selected volume and forward it to every tile or box; a degenerate range is declined so each child can fall back to its own usable scale. Otherwise the guard compares the resolved level with the data maximum (the bounded sample maximum on lazy whole-volume paths), never the configured normalization percentile's high endpoint. With a nonzero `norm_percentile` from YAML, normalization's percentile-derived low endpoint remains an independent lower bound: if it is above the requested floor, that higher endpoint is what is actually subtracted and recorded; if the floor rises above the percentile-derived high endpoint but remains below the data maximum, the high endpoint expands to that maximum so real signal is retained. Under **any** tiling the spec is resolved against the **whole volume**, never against a tile or box crop — which would make abutting regions fit against different baselines and show brightness steps at their boundaries. `uniform` and `content` resolve it once in the parent and hand every tile/box that concrete level (each child's normalization floor is still clamped up to the shared range's low endpoint when that endpoint lies above the level); a `-j N` parent that had to resolve the level to weigh its tiles forwards that concrete level and the raw-input range to its workers, and a hand-run `--tile k/M` worker resolves the same spec against the same whole volume, which agrees because the sampler is deterministic. `batch-fit` extends both across time: one global level for the whole timelapse, resolved at plan time as the **minimum** of the levels measured on a bounded set of evenly spaced `(t, c)` slices spanning the whole store — up to 4 timepoints (always including `t=0` and `t=T-1` when `T > 1`) x up to 4 channel-like coordinates — plus, without denoising, one raw-input range reduced over those same samples and recorded in the manifest (`floor_level`, `norm_range`). Denoising batches leave that automatically sampled range unset so each task resolves on the data it fits: denoise-corrected input for an on-the-fly uniform tile, or the denoised store in `preprocess` mode (including content boxes). An explicit configured `norm_range` remains an intentional override. A minimum is a lower bound on every **sampled** slice's pedestal, so it cannot clip a sampled timepoint or channel to zero (which the tile worker would report as a legitimately empty tile, silently dropping that slice from the merge). Bounded sampling can only bound what it samples: a dimmer NON-sampled slice (a blank/bleached frame between samples, a channel above the cap) can still be erased that way — use `--floor none` or an explicit numeric `--floor N` when a particular slice must survive. With `--denoise` on a `uniform` tiled fit the level is resolved on the **denoised** basis (each tile is denoised before the level is subtracted, so it has to be — #1178), which means cross-worker agreement now also depends on every worker receiving the same `h` (and the same `--denoise-backend`), not only the same volume and spec. That correction is measured on a bounded denoise probe, so it is exact for a volume the probe covers whole and, above that, is applied for a `pNN` floor only: with the default `--floor auto` on a large volume the level stays on the raw basis and one note says so, since a crop-measured histogram-mode shift is dominated by noise. Pass `--floor pNN` (or fit un-tiled) if you want the level itself resolved on denoised data.

An integer `--seeds K` is a **whole-volume** budget (what a default `gsplat cal` reports as K\*): a tiled fit — `--tiling uniform`, a large `--tiling auto` volume, a `--tile k/M` worker — divides it across the tiles that survive floor subtraction and Hann windowing rather than giving each tile the full count. Uniform `fit` and `batch-fit` build the same folded grid; whether the division is **weighted** depends on the entry point. Occupancy-weighted (measure foreground at one shared 10%-of-range level, apply the Hann window and the content planner's sublinear saturation exponent, then divide K proportionally with deterministic rounding): `--tiling uniform` run as one command, both the sequential in-process path and a `-j N` parent forwarding an exact `--tile-seed-count` per worker; and `batch-fit` whenever the plan resolves tile-local reads. Equal share per non-empty tile: a **hand-run** `fit --tile k/M` worker (nobody handed it a count, and scanning the whole volume per worker is the cost the shared plan removes); and a `batch-fit` plan that cannot resolve tile-local reads — `--downscale`, `--denoise`, a deferred or volume-derived floor, or no plan-resolved `--norm-range` — which it announces as `Tile-local reads off: <reason>`. A Slurm plan additionally drops the seed table, with a notice, when the inline table would exceed the script's size cap, so a Slurm run and a local `batch-fit run` of the same plan can allocate differently. The weighting preserves K whenever K is at least the non-empty tile count; a smaller K gives one seed per non-empty tile, and a tile weighted to zero is skipped rather than fitted. A float ratio in `(0, 1]` is scale-free and applied per tile unchanged, so it also scales a narrow edge tile by its voxel count; `--tiling content` ignores `--seeds` entirely (per-box budgets come from the density plan).

A folded tile is the one case where a tile is **larger** than `--tile-size`: the predecessor that absorbs a trailing sliver spans up to `tile_size + overlap - 1` voxels on that axis — 287 at the 256/32 default, i.e. 1.41x the voxels of a full tile in 3D (256/64 is 1.93x). Size `--tile-size` for that worst case. `-j auto` and `batch-fit run` already measure the real maximum when sizing workers; the sequential path neither accounts for it nor warns.

#### `luxar gsplat convert`
Convert .gsplats.zarr to a Luxar scene for the web viewer.
```bash
luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --center
luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --scale-intensity 0.1
```

#### `luxar gsplat render`
Render gsplats back to a volume for quality comparison. Partition and nested
trees render their default-selected leaves without first flattening the store.
```bash
luxar gsplat render fitted.gsplats.zarr rendered.npy --shape 128,128,128
luxar gsplat render fitted.gsplats.zarr rendered.tiff --device cuda
```

#### `luxar gsplat merge`
Combine multiple flat gsplat datasets (concatenation, new dimension, or channel
colors). Partition/nested inputs must be flattened first with `luxar gsplat flatten`.
```bash
luxar gsplat merge a.zarr b.zarr -o merged.zarr
luxar gsplat merge t0.zarr t1.zarr t2.zarr -o 4d.zarr --as-dimension --values 0,1,2
luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr --channel-colors "#ff0080,#00ff00"
```
The inputs' authored appearance (blending mode, opacity, colormap, …) is carried
onto the merged root wherever the inputs **agree**; an input with no opinion on a
key casts no vote, and a key they genuinely disagree on is dropped with a warning
naming the differing values (each with the input it came from) and what lands
instead, rather than one input's choice being promoted. A plain save STAMPS the identity values (`opacity=1.0`,
`colormap="gray"`, …), and a value equal to such a stamp counts as no opinion —
so merging a tuned dataset with a freshly fitted one keeps the tuned look instead
of dropping seven keys back to those same defaults. The flip side: a deliberately
authored identity is indistinguishable from an untouched store and loses to a
sibling's value. `visible` is the exception where ABSENCE votes ("shown"), so
`visible=false` is carried only when every input hides. `colormap` is dropped
even under agreement whenever the merge gave the output per-splat RGB an input
did not have — `--channel-colors`, or the white fill a mixed colored/colorless
merge applies — whenever a colored input authored no palette and therefore
relies on its per-splat RGB, and whenever any input declares a custom LUT.

#### `luxar gsplat cull`
Remove low-contribution splats to reduce dataset size while preserving visual
quality. Partition/nested inputs must be flattened first with `luxar gsplat flatten`.
```bash
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr                            # Auto (cumulative, keep 95%)
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90      # Keep 90% amplitude
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m redundancy --shape 41,512,512  # GPU, no target
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy           # Error-budget (most principled)
```

#### `luxar gsplat filter`
Filter splats by multiple criteria (AND logic). Partition/nested inputs must be
flattened first with `luxar gsplat flatten`.
```bash
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --scale-max p90            # drop largest 10% (diffuse background)
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --scale-max p90 --dry-run  # preview impact, write nothing
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --isolation-max p99        # strip the 1% most-isolated (noise)
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --soft-highpass p90        # attenuate large splats (no popping)
```

**Criteria**: `--bbox`, `--amplitude-min/max`, `--scale-min/max` (characteristic size = geometric-mean spatial sigma; timelapse-safe — `--scale-max` is the recommended background-removal knob), `--volume-min/max`, `--eccentricity-min/max`, `--mass-min/max`, `--sigma-axis`/`--sigma-min/max`, `--isolation-max` (nearest-neighbour distance), `--min-neighbors`/`--neighbor-radius` (local density), `--soft-highpass`/`--soft-lowpass`/`--soft-width` (soft reweighting: attenuate amplitude by scale instead of deleting). Every min/max threshold accepts a plain number OR a percentile written as `pNN`/`NN%`. Supports `--*-normalized` flags (volume/scale/amplitude/mass). Extras: `--spatial-dims` (override auto axis detection), `--dry-run` (report impact without saving), `--truncate` (sigma truncation for volume computation).

#### `luxar gsplat partition`
Partition a dataset into a single `kind=partition` file via spatial BSP
(`--rule median|midpoint|sah`). The input must be flat (matrix-shaped); flatten
partition or nested-tree inputs first with `luxar gsplat flatten`.
```bash
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --parts 4
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --max-elements 100000
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --parts 3 --rule sah --compress zip
```

#### `luxar gsplat slice`
Slice splats by coordinate ranges (numpy-style syntax). Partition/nested inputs
must be flattened first with `luxar gsplat flatten`.
```bash
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "0:50, :, 10:90"
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr ":50, 20:80, :"
```

#### `luxar gsplat compare`
Compare reconstruction quality against a reference volume (PSNR, SSIM, MSE). Partition and nested stores are scored over their default-rendered selection: all parts and each LOD group's finest level. The reported compression ratio covers the whole store, including coarse levels that are not scored.
```bash
luxar gsplat compare fitted.gsplats.zarr original.tiff
luxar gsplat compare fitted.gsplats.zarr original.npy --device cuda --output-json metrics.json
```

#### `luxar gsplat cal`
Calibrate the splat count `K` via blind-spot cross-validation. Sweeps `K`, identifies the held-out PSNR peak (`K*`) using the manuscript's Noise2Self protocol (5% donut-median masking), and reports the noise floor. Defaults to the `n2s` fit preset so the held-out curve has enough optimizer budget to reach the overfit regime.
```bash
luxar gsplat cal volume.tiff cal.json                            # 10-point sweep, [1K, 512K]
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000  # Faster sweep
luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'  # Explicit
luxar gsplat cal volume.tiff cal.json --pdf cal_report.pdf       # Multi-page PDF report
luxar gsplat cal volume.zarr cal.json --progression power --power 2  # Polynomial K spacing
```

**Options**: `--k-grid` (explicit comma-separated K values), `--n-grid` (default 10), `--k-min` (default 1000), `--k-max` (default 512000), `--progression` (exp/power), `--power`, `--mask-seed`, `--mask-fraction` (default 0.05), `--preset` (default n2s), `--config`, `--floor` (background floor / DC-offset suppression, default `auto` — K* is measured on floor-suppressed data, matching `fit`; pass `none` for the legacy hard-min behavior), `--device/-d`, `--k-star-metric` (psnr_minmax/psnr_foreground/psnr_fg_weighted/gain), `--fg-bg-ratio` (default 1, equal total foreground/background weight for `psnr_fg_weighted`), `--auto-region/--no-auto-region` + `--region-size`/`--region-strategy` (calibrate on a content-rich sub-region), `--feature-metric`, `--saturation-exponent`, `--rd-model/--no-rd-model`, `--fit-exponent`/`--exponent-scales` (measure the saturation exponent alpha instead of assuming the default), `--pdf` (multi-page PDF report), `--keep-fits` (persist per-K fits), `--quiet/-q`, plus volume-loader pass-through (`--channel/-c`, `--timepoint`, `--array-key`, `--axes`). For sparse or deconvolved volumes, prefer `psnr_fg_weighted`: it balances signal fidelity against background haze. `gain` remains a useful reported diagnostic, but its K* is identical to `psnr_minmax` because the predict-zero baseline is constant across K.

#### `luxar gsplat migrate-format`
Convert a legacy `.gsplats.zarr` layout to the current node-tree format. Five input shapes are auto-detected: v1.0 (single flat splat set), v1.1 (multi-LOD additive `/splats/lod_<i>/` subgroups), a pre-v2.0 substitutive directory (`manifest.json` + `level_<i>.gsplats.zarr`), the v2.0 `substitutive_<s>/additive_<a>/` matrix, and a v3.0/v3.1 store whose `kind=lod` groups still carry the pre-v3.2 `pixel_size` selector attrs (rewritten as `selector: "screen-area"` + derived per-child `coverage_fraction`). All migrate to a single current-format (v3.4) `.gsplats.zarr`. By default the output adopts the AUTO encoding policy, so legacy float32 Cholesky factors are re-encoded as the split diagonal/off-diagonal arrays with certified uint8 per-column quantization (the encode-time covariance certificate escalates to uint16 when the measured Σ error demands it); pass `--lossless` to keep them float32 for archival fidelity.
```bash
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr             # single file (AUTO encoding)
luxar gsplat migrate-format old_pyr/ v3.gsplats.zarr                        # substitutive directory
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr --lossless  # preserve float32 Cholesky exactly
```

**Options**: `--overwrite`, `--lossless` (preserve float32 Cholesky / PRECISION encoding), `--quiet/-q`.

#### `luxar gsplat reencode`
Re-quantize a **current-format** `.gsplats.zarr`'s Cholesky encoding (writes a re-quantized copy to a new path) — a structure-preserving round-trip: the whole node tree (leaf / additive ladder / `kind=lod` / partition / nested) and its `fitting` / `provenance` / `pipeline` groups carry over verbatim. Splat count and structure are unchanged and decode is always to float32, so viewer/GPU/WASM paths are unaffected — but the Cholesky factors are not the only array re-encoded: `auto` and `memory` also re-quantize the CENTERS to per-axis uint16 fixed-point (float32 once an axis spans 2¹⁶, or once a *non-gridded* axis's grid would displace **more than 0.1% of the splats** past their own σ — a smaller degenerate population is quantized away silently; a *gridded* axis such as a stacked `sigma=0` time axis instead keeps uint16 with its grid snapped onto the data's own spacing, which is also exact), so on ordinary spatial data center values are bit-exact only under `precision`. Unlike `migrate-format` (legacy → current, exposing only float32 vs the AUTO uint8-first policy via `--lossless`), this exposes the full ladder — including `memory` (uint8) — and works on already-current files. The clean way to change quantization after fitting.
```bash
luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory      # uint8 (smallest, ~93 dB)
luxar gsplat reencode fit.gsplats.zarr fit_auto.gsplats.zarr -e auto       # adaptive u8→u16→f32 ladder (near-lossless by certificate)
luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision   # float32 (exact/archival)
```
**Options**: `--encoding/-e` (`auto`|`precision`|`memory`, default `memory`), `--ordering` (`hilbert`|`morton`|`none`), `--quiet/-q`.

#### `luxar gsplat lod`
Build a **representation topology** from a pre-fitted `.gsplats.zarr` via a single
required `--recipe` flag. Recipes are scale-ordered: `flat`, `stream`, `levels`,
`tiles`, `overview`, `adaptive`. (The pre-2026-07 names — additive, substitutive,
pyramid, partitioned, multiscale, mosaic — error with a pointer to the new name.)
Output is a standalone `.gsplats.zarr` (graft into a scene from Python via
`add_gsplats_from_file` / `gsplat convert`).

```bash
# flat / stream — single leaf, optionally with an additive (prefix-sum) ladder
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe flat
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream \
    --breakpoints energy:0.5,0.9,0.99,1.0                                  # cumulative energy fractions
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --add-method self_energy

# tiles / overview — BSP parts each with an additive ladder; overview
# adds a coarse substitutive cap (far view) above the tiled fine branch
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --parts 8 --partition-rule sah
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview --compression-factor 8

# levels — synthesised representative levels (substitutive pyramid)
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels -K 4 -L 3 \
    --subst-method kmeans-lloyd --lloyd-iters 5
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels -K 4 -L 3 --n-lods 4

# volume re-fit — warm-start re-fit each coarse level against the SOURCE volume
# (highest fidelity; each level keeps whichever of merge/re-fit renders closer)
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels \
    --target volume.tiff --refine volume --refine-iters 300
```

An option irrelevant to the chosen recipe (e.g. `--max-elements` with `--recipe stream`) is rejected with a clear error.

#### `luxar gsplat additive`
Give every leaf of an **existing** gsplat tree an additive (streaming) ladder, structure-preservingly — `kind=lod` levels, partition parts, and adaptive groups all keep their shape. The per-leaf counterpart of `lod --recipe stream` (which needs a flat input) and the inverse companion of `flatten`. Explicit `counts:` breakpoints are clamped per leaf; an existing ladder is rebuilt from its union.
```bash
luxar gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200   # ~200ms first paint per leaf
luxar gsplat additive in.gsplats.zarr out.gsplats.zarr -b stream:14000
luxar gsplat additive in.gsplats.zarr out.gsplats.zarr --n-lods 4         # classic equal-count
```

**Options**: `--n-lods` (default 4, equal-count), `--add-method/-m` (auto/greedy/self_energy/mass/amplitude/spectral/random/radial ordering; `radial` reveals outward from the bbox centre and carries no energy stamps), `--breakpoints/-b` (`equal-count` | `stream:C` | explicit `counts:`/`energy:` lists), `--target-ms` (+ `--bandwidth-mbps`, default 25; `--bytes-per-splat` override) to size the first chunk from a download budget, `--encoding/-e`, `--compress/-c`, `--overwrite`.

#### `luxar gsplat flatten`
Collapse **any** gsplat tree (leaf, LOD/matrix tree, partition, nested) into one flat matrix-shaped leaf. Use for compatibility with tools that expect a flat `.gsplats.zarr`, or before rebuilding a new global LOD from a tiled/partitioned result. Flatten retains every default-selected splat but drops additive ladder structure; run `gsplat lod --recipe stream` on the result to build a new global ladder.
```bash
luxar gsplat flatten partitioned.gsplats.zarr flat.gsplats.zarr
```

**Options**: `--encoding/-e precision` (the required default), `--compress/-c` (zip/tar.gz), `--overwrite`. Streaming flatten rejects `auto`/`memory` because their whole-array analysis defeats the memory bound. Precision stores float32 centers and Cholesky factors, so expect a materially larger archive than auto-encoded output; disk staging is created beside the destination and temporarily needs roughly one uncompressed flat payload of free space.

#### `luxar gsplat annotate-quality`
Retrofit Q·e quality stamps onto an **existing** `.gsplats.zarr`, in place (no refit / re-ladder): the cumulative energy fraction `e(k)` per additive sub-LOD plus a reference energy `w` per leaf (cheap O(N); enables the viewer's early energy-threshold LOD upgrades on legacy datasets). The root `content_hash` is re-stamped so viewer caches invalidate automatically. Directory stores only — unpack `.zip`/`.tar.gz` first. New builds stamp by default.
```bash
luxar gsplat annotate-quality splats.gsplats.zarr                 # e(k) + w only (fast)
luxar gsplat annotate-quality splats.gsplats.zarr --with-quality  # + measured Q per level
luxar gsplat annotate-quality splats.gsplats.zarr --dry-run       # print stamps, write nothing
```

**Options**: `--with-quality` (also measure per-level mixture-L2 quality Q vs each lod group's finest content; slower), `--max-pair-splats` (subsample cap for the Q measurement, default 2000000), `--device` (auto/cpu/cuda/mps), `--dry-run`.

#### Tiled Fitting
For large volumes, use tiled fitting. `--tiling auto` (the default) picks the mode automatically: `none` unless BOTH some dimension exceeds `--tile-size` AND the volume has more than 64 M voxels; when it does tile, `content` if a density is supplied (`--cal`/density knobs), else `uniform`. A tiled fit (`--tiling uniform` or `--tiling content`) emits a `kind=partition` by default (one part per tile/box, for viewer frustum culling); pass `--flat` for a single flat leaf. Whole-volume fits (`--tiling none`/small auto) stay a single leaf.

Uniform tiling uses Hann cosine apodization for seamless stitching:
```bash
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32
```

Every producer builds the SAME uniform grid, folding a trailing tile whose unique coverage is thinner than the overlap into its predecessor (#2838) — the sequential path, the `-j N` parent and its workers, this hand-run `--tile k/M` worker, `batch-fit`, and the `fit_tiled` Python API. So `M` here means what the other commands mean by it; the worker prints the grid it resolved, which is the count to compare against. A stored batch manifest from before the change pins the historical unfolded geometry and its tasks are given `--no-fold-tile-slivers` explicitly.

Content-adaptive tiling packs variable-size boxes where the volume is busy, driven by a calibration or density knobs (`--cal`, `--k-star-ref`, `--n-features-ref`, `--saturation-exponent`, `--saturation-cap`, `--feature-threshold`, `--feature-metric`, `--cell`, `--min-leaf`, `--max-leaf`, `--target-features`, `--overlap`, `-j/--jobs`):
```bash
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json -j 8   # parallel content fit
luxar gsplat fit vol.zarr plan.json --tiling content --cal cal.json --plan-only   # emit box plan, no fit
```
Direct content fits warn and ignore `--denoise`, `--progressive`, and the explicit `--downscale` flag; `downscale:` from `--config` remains supported by the content fitter. Batch content planning rejects on-the-fly denoising and progressive fitting instead.

#### `luxar gsplat transform`
Apply spatial and intensity transforms to a Gaussian splat dataset. Multiple transforms can be combined; they are applied in fixed order: scale, rotate, translate, center, scale-intensity, normalize-intensity.
```bash
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --normalize-intensity 1.0
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --translate 0,100,0 --scale-intensity 0.5
```

**Spatial options**: `--scale/-s` (per-axis factors), `--translate/-t` (per-axis offset), `--rotate-x/--rotate-y/--rotate-z` (degrees; rotates the three `--spatial-dims` center dims, default `0,1,2` — the first three, matching the partitioner and the stack-last time/channel convention — other dims are left unrotated), `--spatial-dims i,j,k` (which 3 center dims the rotation acts on, e.g. `1,2,3` for a direct nD fit whose leading axis is time; the listed order assigns the rotation frame's X/Y/Z roles, so `3,2,1` is a different transform from `1,2,3` — unlike `filter`'s order-insensitive `--spatial-dims`), `--center` (center at amplitude-weighted centroid).
**Intensity options**: `--scale-intensity` (multiply amplitudes), `--normalize-intensity` (normalize max amplitude to value).
**Output options**: `--encoding/-e` (auto/precision/memory), `--compress/-c` (zip/tar.gz).

#### `luxar gsplat denoise`
Denoise a volume using Non-Local Means. Auto-calibrates the denoising strength `h` using Noise2Self unless `--h` is provided. Runs locally (no Slurm). For batch denoising on HPC, use `luxar gsplat batch-fit submit --denoise`.
```bash
luxar gsplat denoise volume.zarr denoised.zarr
luxar gsplat denoise volume.zarr denoised.npy --h 0.03
luxar gsplat denoise data.zarr.zip out.zarr --channel 0 --timepoint 5 --denoise-2d
```

**Options**: `--h` (manual NLM h value), `--patch-size` (default 3), `--search-distance` (default 5), `--backend` (auto/cuda/pytorch/skimage), `--device/-d` (auto/cpu/cuda/mps), `--denoise-2d` (slice-by-slice), `--channel`, `--timepoint`, `--array-key`.

#### `luxar gsplat napari`
Open a Gaussian splat dataset in napari for visual inspection. Partition and
nested trees use their default-selected leaves. Renders the splats back to a
volume and displays them alongside splat center points. Requires `napari` to be
installed (`pip install napari[all]`).
```bash
luxar gsplat napari splats.gsplats.zarr
```

#### `luxar gsplat benchmark`
Benchmark GPU performance for Gaussian splatting to determine optimal tile sizes.
```bash
luxar gsplat benchmark --slurm --partition gpu        # Submit benchmark to Slurm
luxar gsplat benchmark --list                         # Show profiled GPUs
```

The `batch-fit` group fits a whole nD dataset at scale (the scaled-up sibling of `gsplat fit`), either **locally across GPUs** (`run`) or on a **Slurm cluster** (`submit`). Both plan the decomposition once (uniform tiles or a shared content box plan over T×C) and then run a memory-safe streaming merge to a single `kind=partition`. `status`/`validate`/`merge`/`cancel` are shared. Content planning rejects on-the-fly `--denoise` and `--progressive` because content-box workers do not implement them; `batch-fit submit --preprocess` is the supported denoising route because it denoises to a store before the boxes fit. Output remains in voxel index space by default. Pass `--physical` to scale fitted centers, covariance, tile/box origins, partition split planes, and quality scoring with the selected NGFF `coordinateTransformations` spatial scale; an explicit `voxel_size` in `--config` wins. Merged stores record scaled spatial axes at scale 1 and retain the NGFF unit only when it supplied the spacing, so converted scenes can label the scale bar honestly. Planning rejects `--physical` when no spacing is available, when the config requests `output_space: voxel`, or when `--merge-refine volume` would crop a physical-space part on the source voxel grid.

#### `luxar gsplat batch-fit run`
Fit a whole timelapse **locally** across multiple GPUs (no Slurm), then merge. Workers are pinned per GPU via `CUDA_VISIBLE_DEVICES`; automatic concurrency accounts for GPU memory and shared host RAM/CPU limits. Resumable — re-running skips tiles already on disk.
```bash
luxar gsplat batch-fit run vol.zarr out/ --gpus all --tile-size 256                  # uniform, every GPU
luxar gsplat batch-fit run vol.zarr out/ --gpus all --tile-size 256 --physical       # NGFF physical coordinates
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --gpus auto # content plan
luxar gsplat batch-fit run vol.zarr out/ --gpus 0,1 --jobs-per-gpu 2 --timepoints ::10
luxar gsplat batch-fit run vol.zarr out/ --gpus auto --merge-recipe stream --merge-n-lods 4  # per-part LOD
luxar gsplat batch-fit run vol.zarr out/ --gpus cpu                                  # CPU fallback
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --dry-run   # plan only
```
`--gpus` SELECTS devices here: `auto` = visible cards above a free-VRAM floor (skips small cards; override `LUXAR_GPU_VRAM_FLOOR_GB`), `all` = every card, `cpu` = CPU, or an explicit list like `0,1,3`. `LUXAR_AUTO_WORKER_HARD_CAP` overrides the automatic host-wide concurrency cap. Not to be confused with `submit --gpus-per-task`, which is a COUNT.

#### `luxar gsplat batch-fit submit`
Plan and submit HPC Slurm fitting jobs for large OME-Zarr datasets. Submits by default; pass `--dry-run` to plan without submitting.
```bash
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu                    # Submit to Slurm
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --dry-run          # Dry-run plan (no submit)
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --preset draft     # Fast preview
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --physical         # NGFF physical coordinates
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --gpus-per-task 2  # 2 GPUs per Slurm task
```
`--gpus-per-task` is a COUNT of GPUs to request for each task, emitted verbatim as `#SBATCH --gpus-per-task`. It is deliberately not spelled `--gpus`: that means the opposite thing one command over, where `batch-fit run --gpus` SELECTS which local devices to use.

#### `luxar gsplat batch-fit status`
Check the status of a batch fitting run (local `run` or Slurm `submit`).
```bash
luxar gsplat batch-fit status output/
```

#### `luxar gsplat batch-fit merge`
Merge completed tiles from a batch fitting run (local `run` or Slurm `submit`) into a single dataset.
```bash
luxar gsplat batch-fit merge output/
```

#### `luxar gsplat batch-fit validate`
Validate integrity of all tiles in a batch output directory (local `run` or Slurm `submit`). Checks each tile for completeness (metadata, arrays, shapes) and reports OK, MISSING, CORRUPT, and STALE_TMP counts. Use `--fix` to delete corrupt tiles and leftover `.tmp` directories so they get re-fitted on the next submit.
```bash
luxar gsplat batch-fit validate output_dir/
luxar gsplat batch-fit validate output_dir/ --fix
```

#### `luxar gsplat batch-fit cancel`
Cancel all Slurm jobs for a Slurm fitting run. Reads the manifest to find job IDs (calibrate, denoise, fit array, merge) and cancels them via `scancel`.
```bash
luxar gsplat batch-fit cancel output_dir/
```

#### `luxar gsplat batch-fit denoise-calibrate` (internal)
Internal command called by the calibration Slurm job. Reads the batch manifest, calibrates NLM `h` per channel, and writes results back to `denoise_h_values.json`. Not intended for direct use.

#### `luxar gsplat batch-fit denoise-preprocess` (internal)
Internal command called by the denoise Slurm array job, one task per (timepoint, channel) pair. Reads calibrated `h` values and denoises a single volume. Not intended for direct use.


## Key Features

- **Browser Integration**: Automatic browser opening for viewer commands
- **Tree View**: Beautiful hierarchical display of zarr structures
- **Port Management**: Automatic port finding when defaults are occupied
- **Network Simulation**: Test viewer performance under various network conditions (9 profiles)
- **CORS Support**: Proper CORS headers for cross-origin access
- **Directory Listing**: JSON/HTML directory listings for zarr exploration

## Architecture

### DirectoryListingStaticFiles
Custom static file handler that provides:
- Zarr-aware directory traversal
- JSON API for programmatic access
- HTML interface for browser navigation
- Support for .zgroup files

### Viewer Integration
- Automatic viewer building if not built
- Concurrent serving of data and viewer
- Smart URL construction with query parameters

### Utilities
- `open_browser()` - Cross-platform browser opening
- `check_port_available()` - Check if a port is available for binding (probes with
  `SO_REUSEADDR` on POSIX, matching uvicorn's bind, so a lingering socket from a
  previous run doesn't read as busy)
- `check_viewer_built()` - Verify viewer dist exists
- `get_viewer_dist_path()` - Get path to viewer distribution directory
- `build_viewer()` - Build viewer using pnpm
- `find_available_port()` - Find free ports for servers (supports end_port shorthand)
- `format_tree_node()` - Format hierarchical displays
- `format_memory_size()` - Format bytes to human-readable string
- `get_zarr_info()` - Extract comprehensive zarr metadata
- `validate_zarr_store()` - Validate that a path is a valid Zarr store

## Testing

The CLI is thoroughly tested with:
- Unit tests for all utility functions
- Integration tests for command workflows
- Mocked server tests to avoid blocking
- Comprehensive test suite with good coverage

## Dependencies

- `typer` - Modern CLI framework
- `uvicorn` - ASGI server for FastAPI
- `fastapi` - Web framework for serving
- `zarr` - Zarr data format support
- `arbol` - Beautiful console output
