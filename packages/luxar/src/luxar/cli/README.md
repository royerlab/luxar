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

**What Each Does**:
- `luxar demo` - Lists the bundled demos; `luxar demo run <key>` generates one and opens it in the viewer
- `luxar serve --viewer` - Serves your data via HTTP alongside the viewer (add `--open` to launch the browser)
- `luxar info --stats` - Shows dataset structure, dimensions, and compression stats

**Pro Tips**:
- Browser launch: `demo`, `viewer`, and `gsplat view` open the browser by default (add `--no-open` to skip); `serve` and `export` do not (add `--open` to launch it)
- Use `luxar profiles` to list network simulation profiles
- Use `luxar serve --help` for all serving options
- Use `luxar export --native macos` (or `linux-amd64`/`linux-arm64`) for double-clickable native bundles backed by an embedded Go launcher (requires `make build-launchers` first)
- For complete working scripts that generate datasets for these commands, see `packages/luxar/examples/` and repository-level `examples/` when present.

## Module Structure

- `__init__.py` - Package initialization, exports the main app
- `main.py` - Main CLI application with the top-level commands
- `serving.py` - HTTP serving internals (`create_server_app`, data/viewer servers; re-exported by `main.py`)
- `info_command.py` - The `luxar info` command implementation
- `gsplat_commands.py` - Thin registration hub (~56 lines) that assembles the `gsplat` sub-app: fit, cal, render, denoise, lod, convert, migrate-format, reencode, info, napari, view, compare, annotate-quality, transform, merge, cull, filter, slice, partition, flatten, additive, benchmark; the `batch-fit` group: run/submit/status/validate/cancel/merge/denoise-calibrate/denoise-preprocess
- `gsplat_ops/` - The gsplat subcommand implementations (~35 modules: fitting, calibration, transforms, batch orchestration, inspection, ...)
- `lod.py` - the unified `lod --recipe {flat,stream,levels,tiles,overview,adaptive}` command (thin wrapper over `gsplats/lod/recipes.py`; registered onto the `gsplat` app)
- `gsplat_config.py` - Config system: presets, YAML loading, volume loaders, helpers
- `utils.py` - Utility functions for CLI operations
- `export.py` - Standalone scene export (viewer + data + serve script)
- `native_app.py` - Native bundle producers (macOS `.app`, Linux portable folder) for `luxar export --native`
- `network_simulation.py` - Network simulation middleware and profile definitions
- `_launchers/` - Go-compiled launcher binaries (populated by `make build-launchers`; ride along in wheel builds)
- `_launcher_assets/` - Bundle icons (`luxar-logo.png`, `AppIcon.icns`) used by `native_app.py`

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
luxar demo cache list             # Inventory ~/.cache/luxar demo caches
luxar demo cache clear lorenz --dry-run   # Preview a cache clear
luxar demo deps                   # Which optional demo deps are missing?
luxar demo deps --install         # Install the extras that provide them
luxar demo deps --extra io        # Restrict to one extra (demos / io / gsplats)
```
`deps` exits 1 when anything is missing, so it doubles as a CI/setup gate. It
reports the *constrained* requirement from `luxar.demos.INSTALL_SPECS` — the same
table the runtime `require_module` gate uses — and surveys with `find_spec`, so
it never imports `torch` just to tell you `torch` is present.

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

**Runtime fallback**: setting `LUXAR_LAUNCHER_NO_WEBVIEW=1` makes the launcher open the user's default browser instead of an embedded WebView — useful for headless smoke tests and minimal Linux installs without `libwebkit2gtk`.

See `packages/luxar-launcher/README.md` for the launcher source itself.


### GSplat Processing Commands

#### `luxar gsplat info`
Print dataset statistics for a `.gsplats.zarr` (splat count, dimensions, bounds, LOD structure).
```bash
luxar gsplat info splats.gsplats.zarr
```

#### `luxar gsplat view`
Open a `.gsplats.zarr` in the web viewer. The standalone `.gsplats.zarr` is a v3.3 node subtree — exactly what the viewer renders inside a scene — so it is served **directly** via `?src=` with no scene-compile round-trip (works for a single leaf, an additive ladder, a `kind=lod` hierarchy, a `kind=partition` split, and arbitrary nestings; archives are extracted first).
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

`--floor` (default `auto`) subtracts a background pedestal (clip at 0) before normalization, so output amplitudes are background-relative. `auto` = histogram-mode estimate (a no-op on clean data); `pNN` subtracts that percentile, a plain number a fixed value, `none` disables it (legacy hard-min).

#### `luxar gsplat convert`
Convert .gsplats.zarr to a Luxar scene for the web viewer.
```bash
luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --center
luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --scale-intensity 0.1
```

#### `luxar gsplat render`
Render gsplats back to a volume for quality comparison.
```bash
luxar gsplat render fitted.gsplats.zarr rendered.npy --shape 128,128,128
luxar gsplat render fitted.gsplats.zarr rendered.tiff --device cuda
```

#### `luxar gsplat merge`
Combine multiple gsplat datasets (concatenation, new dimension, or channel colors).
```bash
luxar gsplat merge a.zarr b.zarr -o merged.zarr
luxar gsplat merge t0.zarr t1.zarr t2.zarr -o 4d.zarr --as-dimension --values 0,1,2
luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr --channel-colors "#ff0080,#00ff00"
```

#### `luxar gsplat cull`
Remove low-contribution splats to reduce dataset size while preserving visual quality.
```bash
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr                            # Auto (cumulative, keep 95%)
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90      # Keep 90% amplitude
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m redundancy --shape 41,512,512  # GPU, no target
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy           # Error-budget (most principled)
```

#### `luxar gsplat filter`
Filter splats by multiple criteria (AND logic).
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
(`--rule median|midpoint|sah`).
```bash
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --parts 4
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --max-elements 100000
luxar gsplat partition input.gsplats.zarr part.gsplats.zarr --parts 3 --rule sah --compress zip
```

#### `luxar gsplat slice`
Slice splats by coordinate ranges (numpy-style syntax).
```bash
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "0:50, :, 10:90"
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr ":50, 20:80, :"
```

#### `luxar gsplat compare`
Compare reconstruction quality against a reference volume (PSNR, SSIM, MSE).
```bash
luxar gsplat compare fitted.gsplats.zarr original.tiff
luxar gsplat compare fitted.gsplats.zarr original.npy --device cuda --output-json metrics.json
```

#### `luxar gsplat cal`
Calibrate the splat count `K` via blind-spot cross-validation. Sweeps `K`, identifies the held-out PSNR peak (`K*`) using the manuscript's Noise2Self protocol (5% donut-median masking), and reports the noise floor. Defaults to the `n2s` fit preset so the held-out curve has enough optimiser budget to reach the overfit regime.
```bash
luxar gsplat cal volume.tiff cal.json                            # 10-point sweep, [1K, 512K]
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000  # Faster sweep
luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'  # Explicit
luxar gsplat cal volume.tiff cal.json --pdf cal_report.pdf       # Multi-page PDF report
luxar gsplat cal volume.zarr cal.json --progression power --power 2  # Polynomial K spacing
```

**Options**: `--k-grid` (explicit comma-separated K values), `--n-grid` (default 10), `--k-min` (default 1000), `--k-max` (default 512000), `--progression` (exp/power), `--power`, `--mask-seed`, `--mask-fraction` (default 0.05), `--preset` (default n2s), `--config`, `--floor` (background floor / DC-offset suppression, default `auto` — K* is measured on floor-suppressed data, matching `fit`; pass `none` for the legacy hard-min behavior), `--device/-d`, `--k-star-metric` (psnr_minmax/psnr_foreground/gain), `--auto-region/--no-auto-region` + `--region-size`/`--region-strategy` (calibrate on a content-rich sub-region), `--feature-metric`, `--saturation-exponent`, `--rd-model/--no-rd-model`, `--fit-exponent`/`--exponent-scales` (measure the saturation exponent alpha instead of assuming the default), `--pdf` (multi-page PDF report), `--keep-fits` (persist per-K fits), `--quiet/-q`, plus volume-loader pass-through (`--channel/-c`, `--timepoint`, `--array-key`, `--axes`).

#### `luxar gsplat migrate-format`
Convert a legacy `.gsplats.zarr` layout to the current node-tree format. Five input shapes are auto-detected: v1.0 (single flat splat set), v1.1 (multi-LOD additive `/splats/lod_<i>/` subgroups), a pre-v2.0 substitutive directory (`manifest.json` + `level_<i>.gsplats.zarr`), the v2.0 `substitutive_<s>/additive_<a>/` matrix, and a v3.0/v3.1 store whose `kind=lod` groups still carry the pre-v3.2 `pixel_size` selector attrs (rewritten as `selector: "coverage"` + derived per-child `coverage_fraction`). All migrate to a single current-format (v3.3) `.gsplats.zarr`. By default the output adopts the AUTO encoding policy, so legacy float32 Cholesky factors are re-encoded as the split diagonal/off-diagonal arrays with certified uint8 per-column quantization (the encode-time covariance certificate escalates to uint16 when the measured Σ error demands it); pass `--lossless` to keep them float32 for archival fidelity.
```bash
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr             # single file (AUTO encoding)
luxar gsplat migrate-format old_pyr/ v3.gsplats.zarr                        # substitutive directory
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr --lossless  # preserve float32 Cholesky exactly
```

**Options**: `--overwrite`, `--lossless` (preserve float32 Cholesky / PRECISION encoding), `--quiet/-q`.

#### `luxar gsplat reencode`
Re-quantize a **current-format** `.gsplats.zarr`'s Cholesky encoding (writes a re-quantized copy to a new path) — a structure-preserving round-trip: the whole node tree (leaf / additive ladder / `kind=lod` / partition / nested) and its `fitting` / `provenance` / `pipeline` groups carry over verbatim; only the on-disk Cholesky encoding changes. Splat count and geometry are unchanged and decode is always to float32, so viewer/GPU/WASM paths are unaffected. Unlike `migrate-format` (legacy → current, exposing only float32 vs the AUTO uint16 default via `--lossless`), this exposes the full ladder — including `memory` (uint8) — and works on already-current files. The clean way to change quantization after fitting.
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
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --method self_energy

# tiles / overview — BSP parts each with an additive ladder; overview
# adds a coarse substitutive cap (far view) above the tiled fine branch
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --parts 8 --partition-rule sah
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview --compression-factor 8

# levels — synthesised representative levels (substitutive pyramid)
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels -K 4 -L 3 \
    --substitutive-method kmeans-lloyd --lloyd-iters 5
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

**Options**: `--n-lods` (default 4, equal-count), `--method/-m` (auto/greedy/self_energy/mass/amplitude/spectral/random ordering), `--breakpoints/-b` (`equal-count` | `stream:C` | explicit `counts:`/`energy:` lists), `--target-ms` (+ `--bandwidth-mbps`, default 25; `--bytes-per-splat` override) to size the first chunk from a download budget, `--encoding/-e`, `--compress/-c`, `--overwrite`.

#### `luxar gsplat flatten`
Collapse **any** gsplat tree (leaf, LOD/matrix tree, partition, nested) into one flat matrix-shaped leaf. Use for compatibility with tools that expect a flat `.gsplats.zarr`, or before rebuilding a new global LOD from a tiled/partitioned result.
```bash
luxar gsplat flatten partitioned.gsplats.zarr flat.gsplats.zarr
```

**Options**: `--encoding/-e` (auto/precision/memory), `--compress/-c` (zip/tar.gz), `--overwrite`.

#### `luxar gsplat annotate-quality`
Retrofit Q·e quality stamps onto an **existing** `.gsplats.zarr`, in place (no refit / re-ladder): the cumulative energy fraction `e(k)` per additive sub-LOD plus a reference energy `w` per leaf (cheap O(N); enables the viewer's early energy-threshold LOD upgrades on legacy datasets). The root `content_hash` is re-stamped so viewer caches invalidate automatically. Directory stores only — unpack `.zip`/`.tar.gz` first. New builds stamp by default.
```bash
luxar gsplat annotate-quality splats.gsplats.zarr                 # e(k) + w only (fast)
luxar gsplat annotate-quality splats.gsplats.zarr --with-quality  # + measured Q per level
luxar gsplat annotate-quality splats.gsplats.zarr --dry-run       # print stamps, write nothing
```

**Options**: `--with-quality` (also measure per-level mixture-L2 quality Q vs each lod group's finest content; slower), `--max-pair-splats` (subsample cap for the Q measurement, default 2000000), `--device` (auto/cpu/cuda/mps), `--dry-run`.

#### Tiled Fitting
For large volumes, use tiled fitting. `--tiling auto` (the default) picks the mode automatically: `none` if the volume fits one tile, `content` if a density is supplied (`--cal`/density knobs), else `uniform`. A tiled fit (`--tiling uniform` or `--tiling content`) emits a `kind=partition` by default (one part per tile/box, for viewer frustum culling); pass `--flat` for a single flat leaf. Whole-volume fits (`--tiling none`/small auto) stay a single leaf.

Uniform tiling uses Hann cosine apodization for seamless stitching:
```bash
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32
```

Content-adaptive tiling packs variable-size boxes where the volume is busy, driven by a calibration or density knobs (`--cal`, `--k-star-ref`, `--n-features-ref`, `--saturation-exponent`, `--saturation-cap`, `--feature-threshold`, `--feature-metric`, `--cell`, `--min-leaf`, `--max-leaf`, `--target-features`, `--overlap`, `-j/--jobs`):
```bash
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json -j 8   # parallel content fit
luxar gsplat fit vol.zarr plan.json --tiling content --cal cal.json --plan-only   # emit box plan, no fit
```

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
Open a Gaussian splat dataset in napari for visual inspection. Renders the splats back to a volume and displays them alongside splat center points. Requires `napari` to be installed (`pip install napari[all]`).
```bash
luxar gsplat napari splats.gsplats.zarr
```

#### `luxar gsplat benchmark`
Benchmark GPU performance for Gaussian splatting to determine optimal tile sizes.
```bash
luxar gsplat benchmark --slurm --partition gpu        # Submit benchmark to Slurm
luxar gsplat benchmark --list                         # Show profiled GPUs
```

The `batch-fit` group fits a whole nD dataset at scale (the scaled-up sibling of `gsplat fit`), either **locally across GPUs** (`run`) or on a **Slurm cluster** (`submit`). Both plan the decomposition once (uniform tiles or a shared content box plan over T×C) and then run a memory-safe streaming merge to a single `kind=partition`. `status`/`validate`/`merge`/`cancel` are shared.

#### `luxar gsplat batch-fit run`
Fit a whole timelapse **locally** across multiple GPUs (no Slurm), then merge. One worker is pinned per GPU via `CUDA_VISIBLE_DEVICES`; per-GPU concurrency is sized from each card's free VRAM. Resumable — re-running skips tiles already on disk.
```bash
luxar gsplat batch-fit run vol.zarr out/ --gpus all --tile-size 256                  # uniform, every GPU
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --gpus auto # content plan
luxar gsplat batch-fit run vol.zarr out/ --gpus 0,1 --jobs-per-gpu 2 --timepoints ::10
luxar gsplat batch-fit run vol.zarr out/ --gpus auto --merge-recipe stream --merge-n-lods 4  # per-part LOD
luxar gsplat batch-fit run vol.zarr out/ --gpus cpu                                  # CPU fallback
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --dry-run   # plan only
```
`--gpus`: `auto` = visible cards above a free-VRAM floor (skips small cards; override `LUXAR_GPU_VRAM_FLOOR_GB`), `all` = every card, `cpu` = CPU, or an explicit list like `0,1,3`.

#### `luxar gsplat batch-fit submit`
Plan and submit HPC Slurm fitting jobs for large OME-Zarr datasets. Submits by default; pass `--dry-run` to plan without submitting.
```bash
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu                    # Submit to Slurm
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --dry-run          # Dry-run plan (no submit)
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --preset draft     # Fast preview
```

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
- `check_port_available()` - Check if a port is available for binding
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
