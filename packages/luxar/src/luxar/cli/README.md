# Luxar CLI Package

This package contains the command-line interface (CLI) for Luxar, providing tools for serving, viewing, and inspecting Luxar zarr datasets.

## Quick Start

Essential CLI commands in 3 steps:

```bash
# 1. Generate and view a demo (fastest way to see Luxar)
luxar demo

# 2. Serve your own dataset with the viewer
luxar serve my_data.zarr --viewer

# 3. Get dataset information and statistics
luxar info my_data.zarr --stats
```

**What Each Does**:
- `luxar demo` - Creates a demo dataset and opens it in the viewer automatically
- `luxar serve --viewer` - Serves your data via HTTP and launches the viewer
- `luxar info --stats` - Shows dataset structure, dimensions, and compression stats

**Pro Tips**:
- Add `--no-open` to any command to skip browser launch
- Use `luxar profiles` to list network simulation profiles
- Use `luxar serve --help` for all serving options

## Module Structure

- `__init__.py` - Package initialization, exports the main app
- `main.py` - Main CLI application with all commands
- `gsplat_commands.py` - Gaussian splat subcommands (info, view, prune, fit, convert, render, merge, filter, split, slice, compare, transform, denoise, napari, batch validate/cancel/denoise-calibrate/denoise-preprocess)
- `gsplat_config.py` - Config system: presets, YAML loading, volume loaders, helpers
- `utils.py` - Utility functions for CLI operations
- `network_simulation.py` - Network simulation middleware and profile definitions

## Available Commands

### `luxar demo`
Quick demo generation with automatic viewer launch.
```bash
luxar demo                    # Generate demo and open in browser
luxar demo --points 10000     # Custom point count
luxar demo --no-open          # Don't open browser
luxar demo --no-serve --output demo.zarr  # Generate demo without serving
luxar demo --no-serve --output demo.zarr --points 100000  # Custom point count, no serve
```

### `luxar serve`
Serve zarr datasets or directories via HTTP.
```bash
luxar serve data.zarr         # Serve data
luxar serve data.zarr --viewer # Serve with viewer
luxar serve --viewer-only      # Serve only viewer
```

### `luxar viewer`
Serve the Luxar viewer with optional data.
```bash
luxar viewer                  # Serve viewer
luxar viewer --data data.zarr # Serve viewer with data
luxar viewer --no-open        # Don't open browser
```

### `luxar info`
Display detailed information about zarr datasets.
```bash
luxar info data.zarr          # Basic info with tree view
luxar info data.zarr --stats  # Include detailed statistics
luxar info data.zarr --format json # JSON output
```

### `luxar profiles`
List available network simulation profiles for testing.
```bash
luxar profiles                # Display all network profiles with descriptions
```

**Available profiles:** 3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested

Use these profiles with `serve`, `viewer`, or `demo` commands via the `--network-profile` option to simulate various network conditions for testing.


### `luxar export`
Export a zarr scene and the Luxar viewer as a standalone offline folder. The output is self-contained: anyone can view the scene with just Python 3 and a browser by running `serve.py`.
```bash
luxar export my_scene.zarr -o my_export/              # Export scene + viewer
luxar export my_scene.zarr -o my_export/ --overwrite   # Overwrite existing export
luxar export my_scene.zarr -o my_export/ --open        # Export and serve in browser
luxar export my_scene.zarr -o my_export/ --open --port 9000  # Custom port
```

**Options**: `--output/-o` (required), `--overwrite`, `--open` (serve and launch browser), `--port/-p` (port for local server, default 8000).


### GSplat Processing Commands

#### `luxar gsplat fit`
Fit Gaussian splats to a volume with preset or YAML config.
```bash
luxar gsplat fit volume.npy splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit volume.tiff splats.gsplats.zarr --config params.yaml
luxar gsplat fit --dump-config --preset hifi > config.yaml  # Generate config template
```

**Presets:** `draft` (fast preview), `standard` (balanced), `hifi` (max quality)

#### `luxar gsplat convert`
Convert .gsplats.zarr to a Luxar scene for the web viewer.
```bash
luxar gsplat convert fitted.gsplats.zarr scene.zarr --center
luxar gsplat convert fitted.gsplats.zarr scene.zarr --scale-intensity 0.1
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

#### `luxar gsplat filter`
Filter splats by multiple criteria (AND logic).
```bash
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100
luxar gsplat filter input.gsplats.zarr out.gsplats.zarr --mass-min 0.01
```

**Criteria**: `--bbox`, `--amplitude-min/max`, `--volume-min/max`, `--eccentricity-min/max`, `--mass-min/max`, `--sigma-axis`/`--sigma-min/max`. Supports `--*-normalized` flags.

#### `luxar gsplat split`
Split a dataset into multiple parts.
```bash
luxar gsplat split input.gsplats.zarr output_dir/ --parts 4
luxar gsplat split input.gsplats.zarr output_dir/ --indices "100,500"
luxar gsplat split input.gsplats.zarr output_dir/ --parts 3 --compress zip
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

#### Tiled Fitting
For large volumes, use tiled fitting with Hann cosine apodization:
```bash
luxar gsplat fit large.zarr splats.gsplats.zarr --tiled --tile-size 256 --overlap 32
luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32
```

#### `luxar gsplat transform`
Apply spatial and intensity transforms to a Gaussian splat dataset. Multiple transforms can be combined; they are applied in fixed order: scale, rotate, translate, center, scale-intensity, normalize-intensity.
```bash
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --normalize-intensity 1.0
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --translate 0,100,0 --scale-intensity 0.5
```

**Spatial options**: `--scale/-s` (per-axis factors), `--translate/-t` (per-axis offset), `--rotate-x/--rotate-y/--rotate-z` (degrees, 3D only), `--center` (center at amplitude-weighted centroid).
**Intensity options**: `--scale-intensity` (multiply amplitudes), `--normalize-intensity` (normalize max amplitude to value).
**Output options**: `--encoding/-e` (auto/precision/memory), `--compress/-c` (zip/tar.gz).

#### `luxar gsplat denoise`
Denoise a volume using Non-Local Means. Auto-calibrates the denoising strength `h` using Noise2Self unless `--h` is provided. Runs locally (no Slurm). For batch denoising on HPC, use `luxar gsplat batch plan --denoise`.
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

#### `luxar gsplat batch validate`
Validate integrity of all tiles in a batch output directory. Checks each tile for completeness (metadata, arrays, shapes) and reports OK, MISSING, CORRUPT, and STALE_TMP counts. Use `--fix` to delete corrupt tiles and leftover `.tmp` directories so they get re-fitted on the next submit.
```bash
luxar gsplat batch validate output_dir/
luxar gsplat batch validate output_dir/ --fix
```

#### `luxar gsplat batch cancel`
Cancel all Slurm jobs for a batch fitting run. Reads the manifest to find job IDs (calibrate, denoise, fit array, merge) and cancels them via `scancel`.
```bash
luxar gsplat batch cancel output_dir/
```

#### `luxar gsplat batch denoise-calibrate` (internal)
Internal command called by the calibration Slurm job. Reads the batch manifest, calibrates NLM `h` per channel, and writes results back to `denoise_h_values.json`. Not intended for direct use.

#### `luxar gsplat batch denoise-preprocess` (internal)
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
- `check_viewer_built()` - Verify viewer dist exists
- `build_viewer()` - Build viewer using pnpm
- `find_available_port()` - Find free ports for servers (supports end_port shorthand)
- `format_tree_node()` - Format hierarchical displays
- `get_zarr_info()` - Extract comprehensive zarr metadata

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
