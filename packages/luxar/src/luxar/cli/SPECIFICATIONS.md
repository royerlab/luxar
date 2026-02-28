# luxar.cli - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2026-02-28

## Purpose

The `cli` package provides command-line interface for building, serving, and inspecting Luxar Zarr scenes. Built using Typer for argument parsing and FastAPI/uvicorn for HTTP serving.

---

## Commands

### `luxar demo`

**Purpose**: Generate demonstration datasets and optionally serve with viewer

**Parameters**:
- `--output, -o`: Output path for zarr (required if --no-serve)
- `--points, -n`: Number of points (default: 10000)
- `--type, -t`: Demo type (default: "lorenz")
- `--seed, -s`: Random seed for reproducibility
- `--serve/--no-serve`: Whether to serve with viewer (default: True)
- `--open/--no-open`: Open browser automatically (default: True)
- `--port, -p`: Data server port (default: 8000)
- `--viewer-port`: Viewer port (default: 5173)

**Behavior**:
1. If --no-serve: Generate dataset and exit
2. If --serve:
   - Generate dataset (to temp dir if no output path)
   - Build viewer (if not already built)
   - Find available ports (try requested, increment if busy)
   - Start data server in background thread
   - Start viewer server (blocks main thread)
   - Open browser if requested
3. Handle Ctrl+C gracefully

**Demo Types**:
- "lorenz": Lorenz attractor with time-based colors
- More types can be added in utils/demos.py

---

### `luxar serve`

**Purpose**: Serve a directory or Zarr dataset via HTTP

**Parameters**:
- `path`: Directory or Zarr to serve (required unless --viewer-only)
- `--host`: Host address (default: "127.0.0.1")
- `--port, -p`: Port number (default: 8000)
- `--viewer`: Also serve the viewer (default: False)
- `--viewer-port`: Port for viewer (default: 5173)
- `--open, -o`: Open browser (default: False)
- `--viewer-only`: Serve only viewer, no data (default: False)

**Behavior**:
1. Determine what to serve (directory, zarr, or viewer-only)
2. Create FastAPI app with CORS middleware
3. Mount DirectoryListingStaticFiles handler
4. If --viewer: Start viewer in background thread
5. Start data server (blocks)

**Directory Listing**:
- Provides JSON directory listings for zarr browser
- Handles .zgroup files specially
- Returns JSON for API requests, HTML for browsers
- Supports CORS for cross-origin requests

---

### `luxar viewer`

**Purpose**: Serve the Luxar viewer with optional data

**Parameters**:
- `--data, -d`: Zarr data to load (optional)
- `--host`: Host address (default: "127.0.0.1")
- `--port, -p`: Viewer port (default: 5173)
- `--data-port`: Port for data server (default: 8000)
- `--open/--no-open`: Open browser (default: True)

**Behavior**:
1. Check if viewer is built (build if not)
2. If data provided: Start data server in background
3. Start viewer server with data URL parameter
4. Open browser if requested

---

### `luxar info`

**Purpose**: Display detailed information about a Zarr scene

**Parameters**:
- `path`: Path to Zarr store (required)
- `--tree/--no-tree`: Show tree view (default: True)
- `--stats, -s`: Show detailed statistics (default: False)
- `--depth, -d`: Maximum tree depth (default: unlimited)
- `--format`: Output format "text" or "json" (default: "text")

**Behavior**:
1. Open Zarr store in read mode
2. Traverse hierarchy depth-first
3. Collect statistics (n_groups, n_arrays, n_points_total)
4. If --tree: Print hierarchical tree view
5. If --stats: Include per-object details
6. If --format=json: Output structured JSON

**Tree Format**:
```
/
├── group1 (group)
│   └── points1 (points: 1000 points)
└── group2 (group)
    └── points2 (points: 500 points)
```

---

### `luxar export`

**Purpose**: Export a zarr scene and the Luxar viewer into a standalone offline folder. The exported folder contains everything needed to view the scene without any external dependencies beyond Python 3 and a web browser.

**Parameters**:
- `source`: Path to the zarr dataset (required, positional)
- `--output, -o`: Output folder for the standalone export (required)
- `--overwrite`: Overwrite existing output folder (default: False)
- `--open`: Serve and open browser after export (default: False)
- `--port, -p`: Port for local server when using --open (default: 8000)

**Behavior**:
1. Validate the source is a valid zarr store
2. Check that the viewer has been built (error if not)
3. If output exists and --overwrite not set, raise FileExistsError
4. Copy viewer dist files (HTML, JS, CSS, WASM) to `output/viewer/`
5. Rewrite absolute asset paths to relative (for serving from subdirectory)
6. Copy zarr data to `output/data/`
7. Generate `serve.py` (Python 3 stdlib only, no pip install needed)
8. Generate `README.txt` with usage instructions
9. If `--open`: Launch the generated serve.py script

**Output Structure**:
```
output/
  viewer/         # Luxar viewer (HTML, JS, CSS, WASM)
  data/           # Zarr dataset (copied as-is)
  serve.py        # Local HTTP server (Python stdlib only)
  README.txt      # Usage instructions
```

**Example**:
```bash
luxar export my_scene.zarr -o my_export/
luxar export my_scene.zarr -o my_export/ --overwrite
luxar export my_scene.zarr -o my_export/ --open
```

---

### `luxar gsplat`

**Purpose**: Subcommand group for Gaussian splat tools. Provides operations for inspecting, viewing, and pruning `.gsplats.zarr` datasets.

**Subcommands**:
- `luxar gsplat info` - Show detailed dataset statistics
- `luxar gsplat napari` - Open dataset in napari for inspection
- `luxar gsplat view` - Quick view in the Luxar web viewer
- `luxar gsplat prune` - Remove low-impact splats to reduce dataset size

---

### `luxar gsplat info`

**Purpose**: Display comprehensive statistics about a Gaussian splat dataset.

**Parameters**:
- `path`: Path to `.gsplats.zarr` dataset or compressed archive (required, positional)
- `--histograms/--no-histograms`: Show ASCII histograms (default: True)
- `--bins, -b`: Number of bins for histograms (default: 40, range: 10-100)

**Behavior**:
1. Load the `.gsplats.zarr` dataset (supports `.zip` and `.tar.gz`)
2. Display basic info: file size, splat count, dimensionality, color/sharpness presence
3. Display bounding box per dimension with ranges
4. Display amplitude statistics with percentile distribution and cumulative contribution analysis
5. Display volume statistics (3-sigma ellipsoid volumes from Cholesky factors)
6. Display sharpness statistics (identify standard Gaussians with s=2.0)
7. Display color channel statistics (if colors present)
8. Display metadata (fitting info, provenance, format version)
9. Provide pruning recommendation if significant splats are low-contribution

**Example**:
```bash
luxar gsplat info dataset.gsplats.zarr.zip
luxar gsplat info dataset.gsplats.zarr.zip --no-histograms
luxar gsplat info dataset.gsplats.zarr.zip --bins 60
```

---

### `luxar gsplat napari`

**Purpose**: Open a Gaussian splat dataset in napari for interactive 3D inspection.

**Parameters**:
- `path`: Path to `.gsplats.zarr` dataset or compressed archive (required, positional)

**Behavior**:
1. Load the `.gsplats.zarr` dataset
2. Render splats back to a volume (auto-detects best device: cuda/mps/cpu)
3. Open napari with the rendered volume (viridis colormap, additive blending) and splat centers as points (red, semi-transparent)

**Requires**: `napari` must be installed (`pip install napari[all]`).

---

### `luxar gsplat view`

**Purpose**: Quick view of a Gaussian splat dataset in the Luxar web viewer. Converts the dataset to a Luxar scene, serves it, and opens the viewer.

**Parameters**:
- `path`: Path to `.gsplats.zarr` dataset or compressed archive (required, positional)
- `--port, -p`: Data server port (default: 8000)
- `--viewer-port`: Viewer port (default: 5173)
- `--open/--no-open`: Open browser automatically (default: True)

**Behavior**:
1. Build viewer if not already built
2. Load the `.gsplats.zarr` dataset
3. Center data at its centroid for optimal viewing
4. Create a temporary Luxar scene with appropriate dimensions
5. Start data server and viewer server
6. Open browser with the viewer URL

---

### `luxar gsplat prune`

**Purpose**: Remove low-impact splats from a dataset to reduce file size and rendering cost while preserving quality.

**Parameters**:
- `input_path`: Input `.gsplats.zarr` dataset (required, positional)
- `output_path`: Output `.gsplats.zarr` dataset (required, positional)
- `--method, -m`: Pruning strategy (default: "cumulative")
  - `"cumulative"`: Keep top splats contributing X% of total amplitude
  - `"amplitude_percentile"`: Remove bottom X percentile by amplitude
  - `"combined"`: Remove splats with low amplitude OR large volume
- `--retention, -r`: Amplitude retention fraction for cumulative method (default: 0.95, range: 0.0-1.0)
- `--amplitude-percentile, -a`: Bottom percentile to remove (default: 5.0, range: 0-100)
- `--volume-percentile, -v`: Volume percentile threshold for combined method (default: 95.0, range: 0-100)
- `--encoding, -e`: Encoding mode for output: "auto", "precision", or "memory" (default: "auto")
- `--compress, -c`: Compress output as "zip" or "tar.gz" (default: None)
- `--napari, -n`: Open napari comparison viewer after pruning (default: False)

**Behavior**:
1. Load the input dataset with statistics
2. Apply the chosen pruning method
3. Save the pruned dataset with optional compression
4. Report removal statistics (count, percentage, amplitude retention)
5. If `--napari`: Render both original and pruned to volumes and open napari with comparison layers (green=original, magenta=pruned, red=difference)

**Example**:
```bash
# Keep 95% of amplitude (recommended)
luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \
    --method cumulative --retention 0.95

# With compression
luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \
    --method cumulative --retention 0.95 --compress zip

# Compare before/after in napari
luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \
    --method cumulative --retention 0.95 --napari
```

---

## Network Simulation

Luxar CLI provides comprehensive network simulation capabilities for testing viewer performance under various network conditions (3G, 4G, 5G, broadband, satellite, etc.).

**Available Commands**:
- `luxar profiles` - List all 9 network simulation profiles with detailed parameters
- Network parameters available on: `serve`, `viewer`, `demo` commands

**CLI Options** (applicable to serve/viewer/demo):
- `--profile PROFILE` - Apply preset network profile (e.g., '3g', 'broadband', 'satellite')
- `--bandwidth, -b BANDWIDTH` - Limit bandwidth (e.g., '1mbps', '500kbps', '10mbps')
- `--latency, -l LATENCY` - Add network latency (e.g., '100ms', '500ms')
- `--jitter, -j JITTER` - Add latency jitter (e.g., '10%', '0.1')
- `--packet-loss LOSS` - Simulate packet loss (e.g., '1%', '0.01')

**Example Usage**:
```bash
# Test with 3G mobile network conditions
luxar serve data.zarr --profile 3g --viewer --open

# Custom worst-case scenario
luxar demo --bandwidth 100kbps --latency 500ms --jitter 50% --packet-loss 5%
```

**Implementation**:
- Middleware: `luxar/cli/network_simulation.py` (NetworkSimulationMiddleware)
- Integration: Automatically applied to serve/viewer/demo commands when network options specified

**Design Decision - Pure ASGI Middleware**:

The implementation uses a pure ASGI wrapper approach (wrapping the complete FastAPI app before passing to uvicorn) rather than alternatives:

- ✅ **Chosen**: Pure ASGI middleware wrapper
  - Full control over ASGI message flow (can intercept receive/send)
  - Enables true packet loss simulation (can drop requests entirely)
  - Works with streaming responses
  - Clean separation from FastAPI app configuration
  - Easy to conditionally enable/disable

- ❌ **Rejected**: Starlette BaseHTTPMiddleware
  - Limited control (request-response pattern only)
  - Cannot implement true packet loss (can't drop requests cleanly)
  - Less flexible for low-level simulation

- ❌ **Rejected**: FastAPI @middleware decorator
  - Request-response pattern, not suitable for packet loss
  - Less control over streaming behavior

**Integration Point**: Middleware wraps the complete app after all routes/middleware configured, applied only to data server (not viewer HTML server).

**Detailed Documentation**: See `docs/guides/developer/NETWORK_SIMULATION_SPEC.md` for complete specification including algorithms, profiles, and testing guidelines.

---

## HTTP Server Specification

### DirectoryListingStaticFiles

**Purpose**: Serve static files with JSON directory listing support

**Key Behaviors**:
1. Serve files normally for file requests
2. Provide directory listings for directory requests
3. Return JSON if Accept: application/json header
4. Return HTML otherwise (for browser)
5. Handle CORS OPTIONS requests
6. Special handling for .zgroup files (zarr metadata)

**Directory Listing Format** (JSON):
```json
{
  "entries": [
    {"name": "file.txt", "type": "file", "size": 1234},
    {"name": "subdir", "type": "directory", "size": null},
    {"name": "data.zarr", "type": "zarr", "size": null}
  ]
}
```

### CORS Configuration
- Allow all origins (development use)
- Allow all methods (GET, POST, OPTIONS, etc.)
- Allow all headers
- Allow credentials

---

## Utility Functions

### Port Management
- `check_port_available(port, host="127.0.0.1")` - Test if port is free
- `find_available_port(start_port, max_attempts=100)` - Find next available port

**Algorithm**:
```
Try port, port+1, port+2, ... up to max_attempts
For each port:
  Try to bind socket
  If successful: return port
  If fails: try next
If all fail: return None
```

### Viewer Management
- `check_viewer_built()` - Check if dist/ exists with index.html
- `build_viewer()` - Run pnpm build in luxar-viewer directory
- `get_viewer_dist_path()` - Get path to built viewer
- `open_browser(url)` - Open URL in default browser

### Zarr Information
- `get_zarr_info(path)` - Extract comprehensive stats from Zarr store

**Returns**:
```python
{
  'size': total_bytes,
  'n_groups': count,
  'n_arrays': count,
  'n_points_total': sum,
  'points_objects': [
    {
      'path': str,
      'n_points': int,
      'n_dims': int,
      'has_colors': bool,
      'has_radii': bool,
      'has_sharpness': bool
    },
    ...
  ]
}
```

### Formatting
- `format_memory_size(bytes)` - Human-readable (e.g., "1.5 MB", "3.2 GB")
- `format_tree_node(name, depth, is_last, prefix, type, attrs)` - Tree visualization

---

## Server Lifecycle

### Demo Command with Serve:
```
1. Generate demo → temp directory
2. Build viewer (if needed)
3. Find available ports
4. Start data server (background thread, daemon)
5. Wait 1 second (let server start)
6. Start viewer server (blocks main thread)
7. On Ctrl+C: Cleanup temp directory
```

### Multi-Server Management:
- Data server runs in daemon thread (dies with main)
- Viewer server runs in main thread (keeps process alive)
- Browser opens after 1-second delay (ensure servers ready)
- All servers log at WARNING level (reduce noise)

---

## Error Handling

**Patterns**:
- File not found → Exit code 1 with clear message
- Port unavailable → Try alternatives or report failure
- Build failed → Suggest manual build command
- Viewer not built → Offer to build automatically
- Invalid command → Show help

**Exit Codes**:
- 0: Success
- 1: Error (file not found, invalid args, etc.)

---

## Dependencies

**External**:
- `typer`: Command-line argument parsing
- `uvicorn`: ASGI server
- `fastapi`: Web framework
- `zarr`: Reading Zarr stores for info command
- `arbol`: Structured logging

**Internal**:
- `luxar.utils.demos`: Demo generation
- `luxar.io`: Zarr operations

---

## This specification provides sufficient detail to re-implement the CLI with equivalent user experience and behavior.

---

## Changelog

### v1.2.0 (2026-02-28)
- Added `luxar export` command documentation (standalone offline scene export)
- Added `luxar gsplat` subcommand group documentation
- Added `luxar gsplat info` command documentation (dataset statistics)
- Added `luxar gsplat napari` command documentation (napari viewer)
- Added `luxar gsplat view` command documentation (quick web viewer)
- Added `luxar gsplat prune` command documentation (dataset pruning)

### v1.1.0 (2025-12-11)
- Added network simulation documentation (Section "Network Simulation")
- Documented `luxar profiles` command
- Documented network simulation options for serve/viewer/demo commands (--profile, --bandwidth, --latency, --jitter, --packet-loss)
- Cross-referenced NETWORK_SIMULATION_SPEC.md for detailed specification
- Note: Network simulation feature was implemented 2025-12-06 (commit 87c078e)

### v1.0.0 (2025-11-27)
- Documented demo, serve, viewer, info commands
- Specified HTTP serving architecture
- Defined exit codes and dependencies
