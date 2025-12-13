# luxar.cli - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2025-12-11

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

**Detailed Documentation**: See `docs/NETWORK_SIMULATION_SPEC.md` for complete specification including algorithms, profiles, and testing guidelines.

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
- `check_port_available(port)` - Test if port is free
- `find_available_port(start_port, max_tries=100)` - Find next available port

**Algorithm**:
```
Try port, port+1, port+2, ... up to max_tries
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

### v1.1.0 (2025-12-11)
- Added network simulation documentation (Section "Network Simulation")
- Documented `luxar profiles` command
- Documented network simulation options for serve/viewer/demo commands (--profile, --bandwidth, --latency, --jitter, --packet-loss)
- Cross-referenced NETWORK_SIMULATION_SPEC.md for detailed specification
- Note: Network simulation feature was implemented 2025-12-06 (commit 87c078e)



- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented demo, serve, viewer, info commands
  - Specified HTTP serving architecture
  - Defined exit codes and dependencies
