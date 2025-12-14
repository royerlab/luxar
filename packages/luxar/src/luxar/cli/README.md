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
- `find_available_port()` - Find free ports for servers
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