# Luxar CLI Package

This package contains the command-line interface (CLI) for Luxar, providing tools for serving, viewing, and inspecting Luxar zarr datasets.

## Module Structure

- `__init__.py` - Package initialization, exports the main app
- `main.py` - Main CLI application with all commands
- `utils.py` - Utility functions for CLI operations

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


## Key Features

- **Browser Integration**: Automatic browser opening for viewer commands
- **Tree View**: Beautiful hierarchical display of zarr structures
- **Port Management**: Automatic port finding when defaults are occupied
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
- 40+ tests with 75% coverage

## Dependencies

- `typer` - Modern CLI framework
- `uvicorn` - ASGI server for FastAPI
- `fastapi` - Web framework for serving
- `zarr` - Zarr data format support
- `arbol` - Beautiful console output