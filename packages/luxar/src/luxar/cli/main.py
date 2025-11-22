"""luxar.cli – Command-line interface for building, serving, and inspecting Luxar Zarr scenes."""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, MutableMapping, Optional, Tuple

import typer
import uvicorn
import zarr
from arbol import aprint, asection
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .utils import (
    build_viewer,
    check_viewer_built,
    find_available_port,
    format_memory_size,
    format_tree_node,
    get_viewer_dist_path,
    get_zarr_info,
)
from .utils import (
    open_browser as open_browser_func,
)


class DirectoryListingStaticFiles(StaticFiles):
    """Static files handler with JSON directory listing support."""

    async def get_response(self, path: str, scope: MutableMapping[str, Any]) -> Any:
        """Override to provide directory listing."""
        # Handle OPTIONS requests for CORS
        if scope.get("method") == "OPTIONS":
            from starlette.responses import Response

            return Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )

        if self.directory is None:
            raise ValueError("Directory not set")
        full_path = Path(self.directory) / path if path else Path(self.directory)

        # Handle .zgroup files for zarr directories
        if path.endswith(".zgroup") and full_path.exists() and full_path.is_file():
            from starlette.responses import FileResponse

            return FileResponse(full_path)

        # If it's a directory, provide listing
        if full_path.exists() and full_path.is_dir():
            # Check Accept header
            headers = dict(scope.get("headers", []))
            accept = headers.get(b"accept", b"").decode("utf-8")

            # Generate directory listing
            entries = []
            try:
                for item in sorted(full_path.iterdir()):
                    # Skip hidden files except .zgroup
                    if item.name.startswith(".") and item.name != ".zgroup":
                        continue

                    item_type = "directory" if item.is_dir() else "file"
                    # Check if it's a zarr directory
                    if item.is_dir() and item.name.endswith(".zarr"):
                        item_type = "zarr"
                    elif item.is_dir() and (item / ".zgroup").exists():
                        item_type = "zarr"

                    entries.append(
                        {
                            "name": item.name,
                            "type": item_type,
                            "size": item.stat().st_size if item.is_file() else None,
                        }
                    )
            except PermissionError:
                from starlette.responses import Response

                return Response("Permission denied", status_code=403)

            # Return JSON for API requests
            if "application/json" in accept:
                from starlette.responses import JSONResponse

                return JSONResponse({"entries": entries})

            # Return HTML for browser requests
            from starlette.responses import HTMLResponse

            html_content = "<html><body><h1>Directory Listing</h1><ul>"
            if path:
                html_content += '<li><a href="../">../</a></li>'
            for entry in entries:
                name = str(entry["name"])
                if entry["type"] in ("directory", "zarr"):
                    name += "/"
                html_content += f'<li><a href="{name}">{name}</a></li>'
            html_content += "</ul></body></html>"
            return HTMLResponse(content=html_content)

        # Fall back to default static file serving
        return await super().get_response(path, scope)


app = typer.Typer(help="luxar – build and serve Zarr-backed 3-D scenes")


# ────────────────────────────── serve ────────────────────────────────────────
@app.command()
def serve(
    path: Optional[Path] = typer.Argument(None, exists=True, readable=True),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address to bind to (use 0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(8000, "--port", "-p"),
    viewer: bool = typer.Option(False, "--viewer", help="Also serve the viewer"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Port for viewer"),
    open_browser: bool = typer.Option(False, "--open", "-o", help="Open browser"),
    viewer_only: bool = typer.Option(
        False, "--viewer-only", help="Serve only the viewer"
    ),
) -> None:
    """Serve a directory, Zarr dataset, or viewer via HTTP.

    Args:
        path (Path, optional): Path to directory or Zarr dataset to serve.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 8000.
        viewer (bool, optional): Also serve the viewer. Defaults to False.
        viewer_port (int, optional): Port for viewer. Defaults to 5173.
        open_browser (bool, optional): Open browser. Defaults to False.
        viewer_only (bool, optional): Serve only the viewer. Defaults to False.
    """
    try:
        # Handle viewer-only mode
        if viewer_only:
            if not check_viewer_built():
                aprint(
                    "❌ Viewer not built. Run: cd packages/luxar-viewer && pnpm build"
                )
                raise typer.Exit(1)

            # Find available port
            actual_viewer_port = find_available_port(viewer_port)
            if actual_viewer_port is None:
                aprint(f"❌ Error: No available ports found near {viewer_port}")
                raise typer.Exit(1)
            if actual_viewer_port != viewer_port:
                aprint(f"⚠️  Viewer port {viewer_port} busy, using {actual_viewer_port} instead")

            _serve_viewer(host, actual_viewer_port, None, open_browser)
            return

        # Require path for data serving
        if path is None:
            aprint("❌ Error: Path required unless using --viewer-only")
            raise typer.Exit(1)

        # Determine what we're serving
        if path.is_dir():
            serve_path = path
            if path.name.endswith(".zarr"):
                aprint(f"📁 Serving Zarr dataset: {path}")
            else:
                aprint(f"📂 Serving directory: {path}")
        else:
            aprint(f"❌ Error: {path} is not a directory")
            raise typer.Exit(1)

        # Find available ports (auto-increment if requested ports are busy)
        actual_port = find_available_port(port)
        if actual_port is None:
            aprint(f"❌ Error: No available ports found near {port}")
            raise typer.Exit(1)

        if actual_port != port:
            aprint(f"⚠️  Port {port} busy, using {actual_port} instead")

        if viewer:
            actual_viewer_port = find_available_port(viewer_port)
            if actual_viewer_port is None:
                aprint(f"❌ Error: No available ports found near {viewer_port}")
                raise typer.Exit(1)
            if actual_viewer_port != viewer_port:
                aprint(f"⚠️  Viewer port {viewer_port} busy, using {actual_viewer_port} instead")
        else:
            actual_viewer_port = viewer_port

        api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)

        # Add CORS middleware to allow requests from the viewer
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount the static files handler with directory listing
        api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

        aprint(f"🛰️  Serving {serve_path} at http://{host}:{actual_port}/")

        # Also serve viewer if requested
        if viewer:
            if not check_viewer_built():
                aprint("⚠️  Viewer not built. Skipping viewer serving.")
                aprint("💡 To build: cd packages/luxar-viewer && pnpm build")
            else:
                # Start viewer in a separate thread
                viewer_thread = threading.Thread(
                    target=_serve_viewer,
                    args=(host, actual_viewer_port, f"http://{host}:{actual_port}/", False),
                    daemon=True,
                )
                viewer_thread.start()
                time.sleep(1)  # Give viewer time to start
        else:
            aprint(f"📊 Viewer URL: http://localhost:5173/?src=http://{host}:{actual_port}/")

        # Open browser if requested
        if open_browser:
            url = f"http://{host}:{actual_viewer_port if viewer else 5173}/?src=http://{host}:{actual_port}/"
            time.sleep(1)  # Give servers time to start
            open_browser_func(url)

        uvicorn.run(api, host=host, port=actual_port, reload=False, log_level="warning")
    except Exception as e:
        aprint(f"❌ Error serving path: {e}")
        raise typer.Exit(1)


def _serve_viewer(
    host: str, port: int, data_url: Optional[str] = None, open_browser_flag: bool = True
) -> None:
    """Internal function to serve the viewer."""
    viewer_dist = get_viewer_dist_path()

    api = FastAPI(title="Luxar Viewer", docs_url=None, redoc_url=None)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount viewer static files
    api.mount("/", StaticFiles(directory=str(viewer_dist), html=True))

    viewer_url = f"http://{host}:{port}/"
    if data_url:
        viewer_url += f"?src={data_url}"

    aprint(f"🌐 Viewer available at: {viewer_url}")

    if open_browser_flag:
        time.sleep(1)
        open_browser_func(viewer_url)

    uvicorn.run(api, host=host, port=port, reload=False, log_level="warning")


# ────────────────────────────── viewer ──────────────────────────────────────
@app.command()
def viewer(
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Zarr data to load"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
    port: int = typer.Option(5173, "--port", "-p", help="Port number"),
    data_port: int = typer.Option(8000, "--data-port", help="Port for data server"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
) -> None:
    """Serve the Luxar viewer, optionally with data.

    Args:
        data (Path, optional): Zarr data to load.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 5173.
        data_port (int, optional): Port for data server. Defaults to 8000.
        open_browser (bool, optional): Open browser. Defaults to True.
    """
    try:
        # Check if viewer is built
        if not check_viewer_built():
            aprint("❌ Viewer not built. Building now...")
            if not build_viewer():
                aprint("❌ Failed to build viewer")
                raise typer.Exit(1)

        # If data provided, serve it in background
        data_url = None
        if data:
            if not data.exists():
                aprint(f"❌ Data path does not exist: {data}")
                raise typer.Exit(1)

            # Find available port for data server
            actual_data_port = find_available_port(data_port)
            if actual_data_port is None:
                aprint(f"❌ No available ports near {data_port}")
                raise typer.Exit(1)

            # Start data server in background thread
            data_thread = threading.Thread(
                target=_serve_data,
                args=(data, host, actual_data_port),
                daemon=True,
            )
            data_thread.start()
            time.sleep(1)  # Give data server time to start

            data_url = f"http://{host}:{actual_data_port}/"
            if data.name.endswith(".zarr"):
                data_url += data.name

        # Find available port for viewer
        actual_viewer_port = find_available_port(port)
        if actual_viewer_port is None:
            aprint(f"❌ No available ports near {port}")
            raise typer.Exit(1)
        if actual_viewer_port != port:
            aprint(f"⚠️  Viewer port {port} busy, using {actual_viewer_port} instead")

        # Serve viewer
        _serve_viewer(host, actual_viewer_port, data_url, open_browser)

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
    except Exception as e:
        aprint(f"❌ Error: {e}")
        raise typer.Exit(1)


def _serve_data(path: Path, host: str, port: int) -> None:
    """Internal function to serve data in background."""
    api = FastAPI(title="Luxar Data Server", docs_url=None, redoc_url=None)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Determine serve path
    if path.is_dir():
        serve_path = path.parent if path.name.endswith(".zarr") else path
    else:
        serve_path = path.parent

    api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

    aprint(f"💾 Data server running at http://{host}:{port}/")
    uvicorn.run(api, host=host, port=port, reload=False, log_level="warning")


# ────────────────────────────── demo ─────────────────────────────────────────
@app.command()
def demo(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    n_points: int = typer.Option(10000, "--points", "-n", help="Number of points"),
    demo_type: str = typer.Option("lorenz", "--type", "-t", help="Demo type"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    serve: bool = typer.Option(True, "--serve/--no-serve", help="Serve with viewer"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
    port: int = typer.Option(8000, "--port", "-p", help="Data server port"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Viewer port"),
) -> None:
    """Generate a demo dataset and optionally serve with viewer.

    Examples:
        # Generate and serve (default behavior)
        luxar demo

        # Just generate without serving (replaces old 'random' command)
        luxar demo --no-serve --output my_demo.zarr

        # Generate with specific parameters
        luxar demo --points 100000 --seed 42 --no-serve -o data.zarr

    Args:
        output (Path, optional): Output path. Required when --no-serve.
        n_points (int, optional): Number of points. Defaults to 10000.
        demo_type (str, optional): Demo type. Defaults to "lorenz".
        seed (int, optional): Random seed.
        serve (bool, optional): Serve with viewer. Defaults to True.
        open_browser (bool, optional): Open browser. Defaults to True.
        port (int, optional): Data server port. Defaults to 8000.
        viewer_port (int, optional): Viewer port. Defaults to 5173.
    """
    try:
        with asection("Demo Configuration and Generation"):
            # Validate arguments
            if not serve and output is None:
                aprint("❌ Error: --output required when using --no-serve")
                raise typer.Exit(1)

            # Determine output path
            if output is None:
                # Create temp directory for serving mode
                temp_dir = Path(tempfile.mkdtemp(prefix="luxar_demo_"))
                output = temp_dir / f"{demo_type}_demo.zarr"
                aprint(f"📂 Using temporary directory: {temp_dir}")

            # Generate demo
            aprint(f"🎲 Generating {demo_type} demo with {n_points:,} points...")

            if demo_type == "lorenz":
                from luxar.utils.demos import create_lorenz_attractor

                create_lorenz_attractor(output, n_points=n_points, seed=seed)
            else:
                aprint(f"❌ Unknown demo type: {demo_type}")
                aprint("💡 Available types: lorenz")
                raise typer.Exit(1)

            aprint(f"✅ Generated {n_points:,} points → {output}")

        # If not serving, we're done (replaces old 'random' command)
        if not serve:
            return

        with asection("Viewer Setup and Port Management"):
            # Check viewer is built
            if not check_viewer_built():
                aprint("🔨 Building viewer...")
                if not build_viewer():
                    aprint("❌ Failed to build viewer")
                    raise typer.Exit(1)

            # Find available ports
            actual_port = find_available_port(port)
            actual_viewer_port = find_available_port(viewer_port)

            if actual_port is None or actual_viewer_port is None:
                aprint("❌ Could not find available ports")
                raise typer.Exit(1)

        with asection("Server Startup"):
            # Start data server in background
            data_thread = threading.Thread(
                target=_serve_data,
                args=(output, "127.0.0.1", actual_port),
                daemon=True,
            )
            data_thread.start()
            time.sleep(1)

            # Construct data URL
            data_url = f"http://127.0.0.1:{actual_port}/{output.name}"

            # Serve viewer (this blocks)
            aprint("\n🎉 Demo ready! Starting viewer...")
            _serve_viewer("127.0.0.1", actual_viewer_port, data_url, open_browser)

    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down demo...")
    except Exception as e:
        aprint(f"❌ Error: {e}")
        raise typer.Exit(1)


# ────────────────────────────── info ─────────────────────────────────────────
@app.command()
def info(
    path: Path,
    tree: bool = typer.Option(True, "--tree/--no-tree", help="Show tree view"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show detailed statistics"),
    depth: Optional[int] = typer.Option(None, "--depth", "-d", help="Max tree depth"),
    format: str = typer.Option("text", "--format", help="Output format (text/json)"),
) -> None:
    """Show detailed information about a Zarr scene.

    Args:
        path (Path): Path to the Zarr store.
        tree (bool, optional): Show tree view. Defaults to True.
        stats (bool, optional): Show detailed statistics. Defaults to False.
        depth (int, optional): Max tree depth. Defaults to None (unlimited).
        format (str, optional): Output format. Defaults to "text".
    """
    try:
        if not path.exists():
            aprint("❌ Path does not exist.")
            raise typer.Exit(1)

        # Get zarr info
        info_dict = get_zarr_info(path)

        if format == "json":
            import json

            print(json.dumps(info_dict, indent=2))
            return

        # Text format output
        root = zarr.open_group(path, mode="r")

        # Header
        aprint(f"\n📁 Zarr Store: {path}")
        aprint(f"💾 Size: {format_memory_size(info_dict['size'])}")
        aprint("")

        # Root attributes
        if root.attrs:
            aprint("🎯 Root Attributes:")
            for k, v in root.attrs.items():
                if k == "scene_dimensions":
                    # Special formatting for dimensions
                    aprint(f"  {k}:")
                    if isinstance(v, dict) and "dimensions" in v:
                        for dim in v["dimensions"]:
                            aprint(
                                f"    - {dim.get('name', '?')}: {dim.get('unit', '?')} [display: {dim.get('display', False)}]"
                            )
                elif isinstance(v, (dict, list)) and len(str(v)) > 80:
                    aprint(f"  {k}: <{type(v).__name__} with {len(v)} items>")
                else:
                    aprint(f"  {k}: {v}")
            aprint("")

        # Tree view
        if tree:
            aprint("🌳 Scene Hierarchy:")
            _print_tree(root, max_depth=depth, show_stats=stats)
            aprint("")

        # Statistics
        aprint("📊 Summary Statistics:")
        aprint(f"  🗂️  Groups: {info_dict['n_groups']}")
        aprint(f"  📦 Arrays: {info_dict['n_arrays']}")
        aprint(f"  ⭕ Points objects: {len(info_dict['points_objects'])}")
        aprint(f"  ✨ Total points: {info_dict['n_points_total']:,}")

        if stats and info_dict["points_objects"]:
            aprint("\n📦 Points Objects Details:")
            for pc in info_dict["points_objects"]:
                aprint(f"  {pc['path']}:")
                aprint(f"    Points: {pc['n_points']:,}")
                aprint(f"    Dimensions: {pc['n_dims']}")
                aprint(f"    Has colors: {pc['has_colors']}")
                aprint(f"    Has radii: {pc['has_radii']}")
                aprint(f"    Has sharpness: {pc['has_sharpness']}")
    except Exception as e:
        aprint(f"❌ Error reading info for {path}: {e}")
        raise typer.Exit(1)


def _print_tree(
    group: zarr.Group,
    depth: int = 0,
    max_depth: Optional[int] = None,
    prefix: str = "",
    is_last: bool = True,
    show_stats: bool = False,
) -> None:
    """Print a tree view of the zarr hierarchy."""
    if max_depth is not None and depth > max_depth:
        return

    # Determine node type and gather info
    node_type = "scene" if depth == 0 else "group"
    attrs = {}

    if "positions" in group:
        node_type = "points"
        positions = group["positions"]
        attrs["n_points"] = positions.shape[0]
        if show_stats:
            attrs["shape"] = positions.shape
            attrs["dtype"] = str(positions.dtype)

    # Print node
    if depth == 0:
        print(format_tree_node("/", depth, is_last, prefix, node_type, attrs))
    else:
        name = group.basename or "?"
        print(format_tree_node(name, depth, is_last, prefix, node_type, attrs))

    # Update prefix for children
    if depth > 0:
        if is_last:
            new_prefix = prefix + "    "
        else:
            new_prefix = prefix + "│   "
    else:
        new_prefix = ""

    # Get children
    subgroups = list(group.group_keys())

    # Print children
    for i, subgroup_name in enumerate(subgroups):
        is_last_child = i == len(subgroups) - 1
        subgroup = group[subgroup_name]
        _print_tree(
            subgroup, depth + 1, max_depth, new_prefix, is_last_child, show_stats
        )


def _dfs(
    group: zarr.Group, depth: int = 0
) -> Generator[Tuple[int, zarr.Group], None, None]:
    """Depth-first walk that yields (depth, group) for every subgroup.

    Args:
        group (zarr.Group): Zarr group to traverse.
        depth (int, optional): Current depth. Defaults to 0.

    Yields:
        Tuple[int, zarr.Group]: (depth, group) for each subgroup.
    """
    try:
        yield depth, group
        for name in group.group_keys():
            yield from _dfs(group[name], depth + 1)
    except Exception as e:
        aprint(f"Error traversing Zarr group hierarchy: {e}")
        raise


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
