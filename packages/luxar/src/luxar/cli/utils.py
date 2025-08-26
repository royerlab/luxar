"""luxar.cli_utils – Helper utilities for the Luxar CLI."""

from __future__ import annotations

import socket
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Optional

import zarr
from arbol import aprint


def open_browser(url: str, suppress_errors: bool = False) -> bool:
    """Open a URL in the default web browser.

    Args:
        url: URL to open.
        suppress_errors: If True, don't print error messages.

    Returns:
        True if successful, False otherwise.
    """
    try:
        webbrowser.open(url)
        return True
    except Exception as e:
        if not suppress_errors:
            aprint(f"⚠️ Could not open browser: {e}")
        return False


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available for binding.

    Args:
        port: Port number to check.
        host: Host address to check.

    Returns:
        True if port is available, False if in use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def find_available_port(
    start_port: int = 8000, max_attempts: int = 100
) -> Optional[int]:
    """Find an available port starting from the given port.

    Args:
        start_port: Port to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number, or None if none found.
    """
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    return None


def check_viewer_built() -> bool:
    """Check if the Luxar viewer is built.

    Returns:
        True if viewer dist directory exists, False otherwise.
    """
    viewer_dist = get_viewer_dist_path()
    return viewer_dist.exists() and (viewer_dist / "index.html").exists()


def get_viewer_dist_path() -> Path:
    """Get the path to the viewer distribution directory.

    Returns:
        Path to the viewer dist directory.
    """
    # Find the project root by looking for pyproject.toml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            viewer_dist = current / "packages" / "luxar-player" / "dist"
            if viewer_dist.exists():
                return viewer_dist
        current = current.parent

    # Fallback to relative path
    return (
        Path(__file__).parent.parent.parent.parent.parent
        / "packages"
        / "luxar-player"
        / "dist"
    )


def build_viewer() -> bool:
    """Build the Luxar viewer using pnpm.

    Returns:
        True if build successful, False otherwise.
    """
    viewer_path = get_viewer_dist_path().parent
    try:
        subprocess.run(
            ["pnpm", "build"],
            cwd=viewer_path,
            capture_output=True,
            text=True,
            check=True,
        )
        aprint("✅ Viewer built successfully")
        return True
    except subprocess.CalledProcessError as e:
        aprint(f"❌ Failed to build viewer: {e.stderr}")
        return False
    except FileNotFoundError:
        aprint("❌ pnpm not found. Please install pnpm first: npm install -g pnpm")
        return False


def format_tree_node(
    name: str,
    depth: int,
    is_last: bool,
    prefix: str = "",
    node_type: Optional[str] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> str:
    """Format a tree node for display.

    Args:
        name: Node name.
        depth: Current depth in tree.
        is_last: Whether this is the last child.
        prefix: Prefix for the current line.
        node_type: Type of node (scene, group, points).
        attrs: Node attributes to display.

    Returns:
        Formatted tree node string.
    """
    if depth == 0:
        line = f"📊 {name}"
    else:
        connector = "└─" if is_last else "├─"
        line = f"{prefix}{connector} {name}"

    # Add type indicator
    if node_type == "scene":
        line += " 🌐"
    elif node_type == "group":
        line += " 📁"
    elif node_type == "points":
        line += " ⚫"

    # Add selected attributes
    if attrs:
        important_attrs = []
        if "n_points" in attrs:
            important_attrs.append(f"n={attrs['n_points']:,}")
        if "shape" in attrs:
            important_attrs.append(f"shape={attrs['shape']}")
        if "dtype" in attrs:
            important_attrs.append(f"dtype={attrs['dtype']}")
        if important_attrs:
            line += f" [{', '.join(important_attrs)}]"

    return line


def format_memory_size(bytes_size: float) -> str:
    """Format bytes size to human-readable string.

    Args:
        bytes_size: Size in bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_zarr_info(store_path: Path) -> dict[str, Any]:
    """Get detailed information about a Zarr store.

    Args:
        store_path: Path to the Zarr store.

    Returns:
        Dictionary with store information.
    """
    info = {
        "path": str(store_path),
        "exists": store_path.exists(),
        "size": 0,
        "n_groups": 0,
        "n_arrays": 0,
        "n_points_total": 0,
        "point_clouds": [],
    }

    if not info["exists"]:
        return info

    try:
        # Calculate total size
        if store_path.is_dir():
            info["size"] = sum(
                f.stat().st_size for f in store_path.rglob("*") if f.is_file()
            )

        # Open and analyze store
        root = zarr.open_group(store_path, mode="r")

        def analyze_group(group: zarr.Group, path: str = "") -> None:
            """Recursively analyze a Zarr group."""
            info["n_groups"] += 1  # type: ignore

            # Check if this is a points group
            if "positions" in group:
                positions = group["positions"]
                point_info = {
                    "path": path or "/",
                    "n_points": positions.shape[0],
                    "n_dims": positions.shape[1] if len(positions.shape) > 1 else 1,
                    "has_colors": "colors" in group,
                    "has_radii": "radii" in group,
                    "has_sharpness": "sharpness" in group,
                }
                info["point_clouds"].append(point_info)  # type: ignore
                info["n_points_total"] += point_info["n_points"]

            # Count arrays
            for array_name in group.array_keys():
                info["n_arrays"] += 1  # type: ignore

            # Recurse into subgroups
            for subgroup_name in group.group_keys():
                analyze_group(group[subgroup_name], f"{path}/{subgroup_name}")

        analyze_group(root)

    except Exception as e:
        info["error"] = str(e)

    return info


def validate_zarr_store(store_path: Path) -> tuple[bool, Optional[str]]:
    """Validate that a path is a valid Zarr store.

    Args:
        store_path: Path to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not store_path.exists():
        return False, "Path does not exist"

    if not store_path.is_dir():
        return False, "Path is not a directory"

    try:
        zarr.open_group(store_path, mode="r")
        return True, None
    except Exception as e:
        return False, f"Not a valid Zarr store: {e}"
