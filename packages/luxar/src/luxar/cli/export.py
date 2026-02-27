"""Export a Luxar zarr scene + viewer into a standalone offline folder.

The exported folder contains everything needed to view the scene:
- The Luxar viewer (HTML, JS, CSS, WASM)
- The zarr dataset (copied as-is)
- A serve.py script (Python 3 stdlib only)
- A README.txt with usage instructions

Usage:
    luxar export my_scene.zarr -o my_export/
    cd my_export && python serve.py
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

from arbol import aprint, asection

from .utils import check_viewer_built, get_viewer_dist_path, validate_zarr_store


def export_scene(
    source: Path,
    output: Path,
    data_dir_name: str = "data",
    overwrite: bool = False,
) -> Path:
    """Export a zarr scene with viewer into a standalone folder.

    Args:
        source: Path to the source zarr store.
        output: Path to the output folder.
        data_dir_name: Name for the data directory inside output.
        overwrite: Whether to overwrite existing output.

    Returns:
        Path to the created output folder.

    Raises:
        FileExistsError: If output exists and overwrite is False.
        FileNotFoundError: If source doesn't exist or viewer not built.
        ValueError: If source is not a valid zarr store.
    """
    # Validate source
    is_valid, error = validate_zarr_store(source)
    if not is_valid:
        raise ValueError(f"Invalid zarr store: {error}")

    # Check viewer is built
    if not check_viewer_built():
        raise FileNotFoundError(
            "Viewer not built. Run: cd packages/luxar-viewer && pnpm build"
        )

    # Handle output directory
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    with asection("Exporting standalone scene"):
        # Step 1: Copy viewer
        _copy_viewer(output / "viewer")

        # Step 2: Copy zarr data
        _copy_zarr_data(source, output / data_dir_name)

        # Step 3: Generate serve.py
        _generate_serve_script(output, data_dir_name)

        # Step 4: Generate README.txt
        _generate_readme(output, data_dir_name)

    return output


def _copy_viewer(dest: Path) -> None:
    """Copy viewer dist files to destination.

    Args:
        dest: Destination directory for viewer files.
    """
    viewer_dist = get_viewer_dist_path()
    with asection("Copying viewer files"):
        shutil.copytree(viewer_dist, dest)
        # Count files for user feedback
        n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
        aprint(f"Copied {n_files} viewer files to {dest}")


def _copy_zarr_data(source: Path, dest: Path) -> None:
    """Copy zarr data to destination.

    Args:
        source: Source zarr store path.
        dest: Destination directory for zarr data.
    """
    with asection("Copying zarr data"):
        shutil.copytree(source, dest)
        # Calculate size for user feedback
        total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        aprint(f"Copied zarr data ({size_mb:.1f} MB) to {dest}")


def _generate_serve_script(output: Path, data_dir_name: str) -> None:
    """Generate the serve.py script.

    Args:
        output: Output directory root.
        data_dir_name: Name of the data directory.
    """
    script_content = _get_serve_script_content(data_dir_name)
    serve_path = output / "serve.py"
    serve_path.write_text(script_content)
    # Make executable on Unix
    serve_path.chmod(
        serve_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    aprint(f"Generated {serve_path}")


def _get_serve_script_content(data_dir_name: str) -> str:
    """Return the serve.py script content.

    Uses only Python 3 stdlib -- no external dependencies required.

    Args:
        data_dir_name: Name of the data directory (embedded in the script).

    Returns:
        Complete serve.py script as a string.
    """
    return f'''#!/usr/bin/env python3
"""Serve this exported Luxar scene locally.

Usage:
    python serve.py [--port PORT] [--no-open]

Requirements:
    Python 3 (stdlib only -- no pip install needed)
"""

import argparse
import http.server
import socket
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR_NAME = "{data_dir_name}"


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers for cross-origin zarr access."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress request logging for cleaner output."""
        pass


def find_port(start=8000, attempts=100):
    """Find an available port starting from start."""
    for i in range(attempts):
        port = start + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Serve exported Luxar scene")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    port = find_port(args.port)
    if port is None:
        print("Error: No available port found near {{}}".format(args.port))
        sys.exit(1)

    handler = partial(CORSHandler, directory=str(SCRIPT_DIR))
    server = http.server.HTTPServer(("127.0.0.1", port), handler)

    data_url = "http://127.0.0.1:{{}}/{{}}".format(port, DATA_DIR_NAME)
    viewer_url = "http://127.0.0.1:{{}}/viewer/?src={{}}".format(port, data_url)

    print("Luxar viewer: {{}}".format(viewer_url))
    print("Press Ctrl+C to stop")

    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=[viewer_url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
'''


def _generate_readme(output: Path, data_dir_name: str) -> None:
    """Generate README.txt with usage instructions.

    Args:
        output: Output directory root.
        data_dir_name: Name of the data directory.
    """
    readme = f"""Luxar Exported Scene
====================

This folder contains a self-contained Luxar scene viewer.

Quick Start
-----------
    python serve.py

This starts a local HTTP server and opens the viewer in your browser.

Requirements
------------
- Python 3 (any version, no extra packages needed)
- A modern web browser (Chrome, Firefox, Safari, Edge)

Options
-------
    python serve.py --port 9000     Use a custom port
    python serve.py --no-open       Don't open browser automatically

Folder Structure
----------------
    viewer/         The Luxar viewer (HTML, JS, CSS, WASM)
    {data_dir_name}/             The zarr dataset
    serve.py        Local HTTP server (Python stdlib only)
    README.txt      This file

Notes
-----
- Do NOT open viewer/index.html directly -- file:// protocol does not work
- The serve.py script uses only Python stdlib -- no pip install needed
- To share this scene: zip the entire folder and send it
"""
    (output / "README.txt").write_text(readme)
    aprint("Generated README.txt")
