"""Export a Luxar zarr scene + viewer into a standalone offline folder.

The exported folder contains everything needed to view the scene:
- The Luxar viewer (HTML, JS, CSS, WASM) -- including the touch-panel page
- The zarr dataset (copied as-is)
- A serve.py script (Python 3 stdlib only), which can also host the
  remote-control relay so the folder alone runs a kiosk
- A README.txt with usage instructions

Usage:
    luxar export my_scene.luxar.zarr -o my_export/
    cd my_export && python serve.py
    cd my_export && python serve.py --control --host 0.0.0.0   # kiosk
"""

from __future__ import annotations

import re
import shutil
import stat
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from arbol import aprint, asection

from ..utils.atomic_copy import atomic_copytree
from .utils import (
    check_viewer_built,
    dataset_title,
    format_memory_size,
    get_viewer_dist_path,
    validate_zarr_store,
)


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

    # Preserve the existing-output error when overwrite was not requested.
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")

    # Preflight the viewer before an overwrite can remove the user's old export.
    if not check_viewer_built():
        raise FileNotFoundError(
            "Viewer not built. Run: cd packages/luxar-viewer && pnpm build"
        )
    _require_third_party_notices(get_viewer_dist_path())

    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    with asection("Exporting standalone scene"):
        # Step 1: Copy viewer
        _copy_viewer(output / "viewer")

        # Step 2: Copy zarr data
        _copy_zarr_data(source, output / data_dir_name)

        # Step 3: Generate serve.py
        _generate_serve_script(output, data_dir_name, dataset_title(source))

        # Step 4: Generate README.txt
        _generate_readme(output, data_dir_name)

    return output


#: Emitted into the viewer build by
#: ``packages/luxar-viewer/scripts/generate-third-party-licenses.mjs``.
THIRD_PARTY_LICENSES = "THIRD_PARTY_LICENSES.txt"


def _require_third_party_notices(viewer_dist: Path) -> None:
    """Refuse to export a bundle whose third-party notices are missing.

    An export folder is redistribution: it is zipped, hosted, and handed to
    people who never see this repository. MIT / Apache-2.0 / MPL-2.0 / BSD all
    require their notices to travel with the binary, so a folder without them
    is not one we should be able to produce by accident.

    The production build always emits this file, so its absence means the dist
    predates that change or was assembled by hand -- both of which are worth
    stopping for, because the resulting folder looks complete.
    """
    if (viewer_dist / THIRD_PARTY_LICENSES).is_file():
        return
    raise FileNotFoundError(
        f"{viewer_dist} has no {THIRD_PARTY_LICENSES}, so exporting it would "
        f"redistribute third-party code without the notices its licences "
        f"require. Rebuild the viewer (`make build-viewer`, or `pnpm build` in "
        f"packages/luxar-viewer) — the production build generates it."
    )


def _copy_viewer(dest: Path) -> None:
    """Copy viewer dist files to destination.

    Args:
        dest: Destination directory for viewer files.
    """
    viewer_dist = get_viewer_dist_path()
    _require_third_party_notices(viewer_dist)
    with asection("Copying viewer files"):
        # CL-1: atomic copy — partial output on crash leaves no half-written
        # `dest` for the user to clean up.
        atomic_copytree(viewer_dist, dest)
        # Rewrite absolute asset paths to relative so the viewer works
        # when served from a subdirectory (e.g. /viewer/).
        _rewrite_absolute_paths(dest)
        # Count files for user feedback
        n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
        aprint(f"Copied {n_files} viewer files to {dest}")


def _rewrite_absolute_paths(viewer_dir: Path) -> None:
    """Rewrite absolute asset/wasm paths to relative in all viewer files.

    Vite builds may produce absolute paths like ``/assets/index-xxx.js``
    and JS bundles reference ``"/wasm/..."`` and ``"/assets/..."``.
    These break when the viewer is served from a subdirectory
    (e.g. ``/viewer/``). This rewrites them to ``./`` relative paths.

    Args:
        viewer_dir: Root directory of the copied viewer files.
    """
    count = 0
    for path in viewer_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".html", ".js", ".css"):
            continue
        text = path.read_text()
        # Compute the relative prefix from this file back to the viewer root.
        # Files in viewer/ use "./" while files in viewer/assets/ use "../".
        rel = path.parent.relative_to(viewer_dir)
        if rel == Path("."):
            prefix = "./"
        else:
            prefix = "../" * len(rel.parts)
        # Rewrite /assets/ and /wasm/ absolute paths to relative.
        # For import() calls, paths resolve relative to the module URL,
        # so files in assets/ need "../" to reach sibling dirs like wasm/.
        updated = re.sub(r'(["\'])/assets/', rf"\1{prefix}assets/", text)
        updated = re.sub(r'(["\'])/wasm/', rf"\1{prefix}wasm/", updated)
        # Rewrite Vite's base path resolver: function(n){return"/"+n}
        # This function creates <link> elements in document.head for
        # modulepreload. Links resolve relative to the DOCUMENT URL
        # (not the module URL), so always use "./" here.
        updated = re.sub(
            r'function\(n\)\{return"/"\+n\}',
            'function(n){return"./"+n}',
            updated,
        )
        if updated != text:
            path.write_text(updated)
            count += 1
    if count:
        aprint(f"Rewrote absolute paths to relative in {count} file(s)")


def _copy_zarr_data(source: Path, dest: Path) -> None:
    """Copy zarr data to destination.

    Args:
        source: Source zarr store path.
        dest: Destination directory for zarr data.
    """
    with asection("Copying zarr data"):
        # CL-1: atomic copy.
        atomic_copytree(source, dest)
        # Calculate size for user feedback
        total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        aprint(f"Copied zarr data ({format_memory_size(total_size)}) to {dest}")


def _generate_serve_script(
    output: Path, data_dir_name: str, title: Optional[str] = None
) -> None:
    """Generate the serve.py script.

    Args:
        output: Output directory root.
        data_dir_name: Name of the data directory.
        title: Optional browser-tab title baked into the viewer URL (derived
            from the source scene's file name), so exported-scene tabs are
            tellable apart like every other serve-family command's.
    """
    script_content = _get_serve_script_content(data_dir_name, title)
    serve_path = output / "serve.py"
    serve_path.write_text(script_content)
    # Make executable on Unix
    serve_path.chmod(
        serve_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    aprint(f"Generated {serve_path}")


#: The generated ``serve.py`` is this module, copied with two values
#: substituted. It is a real module rather than a string literal so ruff, mypy
#: and the test suite all see it -- a WebSocket relay hidden inside an f-string
#: (where every brace has to be doubled) is unreviewable, and this one is the
#: fourth implementation of a relay that three other files already agree on.
SERVE_TEMPLATE = Path(__file__).with_name("_export_serve_template.py")

#: Assignments substituted in the template, as ``name -> value`` at export time.
#: Matched anchored at line start, so the constants' *uses* are untouched.
_SUBSTITUTED = ("DATA_DIR_NAME", "TITLE_QUERY")


def _substitute(script: str, name: str, value: str) -> str:
    """Replace the template's ``name = ...`` line with ``value``.

    Raises rather than silently letting the template's own default through: a
    missed substitution would export a folder whose serve.py points at a data
    directory that is not there, which looks like a broken scene rather than a
    broken export.
    """
    pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
    script, count = pattern.subn(f"{name} = {value!r}", script, count=1)
    if count != 1:
        raise RuntimeError(
            f"{SERVE_TEMPLATE.name} has no `{name} = ...` line to substitute; "
            f"the template and the exporter have drifted apart."
        )
    return script


def _get_serve_script_content(data_dir_name: str, title: Optional[str] = None) -> str:
    """Return the serve.py script content.

    Uses only Python 3 stdlib -- no external dependencies required, because
    this folder gets zipped and handed to people who have never installed
    Luxar.

    Args:
        data_dir_name: Name of the data directory (embedded in the script).
        title: Optional tab title, baked in PRE-ENCODED as a ready-to-append
            query fragment.

    Returns:
        Complete serve.py script as a string.
    """
    script = SERVE_TEMPLATE.read_text()
    values = {
        "DATA_DIR_NAME": data_dir_name,
        "TITLE_QUERY": f"&title={quote(title)}" if title else "",
    }
    for name in _SUBSTITUTED:
        script = _substitute(script, name, values[name])
    return script


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
- Python 3.9 or newer (stdlib only, no extra packages needed)
- A modern web browser (Chrome, Firefox, Safari, Edge)

Options
-------
    python serve.py --port 9000     Use a custom port
    python serve.py --no-open       Don't open browser automatically
    python serve.py --host 0.0.0.0  Accept connections from the network
    python serve.py --control       Host the touch-panel relay (see below)

Kiosk mode: driving the display from a tablet
---------------------------------------------
    python serve.py --control --host 0.0.0.0 --control-token SECRET

With --control this folder runs a kiosk on its own: the script prints two
URLs, one for the big display and one for a tablet. Open the first on the
display, the second on the tablet, and tapping a tile moves the display.

It is off by default, because a folder you double-click should not start
listening for anything that wants to drive it. --host 0.0.0.0 is what makes
the tablet able to reach it at all, and on any network you do not control,
pass --control-token: without one, anything that can reach this machine can
drive the display. The token travels in the URL, so it is visible in the
tablet's address bar and its history -- fine for a LAN exhibit, not a
password.

Folder Structure
----------------
    viewer/         The Luxar viewer (HTML, JS, CSS, WASM)
    {data_dir_name}/             The zarr dataset
    serve.py        Local HTTP server + control relay (Python stdlib only)
    README.txt      This file

Notes
-----
- Do NOT open the viewer's HTML files directly. Both pages
  (viewer/index.html and the touch panel viewer/control.html) need a server:
  the file:// protocol cannot fetch the dataset.
- The serve.py script uses only Python stdlib -- no pip install needed
- To share this scene: zip the entire folder and send it
"""
    (output / "README.txt").write_text(readme)
    aprint("Generated README.txt")
