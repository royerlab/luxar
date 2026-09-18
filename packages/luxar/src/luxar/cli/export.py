"""Export a Luxar zarr scene + viewer into a standalone offline folder.

The exported folder contains everything needed to view the scene:

- The Luxar viewer (HTML, JS, CSS, WASM) -- including the touch-panel page
- The zarr dataset (copied as-is)
- A serve.py script (Python 3 stdlib only), which can also host the
  remote-control relay so the folder alone runs a kiosk
- A README.txt with usage instructions

Usage::

    luxar export my_scene.luxar.zarr -o my_export/
    cd my_export && python3 serve.py
    cd my_export && python3 serve.py --control --host 0.0.0.0   # kiosk
"""

from __future__ import annotations

import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from arbol import aprint, asection

from .._zarr_compat import read_node_attrs
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

    facts = read_scene_facts(source)

    with asection("Exporting standalone scene"):
        # Step 1: Copy viewer
        _copy_viewer(output / "viewer")

        # Step 2: Copy zarr data
        _copy_zarr_data(source, output / data_dir_name)

        # Step 3: Generate serve.py, plus the QR encoder it imports
        _generate_serve_script(output, data_dir_name, dataset_title(source), facts)
        _copy_qr_module(output)

        # Step 4: Generate README.txt, plus a test note for a kiosk package
        _generate_readme(output, data_dir_name, facts)
        if facts.has_control_panel:
            _generate_testing_notes(output, facts)
            aprint(
                "This scene declares a control panel, so serve.py defaults to "
                "--control --host 0.0.0.0 and prints a QR for the tablet."
            )

    return output


@dataclass(frozen=True)
class SceneFacts:
    """What the exporter can learn about a scene from its own attributes.

    Read through :func:`luxar._zarr_compat.read_node_attrs`, never by naming a
    metadata document: the attributes live in ``.zattrs`` at zarr format 2 and
    nested under ``attributes`` in ``zarr.json`` at format 3, and a tree
    holding both is the normal steady state here. A literal filename would
    answer "no control panel" for half the stores in existence, and the export
    would look like it worked.
    """

    title: str | None = None
    #: A scene authored to be driven from a tablet: it carries a
    #: ``viewer_config.control_panel``. Decides the exported serve.py's
    #: defaults and what the README leads with.
    has_control_panel: bool = False
    #: The dimension the panel steps, and how many stops it names.
    chapter_dimension: str | None = None
    #: Authored ``Chapter`` entries. NOT the number of tiles: a chapter only
    #: supplies a slot's sublabel, so a dimension coordinate with no authored
    #: chapter still gets a tile, labelled from the dimension's categories.
    chapter_count: int = 0
    #: Tiles the panel actually draws -- one per coordinate of the chapter
    #: dimension. The protein-universe scene authors 20 chapters on a 21-value
    #: dimension (slot 0 is the Overview), so quoting `chapter_count` in the
    #: testing notes told the operator to expect 20 tiles and count 21.
    panel_tiles: int = 0
    #: Dimension names in order, and which of them the viewer displays.
    dimensions: tuple[str, ...] = ()
    displayed: tuple[str, ...] = ()
    citation: str | None = None

    @property
    def stops(self) -> int:
        """Tiles the panel draws, for operator-facing text.

        Falls back to the authored chapter count so a `SceneFacts` built by
        hand -- a test, or any caller not going through `read_scene_facts` --
        still reports a sensible number instead of silently printing zero and
        dropping the scene summary from the README entirely.
        """
        return self.panel_tiles or self.chapter_count


def _coordinate_count(dim: dict) -> int:
    """How many coordinates a dimension has, and so how many tiles it draws.

    Prefers the explicit ``categories`` list, which is what labels the tiles.
    Falls back to the inclusive span of ``range`` in units of ``step`` for a
    continuous or discrete dimension that names no categories. Returns 0 when
    neither is readable, so the caller can choose its own fallback.
    """
    cats = dim.get("categories")
    if isinstance(cats, list) and cats:
        return len(cats)
    rng = dim.get("range")
    step = dim.get("step") or 1
    if isinstance(rng, (list, tuple)) and len(rng) == 2 and step:
        try:
            return int(round((float(rng[1]) - float(rng[0])) / float(step))) + 1
        except (TypeError, ValueError, ZeroDivisionError):
            return 0
    return 0


def read_scene_facts(source: Path) -> SceneFacts:
    """Introspect ``source`` for the facts the export needs. Never raises.

    A store this cannot read is not an export failure: the folder is still
    valid, it just gets the generic README and the conservative serve.py
    defaults. So every lookup is defensive.
    """
    try:
        attrs = read_node_attrs(source) or {}
    except Exception:  # noqa: BLE001 - introspection must not fail an export
        return SceneFacts()

    viewer = attrs.get("viewer_config") or {}
    panel = viewer.get("control_panel") or {}
    chapters = panel.get("chapters") or {}
    # `scene_dimensions.dimensions`, not a top-level `dimensions`. Getting
    # this wrong is silent: the lookup returns nothing, `read_scene_facts`
    # reports no dimensions, and the README simply omits those lines rather
    # than complaining -- so the shape is asserted in the export tests.
    dims = (attrs.get("scene_dimensions") or {}).get("dimensions") or []
    names, shown = [], []
    panel_tiles = 0
    for dim in dims if isinstance(dims, list) else []:
        if not isinstance(dim, dict):
            continue
        name = str(dim.get("name", ""))
        if not name:
            continue
        names.append(name)
        if dim.get("display"):
            shown.append(name)
        if name == panel.get("chapter_dimension"):
            panel_tiles = _coordinate_count(dim)
    citation = attrs.get("citation") or {}
    return SceneFacts(
        title=viewer.get("title") or None,
        has_control_panel=bool(panel),
        chapter_dimension=panel.get("chapter_dimension") or None,
        chapter_count=len(chapters) if isinstance(chapters, dict) else 0,
        panel_tiles=panel_tiles,
        dimensions=tuple(names),
        displayed=tuple(shown),
        citation=(citation.get("ref") or citation.get("short") or None)
        if isinstance(citation, dict)
        else None,
    )


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
    output: Path,
    data_dir_name: str,
    title: Optional[str] = None,
    facts: Optional["SceneFacts"] = None,
) -> None:
    """Generate the serve.py script.

    Args:
        output: Output directory root.
        data_dir_name: Name of the data directory.
        title: Optional browser-tab title baked into the viewer URL (derived
            from the source scene's file name), so exported-scene tabs are
            tellable apart like every other serve-family command's.
        facts: What the scene says about itself. A scene declaring a control
            panel gets a serve.py whose relay and network binding are on by
            default, because an exhibit operator should not have to read a
            README to make the tablet work.
    """
    script_content = _get_serve_script_content(data_dir_name, title, facts)
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
_SUBSTITUTED = ("DATA_DIR_NAME", "TITLE_QUERY", "HAS_CONTROL_PANEL")


def _substitute(script: str, name: str, value: object) -> str:
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


def _generate_testing_notes(output: Path, facts: "SceneFacts") -> None:
    """Write TESTING.txt: what to try, and what has NOT been checked.

    Only for a scene with a control panel. A plain export has nothing
    non-obvious to test -- you open it and it either draws or it does not --
    whereas pairing a tablet has an order to it, a failure mode that looks
    like the panel being broken when it is the network, and limits worth
    stating before someone finds them and assumes they are bugs.
    """
    stops = facts.stops
    notes = f"""Testing this package
====================

Read README.txt first for how to run it. This file is what to check, and what
has not been checked.

1. The display on its own
-------------------------
    python3 serve.py --no-control

Open the Display URL. The scene should draw within a few seconds on a laptop
GPU, and the on-screen panel should name the stop. Step the story dimension
with `]` and `[`, or press 1 then `]`. Nothing here needs the network.

2. The display plus a tablet
----------------------------
    python3 serve.py

This is the kiosk default for this scene. The script prints two URLs and a QR.

    a. Open the Display URL on the big screen. Wait for it to finish loading.
    b. Scan the QR with the tablet, or type the Control panel URL.
       The tablet and this machine must be on the SAME network.
    c. The tablet should show {stops} tiles, each naming a stop.
    d. Tap a tile. The display should fly to that stop within a second or two.

If the tablet shows tiles but tapping does nothing, the page loaded over HTTP
and the WebSocket did not connect -- almost always a firewall or a guest
network that blocks client-to-client traffic, not the software. If the tablet
cannot load the page at all, check that serve.py printed a 192.168.x or 10.x
address rather than 127.0.0.1.

IF THE TABLET SHOWS NO TILES AT ALL, and especially if you are testing both
pages on ONE machine in two tabs of the same browser: that is the likely
cause, not a bug. The panel asks the display for its list of stops and gives
up after 30 seconds, and a browser throttles a background tab hard enough to
miss the question. Measured here: two tabs in one browser -> no tiles; the
same two pages in two separate browser windows against the same server ->
21 tiles and taps landing correctly. On a real kiosk the display is a
different machine, so this does not arise.

3. Worth trying
---------------
- Close and reopen the tablet page while the display keeps running.
- Two tablets at once: both should work, last tap wins.
- Restart serve.py: the tablet page needs reloading, the URL is the same
  unless the port changed.
- `--control-token WORD`, then scan the new QR. An old QR should stop working.

Known limits, so you do not report these as bugs
------------------------------------------------
- NO TOKEN BY DEFAULT. Anyone who can reach this machine on the network can
  drive the display. A foreign web page cannot -- the relay checks the origin
  on the WebSocket handshake -- but a person on the same network who opens the
  panel URL can. Use --control-token on a network you do not control.
- The token travels in the URL, so it is visible in the tablet's address bar
  and history. It is an exhibit lock, not a password.
- Verified on macOS with Python 3.9.6 and on Linux with Python 3.12, both
  serving a tablet-style client over the network. Windows is untested: the
  scripts are stdlib-only and should work with `python serve.py`, but nobody
  has run it.
- Narration and ambient sound need a user gesture before most browsers will
  play audio. Click the display once if it is silent.
- The panel page is served even with --no-control. It will load and then fail
  to connect, because the relay is not running. That is expected.
- Opening the HTML files directly with file:// cannot work; the dataset is
  fetched over HTTP.

Reporting something
-------------------
Useful to include: what you ran, the two URLs it printed, your OS and
`python3 --version`, and the browser console on whichever page misbehaved.
"""
    (output / "TESTING.txt").write_text(notes)
    aprint("Generated TESTING.txt")


#: The QR encoder, copied beside serve.py under this name. serve.py imports it
#: by this exact name, so the two move together.
QR_MODULE_SOURCE = Path(__file__).with_name("_qr.py")
QR_MODULE_NAME = "luxar_qr.py"


def _copy_qr_module(output: Path) -> None:
    """Ship the QR encoder next to serve.py.

    A separate file rather than text pasted into serve.py: both stay readable,
    ruff and mypy see the real module in the source tree, and ``test_qr.py``
    tests the same bytes that get shipped. serve.py degrades to printing URLs
    alone if it is missing, so deleting it is survivable rather than fatal.
    """
    shutil.copy2(QR_MODULE_SOURCE, output / QR_MODULE_NAME)
    aprint(f"Generated {output / QR_MODULE_NAME}")


def _get_serve_script_content(
    data_dir_name: str,
    title: Optional[str] = None,
    facts: Optional["SceneFacts"] = None,
) -> str:
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
    values: dict[str, object] = {
        "DATA_DIR_NAME": data_dir_name,
        "TITLE_QUERY": f"&title={quote(title)}" if title else "",
        "HAS_CONTROL_PANEL": bool(facts and facts.has_control_panel),
    }
    for name in _SUBSTITUTED:
        script = _substitute(script, name, values[name])
    return script


def _generate_readme(
    output: Path, data_dir_name: str, facts: Optional["SceneFacts"] = None
) -> None:
    """Generate README.txt: how to run the folder, and what is in it.

    Two things beyond the mechanics, both read from the scene's own
    attributes rather than assumed. A scene that declares a control panel gets
    a kiosk section that LEADS, with the exact command, because that is what
    the folder is for; a scene without one gets kiosk mode as a footnote, as
    before. And every scene gets a short summary of itself -- title, what the
    dimensions are, how many stops the panel names, the citation -- so the
    folder explains the thing it contains and not only how to serve it.
    """
    facts = facts or SceneFacts()
    scene_lines = []
    if facts.title:
        scene_lines.append(f"    Title        {facts.title}")
    if facts.dimensions:
        shown = ", ".join(facts.displayed) or "(none)"
        hidden = ", ".join(d for d in facts.dimensions if d not in facts.displayed)
        scene_lines.append(f"    Displayed    {shown}")
        if hidden:
            scene_lines.append(f"    Steppable    {hidden}")
    if facts.has_control_panel and facts.chapter_dimension:
        scene_lines.append(
            f"    Panel        {facts.stops} stops along '{facts.chapter_dimension}'"
        )
    if facts.citation:
        scene_lines.append(f"    Data         {facts.citation}")
    about = (
        "About this scene\n----------------\n" + "\n".join(scene_lines) + "\n\n"
        if scene_lines
        else ""
    )

    # Column-aligned whatever the data directory is called; the old listing
    # hard-coded the padding for "data" and went crooked for anything else.
    entries = [
        ("viewer/", "The Luxar viewer (HTML, JS, CSS, WASM)"),
        (f"{data_dir_name}/", "The zarr dataset"),
        ("serve.py", "Local HTTP server + control relay (stdlib only)"),
        ("luxar_qr.py", "QR encoder, imported by serve.py for the panel URL"),
        ("README.txt", "This file"),
    ]
    if facts.has_control_panel:
        entries.append(("TESTING.txt", "What to check, and what is unverified"))
    width = max(len(name) for name, _ in entries)
    structure = "\n".join(f"    {name:<{width}}  {what}" for name, what in entries)

    if facts.has_control_panel:
        kiosk = """Kiosk mode: driving the display from a tablet
---------------------------------------------
THIS SCENE HAS A CONTROL PANEL, so `python3 serve.py` already starts it: the
relay is on and the server binds every interface, because that is the only way
a tablet can reach it. On start the script prints two URLs and a QR code:

    Display         open on the big screen
    Control panel   scan the QR with the tablet, or type the URL

The QR is also written to control-qr.png, which you can print or show on a
second screen.

    python3 serve.py --no-control          Display only, no relay
    python3 serve.py --host 127.0.0.1      Local only, no tablet
    python3 serve.py --control-token WORD  Require a secret

Anyone who can reach this machine on the network can drive the display. A
foreign web page cannot -- the relay checks the request origin on the
WebSocket handshake -- but a person on the same network who opens the panel
URL can. On a network you do not control, pass --control-token. The token
travels in the URL, so it is visible in the tablet's address bar and its
history: an exhibit lock, not a password.

"""
    else:
        kiosk = """Kiosk mode: driving the display from a tablet
---------------------------------------------
    python3 serve.py --control --host 0.0.0.0 --control-token SECRET

This scene declares no control panel, so the relay is off by default and
there are no authored stops for a panel to show. With --control the script
still prints two URLs and a QR for the second one, and a tap on the tablet
still moves the display.

It is off by default because a folder you double-click should not start
listening for anything that wants to drive it. --host 0.0.0.0 is what makes
the tablet able to reach it at all, and on any network you do not control,
pass --control-token: without one, anything that can reach this machine can
drive the display. The token travels in the URL, so it is visible in the
tablet's address bar and its history -- fine for a LAN exhibit, not a
password.

"""

    readme = f"""Luxar Exported Scene
====================

This folder contains a self-contained Luxar scene viewer.

{about}Quick Start
-----------
    python3 serve.py

This starts a local HTTP server and opens the viewer in your browser. On
Windows the command is `python serve.py`; on macOS and Linux use `python3`,
because a stock macOS has no `python` at all.

Requirements
------------
- Python 3.9 or newer. Nothing to install: the scripts here use only the
  standard library (tested on macOS's own Python 3.9).
- A modern web browser (Chrome, Firefox, Safari, Edge)

Options
-------
    python3 serve.py --port 9000    Use a custom port
    python3 serve.py --no-open      Don't open a browser automatically

If the port is busy the script picks the next free one and prints what it
chose. Press Ctrl+C to stop it.

{kiosk}Folder Structure
----------------
{structure}
Notes
-----
- Do NOT open the viewer's HTML files directly. Both pages
  (viewer/index.html and the touch panel viewer/control.html) need a server:
  the file:// protocol cannot fetch the dataset.
- Nothing here needs `pip install`. serve.py is standard library only, and
  luxar_qr.py -- which it imports to draw the QR -- is a plain file shipped
  in this folder, not a package. Delete it and the server still runs; it
  just prints the panel URL without a QR beside it.
- To share this scene: zip the entire folder and send it
"""
    (output / "README.txt").write_text(readme)
    aprint("Generated README.txt")
