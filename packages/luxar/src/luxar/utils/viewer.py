"""Viewer launch and stable demo-port helpers."""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Optional, Union

from arbol import aprint

from .process import run_child_process

# its own `?src=` data URL.
_DEMO_DATA_PORT_BASE = 8001
_DEMO_VIEWER_PORT_BASE = 5200
_DEMO_PORT_SLOTS = 499


def demo_ports(output_path: Union[str, Path]) -> tuple[int, int]:
    """Stable per-dataset ``(data_port, viewer_port)`` pair for demo serving.

    Derived from the dataset's file name, so re-running the same demo lands on
    the same URL (an old tab for that demo stays valid after a reload) while
    different demos get different URLs. The two slots are drawn from
    INDEPENDENT hash digits: with one shared slot, ~1/6 of the bundled demo
    names collided pairwise (birthday at 499 slots), and an identical
    ``(data, viewer)`` pair reproduces the very same-URL stale-tab trap this
    derivation exists to prevent. Independent slots square the pair space
    (~249k), making a full-pair collision vanishingly rare.

    The dataset NAME is the identity: two datasets that share a file name in
    different directories deliberately share a pair (that is what makes a
    demo's URL stable across output directories), so re-running the same demo
    from two workspaces still hands the second one a shifted port.
    """
    digest = zlib.crc32(Path(output_path).name.encode("utf-8"))
    data_slot = digest % _DEMO_PORT_SLOTS
    viewer_slot = (digest // _DEMO_PORT_SLOTS) % _DEMO_PORT_SLOTS
    return _DEMO_DATA_PORT_BASE + data_slot, _DEMO_VIEWER_PORT_BASE + viewer_slot


def _serve_command(
    output_path: Union[str, Path],
    open_browser: bool,
    serve_args: Optional[list[str]],
) -> list[str]:
    """Build the ``luxar serve`` argv for a demo, with derived default ports.

    The derived pair is appended only when ``serve_args`` doesn't already pin
    the respective option, so a demo passing an explicit ``--port`` /
    ``--viewer-port`` keeps full control.
    """
    args = list(serve_args or [])
    cmd = [sys.executable, "-m", "luxar", "serve", str(output_path), "--viewer"]
    cmd.extend(args)
    data_port, viewer_port = demo_ports(output_path)
    # `serve` also spells the data port `-p` (separate or attached, `-p 9` /
    # `-p9`), and Click silently keeps the LAST occurrence of a repeated
    # option — so missing a pinned spelling here would OVERRIDE the demo's
    # explicit choice with the derived port. No other serve option starts
    # with `-p`.
    data_pinned = any(
        a == "--port"
        or a.startswith("--port=")
        or (a.startswith("-p") and not a.startswith("--"))
        for a in args
    )
    if not data_pinned:
        cmd.extend(["--port", str(data_port)])
    if not any(a == "--viewer-port" or a.startswith("--viewer-port=") for a in args):
        cmd.extend(["--viewer-port", str(viewer_port)])
    if open_browser:
        cmd.append("--open")
    return cmd


def launch_viewer(
    output_path: Union[str, Path],
    open_browser: bool = True,
    serve_args: Optional[list[str]] = None,
) -> None:
    """Launch the Luxar viewer to display a dataset.

    This function uses sys.executable to ensure it works regardless of how
    the demo script was launched (hatch run, hatch shell, conda, etc.).

    Each dataset serves on its own stable derived port pair (see
    :func:`demo_ports`) rather than everyone contending for 8000/5173.

    Args:
        output_path: Path to the .zarr dataset to view
        open_browser: Whether to automatically open a browser window
        serve_args: Extra arguments forwarded verbatim to ``luxar serve`` — e.g.
            ``["--profile", "3g"]`` for the network-simulation demo. These let a
            demo drive server-side behaviour (bandwidth throttling, latency,
            packet loss) that ``serve`` already supports. An explicit
            ``--port`` / ``--viewer-port`` here overrides the derived pair.

    Raises:
        SystemExit: If the viewer fails to launch
    """
    cmd = _serve_command(output_path, open_browser, serve_args)

    # isolate_group=False: keep the `luxar serve` child in this process's group
    # so an ancestor's group-kill (from `luxar demo run`) still cascades to it.
    # When the demo is run directly (`python -m luxar.demos.demo_X`), the
    # helper's per-PID teardown still cleanly kills a hung server on Ctrl-C.
    code = run_child_process(cmd, label="Luxar viewer", isolate_group=False)
    if code in (0, 130):  # clean exit or user Ctrl-C
        return
    # Genuine failure. returncode 1 usually means luxar/module import failure;
    # anything else typically means the viewer isn't built.
    if code == 1:
        aprint("\n❌ Error running luxar (exit 1).")
        aprint("Make sure luxar is installed in your Python environment.")
        aprint("If using hatch: run this demo with 'hatch run python <demo.py>'")
    else:
        aprint(f"\n❌ Error: luxar serve exited with code {code}.")
        aprint("Make sure the viewer is built: cd packages/luxar-viewer && pnpm build")
    sys.exit(1)
