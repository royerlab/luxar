"""``luxar env bake``: capture a scene's environment headlessly and attach it.

The capture is the viewer's own (``rendering/environment/`` in the viewer; spec
``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.3): this module only arranges for it to
run unattended — serve the store and the built viewer from one process, drive
a browser to ``?bake-env&probe=…&env-resolution=…`` through the Node Playwright
script ``packages/luxar-viewer/scripts/bake-env.mjs``, collect the container it
hands back, and :func:`~luxar.environment.attach.attach_environment` it.

Playwright is a devDependency of the viewer package, not a Python dependency,
so the driver is Node and this command needs a development checkout (the
viewer's ``node_modules``). From an installed wheel the failure is explicit and
names what is missing; the manual half — ``luxar env attach`` — needs nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import quote

from arbol import aprint, asection

from ..cli.serving import served_store
from ..cli.utils import _find_dev_repo_root, ensure_viewer_built
from .attach import AttachReport, attach_environment
from .container import DEFAULT_RESOLUTION, unpack

#: The Playwright driver, relative to the repository root.
DRIVER_SCRIPT = Path("packages") / "luxar-viewer" / "scripts" / "bake-env.mjs"

__all__ = ["BakeReport", "DEFAULT_RESOLUTION", "DRIVER_SCRIPT", "bake_environment"]

#: The type of the injectable driver: ``(viewer_url, out_path, timeout_s)``.
Driver = Callable[[str, Path, float], None]


@dataclass
class BakeReport:
    """What :func:`bake_environment` produced."""

    store: Path
    container: Path
    resolution: int
    probe: str
    attach: Optional[AttachReport]


def bake_environment(
    store: Union[str, Path],
    *,
    probe: str = "auto",
    resolution: int = DEFAULT_RESOLUTION,
    out: Optional[Union[str, Path]] = None,
    attach: bool = True,
    force: bool = False,
    build: bool = False,
    timeout: float = 300.0,
    driver: Optional[Driver] = None,
) -> BakeReport:
    """Serve ``store`` + the viewer, drive the bake, and attach the result.

    ``probe`` is ``auto`` | ``node:<path>`` | ``x,y,z``; ``resolution`` the cube face
    size in pixels. ``out`` keeps the container file (default: a temporary file
    that is removed after a successful attach). ``build`` lets a stale viewer dist
    be rebuilt first (never by default — a bake must not kick off a multi-minute
    build). ``driver`` replaces the Node script (tests).
    """
    store_path = Path(store)
    if not store_path.exists():
        raise FileNotFoundError(f"Scene not found: {store_path}")
    if resolution < 16 or resolution > 1024:
        raise ValueError(f"--resolution must be between 16 and 1024, got {resolution}")
    if not ensure_viewer_built(auto_build=build):
        raise RuntimeError(
            "The viewer is not built. In a development checkout run "
            "`cd packages/luxar-viewer && pnpm build`, or pass --build."
        )
    run_driver = driver or _node_driver()

    out_path = Path(out) if out is not None else None
    keep = out_path is not None
    if out_path is None:
        handle, tmp_name = tempfile.mkstemp(prefix="luxar-env-", suffix=".env.bin")
        os.close(handle)
        out_path = Path(tmp_name)

    with asection(f"Baking environment for {store_path.name}"):
        with served_store(store_path) as served:
            url = (
                f"{served.viewer_url}?src={quote(served.data_url, safe=':/')}"
                f"&debug&bake-env&probe={quote(probe, safe=':,/')}"
                f"&env-resolution={int(resolution)}"
            )
            aprint(f"driving {url}")
            run_driver(url, out_path, timeout)

        header, _faces = unpack(out_path.read_bytes())
        aprint(
            f"container: {out_path} ({out_path.stat().st_size} bytes, "
            f"{header['resolution']}px, probe {header['probe']})"
        )
        report: Optional[AttachReport] = None
        if attach:
            report = attach_environment(store_path, out_path, force=force)
            if not keep:
                out_path.unlink(missing_ok=True)
        return BakeReport(
            store=store_path,
            container=out_path,
            resolution=int(header["resolution"]),
            probe=str(header["probe"].get("spec", probe))
            if isinstance(header["probe"], dict)
            else str(header["probe"]),
            attach=report,
        )


def _node_driver() -> Driver:
    """The real driver: ``node packages/luxar-viewer/scripts/bake-env.mjs``."""
    repo_root = _find_dev_repo_root()
    script = repo_root / DRIVER_SCRIPT if repo_root else None
    if script is None or not script.exists():
        raise RuntimeError(
            "`luxar env bake` drives the viewer with Playwright, which is a devDependency "
            "of the viewer package: it needs a development checkout with "
            f"{DRIVER_SCRIPT} and the viewer's node_modules. From an installed wheel, "
            "bake on a checkout and attach the result with `luxar env attach`."
        )
    node = shutil.which("node")
    if node is None:
        raise RuntimeError(
            "`node` was not found on PATH; the bake driver needs Node.js."
        )

    def run(url: str, out_path: Path, timeout_s: float) -> None:
        cmd = [
            node,
            str(script),
            "--url",
            url,
            "--out",
            str(out_path),
            "--timeout",
            str(int(timeout_s * 1000)),
        ]
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd,
            cwd=str(script.parent.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s + 60,
            check=False,
        )
        for line in result.stdout.splitlines():
            aprint(f"driver: {line}")
        if result.returncode != 0:
            raise RuntimeError(
                f"The bake driver exited with {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    return run
