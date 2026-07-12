#!/usr/bin/env python3
"""Assert the Python and viewer versions describe the same release.

The Python ``__version__`` (CalVer, zero-padded ``YYYY.MM.DD``) and the viewer
``package.json`` version (semver-normalized, leading zeros stripped) are two
representations of ONE release. ``scripts/set_version.py`` writes both; this
script is the gate that keeps them from drifting — run in CI, ``hatch run
check``, and the release preflight.

Exit code 0 if consistent, 1 if they disagree (with a clear diff), 2 on a
read/parse error.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INIT = REPO / "packages/luxar/src/luxar/__init__.py"
PKG_JSON = REPO / "packages/luxar-viewer/package.json"

CALVER_RE = re.compile(r'^__version__ = "([^"]*)"', re.MULTILINE)


def _normalize_semver(calver: str) -> str:
    """Strip zero-padding so ``2026.06.05`` -> ``2026.6.5`` (npm/semver form)."""
    return ".".join(str(int(p)) for p in calver.split("."))


def main() -> int:
    try:
        init_text = INIT.read_text()
    except OSError as exc:
        print(f"error: cannot read {INIT}: {exc}", file=sys.stderr)
        return 2
    m = CALVER_RE.search(init_text)
    if not m:
        print(f"error: no __version__ assignment in {INIT}", file=sys.stderr)
        return 2
    py_version = m.group(1)

    try:
        pkg = json.loads(PKG_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {PKG_JSON}: {exc}", file=sys.stderr)
        return 2
    viewer_version = pkg.get("version")
    if not viewer_version:
        print(f"error: no 'version' field in {PKG_JSON}", file=sys.stderr)
        return 2

    expected_viewer = _normalize_semver(py_version)
    if viewer_version != expected_viewer:
        print(
            "Version mismatch between Python package and viewer:\n"
            f"  Python  __version__      = {py_version!r}  "
            f"(-> semver {expected_viewer!r})\n"
            f"  Viewer  package.json     = {viewer_version!r}\n"
            "They must describe the same release. Run "
            f"`make set-version {py_version}` to sync, then commit.",
            file=sys.stderr,
        )
        return 1

    print(f"versions consistent: Python {py_version} == viewer {viewer_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
