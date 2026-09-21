#!/usr/bin/env python3
"""Assert the Python, viewer and citation versions describe the same release.

The Python ``__version__`` (CalVer, zero-padded ``YYYY.MM.DD``), the viewer
``package.json`` version (semver-normalized, leading zeros stripped) and
``CITATION.cff`` (zero-padded, plus a ``date-released`` derived from the same
date) are three representations of ONE release. ``scripts/set_version.py``
writes all three; this script is the gate that keeps them from drifting — run
in CI, ``hatch run check``, and the release preflight. The viewer bundle needs
no fourth copy: Vite reads ``package.json`` at build time and injects it as
``VIEWER_VERSION`` (``packages/luxar-viewer/src/version.ts``).

With ``--expect-tag v<version>`` an independently supplied git tag joins the
comparison. The publish workflows are the intended callers because their tag
comes from the triggering ref. The release preflight also passes its derived
tag as a restatement of the tree version.

Exit code 0 if consistent, 1 if they disagree (with a clear diff), 2 on a
read/parse error or a malformed ``--expect-tag``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INIT = REPO / "packages/luxar/src/luxar/__init__.py"
PKG_JSON = REPO / "packages/luxar-viewer/package.json"
CITATION = REPO / "CITATION.cff"

CALVER_RE = re.compile(r'^__version__ = "([^"]*)"', re.MULTILINE)
CFF_VERSION_RE = re.compile(r"^version:[^\S\r\n]*(.*?)[^\S\r\n]*$", re.MULTILINE)
CFF_DATE_RE = re.compile(r"^date-released:[^\S\r\n]*(.*?)[^\S\r\n]*$", re.MULTILINE)


def _normalize_semver(calver: str) -> str:
    """Strip zero-padding so ``2026.06.05`` -> ``2026.6.5`` (npm/semver form)."""
    return ".".join(str(int(p)) for p in calver.split("."))


def _cff_scalar(match: re.Match[str]) -> str:
    """Return a simple YAML scalar without an inline comment or quotes."""
    value = match.group(1).split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _read_python_version() -> str | None:
    """Read and validate the package's zero-padded CalVer."""
    try:
        init_text = INIT.read_text()
    except OSError as exc:
        print(f"error: cannot read {INIT}: {exc}", file=sys.stderr)
        return None
    m = CALVER_RE.search(init_text)
    if not m:
        print(f"error: no __version__ assignment in {INIT}", file=sys.stderr)
        return None
    py_version = m.group(1)
    try:
        if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", py_version):
            raise ValueError
        date.fromisoformat(py_version.replace(".", "-"))
    except ValueError:
        print(
            f"error: {INIT} __version__ {py_version!r} is not valid zero-padded "
            "CalVer YYYY.MM.DD",
            file=sys.stderr,
        )
        return None
    return py_version


def _check_expected_tag(expect_tag: str, py_version: str) -> int:
    """Compare a release tag against the package version. 0 ok, 1 mismatch, 2 malformed."""
    if not expect_tag.startswith("v"):
        print(
            f"error: --expect-tag {expect_tag!r} does not start with 'v'; "
            "release tags are 'v<version>' (e.g. v2026.06.05)",
            file=sys.stderr,
        )
        return 2
    tag_version = expect_tag[1:]
    if tag_version != py_version:
        print(
            "Version mismatch between the release tag and the Python package:\n"
            f"  tag                      = {expect_tag!r}  "
            f"(-> version {tag_version!r})\n"
            f"  Python  __version__      = {py_version!r}\n"
            "The tag selects what gets published, so this disagreement would ship "
            f"a wrong version. Run `make set-version DATE={tag_version}` and promote "
            "it, or tag the version the tree actually declares.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] = ()) -> int:
    # argv defaults to EMPTY, not sys.argv: `main()` is called directly by
    # scripts/tests/test_set_version.py, where sys.argv holds pytest's flags and
    # argparse would exit 2 on them.
    # The CLI entry point below passes sys.argv[1:] explicitly.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--expect-tag",
        metavar="vX.Y.Z",
        help="also require this release tag to match the package version",
    )
    args = ap.parse_args(list(argv))

    py_version = _read_python_version()
    if py_version is None:
        return 2

    if args.expect_tag is not None:
        rc = _check_expected_tag(args.expect_tag, py_version)
        if rc != 0:
            return rc

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
            f"`make set-version DATE={py_version}` to sync, then commit.",
            file=sys.stderr,
        )
        return 1

    # --- CITATION.cff: zero-padded version + a date-released derived from it ---
    # Cheap to check and easy to forget: nothing else reads this file, so drift
    # here is invisible until a citation manager or Zenodo shows the wrong year.
    try:
        cff_text = CITATION.read_text()
    except OSError as exc:
        print(f"error: cannot read {CITATION}: {exc}", file=sys.stderr)
        return 2
    m_ver = CFF_VERSION_RE.search(cff_text)
    m_date = CFF_DATE_RE.search(cff_text)
    if not m_ver or not m_date:
        print(
            f"error: {CITATION.name} needs both a 'version:' and a 'date-released:' "
            "line for the release gate to check them",
            file=sys.stderr,
        )
        return 2
    cff_version, cff_date = _cff_scalar(m_ver), _cff_scalar(m_date)
    expected_date = py_version.replace(".", "-")
    if cff_version != py_version or cff_date != expected_date:
        print(
            "Version mismatch between Python package and CITATION.cff:\n"
            f"  Python  __version__      = {py_version!r}\n"
            f"  CITATION.cff version     = {cff_version!r}  "
            f"(expected {py_version!r})\n"
            f"  CITATION.cff date-released = {cff_date!r}  "
            f"(expected {expected_date!r})\n"
            "They must describe the same release. Run "
            f"`make set-version DATE={py_version}` to sync, then commit.",
            file=sys.stderr,
        )
        return 1

    summary = (
        f"versions consistent: Python {py_version} == viewer {viewer_version} "
        f"== citation {cff_version} ({cff_date})"
    )
    if args.expect_tag is not None:
        summary += f" == tag {args.expect_tag}"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
