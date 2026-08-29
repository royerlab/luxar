#!/usr/bin/env python3
"""Set the Luxar release version (CalVer YYYY.MM.DD).

Updates the single source of truth — ``__version__`` in
``packages/luxar/src/luxar/__init__.py`` — and keeps the viewer's
``package.json`` and the root ``CITATION.cff`` in sync. Because ``main`` is
branch-protected, this only edits files locally; commit the change on a branch
and open a PR, then tag the release with ``make release`` once it has merged
with CI green.

Usage:
    python scripts/set_version.py            # today, zero-padded YYYY.MM.DD
    python scripts/set_version.py 2026.06.29 # explicit date

Note on formats:
    * Python/PyPI (PEP 440) and the git tag use the zero-padded form (2026.06.29).
    * npm/semver forbids leading zeros, so package.json gets the normalized form
      (2026.6.29). They refer to the same release; this divergence is expected.
    * CITATION.cff carries the zero-padded form, matching the git tag, plus a
      ``date-released`` derived from the same date.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INIT = REPO / "packages/luxar/src/luxar/__init__.py"
PKG_JSON = REPO / "packages/luxar-viewer/package.json"
CITATION = REPO / "CITATION.cff"

CALVER_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
CFF_VERSION_LINE_RE = re.compile(r"^version:[^\S\r\n]*.*$", re.MULTILINE)
CFF_DATE_LINE_RE = re.compile(r"^date-released:[^\S\r\n]*.*$", re.MULTILINE)


def main(argv: list[str]) -> int:
    version = argv[1] if len(argv) > 1 else datetime.date.today().strftime("%Y.%m.%d")
    if not CALVER_RE.match(version):
        print(
            f"error: '{version}' is not CalVer YYYY.MM.DD (zero-padded, e.g. 2026.06.29)",
            file=sys.stderr,
        )
        return 2
    try:
        datetime.date.fromisoformat(version.replace(".", "-"))
    except ValueError:
        print(
            f"error: '{version}' is not CalVer YYYY.MM.DD (zero-padded, e.g. 2026.06.29)",
            file=sys.stderr,
        )
        return 2

    cff = CITATION.read_text() if CITATION.exists() else None
    if cff is not None and (
        not CFF_VERSION_LINE_RE.search(cff) or not CFF_DATE_LINE_RE.search(cff)
    ):
        print(
            f"error: {CITATION.name} is missing a 'version:' or 'date-released:' "
            "line to stamp — add them (see check_version_consistency.py)",
            file=sys.stderr,
        )
        return 1

    # --- Python __version__ (zero-padded, authoritative) ---
    text = INIT.read_text()
    new_text, n = re.subn(
        r'^__version__ = "[^"]*"',
        f'__version__ = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        print(
            f"error: could not find __version__ assignment in {INIT}", file=sys.stderr
        )
        return 1
    changed = new_text != text
    INIT.write_text(new_text)
    print(
        f"{'updated' if changed else 'unchanged'}: {INIT.relative_to(REPO)} -> {version}"
    )

    # --- viewer package.json (semver-normalized: strip leading zeros) ---
    semver = ".".join(str(int(p)) for p in version.split("."))
    if PKG_JSON.exists():
        data = json.loads(PKG_JSON.read_text())
        old = data.get("version")
        data["version"] = semver
        # Preserve 2-space indentation and trailing newline (matches the repo).
        PKG_JSON.write_text(json.dumps(data, indent=2) + "\n")
        print(
            f"{'updated' if old != semver else 'unchanged'}: "
            f"{PKG_JSON.relative_to(REPO)} -> {semver}"
        )
    else:
        print(
            f"warning: {PKG_JSON.relative_to(REPO)} not found; skipped viewer version",
            file=sys.stderr,
        )

    # --- CITATION.cff (zero-padded, plus the release date) ---
    # Not merely cosmetic: this is what a citation manager and Zenodo read, and
    # nothing else stamps it — before this it was a launch-day hand-edit that the
    # release runbook never mentioned. `check_version_consistency.py` gates it.
    if cff is not None:
        original_cff = cff
        released = version.replace(".", "-")
        cff, n_ver = CFF_VERSION_LINE_RE.subn(f'version: "{version}"', cff, count=1)
        cff, n_date = CFF_DATE_LINE_RE.subn(
            f'date-released: "{released}"', cff, count=1
        )
        assert n_ver == 1 and n_date == 1
        CITATION.write_text(cff)
        print(
            f"{'updated' if cff != original_cff else 'unchanged'}: "
            f"{CITATION.relative_to(REPO)} -> {version} ({released})"
        )
    else:
        print(
            f"warning: {CITATION.name} not found; skipped citation version",
            file=sys.stderr,
        )

    print()
    print("Next steps (main is branch-protected — no direct push):")
    print(f"  git switch -c release/v{version}")
    print(f"  git commit -am 'release: v{version}'")
    print("  gh pr create --fill   # merge once CI is green")
    print(
        "  make release          # tags v{0} and triggers the PyPI publish".format(
            version
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
