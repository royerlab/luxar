#!/usr/bin/env python3
"""Set the Luxar release version (CalVer YYYY.MM.DD).

Updates the single source of truth — ``__version__`` in
``packages/luxar/src/luxar/__init__.py`` — and keeps the viewer's
``package.json`` in sync. Because ``main`` is branch-protected, this only edits
files locally; commit the change on a branch and open a PR, then tag the release
with ``make release`` once it has merged with CI green.

Usage:
    python scripts/set_version.py            # today, zero-padded YYYY.MM.DD
    python scripts/set_version.py 2026.06.29 # explicit date

Note on formats:
    * Python/PyPI (PEP 440) and the git tag use the zero-padded form (2026.06.29).
    * npm/semver forbids leading zeros, so package.json gets the normalized form
      (2026.6.29). They refer to the same release; this divergence is expected.
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

CALVER_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")


def main(argv: list[str]) -> int:
    version = argv[1] if len(argv) > 1 else datetime.date.today().strftime("%Y.%m.%d")
    if not CALVER_RE.match(version):
        print(
            f"error: '{version}' is not CalVer YYYY.MM.DD (zero-padded, e.g. 2026.06.29)",
            file=sys.stderr,
        )
        return 2

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
        print(f"error: could not find __version__ assignment in {INIT}", file=sys.stderr)
        return 1
    changed = new_text != text
    INIT.write_text(new_text)
    print(f"{'updated' if changed else 'unchanged'}: {INIT.relative_to(REPO)} -> {version}")

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
        print(f"warning: {PKG_JSON.relative_to(REPO)} not found; skipped viewer version", file=sys.stderr)

    print()
    print("Next steps (main is branch-protected — no direct push):")
    print(f"  git switch -c release/v{version}")
    print(f"  git commit -am 'release: v{version}'")
    print("  gh pr create --fill   # merge once CI is green")
    print("  make release          # tags v{0} and triggers the PyPI publish".format(version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
