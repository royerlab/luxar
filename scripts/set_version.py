#!/usr/bin/env python3
"""Set the Luxar release version (CalVer YYYY.MM.DD).

Updates the single source of truth — ``__version__`` in
``packages/luxar/src/luxar/__init__.py`` — and keeps the viewer's
``package.json`` and the root ``CITATION.cff`` in sync. Because ``main`` is
branch-protected, this only edits files locally; commit the change on a branch
and open a PR, then switch to ``main`` and tag the release with ``make release``
once the bump has been promoted with CI green.

Usage:
    python scripts/set_version.py            # today, zero-padded YYYY.MM.DD
    python scripts/set_version.py 2026.06.29 # explicit date

Note on formats:
    * The files and the git tag use the zero-padded form (2026.06.29):
      ``__version__``, ``CITATION.cff`` and ``v2026.06.29``.
    * BOTH registries strip the zeros, and they do it the same way. npm/semver
      forbids leading zeros, so ``package.json`` is written normalized
      (2026.6.29) — but PEP 440 normalizes identically, so the PyPI metadata
      reads 2026.6.29 as well, whatever this file says. (Measured:
      ``Version("2026.10.01")`` is ``2026.10.1``.) Only the tag and the
      in-repo text keep the padding. All of it is one release.
    * ``CITATION.cff`` also carries a ``date-released`` derived from the date.

This script is one step of release prep, not the whole of it. The order is
``make changelog`` (fold the fragments) -> this script -> ``make
changelog-release`` (cut the section, which is named after ``__version__`` and
so must run after the bump). ``scripts/release.sh`` now refuses to tag when
either changelog step has been skipped.
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


def _package_payload(semver: str) -> tuple[str | None, object | None] | None:
    if not PKG_JSON.exists():
        return None, None
    try:
        data = json.loads(PKG_JSON.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"error: cannot parse {PKG_JSON.relative_to(REPO)}: {exc}\n"
            "  Nothing has been written; fix the JSON and re-run.",
            file=sys.stderr,
        )
        return None
    old_version = data.get("version")
    data["version"] = semver
    return json.dumps(data, indent=2) + "\n", old_version


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

    # ---------------------------------------------------------------------
    # PARSE EVERYTHING FIRST, WRITE NOTHING.
    #
    # This used to write __init__.py and only then parse package.json, so a
    # malformed package.json raised a JSONDecodeError with the Python version
    # already bumped on disk: a half-stamped tree that
    # check_version_consistency.py would then reject, leaving the operator to
    # work out which file had moved. Every read and every regex below is
    # validated before the first byte is written.
    # ---------------------------------------------------------------------

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

    # --- viewer package.json (semver-normalized: strip leading zeros) ---
    semver = ".".join(str(int(p)) for p in version.split("."))
    package = _package_payload(semver)
    if package is None:
        return 1
    pkg_payload, pkg_old = package

    # --- CITATION.cff substitutions (validated above; computed here) ---
    cff_payload = None
    released = version.replace(".", "-")
    if cff is not None:
        staged, n_ver = CFF_VERSION_LINE_RE.subn(f'version: "{version}"', cff, count=1)
        staged, n_date = CFF_DATE_LINE_RE.subn(
            f'date-released: "{released}"', staged, count=1
        )
        if n_ver != 1 or n_date != 1:
            print(
                f"error: could not stamp {CITATION.name} "
                f"(version matches={n_ver}, date-released matches={n_date}).\n"
                "  Nothing has been written.",
                file=sys.stderr,
            )
            return 1
        cff_payload = staged

    # ---------------------------------------------------------------------
    # COMMIT: everything parsed cleanly, so write.
    # ---------------------------------------------------------------------
    changed = new_text != text
    INIT.write_text(new_text)
    print(
        f"{'updated' if changed else 'unchanged'}: {INIT.relative_to(REPO)} -> {version}"
    )

    if pkg_payload is not None:
        PKG_JSON.write_text(pkg_payload)
        print(
            f"{'updated' if pkg_old != semver else 'unchanged'}: "
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
    if cff_payload is not None:
        CITATION.write_text(cff_payload)
        print(
            f"{'updated' if cff_payload != cff else 'unchanged'}: "
            f"{CITATION.relative_to(REPO)} -> {version} ({released})"
        )
    else:
        print(
            f"warning: {CITATION.name} not found; skipped citation version",
            file=sys.stderr,
        )

    print()
    print("Next steps (main is branch-protected — no direct push):")
    print(
        "  make changelog        # fold changelog.d/*.md into CHANGELOG.md, if not done yet"
    )
    print("  make changelog-release  # cut the Unreleased section as " + version)
    print(
        "                        # (after this script: the cut is named for __version__)"
    )
    print(f"  git switch -c release/v{version}")
    print(f"  git commit -am 'release: v{version}'")
    # `--base dev` is explicit on purpose: without it `gh` targets the repo's
    # DEFAULT branch, so this hint would silently change meaning if the default
    # ever moves to `main` (which is protected and would reject the PR anyway).
    print("  gh pr create --base dev --fill   # merge once CI is green")
    print(
        "  make release          # switch to main after the bump is promoted; "
        "tags v{0} and publishes".format(version)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
