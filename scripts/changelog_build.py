#!/usr/bin/env python3
"""Assemble ``changelog.d/*.md`` fragments into CHANGELOG.md, then delete them.

Each fragment file is ONE narrative changelog entry — the very same ``#### Title``
+ prose block an author would otherwise write by hand under the current month in
CHANGELOG.md. Writing them as separate per-PR files is what removes CHANGELOG.md
as a merge-conflict magnet: every PR used to edit CHANGELOG.md (≈a third of them),
so under a moving main each rebase re-conflicted on it. With fragments, a PR touches
only its own new file in ``changelog.d/`` and CHANGELOG.md changes exactly once — at
release assembly — so it never conflicts.

Workflow
--------
* While working on a PR, add ``changelog.d/<PR-or-issue-number>.md`` containing the
  entry exactly as it should read in CHANGELOG.md (a ``#### Title`` line followed by
  one or more prose paragraphs — the house style; see ``changelog.d/README.md``).
* At release-prep, run ``make changelog`` (or this script) to fold every fragment
  into CHANGELOG.md under ``## [Unreleased]`` → ``### <Month Year>`` and delete the
  fragments. Commit that via the normal version-bump PR, then tag the release.

Usage::

    python scripts/changelog_build.py --draft                 # preview; change nothing
    python scripts/changelog_build.py                         # fold in + delete fragments
    python scripts/changelog_build.py --month "August 2026"   # pin the month heading

Stdlib-only (no dependency, no build env needed).
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAG_DIR = REPO_ROOT / "changelog.d"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"

# Files in changelog.d/ that are NOT entries.
NON_FRAGMENTS = {"README.md", ".gitkeep"}


def _natural_key(p: Path):
    """Sort by the leading integer in the filename when present (so #900 < #1000),
    then lexically — numeric fragments (named after a PR/issue) come first, in order."""
    m = re.match(r"(\d+)", p.stem)
    return (0, int(m.group(1))) if m else (1, p.name)


def _fragments() -> list[Path]:
    if not FRAG_DIR.is_dir():
        return []
    return sorted(
        (p for p in FRAG_DIR.glob("*.md") if p.name not in NON_FRAGMENTS),
        key=_natural_key,
    )


def _read_block(p: Path) -> str:
    text = p.read_text(encoding="utf-8").strip("\n")
    if not text.strip():
        raise SystemExit(f"✗ fragment {p.name} is empty")
    first = text.lstrip().splitlines()[0]
    if not first.startswith("#### "):
        raise SystemExit(
            f"✗ fragment {p.name} must start with a '#### Title' heading "
            f"(house style); got: {first!r}"
        )
    return text


def _fold(changelog: str, month: str, blocks: list[str]) -> str:
    lines = changelog.splitlines(keepends=False)
    try:
        u = next(i for i, ln in enumerate(lines) if ln.strip() == UNRELEASED)
    except StopIteration:
        raise SystemExit(f"✗ '{UNRELEASED}' header not found in CHANGELOG.md")

    entry = "\n\n".join(blocks)
    month_hdr = f"### {month}"

    # Look for an existing '### <month>' ANYWHERE between Unreleased and the next
    # '## ' header. Every month lives under Unreleased, so the requested one is not
    # necessarily the newest: `--month "July 2026"` must land in the existing July
    # section rather than mint a second one above August.
    j = u + 1
    existing = None
    while j < len(lines) and not lines[j].startswith("## "):
        if lines[j].strip() == month_hdr:
            existing = j
            break
        j += 1

    if existing is not None:
        # Insert new entries at the TOP of that month (newest-first), right
        # after the header and its blank line.
        insert_at = existing + 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        new = lines[:insert_at] + [entry, ""] + lines[insert_at:]
    else:
        # No current-month section yet: create one directly under Unreleased.
        insert_at = u + 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        new = lines[:insert_at] + [month_hdr, "", entry, ""] + lines[insert_at:]

    return "\n".join(new) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draft", action="store_true", help="preview only; change nothing")
    ap.add_argument(
        "--month",
        default=None,
        help='month heading, e.g. "August 2026" (default: current)',
    )
    args = ap.parse_args()

    frags = _fragments()
    if not frags:
        print("• no changelog fragments in changelog.d/ — nothing to assemble")
        return 0

    month = args.month or datetime.datetime.now().strftime("%B %Y")
    blocks = [_read_block(p) for p in frags]

    if not CHANGELOG.is_file():
        raise SystemExit("✗ CHANGELOG.md not found")
    folded = _fold(CHANGELOG.read_text(encoding="utf-8"), month, blocks)

    names = ", ".join(p.name for p in frags)
    if args.draft:
        print(
            f"• DRAFT: would fold {len(frags)} fragment(s) into '### {month}': {names}"
        )
        print("─" * 72)
        print("\n\n".join(blocks))
        print("─" * 72)
        return 0

    CHANGELOG.write_text(folded, encoding="utf-8")
    for p in frags:
        p.unlink()
    print(
        f"✓ folded {len(frags)} fragment(s) into CHANGELOG.md under '### {month}' and removed them: {names}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
