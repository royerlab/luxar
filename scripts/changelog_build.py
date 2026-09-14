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

Each fragment is filed under the month it was WRITTEN, taken from the commit that
added it, not under the month the fold happens to run. Those are usually not the
same: at the time this was written, 440 pending fragments spanned two months and
all 440 would have been filed under the current one, mis-dating 408 of them into
a single unnavigable heading.

``--release`` closes the other half: it renames ``## [Unreleased]`` to a version
section and opens a fresh empty one, so the file accumulates release history
instead of one ever-growing ``[Unreleased]``. The version is read from
``__version__`` — the same single source of truth ``scripts/set_version.py``
writes — because the release version is a DATE that is deliberately set last.

Usage::

    python scripts/changelog_build.py --check                 # validate fragments only
    python scripts/changelog_build.py --draft                 # preview; change nothing
    python scripts/changelog_build.py                         # fold in + delete fragments
    python scripts/changelog_build.py --month "August 2026"   # pin every entry to one month
    python scripts/changelog_build.py --release               # cut [Unreleased] → [<version>]

Stdlib-only (no dependency, no build env needed).
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAG_DIR = REPO_ROOT / "changelog.d"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
#: Single source of truth for the version, written by scripts/set_version.py.
INIT = REPO_ROOT / "packages/luxar/src/luxar/__init__.py"

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


def _validate_block(p: Path) -> str | None:
    """Return a fragment validation error, or None when it is valid."""
    try:
        text = p.read_text(encoding="utf-8").strip("\n")
    except UnicodeDecodeError:
        return f"fragment {p.name} is not valid UTF-8"
    except OSError as exc:
        return f"fragment {p.name} could not be read: {exc}"

    if not text.strip():
        return f"fragment {p.name} is empty"
    first = text.lstrip().splitlines()[0]
    if not first.startswith("#### "):
        return (
            f"fragment {p.name} must start with a '#### Title' heading "
            f"(house style); got: {first!r}"
        )
    return None


def _read_block(p: Path) -> str:
    failure = _validate_block(p)
    if failure:
        raise SystemExit(f"✗ {failure}")
    try:
        return p.read_text(encoding="utf-8").strip("\n")
    except UnicodeDecodeError:
        raise SystemExit(f"✗ fragment {p.name} is not valid UTF-8") from None
    except OSError as exc:
        raise SystemExit(f"✗ fragment {p.name} could not be read: {exc}") from None


def _read_blocks(frags: list[Path]) -> dict[Path, str]:
    """Read every fragment, reporting all invalid entries in one failure."""
    blocks: dict[Path, str] = {}
    failures: list[str] = []
    for fragment in frags:
        failure = _validate_block(fragment)
        if failure:
            failures.append(failure)
            continue
        blocks[fragment] = _read_block(fragment)

    if failures:
        noun = "fragment" if len(failures) == 1 else "fragments"
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"✗ {len(failures)} invalid changelog {noun}:\n{details}")
    return blocks


def _authored_month(p: Path) -> str | None:
    """Month heading for the commit that ADDED ``p``, or None if git cannot say.

    The oldest add-commit, not the newest: a fragment that was deleted and
    restored should keep the month it was written in.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--format=%ad",
                "--date=format:%B %Y",
                # Absolute: a path outside the repo makes git return
                # nothing, which becomes the deliberate "cannot date"
                # failure rather than a ValueError traceback.
                "--",
                str(p),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _group_by_month(frags: list[Path], pinned: str | None) -> dict[str, list[Path]]:
    """Map month heading → fragments, oldest month first.

    Fails rather than guessing. A fragment whose month git cannot supply -- an
    unstaged file, a source tarball with no history -- would otherwise be filed
    under whatever month the release happens to run in, which is exactly the
    mis-dating this exists to prevent. ``--month`` is the explicit override.
    """
    if pinned:
        return {pinned: list(frags)}

    groups: dict[str, list[Path]] = {}
    unknown: list[str] = []
    for p in frags:
        month = _authored_month(p)
        if month is None:
            unknown.append(p.name)
            continue
        groups.setdefault(month, []).append(p)

    if unknown:
        raise SystemExit(
            f"✗ cannot determine the authoring month of {len(unknown)} fragment(s) "
            f"from git: {', '.join(unknown[:8])}"
            f"{' …' if len(unknown) > 8 else ''}\n"
            f"  Commit them first, or pin every entry to one heading with "
            f'--month "<Month Year>".'
        )

    return dict(
        sorted(
            groups.items(), key=lambda kv: datetime.datetime.strptime(kv[0], "%B %Y")
        )
    )


def _current_version() -> str:
    """``__version__`` from the package -- what ``set_version.py`` writes."""
    m = re.search(r'^__version__ = "([^"]+)"', INIT.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"✗ no __version__ assignment found in {INIT}")
    return m.group(1)


def _release_date(version: str) -> str:
    """The date to stamp beside ``version``.

    Luxar versions are CalVer -- `make set-version DATE=YYYY.MM.DD` -- so the
    version already IS the release date. Deriving the stamp from it keeps the
    header from contradicting itself when the cut runs on a different day from
    the bump, which it usually does: the bump lands in a PR and the cut follows
    review. Falls back to today for a version that is not a date.
    """
    m = re.fullmatch(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", version)
    if m:
        year, month, day = (int(g) for g in m.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _cut_release(changelog: str, version: str, date: str) -> str:
    """Rename ``## [Unreleased]`` to a version section, and open a fresh one."""
    lines = changelog.splitlines()
    try:
        u = next(i for i, ln in enumerate(lines) if ln.strip() == UNRELEASED)
    except StopIteration:
        raise SystemExit(f"✗ '{UNRELEASED}' header not found in CHANGELOG.md") from None
    # Renaming this header in place is what puts the new section ABOVE any
    # previous ones: everything already under [Unreleased] belongs to the
    # version being cut, and earlier releases are further down the file.
    lines[u] = f"## [{version}] - {date}"
    return "\n".join(lines[:u] + [UNRELEASED, ""] + lines[u:]) + "\n"


def _fold(changelog: str, month: str, blocks: list[str]) -> str:
    lines = changelog.splitlines(keepends=False)
    try:
        u = next(i for i, ln in enumerate(lines) if ln.strip() == UNRELEASED)
    except StopIteration:
        raise SystemExit(f"✗ '{UNRELEASED}' header not found in CHANGELOG.md") from None

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


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--draft", action="store_true", help="preview only; change nothing")
    ap.add_argument(
        "--check",
        action="store_true",
        help="validate every fragment without reading git history or CHANGELOG.md",
    )
    ap.add_argument(
        "--month",
        default=None,
        help='pin EVERY entry to one heading, e.g. "August 2026" '
        "(default: each fragment is filed under the month it was written)",
    )
    ap.add_argument(
        "--release",
        action="store_true",
        help="rename '## [Unreleased]' to the current __version__ and open a "
        "fresh empty one; refuses while fragments are still pending",
    )
    return ap, ap.parse_args()


def _run_check(ap: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Validate fragments without touching git history or CHANGELOG.md."""
    if args.draft or args.month or args.release:
        ap.error("--check cannot be combined with --draft, --month, or --release")
    blocks = _read_blocks(_fragments())
    noun = "fragment" if len(blocks) == 1 else "fragments"
    print(f"✓ validated {len(blocks)} changelog {noun}")
    return 0


def _run_fold(args: argparse.Namespace) -> int:
    """Fold all pending fragments into CHANGELOG.md."""
    frags = _fragments()
    if not frags:
        print("• no changelog fragments in changelog.d/ — nothing to assemble")
        return 0

    blocks = _read_blocks(frags)
    groups = _group_by_month(frags, args.month)
    # Oldest month first: `_fold` mints a new section directly under
    # [Unreleased], so folding in chronological order leaves the newest on top.
    folded = CHANGELOG.read_text(encoding="utf-8")
    summary = []
    for month, month_frags in groups.items():
        month_blocks = [blocks[p] for p in month_frags]
        folded = _fold(folded, month, month_blocks)
        summary.append(f"{len(month_frags)} into '### {month}'")

    detail = "; ".join(summary)
    if args.draft:
        print(f"• DRAFT: would fold {len(frags)} fragment(s) — {detail}")
        for month, month_frags in groups.items():
            print(f"    {month}: {', '.join(p.name for p in month_frags)}")
        print("─" * 72)
        for month, month_frags in groups.items():
            print(f"### {month}\n")
            print("\n\n".join(blocks[p] for p in month_frags))
            print()
        print("─" * 72)
        return 0

    CHANGELOG.write_text(folded, encoding="utf-8")
    for p in frags:
        p.unlink()
    print(f"✓ folded {len(frags)} fragment(s) and removed them — {detail}")
    return 0


def main() -> int:
    ap, args = _parse_args()
    if args.check:
        return _run_check(ap, args)
    if not CHANGELOG.is_file():
        raise SystemExit("✗ CHANGELOG.md not found")
    if args.release:
        return _run_release(args.draft)
    return _run_fold(args)


def _run_release(draft: bool) -> int:
    """Cut ``[Unreleased]`` into a version section."""
    pending = _fragments()
    if pending:
        raise SystemExit(
            f"✗ {len(pending)} changelog fragment(s) are still pending. Fold them "
            f"first (`make changelog`) — cutting the release now would ship them "
            f"in the NEXT version instead of this one."
        )
    version = _current_version()
    date = _release_date(version)
    text = CHANGELOG.read_text(encoding="utf-8")
    if f"## [{version}]" in text:
        raise SystemExit(
            f"✗ CHANGELOG.md already has a '## [{version}]' section. Bump "
            f"__version__ (`make set-version DATE=…`) before cutting again."
        )
    cut = _cut_release(text, version, date)
    if draft:
        print(f"• DRAFT: would cut '## [Unreleased]' → '## [{version}] - {date}'")
        return 0
    CHANGELOG.write_text(cut, encoding="utf-8")
    print(f"✓ cut '## [Unreleased]' → '## [{version}] - {date}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
