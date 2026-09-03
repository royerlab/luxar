#!/usr/bin/env python3
"""Report whether the README's live-demo count matches the gallery being deployed.

The README states how many demos are live on the site, in two places. Nothing
watched that number: ``sync_demo_counts.py`` owns the *bundled* demo count and
the sample banner, not this one. It has drifted twice.

Report-only by design, and intended to run at DEPLOY time rather than in CI:

* Deploy time is when the fact changes, so the check fires at the only moment
  anyone can act on it. The same check in CI fires later, on an unrelated PR.
* A stale README is a documentation bug. It must never block correct data from
  reaching the site, so this always exits 0.

Counts tiles in the page about to be deployed, not the currently-live one — that
is the number the README will be wrong about a minute later.

Claims are located by their surrounding words rather than by line number, so the
check survives the README being reorganised. If BOTH patterns stop matching it
says so loudly instead of passing on an empty match set, which is the failure
mode that lets a guard quietly stop guarding.

Usage::

    audit_readme_demo_count.py --page dist/index.html
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
TILE_PATTERN = re.compile(r'href="/viewer/index\.html\?src=')
CLAIM_PATTERNS = (
    (re.compile(r"(\d+)\s+live demos"), "intro banner"),
    (re.compile(r"(\d+)\s+demos as interactive scenes"), "docs table row"),
)


def count_tiles(page_html: str) -> int:
    return len(TILE_PATTERN.findall(page_html))


def find_claims(readme_text: str) -> list[tuple[int, str]]:
    """Every stated live-demo count, as ``(number, where)``."""
    claims: list[tuple[int, str]] = []
    for pattern, where in CLAIM_PATTERNS:
        for match in pattern.finditer(readme_text):
            claims.append((int(match.group(1)), where))
    return claims


def audit(
    page_html: str, readme_text: str
) -> tuple[int, list[tuple[int, str]], list[tuple[int, str]]]:
    """Return ``(tiles, claims, stale_claims)``."""
    tiles = count_tiles(page_html)
    claims = find_claims(readme_text)
    stale = [(n, where) for n, where in claims if n != tiles]
    return tiles, claims, stale


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", required=True, type=Path, help="built index.html")
    parser.add_argument("--readme", type=Path, default=README_PATH)
    args = parser.parse_args(argv)

    if not args.page.exists():
        print("readme-count audit: no built page to compare against; skipped")
        return 0
    if not args.readme.exists():
        print("readme-count audit: no README to compare against; skipped")
        return 0

    tiles, claims, stale = audit(args.page.read_text(), args.readme.read_text())
    print(f"readme-count audit: deploying {tiles} tiles")

    if tiles == 0:
        print(
            "  No gallery tiles matched — the page changed, or the tile pattern "
            "needs updating. NOT verified."
        )
        return 0

    if not claims:
        print(
            "  README states no live-demo count in either known spot — it was "
            "reworded, or these patterns need updating. NOT verified."
        )
        return 0

    for number, where in claims:
        status = "ok" if number == tiles else f"STALE (says {number})"
        print(f"  README {where}: {number} -> {status}")
    if stale:
        print(
            f"  WARNING: {len(stale)} README claim(s) disagree with the {tiles} "
            "tiles being deployed. Report only — not blocking this deploy."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
