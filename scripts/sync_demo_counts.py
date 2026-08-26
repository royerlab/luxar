#!/usr/bin/env python3
"""Synchronize the live demo and example counts in repository documentation.

Usage::

    hatch run sync-demo-counts
    hatch run check-demo-counts

The script rewrites only the four current count sites. Historical changelog
counts are intentionally outside its scope. ``--check`` renders in memory and
exits 1 when a committed file has drifted, without writing anything.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Callable, Match, Pattern

from luxar.demos.registry import DemoMetaError, iter_demos

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
CLAUDE = REPO / "CLAUDE.md"
VISUALIZATION_SKILL = REPO / ".agents/skills/luxar-visualization/SKILL.md"
EXAMPLES_DIR = REPO / "packages/luxar/examples"


class SyncError(RuntimeError):
    """A live count site is missing, ambiguous, or internally inconsistent."""


Replacement = str | Callable[[Match[str]], str]


def _replace_once(
    text: str,
    pattern: Pattern[str],
    replacement: Replacement,
    *,
    site: str,
) -> str:
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise SyncError(f"{site}: expected exactly one match, found {count}")
    return updated


def _sync_readme(text: str, demo_count: int) -> str:
    text = _replace_once(
        text,
        re.compile(r"^(luxar demo\s+# Browse the )\d+( bundled demos)$", re.MULTILINE),
        rf"\g<1>{demo_count}\g<2>",
        site="README quick-start count",
    )

    banner = re.compile(
        r"^🎬 (?P<total>\d+) Luxar demos  ·  (?P<built>\d+) built  ·  "
        r"(?P<cached>\d+) cached  ·  (?P<missing>\d+) not generated yet$",
        re.MULTILINE,
    )

    def replace_banner(match: Match[str]) -> str:
        total = int(match.group("total"))
        built = int(match.group("built"))
        cached = int(match.group("cached"))
        missing = int(match.group("missing"))
        if built + cached + missing != total:
            raise SyncError(
                "README sample banner: built, cached, and not-generated counts "
                f"sum to {built + cached + missing}, not {total}"
            )
        new_missing = missing + demo_count - total
        if new_missing < 0:
            raise SyncError(
                "README sample banner: cannot absorb the demo-count decrease in "
                "the not-generated bucket"
            )
        return (
            f"🎬 {demo_count} Luxar demos  ·  {built} built  ·  {cached} cached  ·  "
            f"{new_missing} not generated yet"
        )

    return _replace_once(
        text,
        banner,
        replace_banner,
        site="README sample banner",
    )


def _sync_claude(text: str, demo_count: int) -> str:
    return _replace_once(
        text,
        re.compile(
            r"^(luxar demo\s+# List the )\d+( bundled demos \(table\))$",
            re.MULTILINE,
        ),
        rf"\g<1>{demo_count}\g<2>",
        site="CLAUDE.md demo count",
    )


def _sync_visualization_skill(text: str, demo_count: int, example_count: int) -> str:
    text = _replace_once(
        text,
        re.compile(
            r"^(- `packages/luxar/src/luxar/demos/demo_\*\.py` \()\d+"
            r"( complete demos\))$",
            re.MULTILINE,
        ),
        rf"\g<1>{demo_count}\g<2>",
        site="visualization skill demo count",
    )
    return _replace_once(
        text,
        re.compile(
            r"^(- `packages/luxar/examples/\*_example\.py` \()\d+"
            r"( focused examples\))$",
            re.MULTILINE,
        ),
        rf"\g<1>{example_count}\g<2>",
        site="visualization skill example count",
    )


def _render_outputs(
    demo_count: int,
    example_count: int,
    *,
    readme: Path,
    claude: Path,
    visualization_skill: Path,
) -> dict[Path, tuple[str, str]]:
    readme_text = readme.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8")
    skill_text = visualization_skill.read_text(encoding="utf-8")
    return {
        readme: (readme_text, _sync_readme(readme_text, demo_count)),
        claude: (claude_text, _sync_claude(claude_text, demo_count)),
        visualization_skill: (
            skill_text,
            _sync_visualization_skill(skill_text, demo_count, example_count),
        ),
    }


def _example_count(examples_dir: Path) -> int:
    return sum(path.is_file() for path in examples_dir.glob("*_example.py"))


def _diff(path: Path, actual: str, expected: str, *, repo: Path) -> str:
    return "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=str(path.relative_to(repo)),
            tofile=f"{path.relative_to(repo)} (synchronized)",
        )
    )


def synchronize(
    demo_count: int,
    example_count: int,
    *,
    check: bool,
    repo: Path = REPO,
    readme: Path = README,
    claude: Path = CLAUDE,
    visualization_skill: Path = VISUALIZATION_SKILL,
) -> int:
    try:
        outputs = _render_outputs(
            demo_count,
            example_count,
            readme=readme,
            claude=claude,
            visualization_skill=visualization_skill,
        )
        changed = {
            path: contents
            for path, contents in outputs.items()
            if contents[0] != contents[1]
        }
        if check:
            for path, (actual, expected) in changed.items():
                print(_diff(path, actual, expected, repo=repo), end="")
            return 1 if changed else 0

        for path, (_, expected) in changed.items():
            path.write_text(expected, encoding="utf-8")
    except (OSError, SyncError) as error:
        print(f"demo-count sync failed: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without rewriting files",
    )
    args = parser.parse_args(argv)

    try:
        demo_count = len(iter_demos(refresh=True))
        example_count = _example_count(EXAMPLES_DIR)
    except (DemoMetaError, OSError) as error:
        print(f"demo-count sync failed: {error}", file=sys.stderr)
        return 2
    return synchronize(
        demo_count,
        example_count,
        check=args.check,
        repo=REPO,
        readme=README,
        claude=CLAUDE,
        visualization_skill=VISUALIZATION_SKILL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
