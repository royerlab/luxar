#!/usr/bin/env python3
"""Synchronize the demo, example, and hosted-demo counts in repository documentation.

Usage::

    hatch run sync-demo-counts
    hatch run check-demo-counts

Three facts, each with exactly one source of truth, are projected onto the
current count sites:

* the **bundled** demo count (``luxar.demos.registry.iter_demos``) → the README
  quick-start line and sample banner, ``CLAUDE.md``, and the visualization skill;
* the **example** count (``packages/luxar/examples/*_example.py``) → the skill;
* the **hosted** demo count (``len(scripts/gallery/manifest.json["demos"])``) →
  the README intro ("N live demos"), the README docs-table row ("N demos as
  interactive scenes"), and ``docs/index.rst`` ("hosts N of the bundled demos").

Historical changelog counts are intentionally outside its scope. ``--check``
renders in memory and exits 1 when a committed file has drifted, without writing
anything. Nothing from ``luxar`` is imported at module level, so
``scripts/gallery/audit_readme_demo_count.py`` — which runs at deploy time in an
environment without the package — can load this file for the shared
:data:`HOSTED_README_CLAIMS` table.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Match, Pattern

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
CLAUDE = REPO / "CLAUDE.md"
VISUALIZATION_SKILL = REPO / ".agents/skills/luxar-visualization/SKILL.md"
DOCS_INDEX = REPO / "docs/index.rst"
EXAMPLES_DIR = REPO / "packages/luxar/examples"
MANIFEST = REPO / "scripts/gallery/manifest.json"

#: Where the README states the HOSTED demo count, as ``(pattern, where)``. Group 1
#: is the number; group 2 the words that locate it, so the sites survive the README
#: being reorganised. Owned here (the CI-gated sync) and read by the deploy-time
#: ``scripts/gallery/audit_readme_demo_count.py`` so the two can never disagree
#: about where the claims live.
HOSTED_README_CLAIMS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"(\d+)(\s+live demos\b)"), "intro banner"),
    (re.compile(r"(\d+)(\s+demos as interactive scenes\b)"), "docs table row"),
)


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


def find_hosted_claims(readme_text: str) -> list[tuple[int, str]]:
    """Every hosted-demo count the README states, as ``(number, where)``."""
    claims: list[tuple[int, str]] = []
    for pattern, where in HOSTED_README_CLAIMS:
        for match in pattern.finditer(readme_text):
            claims.append((int(match.group(1)), where))
    return claims


def _sync_readme(text: str, demo_count: int, hosted_count: int) -> str:
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

    text = _replace_once(
        text,
        banner,
        replace_banner,
        site="README sample banner",
    )

    for pattern, where in HOSTED_README_CLAIMS:
        text = _replace_once(
            text,
            pattern,
            rf"{hosted_count}\g<2>",
            site=f"README hosted count ({where})",
        )
    return text


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


def _sync_docs_index(text: str, hosted_count: int) -> str:
    # The RST admonition wraps mid-sentence ("hosts 88 of the\n   bundled demos"),
    # so the locating words span a line break; ``\s+`` keeps that wrap intact.
    return _replace_once(
        text,
        re.compile(r"(hosts )\d+( of the\s+bundled demos)"),
        rf"\g<1>{hosted_count}\g<2>",
        site="docs/index.rst hosted count",
    )


def _render_outputs(
    demo_count: int,
    example_count: int,
    hosted_count: int,
    *,
    readme: Path,
    claude: Path,
    visualization_skill: Path,
    docs_index: Path,
) -> dict[Path, tuple[str, str]]:
    readme_text = readme.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8")
    skill_text = visualization_skill.read_text(encoding="utf-8")
    index_text = docs_index.read_text(encoding="utf-8")
    return {
        readme: (readme_text, _sync_readme(readme_text, demo_count, hosted_count)),
        claude: (claude_text, _sync_claude(claude_text, demo_count)),
        visualization_skill: (
            skill_text,
            _sync_visualization_skill(skill_text, demo_count, example_count),
        ),
        docs_index: (index_text, _sync_docs_index(index_text, hosted_count)),
    }


def _example_count(examples_dir: Path) -> int:
    return sum(path.is_file() for path in examples_dir.glob("*_example.py"))


def _hosted_count(manifest: Path) -> int:
    """How many demos the gallery hosts: the length of the manifest's ``demos``."""
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SyncError(f"{manifest.name}: not valid JSON ({error})") from error
    demos = document.get("demos") if isinstance(document, dict) else None
    if not isinstance(demos, list):
        raise SyncError(f"{manifest.name}: expected an object with a 'demos' list")
    return len(demos)


def _bundled_count() -> int:
    """How many demos the package bundles, from the registry (imported lazily)."""
    from luxar.demos.registry import DemoMetaError, iter_demos

    try:
        return len(iter_demos(refresh=True))
    except DemoMetaError as error:
        raise SyncError(f"demo registry: {error}") from error


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
    hosted_count: int,
    *,
    check: bool,
    repo: Path = REPO,
    readme: Path = README,
    claude: Path = CLAUDE,
    visualization_skill: Path = VISUALIZATION_SKILL,
    docs_index: Path = DOCS_INDEX,
) -> int:
    try:
        outputs = _render_outputs(
            demo_count,
            example_count,
            hosted_count,
            readme=readme,
            claude=claude,
            visualization_skill=visualization_skill,
            docs_index=docs_index,
        )
        changed = {
            path: contents
            for path, contents in outputs.items()
            if contents[0] != contents[1]
        }
        if check:
            for path, (actual, expected) in changed.items():
                print(_diff(path, actual, expected, repo=repo), end="")
            if changed:
                print(
                    "demo documentation counts are stale — run "
                    "`hatch run sync-demo-counts` and commit the result.",
                    file=sys.stderr,
                )
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
        demo_count = _bundled_count()
        example_count = _example_count(EXAMPLES_DIR)
        hosted_count = _hosted_count(MANIFEST)
    except (SyncError, OSError) as error:
        print(f"demo-count sync failed: {error}", file=sys.stderr)
        return 2
    return synchronize(
        demo_count,
        example_count,
        hosted_count,
        check=args.check,
        repo=REPO,
        readme=README,
        claude=CLAUDE,
        visualization_skill=VISUALIZATION_SKILL,
        docs_index=DOCS_INDEX,
    )


if __name__ == "__main__":
    raise SystemExit(main())
