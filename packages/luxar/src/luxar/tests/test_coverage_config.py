"""Tests for repository-wide coverage configuration invariants."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
DELIBERATELY_UNANCHORED_EXCLUSIONS = {
    "pragma: no cover",
    "if __name__ == .__main__.:",
}


def test_coverage_exclusions_are_anchored_or_explicitly_exempt() -> None:
    """New exclusion patterns must not match incidental substrings."""
    with (REPO / "pyproject.toml").open("rb") as file:
        exclusions = tomllib.load(file)["tool"]["coverage"]["report"]["exclude_lines"]

    unanchored = {pattern for pattern in exclusions if not pattern.startswith("^")}

    assert unanchored == DELIBERATELY_UNANCHORED_EXCLUSIONS
