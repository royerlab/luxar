"""Repository-wide invariants for demo click-through templates."""

from __future__ import annotations

import ast
from pathlib import Path

DEMOS_DIR = Path(__file__).resolve().parents[1]
GENECARDS_LINK = "https://www.genecards.org/card/{hover_key}"


def test_genecards_links_use_canonical_card_urls() -> None:
    found: list[tuple[Path, int, str]] = []
    for path in sorted(DEMOS_DIR.glob("demo_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                value = keyword.value
                if (
                    keyword.arg == "link"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and "www.genecards.org" in value.value
                ):
                    found.append((path, value.lineno, value.value))

    assert found, "expected at least one GeneCards demo link"
    assert all(value == GENECARDS_LINK for _, _, value in found), found
