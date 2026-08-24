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
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "www.genecards.org" in node.value:
                    found.append((path, node.lineno, node.value))

    assert found, "expected at least one GeneCards demo link"
    assert all(value == GENECARDS_LINK for _, _, value in found), found
