"""Static guard for canonical click-through templates in demo modules.

Demo links may be authored in shared helpers, mapping literals, module constants,
or direct geometry-adder calls. Runtime coverage would require building every
demo and still would not pin the destination's canonical URL shape, so this lint
scans every string literal in the same demo-module set as the other static
guards. When it trips, replace the reported GeneCards URL with
:data:`GENECARDS_LINK`.

This rule is destination-specific, not a blanket ban on legacy-looking paths.
For example, ``genome.ucsc.edu/cgi-bin/hgTracks`` is UCSC's canonical URL and
must remain valid.
"""

from __future__ import annotations

import ast
from pathlib import Path
from urllib.parse import urlsplit

from ._scanned_modules import scanned_demo_modules

GENECARDS_LINK = "https://www.genecards.org/card/{hover_key}"
GENECARDS_DOMAIN = urlsplit(GENECARDS_LINK).netloc.removeprefix("www.")


def test_genecards_links_use_canonical_card_urls() -> None:
    found: list[tuple[Path, int, str]] = []
    offenders: list[tuple[Path, int, str]] = []
    for path in scanned_demo_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and GENECARDS_DOMAIN in node.value
            ):
                match = (path, node.lineno, node.value)
                found.append(match)
                if node.value != GENECARDS_LINK:
                    offenders.append(match)

    assert found, "expected at least one GeneCards demo link"
    assert not offenders, offenders
