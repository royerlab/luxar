"""Pin every demo click-through to its destination's canonical URL shape.

HTTP success is not enough for these links: search endpoints often return 200
for nonsense, while GeneCards returns the same Cloudflare 403 for valid and
invalid symbols. This offline guard instead records the canonical template that
was verified for each destination and rejects any unreviewed path or host.

The rule is destination-specific, not a blanket ban on legacy-looking paths.
For example, ``genome.ucsc.edu/cgi-bin/hgTracks`` is UCSC's canonical URL.

The registry check reads ``link=`` keywords and ``"link"`` mapping entries, so a
GeneCards URL written anywhere else — a caption, a docstring, a helper constant
that never reaches a link argument — would slip past it. A second, narrower rule
therefore scans every string literal for the GeneCards host and pins its shape,
and asserts the corpus still carries at least one such link so the coverage
cannot quietly disappear.
"""

from __future__ import annotations

import ast
from pathlib import Path
from urllib.parse import urlsplit

from ._scanned_modules import scanned_demo_modules

GENECARDS_LINK = "https://www.genecards.org/card/{hover_key}"
GENECARDS_DOMAIN = urlsplit(GENECARDS_LINK).netloc.removeprefix("www.")

CANONICAL_LINKS = frozenset(
    {
        "https://bgp.he.net/AS{hover_key}",
        "https://codex.flywire.ai/app/cell_details?root_id={hover_key}",
        "https://doi.org/{hover_key}",
        "https://earthquake.usgs.gov/earthquakes/eventpage/{hover_key}",
        "https://en.wikipedia.org/wiki/Special:Search?search={hover_label}",
        "https://en.wikipedia.org/wiki/Special:Search?search={hover_key}",
        "https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&position={hover_key}",
        "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Betelgeuse",
        "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Rigel",
        "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Sun",
        "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={hover_key}",
        "https://www.ebi.ac.uk/ols4/search?q={hover_key}",
        "https://www.ebi.ac.uk/ols4/search?q={hover_label}",
        GENECARDS_LINK,
        "https://www.proteinatlas.org/search/{hover_key}",
        "https://www.uniprot.org/uniprotkb/{hover_key}/entry",
        "https://www.youtube.com/results?search_query={hover_key}",
    }
)


def _static_strings(tree: ast.Module) -> dict[str, str]:
    """Resolve module-level string constants used to compose link templates."""
    values: dict[str, str] = {}
    pending: list[tuple[str, ast.expr]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                pending.append((target.id, statement.value))
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.value is not None:
                pending.append((statement.target.id, statement.value))

    changed = True
    while changed:
        changed = False
        for name, expression in pending:
            if name in values:
                continue
            value = _resolve_string(expression, values)
            if value is not None:
                values[name] = value
                changed = True
    return values


def _resolve_string(expression: ast.expr, constants: dict[str, str]) -> str | None:
    """Evaluate the deliberately small static-string subset used by demos."""
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Name):
        return constants.get(expression.id)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _resolve_string(expression.left, constants)
        right = _resolve_string(expression.right, constants)
        return None if left is None or right is None else left + right
    if isinstance(expression, ast.JoinedStr):
        parts: list[str] = []
        for part in expression.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                value = _resolve_string(part.value, constants)
                if value is None:
                    return None
                parts.append(value)
            else:
                return None
        return "".join(parts)
    return None


def _link_expressions(tree: ast.Module) -> list[tuple[int, ast.expr]]:
    """Find link values authored as kwargs or mapping entries."""
    expressions: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "link":
            expressions.append((node.value.lineno, node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "link":
                    expressions.append((value.lineno, value))
    return expressions


def _demo_links(path: Path) -> list[tuple[Path, int, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _static_strings(tree)
    return [
        (path, line, _resolve_string(expression, constants))
        for line, expression in _link_expressions(tree)
    ]


def _audit_links(
    paths: list[Path], canonical_links: frozenset[str]
) -> tuple[
    list[tuple[Path, int, str | None]],
    list[tuple[Path, int, str | None]],
    list[tuple[Path, int, str]],
]:
    """Return all links, unresolved expressions, and unregistered templates."""
    links = [match for path in paths for match in _demo_links(path)]
    unresolved = [match for match in links if match[2] is None]
    unregistered = [
        (path, line, value)
        for path, line, value in links
        if value is not None and value not in canonical_links
    ]
    return links, unresolved, unregistered


def test_demo_links_use_registered_canonical_templates() -> None:
    links, unresolved, unregistered = _audit_links(
        scanned_demo_modules(), CANONICAL_LINKS
    )
    assert len(links) >= 15, (
        f"expected at least 15 demo link call sites, found {len(links)}"
    )
    assert not unresolved, (
        f"demo link templates must be statically resolvable: {unresolved}"
    )
    assert not unregistered, f"unregistered demo link templates: {unregistered}"


def test_demo_link_extractor_covers_supported_authoring_forms(tmp_path: Path) -> None:
    module = tmp_path / "demo_forms.py"
    module.write_text(
        """
HOST = "https://example.org"
PATH = "/entry/"
ASSEMBLY = "hg19"
DIRECT = HOST + PATH + "{hover_key}"
attrs = {"link": DIRECT}
other = dict(link=f"https://example.org/view?db={ASSEMBLY}&q={{hover_label}}")
scene.add_points(link=("https://example.org/" "fixed"))
""",
        encoding="utf-8",
    )

    assert _demo_links(module) == [
        (module, 6, "https://example.org/entry/{hover_key}"),
        (module, 7, "https://example.org/view?db=hg19&q={hover_label}"),
        (module, 8, "https://example.org/fixed"),
    ]


def test_demo_link_extractor_rejects_runtime_templates(tmp_path: Path) -> None:
    module = tmp_path / "demo_dynamic.py"
    module.write_text(
        'scene.add_points(link=make_link("{hover_key}"))\n', encoding="utf-8"
    )

    assert _demo_links(module) == [(module, 1, None)]


def test_demo_link_registry_rejects_wrong_path_shape(tmp_path: Path) -> None:
    module = tmp_path / "demo_wrong_path.py"
    module.write_text(
        'scene.add_points(link="https://www.uniprot.org/legacy/{hover_key}")\n',
        encoding="utf-8",
    )

    links, unresolved, unregistered = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 1, "https://www.uniprot.org/legacy/{hover_key}")]
    assert unresolved == []
    assert unregistered == links


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
