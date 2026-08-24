"""Pin every demo click-through to its destination's canonical URL shape.

HTTP success is not enough for these links: search endpoints often return 200
for nonsense, while GeneCards returns the same Cloudflare 403 for valid and
invalid symbols. This offline guard records the canonical templates established
by hand review of each destination's current URL scheme and rejects any
unreviewed path or host.

The guard runs two passes: registry resolution for supported link forms, then a
backstop over unclaimed link-like literals that subsumes the GeneCards lint.

The rule is destination-specific, not a blanket ban on legacy-looking paths.
For example, ``genome.ucsc.edu/cgi-bin/hgTracks`` is UCSC's canonical URL.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from ._scanned_modules import scanned_demo_modules

CANONICAL_LINKS_BY_HOST = {
    "bgp.he.net": frozenset({"https://bgp.he.net/AS{hover_key}"}),
    "codex.flywire.ai": frozenset(
        {"https://codex.flywire.ai/app/cell_details?root_id={hover_key}"}
    ),
    "doi.org": frozenset({"https://doi.org/{hover_key}"}),
    "earthquake.usgs.gov": frozenset(
        {"https://earthquake.usgs.gov/earthquakes/eventpage/{hover_key}"}
    ),
    # Both placeholders intentionally drive the same Wikipedia search endpoint.
    "en.wikipedia.org": frozenset(
        {
            "https://en.wikipedia.org/wiki/Special:Search?search={hover_label}",
            "https://en.wikipedia.org/wiki/Special:Search?search={hover_key}",
        }
    ),
    # Keep this literal aligned with demo_dipc_3d_genome.GENOME_ASSEMBLY.
    "genome.ucsc.edu": frozenset(
        {"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&position={hover_key}"}
    ),
    "ned.ipac.caltech.edu": frozenset(
        {"https://ned.ipac.caltech.edu/byname?objname={hover_key}"}
    ),
    # These named-star links are deliberately placeholder-free.
    "simbad.cds.unistra.fr": frozenset(
        {
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Betelgeuse",
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Rigel",
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Sun",
        }
    ),
    "ssd.jpl.nasa.gov": frozenset(
        {"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={hover_key}"}
    ),
    # OLS supports either the key or the display label as its search query.
    "www.ebi.ac.uk": frozenset(
        {
            "https://www.ebi.ac.uk/ols4/search?q={hover_key}",
            "https://www.ebi.ac.uk/ols4/search?q={hover_label}",
        }
    ),
    "www.genecards.org": frozenset({"https://www.genecards.org/card/{hover_key}"}),
    "www.proteinatlas.org": frozenset(
        {"https://www.proteinatlas.org/search/{hover_key}"}
    ),
    "www.uniprot.org": frozenset(
        {
            "https://www.uniprot.org/uniprotkb/{hover_key}/entry",
            "https://www.uniprot.org/uniprotkb?query={hover_key}",
        }
    ),
    "www.youtube.com": frozenset(
        {"https://www.youtube.com/results?search_query={hover_key}"}
    ),
}
CANONICAL_LINKS = frozenset(
    template for templates in CANONICAL_LINKS_BY_HOST.values() for template in templates
)
# Keep in sync with hover-template.ts's PLACEHOLDER_PATTERN; hover_image_label is HTML-only.
LINK_PLACEHOLDERS = ("hover_key", "hover_label", "hover_node", "hover_index")


def _static_strings(tree: ast.Module) -> dict[str, str]:
    """Resolve unambiguous module-level strings used in link templates."""
    binding_counts = Counter(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    binding_counts.update(
        argument.arg for node in ast.walk(tree) for argument in _arguments(node)
    )

    values: dict[str, str] = {}
    pending: list[tuple[str, ast.expr]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and binding_counts[target.id] == 1:
                pending.append((target.id, statement.value))
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.value is not None and binding_counts[statement.target.id] == 1:
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


def _arguments(node: ast.AST) -> list[ast.arg]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = node.args
        return [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *([arguments.vararg] if arguments.vararg is not None else []),
            *([arguments.kwarg] if arguments.kwarg is not None else []),
        ]
    return []


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


def _is_link_subscript(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "link"
    )


def _link_expressions(tree: ast.Module) -> list[tuple[int, ast.expr]]:
    """Find link values authored as kwargs, mapping entries, or assignments."""
    expressions: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "link":
            expressions.append((node.value.lineno, node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "link":
                    expressions.append((value.lineno, value))
        elif isinstance(node, ast.Assign) and any(
            _is_link_subscript(target) for target in node.targets
        ):
            expressions.append((node.value.lineno, node.value))
        elif isinstance(node, ast.AnnAssign) and _is_link_subscript(node.target):
            if node.value is not None:
                expressions.append((node.value.lineno, node.value))
    return sorted(expressions, key=lambda match: match[0])


def _demo_links(path: Path) -> list[tuple[Path, int, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _static_strings(tree)
    return [
        (path, line, _resolve_string(expression, constants))
        for line, expression in _link_expressions(tree)
    ]


def _url_domain(value: str) -> str | None:
    try:
        host = urlsplit(value).netloc
    except ValueError:
        return None
    return host.removeprefix("www.") or None


def _has_link_placeholder(value: str) -> bool:
    return any(f"{{{placeholder}}}" in value for placeholder in LINK_PLACEHOLDERS)


def _unclaimed_link_literals(
    path: Path,
    tree: ast.Module,
    expressions: list[tuple[int, ast.expr]],
    resolved_links: set[str],
    canonical_links: frozenset[str],
    literal_backstop_domains: frozenset[str],
) -> list[tuple[Path, int, str]]:
    claimed_nodes = {
        id(node) for _line, expression in expressions for node in ast.walk(expression)
    }
    unclaimed: list[tuple[Path, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in claimed_nodes
            and node.value not in resolved_links
            and node.value not in canonical_links
            and not any(link.startswith(node.value) for link in resolved_links)
        ):
            continue
        domain = _url_domain(node.value)
        if domain and (
            _has_link_placeholder(node.value) or domain in literal_backstop_domains
        ):
            unclaimed.append((path, node.lineno, node.value))
    return unclaimed


def _audit_links(
    paths: list[Path], canonical_links: frozenset[str]
) -> tuple[
    list[tuple[Path, int, str | None]],
    list[tuple[Path, int, str | None]],
    list[tuple[Path, int, str]],
    list[tuple[Path, int, str]],
]:
    """Return links, unresolved expressions, unregistered links, and stray literals."""
    links: list[tuple[Path, int, str | None]] = []
    unclaimed: list[tuple[Path, int, str]] = []
    literal_backstop_domains = frozenset(
        domain
        for link in canonical_links
        if (domain := _url_domain(link)) is not None
        if not _has_link_placeholder(link)
    )
    # Preserve #2019's GeneCards-wide net; widening it flags citations and
    # non-link endpoints on other registered hosts.
    literal_backstop_domains |= {"genecards.org"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _static_strings(tree)
        expressions = _link_expressions(tree)
        path_links = [
            (path, line, _resolve_string(expression, constants))
            for line, expression in expressions
        ]
        links.extend(path_links)
        resolved_links = {
            value for _path, _line, value in path_links if value is not None
        }
        unclaimed.extend(
            _unclaimed_link_literals(
                path,
                tree,
                expressions,
                resolved_links,
                canonical_links,
                literal_backstop_domains,
            )
        )

    unresolved = [match for match in links if match[2] is None]
    unregistered = [
        (path, line, value)
        for path, line, value in links
        if value is not None and value not in canonical_links
    ]
    return links, unresolved, unregistered, unclaimed


def _locations(matches: list[tuple[Path, int, object]]) -> str:
    return ", ".join(f"{path.name}:{line}" for path, line, _value in matches)


def test_demo_links_use_registered_canonical_templates() -> None:
    links, unresolved, unregistered, unclaimed = _audit_links(
        scanned_demo_modules(), CANONICAL_LINKS
    )
    # Low extractor tripwire; exact template equality below enforces registry coverage.
    assert len(links) >= 15, f"demo link extractor found only {len(links)} call sites"
    assert not unresolved, (
        "demo link templates must be statically resolvable; hoist each template to "
        f"one unshadowed module-level constant: {_locations(unresolved)}"
    )
    assert not unregistered, (
        "verify each destination and add its canonical template to the registry: "
        f"{_locations(unregistered)}"
    )
    assert not unclaimed, (
        "link-like literals must be authored through a supported link form: "
        f"{_locations(unclaimed)}"
    )
    resolved_templates = {value for _path, _line, value in links if value is not None}
    assert resolved_templates == CANONICAL_LINKS, (
        f"unused canonical templates: {sorted(CANONICAL_LINKS - resolved_templates)}"
    )


def test_demo_link_guard_collects_by_module_with_doctests_enabled() -> None:
    # Regression guard: doctest discovery changes pytest's module collection path.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--pyargs",
            __name__,
            "--doctest-modules",
            "--collect-only",
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_demo_links_use_registered_canonical_templates" in result.stdout


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
attrs["link"] = "https://example.org/assigned/{hover_key}"
attrs["link"]: str = "https://example.org/annotated/{hover_label}"
""",
        encoding="utf-8",
    )

    assert _demo_links(module) == [
        (module, 6, "https://example.org/entry/{hover_key}"),
        (module, 7, "https://example.org/view?db=hg19&q={hover_label}"),
        (module, 8, "https://example.org/fixed"),
        (module, 9, "https://example.org/assigned/{hover_key}"),
        (module, 10, "https://example.org/annotated/{hover_label}"),
    ]


def test_demo_link_extractor_rejects_runtime_and_ambiguous_templates(
    tmp_path: Path,
) -> None:
    module = tmp_path / "demo_dynamic.py"
    module.write_text(
        """LINK = "https://example.org/canonical/{hover_key}"
LINK = "https://example.org/rebound/{hover_key}"
scene.add_points(link=LINK)
scene.add_points(link=make_link("{hover_key}"))
""",
        encoding="utf-8",
    )

    assert _demo_links(module) == [(module, 3, None), (module, 4, None)]


def test_demo_link_extractor_rejects_function_local_shadow(tmp_path: Path) -> None:
    module = tmp_path / "demo_shadow.py"
    module.write_text(
        """LINK = "https://example.org/canonical/{hover_key}"
def build():
    LINK = "https://example.org/shadow/{hover_key}"
    scene.add_points(link=LINK)
""",
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 4, None)]
    assert unresolved == links
    assert unregistered == []
    assert unclaimed == [
        (module, 1, "https://example.org/canonical/{hover_key}"),
        (module, 3, "https://example.org/shadow/{hover_key}"),
    ]


def test_demo_link_extractor_rejects_parameter_shadow(tmp_path: Path) -> None:
    module = tmp_path / "demo_parameter_shadow.py"
    module.write_text(
        """LINK = "https://example.org/canonical/{hover_key}"
def build(LINK):
    scene.add_points(link=LINK)
""",
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 3, None)]
    assert unresolved == links
    assert unregistered == []
    assert unclaimed == [(module, 1, "https://example.org/canonical/{hover_key}")]


def test_demo_link_audit_rejects_unclaimed_link_literal(tmp_path: Path) -> None:
    module = tmp_path / "demo_hidden.py"
    module.write_text(
        """NODE = "https://www.uniprot.org/uniprot/{hover_node}"
INDEX = "https://www.proteinatlas.org/legacy/{hover_index}"
LEGACY = "https://www.genecards.org/cgi-bin/carddisp.pl?gene={hover_key}"
NO_WWW = "https://genecards.org/legacy"
scene.add_points(link="https://www.genecards.org/card/{hover_key}")
""",
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 5, "https://www.genecards.org/card/{hover_key}")]
    assert unresolved == []
    assert unregistered == []
    assert unclaimed == [
        (module, 1, "https://www.uniprot.org/uniprot/{hover_node}"),
        (module, 2, "https://www.proteinatlas.org/legacy/{hover_index}"),
        (
            module,
            3,
            "https://www.genecards.org/cgi-bin/carddisp.pl?gene={hover_key}",
        ),
        (module, 4, "https://genecards.org/legacy"),
    ]


def test_demo_link_audit_ignores_malformed_url_literals(tmp_path: Path) -> None:
    module = tmp_path / "demo_malformed.py"
    module.write_text(
        """MIRROR = "https://[hover_key].example.org/x"
scene.add_points(link="https://www.uniprot.org/uniprotkb/{hover_key}/entry")
""",
        encoding="utf-8",
    )
    malformed_canonical = "https://[dead:beef/query"

    links, unresolved, unregistered, unclaimed = _audit_links(
        [module], CANONICAL_LINKS | {malformed_canonical}
    )

    assert links == [(module, 2, "https://www.uniprot.org/uniprotkb/{hover_key}/entry")]
    assert unresolved == []
    assert unregistered == []
    assert unclaimed == []


def test_demo_link_audit_allows_canonical_helper_literal(tmp_path: Path) -> None:
    module = tmp_path / "demo_helper.py"
    module.write_text(
        """UNIPROT_LINK = "https://www.uniprot.org/uniprotkb/{hover_key}/entry"
build_protein_layer(scene, link_template=UNIPROT_LINK)
""",
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == []
    assert unresolved == []
    assert unregistered == []
    assert unclaimed == []


def test_demo_link_audit_allows_resolved_link_prefix_literal(tmp_path: Path) -> None:
    module = tmp_path / "demo_host_constant.py"
    module.write_text(
        """HOST = "https://www.genecards.org"
scene.add_points(link=f"{HOST}/card/{{hover_key}}")
""",
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 2, "https://www.genecards.org/card/{hover_key}")]
    assert unresolved == []
    assert unregistered == []
    assert unclaimed == []


def test_demo_link_registry_rejects_wrong_path_shape(tmp_path: Path) -> None:
    module = tmp_path / "demo_wrong_path.py"
    module.write_text(
        'scene.add_points(link="https://www.uniprot.org/legacy/{hover_key}")\n',
        encoding="utf-8",
    )

    links, unresolved, unregistered, unclaimed = _audit_links([module], CANONICAL_LINKS)

    assert links == [(module, 1, "https://www.uniprot.org/legacy/{hover_key}")]
    assert unresolved == []
    assert unregistered == links
    assert unclaimed == []
