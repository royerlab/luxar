"""Corpus-wide guards for demo-authored element-texture budgets."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from luxar.core.group.partition import DEFAULT_MAX_ELEMENTS
from luxar.demos import registry
from luxar.typing_utils.constants import max_elements_per_node

DEMO_PATHS = sorted(registry._DEMOS_DIR.glob("demo_*.py"))
_GEOMETRY_BY_ADDER = {
    "add_points": "points",
    # Lines partition by vertices while the viewer cap counts segments. Since
    # segments <= vertices, enforcing the segment cap here is conservative.
    "add_lines": "lines",
    "add_gsplats": "gsplats",
}
# Mesh is intentionally absent: its partition budget counts faces, and mesh has
# no per-element texture or corresponding max_elements_per_node() capacity.
# Existing ``link_attrs`` spreads contain only link-template render metadata;
# keep that established exception explicit while rejecting every opaque spread.
_KNOWN_NON_BUDGET_SPREADS = {"link_attrs"}


@dataclass(frozen=True)
class AuthoredElementBudget:
    """One per-node element budget authored by a demo call site."""

    path: Path
    line: int
    geometry_type: str
    max_elements: int


def _module_integer_constants(tree: ast.Module) -> dict[str, int]:
    constants: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            value = statement.value
            if (
                value is not None
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
            ):
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value.value
    return constants


def _resolve_integer(
    expression: ast.expr, constants: dict[str, int], *, site: str
) -> int:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    if isinstance(expression, ast.Name) and expression.id in constants:
        return constants[expression.id]
    raise AssertionError(
        f"{site}: demo element budgets must be integer literals or module-level integer "
        "constants so the corpus-wide cap gate can verify them"
    )


def _default_budget_expression() -> ast.Constant:
    return ast.Constant(value=DEFAULT_MAX_ELEMENTS)


def _partition_max_elements(expression: ast.expr, *, site: str) -> ast.expr | None:
    if isinstance(expression, ast.Constant):
        if expression.value is True:
            return _default_budget_expression()
        if expression.value is False or expression.value is None:
            return None
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        if expression.func.id == "dict":
            if expression.args or any(
                keyword.arg is None for keyword in expression.keywords
            ):
                raise AssertionError(
                    f"{site}: demo partition= values cannot use opaque dict inputs because the "
                    "corpus-wide element-cap gate cannot silently miss their budget"
                )
            for keyword in expression.keywords:
                if keyword.arg == "max_elements":
                    return keyword.value
            return _default_budget_expression()
    if isinstance(expression, ast.Dict):
        for key, value in zip(expression.keys, expression.values):
            if key is None:
                raise AssertionError(
                    f"{site}: demo partition= values cannot use dictionary unpacking because "
                    "the corpus-wide element-cap gate cannot silently miss their budget"
                )
            if isinstance(key, ast.Constant) and key.value == "max_elements":
                return value
        return _default_budget_expression()
    raise AssertionError(
        f"{site}: demo partition= values must be statically readable so the corpus-wide "
        "element-cap gate cannot silently miss their per-node budget"
    )


def authored_element_budgets(path: Path) -> list[AuthoredElementBudget]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _module_integer_constants(tree)
    budgets: list[AuthoredElementBudget] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        site = f"{path.name}:{node.lineno}"
        opaque_spreads = [
            keyword.value
            for keyword in node.keywords
            if keyword.arg is None
            and not (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in _KNOWN_NON_BUDGET_SPREADS
            )
        ]
        if opaque_spreads:
            relevant_call = (
                isinstance(node.func, ast.Name) and node.func.id == "LuxarZarrCompiler"
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _GEOMETRY_BY_ADDER
            )
            if relevant_call:
                raise AssertionError(
                    f"{site}: demo geometry and compiler calls cannot use ** keyword spreads "
                    "because the corpus-wide element-cap gate cannot silently miss "
                    "their budgets"
                )
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if isinstance(node.func, ast.Name) and node.func.id == "LuxarZarrCompiler":
            budget_expression = keywords.get("auto_partition_max_elements")
            if budget_expression is None or (
                isinstance(budget_expression, ast.Constant)
                and budget_expression.value is None
            ):
                continue
            max_elements = _resolve_integer(budget_expression, constants, site=site)
            for geometry_type in ("points", "gsplats"):
                budgets.append(
                    AuthoredElementBudget(
                        path=path,
                        line=getattr(budget_expression, "lineno", node.lineno),
                        geometry_type=geometry_type,
                        max_elements=max_elements,
                    )
                )
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        geometry_type = _GEOMETRY_BY_ADDER.get(node.func.attr)
        if geometry_type is None:
            continue
        budget_expression = None
        if "partition" in keywords:
            budget_expression = _partition_max_elements(
                keywords["partition"], site=site
            )
        if budget_expression is None:
            continue
        budgets.append(
            AuthoredElementBudget(
                path=path,
                line=getattr(budget_expression, "lineno", node.lineno),
                geometry_type=geometry_type,
                max_elements=_resolve_integer(budget_expression, constants, site=site),
            )
        )
    return sorted(budgets, key=lambda budget: budget.line)


def over_cap_budgets(
    budgets: list[AuthoredElementBudget],
) -> list[AuthoredElementBudget]:
    return [
        budget
        for budget in budgets
        if budget.max_elements > max_elements_per_node(budget.geometry_type)
    ]


def test_demo_authored_element_budgets_fit_the_viewer_floor() -> None:
    budgets = [
        budget for path in DEMO_PATHS for budget in authored_element_budgets(path)
    ]
    assert budgets, "no demo element budgets found — discovery or authoring changed"
    over_cap = over_cap_budgets(budgets)
    assert not over_cap, (
        "demo per-node budgets exceed the 4096-class viewer cap: "
        + ", ".join(
            f"{budget.path.name}:{budget.line} {budget.max_elements:,} "
            f"{budget.geometry_type}"
            for budget in over_cap
        )
    )


def test_budget_discovery_covers_partition_spellings(tmp_path: Path) -> None:
    demo = tmp_path / "demo_example.py"
    demo.write_text(
        """
POINT_BUDGET = 5_591_040
scene.add_points('points', positions, partition=dict(max_elements=POINT_BUDGET))
scene.add_gsplats('splats', centers, amplitudes, cholesky,
                  partition={'max_elements': 4_194_304})
scene.add_lines('lines', vertices, widths=0.1, partition=True)
scene.add_lines('no_partition', vertices, widths=0.1, partition=False)
scene.add_points('default_dict', positions, partition=dict(rule='sah'))
scene.add_points('default_literal', positions, partition={'rule': 'sah'})
""",
        encoding="utf-8",
    )
    assert [
        (budget.geometry_type, budget.max_elements, budget.line)
        for budget in authored_element_budgets(demo)
    ] == [
        ("points", 5_591_040, 3),
        ("gsplats", 4_194_304, 5),
        ("lines", DEFAULT_MAX_ELEMENTS, 6),
        ("points", DEFAULT_MAX_ELEMENTS, 8),
        ("points", DEFAULT_MAX_ELEMENTS, 9),
    ]


def test_budget_discovery_covers_compiler_auto_partition(tmp_path: Path) -> None:
    demo = tmp_path / "demo_example.py"
    demo.write_text(
        "LuxarZarrCompiler('out', auto_partition_max_elements=1_000_000)\n",
        encoding="utf-8",
    )
    assert [
        (budget.geometry_type, budget.max_elements)
        for budget in authored_element_budgets(demo)
    ] == [("points", 1_000_000), ("gsplats", 1_000_000)]


def test_budget_discovery_rejects_an_unreadable_partition(tmp_path: Path) -> None:
    demo = tmp_path / "demo_example.py"
    demo.write_text(
        "scene.add_points('points', positions, partition=partition_config)\n",
        encoding="utf-8",
    )
    with pytest.raises(
        AssertionError, match=r"demo_example\.py:1: .*cannot silently miss"
    ):
        authored_element_budgets(demo)


@pytest.mark.parametrize(
    "source",
    [
        "scene.add_points('points', positions, **opts)\n",
        "LuxarZarrCompiler('out', **opts)\n",
    ],
)
def test_budget_discovery_rejects_keyword_spreads(tmp_path: Path, source: str) -> None:
    demo = tmp_path / "demo_example.py"
    demo.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError, match=r"demo_example\.py:1: .*keyword spreads"):
        authored_element_budgets(demo)


def test_budget_discovery_allows_known_link_attribute_spreads(tmp_path: Path) -> None:
    demo = tmp_path / "demo_example.py"
    demo.write_text(
        "scene.add_points('points', positions, **link_attrs)\n",
        encoding="utf-8",
    )
    assert authored_element_budgets(demo) == []


def test_over_cap_detection_is_strict_at_the_viewer_floor() -> None:
    budgets = [
        AuthoredElementBudget(Path("at-cap.py"), 1, "points", 5_591_040),
        AuthoredElementBudget(Path("over-cap.py"), 2, "points", 5_591_041),
    ]
    assert over_cap_budgets(budgets) == [budgets[1]]
