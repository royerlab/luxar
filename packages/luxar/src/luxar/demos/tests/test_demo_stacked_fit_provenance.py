"""Published stacking demos must classify and persist their component fits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_DEMOS = Path(__file__).resolve().parents[1]
_PUBLISHED_STACKS = {
    "demo_gsplats_4d_celegans_tracking.py": "preprocessed",
    "demo_gsplats_4d_cell_tracking_challenge.py": "preprocessed",
    "demo_gsplats_4d_nexrad_supercell.py": "synthetic",
    "demo_gsplats_4d_zebrafish_timelapse.py": "preprocessed",
}
_EXEMPT_STACKS = {
    "demo_gsplats_3d_culling_study.py": (
        "categorical cull views derived from one fit, not independently fitted parts"
    ),
    "demo_quantum_orbitals.py": "not published as a Zenodo gsplat archive",
}


def _stack_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "combine_as_new_dimension"
    ]


def test_every_stacking_demo_is_classified() -> None:
    discovered = {path.name for path in _DEMOS.glob("*.py") if _stack_calls(path)}
    classified = set(_PUBLISHED_STACKS) | set(_EXEMPT_STACKS)
    assert discovered == classified, (
        "stacking demos must supply part_provenance or have a named exemption: "
        f"unclassified={sorted(discovered - classified)}, "
        f"stale={sorted(classified - discovered)}"
    )
    assert all(_EXEMPT_STACKS.values())


@pytest.mark.parametrize(
    ("module", "reference_kind"),
    list(_PUBLISHED_STACKS.items()),
)
def test_stacking_demo_supplies_part_provenance(
    module: str, reference_kind: str
) -> None:
    tree = ast.parse((_DEMOS / module).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    collectors = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "collect_part_provenance"
    ]
    assert len(collectors) == 1
    reference = next(
        keyword.value
        for keyword in collectors[0].keywords
        if keyword.arg == "fit_reference"
    )
    assert isinstance(reference, ast.Dict)
    fields = {
        key.value: value.value
        for key, value in zip(reference.keys, reference.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }
    assert fields["kind"] == reference_kind

    stack_calls = _stack_calls(_DEMOS / module)
    assert any(
        any(keyword.arg == "part_provenance" for keyword in call.keywords)
        for call in stack_calls
    )
