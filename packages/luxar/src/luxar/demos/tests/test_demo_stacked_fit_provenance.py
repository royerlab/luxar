"""Published stacking demos must classify and persist their component fits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_DEMOS = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module", "reference_kind"),
    [
        ("demo_gsplats_4d_celegans_tracking.py", "preprocessed"),
        ("demo_gsplats_4d_cell_tracking_challenge.py", "preprocessed"),
        ("demo_gsplats_4d_nexrad_supercell.py", "synthetic"),
        ("demo_gsplats_4d_zebrafish_timelapse.py", "preprocessed"),
    ],
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

    stack_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "combine_as_new_dimension"
    ]
    assert any(
        any(keyword.arg == "part_provenance" for keyword in call.keywords)
        for call in stack_calls
    )
