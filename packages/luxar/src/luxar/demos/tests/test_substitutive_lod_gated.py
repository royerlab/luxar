"""Guard: no demo may request substitutive LOD without gating its dependencies.

Building a substitutive LOD imports :mod:`luxar.gsplats.lod` — its coarsening
kernels need ``torch`` and its additive sibling does ``from scipy import sparse``
at module load. Neither is a core dependency, so a demo that passes a bare
``substitutive_lod=dict(...)`` dies with a ``ModuleNotFoundError`` mid-build on a
machine that holds every cache it needs — the exact "warm cache runs anywhere"
contract these demos advertise (#712, #1107).

The fix is :func:`luxar.demos.substitutive_lod_or_flat`, which passes the spec
through when both modules are importable and returns ``None`` (flat leaf, plus a
notice) when they are not. This test is what keeps it fixed for the whole class
rather than one demo at a time: it fails if any demo passes a ``substitutive_lod``
that was not resolved through the helper.

Accepted values for the keyword: a direct call to the helper, a local/module name
bound from such a call (the shape used when one spec feeds several ``add_*``
calls and the notice must print once), or a literal ``None``.

Not covered on purpose: a helper call reached through an alias or an attribute
(``deps.substitutive_lod_or_flat(...)``), and a name bound through an
intermediate variable. The analysis is deliberately simple; every demo today is
written in one of the accepted shapes, and the parametrized scan below proves it
stays that way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HELPER = "substitutive_lod_or_flat"

DEMOS_DIR = Path(__file__).resolve().parent.parent
DEMO_FILES = sorted(DEMOS_DIR.glob("demo_*.py"))


def _gated_names(tree: ast.AST) -> set[str]:
    """Names bound anywhere in the module from a ``HELPER(...)`` call."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and _callee(call) == HELPER):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _callee(call: ast.Call) -> str | None:
    """The bare function name of ``call``, or ``None`` for attribute calls."""
    return call.func.id if isinstance(call.func, ast.Name) else None


def _ungated_sites(path: Path) -> list[int]:
    """Line numbers of ``substitutive_lod=`` arguments that skip the helper."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    gated = _gated_names(tree)

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "substitutive_lod":
                continue
            value = kw.value
            ok = (
                (isinstance(value, ast.Call) and _callee(value) == HELPER)
                or (isinstance(value, ast.Name) and value.id in gated)
                or (isinstance(value, ast.Constant) and value.value is None)
            )
            if not ok:
                offenders.append(kw.value.lineno)
    return offenders


@pytest.mark.parametrize("demo", DEMO_FILES, ids=lambda p: p.stem)
def test_demo_gates_substitutive_lod(demo: Path) -> None:
    lines = _ungated_sites(demo)
    assert not lines, (
        f"{demo.name} requests substitutive LOD at line(s) "
        f"{', '.join(str(n) for n in lines)} without gating torch/scipy. Wrap the "
        f"spec in luxar.demos.{HELPER}(...) so a warm-cache run on a machine "
        "without them falls back to a flat leaf instead of crashing."
    )


def test_guard_catches_a_bare_spec(tmp_path: Path) -> None:
    """The guard must actually fail on the shape it exists to prevent.

    Without this, a bug in the AST walk (a renamed keyword, a missed node type)
    would make every real-demo case above pass vacuously.
    """
    bare = tmp_path / "demo_bare.py"
    bare.write_text(
        "scene.add_points('x', substitutive_lod=dict(compression_factor=8))\n"
    )
    assert _ungated_sites(bare) == [1]

    gated = tmp_path / "demo_gated.py"
    gated.write_text(
        "scene.add_points('x', substitutive_lod=substitutive_lod_or_flat(\n"
        "    dict(compression_factor=8)\n"
        "))\n"
    )
    assert _ungated_sites(gated) == []

    via_name = tmp_path / "demo_via_name.py"
    via_name.write_text(
        "lod = substitutive_lod_or_flat(LOD)\n"
        "scene.add_points('x', substitutive_lod=lod)\n"
    )
    assert _ungated_sites(via_name) == []

    ungated_name = tmp_path / "demo_ungated_name.py"
    ungated_name.write_text(
        "lod = dict(compression_factor=8)\nscene.add_points('x', substitutive_lod=lod)\n"
    )
    assert _ungated_sites(ungated_name) == [2]


def test_the_guard_sees_real_call_sites() -> None:
    """At least one demo must actually pass ``substitutive_lod``.

    A refactor that renamed the keyword would otherwise leave this whole file
    green while guarding nothing.
    """
    with_lod = [
        p.name
        for p in DEMO_FILES
        if "substitutive_lod=" in p.read_text(encoding="utf-8")
    ]
    assert with_lod, (
        "no demo passes substitutive_lod — is the keyword still named that?"
    )
