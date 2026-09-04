"""Tests for the layering gate (audit A1-01).

The script guards a contract whose failure mode is silence: import-linter will
happily enforce a stale order forever, and will happily accept a new violation
that someone appended to `ignore_imports`. These tests pin both directions, plus
the fail-closed cases — a gate that scores an empty graph would report a perfect
layering for a repository it never read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_layer_order.py"
_spec = importlib.util.spec_from_file_location("check_layer_order", SCRIPT)
assert _spec and _spec.loader
clo = importlib.util.module_from_spec(_spec)
sys.modules["check_layer_order"] = clo
_spec.loader.exec_module(clo)


ORDER = ["cli", "demos", "io", "core", "gsplats", "typing_utils"]


def _edges(pairs: dict[tuple[str, str], int]):
    """Synthetic edge map: pair -> that many distinct (importer, imported)."""
    return {
        (src, dst): [(f"luxar.{src}.m{i}", f"luxar.{dst}.n{i}") for i in range(n)]
        for (src, dst), n in pairs.items()
    }


class TestScore:
    def test_downward_edges_are_free(self):
        edges = _edges({("cli", "core"): 3, ("io", "typing_utils"): 5})
        total, bad = clo._score(ORDER, edges)
        assert total == 0
        assert bad == {}

    def test_upward_edges_are_counted_with_their_multiplicity(self):
        edges = _edges({("typing_utils", "cli"): 4, ("core", "io"): 2})
        total, bad = clo._score(ORDER, edges)
        assert total == 6
        assert bad == {("typing_utils", "cli"): 4, ("core", "io"): 2}

    def test_same_layer_is_not_a_violation(self):
        # A package cannot import itself across the boundary; `_edges` never
        # emits a self-pair, and the scorer must not invent one.
        total, _ = clo._score(ORDER, _edges({("core", "core"): 3}))
        assert total == 0

    def test_a_package_outside_the_declared_order_is_ignored(self):
        """An unlisted package must not be scored as if it were at rank 0.

        `dict.get(..., 0)` would silently rank an unknown package as the top
        layer and report every edge out of it as a violation.
        """
        total, _ = clo._score(ORDER, _edges({("brandnew", "cli"): 9}))
        assert total == 0


class TestReadsTheRealContract:
    def test_declared_order_is_the_one_in_pyproject(self):
        order = clo._declared_order()
        assert order[0] == "cli", "cli must be the top layer"
        assert "gsplats" in order and "core" in order
        # The decision this contract records: gsplats is the data model, below
        # the scene graph and the compiler that consume it.
        assert order.index("gsplats") > order.index("core")
        assert order.index("gsplats") > order.index("io")

    def test_production_ignores_exclude_the_test_globs(self):
        ignores = clo._production_ignores()
        assert ignores, "the debt list came back empty — nothing would be checked"
        assert all("*" not in entry for entry in ignores)
        assert all(".tests." not in entry for entry in ignores)
        assert all(" -> " in entry for entry in ignores)

    def test_production_ignores_exclude_root_package_exemptions(self, monkeypatch):
        monkeypatch.setattr(clo, "_declared_order", lambda: ["encoding", "utils"])
        monkeypatch.setattr(
            clo,
            "_contract",
            lambda: {
                "ignore_imports": [
                    "luxar.encoding.foo -> luxar",
                    "luxar.encoding.foo -> luxar.utils.bar",
                ]
            },
        )
        assert clo._production_ignores() == ["luxar.encoding.foo -> luxar.utils.bar"]


class TestGate:
    def test_passes_on_the_committed_tree(self):
        assert clo.main([]) == 0

    def test_report_mode_does_not_gate(self, capsys):
        assert clo.main(["--report"]) == 0
        out = capsys.readouterr().out
        assert "declared order" in out
        assert "violating edges by pair" in out

    def test_fails_when_violations_outnumber_declared_debt(self, monkeypatch, capsys):
        """A new violation must not be waveable by appending an ignore."""
        real = clo._edges
        monkeypatch.setattr(
            clo,
            "_edges",
            lambda: {
                **real(),
                ("typing_utils", "cli"): [("luxar.typing_utils.x", "luxar.cli.y")],
            },
        )
        assert clo.main([]) == 1
        assert "must be fixed, not appended" in capsys.readouterr().err

    def test_fails_when_violations_and_debt_grow_together(self, monkeypatch, capsys):
        real_edges = clo._edges
        real_ignores = clo._production_ignores
        monkeypatch.setattr(
            clo,
            "_edges",
            lambda: {
                **real_edges(),
                ("typing_utils", "cli"): [("luxar.typing_utils.x", "luxar.cli.y")],
            },
        )
        monkeypatch.setattr(
            clo,
            "_production_ignores",
            lambda: real_ignores() + ["luxar.typing_utils.x -> luxar.cli.y"],
        )
        assert clo.main([]) == 1
        assert "exceeds the committed maximum" in capsys.readouterr().err

    def test_fails_when_declared_debt_outnumbers_violations(self):
        """Paid-down debt left in the list is a failure, not a pass.

        import-linter catches this too, by refusing an ignore that matches
        nothing. Checked here as well so the two cannot both be relaxed at once
        without a test going red.
        """
        problem = clo._debt_problem(clo.MAX_DEBT - 1, ["edge"] * clo.MAX_DEBT)
        assert problem is not None
        assert "1 named edge no longer exists" in problem

    def test_fails_when_a_better_order_exists(self, monkeypatch, capsys):
        declared = clo._declared_order()
        swapped = list(declared)
        i, j = swapped.index("io"), swapped.index("gsplats")
        swapped[i], swapped[j] = swapped[j], swapped[i]
        monkeypatch.setattr(clo, "_declared_order", lambda: swapped)
        assert clo.main([]) == 1
        assert "achievable with" in capsys.readouterr().err


class TestFailsClosed:
    def test_an_empty_graph_is_an_error_not_a_perfect_score(self, monkeypatch):
        """A graph that came back empty scores zero violations — a fake pass.

        This is the failure this whole family of gates keeps hitting: the scan
        finds nothing, every assertion is vacuously true, and the build is green
        for a repository it never read.
        """
        import grimp

        class _Empty:
            modules: tuple[str, ...] = ()

            def find_modules_directly_imported_by(self, _module):  # pragma: no cover
                return ()

        monkeypatch.setattr(grimp, "build_graph", lambda *a, **k: _Empty())
        with pytest.raises(SystemExit):
            clo._edges()

    def test_an_unlisted_subpackage_is_an_error(self, monkeypatch):
        import grimp

        class _Graph:
            modules = ("luxar", "luxar.brandnew", "luxar.cli")

            def find_modules_directly_imported_by(self, module):
                if module == "luxar.brandnew":
                    return {"luxar.cli"}
                return set()

        monkeypatch.setattr(grimp, "build_graph", lambda *a, **k: _Graph())
        monkeypatch.setattr(clo, "_declared_order", lambda: ["cli"])
        with pytest.raises(SystemExit):
            clo._edges()

    def test_graph_uses_the_configured_type_checking_setting(self, monkeypatch):
        import grimp

        seen = {}

        class _Graph:
            modules = ("luxar", "luxar.cli", "luxar.core")

            def find_modules_directly_imported_by(self, module):
                if module == "luxar.cli":
                    return {"luxar.core"}
                return set()

        def build_graph(*args, **kwargs):
            seen.update(kwargs)
            return _Graph()

        monkeypatch.setattr(grimp, "build_graph", build_graph)
        monkeypatch.setattr(clo, "_exclude_type_checking_imports", lambda: False)
        monkeypatch.setattr(clo, "_declared_order", lambda: ["cli", "core"])
        clo._edges()
        assert seen["exclude_type_checking_imports"] is False

    def test_a_missing_contract_is_an_error(self, monkeypatch):
        monkeypatch.setattr(clo, "CONTRACT_NAME", "no such contract")
        with pytest.raises(SystemExit):
            clo._contract()

    def test_a_contract_with_no_layers_is_an_error(self, monkeypatch):
        monkeypatch.setattr(clo, "_contract", lambda: {"name": clo.CONTRACT_NAME})
        with pytest.raises(SystemExit):
            clo._declared_order()
