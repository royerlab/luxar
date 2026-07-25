"""Tests for the shared optional-dependency gate (``luxar.demos._dependencies``).

Two things are guarded here:

* :func:`require_module` behaviour — it must raise (never exit, never print) and
  the message must name the CONSTRAINED requirement, so a user never gets told
  to run a bare ``pip install`` that could drag in an incompatible transitive
  dependency.
* Spec/pin agreement — every entry in :data:`INSTALL_SPECS` must stay compatible
  with the corresponding pin in ``pyproject.toml``. This is what stops the hint
  and the packaging metadata from drifting apart; the anndata ceiling in
  particular is load-bearing (``>=0.13`` would upgrade zarr past Luxar's pin and
  break every store on disk).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from luxar.demos import INSTALL_SPECS, MissingDependencyError, require_module
from luxar.demos._dependencies import DependencySpec

# Requirement names that intentionally live outside every Luxar extra.
NOT_IN_ANY_EXTRA = {"gdown"}


def _pyproject() -> Path:
    # tests/ -> demos/ -> luxar/ -> src/ -> luxar/ -> packages/ -> repo root
    return Path(__file__).resolve().parents[6] / "pyproject.toml"


class TestRequireModule:
    def test_present_module_is_returned(self) -> None:
        assert require_module("json") is sys.modules["json"]

    def test_missing_module_raises_missing_dependency_error(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            require_module("no_such_package_xyz")
        # Subclasses ImportError so existing handlers keep working.
        assert isinstance(excinfo.value, ImportError)
        assert "no_such_package_xyz" in str(excinfo.value)

    def test_it_raises_rather_than_exiting(self, monkeypatch) -> None:
        """A gate that called sys.exit could not be handled by its caller."""

        def boom(_code: int = 0) -> None:  # pragma: no cover - must not run
            raise AssertionError("require_module must not call sys.exit")

        monkeypatch.setattr(sys, "exit", boom)
        with pytest.raises(MissingDependencyError):
            require_module("no_such_package_xyz")

    def test_it_does_not_print(self, capsys) -> None:
        """Single-reporter rule: callers report the exception, the gate does not.

        A gate that printed as well showed the user the same message twice, once
        from the helper and once from the entry point's error handler.
        """
        with pytest.raises(MissingDependencyError):
            require_module("no_such_package_xyz")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_known_module_advertises_the_constrained_spec_and_extra(
        self, monkeypatch
    ) -> None:
        monkeypatch.setitem(
            INSTALL_SPECS,
            "nonexistent_pinned_xyz",
            DependencySpec("pkg>=1,<2", "demos", "Because reasons."),
        )

        with pytest.raises(MissingDependencyError) as excinfo:
            require_module("nonexistent_pinned_xyz")

        message = str(excinfo.value)
        assert "pip install 'pkg>=1,<2'" in message
        assert "luxar[demos]" in message
        assert "Because reasons." in message

    def test_spec_without_an_extra_omits_the_extra_hint(self, monkeypatch) -> None:
        monkeypatch.setitem(
            INSTALL_SPECS, "nonexistent_bare_xyz", DependencySpec("pkg", "")
        )
        with pytest.raises(MissingDependencyError) as excinfo:
            require_module("nonexistent_bare_xyz")
        assert "luxar[" not in str(excinfo.value)

    def test_explicit_pip_name_overrides_the_table(self, monkeypatch) -> None:
        monkeypatch.setitem(
            INSTALL_SPECS,
            "nonexistent_pinned_xyz",
            DependencySpec("pkg>=1,<2", "demos"),
        )
        with pytest.raises(MissingDependencyError) as excinfo:
            require_module("nonexistent_pinned_xyz", pip_name="other-pkg")
        assert "other-pkg" in str(excinfo.value)
        assert "pkg>=1,<2" not in str(excinfo.value)

    def test_submodule_inherits_its_package_spec(self) -> None:
        """`require_module("PIL.Image")` must advertise Pillow, not "PIL.Image"."""
        with pytest.raises(MissingDependencyError) as excinfo:
            require_module("PIL.no_such_submodule_xyz")
        message = str(excinfo.value)
        assert "Pillow>=9.0.0" in message
        assert "luxar[demos]" in message

    def test_unknown_module_still_names_itself(self) -> None:
        with pytest.raises(MissingDependencyError, match="totally_unknown_xyz"):
            require_module("totally_unknown_xyz")


class TestSpecsAreValidRequirements:
    @pytest.mark.parametrize("module", sorted(INSTALL_SPECS))
    def test_spec_parses_as_a_pep440_requirement(self, module: str) -> None:
        Requirement = pytest.importorskip("packaging.requirements").Requirement
        Requirement(INSTALL_SPECS[module].spec)  # raises on a malformed spec

    def test_anndata_ceiling_is_present(self) -> None:
        """Load-bearing: anndata >= 0.13 requires zarr >= 3.1, breaking our pin."""
        spec = INSTALL_SPECS["anndata"]
        assert "<0.13" in spec.spec, f"anndata lost its upper bound: {spec.spec}"
        assert "zarr" in spec.note, "the reason for the bound must travel with it"


class TestSpecsMatchPyproject:
    """Every spec must accept exactly the versions its pyproject pin accepts."""

    @pytest.mark.parametrize("module", sorted(INSTALL_SPECS))
    def test_spec_is_equivalent_to_the_pin(self, module: str) -> None:
        pyproject = _pyproject()
        if not pyproject.is_file():  # installed wheel — no source tree to check
            pytest.skip("pyproject.toml not available (installed package)")
        packaging = pytest.importorskip("packaging.requirements")
        specifiers = pytest.importorskip("packaging.specifiers")
        version_mod = pytest.importorskip("packaging.version")

        want = packaging.Requirement(INSTALL_SPECS[module].spec)
        text = pyproject.read_text(encoding="utf-8")
        pins = set()
        for raw in re.findall(r'"([A-Za-z0-9_.\-]+(?:\[[^\]]*\])?[<>=!~][^"]*)"', text):
            try:
                req = packaging.Requirement(raw)
            except Exception:  # noqa: BLE001 - not a requirement string
                continue
            if req.name.lower() == want.name.lower():
                pins.add(str(req.specifier))

        if want.name.lower() in NOT_IN_ANY_EXTRA:
            assert not pins, (
                f"{want.name} is documented as outside every extra but appears "
                f"in pyproject.toml as {pins} — update NOT_IN_ANY_EXTRA or the spec"
            )
            return

        assert pins, (
            f"{want.name} is advertised by INSTALL_SPECS but is not pinned "
            "anywhere in pyproject.toml"
        )

        # Behavioural equivalence beats string equality: `>=2.2` and `>=2.2.0`
        # are the same requirement under PEP 440.
        samples = [
            version_mod.Version(v)
            for v in (
                "0.4",
                "0.5.0",
                "0.9",
                "0.10",
                "0.10.0",
                "0.11.4",
                "0.12.19",
                "0.13",
                "1.0",
                "1.5.0",
                "1.15.0",
                "2.2",
                "2.2.0",
                "2.31.0",
                "3.0",
                "3.0.0",
                "9.0.0",
                "10.0",
            )
        ]
        mine = {str(v): v in want.specifier for v in samples}
        assert any(
            mine == {str(v): v in specifiers.SpecifierSet(pin) for v in samples}
            for pin in pins
        ), (
            f"INSTALL_SPECS[{module!r}] = {want} accepts different versions than "
            f"pyproject's {sorted(pins)} — one of them has drifted"
        )
