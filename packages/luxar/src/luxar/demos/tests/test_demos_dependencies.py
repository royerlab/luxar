"""Tests for the shared optional-dependency gate (``luxar.demos._dependencies``).

Two things are guarded here:

* :func:`require_module` behaviour — it must raise (never exit, never print) and
  the message must name the CONSTRAINED requirement, so a user never gets told
  to run a bare ``pip install`` that could drag in an incompatible transitive
  dependency.
* Spec/pin agreement — every entry in :data:`INSTALL_SPECS` must stay compatible
  with the corresponding pin in ``pyproject.toml``. This is what stops the hint
  and the packaging metadata from drifting apart; the metpy floor is the live
  example (``metpy<1.6.3`` declares only ``numpy>=1.20`` and breaks at runtime
  against Luxar's ``numpy>=2.0``).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from luxar.demos import INSTALL_SPECS, MissingDependencyError, require_module
from luxar.demos._dependencies import (
    SUBSTITUTIVE_LOD_MODULES,
    DependencySpec,
    _version_satisfied,
    extras_for,
    is_installed,
    substitutive_lod_or_flat,
    survey,
)
from luxar.demos.registry import iter_demos

from ._scanned_modules import EXCLUDED, REQUIRED_SHARED_HELPERS, scanned_demo_modules

# Requirement names that intentionally live outside every Luxar extra.
NOT_IN_ANY_EXTRA = {"gdown", "kaggle"}


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

    def test_every_upper_bound_explains_itself(self) -> None:
        """A ceiling must arrive with its reason, or nobody can ever lift it.

        This generalizes a test that used to pin one specific ceiling
        (``anndata<0.13``, which existed because anndata 0.13 needs zarr >= 3.1
        and the project was pinned to zarr 2). That cap is gone now Luxar is on
        zarr 3 — and the lesson is that an unexplained bound outlives its cause.
        So the invariant is no longer "this bound exists" but "whatever bounds
        exist, say why", which stays true as caps come and go.
        """
        unexplained = [
            module
            for module, spec in INSTALL_SPECS.items()
            if ("<" in spec.spec or "!=" in spec.spec) and len(spec.note.strip()) < 20
        ]
        assert not unexplained, (
            "these specs cap a version without explaining the cap, so a future "
            f"reader cannot tell whether it is still needed: {unexplained}"
        )


class TestHatchEnvironments:
    def test_moderngl_stays_out_of_python_matrix_environments(self) -> None:
        """The optional GPU renderer must not block Python matrix setup."""
        pyproject = _pyproject()
        if not pyproject.is_file():  # installed wheel — no source tree to check
            pytest.skip("pyproject.toml not available (installed package)")

        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        environments = data["tool"]["hatch"]["envs"]
        default_environment = environments["default"]
        test_environment = environments["test"]
        Requirement = pytest.importorskip("packaging.requirements").Requirement
        default_names = {
            Requirement(raw).name.lower() for raw in default_environment["dependencies"]
        }
        test_names = {
            Requirement(raw).name.lower() for raw in test_environment["dependencies"]
        }

        assert "moderngl" not in default_names, (
            "moderngl is demo-only and moderngl/glcontext lack CPython 3.14 wheels; "
            "putting it in the default Hatch environment makes the Python 3.14 "
            "CI leg require system OpenGL headers before tests can run"
        )
        assert "moderngl" not in test_names, (
            "moderngl is demo-only and moderngl/glcontext lack CPython 3.14 wheels; "
            "putting it in the test Hatch environment breaks its Python 3.14 matrix"
        )
        assert "demos" not in default_environment["features"], (
            "the demos feature includes moderngl/glcontext, which lack CPython 3.14 "
            "wheels and must stay out of the default Hatch environment"
        )
        assert "demos" not in test_environment["features"], (
            "the demos feature includes moderngl/glcontext, which lack CPython 3.14 "
            "wheels and must stay out of the test Hatch environment"
        )

    def test_demos_environment_can_run_gpu_renderer_tests(self) -> None:
        """The GL tests retain an explicit environment with pytest and demo deps."""
        pyproject = _pyproject()
        if not pyproject.is_file():  # installed wheel — no source tree to check
            pytest.skip("pyproject.toml not available (installed package)")

        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        demos_environment = data["tool"]["hatch"]["envs"]["demos"]

        assert demos_environment["template"] == "demos"
        assert set(demos_environment["features"]) == {"test", "demos"}
        assert demos_environment["env-vars"] == {
            name: f"{{env:{name}:1}}"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        assert demos_environment["scripts"]["pytest"].startswith("python -m pytest ")


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

        spec = INSTALL_SPECS[module]
        want = packaging.Requirement(spec.spec)

        # Parse the TOML rather than regexing it, so each pin stays attached to
        # the EXTRA that declares it. A name-only match is too weak: scipy is
        # pinned twice (demos >=1.15.0, gsplats >=1.9.0), and a spec that had
        # silently relaxed to the gsplats floor would match "some pin" and pass.
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        optional = data["project"].get("optional-dependencies", {})

        by_extra: dict[str, str] = {}
        for extra_name, reqs in optional.items():
            for raw in reqs:
                try:
                    req = packaging.Requirement(raw)
                except Exception:  # noqa: BLE001 - not a requirement string
                    continue
                if req.name.lower() == want.name.lower():
                    by_extra[extra_name] = str(req.specifier)

        if want.name.lower() in NOT_IN_ANY_EXTRA:
            assert not by_extra, (
                f"{want.name} is documented as outside every extra but appears "
                f"in pyproject.toml as {by_extra} — update NOT_IN_ANY_EXTRA or "
                "the spec"
            )
            return

        assert by_extra, (
            f"{want.name} is advertised by INSTALL_SPECS but is not pinned "
            "in any pyproject.toml extra"
        )
        # The spec must be provided by the extra it CLAIMS to be provided by.
        assert spec.extra in by_extra, (
            f"INSTALL_SPECS[{module!r}] claims extra {spec.extra!r}, but "
            f"{want.name} is only pinned in {sorted(by_extra)}"
        )
        # Compare against THAT extra's pin only — not any pin sharing the name.
        pins = {by_extra[spec.extra]}

        # Behavioural equivalence beats string equality: `>=2.2` and `>=2.2.0`
        # are the same requirement under PEP 440.
        # One sample must land in EVERY gap between the floors we pin, or two
        # different pins map every sample identically and the equivalence check
        # passes vacuously. Keep a value just below and just above each floor.
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
                "0.19.0",
                "0.22",
                "1.0",
                "1.4.0",
                "1.5.0",
                "1.6.0",
                # 1.9/1.14 separate the two scipy floors: the demos extra pins
                # >=1.15.0 and gsplats pins >=1.9.0. Without a sample in
                # [1.9, 1.15) the two are indistinguishable here, and because
                # the check accepts ANY matching pin it would greenlight an
                # INSTALL_SPECS entry that had silently relaxed to >=1.9.0.
                "1.9.0",
                "1.14.0",
                "1.15.0",
                # Straddles the 2.0/2.2 boundary. Without it, sentence-
                # transformers >=2.2.0 and torch >=2.2 read identically to a
                # relaxed >=2.0, and scipy's load-bearing <2.0 cap reads
                # identically to <2.2.
                "2.1.0",
                "2.2",
                "2.2.0",
                "2.3.0",
                "2.31.0",
                "3.0",
                "3.0.0",
                "3.5.0",
                # Between the nibabel floors: the demos extra pins >=5.0.0, so
                # without a sample in [4.0, 5.0) a spec silently relaxed to
                # >=4.0.0 would accept the same sample set and pass vacuously.
                "4.0.0",
                "5.0.0",
                "6.0.0",
                "9.0.0",
                "10.0",
                "12.0.0",
                # A date-versioned sample below the 2023.1.0 floor shared by
                # tifffile/imagecodecs: without one in [12.0.0, 2023.1.0) those
                # rows have a decade-wide blind window where a relaxed pin reads
                # identical to the current one.
                "2020.1.1",
                "2023.1.0",
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


class TestSurvey:
    """``survey`` backs ``luxar demo deps``, so it must be cheap and total."""

    def test_survey_covers_the_whole_table(self) -> None:
        assert {r.module for r in survey()} == set(INSTALL_SPECS)

    def test_survey_is_sorted_case_insensitively(self) -> None:
        """The CLI prints rows in survey order, so it must be stable."""
        modules = [r.module for r in survey()]
        assert modules == sorted(modules, key=str.lower)

    def test_survey_filters_by_extra(self) -> None:
        rows = survey("demos")
        assert rows, "the demos extra must contribute specs"
        assert all(r.spec.extra == "demos" for r in rows)
        # gdown belongs to no extra, so an extra-filtered survey must exclude it.
        assert "gdown" not in {r.module for r in rows}

    def test_survey_does_not_import_the_modules(self) -> None:
        """A survey that imported torch/esm would cost seconds and CUDA context."""
        before = set(sys.modules)
        survey()
        new = set(sys.modules) - before
        assert not (new & set(INSTALL_SPECS)), f"survey imported {new}"

    def test_installed_flag_tracks_importability(self) -> None:
        rows = {r.module: r for r in survey()}
        # scipy is a hard dependency of the test env, numpy-adjacent and always
        # present. The missing-module direction is asserted separately in
        # test_is_installed_is_false_for_a_missing_module.
        assert rows["scipy"].installed is True

    def test_is_installed_is_false_for_a_missing_module(self) -> None:
        assert is_installed("no_such_package_xyz") is False

    def test_is_installed_does_not_raise_on_a_missing_parent(self) -> None:
        """find_spec raises ModuleNotFoundError for a submodule of a missing pkg."""
        assert is_installed("no_such_package_xyz.submodule") is False


class TestVersionAwareness:
    """``survey`` must not report ``ok`` for a package below its pinned floor.

    The concrete bug: the ``demos`` extra floors ``scipy>=1.15`` (for
    ``demo_quantum_orbitals``) while ``gsplats`` floors it at ``1.9``. On an env
    satisfying only the gsplats floor, an import-only survey said ``ok`` and the
    demo then died with an ``AttributeError``. scipy is a hard dependency of the
    test env, so we drive the check by monkeypatching the reported version.
    """

    def test_below_pin_is_not_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("packaging")
        import importlib.metadata as md

        real = md.version
        monkeypatch.setattr(
            md, "version", lambda n: "1.9.0" if n == "scipy" else real(n)
        )
        row = {r.module: r for r in survey()}["scipy"]
        # scipy still imports (installed), but 1.9.0 < the demos floor 1.15.0.
        assert row.installed is True
        assert row.satisfied is False

    def test_meeting_the_pin_is_satisfied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("packaging")
        import importlib.metadata as md

        real = md.version
        monkeypatch.setattr(
            md, "version", lambda n: "1.15.3" if n == "scipy" else real(n)
        )
        assert {r.module: r for r in survey()}["scipy"].satisfied is True

    def test_missing_metadata_gets_the_benefit_of_the_doubt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importable but no distribution metadata: never a false OUTDATED."""
        pytest.importorskip("packaging")
        import importlib.metadata as md

        real = md.version

        def fake(name: str) -> str:
            if name == "scipy":
                raise md.PackageNotFoundError(name)
            return real(name)

        monkeypatch.setattr(md, "version", fake)
        assert {r.module: r for r in survey()}["scipy"].satisfied is True

    def test_spec_without_a_version_bound_is_satisfied(self) -> None:
        """A bare requirement has nothing to check, so it is always satisfied."""
        assert _version_satisfied("some_pkg") is True

    @pytest.mark.parametrize("bad", [None, "1.4.3-1ubuntu2"])
    def test_malformed_version_metadata_never_crashes(
        self, monkeypatch: pytest.MonkeyPatch, bad: object
    ) -> None:
        """A report must never crash: absent `Version:` (None → TypeError) or a
        non-PEP440 distro-patched version (InvalidVersion) → benefit of the doubt.
        """
        pytest.importorskip("packaging")
        import importlib.metadata as md

        real = md.version
        monkeypatch.setattr(md, "version", lambda n: bad if n == "scipy" else real(n))
        # No exception, and the un-judgeable row is not flagged OUTDATED.
        assert {r.module: r for r in survey()}["scipy"].satisfied is True

    def test_unreadable_metadata_never_crashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corrupt (non-UTF-8) METADATA makes ``version()`` raise; the report
        must survive it (``UnicodeDecodeError`` is a ``ValueError``)."""
        pytest.importorskip("packaging")
        import importlib.metadata as md

        real = md.version

        def boom(name: str) -> str:
            if name == "scipy":
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad metadata")
            return real(name)

        monkeypatch.setattr(md, "version", boom)
        assert {r.module: r for r in survey()}["scipy"].satisfied is True


class TestExtrasFor:
    def test_it_dedupes_and_sorts(self) -> None:
        rows = survey()
        extras = extras_for(rows)
        assert extras == sorted(set(extras))
        assert "demos" in extras

    def test_it_drops_specs_outside_every_extra(self) -> None:
        """gdown has extra == "" and cannot be installed via luxar[...]."""
        gdown = [r for r in survey() if r.module == "gdown"]
        assert gdown, "gdown should still be in the table"
        assert extras_for(gdown) == []


class TestSubstitutiveLodGate:
    """The LOD gate DEGRADES instead of raising, and says what was lost.

    Its call sites pass the result straight into ``add_points``/``add_lines``, so
    the pass-through must be the spec object itself — a copy would silently drop
    a caller's later mutation, and anything truthy-but-different would change
    what the writer builds.
    """

    def test_the_gated_modules_are_installable(self) -> None:
        """Advice to install them is only actionable if the table carries them."""
        for module in SUBSTITUTIVE_LOD_MODULES:
            assert module in INSTALL_SPECS, f"{module} missing from INSTALL_SPECS"

    def test_spec_passes_through_when_both_modules_are_present(self, capsys) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("scipy")
        spec = dict(compression_factor=8, levels=3)
        assert substitutive_lod_or_flat(spec) is spec
        assert capsys.readouterr().out == "", "quiet path printed a notice"

    @pytest.mark.parametrize("blocked", SUBSTITUTIVE_LOD_MODULES)
    def test_missing_module_drops_the_spec_and_names_it(
        self, blocked, monkeypatch, capsys
    ) -> None:
        # A None entry makes is_installed() False without needing the package
        # to be genuinely absent from the machine running the tests.
        monkeypatch.setitem(sys.modules, blocked, None)
        assert substitutive_lod_or_flat(dict(compression_factor=8)) is None
        out = capsys.readouterr().out
        assert blocked in out, "notice did not name the missing module"
        assert "skipping Points LOD" in out
        assert "luxar[gsplats]" in out, "notice is not actionable"

    def test_geometry_names_the_right_leaf(self, monkeypatch, capsys) -> None:
        """Lines demos must not be told their Points lost coarsening."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert substitutive_lod_or_flat({}, geometry="Lines") is None
        assert "skipping Lines LOD" in capsys.readouterr().out


class TestScannedModuleSet:
    """The guards below are only as good as the set of files they read.

    Seventeen guards read :func:`scanned_demo_modules` — four here, plus the
    entry-point preflight, substitutive-LOD, Layers-panel, import-spelling,
    tone-mapping-policy, fit-provenance, caption-coverage, canonical-link and
    cinematic-mode guards next door.

    A gate that moves out of a ``demo_*.py`` into a shared helper must stay
    covered, so the set is a denylist over ``demos/*.py`` and
    ``demos/_support/**/*.py`` rather than an opt-in filename pattern.
    """

    def test_shared_helpers_are_scanned(self) -> None:
        demos_dir = Path(__file__).resolve().parents[1]
        relative_paths = {
            path.relative_to(demos_dir).as_posix() for path in scanned_demo_modules()
        }
        assert REQUIRED_SHARED_HELPERS <= relative_paths, (
            "shared helper modules escaped the guarded set: "
            f"{sorted(REQUIRED_SHARED_HELPERS - relative_paths)}"
        )

    def test_the_set_is_a_denylist_over_every_module(self) -> None:
        # Not an allowlist: new helpers in either shared-helper location must be
        # picked up with no edit here, or they silently escape all 17 guards.
        demos_dir = Path(__file__).resolve().parents[1]
        expected = {
            *(p for p in demos_dir.glob("*.py") if p.name not in EXCLUDED),
            *(
                p
                for p in (demos_dir / "_support").rglob("*.py")
                if p.name != "__init__.py"
            ),
        }
        assert set(scanned_demo_modules()) == expected

    def test_excluded_modules_are_justified(self) -> None:
        # Exclusions cost coverage, so each one is named and explained in
        # _scanned_modules' docstring. Keep the set tiny and deliberate.
        assert set(EXCLUDED) == {"__init__.py", "_dependencies.py"}

    def test_scanned_demos_match_the_registry_exactly(self) -> None:
        # The registry is what `luxar demo` lists and what the import smoke test
        # exercises; the guards read the filesystem. A divergence either way is a
        # bug: a registered demo the guards never scan, or a demo_*.py on disk
        # that no longer registers (a broken DEMO_META).
        scanned = {p.name for p in scanned_demo_modules() if p.name.startswith("demo_")}
        registered = {info.path.name for info in iter_demos()}
        assert scanned == registered, (
            "scanned demo modules and the registry disagree: "
            f"only on disk {sorted(scanned - registered)}, "
            f"only registered {sorted(registered - scanned)}"
        )


class TestEveryGatedModuleIsInTheTable:
    """A ``require_module("x")`` whose x is absent from the table would advertise
    a BARE ``pip install x`` — exactly the unbounded hint rule 2 forbids — and
    ``luxar demo deps`` would never report it as missing."""

    def test_all_require_module_arguments_are_known(self) -> None:
        pattern = re.compile(r'require_module\(\s*"([^"]+)"')
        unknown: dict[str, set[str]] = {}
        for path in scanned_demo_modules():
            for module in pattern.findall(path.read_text(encoding="utf-8")):
                # Submodules inherit their package's spec (see require_module).
                root = module.split(".")[0]
                if root not in INSTALL_SPECS:
                    unknown.setdefault(root, set()).add(path.name)
        assert not unknown, (
            "require_module() called with modules missing from INSTALL_SPECS: "
            + "; ".join(f"{m} ({', '.join(sorted(f))})" for m, f in unknown.items())
        )


class TestNoRuntimePipInstall:
    """No demo may install packages behind the user's back.

    Two demos shipped the same shape — a swallowed ImportError that shelled out
    to ``pip install -q <pkg>`` with stdout/stderr sent to DEVNULL
    (``demo_zebrahub_velocity_streamlines``, then ``demo_tabula_sapiens``).
    It mutates the environment without consent, is hostile on a shared HPC node
    or in CI, pulls an UNBOUNDED requirement that can walk a core dependency
    past Luxar's pins, and hides the failure it is papering over. `require_module` is the
    sanctioned reaction to a missing dependency; installing is the user's call,
    via the explicit `luxar demo deps --install` (which is why this scans only
    the demo modules, not the CLI that command lives in).

    KNOWN BLIND SPOT: this is a source-literal check. An argv assembled from
    computed pieces (``["pip", verb]``) would slip through. It closes the shape
    that actually shipped twice, not every conceivable spelling.
    """

    #: An argv token that means "the pip executable".
    _PIP_TOKENS = {"pip", "pip3"}
    #: A shell/command string that runs an install, e.g. "pip install foo".
    _SHELL_PIP_INSTALL = re.compile(r"\bpip3?\b[^\n]*\binstall\b")
    #: Substrings of the method names that hand a command to the OS.
    _EXEC_NAMES = ("run", "call", "output", "system", "popen", "spawn", "exec")

    @classmethod
    def _is_pip_token(cls, value: str) -> bool:
        tail = value.rsplit("/", 1)[-1]
        return tail in cls._PIP_TOKENS or value.strip() == "-m pip"

    @classmethod
    def _offending_nodes(cls, tree: ast.AST) -> list[int]:
        """Line numbers of literal argv lists / shell strings that run pip."""
        found: list[int] = []
        for node in ast.walk(tree):
            # `[sys.executable, "-m", "pip", "install", "-q", "h5py"]` — the
            # shape that shipped. Flagged wherever it is written, so hoisting it
            # into a variable before the subprocess call does not evade it.
            if isinstance(node, (ast.List, ast.Tuple)):
                tokens = [
                    e.value
                    for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if any(cls._is_pip_token(t) for t in tokens) and "install" in tokens:
                    found.append(node.lineno)
                continue
            # `subprocess.run("pip install foo", shell=True)` / os.system(...).
            if isinstance(node, ast.Call):
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                name = func.attr.lower()
                if not any(part in name for part in cls._EXEC_NAMES):
                    continue
                for arg in ast.walk(node):
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and cls._SHELL_PIP_INSTALL.search(arg.value)
                    ):
                        found.append(node.lineno)
                        break
        return sorted(found)

    def _scan(self, source: str) -> list[int]:
        return self._offending_nodes(ast.parse(source))

    def test_no_demo_installs_packages_at_runtime(self) -> None:
        files = scanned_demo_modules()
        offenders = {
            path.name: lines
            for path in files
            if (lines := self._scan(path.read_text(encoding="utf-8")))
        }
        assert not offenders, (
            "demo modules run pip at runtime: "
            + "; ".join(
                f"{name}:{','.join(str(n) for n in lines)}"
                for name, lines in sorted(offenders.items())
            )
            + ". Raise via `require_module` instead and let the user install."
        )

    def test_the_guard_itself_detects_the_shape(self) -> None:
        """A guard that cannot fail is worth nothing — this is what shipped."""
        assert self._scan(
            "import subprocess, sys\n"
            "def _ensure():\n"
            "    subprocess.check_call(\n"
            '        [sys.executable, "-m", "pip", "install", "-q", "h5py"],\n'
            "        stdout=subprocess.DEVNULL,\n"
            "    )\n"
        ) == [4]

    def test_the_guard_detects_a_hoisted_argv(self) -> None:
        assert self._scan(
            'cmd = ["pip", "install", "h5py"]\nsubprocess.check_call(cmd)\n'
        ) == [1]

    def test_the_guard_detects_a_shell_string(self) -> None:
        assert self._scan('subprocess.run("pip install h5py", shell=True)\n') == [1]

    def test_the_guard_ignores_an_install_hint(self) -> None:
        """Telling the user what to run is the sanctioned behaviour."""
        assert self._scan('aprint("Install with: pip install matplotlib")\n') == []

    def test_the_guard_ignores_a_non_pip_subprocess(self) -> None:
        """Demos legitimately shell out to curl for a resumable download."""
        assert (
            self._scan(
                'cmd = ["curl", "-L", "-o", str(dest), url]\n'
                "subprocess.run(cmd, check=True)\n"
            )
            == []
        )


#: Third-party modules a demo may import WITHOUT an INSTALL_SPECS entry, and why.
#: Adding a new unlisted third-party import breaks the build until it is either
#: pinned + tabled or justified here, so an unpinned dependency cannot ship by
#: accident (see the blind spot noted on TestNoUnpinnedThirdPartyImports).
UNLISTED_IMPORTS_OK = {
    # Pure accelerator, deliberately NOT routed through `require_module`: absence
    # is a silent fall-back to `umap-learn`. Tabling it would be worse than
    # leaving it out — RAPIDS ships CUDA-only wheels with no macOS build, so
    # `deps` would show a permanently unmet row and `deps --install` would offer
    # an install that cannot succeed on most machines.
    "cuml": "soft optional GPU UMAP; no CPU-only wheel, absence falls back to umap-learn",
    # Soft optional: the demo prints a warning and returns, so it runs fine
    # without napari. Pinned only in the heavyweight `tracksdata` extra
    # (napari + PyQt6); tabling it would make `deps --install` pull all of that
    # to satisfy a dependency no demo actually requires.
    "napari": "soft optional — demo warns and continues; tracksdata extra only",
    # A hard dependency of `requests`, which is a CORE dependency, so it is
    # always importable. Nothing to advertise.
    "urllib3": "transitive of requests (a core dependency) — always present",
    # PyMOL open-source is NOT on PyPI (conda-forge / Homebrew only), so no pip
    # spec can be advertised: `_pdb_turntable.find_pymol` probes the `pymol`
    # executable, then this import, and prints its own two-route install hint.
    # The stories demo builds without turntables when it is absent.
    "pymol": "not pip-installable; probed by find_pymol, own conda/brew hint, soft optional",
    # Optional provider of an ffmpeg binary when none is on PATH; probed inside
    # try/except in `_pdb_turntable.find_ffmpeg`, which prints its own hint.
    "imageio_ffmpeg": "soft optional ffmpeg fallback; probed in try/except with its own hint",
}


class TestNoUnpinnedThirdPartyImports:
    """Every third-party module a demo IMPORTS must be pinned AND tabled.

    `require_module` coverage is not enough on its own: a plain `import foo`
    never touches the gate, so an unpinned direct import can ship silently —
    which is how `scikit-learn` and `matplotlib` were reaching demos only as
    accidental transitives.

    KNOWN BLIND SPOT: this is an import-graph check, so it cannot see an
    *indirect* runtime need. `pooch` is the worked example — no demo imports it;
    it is what scikit-image's `cells3d()`/`kidney()` fetchers require at call
    time. Nothing static can catch that class; only running the demo on a cold
    cache does. Do not read a green run here as "every demo is runnable".
    """

    @staticmethod
    def _core_import_names() -> set[str]:
        """Import names that the CORE dependencies make always-available."""
        # A few core dists import under a different name than they ship as.
        return {
            "numpy",
            "zarr",
            "typer",
            "fastapi",
            "uvicorn",
            "arbol",
            "colors",  # ansicolors
            "requests",
            "aiohttp",
            "fsspec",
            "xxhash",
            "hilbertcurve",
            "yaml",  # pyyaml
            "click",  # via typer
            "pydantic",  # via fastapi
            "starlette",  # via fastapi
        }

    def test_every_third_party_demo_import_is_pinned_and_tabled(self) -> None:
        allowed = (
            set(sys.stdlib_module_names)
            | {"luxar"}
            | self._core_import_names()
            | set(INSTALL_SPECS)
            | set(UNLISTED_IMPORTS_OK)
        )

        offenders: dict[str, set[str]] = {}
        files = scanned_demo_modules()
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    # level > 0 is a relative (first-party) import.
                    if node.level == 0 and node.module:
                        names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root not in allowed:
                        offenders.setdefault(root, set()).add(path.name)

        assert not offenders, (
            "demo modules import third-party packages that are neither in "
            "INSTALL_SPECS nor justified in UNLISTED_IMPORTS_OK: "
            + "; ".join(
                f"{mod} ({', '.join(sorted(f))})"
                for mod, f in sorted(offenders.items())
            )
        )

    def test_the_allowlist_itself_stays_justified(self) -> None:
        """An allowlist entry must carry a reason and must not shadow the table."""
        for module, reason in UNLISTED_IMPORTS_OK.items():
            assert reason.strip(), f"{module} needs a stated reason"
            assert module not in INSTALL_SPECS, (
                f"{module} is now in INSTALL_SPECS — drop it from "
                "UNLISTED_IMPORTS_OK so the table stays the single source"
            )


class TestNoUnboundedInstallHint:
    """The other half of rule 2: a demo may not TELL the user to run an
    unbounded install of a package the table bounds.

    ``TestEveryGatedModuleIsInTheTable`` only enforces the converse — that a
    gated module is tabled. Nothing stopped a hand-written
    ``aprint("pip install matplotlib")`` from sitting next to the tabled
    ``matplotlib>=3.5.0``, which is how eleven of them accumulated. A hint that
    drops the bound is the same defect as the code doing the install: it walks
    the user into the resolution the pin exists to prevent (a bare
    ``pip install metpy`` landing 1.5.x against Luxar's ``numpy>=2.0``).

    Scanned as text, docstrings included — a "Requirements:" block is advice
    the user follows just as readily as a runtime message.

    KNOWN BLIND SPOT: only packages the table BOUNDS are checked. A hint for an
    untabled package (or for ``gdown``, whose spec is deliberately unbounded)
    has no bound to compare against and passes.
    """

    #: Everything on a line after this is prose about the command, not part of it.
    _COMMENT = "#"
    _HINT = re.compile(r"\bpip3?\s+install\s+(?P<rest>.*)")
    _TOKEN = re.compile(
        r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?P<rest>.*)$"
    )
    _VERSION_OPS = ("==", ">=", "<=", "~=", "!=", "<", ">")

    @staticmethod
    def _canonical(name: str) -> str:
        """PEP 503 name normalisation, so ``umap_learn`` == ``umap-learn``."""
        return re.sub(r"[-_.]+", "-", name).lower()

    @classmethod
    def _bounded_distributions(cls) -> dict[str, str]:
        """Canonical distribution name -> the bound the table advertises."""
        Requirement = pytest.importorskip("packaging.requirements").Requirement
        bounded = {}
        for spec in INSTALL_SPECS.values():
            req = Requirement(spec.spec)
            if str(req.specifier):
                bounded[cls._canonical(req.name)] = str(req.specifier)
        return bounded

    @classmethod
    def _scan(cls, source: str, bounded: dict[str, str]) -> list[tuple[int, str]]:
        """(line, package) for every hint that names a bounded dist unbounded."""
        offenders = []
        for lineno, line in enumerate(source.splitlines(), 1):
            match = cls._HINT.search(line.split(cls._COMMENT)[0])
            if match is None:
                continue
            for raw in match.group("rest").split():
                token = raw.strip("'\"`,.;()")
                if not token or token.startswith("-"):
                    continue  # a pip flag such as -q or --index-url
                parsed = cls._TOKEN.match(token)
                if parsed is None:
                    continue
                name = cls._canonical(parsed.group("name"))
                if name not in bounded:
                    continue
                if not any(op in parsed.group("rest") for op in cls._VERSION_OPS):
                    offenders.append((lineno, name))
        return offenders

    def test_no_demo_advertises_an_unbounded_install(self) -> None:
        bounded = self._bounded_distributions()
        files = scanned_demo_modules()

        offenders = {
            path.name: hits
            for path in files
            if (hits := self._scan(path.read_text(encoding="utf-8"), bounded))
        }
        assert not offenders, (
            "demo modules advertise an unbounded install for a package the "
            "table bounds: "
            + "; ".join(
                f"{name}:" + ",".join(f"{line}({pkg})" for line, pkg in hits)
                for name, hits in sorted(offenders.items())
            )
            + ". Raise via `require_module` (it quotes INSTALL_SPECS) or name "
            "the bound, e.g. `pip install 'luxar[demos]'`."
        )

    def test_the_guard_itself_detects_a_bare_hint(self) -> None:
        """A guard that cannot fail is worth nothing — this is what shipped."""
        bounded = self._bounded_distributions()
        assert self._scan(
            'aprint("Install with: pip install matplotlib")\n', bounded
        ) == [(1, "matplotlib")]

    def test_the_guard_reads_docstrings_too(self) -> None:
        assert self._scan(
            '"""Demo.\n\nRequirements:\n    pip install umap-learn\n"""\n',
            self._bounded_distributions(),
        ) == [(4, "umap-learn")]

    def test_the_guard_accepts_a_bounded_hint(self) -> None:
        bounded = self._bounded_distributions()
        assert self._scan("pip install 'networkx>=3.0'\n", bounded) == []
        assert self._scan('pip install "anndata>=0.10,<0.13"\n', bounded) == []

    def test_the_guard_accepts_an_extra(self) -> None:
        """`luxar[demos]` carries every bound with it — that is the point."""
        assert (
            self._scan("pip install 'luxar[demos]'\n", self._bounded_distributions())
            == []
        )

    def test_the_guard_ignores_prose_after_a_comment(self) -> None:
        """`# includes pandas, umap-learn` describes the extra, it is not a command."""
        assert (
            self._scan(
                "pip install 'luxar[demos]'   # includes pandas, umap-learn\n",
                self._bounded_distributions(),
            )
            == []
        )

    def test_the_guard_ignores_pip_flags(self) -> None:
        assert (
            self._scan(
                "pip install --index-url https://example.invalid 'torch>=2.2,<3.0'\n",
                self._bounded_distributions(),
            )
            == []
        )

    def test_the_guard_ignores_an_unbounded_spec(self) -> None:
        """gdown's own spec carries no bound, so there is nothing to demand."""
        assert self._scan("pip install gdown\n", self._bounded_distributions()) == []
