"""Every subpackage must be reachable by its own dotted path.

``luxar/__init__.py`` re-exports a lot, and a re-export of the form
``from .<pkg> import <module> as <pkg>`` silently makes the subpackage
unreachable: ``import luxar.<pkg>.<other>`` resolves ``luxar.<pkg>`` by
*attribute* lookup on the parent package, not through ``sys.modules``, so it
finds the inner module and raises ``ImportError`` for a sibling that plainly
exists on disk.

That happened for real. ``from .validation import base as validation`` bound
``luxar.validation`` to ``validation/base.py``, so::

    import luxar.validation.types
    ImportError: cannot import name 'types' from 'luxar.validation.base'

A ``from . import validation as validation_module`` alias sat next to it as the
escape hatch, with no readers anywhere in the repo — the shape of a workaround
that outlived whatever prompted it.

These tests are derived rather than restated: they walk the package directory,
so a subpackage added later is covered without editing this file. Each probe
runs in a subprocess because ``sys.modules`` caching makes an in-process check
answer "fine" once any sibling has been imported by a ``from ... import`` form.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import luxar

PACKAGE_ROOT = Path(luxar.__file__).parent

# Subpackages that are deliberately not importable in a bare install (the heavy
# optional extras). They still must not be SHADOWED, which the attribute test
# below covers without importing them.
NEEDS_OPTIONAL_DEPS = frozenset({"gsplats"})


def _submodule_import_program(targets: list[str]) -> str:
    """Build a probe that ignores only missing third-party dependencies."""
    lines: list[str] = []
    for index, target in enumerate(targets):
        lines.extend(
            [
                "try:",
                f"    import {target} as _m{index}",
                "except ModuleNotFoundError as exc:",
                '    if (exc.name or "").startswith("luxar"):',
                "        raise",
            ]
        )
    lines.append("print('ok')")
    return "\n".join(lines)


def _subpackages() -> list[str]:
    """Every directory under ``luxar/`` that is an importable package."""
    return sorted(
        p.name
        for p in PACKAGE_ROOT.iterdir()
        if p.is_dir()
        and (p / "__init__.py").exists()
        and not p.name.startswith("_")
        and p.name != "tests"
    )


def test_the_subpackage_scan_found_something() -> None:
    """Fail closed: an empty scan would make every test below vacuous."""
    found = _subpackages()
    assert len(found) >= 5, f"only found {found} under {PACKAGE_ROOT}"
    # The one this file exists because of.
    assert "validation" in found


@pytest.mark.parametrize("pkg", _subpackages())
def test_the_attribute_on_luxar_is_the_subpackage_not_a_module_inside_it(
    pkg: str,
) -> None:
    """``luxar.<pkg>`` must be the package, never a module nested in it.

    This is the actual defect: the attribute is what ``import luxar.<pkg>.<x>``
    resolves through. A subpackage that ``luxar/__init__.py`` does not re-export
    at all has no attribute, which is fine — only a WRONG attribute is a bug.
    """
    attr = getattr(luxar, pkg, None)
    if attr is None:
        pytest.skip(f"luxar/__init__.py does not re-export {pkg!r}")
    assert attr.__name__ == f"luxar.{pkg}", (
        f"luxar.{pkg} is bound to {attr.__name__!r}. A "
        f"`from .{pkg} import <module> as {pkg}` re-export shadows the "
        f"subpackage and breaks `import luxar.{pkg}.<sibling>`."
    )
    init = getattr(attr, "__file__", "") or ""
    assert init.endswith("__init__.py"), (
        f"luxar.{pkg}.__file__ is {init!r}, not an __init__.py — the attribute "
        f"points at a module inside the package rather than the package."
    )


@pytest.mark.parametrize("pkg", sorted(set(_subpackages()) - NEEDS_OPTIONAL_DEPS))
def test_every_public_submodule_is_importable_by_its_dotted_path(pkg: str) -> None:
    """``import luxar.<pkg>.<submodule>`` must work for EVERY public submodule.

    Every one, not a sample. An earlier draft imported only the
    alphabetically-first submodule and so passed on the buggy tree: for
    ``validation`` that is ``base``, which is precisely the module the shadow
    pointed at, so ``import luxar.validation.base`` succeeded while
    ``...types`` raised. A probe that the defect satisfies is not a probe.

    The ``as`` form is deliberate and load-bearing. A bare ``import a.b.c``
    only guarantees ``sys.modules`` entries and binds the ROOT name, so it
    succeeds even when ``a.b`` is shadowed; ``import a.b.c as m`` resolves
    ``c`` by attribute traversal from ``a``, which is the operation the shadow
    breaks. A second draft of this test used the bare form and, again, passed
    on the buggy tree.

    One subprocess per package: a pristine interpreter is the only honest test
    of an import path, because once anything in the session has run
    ``from luxar.validation.types import x`` the ``sys.modules`` entry exists
    and even the attribute route can start resolving.
    """
    submodules = sorted(
        p.stem for p in (PACKAGE_ROOT / pkg).glob("*.py") if not p.stem.startswith("_")
    )
    if not submodules:
        pytest.skip(f"{pkg} has no public top-level submodule")
    targets = [f"luxar.{pkg}.{name}" for name in submodules]
    program = _submodule_import_program(targets)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"importing {len(targets)} submodules of luxar.{pkg} failed "
        f"({', '.join(targets)}):\n{result.stderr}"
    )


def test_submodule_probe_tolerates_a_missing_third_party_dependency(
    tmp_path: Path,
) -> None:
    """Optional imports must not disable the dotted-path shadowing probe."""
    (tmp_path / "optional_probe.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'pandas'\", name='pandas')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", _submodule_import_program(["optional_probe"])],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_submodule_probe_rejects_a_missing_luxar_module() -> None:
    """A missing Luxar target must still fail rather than look optional."""
    missing = "luxar.module_that_does_not_exist"
    result = subprocess.run(
        [sys.executable, "-c", _submodule_import_program([missing])],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert missing in result.stderr


def test_the_validation_shadow_specifically_stays_fixed() -> None:
    """The exact statement that used to raise.

    Kept as its own named case alongside the derived sweep above: this is the
    regression, and a reader tracing the bug should find it spelled out rather
    than having to reconstruct which parametrised case covered it.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import luxar.validation.types as vt; "
            "assert vt.__name__ == 'luxar.validation.types', vt.__name__; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_dead_validation_module_alias_is_gone() -> None:
    """``luxar.validation_module`` was the workaround for the shadow.

    It had zero readers in the repo. Asserting its absence is what stops the
    pair from being reintroduced together, which is how it survived: the alias
    made the shadow harmless enough not to notice.
    """
    assert not hasattr(luxar, "validation_module"), (
        "luxar.validation_module is back — it only existed to work around "
        "`luxar.validation` being bound to validation/base.py. Bind the "
        "subpackage itself instead."
    )
