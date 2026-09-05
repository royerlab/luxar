"""Every public Python subpackage must be reachable from the API reference.

``luxar.mesh`` and ``luxar.shading`` were public packages with ``__all__`` and
zero Sphinx presence, while ``docs/api/cli.rst`` carried thirteen ``automodule``
entries for CLI *internals*. A visitor reading the API reference concluded Luxar
rendered three geometry types and had no way to find
``bake_ambient_occlusion``.

The rule is DERIVED from the tree on both sides — the package list by walking
``luxar/`` for ``__init__.py`` files that declare ``__all__``, and the documented
set by scanning every ``auto*`` directive under ``docs/``. Neither side is a
hand-maintained list, so adding a public package without documenting it fails
here rather than being noticed by a reader some months later.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
SRC = REPO / "packages/luxar/src/luxar"
DOCS = REPO / "docs"

#: Directories under ``luxar/`` that are not importable API surface.
NON_PACKAGE_DIRS = frozenset({"tests"})

#: Matches ``.. automodule:: luxar.x``, ``.. autoclass:: luxar.x.Y``, and friends.
AUTODOC_DIRECTIVE = re.compile(r"^\.\.\s+auto\w+::\s+(luxar[\w.]*)", re.MULTILINE)


def _declares_all(init: Path) -> bool:
    """Whether ``init`` assigns ``__all__`` at module level."""
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return True
    return False


def public_packages() -> set[str]:
    """Top-level ``luxar.*`` subpackages that declare ``__all__``."""
    return {
        f"luxar.{child.name}"
        for child in SRC.iterdir()
        if child.is_dir()
        and not child.name.startswith((".", "_"))
        and child.name not in NON_PACKAGE_DIRS
        and (child / "__init__.py").exists()
        and _declares_all(child / "__init__.py")
    }


def documented_modules() -> set[str]:
    """Every ``luxar`` module named by an autodoc directive anywhere in docs/."""
    return {
        match.group(1)
        for rst in DOCS.rglob("*.rst")
        if "_build" not in rst.parts
        for match in AUTODOC_DIRECTIVE.finditer(rst.read_text(encoding="utf-8"))
    }


def test_the_scan_found_something_to_check() -> None:
    """Fail closed: an empty scan on either side would pass every other test.

    A rename of ``packages/luxar/src/luxar`` or of ``docs/`` would otherwise turn
    this whole module into a no-op that reports success.
    """
    assert SRC.is_dir(), f"source tree not found at {SRC}"
    assert DOCS.is_dir(), f"docs tree not found at {DOCS}"
    assert len(public_packages()) >= 10, (
        f"only found {sorted(public_packages())} — the package walk is broken, "
        "not the documentation"
    )
    assert len(documented_modules()) >= 20, (
        f"only found {len(documented_modules())} autodoc directives under {DOCS} — "
        "the docs scan is broken, not the documentation"
    )


@pytest.mark.parametrize("package", sorted(public_packages()))
def test_public_package_appears_in_the_api_reference(package: str) -> None:
    """Each public package is named by an autodoc directive under ``docs/``.

    Documenting a submodule counts: ``docs/api/core.rst`` reaching
    ``luxar.core.transforms`` means ``luxar.core`` has a page a reader can land
    on. What this rejects is a package with no Sphinx presence at all.
    """
    documented = documented_modules()
    reachable = any(
        name == package or name.startswith(f"{package}.") for name in documented
    )
    assert reachable, (
        f"{package} declares __all__ but no docs/**/*.rst names it in an autodoc "
        f"directive, so it is absent from the published API reference. Add a "
        f"docs/api/{package.removeprefix('luxar.')}.rst with an "
        f".. automodule:: {package} and list it in a docs/index.rst toctree."
    )
