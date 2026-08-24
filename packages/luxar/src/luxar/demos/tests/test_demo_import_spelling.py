"""One spelling for the shared demo helpers: ``from luxar.demos import …``.

``luxar/demos/__init__.py`` is the barrel that re-exports the shared plumbing
(``launch_viewer``, ``parse_demo_flags``, ``cached_download``, …) out of
``luxar/utils/demos.py`` and ``luxar/utils/data_fetch.py``, and
``demos/README.md`` §6 documents that spelling as the way to reach it. Even so,
38 demo scripts reached *past* the barrel with ``from luxar.utils.viewer import
…``. Only 12 of them had to: 4 needed ``is_lfs_pointer`` and 8
``print_data_provenance``, neither of which the barrel re-exported. The other 26
deep-imported names the barrel already had — which is the point. One forced
exception per missing symbol was enough to make the deep form look like the house
style, and it then spread by copy-paste to files that never needed it (issue
#1304, item 5).

Two spellings for one surface is a maintenance tax: the barrel stops being the
place a helper's audience can be read off, and moving ``utils/demos.py`` has to
chase 40 call sites instead of one. So this module carries TWO invariants: no
demo module reaches past the barrel, and the barrel really does re-export
everything the demos ask of it — the second one because its breach is what
produced the second spelling in the first place, and a guard is the only thing
that turns that gap into a failure instead of into another deep import.

Every deep *spelling* counts, not just the one the demos happened to use:
``import luxar.utils.viewer``, ``… as ud`` and ``from luxar.utils import viewer``
reach the same module and are checked too (see
:func:`test_the_guard_detects_every_deep_spelling`). The boundary is import
STATEMENTS — a dynamic ``importlib.import_module("luxar.utils.viewer")`` is out
of scope, as it is for every AST lint in this directory.

BOTH detectors are exercised against synthetic trees as well as against the real
(clean) ones, because a clean tree cannot tell a working detector from a broken
one: see :func:`test_the_guard_detects_every_deep_spelling` and
:func:`test_the_guard_detects_a_missing_barrel_export`.

Scope: the demo MODULES, not the whole package. The ``tests/`` subdirectories are
out of scope as a class, and legitimately so — a test may need the module a
private lives in (``test_demo_meta`` deep-imports ``_DEFAULT_CACHE_ROOT`` to pin
that ``registry.DEMO_CACHE_ROOT`` duplicates it, while the concern suites import
their owning utility modules directly), and a re-export
barrel cannot serve either need. Deep imports elsewhere in the package — a few
unit tests building a Lorenz fixture, ``utils/download.py`` reaching for a
zip-path private, and others — are out of scope for the same reason: the barrel
is a demo-authoring convenience, not a package-wide facade.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import luxar.demos as demos_barrel

from ._scanned_modules import EXCLUDED, scanned_demo_modules

#: ``.../luxar`` — this file lives in ``.../luxar/demos/tests``.
LUXAR_DIR = Path(__file__).resolve().parents[2]

#: Every ``demos`` tree under ``luxar/gsplats``, DISCOVERED rather than
#: enumerated: three exist today (``gsplats/demos``, ``gsplats/seeds/demos``,
#: ``gsplats/multiscale/demos`` — the same trees
#: ``gsplats/demos/tests/test_gsplats_demos_compile.py`` byte-compiles), and a
#: fourth added tomorrow joins the guarded set with no edit here. Discovery is
#: the same reasoning as ``_scanned_modules``' denylist, applied to directories.
GSPLAT_DEMO_DIRS = sorted(
    p for p in (LUXAR_DIR / "gsplats").rglob("demos") if p.is_dir()
)

#: The wheel drops all three trees (``pyproject.toml`` ``exclude``) but ships
#: this test, so an installed-package run (``pytest --pyargs luxar``) legitimately
#: finds none of them. Absence is therefore not a discovery failure, and the
#: floors below only apply when the trees are present. BOTH anchors, because
#: either one alone has a blind spot: discovery alone would read a broken glob as
#: "installed", and the canonical directory alone would read a RENAME of
#: ``gsplats/demos`` as "installed" and silently stop scanning the two trees that
#: are still there. Together: a wheel skips, a glob break trips the tree floor,
#: and a rename trips the module floor.
GSPLAT_TREES_PRESENT = (
    bool(GSPLAT_DEMO_DIRS) or (LUXAR_DIR / "gsplats" / "demos").is_dir()
)

#: The MODULE count is the real tripwire; the tree floor is deliberately slack
#: because merging the one-module ``gsplats/seeds/demos`` into ``gsplats/demos``
#: would be a fine refactor and must not read as broken discovery.
MIN_GSPLAT_DEMO_TREES = 1
MIN_GSPLAT_DEMO_MODULES = 25

#: The helper modules the barrel exists to front. Importing any one
#: directly from a demo — under any spelling — is the drift this module fails on.
DEEP_MODULES = frozenset(
    {
        "luxar.utils.bundles",
        "luxar.utils.cache",
        "luxar.utils.colors",
        "luxar.utils.data_fetch",
        "luxar.utils.device",
        "luxar.utils.flags",
        "luxar.utils.lfs",
        "luxar.utils.payload_agreement",
        "luxar.utils.provenance",
        "luxar.utils.scenes",
        "luxar.utils.viewer",
        "luxar.utils.zip_safety",
    }
)

#: ``(package, leaf)`` PAIRS, so ``from luxar.utils import viewer`` is recognised
#: (it binds the same module object as ``import luxar.utils.viewer``) without a
#: cross product: were a deep module added in another package, this must not turn
#: ``from luxar.io import demos`` into a false positive.
DEEP_PAIRS = frozenset(tuple(m.rsplit(".", 1)) for m in DEEP_MODULES)
DEEP_PARENTS = frozenset(parent for parent, _ in DEEP_PAIRS)

#: The barrel itself — the one module that is *supposed* to import deeply.
BARREL = "luxar.demos"

#: The ONE module exempt from the rule, by repo-relative PATH rather than by
#: file name: the barrel is what must import deeply. The bare ``__init__.py``
#: package markers in the gsplats trees are not barrels and get no exemption —
#: nothing about being an ``__init__`` justifies a deep import. The barrel is put
#: into the candidate set explicitly (see :func:`_guarded_modules`) so this
#: exemption is exercised rather than merely declared.
EXEMPT = frozenset({"luxar/demos/__init__.py"})


def _guarded_modules() -> list[Path]:
    """Every module the demo trees ship, minus :data:`EXEMPT`.

    ``scanned_demo_modules`` covers ``luxar/demos`` but drops both names in
    ``_scanned_modules.EXCLUDED``, for reasons that belong to the guards next
    door (``_dependencies.py`` quotes a bare ``pip install`` as an anti-example,
    which trips their docstring lint). Neither reason has anything to do with
    import spelling, so both are added back — derived from ``EXCLUDED`` rather
    than spelled out, so a rename over there cannot leave a stale path here —
    and the barrel is then removed by :data:`EXEMPT`, which is what makes that
    exemption load-bearing instead of decorative.
    """
    gsplat: list[Path] = []
    if GSPLAT_TREES_PRESENT:
        gsplat = sorted(
            p for directory in GSPLAT_DEMO_DIRS for p in directory.glob("*.py")
        )
        assert len(GSPLAT_DEMO_DIRS) >= MIN_GSPLAT_DEMO_TREES, (
            f"found only {len(GSPLAT_DEMO_DIRS)} demos tree(s) under "
            f"{LUXAR_DIR / 'gsplats'} (expected at least "
            f"{MIN_GSPLAT_DEMO_TREES}) — discovery or the package layout changed"
        )
        assert len(gsplat) >= MIN_GSPLAT_DEMO_MODULES, (
            f"found only {len(gsplat)} modules across {len(GSPLAT_DEMO_DIRS)} "
            f"gsplats demos tree(s) (expected at least "
            f"{MIN_GSPLAT_DEMO_MODULES}) — discovery or the layout changed"
        )
    readded = [LUXAR_DIR / "demos" / name for name in sorted(EXCLUDED)]
    absent = [str(p) for p in readded if not p.exists()]
    assert not absent, (
        f"_scanned_modules.EXCLUDED names module(s) that are not on disk: {absent} "
        "— re-key EXCLUDED after the rename so this guard keeps scanning them"
    )
    everything = [*scanned_demo_modules(), *readded, *gsplat]
    return sorted(p for p in everything if _label(p) not in EXEMPT)


def _label(path: Path) -> str:
    """Repo-relative label, so same-named files in two trees stay distinct."""
    try:
        return path.resolve().relative_to(LUXAR_DIR.parent).as_posix()
    except ValueError:  # a synthetic tree under tmp_path
        return path.name


def _resolved_module(node: ast.ImportFrom, path: Path) -> str:
    """Absolute dotted name of what ``node`` imports *from*.

    Relative spellings must resolve, or the guard would pass vacuously against
    ``from ..utils.viewer import …`` — the form a module inside the package is
    most likely to reach for.
    """
    if not node.level:
        return node.module or ""
    parts = path.resolve().parts
    # The LAST "luxar" directory is the package root (the path also contains
    # .../packages/luxar/src/... above it). A relative import is only meaningful
    # inside the package, so refuse to guess rather than raising a bare
    # tuple.index ValueError at the caller.
    assert "luxar" in parts, (
        f"cannot resolve the relative import on line {node.lineno} of {path}: "
        "the file is not inside a `luxar/` package directory"
    )
    start = len(parts) - 1 - parts[::-1].index("luxar")
    package = parts[start:-1]  # e.g. ("luxar", "demos")
    # An over-deep `level` (more dots than there are packages) is a runtime
    # error anyway; clamp so the arithmetic cannot wrap and truncate from the
    # END, which would resolve some unrelated suffix as if it were a module.
    base = package[: max(0, len(package) - (node.level - 1))]
    return ".".join([*base, node.module] if node.module else base)


def _deep_imports(path: Path) -> list[str]:
    """Every import statement in ``path`` that reaches a deep helper module.

    Walks ``ast.Import`` as well as ``ast.ImportFrom`` — the same two node types
    ``test_demos_dependencies`` walks, and for the same reason: checking only
    one of them leaves the other spelling invisible.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            # `import x` is always absolute — there is no relative form.
            hits += [
                (node.lineno, f"import {a.name}")
                for a in node.names
                if a.name in DEEP_MODULES
            ]
        elif isinstance(node, ast.ImportFrom):
            module = _resolved_module(node, path)
            if module in DEEP_MODULES:
                hits.append((node.lineno, f"from {module} import …"))
            elif module in DEEP_PARENTS:
                # `from luxar.utils import viewer` — the module itself is the
                # imported NAME here, so the check has to look at the names.
                hits += [
                    (node.lineno, f"from {module} import {a.name}")
                    for a in node.names
                    if (module, a.name) in DEEP_PAIRS
                ]
    return [f"line {lineno}: {what}" for lineno, what in sorted(hits)]


def _barrel_imported_names(path: Path) -> list[str]:
    """Every name ``path`` imports out of the ``luxar.demos`` barrel."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.ImportFrom) and _resolved_module(node, path) == BARREL:
            names += [alias.name for alias in node.names]
    return names


def _unexported_barrel_names(
    paths: list[Path], allowed: set[str]
) -> dict[str, list[str]]:
    """Names pulled from the barrel that ``allowed`` does not vouch for.

    Factored out so the synthetic negative control below drives the SAME code
    the real check runs, rather than a paraphrase of it.
    """
    missing: dict[str, list[str]] = {}
    for path in paths:
        for name in _barrel_imported_names(path):
            if name not in allowed:
                missing.setdefault(name, []).append(_label(path))
    return missing


def _barrel_allowed_names() -> set[str]:
    """What ``from luxar.demos import <name>`` may legitimately name.

    Live attributes, not ``__all__``: a name that is merely *declared* cannot
    vouch for an import. Submodules count too — ``from luxar.demos import
    demo_gsplats_lod_tribolium as _LOD`` is the documented way to reuse a
    sibling demo's helpers.
    """
    barrel_dir = Path(demos_barrel.__file__).resolve().parent
    bound = {n for n in demos_barrel.__all__ if hasattr(demos_barrel, n)}
    return bound | {p.stem for p in barrel_dir.glob("*.py")}


def test_no_demo_module_reaches_past_the_barrel() -> None:
    """No demo module may import ``luxar.utils.viewer`` / ``.data_fetch`` directly."""
    offenders: dict[str, list[str]] = {}
    for path in _guarded_modules():
        hits = _deep_imports(path)
        if hits:
            offenders[_label(path)] = hits
    assert not offenders, (
        "demo module(s) import the shared helpers past the luxar.demos barrel:\n"
        + "\n".join(
            f"  {name}: {'; '.join(hits)}" for name, hits in sorted(offenders.items())
        )
        + "\nUse the single spelling `from luxar.demos import …` (demos/README.md "
        "§6). If the barrel is missing the symbol you need, add it there — that "
        "gap is what produced the second spelling in the first place. From "
        "_dependencies.py, which the barrel itself imports, put the "
        "`from luxar.demos import …` inside the function instead: a module-scope "
        "one would be circular."
    )


def test_the_guard_detects_every_deep_spelling(tmp_path: Path) -> None:
    """Detection is exercised against a synthetic tree, not only the clean one.

    The real trees pass, so this is the only thing standing between the guard
    and a silent regression in its own detector. It caught a real one: an
    earlier version walked ``ast.ImportFrom`` alone, which let ``import
    luxar.utils.viewer`` and ``from luxar.utils import viewer`` through.
    """
    fake = tmp_path / "luxar" / "demos" / "demo_fake.py"
    fake.parent.mkdir(parents=True)
    deep = [
        "import luxar.utils.viewer",
        "import luxar.utils.viewer as ud",
        "import luxar.utils.data_fetch",
        "from luxar.utils.viewer import launch_viewer",
        "from luxar.utils.data_fetch import ensure_dataset",
        "from luxar.utils import viewer as ud2",
        "from luxar.utils import data_fetch",
        "from ..utils.viewer import parse_demo_flags",
        "from ..utils import viewer as ud3",
    ]
    fine = [
        "from luxar.demos import launch_viewer",  # the one true spelling
        "from luxar.demos.registry import iter_demos",
        "from luxar import demos",  # the barrel, not the helper module
        "from .registry import iter_demos",
        "from ..utils.paths import get_demos_output_dir",
        "from ..utils import paths",
        "import luxar.utils.paths",
    ]
    fake.write_text("\n".join([*deep, *fine]) + "\n", encoding="utf-8")

    hits = _deep_imports(fake)
    assert len(hits) == len(deep), f"expected {len(deep)} hits, got {hits}"
    # Every offending line number is reported exactly once, and no line from the
    # legitimate half is reported at all.
    assert [h.split(":")[0] for h in hits] == [
        f"line {i}" for i in range(1, len(deep) + 1)
    ], hits


def test_the_guard_detects_a_missing_barrel_export(tmp_path: Path) -> None:
    """The second invariant gets a negative control too.

    Without one it passes vacuously — stub ``_barrel_imported_names`` to return
    ``[]`` and the real check still goes green, which is exactly the rot this
    module claims synthetic trees prevent. It matters most for the gsplats
    trees, where invariant #2 is the ONLY import guard they have (their own
    suite byte-compiles them, and ``py_compile`` never resolves an import).
    """
    absolute = tmp_path / "luxar" / "gsplats" / "demos" / "demo_absolute.py"
    absolute.parent.mkdir(parents=True)
    absolute.write_text(
        "from luxar.demos import launch_viewer, definitely_not_exported\n",
        encoding="utf-8",
    )
    # The relative barrel form resolves to `luxar.demos` too, so it must be seen.
    relative = tmp_path / "luxar" / "demos" / "demo_relative.py"
    relative.parent.mkdir(parents=True)
    relative.write_text("from . import launch_viewer, also_not_exported\n", "utf-8")

    missing = _unexported_barrel_names([absolute, relative], {"launch_viewer"})
    assert missing == {
        "definitely_not_exported": ["demo_absolute.py"],
        "also_not_exported": ["demo_relative.py"],
    }, missing


def test_relative_spellings_resolve(tmp_path: Path) -> None:
    """The resolver must catch the relative forms, else the guard is vacuous."""
    fake = tmp_path / "luxar" / "demos" / "demo_fake.py"
    fake.parent.mkdir(parents=True)
    source = (
        "from ..utils.viewer import launch_viewer\n"
        "from ..utils.data_fetch import ensure_dataset\n"
        "from .registry import iter_demos\n"
        "from luxar.utils.viewer import parse_demo_flags\n"
    )
    fake.write_text(source, encoding="utf-8")

    tree = ast.parse(source)
    resolved = [
        _resolved_module(node, fake)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert resolved == [
        "luxar.utils.viewer",
        "luxar.utils.data_fetch",
        "luxar.demos.registry",
        "luxar.utils.viewer",
    ]
    assert sum(module in DEEP_MODULES for module in resolved) == 3

    # An over-deep `level` (more dots than packages) must not wrap: unclamped,
    # `package[: len(package) - (level - 1)]` truncated from the END and made
    # `from ....utils.viewer import …` resolve to the deep module it cannot
    # possibly reach, i.e. reported an unreachable import as a real offender.
    over_deep = ast.parse("from ....utils.viewer import launch_viewer\n").body[0]
    assert isinstance(over_deep, ast.ImportFrom)
    assert _resolved_module(over_deep, fake) not in DEEP_MODULES


def test_resolver_refuses_a_path_outside_the_package(tmp_path: Path) -> None:
    """A verdict or an explanation — never a bare ``tuple.index`` ValueError."""
    stray = tmp_path / "not_a_package" / "demo_stray.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("from ..utils.viewer import launch_viewer\n", encoding="utf-8")

    node = next(
        n
        for n in ast.parse(stray.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ImportFrom)
    )
    try:
        _resolved_module(node, stray)
    except AssertionError as exc:
        assert "not inside a `luxar/` package directory" in str(exc)
        assert str(stray) in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected an explanatory AssertionError")


def test_barrel_all_entries_are_really_bound() -> None:
    """Every ``__all__`` entry must name an attribute that exists.

    ``__all__`` is a declaration, not evidence: ruff's F822 (undefined name in
    ``__all__``) does not fire on ``__init__.py``, so a name listed there with
    no matching import is caught by nothing — and it would make
    :func:`test_barrel_exports_every_symbol_the_demos_pull_from_it` *bless* an
    import that raises ImportError at runtime.
    """
    declared = set(demos_barrel.__all__)
    unbound = sorted(n for n in declared if not hasattr(demos_barrel, n))
    assert not unbound, (
        f"demos/__init__.py lists {unbound} in __all__ but never imports them — "
        "`from luxar.demos import <name>` would raise ImportError."
    )


def test_barrel_exports_every_symbol_the_demos_pull_from_it() -> None:
    """A name imported from the barrel must actually be re-exported by it.

    This is the invariant whose breach created the deep spelling. For
    ``luxar/demos`` a missing re-export would also surface as an ImportError in
    ``test_all_demos_import``, but the gsplats demo trees are byte-compile-smoke
    only, so that half has no other guard at all.
    """
    missing = _unexported_barrel_names(_guarded_modules(), _barrel_allowed_names())
    assert not missing, (
        "name(s) imported from luxar.demos that the barrel does not export: "
        + "; ".join(f"{name} ({', '.join(sorted(f))})" for name, f in missing.items())
        + " — add them to the imports and __all__ in demos/__init__.py."
    )


def test_readme_helper_catalogue_names_real_exports() -> None:
    """The §6 helper list must not advertise a name the barrel has dropped.

    One direction only. §6 is a curated "prefer these over hand-rolling" list by
    its own wording, not the full catalogue: ``__all__`` is that, with
    ``docs/api/utils.rst`` documenting the helper modules behind it and §7 / the
    ``luxar demo deps`` table documenting the dependency-gate surface. Requiring
    every export to appear in §6 would drag those into a section about caching
    and CLI flags. What IS worth pinning is that nothing in the list has gone
    stale.
    """
    readme = Path(demos_barrel.__file__).resolve().parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    # Slice to §6 FIRST: an unindented `from luxar.demos import (` example added
    # anywhere earlier would otherwise be parsed as the helper list, and the
    # failure would blame §6 for a block it does not contain.
    section = re.search(r"^### 6\. .*?(?=^### )", text, re.M | re.S)
    assert section, "demos/README.md has no `### 6.` shared-helpers section"
    block = re.search(
        r"^from luxar\.demos import \(\n(.*?)^\)$", section.group(0), re.M | re.S
    )
    assert block, "the §6 `from luxar.demos import (...)` helper list is gone"

    # Strip the trailing `# …` annotations (prose has commas too), then parse
    # EVERY comma-separated fragment. Asserting the parse is total is what keeps
    # a missing trailing comma, or an `as` alias, from silently shrinking the
    # checked set instead of failing.
    body = "\n".join(line.split("#")[0] for line in block.group(1).splitlines())
    fragments = [f.strip() for f in body.replace("\n", " ").split(",") if f.strip()]
    cited: list[str] = []
    for fragment in fragments:
        # Any identifier, not just a lowercase one: 5 of the 33 exports are
        # classes or constants (`MissingDependencyError`, `INSTALL_SPECS`, …) and
        # adding one to §6 must not read as a malformed list.
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\s+as\s+\w+)?", fragment)
        assert match, f"§6 lists an entry this guard cannot parse: {fragment!r}"
        cited.append(match.group(1))
    assert len(cited) >= 10, f"only parsed {cited} out of the §6 helper list"

    stale = sorted(name for name in cited if not hasattr(demos_barrel, name))
    assert not stale, (
        f"demos/README.md §6 advertises {stale}, which luxar.demos no longer "
        "exports — fix the list or restore the re-export."
    )
