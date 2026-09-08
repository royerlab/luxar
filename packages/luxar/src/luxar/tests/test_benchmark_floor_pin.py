"""Guard: every gsplat fit in ``scripts/`` declares its background-floor basis.

A benchmark that scores PSNR/SSIM against the RAW volume must fit on the same
intensity basis. The shipped default ``floor="auto"`` estimates and subtracts a
background pedestal, so its output amplitudes are background-relative;
original-referenced metrics then penalise the fit for correctly dropping
non-signal, and move silently whenever the floor estimator changes (#1184, and
the 2026-07-12 ``--floor`` decision in
``docs/guides/developer/BENCHMARK_FLOOR_DECISION.md``).

So: **every call to a module-level gsplat fitting function in any ``*.py`` under
``scripts/`` must declare its floor at the call site** — normally as an explicit
``floor=`` keyword, or in a kwargs dict visible there (see below). The gate is
about a *declared* basis, not a particular value — a harness that genuinely wants
the shipped behaviour simply writes ``floor="auto"`` and passes.

DO NOT ADD A FILE ALLOW-LIST. The requirement is satisfiable entirely inside the
file that adds the call, so a new harness written on one branch can never turn
this guard red after merging with an unrelated branch. An exemption set keyed on
file names would reintroduce exactly that cross-branch failure. For the same
reason an unparseable file is reported as a finding rather than raised out of
the test — see :func:`find_unpinned_fits`.

A ``floor=`` passed as a variable (or any non-literal expression) counts: the
author still had to decide what basis to fit on, which is all this gate asks.
A ``**`` unpack counts only when a dict carrying a ``"floor"`` key is *visible at
the call site*: an inline ``**{"floor": ...}`` / ``**dict(floor=...)``, or
``**NAME`` where ``NAME`` is assigned a dict literal (or a ``dict(...)`` call)
with that key at module level in the same file. That is the idiom the five
luxar-paper harnesses this convention comes from pin with
(``FIT_KWARGS = dict(..., floor="none", ...)``, applied as
``fit_gaussian_splats(V, **FIT_KWARGS)``), so it has to pass — and the offender
message must not tell such an author to add a second ``floor=`` beside the dict,
which is a ``TypeError``. A bare ``**kwargs`` forward inside a function still
does *not* count: nothing visible at the call site carries the decision.

Local names are resolved through ``import``/``from ... import ... as`` aliases in
the same file, so ``from luxar.gsplats import fit_gaussian_splats as fit`` then
``fit(V)`` is caught.

What this gate deliberately does NOT cover
------------------------------------------
*The class API.* :class:`~luxar.gsplats.GaussianSplatFitter`'s
``fit(FitParameters(V, ..., floor="auto"))`` honours the same spec, but an AST gate sees only
the attribute name ``fit``, and demanding ``floor=`` on every ``.fit(`` would
flag ``model.fit(X, y)`` / ``scaler.fit_transform(X)`` (the ``_NOT_A_FITTER``
case below pins that it does not). Hence *module-level* functions only.

*A positional floor.* ``floor`` is the fourth positional parameter of
:func:`~luxar.gsplats.fit_gaussian_splats`
(``V, seeds, norm_percentile, floor``), so ``fit_gaussian_splats(V, 8000, 0.0,
"none")`` genuinely is pinned and the gate still reports it. That is on purpose:
"declared" means a reader of the call site can *see* which argument is the floor,
so the fix is to name it, not to weaken the check.

*A fitter carried as a value.* ``functools.partial(fit_gaussian_splats, ...)``, a
dict of callables, a decorator that returns one — anything that passes the
function around rather than calling it by name — is not matched. Import aliases
are (above); the rest would need the general dataflow analysis this gate
deliberately does not have.

*Generic names.* Matching is by bare name, so a *local* helper in ``scripts/``
called ``fit_tile``/``fit_tiled`` would be asked for a ``floor=`` it may not
accept. None exists today; if one appears, give it a ``floor`` parameter or
rename it.

*argv-driven fits.* ``scripts/calibrate_gsplat_demos.py`` shells out to
``luxar gsplat cal``; a subprocess argv is invisible to a call-name gate. That
one needs no pin anyway: ``calibrate`` subtracts the floor from ``V`` ONCE up
front and then pins its own per-K fits to ``floor="none"``
(``luxar/gsplats/calibration/driver.py``), so fit target, render reference and
held-out truth all sit on one floored basis — self-consistent, unlike an
original-referenced benchmark. (``calibrate`` also takes the spec inside a
``fit_kwargs`` *dict parameter* rather than as a keyword, so a keyword check
could not see it either. :func:`luxar.gsplats.planner.fit_planned` reads the
same way at a glance but does not behave that way — it collects ``**fit_kwargs``,
so a ``floor=`` written at its call site is plainly visible here and honoured at
fit time; it is guarded, see :data:`GUARDED_NAMES`.)

*Anything outside ``scripts/``.* ``scripts/`` is the line because these are the
harnesses whose numbers are consumed as machine verdicts — an autoresearch
``METRIC=`` line, a Pareto-dominance exit code — where a silently drifting basis
changes a decision. The same shape exists in the demos and is left alone on
purpose. Measured with this module's own detector, 40 demo modules fit unpinned —
21 under ``luxar/gsplats/demos/`` and 19 under ``luxar/demos/`` — and about 17 of
those go on to report a PSNR, among them
``luxar/gsplats/demos/demo_3d_dapi_microscopy.py`` and
``luxar/gsplats/demos/demo_performance_metrics.py``, which score the
reconstruction against the raw volume exactly as the benchmarks did. A PSNR a
demo prints is illustrative, though, and nothing gates on it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

#: Names that must declare a floor basis when called from ``scripts/``: the
#: module-level gsplat fitting entry points, plus the two worker-argv builders.
#:
#: ``fit_gaussian_splats`` takes the spec as a named parameter;
#: ``fit_progressive_gaussian_splats`` (via ``**kwargs``) and
#: ``fit_tile``/``fit_tiled`` (via ``**fit_kwargs``) pop it themselves;
#: ``fit_tiled_gaussian_splats`` is the public alias of ``fit_tiled``, exported
#: in ``luxar.gsplats.__all__`` and documented under that name in
#: ``docs/api/gsplats.rst``, so it must be listed too or the gate is bypassable
#: by spelling.
#:
#: ``build_worker_cmd`` (``luxar/gsplats/fit_tiled_parallel.py``) and
#: ``_default_worker_cmd_builder``
#: (``luxar/gsplats/planner/fit_planned_parallel.py``) are argv BUILDERS rather
#: than fitters, and they are the one remaining path that can reproduce #1184:
#: each takes a real ``floor`` keyword defaulting to ``None``, and each OMITS
#: ``--floor`` from the worker argv when it is ``None`` — so the tile worker then
#: resolves the shipped ``auto``. A ``scripts/`` harness that tile-fits via
#: ``fit_tiled_parallel(worker_cmd_builder=build_worker_cmd(...))`` and scores
#: against the raw volume is the original bug again, silently. The requirement is
#: satisfiable on them precisely because they *have* the keyword.
#:
#: ``fit_planned`` (``luxar/gsplats/planner/fit_planned.py``) collects
#: ``**fit_kwargs`` and forwards them to a per-box ``fit_gaussian_splats``, so a
#: ``floor=`` at its call site is both visible here and honoured at fit time —
#: the same mechanism as ``fit_progressive_gaussian_splats``. It needs the
#: declaration more than most: with no floor in ``fit_kwargs`` every box
#: re-estimates ``auto`` against its own crop, which its own docstring warns
#: leaves visible brightness steps at box boundaries, so the value declared there
#: has to be a concrete level or ``"none"``.
#:
#: ``fit_tiled_parallel`` and ``fit_planned_parallel`` are deliberately NOT
#: listed: both are keyword-only with no ``floor`` parameter and no
#: ``**kwargs``, so demanding ``floor=`` there would order an author to write a
#: call that raises ``TypeError``. Their per-tile/per-box floor only ever travels
#: through the caller's ``worker_cmd_builder`` — i.e. through one of the two
#: builders above.
GUARDED_NAMES: frozenset[str] = frozenset(
    {
        "fit_gaussian_splats",
        "fit_progressive_gaussian_splats",
        "fit_tile",
        "fit_tiled",
        "fit_tiled_gaussian_splats",
        "fit_planned",
        "build_worker_cmd",
        "_default_worker_cmd_builder",
    }
)


def _called_name(node: ast.Call) -> str | None:
    """Return the bare function name of a call (``mod.fit_x`` → ``fit_x``)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _guarded_names_in(tree: ast.Module) -> frozenset[str]:
    """:data:`GUARDED_NAMES` plus the local aliases bound to one by an import.

    ``from luxar.gsplats import fit_gaussian_splats as fit`` makes ``fit`` a
    guarded name in *this* source, so renaming on import does not evade the gate.
    Resolution is single-file on purpose: no cross-module chasing.
    """
    names = set(GUARDED_NAMES)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            original = alias.name.rsplit(".", 1)[-1]
            if alias.asname and original in GUARDED_NAMES:
                names.add(alias.asname)
    return frozenset(names)


def _carries_floor_key(node: ast.expr) -> bool:
    """True for a dict literal / ``dict(...)`` call with a ``"floor"`` key."""
    if isinstance(node, ast.Dict):
        return any(
            isinstance(key, ast.Constant) and key.value == "floor" for key in node.keys
        )
    if isinstance(node, ast.Call) and _called_name(node) == "dict":
        return any(kw.arg == "floor" for kw in node.keywords)
    return False


def _module_dicts_with_floor(tree: ast.Module) -> frozenset[str]:
    """Module-level names assigned a dict that carries a ``"floor"`` key.

    This is the whole of the "visible at the call site" rule for ``**NAME``: one
    pass over the module body, no dataflow analysis.
    """
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets: list[ast.expr] = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        else:
            continue
        if stmt.value is None or not _carries_floor_key(stmt.value):
            continue
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    return frozenset(names)


def _declares_floor(node: ast.Call, dict_names: frozenset[str]) -> bool:
    """True when the floor basis is declared *at this call site*."""
    for kw in node.keywords:
        if kw.arg == "floor":
            return True
        if kw.arg is None:  # a ** unpack
            if _carries_floor_key(kw.value):
                return True
            if isinstance(kw.value, ast.Name) and kw.value.id in dict_names:
                return True
    return False


def _guarded_calls(tree: ast.Module) -> list[tuple[ast.Call, str, bool]]:
    """Every guarded call in ``tree`` as ``(node, name, declares a floor)``."""
    guarded = _guarded_names_in(tree)
    dict_names = _module_dicts_with_floor(tree)
    calls: list[tuple[ast.Call, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name is None or name not in guarded:
            continue
        calls.append((node, name, _declares_floor(node, dict_names)))
    return calls


def find_unpinned_fits(source: str | bytes, *, filename: str = "<source>") -> list[str]:
    """Return one message per guarded call in ``source`` with no declared floor.

    "Declared" is :func:`_declares_floor`: a ``floor=`` keyword, or a ``**`` unpack
    of a dict that carries the key and is visible at the call site. Only *calls*
    count: an import, a docstring or a string mentioning a fitter name is not a
    call and cannot satisfy or violate the gate.

    ``source`` is normally the file's raw BYTES, so Python's own source-encoding
    rules apply: a leading UTF-8 BOM and a PEP 263 cookie
    (``# -*- coding: latin-1 -*-``) are both handled, whereas decoding to ``str``
    as UTF-8 first would raise on a valid non-UTF-8 file. Same approach, same
    reason as ``_check_python_file_docstrings`` in
    ``scripts/check_documentation.py``.

    An unparseable file (a WIP syntax error, an undeclared encoding) becomes ONE
    clearly-worded finding naming it instead of a traceback out of this gate. The
    two arms cover what the supported interpreters (``requires-python =
    ">=3.12"``) actually raise for the cases that reach a source gate:
    ``SyntaxError`` for unparseable sources, undeclared non-UTF-8 sources and
    embedded NULs — which 3.12 reclassified from ``ValueError``, so the upstream
    comment's "``ValueError`` the embedded-NUL case on 3.10/3.11" no longer
    applies here — and ``ValueError`` for the ``UnicodeError`` family, since a
    lone surrogate in a ``str`` source raises ``UnicodeEncodeError``, a
    ``ValueError`` subclass. They are not a blanket ``except Exception``: a
    pathological source can still hit the parser's own limits (``ast.parse("-" *
    100000 + "1")`` raises ``MemoryError: Parser stack overflowed``), and that is
    left to propagate — unreachable for a real file, and the same two arms, for
    the same reason, as ``scripts/check_documentation.py``.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError) as exc:
        return [
            f"{filename}: could not be parsed, so any gsplat fit in it went "
            f"unchecked ({type(exc).__name__}: {exc}). THE FLOOR PIN IS NOT "
            "WHAT IS BROKEN HERE — this gate only reads the file. Fix the "
            "file's syntax (or declare its source encoding) and it will pass "
            "again."
        ]
    return [
        f"{filename}:{node.lineno}: {name}(...) does not declare its "
        "background-floor basis at the call site. Declare it there: either a "
        "floor= keyword (a positional fourth argument does not count — name it), "
        "or a kwargs dict visible at the call site, i.e. an inline "
        '**{"floor": ...} or a module-level FIT_KWARGS = dict(..., floor=...) '
        "unpacked as **FIT_KWARGS. If a forwarded dict is already in play, name "
        "floor INSIDE that dict rather than adding a second floor= beside it "
        "(that raises TypeError: got multiple values for keyword argument "
        "'floor'). Benchmarks scored against the raw volume must pin "
        'floor="none" (same basis as the reference); a harness that wants the '
        'shipped background suppression must say floor="auto". See '
        f"{Path(__file__).name} for why."
        for node, name, declared in _guarded_calls(tree)
        if not declared
    ]


def _script_sources() -> list[Path]:
    """Every ``*.py`` under ``scripts/``, recursively."""
    return sorted(SCRIPTS_DIR.rglob("*.py"))


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_scripts_dir_is_present() -> None:
    """Fail loudly rather than pass vacuously if the repo layout moves."""
    assert SCRIPTS_DIR.is_dir(), f"expected a scripts/ directory at {SCRIPTS_DIR}"
    assert _script_sources(), f"no *.py found under {SCRIPTS_DIR}"


def test_every_script_fit_declares_its_floor() -> None:
    """No fitter call under ``scripts/`` may inherit the shipped floor default."""
    offenders: list[str] = []
    for path in _script_sources():
        offenders.extend(
            find_unpinned_fits(
                path.read_bytes(),
                filename=str(path.relative_to(PROJECT_ROOT)),
            )
        )

    assert not offenders, "Unpinned gsplat fits in scripts/:\n" + "\n".join(offenders)


def test_the_gate_actually_sees_a_fitter_call() -> None:
    """Non-vacuity on the real tree: at least one pinned call must be found."""
    pinned = 0
    for path in _script_sources():
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError):
            continue  # the gate above reports it; not this test's business
        pinned += sum(declared for _node, _name, declared in _guarded_calls(tree))
    assert pinned >= 4, (
        "expected the four benchmark pins (the single-pass Pareto fit, the "
        "progressive Pareto fit plus its warm-up, and the progressive PSNR "
        f"fit), found {pinned}"
    )


# ---------------------------------------------------------------------------
# Detector behaviour on planted sources
# ---------------------------------------------------------------------------

_UNPINNED = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

data = fit_gaussian_splats(V, seeds=1000, verbose=False)
"""

_PINNED_LITERAL = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

data = fit_gaussian_splats(V, floor="none", seeds=1000)
"""

_PINNED_AUTO = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

data = fit_gaussian_splats(V, floor="auto", seeds=1000)
"""

_PINNED_VARIABLE = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

spec = "none"
data = fit_gaussian_splats(V, floor=spec, seeds=1000)
"""

_IMPORT_ONLY = '''
"""A module that merely mentions fit_gaussian_splats(V) in prose."""

from luxar.gsplats.fit_gsplats import fit_gaussian_splats

NAME = "fit_gaussian_splats"
__all__ = ["fit_gaussian_splats"]
'''

_QUALIFIED_UNPINNED = """
import luxar.gsplats.fit_progressive_gsplats as fp

data = fp.fit_progressive_gaussian_splats(V, max_splats=100)
"""

_KWARGS_FORWARD = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

def go(V, **kwargs):
    return fit_gaussian_splats(V, **kwargs)
"""

# The luxar-paper pinning idiom: the decision lives in a dict the reader of the
# call site can see, so demanding a second `floor=` beside it would be a
# TypeError, not a fix.
_INLINE_DICT_UNPACK = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

data = fit_gaussian_splats(V, **{"floor": "none", "seeds": 1000})
"""

_MODULE_DICT_UNPACK = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

FIT_KWARGS = dict(seeds=1000, floor="none", verbose=False)

data = fit_gaussian_splats(V, **FIT_KWARGS)
"""

_MODULE_DICT_UNPACK_NO_FLOOR = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

FIT_KWARGS = {"seeds": 1000, "verbose": False}

data = fit_gaussian_splats(V, **FIT_KWARGS)
"""

_IMPORT_ALIAS_UNPINNED = """
from luxar.gsplats import fit_gaussian_splats as fit

data = fit(V, seeds=1000)
"""

_IMPORT_ALIAS_PINNED = """
from luxar.gsplats import fit_gaussian_splats as fit

data = fit(V, seeds=1000, floor="none")
"""

# `floor` is the fourth POSITIONAL parameter of fit_gaussian_splats, so this call
# really is pinned — and is still reported, deliberately: see the module
# docstring's "A positional floor" clause.
_POSITIONAL_FLOOR = """
from luxar.gsplats.fit_gsplats import fit_gaussian_splats

data = fit_gaussian_splats(V, 8000, 0.0, "none")
"""

# The worker-argv builders omit --floor when it is None, leaving the tile worker
# on the shipped `auto` — #1184 through the tiled-parallel path.
_WORKER_CMD_UNPINNED = """
from luxar.gsplats.fit_tiled_parallel import build_worker_cmd, fit_tiled_parallel

builder = build_worker_cmd(argv0, inp, out, 0, 4, 256, 32, seeds="8000")
"""

_WORKER_CMD_PINNED = """
from luxar.gsplats.fit_tiled_parallel import build_worker_cmd

builder = build_worker_cmd(argv0, inp, out, 0, 4, 256, 32, floor="none")
"""

_ALIAS_UNPINNED = """
from luxar.gsplats import fit_tiled_gaussian_splats

data = fit_tiled_gaussian_splats(V, tile_size=256)
"""

_NOT_A_FITTER = """
model.fit(X, y)
scaler.fit_transform(X)
"""

# fit_tiled_parallel has no floor parameter and no **kwargs, so `floor=` there
# is a TypeError, not a pin: the gate must NOT ask for one. Same for
# fit_planned_parallel, whose per-box floor travels through its cmd builder.
_PARALLEL_UNPINNED = """
from luxar.gsplats.fit_tiled_parallel import fit_tiled_parallel

data = fit_tiled_parallel(num_tiles=4, jobs=2, worker_cmd_builder=build)
"""

_PLANNED_PARALLEL_UNPINNED = """
from luxar.gsplats.planner.fit_planned_parallel import fit_planned_parallel

data = fit_planned_parallel(plan, jobs=2, tmp_dir=tmp, worker_cmd_builder=build)
"""

# fit_planned DOES take the spec — inside **fit_kwargs, forwarded to a per-box
# fit_gaussian_splats — so a floor= at the call site is visible and honoured.
_PLANNED_UNPINNED = """
from luxar.gsplats.planner.fit_planned import fit_planned

data = fit_planned(V, plan, device="cuda")
"""

_PLANNED_PINNED = """
from luxar.gsplats.planner.fit_planned import fit_planned

data = fit_planned(V, plan, floor="none", device="cuda")
"""


@pytest.mark.parametrize(
    ("label", "source", "expected"),
    [
        ("unpinned call", _UNPINNED, 1),
        ("pinned none", _PINNED_LITERAL, 0),
        ("pinned auto", _PINNED_AUTO, 0),
        ("pinned via variable", _PINNED_VARIABLE, 0),
        ("import/docstring only", _IMPORT_ONLY, 0),
        ("qualified unpinned call", _QUALIFIED_UNPINNED, 1),
        ("kwargs forwarding", _KWARGS_FORWARD, 1),
        ("inline dict unpack with floor", _INLINE_DICT_UNPACK, 0),
        ("module-level dict with floor", _MODULE_DICT_UNPACK, 0),
        ("module-level dict without floor", _MODULE_DICT_UNPACK_NO_FLOOR, 1),
        ("import alias unpinned", _IMPORT_ALIAS_UNPINNED, 1),
        ("import alias pinned", _IMPORT_ALIAS_PINNED, 0),
        ("positional floor (still flagged)", _POSITIONAL_FLOOR, 1),
        ("worker argv builder unpinned", _WORKER_CMD_UNPINNED, 1),
        ("worker argv builder pinned", _WORKER_CMD_PINNED, 0),
        ("public fit_tiled alias", _ALIAS_UNPINNED, 1),
        ("unrelated .fit()", _NOT_A_FITTER, 0),
        ("fit_tiled_parallel (argv floor)", _PARALLEL_UNPINNED, 0),
        ("fit_planned_parallel (argv floor)", _PLANNED_PARALLEL_UNPINNED, 0),
        ("fit_planned unpinned", _PLANNED_UNPINNED, 1),
        ("fit_planned pinned", _PLANNED_PINNED, 0),
    ],
)
def test_detector_on_planted_sources(label: str, source: str, expected: int) -> None:
    """The detector answers correctly on hand-written positive/negative cases."""
    offenders = find_unpinned_fits(source, filename=f"{label}.py")
    assert len(offenders) == expected, f"{label}: {offenders}"


def test_offender_message_is_actionable() -> None:
    """A failure names the file, the line, the fitter and the fix."""
    (message,) = find_unpinned_fits(_UNPINNED, filename="harness.py")

    assert message.startswith("harness.py:4:")
    assert "fit_gaussian_splats" in message
    assert 'floor="none"' in message


def test_offender_message_does_not_advise_a_typeerror() -> None:
    """The advice must not tell a dict-pinning author to add a second floor=.

    ``fit(V, floor="none", **FIT_KWARGS)`` where the dict already carries the key
    raises ``TypeError: got multiple values for keyword argument 'floor'``, so the
    message has to point at the dict instead — and has to say that a positional
    floor is not what "declared" means.
    """
    (message,) = find_unpinned_fits(_MODULE_DICT_UNPACK_NO_FLOOR, filename="h.py")

    assert "name floor INSIDE that dict" in message
    assert "TypeError" in message
    assert "positional fourth argument does not count" in message


# ---------------------------------------------------------------------------
# Byte sources: encodings this gate must not choke on, and files it cannot read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        # A UTF-8 BOM in front of an otherwise ordinary module.
        ("utf-8 BOM", _UNPINNED.encode("utf-8-sig")),
        # A PEP 263 cookie declaring a non-UTF-8 encoding, with a byte that is
        # not valid UTF-8 (0xE9 = 'é' in latin-1) inside a comment.
        (
            "latin-1 cookie",
            b"# -*- coding: latin-1 -*-\n# caf\xe9 noir\n"
            b"from luxar.gsplats import fit_gaussian_splats\n\n"
            b"data = fit_gaussian_splats(V, seeds=1000)\n",
        ),
    ],
)
def test_unusual_encodings_are_parsed_not_crashed(label: str, raw: bytes) -> None:
    """Reading raw bytes means Python's source-encoding rules apply.

    Both files are perfectly valid Python that a ``read_text(encoding="utf-8")``
    would mangle or reject; the detector must still see the unpinned call.
    """
    offenders = find_unpinned_fits(raw, filename=f"{label}.py")

    assert len(offenders) == 1, f"{label}: {offenders}"
    assert "fit_gaussian_splats" in offenders[0]
    assert "could not be parsed" not in offenders[0]


def test_unparseable_file_becomes_a_finding_not_a_traceback() -> None:
    """A WIP syntax error names the file and says the gate is not the problem."""
    offenders = find_unpinned_fits(b"def broken(:\n    pass\n", filename="wip.py")

    (message,) = offenders
    assert message.startswith("wip.py: could not be parsed")
    assert "SyntaxError" in message
    assert "THE FLOOR PIN IS NOT WHAT IS BROKEN HERE" in message


@pytest.mark.parametrize(
    ("label", "source", "exc_name"),
    [
        # An embedded NUL: SyntaxError since 3.12 ("source code string cannot
        # contain null bytes"), for bytes and str alike.
        ("NUL bytes", b"data = fit_gaussian_splats(V)\n\x00", "SyntaxError"),
        ("NUL str", "data = fit_gaussian_splats(V)\n\x00", "SyntaxError"),
        # A lone surrogate can only reach ast.parse through a str caller, and is
        # the reason the ValueError arm stays: UnicodeEncodeError subclasses it.
        (
            "lone surrogate str",
            "data = fit_gaussian_splats(V)\n\ud800",
            "UnicodeEncodeError",
        ),
    ],
)
def test_pathological_sources_become_findings(
    label: str, source: str | bytes, exc_name: str
) -> None:
    """The ``(SyntaxError, ValueError)`` arms cover what 3.12 actually raises."""
    with pytest.raises(Exception) as raised:  # noqa: B017 - the point is the type
        ast.parse(source)
    assert type(raised.value).__name__ == exc_name, label
    assert isinstance(raised.value, (SyntaxError, ValueError)), label

    (message,) = find_unpinned_fits(source, filename=f"{label}.py")
    assert message.startswith(f"{label}.py: could not be parsed")
    assert exc_name in message


def test_unparseable_temp_script_does_not_raise(tmp_path: Path) -> None:
    """End to end on a real file: read_bytes + parse never escapes as an error."""
    broken = tmp_path / "half_written_harness.py"
    broken.write_bytes(b"\xef\xbb\xbfdef f(:\n")

    (message,) = find_unpinned_fits(broken.read_bytes(), filename=broken.name)

    assert "half_written_harness.py: could not be parsed" in message
