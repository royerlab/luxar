"""One tone-mapping policy for the demos: ACES unless a demo argues otherwise.

The house rule (issue #1459) is that ACES is the right choice for almost every
scene — its filmic rolloff keeps bright structure from clipping flat — and that
where ACES is wrong, the answer is decided by RANGE rather than by reaching for
``"Neutral"``.

Inside [0, 1], ``"None"`` is an exact passthrough: the viewer maps it to
``THREE.NoToneMapping``, which the mega-shader aliases to its Linear mode (a
clamp to [0, 1]), and exposure/offset/gamma still apply because the shader runs
them *before* the tone-mapping switch. ``"Linear"`` aliases to that same clamp
mode, so it is bit-identical to ``"None"``; ``"None"`` is simply the clearer
name for the passthrough, which is why the policy below names that one.

``"Neutral"`` (Khronos PBR Neutral) is not a passthrough. It subtracts an offset
taken from the channel MINIMUM (0.04 once that minimum reaches 0.08,
``x - 6.25*x*x`` below that, which crushes near-black hardest), so even a well
in-gamut colour moves: ``(0.5, 0.5, 0.5) -> (0.46, 0.46, 0.46)``. The one class
it does leave alone is a colour whose minimum is exactly 0 below the knee — a
fully saturated hue, or black — where the offset is 0 too; anything desaturated
at all moves. From peak 0.76 upward it additionally
compresses the peak and mixes toward the grey EQUAL to that compressed peak,
with weight ``g = 1 - 1/(0.15*(peak - newPeak) + 1)``. Running the three.js
formula: ``(1, 0, 0) -> (0.880, 0.016, 0.016)``,
``(8, 0.5, 8) -> (0.992, 0.535, 0.992)`` (g = 0.51) and
``(100, 0, 0) -> (0.999, 0.936, 0.936)`` (g = 0.94). Just above the knee the mix
is negligible (a post-offset peak of 0.9 greys toward 0.848 at g = 0.008); it
only bites well over 1.0.

Because that mix is "scale every channel, then add the same amount to all", what
Neutral costs over range is CHROMA, not hue: it holds the HSV hue angle exactly
(in the shader's linear working space — the sRGB encode that follows can still
move a measured hue reading by a degree or two) and takes ``(100, 0, 0)`` to
saturation 0.06 — essentially white. A hard clamp
is not the clean opposite it looks like: it holds full saturation only where
the darkest channel is already 0 (``(100, 0, 0)`` clamps to ``(1, 0, 0)``,
while ``(2, 0.5, 0.5)`` clamps to ``(1, 0.5, 0.5)`` and drops from saturation
0.75 to 0.5), and ``(2, 1, 0)`` clamps to ``(1, 1, 0)`` and moves from hue 30°
to hue 60°, with every structure above 1.0 going flat. ACES shifts hue by
design. So
no operator is faithful over range, and the choice is which distortion the scene
can afford: where colour is a hue encoding, Neutral is the hue-exact option and
the chroma of the brightest peaks is what it costs.

So over range the operational rule is: bring the scene back into
[0, 1] with exposure/intensity and use ``"None"``, accept ACES's filmic rolloff,
or keep Neutral where hue matters more than peak chroma — and choose between
them with an actual render (as #1459 did for the FlyWire connectome), never a
blind flip.

This guard pins that policy so a new demo cannot copy-paste a ``"Neutral"``
into the tree without stating a reason: every statically known ``tone_mapping``
in the demo package — a literal, a signature default, or a string constant —
must be ``"ACES"`` unless its module is in :data:`EXCEPTIONS` below, which
carries the justification with it.

Scope and method mirror the other AST lints in this directory: the module set
comes from ``_scanned_modules`` (a denylist, so a new demo joins automatically),
paths are resolved relative to THIS file rather than through ``import luxar``
(so the guard always reads the checkout it ships in), and nothing is imported —
the demo modules are heavy. A module that sets no tone mapping at all is out of
scope: the viewer's own default is ACES, so silence already complies.

"Statically known" is a real boundary, not a hedge. The scan resolves exactly
three shapes — an inline literal call keyword, a signature default, and a plain
``NAME = "..."`` string constant — and reads nothing else. Constants are
resolved in the scope they are USED in: a function-local or class-body binding
shadows a module-level one of the same name, as do the parameters of a ``def``
or a ``lambda`` and the ``for`` targets of a comprehension, so a name never
reports a value from a scope it cannot see.
A local binding the scan cannot READ (a conditional, a call, a loop variable,
an import) shadows just as hard — it hides the outer value rather than letting
it leak in, so such a name reports nothing at all.
(A class body is read the permissive way round — its attributes stay visible to
its own methods, which the interpreter would not do. That over-reports rather
than misses, which is the direction this guard errs in everywhere.) Within
one scope the scan does not track statement ORDER; a name rebound to a second
string reports both values, so a later compliant binding cannot mask an earlier
stray one. A conditional expression
(``tone_mapping="Neutral" if hdr else "ACES"``), a ``**{...}`` splat, a
tuple-unpacked constant, a constant bound inside an ``if:``/``try:`` block and
``+=`` string building all read as nothing or as a partial string. That is by
design — the guard reads a static tree, it does not evaluate one — and none of
those spellings occur in the demo package today.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ._scanned_modules import scanned_demo_modules

#: The demos allowed to deviate from ACES: module name -> (allowed values, why).
#:
#: Two shapes of exception, and only two. ``"None"`` — every colour in the
#: scene already sits inside [0, 1], so a passthrough is exact and any tone
#: mapper is pure distortion. ``"Neutral"`` — the scene runs OVER range and its
#: colour is a hue encoding, which is exactly the case Neutral is right for:
#: over range it is the one operator that holds the hue angle, at the cost of
#: the chroma of the brightest peaks (``"None"`` clips and can shift hue when
#: several channels clip unequally; ACES shifts hue by design). That trade is
#: the reason for the pin, so moving one of these — at ACES, or at ``"None"``
#: after an exposure/intensity re-tune — trades hue fidelity for chroma and
#: needs a live A/B, not a blind flip. The one Neutral row that is NOT this
#: case (``_interop_common.py``) says so in its own reason. Each entry must name
#: the driver, and each demo repeats the reason at its own pin. A row exempts
#: the scene that needs it, not the whole module: the rest of the file may
#: still pin the ACES default (see :func:`_allowed_values`), and a row whose
#: module pins nothing but ACES is dead and must be deleted.
EXCEPTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    # --- in-gamut: no tone mapping is the exact choice -------------------
    "demo_ocean_currents_earth.py": (
        ("None",),
        "Blue Marble texture and the blue->white speed LUT are both [0, 1]; "
        "blending is normal, LINE_INTENSITY is 1.0",
    ),
    "demo_gsplats_recipes_tribolium.py": (
        ("None",),
        "hue/shade recipe palette comes from colorsys in [0, 1]; the geometry "
        "is six volumetric gsplat nodes (one per RECIPES entry), and that mode "
        "emits color*(1-exp(-tau))/absorption with tau = absorption*opacity*"
        "intensity. opacity does not drop out of that — the opacity*intensity "
        "ray mass cancels against the tau under the S(tau) self-screening, and "
        "what is left, (1-exp(-tau)), is below 1 for ANY tau >= 0 — so it is "
        "absorption=1.0 that bounds each node at 1, whatever opacity and "
        "intensity do. Their One/OneMinusSrcAlpha composition "
        "keeps the sum bounded too, and the text is a DOM overlay outside the "
        "HDR pass. kappa is a live Layers-panel slider, and dropping it below "
        "1 raises that bound to 1/kappa, so a node can go over range where the "
        "'None' pin clips flat",
    ),
    # --- over range (Neutral is hue-exact there), or range unproven ------
    "_interop_common.py": (
        ("Neutral",),
        "unproven range, needs a render check: imported classical captures "
        "carry per-splat RGB clipped to [0, 1], but "
        "demo_gsplats_interop_sog_matrixcity builds recipe='overview', whose "
        "substitutive merge emits amplitudes with no bound of 1 and which the "
        "gsplat 'normal' shader feeds straight into RGB — so nothing here has "
        "been shown to be in gamut (#1459)",
    ),
    "demo_nd_transforms.py": (
        ("Neutral",),
        "the bench's lit palette is deliberately HDR (C_LIT peaks at 1.9, "
        "CH_LIT at 2.4) and its colour is an exact encoding — a red R must "
        "read as red — so Neutral's hue-exactness is what is being bought, at "
        "the cost of the brightest glyphs' chroma",
    ),
    "demo_biodiversity_planetary_scale.py": (
        ("Neutral",),
        "GLOBE_INTENSITY 4.88 and OCCURRENCE_INTENSITY 100.0, over a "
        "CATEGORICAL taxonomic palette that must stay legible against the "
        "globe: over range Neutral is the hue-exact option, and the chroma of "
        "the hottest occurrence peaks is the accepted cost",
    ),
    "demo_gsplats_3d_tribolium_embryo.py": (
        ("Neutral",),
        "exposure=1.97 (~2 stops) because the blending mode projects peaks "
        "instead of integrating along the ray, so the scene runs over range; "
        "the pin was verified against this volume and Neutral rolls the peaks "
        "off instead of clipping them flat",
    ),
}

#: A floor, not a count: ~34 demo modules pin a tone mapping today. Well below
#: that means the AST scan stopped finding pins (a renamed keyword, a walk that
#: no longer descends) rather than that demos were deleted — without it a
#: broken scanner would pass this guard silently.
MIN_MODULES_WITH_PINS = 25


def _string_constants(body: list[ast.stmt]) -> dict[str, tuple[str, ...]]:
    """``NAME = "..."`` bindings directly in ``body``, annotated ones included.

    ``TONE: Final = "Neutral"`` at module level, then
    ``ViewerConfig(tone_mapping=TONE)``, is a spelling this package's
    ``Final``-constant convention makes likely; without this it would read as
    a non-literal and slip past the policy entirely. A function or class body is
    read the same way, so a local ``TONE = "Neutral"`` is caught in its own
    right (and shadows a module constant of the same name — see
    :func:`_function_scope`).

    A name bound more than once keeps EVERY distinct value it is bound to. The
    scan has no notion of where in a body a use sits, so keeping only the last
    binding would let ``TONE = "Neutral"``,
    ``ViewerConfig(tone_mapping=TONE)``, ``TONE = "ACES"`` report ``"ACES"``
    alone and hide a real pin behind a later compliant one. Reporting both is
    the conservative direction: the policy check sees the ``"Neutral"``, and the
    worst case is a spurious failure on a demo that genuinely rebinds a
    tone-mapping constant — which none does, and which an EXCEPTIONS row
    settles.
    """
    consts: dict[str, list[str]] = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            targets: list[ast.expr] = list(stmt.targets)
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
            value = stmt.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bound = consts.setdefault(target.id, [])
                if value.value not in bound:
                    bound.append(value.value)
    return {name: tuple(bound) for name, bound in consts.items()}


def _as_strings(node: ast.expr, consts: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The ``tone_mapping`` values a node can carry; empty if it carries none."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return consts.get(node.id, ())
    return ()


def _without_params(
    args: ast.arguments, consts: dict[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """``consts`` with every parameter name of ``args`` removed.

    A parameter shadows an outer constant of the same name and carries no
    policy of its own, which is what keeps ``tone_mapping=tone_mapping`` (a
    forwarded value) from being read as whatever a module happens to bind that
    name to.
    """
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    scope = dict(consts)
    for param in params:
        scope.pop(param.arg, None)
    return scope


def _target_names(target: ast.expr) -> set[str]:
    """Every name a comprehension ``for`` target binds, tuple targets included."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _names_bound_here(node: ast.AST) -> set[str]:
    """Every name ``node`` itself binds, ignoring whatever its children bind."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    if isinstance(node, ast.ExceptHandler) and node.name:
        return {node.name}
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return {node.name}
    if isinstance(node, ast.MatchMapping) and node.rest:
        return {node.rest}
    return set()


def _bound_names(body: list[ast.stmt]) -> set[str]:
    """Every name ``body`` binds in a scope of its OWN, whatever shape the value has.

    :func:`_string_constants` reads only the bindings it can resolve to a
    string, so on its own it cannot say "this name is local and I cannot see
    its value". Without that, an unreadable local — ``TONE = "Neutral" if hdr
    else "ACES"``, ``TONE = pick()``, ``for TONE in ...``, ``from x import
    TONE``, a ``case TONE:`` capture — would leave the module's ``TONE``
    visible and the scan
    would report a value this scope never sees. That is wrong in both
    directions: it hides a stray pin behind a compliant module constant, and it
    fails a module for a ``"Neutral"`` the call never receives. So every name
    bound in a body is dropped from the inherited scope first, and only the
    readable ones are put back.

    The walk stops at nested ``def``/``class``/``lambda`` bodies and at
    comprehensions, which bind in scopes of their own (a nested definition's
    own NAME is still bound here).
    """
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(
            node,
            (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            return
        names.update(_names_bound_here(node))
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in body:
        visit(stmt)
    return names


def _body_scope(
    body: list[ast.stmt], consts: dict[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """``consts`` as seen from inside ``body``: its own bindings win."""
    bound = _bound_names(body)
    scope = {name: values for name, values in consts.items() if name not in bound}
    scope.update(_string_constants(body))
    return scope


def _function_scope(
    node: ast.FunctionDef | ast.AsyncFunctionDef, consts: dict[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """``consts`` as seen from inside ``node``'s body: own constants win."""
    return _body_scope(node.body, _without_params(node.args, consts))


def _keyword_values(node: ast.Call, consts: dict[str, tuple[str, ...]]) -> list[str]:
    """``tone_mapping=`` values passed to one call."""
    values: list[str] = []
    for kw in node.keywords:
        if kw.arg == "tone_mapping":
            values.extend(_as_strings(kw.value, consts))
    return values


def _default_values(
    args: ast.arguments, consts: dict[str, tuple[str, ...]]
) -> list[str]:
    """``tone_mapping`` signature defaults of one function."""
    positional = args.posonlyargs + args.args
    pairs = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults)
    ) + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    values: list[str] = []
    for arg, default in pairs:
        if arg.arg == "tone_mapping":
            values.extend(_as_strings(default, consts))
    return values


def _collect_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    consts: dict[str, tuple[str, ...]],
    found: list[str],
) -> None:
    """Collect from a ``def``: defaults and decorators outside, body inside."""
    # Defaults and decorators are evaluated in the ENCLOSING scope; only the
    # body sees the function's own names.
    found.extend(_default_values(node.args, consts))
    for outer in [node.args, *node.decorator_list]:
        _collect(outer, consts, found)
    scope = _function_scope(node, consts)
    for stmt in node.body:
        _collect(stmt, scope, found)


def _collect_class(
    node: ast.ClassDef, consts: dict[str, tuple[str, ...]], found: list[str]
) -> None:
    """Collect from a ``class``: bases/keywords/decorators outside, body inside."""
    # Bases, keywords and decorators are evaluated in the ENCLOSING scope;
    # the body binds its own names on top of it. Methods do not really see
    # those class attributes, but descending with them visible over-reports
    # rather than misses, which is the safe direction for this guard.
    for outer in [*node.bases, *node.keywords, *node.decorator_list]:
        _collect(outer, consts, found)
    scope = _body_scope(node.body, consts)
    for stmt in node.body:
        _collect(stmt, scope, found)


def _collect_lambda(
    node: ast.Lambda, consts: dict[str, tuple[str, ...]], found: list[str]
) -> None:
    """Collect from a ``lambda``: defaults outside, body without its params."""
    # Same split as a `def`, and for the same reason: only the body sees the
    # lambda's parameters, so a forwarded one carries no policy.
    found.extend(_default_values(node.args, consts))
    _collect(node.args, consts, found)
    _collect(node.body, _without_params(node.args, consts), found)


def _collect_comprehension(
    node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    consts: dict[str, tuple[str, ...]],
    found: list[str],
) -> None:
    """Collect from a comprehension, dropping each ``for`` target as it binds."""
    # A comprehension is a scope of its own: its `for` targets are bound
    # inside it, the scan cannot read them, and so — like any other
    # unreadable local — they must hide a same-named outer constant rather
    # than let it leak into the element expression. Only the FIRST iterable
    # is evaluated in the enclosing scope; every later one, the `if`
    # clauses and the element see the targets bound before them.
    scope = consts
    for generator in node.generators:
        _collect(generator.iter, scope, found)
        bound = _target_names(generator.target)
        scope = {n: v for n, v in scope.items() if n not in bound}
        for condition in generator.ifs:
            _collect(condition, scope, found)
    elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
    for element in elements:
        _collect(element, scope, found)


def _collect(
    node: ast.AST, consts: dict[str, tuple[str, ...]], found: list[str]
) -> None:
    """Walk ``node``, appending every statically known ``tone_mapping`` value.

    ``consts`` is the name → string bindings visible AT ``node``: module-level
    constants, overridden by those of each enclosing function, class body or
    comprehension.
    Descending
    with a per-scope map (rather than walking the whole tree against one
    module-level map) is what makes a name resolve to the value the interpreter
    would see, instead of to a same-named module constant it shadows.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        _collect_function(node, consts, found)
        return
    if isinstance(node, ast.ClassDef):
        _collect_class(node, consts, found)
        return
    if isinstance(node, ast.Lambda):
        _collect_lambda(node, consts, found)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        _collect_comprehension(node, consts, found)
        return
    if isinstance(node, ast.Call):
        found.extend(_keyword_values(node, consts))
    for child in ast.iter_child_nodes(node):
        _collect(child, consts, found)


def _tone_mappings(path: Path) -> list[str]:
    """Every statically known ``tone_mapping`` value written in ``path``.

    Covers both spellings the demo package uses: a call keyword
    (``ViewerConfig(tone_mapping="ACES")``) and a function-signature default
    (``_interop_common.build_interop_scene(..., tone_mapping: str = "Neutral")``),
    each either as a literal or as a string constant resolved in its own scope.
    A value that is neither (``tone_mapping=tone_mapping``, a forwarded
    parameter) carries no policy of its own and is skipped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    _collect(tree, _string_constants(tree.body), found)
    return found


def _allowed_values(module_name: str) -> tuple[str, ...]:
    """The tone mappings ``module_name`` may pin: ACES, plus its exception row.

    ACES is legal EVERYWHERE, in an exception module as much as anywhere else.
    A row exempts the scene that needs something else; it must not outlaw the
    house default in the rest of the file. Without this, a module holding one
    justified ``"Neutral"`` scene alongside an ordinary ACES one would have no
    legal spelling at all — the row rejects the ACES pin, widening the row to
    include ``"ACES"`` trips the dead-row check, and deleting the row rejects
    the Neutral pin.
    """
    return EXCEPTIONS.get(module_name, ((), ""))[0] + ("ACES",)


def test_every_demo_tone_mapping_is_aces_or_a_justified_exception() -> None:
    """No demo pins a non-ACES tone mapping without a row in EXCEPTIONS."""
    offenders: list[str] = []
    for path in scanned_demo_modules():
        allowed = _allowed_values(path.name)
        for value in _tone_mappings(path):
            if value not in allowed:
                offenders.append(
                    f"{path.name}: tone_mapping={value!r} (expected "
                    f"{' or '.join(repr(a) for a in allowed)})"
                )
    assert not offenders, (
        "demo tone-mapping policy (#1459): ACES is the house default, and where "
        "it is wrong the choice is made by RANGE — inside [0, 1] 'None' is an "
        "exact passthrough, while 'Neutral' is not (it subtracts a "
        "channel-minimum offset even below its knee, so anything but a fully "
        "saturated colour moves, and over range it keeps the hue angle but "
        "sheds chroma). Fix the pin, or add the module to "
        "EXCEPTIONS with the reason:\n  " + "\n  ".join(sorted(offenders))
    )


def test_the_scan_actually_finds_the_pins() -> None:
    """A broken scanner must fail here rather than pass the policy vacuously."""
    with_pins = [p.name for p in scanned_demo_modules() if _tone_mappings(p)]
    assert len(with_pins) >= MIN_MODULES_WITH_PINS, (
        f"only {len(with_pins)} demo modules were seen to pin a tone mapping "
        f"(expected at least {MIN_MODULES_WITH_PINS}) — the AST scan, not the "
        f"demos, is what changed: {sorted(with_pins)}"
    )


def test_every_exception_is_real_and_justified() -> None:
    """Each EXCEPTIONS row names a module that exists, deviates, and says why."""
    scanned = {p.name: p for p in scanned_demo_modules()}
    for name, (allowed, reason) in EXCEPTIONS.items():
        assert name in scanned, f"EXCEPTIONS names {name}, which is not scanned"
        assert reason.strip(), f"EXCEPTIONS row for {name} carries no reason"
        values = _tone_mappings(scanned[name])
        assert values, (
            f"EXCEPTIONS names {name}, but it pins no tone mapping at all — "
            f"delete the row; silence already means the viewer's ACES default"
        )
        deviations = set(values) - {"ACES"}
        assert deviations, (
            f"{name} pins nothing but ACES, so its EXCEPTIONS row is dead — delete it"
        )
        assert deviations <= set(allowed), (
            f"{name} pins {sorted(deviations)}, not {sorted(allowed)}"
        )


def test_an_exception_row_still_allows_aces() -> None:
    """A row exempts one scene; the rest of its module may still pin ACES.

    An exception is per-MODULE only because that is the granularity a static
    scan has. It must not read as "this file may not use the house default",
    which would leave a module with one justified deviation and one ordinary
    ACES scene nothing legal to write.
    """
    for name, (allowed, _reason) in EXCEPTIONS.items():
        assert "ACES" in _allowed_values(name), name
        assert set(allowed) <= set(_allowed_values(name)), name
    assert _allowed_values("demo_not_in_exceptions.py") == ("ACES",)


def test_the_guard_detects_a_stray_neutral(tmp_path: Path) -> None:
    """The three spellings the scan claims to cover really are covered.

    Those three and no more: an inline literal call keyword, a signature
    default, and a string constant — the last one resolved in the scope it is
    used in (plus the forwarded-parameter case, which correctly carries no
    policy). The dynamic spellings listed in this module's docstring stay out of
    reach on purpose, so this is a completeness check on the documented surface,
    not on every way a string can reach ``tone_mapping``.
    """
    call = tmp_path / "demo_stray_call.py"
    call.write_text('ViewerConfig(tone_mapping="Neutral")\n', encoding="utf-8")
    assert _tone_mappings(call) == ["Neutral"]

    default = tmp_path / "_stray_default.py"
    default.write_text(
        'def build(*, tone_mapping: str = "Neutral") -> None: ...\n', encoding="utf-8"
    )
    assert _tone_mappings(default) == ["Neutral"]

    # A module-level constant, plain and annotated — the `Final` spelling this
    # package favours, which a literals-only scan would miss entirely.
    named = tmp_path / "demo_named_constant.py"
    named.write_text(
        'TONE: Final = "Neutral"\n'
        'FALLBACK = "Neutral"\n'
        "ViewerConfig(tone_mapping=TONE)\n"
        "def build(*, tone_mapping: str = FALLBACK) -> None: ...\n",
        encoding="utf-8",
    )
    assert _tone_mappings(named) == ["Neutral", "Neutral"]

    forwarded = tmp_path / "demo_forwarded.py"
    forwarded.write_text(
        "def build(tone_mapping):\n"
        "    return ViewerConfig(tone_mapping=tone_mapping)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(forwarded) == []

    # A constant is read in the scope it is USED in: a function-local binding
    # is caught in its own right and shadows the module-level one, rather than
    # the scan reporting the module's value for a pin that never uses it.
    shadowed = tmp_path / "demo_shadowed_constant.py"
    shadowed.write_text(
        'TONE = "ACES"\n'
        "def build():\n"
        '    TONE = "Neutral"\n'
        "    return ViewerConfig(tone_mapping=TONE)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(shadowed) == ["Neutral"]

    # A class body is a scope of its own: a constant bound there is resolved
    # for the pins inside it, rather than reading as a non-literal.
    class_scoped = tmp_path / "demo_class_constant.py"
    class_scoped.write_text(
        'TONE = "ACES"\n'
        "class Scene:\n"
        '    TONE = "Neutral"\n'
        "    config = ViewerConfig(tone_mapping=TONE)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(class_scoped) == ["Neutral"]

    # A parameter shadows an outer constant too, so a forwarded value stays
    # policy-free even when the module binds that very name — for a lambda's
    # parameters as much as for a def's.
    param = tmp_path / "demo_param_shadow.py"
    param.write_text(
        'tone_mapping = "Neutral"\n'
        "def build(tone_mapping):\n"
        "    return ViewerConfig(tone_mapping=tone_mapping)\n"
        "make = lambda tone_mapping: ViewerConfig(tone_mapping=tone_mapping)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(param) == []


def test_an_unreadable_local_hides_the_outer_constant(tmp_path: Path) -> None:
    """A local binding the scan cannot read still shadows the module constant.

    Shadowing must not depend on whether the scan happens to understand the
    local VALUE. If it did, ``TONE = "Neutral" if hdr else "ACES"`` inside a
    function would leave the module's own ``TONE`` visible, and the scan would
    report a value that call never receives — a spurious failure where the
    module constant is non-ACES, and a hidden pin where it is ACES.
    """
    conditional = tmp_path / "demo_conditional_local.py"
    conditional.write_text(
        'TONE = "Neutral"\n'
        "def build(hdr):\n"
        '    TONE = "Neutral" if hdr else "ACES"\n'
        "    return ViewerConfig(tone_mapping=TONE)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(conditional) == []

    # Same for the other shapes a local binding takes: a call, a loop variable,
    # an import, a string bound inside an `if:` (which `_string_constants` does
    # not descend into), a `match` capture, and a class attribute.
    for name, source in {
        "demo_local_call.py": (
            'TONE = "Neutral"\n'
            "def build():\n"
            "    TONE = pick()\n"
            "    return ViewerConfig(tone_mapping=TONE)\n"
        ),
        "demo_local_loop.py": (
            'TONE = "Neutral"\n'
            "def build():\n"
            "    for TONE in options():\n"
            "        ViewerConfig(tone_mapping=TONE)\n"
        ),
        "demo_local_import.py": (
            'TONE = "Neutral"\n'
            "def build():\n"
            "    from ._palette import TONE\n"
            "    return ViewerConfig(tone_mapping=TONE)\n"
        ),
        "demo_local_branch.py": (
            'TONE = "Neutral"\n'
            "def build(hdr):\n"
            "    if hdr:\n"
            '        TONE = "ACES"\n'
            "    return ViewerConfig(tone_mapping=TONE)\n"
        ),
        "demo_local_match.py": (
            'TONE = "Neutral"\n'
            "def build(request):\n"
            "    match request:\n"
            "        case {'tone': TONE}:\n"
            "            return ViewerConfig(tone_mapping=TONE)\n"
        ),
        "demo_class_dynamic.py": (
            'TONE = "Neutral"\n'
            "class Scene:\n"
            "    TONE = pick()\n"
            "    config = ViewerConfig(tone_mapping=TONE)\n"
        ),
        # A comprehension's `for` target is a local of the comprehension's own
        # scope — in the element expression, in a later generator's iterable,
        # and in an `if` clause alike.
        "demo_comprehension_target.py": (
            'TONE = "Neutral"\n'
            "scenes = [ViewerConfig(tone_mapping=TONE) for TONE in options()]\n"
        ),
        "demo_comprehension_dict.py": (
            'TONE = "Neutral"\n'
            "scenes = {k: ViewerConfig(tone_mapping=TONE) for k, TONE in items()}\n"
        ),
        "demo_comprehension_if.py": (
            'TONE = "Neutral"\n'
            "scenes = [x for TONE in options() if ViewerConfig(tone_mapping=TONE)]\n"
        ),
    }.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        assert _tone_mappings(path) == [], name

    # But a nested scope is not this one: a `def` or a comprehension binding
    # that name inside the function must not blind the function's own pin.
    nested = tmp_path / "demo_nested_binding.py"
    nested.write_text(
        'TONE = "Neutral"\n'
        "def build():\n"
        "    def inner(TONE):\n"
        "        return TONE\n"
        "    rest = [TONE for TONE in options()]\n"
        "    return ViewerConfig(tone_mapping=TONE), inner, rest\n",
        encoding="utf-8",
    )
    assert _tone_mappings(nested) == ["Neutral"]

    # And the shadowing stops at the names a comprehension actually binds: a
    # pin inside one that does NOT rebind the constant still reads it, as does
    # the first iterable, which is evaluated out in the enclosing scope.
    comp_outer = tmp_path / "demo_comprehension_outer.py"
    comp_outer.write_text(
        'TONE = "Neutral"\n'
        "scenes = [ViewerConfig(tone_mapping=TONE) for _ in options()]\n"
        "more = [x for y in options(ViewerConfig(tone_mapping=TONE))]\n",
        encoding="utf-8",
    )
    assert _tone_mappings(comp_outer) == ["Neutral", "Neutral"]


def test_a_rebound_constant_cannot_mask_a_stray_neutral(tmp_path: Path) -> None:
    """A name bound twice reports both values, so order cannot hide a pin.

    The scan resolves a name per SCOPE, not per statement, so ``TONE`` below is
    ambiguous to it. Reporting only the last binding would let a compliant
    reassignment placed after the call hide the ``"Neutral"`` the scene actually
    receives; reporting both keeps the policy check on the safe side.
    """
    module_level = tmp_path / "demo_rebound_module.py"
    module_level.write_text(
        'TONE = "Neutral"\nViewerConfig(tone_mapping=TONE)\nTONE = "ACES"\n',
        encoding="utf-8",
    )
    assert sorted(_tone_mappings(module_level)) == ["ACES", "Neutral"]

    local = tmp_path / "demo_rebound_local.py"
    local.write_text(
        "def build():\n"
        '    TONE = "Neutral"\n'
        "    cfg = ViewerConfig(tone_mapping=TONE)\n"
        '    TONE = "ACES"\n'
        "    return cfg, TONE\n",
        encoding="utf-8",
    )
    assert sorted(_tone_mappings(local)) == ["ACES", "Neutral"]


def test_each_part_of_a_def_is_visited_exactly_once(tmp_path: Path) -> None:
    """A canary on the walk's own visit order, one visit per part.

    :func:`_collect_function` visits the signature defaults, the decorators and
    the body as three separate steps, so permuting them would silently permute
    the results and nothing else here would notice: every other assertion in
    this file compares a set, a sorted list, or a run of the same value. A
    DROPPED visit is better covered — losing the defaults or the body reddens
    several other tests in this file — but the decorator visit is pinned here
    and nowhere else.

    The order below is that visit order, NOT the interpreter's: CPython
    evaluates a decorator's own expression (``register(tone_mapping="Deco")``)
    *before* the signature defaults, so only "body last" is evaluation-faithful.
    Nothing depends on the order — the policy check reads the values as a set —
    so it is pinned purely as a canary. Reordering the visits to match the
    interpreter is a legitimate change; update this list with it rather than
    reading the failure as a regression.
    """
    ordered = tmp_path / "demo_walk_order.py"
    ordered.write_text(
        '@register(tone_mapping="Deco")\n'
        'def build(*, tone_mapping: str = "Default") -> None:\n'
        '    ViewerConfig(tone_mapping="Body")\n',
        encoding="utf-8",
    )
    assert _tone_mappings(ordered) == ["Default", "Deco", "Body"]
