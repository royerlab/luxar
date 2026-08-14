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

``"Neutral"`` (Khronos PBR Neutral) is not an identity anywhere. It subtracts an
offset (0.04 once the channel minimum reaches 0.08, ``x - 6.25*x*x`` below that,
which crushes near-black hardest), so even a well in-gamut colour moves:
``(0.5, 0.5, 0.5) -> (0.46, 0.46, 0.46)``. From peak 0.76 upward it additionally
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
fails the other way round: ``(100, 0, 0)`` clamps to ``(1, 0, 0)`` at full
saturation, but ``(2, 1, 0)`` clamps to ``(1, 1, 0)`` and moves from hue 30° to
hue 60°, and every structure above 1.0 goes flat. ACES shifts hue by design. So
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
resolved in the scope they are USED in: a function-local binding shadows a
module-level one of the same name, as do the function's own parameters, so a
name never reports a value from a scope it cannot see. A conditional expression
(``tone_mapping="Neutral" if hdr else "ACES"``), a ``**{...}`` splat, a
tuple-unpacked constant, a constant bound inside an ``if:``/``try:`` block,
``+=`` string building and a name bound by a ``lambda`` parameter all read as
nothing or as a partial string. That is by design — the guard reads a static
tree, it does not evaluate one — and none of those spellings occur in the demo
package today.
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
#: the driver, and each demo repeats the reason at its own pin.
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
        "intensity, so opacity cancels identically and it is absorption=1.0 "
        "that bounds each node at 1 — their One/OneMinusSrcAlpha composition "
        "keeps the sum bounded too, and the text is a DOM overlay outside the "
        "HDR pass. kappa is a live Layers-panel slider, so dropping it below 1 "
        "pushes a node over range where the 'None' pin clips flat",
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


def _string_constants(body: list[ast.stmt]) -> dict[str, str]:
    """``NAME = "..."`` bindings directly in ``body``, annotated ones included.

    ``TONE: Final = "Neutral"`` at module level, then
    ``ViewerConfig(tone_mapping=TONE)``, is a spelling this package's
    ``Final``-constant convention makes likely; without this it would read as
    a non-literal and slip past the policy entirely. A function body is read
    the same way, so a local ``TONE = "Neutral"`` is caught in its own right
    (and shadows a module constant of the same name — see
    :func:`_function_scope`).
    """
    consts: dict[str, str] = {}
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
                consts[target.id] = value.value
    return consts


def _as_string(node: ast.expr, consts: dict[str, str]) -> str | None:
    """A ``tone_mapping`` value as a string, or ``None`` if it carries no policy."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    return None


def _function_scope(
    node: ast.FunctionDef | ast.AsyncFunctionDef, consts: dict[str, str]
) -> dict[str, str]:
    """``consts`` as seen from inside ``node``'s body.

    Its own string constants win, and its parameters shadow an outer constant of
    the same name — a parameter carries no policy of its own, which is what
    keeps ``tone_mapping=tone_mapping`` (a forwarded value) from being read as
    whatever a module happens to bind that name to.
    """
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    scope = dict(consts)
    for param in params:
        scope.pop(param.arg, None)
    scope.update(_string_constants(node.body))
    return scope


def _keyword_values(node: ast.Call, consts: dict[str, str]) -> list[str]:
    """``tone_mapping=`` values passed to one call."""
    values: list[str] = []
    for kw in node.keywords:
        if kw.arg == "tone_mapping":
            value = _as_string(kw.value, consts)
            if value is not None:
                values.append(value)
    return values


def _default_values(args: ast.arguments, consts: dict[str, str]) -> list[str]:
    """``tone_mapping`` signature defaults of one function."""
    positional = args.posonlyargs + args.args
    pairs = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults)
    ) + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
    values: list[str] = []
    for arg, default in pairs:
        if arg.arg == "tone_mapping":
            value = _as_string(default, consts)
            if value is not None:
                values.append(value)
    return values


def _collect(node: ast.AST, consts: dict[str, str], found: list[str]) -> None:
    """Walk ``node``, appending every statically known ``tone_mapping`` value.

    ``consts`` is the name → string binding visible AT ``node``: module-level
    constants, overridden by those of each enclosing function body. Descending
    with a per-scope map (rather than walking the whole tree against one
    module-level map) is what makes a name resolve to the value the interpreter
    would see, instead of to a same-named module constant it shadows.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Defaults and decorators are evaluated in the ENCLOSING scope; only the
        # body sees the function's own names.
        found.extend(_default_values(node.args, consts))
        for outer in [node.args, *node.decorator_list]:
            _collect(outer, consts, found)
        scope = _function_scope(node, consts)
        for stmt in node.body:
            _collect(stmt, scope, found)
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


def test_every_demo_tone_mapping_is_aces_or_a_justified_exception() -> None:
    """No demo pins a non-ACES tone mapping without a row in EXCEPTIONS."""
    offenders: list[str] = []
    for path in scanned_demo_modules():
        allowed = EXCEPTIONS.get(path.name, (("ACES",), ""))[0]
        for value in _tone_mappings(path):
            if value not in allowed:
                offenders.append(
                    f"{path.name}: tone_mapping={value!r} (expected "
                    f"{' or '.join(repr(a) for a in allowed)})"
                )
    assert not offenders, (
        "demo tone-mapping policy (#1459): ACES is the house default, and where "
        "it is wrong the choice is made by RANGE — inside [0, 1] 'None' is an "
        "exact passthrough, and 'Neutral' is not an identity anywhere (it "
        "subtracts an offset even below its knee, and over range it keeps the "
        "hue angle but sheds chroma). Fix the pin, or add the module to "
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
        assert set(values) <= set(allowed), (
            f"{name} pins {sorted(set(values))}, not {sorted(allowed)}"
        )
        assert "ACES" not in values, (
            f"{name} pins ACES, so its EXCEPTIONS row is dead — delete it"
        )


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

    # A parameter shadows an outer constant too, so a forwarded value stays
    # policy-free even when the module binds that very name.
    param = tmp_path / "demo_param_shadow.py"
    param.write_text(
        'tone_mapping = "Neutral"\n'
        "def build(tone_mapping):\n"
        "    return ViewerConfig(tone_mapping=tone_mapping)\n",
        encoding="utf-8",
    )
    assert _tone_mappings(param) == []
