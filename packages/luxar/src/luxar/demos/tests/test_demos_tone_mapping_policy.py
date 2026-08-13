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
in the demo package — a literal, a signature default, or a module-level string
constant — must be ``"ACES"`` unless its module is in :data:`EXCEPTIONS` below,
which carries the justification with it.

Scope and method mirror the other AST lints in this directory: the module set
comes from ``_scanned_modules`` (a denylist, so a new demo joins automatically),
paths are resolved relative to THIS file rather than through ``import luxar``
(so the guard always reads the checkout it ships in), and nothing is imported —
the demo modules are heavy. A module that sets no tone mapping at all is out of
scope: the viewer's own default is ACES, so silence already complies.

"Statically known" is a real boundary, not a hedge. The scan resolves exactly
three shapes — an inline literal call keyword, a signature default, and a
module-level string constant — and reads nothing else. A conditional expression
(``tone_mapping="Neutral" if hdr else "ACES"``), a ``**{...}`` splat, a
tuple-unpacked constant, a constant bound inside an ``if:``/``try:`` block and
``+=`` string building all read as nothing or as a partial string, and a
function-LOCAL rebinding of a module-level name resolves to the MODULE value
(so it can report the wrong one). That is by design — the guard reads a static
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
    # --- in flight -------------------------------------------------------
    # #1459's own half of the sweep (PR #1527) moves this demo from "Neutral"
    # to a re-tuned ACES. Both values are accepted until that lands; delete
    # this row once it has, so the demo rejoins the plain ACES rule.
    "demo_flywire_connectome.py": (
        ("Neutral", "ACES"),
        "PR #1527 re-tunes its appearance and flips it to ACES",
    ),
}

#: A floor, not a count: ~34 demo modules pin a tone mapping today. Well below
#: that means the AST scan stopped finding pins (a renamed keyword, a walk that
#: no longer descends) rather than that demos were deleted — without it a
#: broken scanner would pass this guard silently.
MIN_MODULES_WITH_PINS = 25


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "..."`` bindings, annotated ones included.

    ``TONE: Final = "Neutral"`` at module level, then
    ``ViewerConfig(tone_mapping=TONE)``, is a spelling this package's
    ``Final``-constant convention makes likely; without this it would read as
    a non-literal and slip past the policy entirely.
    """
    consts: dict[str, str] = {}
    for stmt in tree.body:
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


def _tone_mappings(path: Path) -> list[str]:
    """Every statically known ``tone_mapping`` value written in ``path``.

    Covers both spellings the demo package uses: a call keyword
    (``ViewerConfig(tone_mapping="ACES")``) and a function-signature default
    (``_interop_common.build_interop_scene(..., tone_mapping: str = "Neutral")``),
    each either as a literal or as a module-level string constant. A value that
    is neither (``tone_mapping=tone_mapping``, a forwarded parameter) carries no
    policy of its own and is skipped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    consts = _module_string_constants(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "tone_mapping":
                    value = _as_string(kw.value, consts)
                    if value is not None:
                        found.append(value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(
                zip(positional[len(positional) - len(args.defaults) :], args.defaults)
            ) + [
                (a, d)
                for a, d in zip(args.kwonlyargs, args.kw_defaults)
                if d is not None
            ]
            for arg, default in pairs:
                if arg.arg == "tone_mapping":
                    value = _as_string(default, consts)
                    if value is not None:
                        found.append(value)
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
        if "ACES" not in allowed:
            # Only rows that FORBID ACES must still carry a pin. An
            # ACES-allowing row is an in-flight flip (PR #1527), which may land
            # either by pinning "ACES" or by dropping the pin and taking the
            # viewer's ACES default — which this guard's own docstring calls
            # compliant. Demanding a pin here would turn main red on that
            # landing; the row is deleted by hand once the PR is in.
            assert values, (
                f"EXCEPTIONS names {name}, but it pins no tone mapping at all — "
                f"delete the row; silence already means the viewer's ACES default"
            )
        assert set(values) <= set(allowed), (
            f"{name} pins {sorted(set(values))}, not {sorted(allowed)}"
        )
        assert "ACES" not in values or len(allowed) > 1, (
            f"{name} pins ACES, so its EXCEPTIONS row is dead — delete it"
        )


def test_the_guard_detects_a_stray_neutral(tmp_path: Path) -> None:
    """The three spellings the scan claims to cover really are covered.

    Those three and no more: an inline literal call keyword, a signature
    default, and a module-level string constant (plus the forwarded-parameter
    case, which correctly carries no policy). The dynamic spellings listed in
    this module's docstring stay out of reach on purpose, so this is a
    completeness check on the documented surface, not on every way a string can
    reach ``tone_mapping``.
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
