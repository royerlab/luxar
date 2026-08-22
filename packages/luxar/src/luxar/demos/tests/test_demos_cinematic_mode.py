"""Static guard: every demo scene ships the cinematic look.

The demos are the gallery — they are what a first-time visitor sees — and the
house choice is that they all open in cinematic mode: ACES, a subtle wide bloom,
detector noise, a vignette and a 35 mm chromatic lens. In a scene that is
``ViewerConfig(cinematic_mode=True)``, one keyword, because the zarr config
bridge expands the preset at load time into every field the scene did NOT set
explicitly (#1591). An author-set field always wins, so a scene keeps its own
tone mapping, bloom or exposure and takes the rest of the look from the preset.

Why a lint rather than a default somewhere: nothing in the write path can supply
this. ``LuxarZarrCompiler.create_scene`` serves library users as well as demos,
and flipping a rendering look on by default there would restyle every scene
anyone writes with Luxar. So each demo states it, and this guard is what keeps
"all of them" true for demo number 87.

Two invariants, and they close different holes:

``test_every_scene_passes_a_viewer_config``
    No ``create_scene`` call may omit ``viewer_config``. Without this a new demo
    could ship with no config at all — the second invariant would have nothing
    to inspect and would pass vacuously.
``test_every_viewer_config_enables_cinematic_mode``
    Every ``ViewerConfig(...)`` built in the demos package sets
    ``cinematic_mode=True``. Checking the CONSTRUCTIONS rather than resolving
    what each ``create_scene`` was handed is deliberate: a config reaches a
    scene inline, through a local (``viewer_config = ViewerConfig(...)``, six
    demos) or out of a helper (``_solar_system_viewer_config()``, whose one
    config serves two scenes), and a static resolver for all three would be
    more fragile than the rule it enforces. The rule holds because in this
    package a ``ViewerConfig`` is only ever built to be handed to a scene — if
    that stops being true, this guard is where to say so.

A literal ``True`` is required, not any truthy expression: a scene whose look
depends on a flag computed at build time is not something a reader can confirm,
and the value is written into the zarr store where only ``true`` expands the
preset.

FIXING A FAILURE is usually one keyword — add ``cinematic_mode=True`` to the
config the message names. Individual preset fields may be pinned when they
would damage the scene's scientific contract: the two quantitative ortho demos
disable lens distortion and detector noise so their scale bars and intensities
remain meaningful, and the biodiversity globe disables both so its categorical
hues remain exact.

The preset's 63° FOV is applied after first-load auto-framing, which happens at
the viewer's 47° default. Auto-framed scenes therefore open 0.71x smaller. A
demo that authors a camera position has a stronger contract: its distance was
composed for a specific FOV, so every such ``CameraConfig`` must pin ``fov`` or
``fov_preset``. The third invariant below enforces that rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ._scanned_modules import scanned_demo_modules

#: A floor, not a count: ~82 configs across ~81 modules today. Well below that
#: means the AST walk stopped finding them (a renamed class, a walk that no
#: longer descends) rather than that demos were deleted.
MIN_VIEWER_CONFIGS = 60

SCIENTIFIC_FIDELITY_OVERRIDES = {
    "demo_biodiversity_planetary_scale.py": frozenset(
        {"chromatic_lens_distortion_enabled", "detector_noise_enabled"}
    ),
    "demo_gsplats_2d_cmu1_pathology.py": frozenset(
        {"chromatic_lens_distortion_enabled", "detector_noise_enabled"}
    ),
    "demo_gsplats_2d_codex_pancreas.py": frozenset(
        {"chromatic_lens_distortion_enabled", "detector_noise_enabled"}
    ),
}

MODULES = scanned_demo_modules()


def _module_ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def _viewer_configs(tree: ast.AST) -> list[ast.Call]:
    """Every ``ViewerConfig(...)`` construction in a parsed module."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ViewerConfig"
    ]


def _create_scene_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``<compiler>.create_scene(...)`` call in a parsed module.

    Matched on the attribute, not the receiver name, because demos spell the
    compiler ``compiler`` or ``c``. A demo's own plain ``create_scene()``
    helper function is not a method call and so is correctly skipped — its body
    holds the real call.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_scene"
    ]


def _camera_configs(tree: ast.AST) -> list[ast.Call]:
    """Every ``CameraConfig(...)`` construction in a parsed module."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CameraConfig"
    ]


def _enables_cinematic(call: ast.Call) -> bool:
    return any(
        kw.arg == "cinematic_mode"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in call.keywords
    )


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _sets_literal(call: ast.Call, name: str, expected: bool) -> bool:
    value = _keyword(call, name)
    return isinstance(value, ast.Constant) and value.value is expected


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_every_scene_passes_a_viewer_config(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    bare = [
        call.lineno
        for call in _create_scene_calls(tree)
        if (value := _keyword(call, "viewer_config")) is None
        or (isinstance(value, ast.Constant) and value.value is None)
    ]
    assert not bare, (
        f"{path.name}: create_scene at line(s) {bare} passes no viewer_config, "
        f"so the scene cannot enable cinematic mode — pass "
        f"viewer_config=ViewerConfig(cinematic_mode=True)"
    )


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_every_authored_camera_pins_its_fov(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    missing = [
        call.lineno
        for call in _camera_configs(tree)
        if _keyword(call, "position") is not None
        and _keyword(call, "fov") is None
        and _keyword(call, "fov_preset") is None
    ]
    assert not missing, (
        f"{path.name}: CameraConfig at line(s) {missing} sets an opening position "
        f"without pinning fov or fov_preset; cinematic mode widens the lens only "
        f"after the viewer frames the scene, so authored poses must pin the FOV "
        f"they were composed for"
    )


@pytest.mark.parametrize("filename", SCIENTIFIC_FIDELITY_OVERRIDES)
def test_scientific_fidelity_overrides_are_explicit(filename: str) -> None:
    path = next(path for path in MODULES if path.name == filename)
    tree = ast.parse(path.read_text(), filename=str(path))
    configs = _viewer_configs(tree)
    assert len(configs) == 1, (
        f"{filename}: expected one ViewerConfig, found {len(configs)}"
    )

    missing = [
        field
        for field in SCIENTIFIC_FIDELITY_OVERRIDES[filename]
        if not _sets_literal(configs[0], field, False)
    ]
    assert not missing, (
        f"{filename}: scientific fidelity requires explicit False for {missing}; "
        f"cinematic mode must not distort the scale/hue contract"
    )


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_every_viewer_config_enables_cinematic_mode(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    missing = [
        call.lineno for call in _viewer_configs(tree) if not _enables_cinematic(call)
    ]
    assert not missing, (
        f"{path.name}: ViewerConfig at line(s) {missing} does not set "
        f"cinematic_mode=True — the demos all open in cinematic mode; see this "
        f"module's docstring for the one-keyword fix"
    )


def test_the_scan_actually_finds_the_configs() -> None:
    total = sum(
        len(_viewer_configs(ast.parse(path.read_text(), filename=str(path))))
        for path in MODULES
    )
    assert total >= MIN_VIEWER_CONFIGS, (
        f"found only {total} ViewerConfig constructions across {len(MODULES)} "
        f"demo modules (expected at least {MIN_VIEWER_CONFIGS}) — the scan "
        f"broke, so both invariants above were passing vacuously"
    )


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("ViewerConfig(cinematic_mode=True)", False),
        ("ViewerConfig(cinematic_mode=True, tone_mapping='ACES')", False),
        ("ViewerConfig(tone_mapping='ACES')", True),
        ("ViewerConfig()", True),
        ("ViewerConfig(cinematic_mode=False)", True),
        # Not a literal True: unreadable statically, and only `true` in the
        # store expands the preset.
        ("ViewerConfig(cinematic_mode=WANT_FILM_LOOK)", True),
        ("ViewerConfig(cinematic_mode=1)", True),
    ],
)
def test_the_guard_reads_the_flag_it_claims_to(source: str, flagged: bool) -> None:
    (call,) = _viewer_configs(ast.parse(source))
    assert _enables_cinematic(call) is not flagged


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("compiler.create_scene(dimensions=d, viewer_config=v)", False),
        ("c.create_scene(dimensions=d, viewer_config=v)", False),
        ("compiler.create_scene(dimensions=d, viewer_config=None)", True),
        ("compiler.create_scene(dimensions=d)", True),
        # A demo's own helper of the same name is not a scene call; the real
        # `compiler.create_scene` inside its body is what gets checked.
        ("create_scene(output_path)", False),
    ],
)
def test_the_guard_finds_the_scene_calls_it_claims_to(
    source: str, flagged: bool
) -> None:
    tree = ast.parse(source)
    bare = [
        call
        for call in _create_scene_calls(tree)
        if (value := _keyword(call, "viewer_config")) is None
        or (isinstance(value, ast.Constant) and value.value is None)
    ]
    assert bool(bare) is flagged
