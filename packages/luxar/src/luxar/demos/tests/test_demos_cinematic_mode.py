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

Six invariants, and they close different holes:

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
``test_line_dominant_demos_allow_high_dpr``
    The documented line-dominant demo set opts into native display resolution,
    and no other demo does so accidentally. Each config in those modules must
    use a literal ``allow_high_dpr=True`` so a new scene path cannot silently
    lose the authored choice.
``test_every_authored_camera_uses_the_cinematic_lens``
    Every ``CameraConfig`` with an opening position leaves the preset's FOV
    unpinned and composes its distance for 63° through
    ``demos/_cinematic_camera.py``. Automatic framing already uses the resolved
    preset FOV, but an authored position suppresses that framing, so its distance
    must still be composed for 63°. This keeps the 35 mm framing and distortion
    together instead of mixing two lenses in one image.
``test_scientific_fidelity_overrides_are_explicit``
    The four demos whose scale, intensity, or categorical hue would be damaged
    by lens distortion and detector noise keep those author overrides explicit.
``test_python_fov_constants_match_the_viewer_contract``
    The Python framing helpers stay locked to the viewer's 35 mm preset and
    default FOV values, so a TypeScript lens retune cannot silently drift demos.

A literal ``True`` is required, not any truthy expression: a scene whose look
depends on a flag computed at build time is not something a reader can confirm,
and the value is written into the zarr store where only ``true`` expands the
preset.

FIXING A FAILURE is usually one keyword — add ``cinematic_mode=True`` to the
config the message names. Individual preset fields may be pinned when they
would damage the scene's scientific contract: the two quantitative ortho demos
disable bloom, vignette, lens distortion and detector noise so their scale bars
and intensities remain meaningful, and the biodiversity globe disables the
last two so its categorical hues remain exact. The nD transform bench also
disables the last two to preserve its exact RGB corner palette.

The preset's 63° FOV is resolved before automatic framing, so auto-framed scenes
keep the fitted subject occupancy intended for that lens. Authored positions
suppress automatic framing, so their distance must instead be composed for the
unpinned 63° preset FOV.

A demo that authors a camera position has a stronger contract, since its distance
was composed for one specific FOV. Every authored pose is composed for 63°
through ``demos/_cinematic_camera.py``. Most preserve their authored framing;
the biodiversity globe preserves its silhouette, while the forest and embryo
line instead preserve subject clearance. All keep the preset's 35 mm framing
and distortion together.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

from luxar.conftest import viewer_source
from luxar.demos import _cinematic_camera
from luxar.demos._cinematic_camera import (
    CINEMATIC_FOV_DEG,
    VIEWER_DEFAULT_FOV_DEG,
    framing_scale,
    pull_in,
)

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
        {
            "bloom_enabled",
            "chromatic_lens_distortion_enabled",
            "detector_noise_enabled",
            "vignette_enabled",
        }
    ),
    "demo_gsplats_2d_codex_pancreas.py": frozenset(
        {
            "bloom_enabled",
            "chromatic_lens_distortion_enabled",
            "detector_noise_enabled",
            "vignette_enabled",
        }
    ),
    "demo_nd_transforms.py": frozenset(
        {"chromatic_lens_distortion_enabled", "detector_noise_enabled"}
    ),
}

HIGH_DPR_DEMOS = frozenset(
    {
        "demo_bioluminescent_ocean.py",
        "demo_caida_as_topology.py",
        "demo_cosmicflows_laniakea.py",
        "demo_dipc_3d_genome.py",
        "demo_dmri_tractography.py",
        # Not line-dominant: a points scene, but authored for ONE dedicated
        # high-DPI kiosk display with a GPU to spare, where the owner asked for
        # full device resolution (a laptop's DPR cap is the wrong default there).
        "demo_esm3_protein_stories.py",
        "demo_esm_protein_universe.py",  # same kiosk display as the stories tour
        "demo_flywire_connectome.py",
        "demo_global_rivers_earth.py",
        "demo_hilbert_curve_3d.py",
        "demo_huri_interactome.py",
        "demo_ocean_currents_earth.py",
        "demo_particle_collision.py",
        "demo_particle_collision_animated.py",
        "demo_ppi_flow_field.py",
        "demo_zebrahub_velocity_streamlines.py",
    }
)

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


def _has_non_none_keyword(call: ast.Call, name: str) -> bool:
    value = _keyword(call, name)
    return value is not None and not (
        isinstance(value, ast.Constant) and value.value is None
    )


def _sets_literal(call: ast.Call, name: str, expected: bool) -> bool:
    value = _keyword(call, name)
    return isinstance(value, ast.Constant) and value.value is expected


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_every_scene_passes_a_viewer_config(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    bare = [
        call.lineno
        for call in _create_scene_calls(tree)
        if not _has_non_none_keyword(call, "viewer_config")
    ]
    assert not bare, (
        f"{path.name}: create_scene at line(s) {bare} passes no viewer_config, "
        f"so the scene cannot enable cinematic mode — pass "
        f"viewer_config=ViewerConfig(cinematic_mode=True)"
    )


def _composes_for_the_cinematic_lens(tree: ast.AST, call: ast.Call) -> bool:
    """Whether an authored pose is demonstrably built for the preset's 63° lens.

    Two routes, both from ``demos/_cinematic_camera.py`` and visible to a static
    reader:

    * ``position=pull_in(...)`` — a distance tuned at an authored lens carried
      over to 63°.
    * the module imports ``CINEMATIC_FOV_DEG`` or ``framing_scale`` — it derives
      a distance from the cinematic lens, so there is no hidden 47° assumption.

    The second test is per-MODULE rather than per-call, which is the looser of
    the two: a module that imports either symbol vouches for every pose in it.
    That is the honest granularity for a demo whose framing comes out of one
    camera-distance helper, and the import is a deliberate enough act to read
    as the statement it is.
    """
    imported_symbols = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("_cinematic_camera")
        for alias in node.names
    }
    position = _keyword(call, "position")
    if isinstance(position, ast.Call) and isinstance(position.func, ast.Name):
        if imported_symbols.get(position.func.id) == "pull_in":
            return True
    return bool({"CINEMATIC_FOV_DEG", "framing_scale"} & set(imported_symbols.values()))


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_every_authored_camera_uses_the_cinematic_lens(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    missing = [
        call.lineno
        for call in _camera_configs(tree)
        if _has_non_none_keyword(call, "position")
        and (
            _has_non_none_keyword(call, "fov")
            or _has_non_none_keyword(call, "fov_preset")
            or not _composes_for_the_cinematic_lens(tree, call)
        )
    ]
    assert not missing, (
        f"{path.name}: CameraConfig at line(s) {missing} does not take the "
        f"cinematic 35 mm lens whole: leave fov/fov_preset unset and compose "
        f"the position for 63° through demos/_cinematic_camera.py"
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


def test_line_dominant_demos_allow_high_dpr() -> None:
    authored: dict[str, list[int]] = {}
    incomplete: dict[str, list[int]] = {}

    for path in MODULES:
        tree = ast.parse(path.read_text(), filename=str(path))
        configs = _viewer_configs(tree)
        enabled = [
            call.lineno
            for call in configs
            if _sets_literal(call, "allow_high_dpr", True)
        ]
        if enabled:
            authored[path.name] = enabled
        if path.name in HIGH_DPR_DEMOS and len(enabled) != len(configs):
            incomplete[path.name] = [call.lineno for call in configs]

    assert not incomplete, (
        "every ViewerConfig in a line-dominant demo must set "
        f"allow_high_dpr=True; config line(s) by module: {incomplete}"
    )
    assert authored.keys() == HIGH_DPR_DEMOS, (
        "allow_high_dpr=True must match the documented line-dominant demo set; "
        f"authored at line(s) {authored}"
    )


def test_the_scan_actually_finds_the_configs() -> None:
    total = sum(
        len(_viewer_configs(ast.parse(path.read_text(), filename=str(path))))
        for path in MODULES
    )
    assert total >= MIN_VIEWER_CONFIGS, (
        f"found only {total} ViewerConfig constructions across {len(MODULES)} "
        f"demo modules (expected at least {MIN_VIEWER_CONFIGS}) — the scan "
        f"broke, so the ViewerConfig invariant above was passing vacuously"
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
        ("CameraConfig(position=p, fov=47.0)", True),
        ("CameraConfig(position=p, fov_preset='50mm')", True),
        ("CameraConfig(position=p)", True),
        ("CameraConfig(position=p, fov=None)", True),
        ("CameraConfig(fov=47.0)", False),
        # Composed for the cinematic lens instead of pinned — the other way to
        # be unambiguous about the FOV a pose assumes.
        (
            "from luxar.demos._cinematic_camera import pull_in\n"
            "CameraConfig(position=pull_in((6.0, 7.0, 19.0)))",
            False,
        ),
        (
            "from luxar.demos._cinematic_camera import pull_in as compose\n"
            "CameraConfig(position=compose((6.0, 7.0, 19.0)))",
            False,
        ),
        (
            "from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG\n"
            "CameraConfig(position=(0.0, 0.0, d))",
            False,
        ),
        (
            "from luxar.demos._cinematic_camera import framing_scale\n"
            "CameraConfig(position=(0.0, 0.0, d * framing_scale(50.0)))",
            False,
        ),
        # A pull_in-looking call that is NOT the helper stays flagged.
        ("CameraConfig(position=other.pull_in(p))", True),
        (
            "def pull_in(position): return position\nCameraConfig(position=pull_in(p))",
            True,
        ),
    ],
)
def test_the_guard_reads_authored_camera_framing(source: str, flagged: bool) -> None:
    tree = ast.parse(source)
    (call,) = _camera_configs(tree)
    missing = bool(
        _has_non_none_keyword(call, "position")
        and (
            _has_non_none_keyword(call, "fov")
            or _has_non_none_keyword(call, "fov_preset")
            or not _composes_for_the_cinematic_lens(tree, call)
        )
    )
    assert missing is flagged


@pytest.mark.parametrize("from_fov_deg", [28.0, 38.0, 42.0, 45.0, 50.0])
def test_pull_in_preserves_framing_from_every_authored_lens(
    from_fov_deg: float,
) -> None:
    target = (4.0, -3.0, 2.0)
    position = (14.0, 17.0, 32.0)
    moved = pull_in(position, target, from_fov_deg=from_fov_deg)

    old_distance = math.dist(position, target)
    new_distance = math.dist(moved, target)
    old_half_height = old_distance * math.tan(math.radians(from_fov_deg / 2.0))
    new_half_height = new_distance * math.tan(math.radians(CINEMATIC_FOV_DEG / 2.0))

    assert new_half_height == pytest.approx(old_half_height)
    assert tuple(moved[i] - target[i] for i in range(3)) == pytest.approx(
        tuple((position[i] - target[i]) * new_distance / old_distance for i in range(3))
    )
    assert new_distance / old_distance == pytest.approx(framing_scale(from_fov_deg))


def test_pull_in_uses_the_shared_framing_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cinematic_camera, "framing_scale", lambda _from_fov_deg: 0.25)

    assert pull_in((8.0, 12.0, 16.0), (4.0, 4.0, 4.0)) == (5.0, 6.0, 7.0)


def test_python_fov_constants_match_the_viewer_contract() -> None:
    camera_source = viewer_source("src/config/sections/camera/data.ts")
    rendering_source = viewer_source("src/config/sections/rendering-controls/data.ts")

    cinematic_match = re.search(
        r"['\"]35mm['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        camera_source.read_text(encoding="utf-8"),
    )
    default_match = re.search(
        r"defaults\s*:\s*\{.*?\bfov\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        rendering_source.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert cinematic_match is not None, (
        "camera/data.ts no longer exposes a numeric 35 mm FOV"
    )
    assert default_match is not None, (
        "rendering-controls/data.ts no longer exposes a numeric default FOV"
    )
    assert float(cinematic_match.group(1)) == CINEMATIC_FOV_DEG
    assert float(default_match.group(1)) == VIEWER_DEFAULT_FOV_DEG


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
        if not _has_non_none_keyword(call, "viewer_config")
    ]
    assert bool(bare) is flagged
