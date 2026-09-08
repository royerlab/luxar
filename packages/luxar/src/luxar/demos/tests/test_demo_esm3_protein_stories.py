"""Unit tests for the ESM protein-stories demo's pure helpers.

The demo itself needs the landscape cache (~50 MB of metadata + UMAP), so the
scene build is exercised manually; these tests pin the parts that decide WHAT
gets highlighted and WHERE the camera goes, on synthetic data.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pytest

from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import demo_esm3_protein_stories as demo
from luxar.demos.demo_esm3_protein_stories import (
    NARRATION_AFTER_FLIGHT_MS,
    STORIES,
    STORY_DIM,
    Story,
    StoryCluster,
    add_story_sounds,
    overview_panel_html,
    select_story_members,
    story_camera,
    story_node_name,
    story_panel_html,
)


def _story(**overrides: object) -> Story:
    base = dict(
        key="Test",
        title="Test story",
        subtitle="sub",
        pattern=r"^Hemoglobin subunit",
        color=(1.0, 0.0, 0.0),
        facts=("fact one", "fact <two>"),
        mystery="why?",
    )
    base.update(overrides)
    return Story(**base)  # type: ignore[arg-type]


def test_select_members_keeps_the_blob_and_drops_stragglers() -> None:
    names = np.array(
        ["Hemoglobin subunit alpha"] * 6
        + ["Hemoglobin subunit beta"]
        + ["Myoglobin"] * 3,
        dtype=object,
    )
    kingdoms = np.array(["Other Eukaryotes"] * 10, dtype=object)
    pos = np.zeros((10, 3), dtype=np.float32)
    pos[:6] = np.random.default_rng(0).normal(scale=0.1, size=(6, 3)) + [5, 5, 5]
    pos[6] = [20, 20, 20]  # a hemoglobin far from the blob: named, not a member
    pos[7:] = [-5, -5, -5]  # myoglobins: not named at all

    cluster = select_story_members(_story(radius=0.8), names, kingdoms, pos)

    assert cluster.n_named == 7
    assert sorted(cluster.indices.tolist()) == [0, 1, 2, 3, 4, 5]
    assert np.allclose(cluster.centre, [5, 5, 5], atol=0.2)
    assert 0 < cluster.r95 < 0.5


def test_select_members_centres_on_the_densest_blob_of_a_split_family() -> None:
    # A family the model splits in two: 20 members at +8 and 6 members at -8.
    # The MEDIAN of such a family lands between the blobs, on nothing;
    # centring on the densest member picks the big blob.
    rng = np.random.default_rng(1)
    big = rng.normal(scale=0.1, size=(20, 3)) + [8, 8, 8]
    small = rng.normal(scale=0.1, size=(6, 3)) + [-8, -8, -8]
    pos = np.vstack([big, small]).astype(np.float32)
    names = np.array(["Hemoglobin subunit alpha"] * 26, dtype=object)
    kingdoms = np.array(["Other Eukaryotes"] * 26, dtype=object)

    cluster = select_story_members(_story(radius=0.6), names, kingdoms, pos)

    assert sorted(cluster.indices.tolist()) == list(range(20))
    assert np.allclose(cluster.centre, [8, 8, 8], atol=0.2)
    assert cluster.n_named == 26


def test_select_members_honours_the_kingdom_filter() -> None:
    names = np.array(["Hemagglutinin"] * 4, dtype=object)
    kingdoms = np.array(
        ["Viruses", "Viruses", "Other Bacteria", "Viruses"], dtype=object
    )
    pos = np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (4, 1))

    cluster = select_story_members(
        _story(pattern=r"^Hemagglutinin", kingdom="Viruses"), names, kingdoms, pos
    )
    assert sorted(cluster.indices.tolist()) == [0, 1, 3]


def test_select_members_fails_loudly_when_nothing_matches() -> None:
    names = np.array(["Myoglobin"], dtype=object)
    with pytest.raises(ValueError, match="matched no protein names"):
        select_story_members(
            _story(), names, np.array(["x"], dtype=object), np.zeros((1, 3), np.float32)
        )


def test_story_camera_looks_at_the_centre_from_outside_the_cloud() -> None:
    cluster = StoryCluster(
        indices=np.arange(3), centre=np.array([4.0, 0.0, 0.0]), r95=0.5, n_named=3
    )
    import math

    from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
    from luxar.demos.demo_esm3_protein_stories import (
        SPARSE_TAIL_RATIO,
        SPHERE_RADIUS_SCALE,
        bubble_radius,
        framing_radius,
    )

    cam = story_camera(
        cluster, _story(frame_fraction=0.46, min_distance=1.0), np.zeros(3)
    )
    assert cam.target == (4.0, 0.0, 0.0)
    assert cam.position is not None
    offset = np.array(cam.position) - np.array(cam.target)
    # The BUBBLE (radius 1.35 * r95) spans 46% of the frame height under the
    # cinematic lens — the panel's height: a sphere of radius R at distance d
    # covers R / (d * tan(fov/2)) of the height, so d = R / (0.46 * tan(fov/2));
    # along +x lifted in +y, never through the cloud. fov is left to cinematic mode.
    r_bubble = SPHERE_RADIUS_SCALE * 0.5
    assert bubble_radius(cluster) == r_bubble
    expected = r_bubble / (0.46 * math.tan(math.radians(CINEMATIC_FOV_DEG) / 2))
    assert np.isclose(np.linalg.norm(offset), expected)
    assert offset[0] > 0 and offset[1] > 0
    assert cam.fov is None
    # min_distance is a floor, not a scale.
    far = story_camera(
        cluster, _story(frame_fraction=0.46, min_distance=20.0), np.zeros(3)
    )
    assert far.position is not None
    assert np.isclose(
        np.linalg.norm(np.array(far.position) - np.array(far.target)), 20.0
    )
    # A sparse cluster (long tail: r95 far beyond r50) is framed on its core, so
    # the camera comes closer than the bubble alone would ask for.
    sparse = StoryCluster(
        indices=np.arange(3), centre=np.zeros(3), r95=1.0, n_named=3, r50=0.2
    )
    dense = StoryCluster(
        indices=np.arange(3), centre=np.zeros(3), r95=1.0, n_named=3, r50=0.6
    )
    assert framing_radius(dense) == bubble_radius(dense)
    assert framing_radius(sparse) < bubble_radius(sparse)
    assert framing_radius(sparse) == max(
        0.35, SPHERE_RADIUS_SCALE * SPARSE_TAIL_RATIO * 0.2
    )
    near = story_camera(sparse, _story(min_distance=0.1), np.zeros(3))
    farther = story_camera(dense, _story(min_distance=0.1), np.zeros(3))
    assert near.position is not None and farther.position is not None
    assert np.linalg.norm(near.position) < np.linalg.norm(farther.position)


def test_story_camera_falls_back_when_the_cluster_sits_at_the_centre() -> None:
    cluster = StoryCluster(indices=np.arange(1), centre=np.zeros(3), r95=0.1, n_named=1)
    cam = story_camera(cluster, _story(), np.zeros(3))
    assert cam.position is not None and cam.position[2] > 0


def test_story_camera_stays_outside_a_sparse_cluster_bubble() -> None:
    from luxar.demos.demo_esm3_protein_stories import (
        BUBBLE_CAMERA_CLEARANCE,
        bubble_radius,
    )

    cluster = StoryCluster(
        indices=np.arange(3),
        centre=np.array([4.0, 0.0, 0.0]),
        r95=2.4,
        r50=0.2,
        n_named=3,
    )
    cam = story_camera(cluster, _story(min_distance=0.1), np.zeros(3))

    assert cam.position is not None
    distance = np.linalg.norm(np.asarray(cam.position) - np.asarray(cam.target))
    assert np.isclose(distance, BUBBLE_CAMERA_CLEARANCE * bubble_radius(cluster))


@pytest.mark.parametrize("n", [0, 1, 3])
def test_icosphere_is_a_closed_unit_manifold(n: int) -> None:
    from luxar.demos.demo_esm3_protein_stories import icosphere

    verts, faces = icosphere(n)
    assert verts.shape == (10 * 4**n + 2, 3)
    assert faces.shape == (20 * 4**n, 3)
    assert verts.dtype == np.float32 and faces.dtype == np.uint32
    # Every vertex on the unit sphere → positions double as normals.
    assert np.allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-6)
    # Closed manifold: every edge is shared by exactly two faces.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    assert faces.max() < len(verts)


def test_panels_escape_html_and_carry_the_counts() -> None:
    panel = story_panel_html(_story(), n_members=42, index=2, total=5)
    assert "Story 2 of 5" in panel
    assert "&lt;two&gt;" in panel  # facts are text, not markup
    assert "42 proteins highlighted" in panel
    assert "Open question: why?" in panel
    assert "1,234 Swiss-Prot proteins" in overview_panel_html(1234)


def test_attribution_closes_the_overview_panel_and_is_not_a_standalone_overlay() -> (
    None
):
    """The credit lives inside the Overview panel (story 0), nowhere else.

    The kiosk keeps the title bare and bottom-right for the Biohub mark, so the
    dataset/model/license credit is the panel's closing line — escaped like the
    rest of the panel — and no overlay is authored from ATTRIBUTION directly.
    """
    import ast
    import inspect

    import luxar.demos.demo_esm3_protein_stories as demo

    citation = demo.DEMO_META["citation"]
    assert citation["ref"] in demo.ATTRIBUTION
    assert citation["license"] in demo.ATTRIBUTION
    panel = overview_panel_html(1234)
    assert html.escape(demo.ATTRIBUTION) in panel
    assert panel.rstrip().endswith(html.escape(demo.ATTRIBUTION) + "</div></div>")

    tree = ast.parse(inspect.getsource(demo))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"add_text", "add_html"}
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ATTRIBUTION"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_html"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "overview_panel_html"
        for node in ast.walk(tree)
    )


def test_story_highlights_declare_their_intentional_additive_blending() -> None:
    import ast
    import inspect

    import luxar.demos.demo_esm3_protein_stories as demo

    tree = ast.parse(inspect.getsource(demo))
    story_points = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_points"
        and node.args
        # The highlight node is the one named through story_node_name(k, s)
        # (the hums' attach_to target).
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "story_node_name"
    ]
    assert len(story_points) == 1
    blending_mode = next(
        (
            keyword.value
            for keyword in story_points[0].keywords
            if keyword.arg == "blending_mode"
        ),
        None,
    )
    assert isinstance(blending_mode, ast.Constant)
    assert blending_mode.value == "additive"


def test_story_narration_is_the_short_spoken_script_not_the_panel() -> None:
    from luxar.demos.demo_esm3_protein_stories import (
        OVERVIEW_NARRATION,
        story_narration,
    )

    for s in STORIES:
        spoken = story_narration(s)
        panel_words = sum(len(f.split()) for f in s.facts) + len(s.mystery.split())
        assert 40 <= len(spoken.split()) <= 110, (s.key, len(spoken.split()))
        assert len(spoken.split()) < 0.7 * panel_words, s.key  # punchier than the panel
        assert not spoken.startswith("Story "), s.key  # no "Story N of 10" prefix
        assert "<" not in spoken and "&" not in spoken, s.key  # plain speech, no markup
        assert spoken.rstrip().endswith((".", "?")), s.key
    assert 20 <= len(OVERVIEW_NARRATION.split()) <= 80
    # A story without an authored script still gets something short to say.
    assert story_narration(_story()) == "Test story. why?"


def test_shipped_stories_are_well_formed_and_author_valid_waypoints() -> None:
    keys = [s.key for s in STORIES]
    assert len(keys) == len(set(keys)) == 10
    for s in STORIES:
        assert "/" not in s.key, f"{s.key!r} doubles as a node name; '/' is refused"
        assert 3 <= len(s.facts) <= 5, s.key
        assert s.mystery.strip(), s.key
        assert s.narration.strip(), s.key  # every shipped story is narrated
        assert all(0.0 <= c <= 1.0 for c in s.color), s.key
    # The waypoint schema accepts what the demo authors (story index → pose).
    from luxar.core.viewer_config import Waypoint

    vc = ViewerConfig(
        waypoints=[
            Waypoint(when={STORY_DIM: k}, camera=CameraConfig(target=(0.0, 0.0, 0.0)))
            for k in range(len(STORIES) + 1)
        ]
    )
    assert len(vc.to_dict()["waypoints"]) == 11


# =============================================================================
# Sound layer: narration text + node assembly
# =============================================================================


def test_add_story_sounds_authors_a_bed_and_one_narration_per_slot(
    tmp_path, monkeypatch
) -> None:
    """The bed download and the TTS are swapped for fakes; the SOUND NODES that land
    in the store are real (through ``scene.add_sound``)."""
    from luxar import Dimension, Dimensions, LuxarZarrCompiler
    from luxar._zarr_compat import open_group

    mp3 = bytes.fromhex("fffb9000") + b"\x00" * 64
    bed = tmp_path / "bed.mp3"
    bed.write_bytes(mp3)
    monkeypatch.setattr(demo, "cached_download", lambda *a, **k: bed)
    spoken: list[str] = []

    def fake_synthesise(text, voice, cache_dir, *, engine=None, engines=None):
        spoken.append(text)
        clip = cache_dir / f"{len(spoken)}.mp3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(mp3)
        return clip

    monkeypatch.setattr(demo, "synthesise", fake_synthesise)
    foa_sources: list[Path] = []

    def fake_foa(src, cache_dir, *, spread_deg, loop_crossfade_s):
        assert loop_crossfade_s == demo.AMBIENT_BED_LOOP_CROSSFADE_S
        foa_sources.append(src)
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip = cache_dir / "foa.m4a"
        clip.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
        return clip

    monkeypatch.setattr(demo, "synthesise_foa_from_clip", fake_foa)
    stories = (
        _story(key="A", frame_fraction=0.72, flight_ms=2000),
        _story(key="B", pattern="^x", flight_ms=4000),
    )
    store = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension(
                        STORY_DIM,
                        unit="",
                        categories=["Overview", "A", "B"],
                        display=False,
                    ),
                    Dimension("x", unit="UMAP"),
                    Dimension("y", unit="UMAP"),
                    Dimension("z", unit="UMAP"),
                ]
            )
        )
        added = add_story_sounds(
            scene,
            stories,
            narration_dir=tmp_path / "narration",
            engine="openai",
            ambisonic_dir=tmp_path / "foa",
        )
    # bed + overview + two narrations — and nothing on the `effects` bus: the
    # per-cluster hums were removed as a distracting drone under the voice.
    assert added == 4
    assert not [
        n for n in open_group(store, mode="r").group_keys() if n.startswith("hum_")
    ]
    assert foa_sources == [bed]
    # The authored scripts, not the panel: the overview script, then each
    # story's `narration` (or its title + open question when none is authored).
    assert spoken[0] == demo.OVERVIEW_NARRATION
    assert spoken[1] == demo.story_narration(stories[0])
    assert not spoken[1].startswith("Story ")
    root = open_group(store, mode="r")
    bed_attrs = dict(root["bed_ambient"].attrs)
    assert bed_attrs["trigger"] == "continuous" and bed_attrs["bus"] == "ambient"
    assert bed_attrs["has_positions"] is False
    assert bed_attrs["license"] == "CC0"
    # The bed became a first-order ambisonic field (the fake encoder "succeeded").
    assert bed_attrs["ambisonic"] == "foa" and bed_attrs["format"] == "aac"
    assert bed_attrs["spatial"] is False
    # The bed is a Layers-panel row (mute / gain); narrations are not.
    assert bed_attrs["layer"] is True
    assert "layer" not in dict(root["narration_B"].attrs)
    narr = dict(root["narration_B"].attrs)
    # Narration starts when the flight lands (waypoint arrival), a beat later.
    assert narr["trigger"] == "on_arrive" and narr["bus"] == "voice"
    assert narr["delay_ms"] == NARRATION_AFTER_FLIGHT_MS
    assert narr["has_positions"] is True and narr["n_positions"] == 1
    overview = dict(root["narration_Overview"].attrs)
    assert overview["trigger"] == "on_arrive"
    # The highlight node name the hums used to attach to is still the demo's.
    assert story_node_name(2, stories[1]) == "Story 2: B"


def test_add_story_sounds_stays_silent_without_an_engine_and_survives_no_bed(
    tmp_path, monkeypatch
) -> None:
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    def offline(*_a, **_k):
        raise OSError("no network")

    monkeypatch.setattr(demo, "cached_download", offline)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("luxar.demos._narration.shutil.which", lambda _n: None)
    store = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension(
                        STORY_DIM, unit="", categories=["Overview", "A"], display=False
                    ),
                    Dimension("x", unit="UMAP"),
                    Dimension("y", unit="UMAP"),
                    Dimension("z", unit="UMAP"),
                ]
            )
        )
        with pytest.warns(UserWarning, match="No narration engine"):
            added = add_story_sounds(
                scene, (_story(key="A"),), narration_dir=tmp_path / "n", engine=None
            )
    assert added == 0


def test_add_story_sounds_keeps_the_stereo_bed_without_an_encoder(
    tmp_path, monkeypatch
) -> None:
    from luxar import Dimension, Dimensions, LuxarZarrCompiler
    from luxar._zarr_compat import open_group

    bed = tmp_path / "bed.mp3"
    bed.write_bytes(bytes.fromhex("fffb9000") + b"\x00" * 64)
    monkeypatch.setattr(demo, "cached_download", lambda *a, **k: bed)
    monkeypatch.setattr(demo, "synthesise_foa_from_clip", lambda *a, **k: None)
    store = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension(
                        STORY_DIM, unit="", categories=["Overview"], display=False
                    ),
                    Dimension("x", unit="UMAP"),
                    Dimension("y", unit="UMAP"),
                    Dimension("z", unit="UMAP"),
                ]
            )
        )
        added = add_story_sounds(scene, (), narration_dir=tmp_path / "n", engine="none")
    assert added == 1
    attrs = dict(open_group(store, mode="r")["bed_ambient"].attrs)
    assert "ambisonic" not in attrs and attrs["format"] == "mp3"


def test_biohub_logo_is_a_bundled_transparent_png() -> None:
    """The bottom-right mark ships inside the package (self-contained store).

    demos/data is excluded from the wheel, so the asset lives beside the module;
    it must be a real PNG with an alpha channel — the glyph is white, and the
    transparency plus `difference` blending is what makes it read on any
    background.
    """
    from luxar.demos.demo_esm3_protein_stories import BIOHUB_LOGO, BIOHUB_LOGO_WIDTH

    assert BIOHUB_LOGO.is_file(), BIOHUB_LOGO
    assert "demos/data" not in BIOHUB_LOGO.as_posix()
    assert BIOHUB_LOGO.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image

    with Image.open(BIOHUB_LOGO) as im:
        assert im.mode == "RGBA"
        alpha = np.asarray(im)[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255  # transparent margin, opaque glyph
    assert 0 < BIOHUB_LOGO_WIDTH <= 0.2


def test_every_story_waypoint_reveals_its_overlays_on_arrival() -> None:
    """The panel, caption and turntable appear when the camera has landed.

    Every ``Waypoint(...)`` the demo authors carries ``reveal="on_arrival"`` —
    the gate the viewer keys on the flight's arrival, the same event that starts
    the narration, so text and voice land together (remote-control spec §4.1).
    """
    import ast
    import inspect

    import luxar.demos.demo_esm3_protein_stories as demo

    tree = ast.parse(inspect.getsource(demo))
    waypoint_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Waypoint"
    ]
    assert len(waypoint_calls) == 2  # the overview and the per-story loop
    for call in waypoint_calls:
        reveal = next((kw.value for kw in call.keywords if kw.arg == "reveal"), None)
        assert isinstance(reveal, ast.Constant), ast.unparse(call)[:80]
        assert reveal.value == "on_arrival"

    video_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_video"
    ]
    assert len(video_calls) == 1
    alpha_matte = next(
        (kw.value for kw in video_calls[0].keywords if kw.arg == "alpha_matte"), None
    )
    assert isinstance(alpha_matte, ast.Constant), ast.unparse(video_calls[0])[:80]
    assert alpha_matte.value == "stacked"
