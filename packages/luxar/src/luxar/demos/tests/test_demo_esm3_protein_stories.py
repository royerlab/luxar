"""Unit tests for the ESM protein-stories demo's pure helpers.

The demo itself needs the landscape cache (~50 MB of metadata + UMAP), so the
scene build is exercised manually; these tests pin the parts that decide WHAT
gets highlighted and WHERE the camera goes, on synthetic data.
"""

from __future__ import annotations

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
    hum_frequency_hz,
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
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    cluster = StoryCluster(
        indices=np.arange(3), centre=np.array([4.0, 0.0, 0.0]), r95=0.5, n_named=3
    )
    import math

    from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG

    cam = story_camera(
        cluster, _story(frame_fraction=0.36, min_distance=2.5), np.zeros(3)
    )
    assert cam.target == (4.0, 0.0, 0.0)
    assert cam.position is not None
    offset = np.array(cam.position) - np.array(cam.target)
    # The blob (diameter 1.0) spans 36% of the frame height under the cinematic
    # lens: distance = r95 / (0.5 * 0.36 * tan(fov/2)); along +x lifted in +y,
    # never through the cloud. The pose leaves fov to cinematic mode.
    expected = 0.5 / (0.5 * 0.36 * math.tan(math.radians(CINEMATIC_FOV_DEG) / 2))
    assert np.isclose(np.linalg.norm(offset), expected)
    assert offset[0] > 0 and offset[1] > 0
    assert cam.fov is None
    # min_distance is a floor, not a scale.
    far = story_camera(
        cluster, _story(frame_fraction=0.36, min_distance=20.0), np.zeros(3)
    )
    assert far.position is not None
    assert np.isclose(
        np.linalg.norm(np.array(far.position) - np.array(far.target)), 20.0
    )


def test_story_camera_falls_back_when_the_cluster_sits_at_the_centre() -> None:
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    cluster = StoryCluster(indices=np.arange(1), centre=np.zeros(3), r95=0.1, n_named=1)
    cam = story_camera(cluster, _story(), np.zeros(3))
    assert cam.position is not None and cam.position[2] > 0


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
    hums: list[float] = []

    def fake_hum(frequency_hz, seconds, cache_dir, **_k):
        hums.append(frequency_hz)
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip = cache_dir / f"hum{len(hums)}.m4a"
        clip.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
        return clip

    monkeypatch.setattr(demo, "synthesise_hum", fake_hum)
    foa_sources: list[Path] = []

    def fake_foa(src, cache_dir, *, spread_deg):
        foa_sources.append(src)
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip = cache_dir / "foa.m4a"
        clip.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
        return clip

    monkeypatch.setattr(demo, "synthesise_foa_from_clip", fake_foa)
    stories = (
        _story(key="A", flight_ms=2000),
        _story(key="B", pattern="^x", flight_ms=4000),
    )
    clusters = [
        StoryCluster(indices=np.array([0]), centre=np.zeros(3), r95=0.5, n_named=1),
        StoryCluster(indices=np.array([1]), centre=np.ones(3), r95=2.0, n_named=1),
    ]
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
            clusters=clusters,
            hum_dir=tmp_path / "hums",
            ambisonic_dir=tmp_path / "foa",
        )
    assert added == 6  # bed + overview + two narrations + two hums
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
    # Bed and hums are Layers-panel rows (mute / gain); narrations are not.
    assert bed_attrs["layer"] is True
    assert dict(root["hum_B"].attrs)["layer"] is True
    assert "layer" not in dict(root["narration_B"].attrs)
    narr = dict(root["narration_B"].attrs)
    # Narration starts when the flight lands (waypoint arrival), a beat later.
    assert narr["trigger"] == "on_arrive" and narr["bus"] == "voice"
    assert narr["delay_ms"] == NARRATION_AFTER_FLIGHT_MS
    assert narr["has_positions"] is True and narr["n_positions"] == 1
    overview = dict(root["narration_Overview"].attrs)
    assert overview["trigger"] == "on_arrive"

    # One hum per cluster, attached to the story's highlight node, scaled by r95.
    assert hums == [hum_frequency_hz(1), hum_frequency_hz(2)]
    assert hum_frequency_hz(2) > hum_frequency_hz(1)
    hum_b = dict(root["hum_B"].attrs)
    assert hum_b["attach_to"] == story_node_name(2, stories[1]) == "Story 2: B"
    assert hum_b["spatial"] is True and hum_b["bus"] == "effects"
    assert hum_b["trigger"] == "continuous" and hum_b["loop"] is True
    assert hum_b["ref_distance"] == pytest.approx(2.0 * demo.HUM_REF_DISTANCE_PER_R95)
    assert hum_b["max_distance"] == pytest.approx(2.0 * demo.HUM_MAX_DISTANCE_PER_R95)
    assert hum_b["has_positions"] is True and hum_b["n_positions"] == 1
    hum_a = dict(root["hum_A"].attrs)
    assert hum_a["ref_distance"] == pytest.approx(
        max(0.5, 0.5 * demo.HUM_REF_DISTANCE_PER_R95)
    )


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
