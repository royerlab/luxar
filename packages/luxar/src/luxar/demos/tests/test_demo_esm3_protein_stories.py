"""Unit tests for the ESM protein-stories demo's pure helpers.

The demo itself needs the landscape cache (~50 MB of metadata + UMAP), so the
scene build is exercised manually; these tests pin the parts that decide WHAT
gets highlighted and WHERE the camera goes, on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos.demo_esm3_protein_stories import (
    STORIES,
    STORY_DIM,
    Story,
    overview_panel_html,
    select_story_members,
    story_camera,
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
    # cinematic lens — the panel's height: distance = R / (0.5 * 0.46 * tan(fov/2));
    # along +x lifted in +y, never through the cloud. fov is left to cinematic mode.
    r_bubble = SPHERE_RADIUS_SCALE * 0.5
    assert bubble_radius(cluster) == r_bubble
    expected = r_bubble / (0.5 * 0.46 * math.tan(math.radians(CINEMATIC_FOV_DEG) / 2))
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
