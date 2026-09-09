"""Unit tests for the ESM protein-universe demo's pure helpers.

The demo itself needs the two hand-delivered parquet files (~900 MB), so the
scene build is exercised manually; these tests pin the parts that decide WHAT a
story highlights and WHERE the camera goes, on synthetic data, plus the
well-formedness of the twelve shipped stories.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.viewer_config import ViewerConfig
from luxar.demos import demo_esm_protein_universe as demo
from luxar.demos.demo_esm3_protein_stories import (
    STORIES as SWISSPROT_STORIES,
)
from luxar.demos.demo_esm3_protein_stories import (
    Story,
    story_camera,
    story_narration,
)
from luxar.demos.demo_esm_protein_universe import (
    BUBBLE_MIN_RADIUS,
    HIGHLIGHT_INTENSITY,
    HIGHLIGHT_REFERENCE_MEMBERS,
    STORIES,
    STORY_DIM,
    Universe,
    UniverseStory,
    backdrop_colors,
    densest_core,
    family_mask,
    highlight_intensity,
    member_labels,
    overview_panel_html,
    phylum_group,
    select_universe_members,
)

spatial = pytest.importorskip("scipy.spatial")


# --------------------------------------------------------------------------- #
# Synthetic universe: a diffuse cloud, one dense pure knot, one dense mixed knot
# --------------------------------------------------------------------------- #


def _universe(seed: int = 0) -> Universe:
    rng = np.random.default_rng(seed)
    # ×4: the spur predicate starts at radius 22 (5.5σ), so no cloud point
    # ever crosses it and the spur count below is exact.
    cloud = rng.normal(size=(4000, 3)) * 4.0
    pure = np.array([6.0, 0.0, 0.0]) + rng.normal(size=(120, 3)) * 0.05
    mixed_fam = np.array([-6.0, 0.0, 0.0]) + rng.normal(size=(150, 3)) * 0.05
    mixed_other = np.array([-6.0, 0.0, 0.0]) + rng.normal(size=(600, 3)) * 0.05
    spur = np.array([0.0, 0.0, 30.0]) + rng.normal(size=(50, 3)) * 0.3
    positions = np.vstack([cloud, pure, mixed_fam, mixed_other, spur]).astype(
        np.float32
    )
    n = len(positions)
    pfam_vocab = np.array(["PF00001", "PF00042", "PF99999"])
    pfam = np.full(n, -1, dtype=np.int32)
    pfam[4000:4120] = 1  # the pure knot is all globin
    pfam[4120:4270] = 1  # so is the family half of the mixed knot
    pfam[4270:4870] = 2  # ... buried in four times as many other clusters
    pfam[:200] = 0
    phylum_vocab = np.array(["Chordata", "Pseudomonadota", "Uroviricota"])
    phylum = np.full(n, -1, dtype=np.int32)
    phylum[:2000] = 1
    phylum[2000:2500] = 2
    phylum[4000:4120] = 0
    pfam[4870:4920] = 0  # the spur is PF00001 here (the shipped one is PF00005)
    pct = np.full(n, 100, dtype=np.uint8)
    pct[2000:2500] = 0  # the phage clusters are dark
    return Universe(
        positions=positions,
        annotation_row=np.arange(n, dtype=np.int32),
        pct_characterized=pct,
        pfam_code=pfam,
        pfam_vocab=pfam_vocab,
        phylum_code=phylum,
        phylum_vocab=phylum_vocab,
    )


def _story(**overrides: object) -> UniverseStory:
    base = dict(
        key="Test",
        title="Test story",
        subtitle="sub",
        pattern="",
        color=(1.0, 0.0, 0.0),
        facts=("fact one", "fact <two>", "fact three"),
        mystery="why?",
        radius=0.3,
    )
    base.update(overrides)
    return UniverseStory(**base)  # type: ignore[arg-type]


def test_universe_masks_select_by_dominant_pfam_phylum_and_darkness() -> None:
    u = _universe()
    assert u.pfam_mask(("PF00042",)).sum() == 270
    assert u.pfam_mask(("PF00042", "PF99999")).sum() == 870
    assert u.phylum_mask("Uroviricota").sum() == 500
    assert u.dark_mask().sum() == 500
    names = u.phylum_names()
    assert names[0] == "Pseudomonadota" and names[4000] == "Chordata"
    assert names[-1] == ""  # no phylum recorded


def test_densest_core_prefers_the_pure_knot_over_the_bigger_mixed_one() -> None:
    u = _universe()
    tree = spatial.cKDTree(u.positions)
    fam = u.pfam_mask(("PF00042",))
    # The mixed knot has MORE globin members (150 vs 120) but is 80% something
    # else; the purity weight lands the seed on the pure knot.
    core = densest_core(u.positions[fam].astype(np.float64), 0.2, tree)
    assert np.linalg.norm(core - [6.0, 0.0, 0.0]) < 0.3


def test_select_members_keeps_the_knot_and_reports_the_family_size() -> None:
    u = _universe()
    tree = spatial.cKDTree(u.positions)
    story = _story(pfam=("PF00042",))
    mask = family_mask(story, u, None)
    cluster = select_universe_members(story, u, mask, tree)
    assert cluster.n_named == 270
    assert set(cluster.indices.tolist()) == set(range(4000, 4120))
    assert np.linalg.norm(cluster.centre - [6.0, 0.0, 0.0]) < 0.1
    assert 0 < cluster.r50 <= cluster.r95 < 0.3


def test_whole_story_lights_every_member_with_no_knot_cut() -> None:
    u = _universe()
    tree = spatial.cKDTree(u.positions)
    story = _story(region="dark", whole=True)
    cluster = select_universe_members(story, u, family_mask(story, u, None), tree)
    assert cluster.n_named == 500 == len(cluster.indices)
    assert set(cluster.indices.tolist()) == set(range(2000, 2500))
    assert cluster.r95 > 1.0  # the whole cloud, not a knot


def test_spur_check_demands_the_abc_family(monkeypatch: pytest.MonkeyPatch) -> None:
    u = _universe()
    tree = spatial.cKDTree(u.positions)
    story = _story(region="spur", whole=True)
    cluster = select_universe_members(story, u, family_mask(story, u, None), tree)
    # The synthetic spur is PF00001-dominated, not PF00005: the guard fires.
    assert demo.spur_pfam_fraction(u, cluster) == 0.0
    with pytest.raises(ValueError, match="only 0%"):
        demo.check_spur_story(u, cluster)
    # ... and passes once the family matches the panel's claim.
    monkeypatch.setattr(demo, "SPUR_PFAM", "PF00001")
    assert demo.check_spur_story(u, cluster) == 1.0


def test_whole_highlight_sample_is_all_or_a_fixed_sorted_subset() -> None:
    small = np.arange(10, 10 + demo.WHOLE_HIGHLIGHT_MAX_POINTS - 1)
    assert demo.whole_highlight_sample(small) is small
    big = np.arange(0, 3 * demo.WHOLE_HIGHLIGHT_MAX_POINTS, 1)
    a = demo.whole_highlight_sample(big)
    b = demo.whole_highlight_sample(big)
    assert len(a) == demo.WHOLE_HIGHLIGHT_MAX_POINTS
    assert np.array_equal(a, b)  # seeded: the same members every build
    assert np.all(np.diff(a) > 0)  # sorted, no duplicates
    assert np.isin(a, big).all()


def test_highlight_unit_states_the_sampling_ratio() -> None:
    assert demo.highlight_unit(159, 159) == "clusters"
    assert demo.highlight_unit(2_025_330, 300_000) == "clusters (one in 7 drawn)"
    assert demo.highlight_unit(580_733, 300_000) == "clusters (one in 2 drawn)"


def test_side_on_camera_looks_across_the_spur_not_along_it() -> None:
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    centre = np.array([0.0, 0.0, 30.0])  # straight out along +z from the origin
    cluster = StoryCluster(
        indices=np.arange(20), centre=centre, r95=6.0, n_named=20, r50=3.0, r_max=9.0
    )
    outside_in = demo.universe_story_camera(cluster, _story(), np.zeros(3))
    side_on = demo.universe_story_camera(cluster, _story(side_on=True), np.zeros(3))
    to_c = lambda pose: np.asarray(pose.position) - centre  # noqa: E731
    radial = centre / np.linalg.norm(centre)
    # Default pose sits along the centre→cluster ray (plus the small lift).
    assert (
        abs(np.dot(to_c(outside_in), radial)) / np.linalg.norm(to_c(outside_in)) > 0.9
    )
    # Side-on sits perpendicular to that ray, at the same distance rule.
    assert abs(np.dot(to_c(side_on), radial)) / np.linalg.norm(to_c(side_on)) < 0.35
    assert np.isclose(np.linalg.norm(to_c(side_on)), np.linalg.norm(to_c(outside_in)))
    assert tuple(side_on.target) == tuple(centre)


def test_region_stories_select_by_predicate_not_by_family() -> None:
    u = _universe()
    assert family_mask(_story(region="spur"), u, None).sum() == 50
    assert family_mask(_story(region="dark"), u, None).sum() == 500
    assert family_mask(_story(region="phage"), u, None).sum() == 500
    with pytest.raises(ValueError, match="unknown region"):
        family_mask(_story(region="nebula"), u, None)


def test_name_pattern_needs_a_name_mask_and_unions_with_pfam() -> None:
    u = _universe()
    story = _story(pattern="hemoglobin", pfam=("PF00001",))
    with pytest.raises(ValueError, match="no names given"):
        family_mask(story, u, None)
    by_name = np.zeros(len(u), dtype=bool)
    by_name[3000:3010] = True
    assert family_mask(story, u, by_name).sum() == 250 + 10  # PF00001: cloud + spur


def test_select_members_fails_loudly_when_nothing_matches() -> None:
    u = _universe()
    tree = spatial.cKDTree(u.positions)
    story = _story(pfam=("PF12345",))
    with pytest.raises(ValueError, match="no cluster matched"):
        select_universe_members(story, u, family_mask(story, u, None), tree)


def test_phylum_groups_cover_the_domains_of_life() -> None:
    assert phylum_group("Pseudomonadota") == "Proteobacteria"
    assert phylum_group("Bacillota") == "Firmicutes & Actino"
    assert phylum_group("Bacteroidota") == "Other Bacteria"
    assert phylum_group("Methanobacteriota") == "Archaea"
    assert phylum_group("Uroviricota") == "Viruses"
    assert phylum_group("Nucleocytoviricota") == "Viruses"
    assert phylum_group("Chordata") == "Other Vertebrates"
    assert phylum_group("Streptophyta") == "Plants"
    assert phylum_group("Ascomycota") == "Fungi"
    assert phylum_group("") == "Other"
    assert phylum_group("Candidatus Neverheardota") == "Other"


def test_backdrop_colours_follow_the_palette_and_dim_dark_clusters() -> None:
    u = _universe()
    rgb = backdrop_colors(u)
    assert rgb.shape == (len(u), 3) and rgb.dtype == np.float32
    from luxar.demos.demo_esm3_protein_landscape import TAXON_COLORS

    assert np.allclose(rgb[0], TAXON_COLORS["Proteobacteria"])
    assert np.allclose(rgb[4000], TAXON_COLORS["Other Vertebrates"])
    assert np.allclose(rgb[-1], TAXON_COLORS["Other"])
    # Dark phage clusters: the virus colour, dimmed.
    assert np.allclose(rgb[2000], np.asarray(TAXON_COLORS["Viruses"]) * demo.DARK_DIM)


def test_highlight_intensity_holds_for_a_knot_and_dims_a_region() -> None:
    assert highlight_intensity(10) == HIGHLIGHT_INTENSITY
    assert highlight_intensity(HIGHLIGHT_REFERENCE_MEMBERS) == HIGHLIGHT_INTENSITY
    four_x = highlight_intensity(4 * HIGHLIGHT_REFERENCE_MEMBERS)
    assert four_x == pytest.approx(HIGHLIGHT_INTENSITY / 2)
    assert highlight_intensity(10_000) < four_x


def test_member_labels_name_the_taxon_and_link_the_uniref_match() -> None:
    labels, keys = member_labels(
        ["Globin", "hypothetical protein", "Lysozyme"],
        ["Chordata", "", ""],
        ["", "domain:Bacteria", ""],
        ["UniRef90_P69905", "", ""],
    )
    assert labels == [
        "Globin — Chordata",
        "hypothetical protein — Bacteria",
        "Lysozyme",
    ]
    assert keys == ["UniRef90_P69905", "hypothetical protein", "Lysozyme"]


def test_overview_panel_carries_the_count_and_the_credit() -> None:
    panel = overview_panel_html(7_714_508)
    assert "7,714,508" in panel
    assert demo.ATTRIBUTION in panel
    assert "Twelve stories" in panel


# --------------------------------------------------------------------------- #
# The shipped stories
# --------------------------------------------------------------------------- #


def test_twelve_shipped_stories_are_well_formed() -> None:
    keys = [s.key for s in STORIES]
    assert len(keys) == len(set(keys)) == 12
    for s in STORIES:
        assert isinstance(s, UniverseStory) and isinstance(s, Story)
        assert "/" not in s.key, f"{s.key!r} doubles as a node name; '/' is refused"
        assert s.pfam or s.pattern or s.region, f"{s.key}: no selector"
        assert s.region in (None, *demo.VALID_REGIONS), s.key
        assert s.kingdom is None, s.key  # Swiss-Prot taxon filter never applies here
        assert 3 <= len(s.facts) <= 5, s.key
        assert s.mystery.strip() and s.narration.strip(), s.key
        assert s.pdb_id, s.key  # every stop has a turntable
        assert all(0.0 <= c <= 1.0 for c in s.color), s.key
        spoken = story_narration(s)
        assert 40 <= len(spoken.split()) <= 110, (s.key, len(spoken.split()))
        assert not spoken.startswith("Story "), s.key
        assert "<" not in spoken and "&" not in spoken, s.key
        assert spoken.rstrip().endswith((".", "?")), s.key
    assert 20 <= len(demo.OVERVIEW_NARRATION.split()) <= 80
    # Three region stories, in this order, so the tour builds up to the map's
    # own stories after the classic families.
    assert [s.region for s in STORIES if s.region] == ["spur", "dark", "phage"]
    # A story about the map itself lights EVERY member (the panel says "a quarter
    # of the map"; the picture must agree), so it has no knot cut and no bubble.
    assert all(s.whole for s in STORIES if s.region)
    assert not any(s.whole for s in STORIES if not s.region)


def test_carried_stories_keep_their_vetted_facts() -> None:
    """A carried story changes only its map-specific lines (see ``_carry``)."""
    base = {s.key: s for s in SWISSPROT_STORIES}
    carried = [s for s in STORIES if s.key in base]
    assert len(carried) == 7
    for s in carried:
        b = base[s.key]
        assert s.mystery == b.mystery, s.key
        assert s.pdb_id == b.pdb_id or s.key == "Viral surface proteins", s.key
        assert len(s.facts) == len(b.facts), s.key
        unchanged = sum(f in b.facts for f in s.facts)
        assert unchanged >= len(b.facts) - 1, s.key  # at most ONE fact rewritten
        assert s.color == b.color, s.key


def test_shipped_stories_author_valid_waypoints() -> None:
    from luxar.core.viewer_config import Waypoint
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    cluster = StoryCluster(
        indices=np.arange(50),
        centre=np.array([3.0, -2.0, 1.0]),
        r95=0.12,
        n_named=200,
        r50=0.06,
    )
    waypoints = [
        Waypoint(
            when={STORY_DIM: k},
            camera=story_camera(cluster, s, np.zeros(3), min_radius=BUBBLE_MIN_RADIUS),
            duration_ms=s.flight_ms,
            reveal="on_arrival",
        )
        for k, s in enumerate(STORIES, start=1)
    ]
    config = ViewerConfig(waypoints=waypoints)
    assert len(config.to_dict()["waypoints"]) == 12
    # A knot smaller than the bubble floor is framed at the story's own
    # distance floor, never closer than the bubble.
    pose = waypoints[0].camera
    distance = np.linalg.norm(np.asarray(pose.position) - np.asarray(pose.target))
    assert distance >= max(STORIES[0].min_distance, 1.2 * BUBBLE_MIN_RADIUS)
