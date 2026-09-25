"""Unit tests for the ESM protein-universe demo's pure helpers.

The demo itself needs the two hand-delivered parquet files (~900 MB), so the
scene build is exercised manually; these tests pin the parts that decide WHAT a
story highlights and WHERE the camera goes, on synthetic data, plus the
well-formedness of the twenty shipped stories.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.dimensions import Dimension
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
    highlight_appearance,
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
    pct = np.full(n, 100, dtype=np.float32)
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


def test_dark_mask_preserves_fractional_values_and_excludes_unknown_joins() -> None:
    u = replace(
        _universe(),
        annotation_row=np.array([0, 1, 2, -1], dtype=np.int32),
        pct_characterized=np.array([0.0, 0.4, 2.5, 0.0], dtype=np.float32),
    )

    assert u.dark_mask().tolist() == [True, False, False, False]


def test_characterized_percentages_reject_nulls_without_truncating_fractions() -> None:
    pa = pytest.importorskip("pyarrow")
    fractional = pa.chunked_array([pa.array([0.0, 0.4]), pa.array([2.5, 100.0])])
    values = demo._characterized_percentages(fractional)
    assert values.dtype == np.float32
    assert values.tolist() == pytest.approx([0.0, 0.4, 2.5, 100.0])

    null_integer = pa.chunked_array([pa.array([0, 5, None, 100], type=pa.int64())])
    with pytest.raises(ValueError, match="null or non-finite"):
        demo._characterized_percentages(null_integer)


def test_backdrop_compositing_lives_only_on_partition_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo, "BACKDROP_TILE_POINTS", 2_000)
    rng = np.random.default_rng(12)
    positions = np.column_stack(
        [
            np.zeros(9_000, dtype=np.float32),
            rng.normal(size=(9_000, 3)).astype(np.float32),
        ]
    )
    colors = rng.random((9_000, 3), dtype=np.float32)
    dims = Dimensions(
        [
            Dimension(demo.STORY_DIM, unit="", categories=["Overview"], display=False),
            Dimension("x", unit="UMAP", display=True),
            Dimension("y", unit="UMAP", display=True),
            Dimension("z", unit="UMAP", display=True),
        ]
    )
    out = tmp_path / "backdrop.luxar.zarr"

    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        demo._add_backdrop(scene, positions, colors, dims)

    store = zarr.open_group(out, mode="r")
    wrapper = store["Backdrop"]
    assert wrapper.attrs["opacity"] == pytest.approx(demo.BACKDROP_OPACITY)
    assert wrapper.attrs["intensity"] == pytest.approx(demo.BACKDROP_INTENSITY)
    for part_name in wrapper.group_keys():
        part = wrapper[part_name]
        for child_name in part.group_keys():
            child = part[child_name]
            assert child.attrs["opacity"] == pytest.approx(1.0)
            assert child.attrs["intensity"] == pytest.approx(1.0)


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
    # The synthetic spur is PF00001-dominated, not an ABC family: the guard fires.
    assert demo.spur_pfam_fraction(u, cluster) == 0.0
    with pytest.raises(ValueError, match="only 0%"):
        demo.check_spur_story(u, cluster)
    # ... and passes once the family matches the panel's claim.
    monkeypatch.setattr(demo, "SPUR_PFAMS", ("PF00001",))
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


def test_spur_mask_keeps_the_streak_and_drops_far_stragglers() -> None:
    u = _universe()
    # Add a straggler equally far out but 90 degrees off the spur's axis.
    positions = np.vstack([u.positions, [[30.0, 0.0, 0.0]]]).astype(np.float32)
    mask = demo.spur_mask(positions)
    assert mask[4870:4920].all()  # the streak along +z
    assert not mask[-1]  # the straggler along +x
    assert mask.sum() == 50
    assert not demo.spur_mask(np.zeros((5, 3), dtype=np.float32)).any()
    symmetric = np.array([[30.0, 0.0, 0.0], [-30.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no dominant direction"):
        demo.spur_mask(symmetric)


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


def test_groups_filter_keeps_only_the_named_branches_of_life() -> None:
    """Hemoglobin's guard: the globin knot the story lands on must be animal."""
    universe = _universe()
    unfiltered = family_mask(_story(pfam=("PF00042",)), universe, None)
    animal = family_mask(
        _story(pfam=("PF00042",), groups=("Other Vertebrates",)), universe, None
    )
    assert unfiltered.sum() == 270
    assert animal.sum() == 120  # the Chordata knot only
    assert np.flatnonzero(animal).min() >= 4000
    assert np.flatnonzero(animal).max() < 4120


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
    assert "Twenty stories" in panel


# --------------------------------------------------------------------------- #
# The shipped stories
# --------------------------------------------------------------------------- #


def test_shipped_stories_are_well_formed() -> None:
    keys = [s.key for s in STORIES]
    assert len(keys) == len(set(keys)) == 20
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
    # The converse does NOT hold: a SCATTER story is whole without being a
    # region (tyrosine decarboxylase selects a family that turns out not to be
    # one). Every non-region whole story must be exactly that case, so a story
    # cannot silently lose its knot cut.
    assert all(s.scatter for s in STORIES if s.whole and not s.region)
    assert all(s.whole for s in STORIES if s.scatter)
    assert [s.key for s in STORIES if s.scatter] == ["Levodopa and the gut"]


def test_carried_stories_keep_their_vetted_facts() -> None:
    """A carried story changes only its map-specific lines (see ``_carry``)."""
    base = {s.key: s for s in SWISSPROT_STORIES}
    carried = [s for s in STORIES if s.key in base]
    assert len(carried) == 7
    for s in carried:
        b = base[s.key]
        # The viral question is re-aimed: this map is built from metagenomes.
        assert s.mystery == b.mystery or s.key == "Viral surface proteins", s.key
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
    assert len(config.to_dict()["waypoints"]) == 20
    # A knot smaller than the bubble floor is framed at the story's own
    # distance floor, never closer than the bubble.
    pose = waypoints[0].camera
    distance = np.linalg.norm(np.asarray(pose.position) - np.asarray(pose.target))
    assert distance >= max(STORIES[0].min_distance, 1.2 * BUBBLE_MIN_RADIUS)


# ---------------------------------------------------------------------------
# Render quality (2026-09-10 review: laptop defaults, kiosk behind a flag)
# ---------------------------------------------------------------------------


def test_default_viewer_config_is_the_laptop_build() -> None:
    """The hosted default avoids kiosk-only render and dolly costs."""
    from luxar.demos import demo_esm_protein_universe as demo

    vc = demo._viewer_config([], (0.0, 0.0, 100.0), auto_rotate=True, audio=False)
    assert vc.ssaa_enabled is False
    assert vc.allow_high_dpr is False
    assert vc.auto_dolly_amplitude_percent == 20.0


def test_auto_dolly_rides_with_the_turntable() -> None:
    """The dolly breathes under the spin, and stops when the spin does.

    `--no-auto-rotate` exists to ask for a still camera; a scene that stopped
    rotating but kept sliding in and out would be a worse answer than either.
    """
    from luxar.demos import demo_esm_protein_universe as demo

    spinning = demo._viewer_config([], (0.0, 0.0, 100.0), auto_rotate=True, audio=False)
    assert spinning.auto_dolly is True
    assert spinning.auto_dolly_amplitude_percent == 20.0
    assert spinning.auto_dolly_period == 58.5

    still = demo._viewer_config([], (0.0, 0.0, 100.0), auto_rotate=False, audio=False)
    assert still.auto_dolly is False
    # Unset, not zero: the tri-state contract is that an omitted field keeps
    # the viewer's own default rather than authoring a degenerate one.
    assert still.auto_dolly_amplitude_percent is None
    assert still.auto_dolly_period is None


def test_high_quality_restores_kiosk_settings() -> None:
    from luxar.demos import demo_esm_protein_universe as demo

    vc = demo._viewer_config(
        [], (0.0, 0.0, 100.0), auto_rotate=True, audio=False, high_quality=True
    )
    assert vc.ssaa_enabled is True
    assert vc.allow_high_dpr is True
    assert vc.auto_dolly_amplitude_percent == 95.0


def test_high_quality_flag_is_spelled_in_main_and_documented() -> None:
    """The CLI switch and its dolly effect stay visible to users."""
    import inspect

    from luxar.demos import demo_esm_protein_universe as demo

    src = inspect.getsource(demo.main)
    assert '"--high-quality" in sys.argv' in src
    assert "high_quality=high_quality" in src
    module_doc = demo.__doc__ or ""
    assert "--high-quality" in module_doc
    assert "95% dolly" in module_doc
    touch_panel = module_doc.index("Touch panel (off by default):")
    assert module_doc.index("--high-quality") < touch_panel
    assert module_doc.index("--coords X.parquet --annotations Y.parquet") < touch_panel
    assert "95% dolly swing" in (demo.build_universe_scene.__doc__ or "")


# ---------------------------------------------------------------------------
# Scatter stories (2026-09-16 review: a requested family that is not one)
# ---------------------------------------------------------------------------


def test_scatter_story_demands_the_whole_selection() -> None:
    """A scatter story is about every member, so it cannot take a knot cut."""
    with pytest.raises(ValueError, match="scatter=True needs whole=True"):
        _story(pattern="x", scatter=True)

    # whole=True makes it legal, and the flag survives construction.
    story = _story(pattern="x", whole=True, scatter=True)
    assert story.scatter and story.whole


def test_highlight_appearance_has_one_regime_per_kind_of_story() -> None:
    """Knot points, a region's sub-pixel shadow, a scatter's countable dots."""
    knot = _story(pfam=("PF00042",))
    region = _story(region="dark", whole=True)
    scatter = _story(pattern="x", whole=True, scatter=True)

    assert highlight_appearance(knot, 50) == (
        demo.HIGHLIGHT_RADIUS,
        highlight_intensity(50),
    )
    # A big knot is dimmed by the member count; a region ignores it.
    assert highlight_appearance(knot, 10_000)[1] < highlight_appearance(knot, 50)[1]
    assert highlight_appearance(region, 300_000) == (
        demo.WHOLE_HIGHLIGHT_RADIUS,
        demo.WHOLE_HIGHLIGHT_INTENSITY,
    )
    assert highlight_appearance(scatter, 52) == (
        demo.SCATTER_HIGHLIGHT_RADIUS,
        demo.SCATTER_HIGHLIGHT_INTENSITY,
    )
    # The whole point of a scatter marker: visible where a region's is not.
    assert demo.SCATTER_HIGHLIGHT_RADIUS > 10 * demo.WHOLE_HIGHLIGHT_RADIUS
    # ... and dimmer than a knot's, because each marker covers many pixels.
    assert demo.SCATTER_HIGHLIGHT_INTENSITY < HIGHLIGHT_INTENSITY


def test_carries_labels_follows_the_member_count_not_the_story_kind() -> None:
    """A scatter story's few dozen members are labelled; a region's millions are not.

    Hover labels are per-member strings in the store, so the dark proteome's
    two million would dominate it; 52 tyrosine decarboxylase clusters are worth
    reading and cost nothing.
    """
    knot = _story(pfam=("PF00042",))
    region = _story(region="dark", whole=True)
    scatter = _story(pattern="x", whole=True, scatter=True)
    cap = demo.LABELLED_MEMBER_CAP

    # A knot is labelled whatever its size (its knot cut keeps it small).
    assert demo.carries_labels(knot, 1)
    assert demo.carries_labels(knot, cap + 1)
    # A whole story is labelled up to the cap and not past it.
    assert demo.carries_labels(scatter, 52)
    assert demo.carries_labels(scatter, cap)
    assert not demo.carries_labels(scatter, cap + 1)
    assert not demo.carries_labels(region, 2_025_330)
    # The cap sits below the whole-map draw budget, so no region story can
    # slip under it by being sampled.
    assert cap < demo.WHOLE_HIGHLIGHT_MAX_POINTS


def test_shipped_stories_label_the_scatter_and_not_the_regions() -> None:
    """The rule, applied to the real tour with its measured member counts."""
    measured = {
        "Dark proteome": 2_025_330,
        "Phage": 580_733,
        "ABC transporters": 8_277,
        "Levodopa and the gut": 52,
    }
    by_key = {s.key: s for s in STORIES}
    assert demo.carries_labels(
        by_key["Levodopa and the gut"], measured["Levodopa and the gut"]
    )
    for key in ("Dark proteome", "Phage", "ABC transporters"):
        assert not demo.carries_labels(by_key[key], measured[key]), key


def test_the_scatter_story_is_the_one_family_that_is_not_a_family() -> None:
    """Tyrosine decarboxylase is authored as a scatter on purpose.

    Measured on this Atlas release (2026-09-16): 52 clusters name it, their
    densest ball holds 21 at 4% PURITY inside the group II PLP decarboxylase
    fold (1,909 clusters in clan CL0061; an earlier draft said 4,170, which
    counted three clans and two unrelated folds), and their phyla are
    scattered. A knot story here would claim a family the map does not show,
    so the panel's last two facts are about the scatter itself.

    It is not a constellation either, and the reason is measured rather than
    stylistic (re-derived 2026-09-17). Two of its components DO clear the
    10-member floor, so the rule would allow a figure: places of 21 and 13,
    joined by exactly ONE line. But a constellation lights only the places it
    joins, so that figure would light 34 of the 52 and HIDE 18 — 35% of the
    story's subject, including every singleton, and the singletons are the
    scatter. The one line would also assert precisely the family relation the
    story exists to say is absent.
    """
    (story,) = [s for s in STORIES if s.scatter]
    assert story.key == "Levodopa and the gut"
    assert story.pattern and story.pfam  # name match UNION the dedicated Pfam
    assert story.region is None  # not a map-wide predicate: a family selector
    assert story.frame_fraction >= 1.0  # pulled back to hold the whole scatter
    assert any("never make a family of their own" in f for f in story.facts)
    assert not story.constellation


def test_the_worm_knot_keeps_its_deliberately_smaller_radius() -> None:
    """The nematode chemoreceptor radius is 0.2, not FAMILY_RADIUS.

    Measured both ways (2026-09-16): the family has several comparable
    components, and at 0.3 the purity-weighted seed abandons the tight knot
    (55 members, r95 0.051, 95% ball purity, every member a nematode) for a
    diffuse 40-member component at 20% purity. This pins the choice so a later
    tidy-up cannot silently widen it.
    """
    (worm,) = [s for s in STORIES if s.key == "Worm chemoreceptors"]
    assert worm.radius == 0.2 < demo.FAMILY_RADIUS


def test_every_new_story_selects_by_pfam_or_name_and_names_its_structure() -> None:
    """The stories added after the 2026-09-16 review and the 09-17 follow-up."""
    added = [
        "Lanthipeptides",
        "Ice-binding proteins",
        "Reverse gyrase",
        "Olfactory receptors",
        "Insect odorant receptors",
        "Worm chemoreceptors",
        "TnpB and Fanzor",
        "Levodopa and the gut",
    ]
    by_key = {s.key: s for s in STORIES}
    assert sorted(added) == sorted(k for k in added if k in by_key)
    # They used to be pinned to the last eight slots, so the kiosk chapter
    # indices of the original twelve would not move. That is no longer the
    # contract: the tour is walked in an explicit narrative order
    # (`TOUR_ORDER`), which interleaves new stops with old ones and therefore
    # DOES renumber the chapters. Nothing persists a chapter index across
    # builds, so the cost is a stale bookmark, not a broken kiosk.
    for key in added:
        s = by_key[key]
        assert s.key not in {c.key for c in SWISSPROT_STORIES}  # not carried
        assert s.pfam or s.pattern, key
        assert s.pdb_id and s.pdb_id == s.pdb_id.upper(), key
        assert s.tags, key
    # Distinct turntables: two stops showing the same molecule would read as a
    # bug on the kiosk.
    pdbs = [s.pdb_id.upper() for s in STORIES]
    assert len(pdbs) == len(set(pdbs))


def test_scatter_camera_frames_the_map_not_the_members_bounding_box() -> None:
    """A scatter story's picture must contain the cloud its panel talks about.

    The bounding-box centre of a few dozen scattered clusters is a biased
    point, so aiming there leaves the map in one half of the frame.
    """
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    cluster = StoryCluster(
        indices=np.arange(52),
        centre=np.array([-7.63, 2.49, 0.2]),  # measured for the shipped scatter
        r95=6.79,
        n_named=52,
        r50=5.94,
        r_max=7.63,
    )
    story = _story(pattern="x", whole=True, scatter=True, frame_fraction=1.0)
    pose = demo.universe_story_camera(cluster, story, np.zeros(3), map_radius=20.0)

    # Aimed at the MAP's centre, not the members'.
    assert tuple(pose.target) == (0.0, 0.0, 0.0)
    # Far enough back that the map's radius fits the frame height.
    distance = float(np.linalg.norm(np.asarray(pose.position)))
    assert distance > 20.0
    # Approached from the members' own side of the cloud, so the shot differs
    # from the Overview's.
    to_camera = np.asarray(pose.position)
    assert float(np.dot(to_camera[[0, 2]], cluster.centre[[0, 2]])) > 0

    # A knot story is unaffected and needs no map radius.
    knot = _story(pfam=("PF00042",))
    assert demo.universe_story_camera(cluster, knot, np.zeros(3)).target != (
        0.0,
        0.0,
        0.0,
    )


def test_a_scatter_story_refuses_to_be_framed_without_the_map_radius() -> None:
    """Fail loudly rather than silently framing a scatter like a knot."""
    from luxar.demos.demo_esm3_protein_stories import StoryCluster

    cluster = StoryCluster(
        indices=np.arange(3),
        centre=np.array([1.0, 0.0, 0.0]),
        r95=5.0,
        n_named=3,
        r50=4.0,
    )
    with pytest.raises(ValueError, match="pass map_radius"):
        demo.universe_story_camera(
            cluster, _story(pattern="x", whole=True, scatter=True), np.zeros(3)
        )


def test_only_one_story_claims_the_tour_is_tightest_knot() -> None:
    """Superlatives are measured, so only one story may hold each.

    Measured r95 over the fifteen knot stories (2026-09-16): reverse gyrase
    0.020 is the tightest, then ice-binding 0.044 and the worm chemoreceptors
    0.051; the vertebrate olfactory knot is EIGHTH at 0.123, so an earlier
    draft calling it "among the tightest" was wrong and now says it is the
    largest of the three smell knots instead. This guard keeps two stories from
    claiming the same crown after a later edit.
    """
    unscoped, scoped = [], []
    for s in STORIES:
        body = " ".join(s.facts + (s.subtitle, s.title)).lower()
        if "tightest" not in body:
            continue
        (scoped if "of the three" in body else unscoped).append(s.key)
    assert unscoped == ["Reverse gyrase"], unscoped
    assert scoped == ["Worm chemoreceptors"], scoped
    # And the one that claims it is the smallest knot is the same story.
    smallest = [s.key for s in STORIES if "smallest" in " ".join(s.facts).lower()]
    assert smallest == ["Reverse gyrase"], smallest


def test_the_ice_story_never_selects_the_mislabelled_pfam_again() -> None:
    """PF07589 is a protein-sorting motif, not an ice-binding domain.

    The Atlas's own ``cluster_top_pfam_names`` column calls PF07589
    "Ice-binding protein, C-terminal domain"; InterPro and current Pfam call it
    "PEP-CTERM protein-sorting motif". Including it drew a 94-cluster knot of
    PVC bacteria carrying a secretion signal, and it supplied 2,076 of the
    2,587 clusters that story used to claim. This pins the corrected selector
    so the Atlas's label cannot lead us back.
    """
    (ice,) = [s for s in STORIES if s.key == "Ice-binding proteins"]
    assert "PF07589" not in ice.pfam
    assert ice.pfam == ("PF11999", "PF20597", "PF21300")
    # No other story may pick it up either.
    for s in STORIES:
        assert "PF07589" not in (s.pfam or ()), s.key


# ---------------------------------------------------------------------------
# Constellations (2026-09-16: light every place a family sits, join them with
# lines, and frame the whole figure)
# ---------------------------------------------------------------------------

_MAP_CENTRE = np.zeros(3)


def _figure_with_r_max(r_max: float) -> demo.Constellation:
    return demo.Constellation(
        places=(np.array([0]), np.array([1])),
        centroids=np.zeros((2, 3), dtype=np.float32),
        edges=np.array([[0, 1]]),
        members=np.array([0, 1]),
        centre=np.zeros(3),
        r_max=r_max,
        view=np.array([0.0, 0.0, 1.0]),
    )


def test_constellation_joins_the_big_places_and_skips_the_stragglers() -> None:
    """Two dense places plus a singleton: one line, and the singleton is out."""
    rng = np.random.default_rng(3)
    a = np.array([0.0, 0.0, 0.0]) + rng.normal(size=(40, 3)) * 0.02
    b = np.array([5.0, 0.0, 0.0]) + rng.normal(size=(30, 3)) * 0.02
    stray = np.array([[-9.0, 0.0, 0.0]])
    other = rng.normal(size=(200, 3)) * 3.0  # not in the family
    positions = np.vstack([a, b, stray, other]).astype(np.float32)
    mask = np.zeros(len(positions), dtype=bool)
    mask[: len(a) + len(b) + 1] = True

    figure = demo.constellation_of(positions, mask, _MAP_CENTRE)
    assert figure is not None

    assert len(figure.centroids) == 2, "the one-member place is below the floor"
    assert figure.edges.shape == (1, 2)
    got = sorted(float(c[0]) for c in figure.centroids)
    assert got[0] == pytest.approx(0.0, abs=0.02)
    assert got[1] == pytest.approx(5.0, abs=0.02)
    # An MST over n places has exactly n-1 edges.
    assert len(figure.edges) == len(figure.centroids) - 1
    # The stray is NOT lit: the highlight is exactly the joined places.
    assert len(figure.members) == len(a) + len(b)
    assert np.array_equal(figure.members, np.sort(np.concatenate(figure.places)))


def test_constellation_endpoints_sit_on_lit_clusters() -> None:
    """The bug the redesign fixes: a line must start and end on a lit place.

    Every centroid is the mean of one place, and every place is a subset of
    the lit members — so an endpoint can never float free of the blob it
    appears to leave. Before the redesign the lines were drawn from the
    family's components while the highlight lit a KNOT ball around a
    purity-weighted seed, and the two disagreed by up to 0.45 world units.
    """
    rng = np.random.default_rng(11)
    places = [
        np.array(c) + rng.normal(size=(25, 3)) * 0.02
        for c in ([0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [-3.0, 2.0, 1.0])
    ]
    positions = np.vstack([*places, rng.normal(size=(150, 3)) * 5.0]).astype(np.float32)
    mask = np.zeros(len(positions), dtype=bool)
    mask[: 25 * 3] = True

    figure = demo.constellation_of(positions, mask, _MAP_CENTRE)
    assert figure is not None
    assert len(figure.places) == 3 and len(figure.edges) == 2

    lit = set(figure.members.tolist())
    for place, centroid in zip(figure.places, figure.centroids, strict=True):
        assert set(place.tolist()) <= lit
        assert centroid == pytest.approx(positions[place].mean(axis=0), abs=1e-5)
    # Every endpoint index addresses a real place.
    assert int(figure.edges.min()) >= 0
    assert int(figure.edges.max()) < len(figure.centroids)


def test_constellation_is_none_when_the_family_sits_in_one_place() -> None:
    """A family that is not scattered has no figure — the caller must not draw one."""
    rng = np.random.default_rng(4)
    positions = np.vstack(
        [np.zeros((50, 3)) + rng.normal(size=(50, 3)) * 0.02, rng.normal(size=(50, 3))]
    ).astype(np.float32)
    mask = np.zeros(len(positions), dtype=bool)
    mask[:50] = True

    assert demo.constellation_of(positions, mask, _MAP_CENTRE) is None
    # And an empty family is handled rather than raising.
    empty = np.zeros(len(positions), dtype=bool)
    assert demo.constellation_of(positions, empty, _MAP_CENTRE) is None


def test_constellation_respects_its_floor_and_radius() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(12, 3)) * 0.02
    b = np.array([4.0, 0.0, 0.0]) + rng.normal(size=(12, 3)) * 0.02
    positions = np.vstack([a, b]).astype(np.float32)
    mask = np.ones(len(positions), dtype=bool)

    assert demo.constellation_of(positions, mask, _MAP_CENTRE) is not None
    # Raise the floor above both places and there is no figure.
    assert demo.constellation_of(positions, mask, _MAP_CENTRE, min_members=20) is None
    # Widen the linkage until the two places merge into one.
    assert demo.constellation_of(positions, mask, _MAP_CENTRE, radius=9.0) is None


def test_constellation_view_axis_is_the_thin_one_and_points_outward() -> None:
    """The face-on direction: along the axis the figure is flattest."""
    rng = np.random.default_rng(7)
    # Three places strung along x and spread in y, but flat in z: the thin
    # axis is z, so the camera must look down z. (The y spread matters — three
    # isotropic blobs on a line leave y and z tied, and the "thinnest" axis of
    # a tie is arbitrary.)
    positions = np.vstack(
        [
            np.array([x, 0.0, 0.0])
            + rng.normal(size=(30, 3)) * np.array([0.02, 1.5, 0.02])
            for x in (-4.0, 0.0, 4.0)
        ]
    ).astype(np.float32)
    map_centre = np.array([0.0, 0.0, -50.0])

    figure = demo.constellation_of(
        positions, np.ones(len(positions), dtype=bool), map_centre
    )
    assert figure is not None
    assert abs(float(figure.view[2])) > 0.9, figure.view
    # Signed AWAY from the map centre, so the camera sits outside looking in.
    assert float(figure.view @ (figure.centre - map_centre)) > 0


def test_constellation_camera_frames_the_whole_figure_from_every_angle() -> None:
    """Distance is set from r_max, so auto-rotate cannot swing a node out of frame."""
    rng = np.random.default_rng(9)
    positions = np.vstack(
        [
            np.array([x, 0.0, 0.0]) + rng.normal(size=(30, 3)) * 0.02
            for x in (-6.0, 0.0, 6.0)
        ]
    ).astype(np.float32)
    figure = demo.constellation_of(
        positions, np.ones(len(positions), dtype=bool), _MAP_CENTRE
    )
    assert figure is not None
    story = _story(pfam=("PF00042",), constellation=True)
    camera = demo.constellation_camera(figure, story)

    distance = float(np.linalg.norm(np.asarray(camera.position) - figure.centre))
    assert camera.target == pytest.approx(tuple(figure.centre), abs=1e-5)
    half_height = distance * float(np.tan(np.radians(demo.STORY_LENS_FOV_DEG) / 2))
    # Every member is inside the frame even at the worst viewing angle, where
    # its whole distance from the centre is perpendicular to the view.
    assert figure.r_max <= half_height
    # And it is not framed so wide that the figure becomes a speck.
    assert figure.r_max / half_height == pytest.approx(
        demo.CONSTELLATION_FRAME_FRACTION, rel=1e-6
    )


def test_constellation_beads_hold_a_constant_angular_size() -> None:
    """Marker radius scales with the figure, so a small one is not sub-pixel."""
    story = _story(pfam=("PF00042",), constellation=True)
    r_small, _ = demo.highlight_appearance(story, 100, figure=_figure_with_r_max(2.0))
    r_large, _ = demo.highlight_appearance(story, 100, figure=_figure_with_r_max(12.0))
    assert r_large / r_small == pytest.approx(6.0, rel=1e-6)
    # Never below the knot marker, whatever the figure's scale.
    tiny, _ = demo.highlight_appearance(story, 100, figure=_figure_with_r_max(0.01))
    assert tiny == pytest.approx(demo.HIGHLIGHT_RADIUS)
    # Bigger highlights are dimmed, as everywhere else on the tour.
    _, bright = demo.highlight_appearance(story, 100, figure=_figure_with_r_max(5.0))
    _, dim = demo.highlight_appearance(story, 900, figure=_figure_with_r_max(5.0))
    assert dim < bright
    # A constellation without its figure is a programming error, not a default.
    with pytest.raises(ValueError, match="pass its figure"):
        demo.highlight_appearance(story, 100)


def test_constellation_lines_are_bright_and_still_tinted() -> None:
    """The owner's brief: nearly white, holding some of the story's hue."""
    color = (0.9, 0.25, 0.2)
    line = demo.constellation_line_color(color)
    assert line.min() > 0.7, "every channel is bright — the line reads as light"
    assert line.max() <= 1.0
    # The hue order survives, so the line still belongs to its story.
    assert line[0] > line[1] > line[2]
    # Brighter than the story colour it came from, channel by channel.
    assert np.all(line >= np.asarray(color, dtype=np.float32))
    # White stays white rather than overshooting.
    assert demo.constellation_line_color((1.0, 1.0, 1.0)) == pytest.approx(
        np.ones(3), abs=1e-6
    )


def test_only_pfam_selected_stories_may_be_constellations() -> None:
    """The line means "same Pfam family", so there must be a Pfam selector."""
    with pytest.raises(ValueError, match="constellation needs a pfam selector"):
        _story(pattern="x", constellation=True)
    assert _story(pfam=("PF00042",), constellation=True).constellation


def test_constellation_excludes_every_other_framing_mode() -> None:
    """A constellation is its own mode: no knot cut, no bubble, no groups filter."""
    for kwargs in (
        {"whole": True},
        {"whole": True, "scatter": True},
        {"side_on": True},
    ):
        with pytest.raises(ValueError, match="cannot be combined"):
            _story(pfam=("PF00042",), constellation=True, **kwargs)
    with pytest.raises(ValueError, match="cannot take a groups filter"):
        _story(pfam=("PF00042",), constellation=True, groups=("Bacteria",))


def test_the_shipped_constellations_are_the_three_whose_split_means_something() -> None:
    """Reduced from six on 2026-09-16: lines assert the split is meaningful.

    The three smell families were measured and dropped — every place they
    occupy is the same protein in another spot (four Chordata olfactory places
    with nothing telling them apart; one big insect place plus a crumb; six
    worm places all of mixed Sr families) — so a line between them would
    assert a structure the data does not carry.
    """
    assert [s.key for s in STORIES if s.constellation] == [
        "Hemoglobin",
        "Photosystem II",
        "Lanthipeptides",
    ]
    for s in STORIES:
        if s.constellation:
            assert s.pfam, s.key
            assert not s.whole and not s.scatter and not s.groups, s.key
            # The panel must say what the lines mean, or the figure is decoration.
            assert "lines join" in " ".join(s.facts).lower(), s.key


# ---------------------------------------------------------------------------
# Tour order (2026-09-16: an explicit narrative order, not authoring order)
# ---------------------------------------------------------------------------


def test_tour_order_is_a_permutation_of_the_authored_pool() -> None:
    assert len(demo.TOUR_ORDER) == len(set(demo.TOUR_ORDER)) == len(demo._STORY_POOL)
    assert set(demo.TOUR_ORDER) == {s.key for s in demo._STORY_POOL}
    assert [s.key for s in STORIES] == list(demo.TOUR_ORDER)


def test_tour_order_rejects_a_list_that_is_not_a_permutation() -> None:
    pool = demo._STORY_POOL
    with pytest.raises(ValueError, match="missing"):
        demo._ordered(pool, demo.TOUR_ORDER[:-1])
    with pytest.raises(ValueError, match="unknown"):
        demo._ordered(pool, (*demo.TOUR_ORDER, "Not a story"))


def test_tour_order_keeps_the_three_sequences_that_depend_on_it() -> None:
    """Three adjacencies are load-bearing, not taste."""
    order = list(demo.TOUR_ORDER)
    # The owner asked for these two together (2026-09-16): Cas12 links the
    # CRISPR system to its TnpB ancestors, so CRISPR hands off to TnpB.
    assert order.index("TnpB and Fanzor") == order.index("CRISPR-Cas") + 1
    # Each smell narration counts itself: "a second time", "a third time".
    smell = ["Olfactory receptors", "Insect odorant receptors", "Worm chemoreceptors"]
    first = order.index(smell[0])
    assert order[first : first + 3] == smell
    # The map's own geography, in the order the dark story sets up.
    assert [s.region for s in STORIES if s.region] == ["spur", "dark", "phage"]


# ---------------------------------------------------------------------------
# Text/picture agreement (2026-09-16: the panels must describe what is drawn)
# ---------------------------------------------------------------------------


def _story_text(story: UniverseStory) -> str:
    return " ".join([story.subtitle, *story.facts, story.narration]).lower()


def test_no_story_describes_a_framing_it_does_not_use() -> None:
    """The panel vocabulary has to match the mode the story is drawn in.

    Every mode has its own words: a knot story is "this knot" inside a bubble,
    a constellation says "the lines join N places", a whole-map story lights a
    region. Using another mode's words is the drift that is hardest to see,
    because the sentence still reads perfectly well.
    """
    for s in STORIES:
        text = _story_text(s)
        assert "bubble" not in text, f"{s.key}: panels never name the bubble"
        if s.constellation:
            assert "lines join" in text, f"{s.key}: a constellation must say so"
            assert "this knot" not in text, f"{s.key}: no knot is cut here"
        else:
            assert "lines join" not in text, (
                f"{s.key}: only a constellation draws lines"
            )
        if s.whole and not s.scatter:
            assert "this knot" not in text, f"{s.key}: no knot is cut here"


def test_the_scatter_story_does_not_claim_the_whole_map() -> None:
    """Its members span 17% of the map diagonal, not the map.

    Measured 2026-09-16: bounding box 15.3 units against the map's 89.4, and
    the 52 clusters draw as ~13 blobs because their median nearest-neighbour
    distance (0.022) is a fifth of `SCATTER_HIGHLIGHT_RADIUS`. The panel used
    to say "scattered across the whole map" and "right across the cloud", and
    then showed a dozen beads inside a fifth of the cloud.
    """
    scatter = [s for s in STORIES if s.scatter]
    assert scatter, "the tour has a scatter story"
    for s in scatter:
        text = _story_text(s)
        for claim in ("whole map", "across the cloud", "across the whole"):
            assert claim not in text, f"{s.key}: overclaims the extent ({claim!r})"
        # It must explain the gap between its count and its dots instead.
        assert "specks" in text, f"{s.key}: say what the picture actually shows"


def test_the_cross_story_reference_still_points_at_story_one() -> None:
    """The CRISPR panel calls sickle-cell "the illness of the first story".

    A reference by POSITION is the one thing `TOUR_ORDER` can silently break,
    so it is pinned here rather than left to a reading.
    """
    referrers = [s for s in STORIES if "first story on this tour" in _story_text(s)]
    assert referrers, "if the phrase is gone, delete this test with it"
    assert STORIES[0].key == "Hemoglobin"
    assert "sickle-cell" in _story_text(STORIES[0])
