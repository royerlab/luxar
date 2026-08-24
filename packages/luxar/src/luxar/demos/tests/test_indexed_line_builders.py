"""Invariant coverage for demos converted to indexed Lines authoring.

These tests pin the contracts that make thick curves continuous: every edge is
in range, every per-vertex attribute matches ``V``, no emitted vertex is orphaned,
and interior joints / branch hubs are represented by shared vertex indices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from luxar.demos import demo_bioluminescent_ocean as ocean
from luxar.demos import demo_dipc_3d_genome as dipc
from luxar.demos import demo_gsplats_4d_celegans_tracking as celegans
from luxar.demos import demo_lsystem_forest as forest
from luxar.demos import demo_particle_collision as collision
from luxar.demos import demo_particle_collision_animated as collision_animated


def _assert_indexed_geometry(
    vertices: np.ndarray,
    edges: np.ndarray,
    *attributes: Sequence[Any],
    require_shared_joint: bool = True,
) -> None:
    """Assert the common in-memory indexed-lines builder contract."""
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    n_vertices = len(vertices)

    assert vertices.ndim == 2
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert np.issubdtype(edges.dtype, np.integer)
    assert len(edges) > 0
    assert int(edges.min()) >= 0
    assert int(edges.max()) < n_vertices

    degree = np.bincount(edges.reshape(-1), minlength=n_vertices)
    assert np.all(degree > 0), "builder emitted an unreferenced vertex"
    if require_shared_joint:
        assert np.any(degree >= 2), "no joint reuses a shared vertex index"

    for attribute in attributes:
        assert len(attribute) == n_vertices


def _assert_stored_indexed_node(store_path: Path, node_name: str) -> None:
    """Assert the same contract after compiler ordering and serialization."""
    root = zarr.open_group(store_path, mode="r")
    node = root[node_name]
    n_vertices = int(node.attrs["n_vertices"])
    edges = np.asarray(node["segments"][:])

    assert node.attrs["original_line_type"] == "indexed"
    assert node["vertices"].shape[0] == n_vertices
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert len(edges) > 0
    assert int(edges.min()) >= 0
    assert int(edges.max()) < n_vertices

    degree = np.bincount(edges.reshape(-1), minlength=n_vertices)
    assert np.all(degree > 0), f"{node_name} contains orphaned vertices"
    assert np.any(degree >= 2), f"{node_name} contains no shared joints"

    for dataset_name in ("widths", "colors", "sharpnesses"):
        if dataset_name in node:
            assert node[dataset_name].shape[0] in (1, n_vertices)


def _assert_edges_stay_within_chains(
    edges: np.ndarray, chain_lengths: Sequence[int]
) -> None:
    """Ensure offset accumulation never connects two concatenated chains."""
    chain_of_vertex = np.empty(sum(chain_lengths), dtype=np.int64)
    offset = 0
    for chain_index, length in enumerate(chain_lengths):
        chain_of_vertex[offset : offset + length] = chain_index
        offset += length
    assert np.all(chain_of_vertex[edges[:, 0]] == chain_of_vertex[edges[:, 1]])


@pytest.mark.parametrize(
    "builder",
    [
        collision.generate_detector_geometry,
        collision_animated.generate_detector_geometry,
    ],
    ids=["static-collision", "animated-collision"],
)
def test_detector_builders_preserve_indexed_invariants(
    builder: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    vertices, widths, colors, sharpness, edges = builder(np.random.default_rng(0))
    _assert_indexed_geometry(vertices, edges, widths, colors, sharpness)


@pytest.mark.parametrize(
    "scene_builder,kwargs,node_names",
    [
        (
            collision.generate_detector_scene,
            {"n_events": 1, "n_jets_per_event": 1},
            ("detector_geometry", "particle_tracks"),
        ),
        (
            collision_animated.generate_animated_detector_scene,
            {"n_events": 1, "n_jets_per_event": 1, "n_frames": 4},
            ("detector_geometry", "particle_tracks"),
        ),
    ],
    ids=["static-collision", "animated-collision"],
)
def test_collision_scene_track_batches_survive_serialization(
    tmp_path: Path,
    scene_builder: Callable[..., tuple[int, int]],
    kwargs: dict[str, int],
    node_names: tuple[str, str],
) -> None:
    output = tmp_path / "collision.luxar.zarr"
    n_segments, _n_points = scene_builder(output, **kwargs)
    assert n_segments > 0
    for node_name in node_names:
        _assert_stored_indexed_node(output, node_name)


@pytest.mark.parametrize(
    "builder,chain_count,chain_length",
    [
        (ocean.generate_tentacles, ocean.N_TENTACLES, ocean.TENTACLE_SEGMENTS),
        (ocean.generate_oral_arms, ocean.N_ORAL_ARMS, 25),
    ],
    ids=["tentacles", "oral-arms"],
)
def test_ocean_curve_builders_preserve_offsets(
    builder: Callable[..., tuple[np.ndarray, ...]],
    chain_count: int,
    chain_length: int,
) -> None:
    jelly = ocean.create_jellyfish(1, np.random.default_rng(1))[0]
    vertices, widths, colors, sharpness, edges = builder(jelly, 1, 4)
    _assert_indexed_geometry(vertices, edges, widths, colors, sharpness)
    _assert_edges_stay_within_chains(edges, [chain_length] * chain_count)


def test_ocean_cross_frame_batch_has_no_orphans(tmp_path: Path) -> None:
    output = tmp_path / "ocean.luxar.zarr"
    n_segments, _n_points = ocean.generate_ocean_scene(
        output, n_jellyfish=2, n_frames=2
    )
    assert n_segments > 0
    _assert_stored_indexed_node(output, "tentacles")


def test_dipc_haplotype_builder_preserves_arm_boundaries() -> None:
    polylines = [
        {
            "vertices": np.arange(12, dtype=np.float32).reshape(4, 3),
            "positions": np.arange(4, dtype=np.float64) * 1_000_000,
            "color": np.array([0.8, 0.4, 0.2], dtype=np.float32),
            "chrom": "1",
            "haplotype": 0,
        },
        {
            "vertices": np.arange(9, dtype=np.float32).reshape(3, 3) + 100,
            "positions": np.arange(3, dtype=np.float64) * 1_000_000,
            "color": np.array([0.2, 0.6, 0.9], dtype=np.float32),
            "chrom": "2",
            "haplotype": 0,
        },
    ]
    vertices, colors, labels, keys, edges = dipc._haplotype_geometry(polylines, 0)
    _assert_indexed_geometry(vertices, edges, colors, labels, keys)
    _assert_edges_stay_within_chains(edges, [4, 3])
    # Keys are the UCSC locus for the same particle the label names (#1917), so
    # they are per-vertex like everything else and must stay aligned with it.
    assert len(keys) == len(vertices)

    # The locus itself, not merely its shape. The fixture's first bead sits at
    # bp 0, which is the clamp case: UCSC is 1-based and rejects a start below
    # 1, so the window has to open at 1 rather than -10000.
    assert keys[0] == "chr1:1-10000"
    # Away from the boundary the window is the documented 20 kb — Dip-C's own
    # resolution — centred on the bead.
    assert keys[1] == "chr1:990000-1010000"
    # The second arm keeps its own chromosome.
    assert keys[-1].startswith("chr2:")
    for locus in keys:
        span = locus.rsplit(":", 1)[1]
        start, end = (int(x) for x in span.split("-"))
        assert start >= 1, f"{locus} starts below 1"
        assert end > start


def test_lsystem_turtle_reuses_branch_hub() -> None:
    lsystem = forest.LSystem(
        axiom="F[+F]F",
        rules={},
        randomness=0.0,
    )
    vertices, edges, edge_depths, vertex_depths = forest.derive_tree(
        lsystem,
        iterations=0,
        seed=7,
    )
    _assert_indexed_geometry(vertices, edges, vertex_depths)
    assert len(edge_depths) == len(edges)

    degree = np.bincount(edges.reshape(-1), minlength=len(vertices))
    assert int(degree.max()) == 3, "the pushed/popped branch point was duplicated"


def test_lsystem_stochastic_expansion_is_seeded_and_normalized() -> None:
    """Stochastic productions sample deterministically from the seeded rng."""
    lsystem = forest.LSystem(
        axiom="X",
        rules={"X": [(0.5, "F[+X]"), (0.5, "F[-X]")]},
        randomness=0.0,
    )
    a = lsystem.expand(6, np.random.default_rng(11))
    b = lsystem.expand(6, np.random.default_rng(11))
    c = lsystem.expand(6, np.random.default_rng(12))
    assert a == b, "same seed must reproduce the same derivation"
    assert a != c, "different seeds should (overwhelmingly) diverge"
    assert set(a) <= set("F[]+-X")


def test_lsystem_tropism_bends_branches_not_the_trunk() -> None:
    """Tropism applies only at branching depth >= 1: trunks stay straight.

    The trunk is TILTED (leading ``+``) on purpose: a vertical trunk is
    antiparallel to the gravity tropism, where the bend is a no-op anyway
    (``H x T = 0``), so a vertical-trunk assertion would stay green even
    with the depth gate deleted.
    """
    axiom = "+FFFF[+FFFF]"
    straight = forest.LSystem(axiom=axiom, rules={}, randomness=0.0)
    drooped = forest.LSystem(
        axiom=axiom,
        rules={},
        randomness=0.0,
        tropism=(0.0, 0.0, -1.0),
        tropism_strength=0.4,
    )
    v_straight, e_straight, _, _ = forest.derive_tree(straight, iterations=0, seed=1)
    v_drooped, e_drooped, d_drooped, _ = forest.derive_tree(
        drooped, iterations=0, seed=1
    )
    assert np.array_equal(e_straight, e_drooped)
    # Trunk vertices (introduced at depth 0) are identical...
    trunk = v_straight[:5]
    np.testing.assert_allclose(v_drooped[:5], trunk, atol=1e-6)
    # ...while the branch tip ends up strictly lower under gravity droop.
    tip_straight = v_straight[e_straight[d_drooped >= 1][-1, 1]]
    tip_drooped = v_drooped[e_drooped[d_drooped >= 1][-1, 1]]
    assert tip_drooped[2] < tip_straight[2]


def test_lsystem_frame_survives_long_derivations() -> None:
    """Regression: frame vectors must stay unit through long command paths.

    Rodrigues rotation about a frame vector assumes a UNIT axis; without
    per-step renormalization the ~1e-16 float error per rotation compounds
    geometrically and the frame collapses to zero within ~250 rotations
    (measured), shrinking every drawn segment along the way.
    """
    lsystem = forest.LSystem(
        axiom="F" + "+F-F^F&F/F\\F" * 200,  # 1,200 rotations along ONE path
        rules={},
        randomness=0.3,
    )
    vertices, edges, _, _ = forest.derive_tree(lsystem, iterations=0, seed=5)
    segments = (
        vertices[edges[:, 1].astype(np.int64)] - vertices[edges[:, 0].astype(np.int64)]
    )
    lengths = np.linalg.norm(segments, axis=1)
    assert float(lengths.min()) > 0.999, "heading norm decayed along the walk"
    assert float(lengths.max()) < 1.001


def test_forest_foliage_cholesky_packs_the_5d_tril_layout() -> None:
    """The hand-packed 15-wide factors reproduce the intended covariance."""
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    sigma_along = np.array([0.5, 0.4])
    sigma_perp = np.array([0.3, 0.2])
    packed = forest._pack_spatial_cholesky_5d(
        forest._foliage_cholesky(dirs, sigma_along, sigma_perp)
    )
    assert packed.shape == (2, 15)
    for k in range(2):
        lower = np.zeros((5, 5))
        lower[np.tril_indices(5)] = packed[k]
        cov = lower @ lower.T
        # Stacked (season, growth) axes: near-zero isotropic, no cross terms.
        np.testing.assert_allclose(
            np.diag(cov)[:2], forest.STACKED_AXIS_SIGMA**2, rtol=1e-5
        )
        assert np.all(cov[:2, 2:] == 0.0)
        # Spatial block: principal sigmas are exactly (along, perp, perp).
        eigenvalues = np.linalg.eigvalsh(cov[2:, 2:])
        np.testing.assert_allclose(
            np.sqrt(eigenvalues.max()), sigma_along[k], atol=1e-6
        )
        np.testing.assert_allclose(np.sqrt(eigenvalues.min()), sigma_perp[k], atol=1e-6)


def test_forest_stagger_delays_development_but_converges() -> None:
    """A staggered tree lags through the middle slots yet ends ancient.

    A plain ``growth - offset`` clamp leaves offset trees permanently short
    of the final stage — most of the forest would never be ancient in the
    authored ancient poster slice.
    """
    last = forest.N_STAGES - 1
    for offset in (0, 1, 2):
        stages = [forest._effective_stage(g, offset) for g in range(forest.N_STAGES)]
        assert stages[0] == 0, (offset, stages)
        assert stages[-1] == last, f"offset {offset} never reaches ancient: {stages}"
        assert all(b >= a for a, b in zip(stages, stages[1:])), (offset, stages)
        if offset:
            assert sum(stages) < sum(range(forest.N_STAGES)), (
                f"offset {offset} does not actually delay development"
            )


def test_forest_palm_grows_foliage() -> None:
    """Palm fronds live at branch depth 1; the foliage gate must accept them.

    A ``max_depth >= 2`` gate silently leaves palms bare in every season —
    the composite layer test only proves SOME species produced gsplats.
    """
    palm_index = next(i for i, s in enumerate(forest.SPECIES) if s.key == "palm")
    palm = forest.SPECIES[palm_index]
    plan = forest.TreePlan(
        index=0,
        species_index=palm_index,
        x=0.0,
        y=0.0,
        z=0.0,
        seed=5,
        scale=1.0,
        rotation=0.0,
        stage_offset=0,
        tint_shift=np.zeros(3, dtype=np.float32),
        brightness=1.0,
        lsystem=palm.lsystem,
    )
    lines_out = forest.SpeciesArrays()
    foliage_out = forest.FoliageArrays()
    forest._build_tree(
        plan, lines_out, foliage_out, final_iterations=4 + palm.iter_bonus
    )
    assert foliage_out.n_splats > 0, "palm grew no foliage splats"
    assert all("Palm" in label for _, label in foliage_out.label_runs)


def _first_branch_direction(lsystem: forest.LSystem, iterations: int) -> np.ndarray:
    """Unit direction of the first depth-1 segment of a derivation."""
    v, e, ed, _ = forest.derive_tree(lsystem, iterations, seed=77)
    first = int(np.argmax(ed == 1))
    segment = v[int(e[first, 1])] - v[int(e[first, 0])]
    return segment / np.linalg.norm(segment)


#: The one shipped grammar whose crown ordinals are NOT depth-stable: the palm
#: axiom is ``TC`` and ``T -> F/T`` inserts a top-level roll ahead of the crown
#: on every derivation step. See :func:`demo_lsystem_forest._branch_jitter`.
_ORDINAL_UNSTABLE = {"palm"}


@pytest.mark.parametrize(
    "key", [s.key for s in forest.SPECIES if s.key not in _ORDINAL_UNSTABLE]
)
def test_lsystem_growth_stages_keep_existing_branch_orientations(key: str) -> None:
    """Re-deriving a tree one iteration deeper must not re-roll its angles.

    Growth stages are re-derivations at increasing depth. Jitter therefore
    cannot come from a sequential rng stream (the deeper expansion consumes
    a different number of draws, shifting every subsequent sample and
    visibly popping branches during the growth time-lapse); it is a pure
    function of each branch's bracket path. The first branch exists at
    every depth, so its opening direction must match exactly across stages.

    Parametrized over every species so the docstring's scope is CHECKED
    rather than asserted: the property depends on each grammar appending its
    recursion last, which is a per-grammar fact, not a property of
    :func:`demo_lsystem_forest._branch_jitter` alone.
    """
    lsystem = next(s for s in forest.SPECIES if s.key == key).lsystem
    d3 = _first_branch_direction(lsystem, 3)
    d4 = _first_branch_direction(lsystem, 4)
    d5 = _first_branch_direction(lsystem, 5)
    np.testing.assert_allclose(d3, d4, atol=1e-5)
    np.testing.assert_allclose(d4, d5, atol=1e-5)


def test_palm_crown_reroll_is_the_documented_exception() -> None:
    """The palm's crown DOES re-roll across depths — keep the doc honest.

    If a future grammar edit makes the palm depth-stable (e.g. bracketing the
    crown so it gets its own ordinal space), this test fails and the
    ``_branch_jitter`` caveat plus ``_ORDINAL_UNSTABLE`` must be retired
    rather than left as stale prose.
    """
    palm = next(s for s in forest.SPECIES if s.key == "palm").lsystem
    d4 = _first_branch_direction(palm, 4)
    d5 = _first_branch_direction(palm, 5)
    assert not np.allclose(d4, d5, atol=1e-5)


class _RecordingScene:
    """Small scene sink that records geometry submitted by demo builders."""

    def __init__(self) -> None:
        self.lines: dict[str, dict[str, Any]] = {}
        self.points: dict[str, dict[str, Any]] = {}

    def add_lines(self, name: str, **kwargs: Any) -> None:
        self.lines[name] = kwargs

    def add_points(self, name: str, **kwargs: Any) -> None:
        self.points[name] = kwargs


def _tracking_data() -> dict[str, Any]:
    positions = [(t, 10.0 + 0.1 * t, 20.0 + 0.2 * t, 30.0 + 0.3 * t) for t in range(8)]
    return {
        "tracks": {1: positions, 2: [(0, 1.0, 2.0, 3.0)]},
        "colors": {1: (0.8, 0.4, 0.2), 2: (0.2, 0.8, 0.4)},
    }


def test_celegans_cell_track_builder_skips_orphans_and_shares_joints() -> None:
    scene = _RecordingScene()
    celegans.add_cell_tracks(scene, _tracking_data(), np.zeros(3, dtype=np.float32))

    node = scene.lines["cell_tracks"]
    vertices = np.asarray(node["vertices"])
    edges = np.asarray(node["indices"])
    colors = np.asarray(node["colors"])
    _assert_indexed_geometry(vertices, edges, colors)
    assert len(vertices) == 8  # the one-position track is intentionally omitted


def test_celegans_fading_trail_chain_index_has_no_orphans() -> None:
    scene = _RecordingScene()
    celegans.add_fading_trail_tracks(
        scene,
        _tracking_data(),
        np.zeros(3, dtype=np.float32),
        n_timepoints=8,
    )

    node = scene.lines["cell_tracks_trail"]
    vertices = np.asarray(node["vertices"])
    edges = np.asarray(node["indices"])
    colors = np.asarray(node["colors"])
    _assert_indexed_geometry(vertices, edges, colors)
    assert "current_positions" in scene.points
