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
    vertices, colors, labels, edges = dipc._haplotype_geometry(polylines, 0)
    _assert_indexed_geometry(vertices, edges, colors, labels)
    _assert_edges_stay_within_chains(edges, [4, 3])


def test_lsystem_turtle_reuses_branch_hub() -> None:
    lsystem = forest.LSystem(
        axiom="F[+F]F",
        rules={},
        randomness=0.0,
    )
    vertices, widths, colors, sharpness, edges, _leaves = forest.create_tree(
        lsystem,
        iterations=0,
        add_leaves=False,
    )
    _assert_indexed_geometry(vertices, edges, widths, colors, sharpness)

    degree = np.bincount(edges.reshape(-1), minlength=len(vertices))
    assert int(degree.max()) == 3, "the pushed/popped branch point was duplicated"


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
