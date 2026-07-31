"""Lines authoring lint and indexed-layout boundary coverage.

The viewer's joint continuity matches shared vertex indices, not equal endpoint
coordinates. The writer warns only for strong forward-chain evidence, avoiding
false positives for common graph edge orderings, and warns once per logical node.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler

WARNING_MARKER = "exploded into independent segments"


def _explode(points: np.ndarray) -> np.ndarray:
    """Duplicate interior vertices: ``(N, 3)`` polyline → endpoint pairs."""
    n_segments = len(points) - 1
    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]
    vertices[1::2] = points[1:]
    return vertices


def _edge_vertices(edges: list[tuple[int, int]]) -> np.ndarray:
    """Expand integer graph edges into independent coordinate pairs."""
    return np.asarray(
        [[[float(a), 0.0, 0.0], [float(b), 0.0, 0.0]] for a, b in edges],
        dtype=np.float32,
    ).reshape(-1, 3)


def _write_lines(tmp_path, capsys, *, name: str = "lines", **kwargs) -> str:
    with LuxarZarrCompiler(tmp_path / "lint.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(name, widths=0.5, **kwargs)
    return capsys.readouterr().out


def _chain_points(n: int) -> np.ndarray:
    return np.column_stack(
        [np.arange(n, dtype=np.float32), np.zeros((n, 2), dtype=np.float32)]
    )


def test_exploded_continuous_segments_warns_after_named_header(
    tmp_path, capsys
) -> None:
    out = _write_lines(
        tmp_path,
        capsys,
        name="tracks",
        vertices=_explode(_chain_points(40)),
        line_type="segments",
    )
    assert WARNING_MARKER in out
    assert "Node 'tracks'" in out
    assert out.index("Writing 78 line vertices") < out.index(WARNING_MARKER)


def test_disconnected_segments_stay_silent(tmp_path, capsys) -> None:
    vertices = np.random.default_rng(1).uniform(-10, 10, (80, 3)).astype(np.float32)
    out = _write_lines(tmp_path, capsys, vertices=vertices, line_type="segments")
    assert WARNING_MARKER not in out


def test_dfs_ordered_graph_edges_stay_silent(tmp_path, capsys) -> None:
    # Eight of eleven adjacent edge pairs share end→start coordinates: the old
    # >50% heuristic warned even though branch traversals are graph edge soup.
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 4),
        (4, 5),
        (5, 6),
        (1, 7),
        (7, 8),
        (8, 9),
        (1, 10),
        (10, 11),
        (11, 12),
    ]
    out = _write_lines(
        tmp_path,
        capsys,
        vertices=_edge_vertices(edges),
        line_type="segments",
    )
    assert WARNING_MARKER not in out


def test_symmetric_directed_edges_stay_silent(tmp_path, capsys) -> None:
    edges = [(0, 1), (1, 0)] * 8
    out = _write_lines(
        tmp_path,
        capsys,
        vertices=_edge_vertices(edges),
        line_type="segments",
    )
    assert WARNING_MARKER not in out


@pytest.mark.parametrize("n_points,should_warn", [(8, False), (9, True)])
def test_lint_vertex_floor_is_pinned(
    tmp_path, capsys, n_points: int, should_warn: bool
) -> None:
    out = _write_lines(
        tmp_path,
        capsys,
        vertices=_explode(_chain_points(n_points)),
        line_type="segments",
    )
    assert (WARNING_MARKER in out) is should_warn


def test_exactly_ninety_percent_forward_adjacency_stays_silent(
    tmp_path, capsys
) -> None:
    vertices = _explode(_chain_points(12))  # 11 segments → 10 adjacencies
    vertices[2] = np.array([100.0, 0.0, 0.0], dtype=np.float32)  # break one
    out = _write_lines(tmp_path, capsys, vertices=vertices, line_type="segments")
    assert WARNING_MARKER not in out


def test_partitioned_segments_warn_once_for_logical_node(tmp_path, capsys) -> None:
    out = _write_lines(
        tmp_path,
        capsys,
        name="partitioned",
        vertices=_explode(_chain_points(41)),
        line_type="segments",
        partition={"max_elements": 32},
    )
    assert out.count(WARNING_MARKER) == 1
    assert "Node 'partitioned'" in out
    assert "Node 'partitioned/part_" not in out


def test_indexed_authoring_stays_silent(tmp_path, capsys) -> None:
    points = _chain_points(40)
    idx = np.arange(len(points) - 1, dtype=np.uint32)
    out = _write_lines(
        tmp_path,
        capsys,
        vertices=points,
        indices=np.column_stack([idx, idx + 1]),
        line_type="indexed",
    )
    assert WARNING_MARKER not in out


def test_indexed_accepts_odd_edge_count_pairs(tmp_path, capsys) -> None:
    points = _chain_points(4)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    out = _write_lines(
        tmp_path, capsys, vertices=points, indices=edges, line_type="indexed"
    )
    assert "Converted indexed to 3" in out


@pytest.mark.parametrize(
    "indices,error_pattern",
    [
        (np.array([0], dtype=np.uint32), "at least 2 indices"),
        (np.array([0, 1, 2], dtype=np.uint32), "even element count"),
    ],
)
def test_indexed_rejects_incomplete_flat_pairs(
    tmp_path, indices: np.ndarray, error_pattern: str
) -> None:
    with LuxarZarrCompiler(tmp_path / "invalid.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match=error_pattern):
            scene.add_lines(
                "lines",
                vertices=_chain_points(4),
                widths=0.5,
                indices=indices,
                line_type="indexed",
            )
