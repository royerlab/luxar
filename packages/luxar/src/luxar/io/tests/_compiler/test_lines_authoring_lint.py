"""Authoring lint for lines: warn when ``line_type="segments"`` input is a
chain of exploded continuous polylines (duplicated joint coordinates).

The viewer's joint continuity (endpoint-cap suppression) matches joints by
shared vertex INDEX, so exploded authoring renders every interior joint of a
thick line as a dark bead. The writer warns — never rejects — and stays
silent for genuinely disconnected segment soup and for indexed authoring.
"""

from __future__ import annotations

import numpy as np

from luxar import Dimensions, LuxarZarrCompiler

WARNING_MARKER = "exploded into independent segments"


def _explode(points: np.ndarray) -> np.ndarray:
    """Duplicate interior vertices: (N, 3) polyline -> (2(N-1), 3) pairs."""
    n_segments = len(points) - 1
    vertices = np.zeros((n_segments * 2, 3), dtype=np.float32)
    vertices[0::2] = points[:-1]
    vertices[1::2] = points[1:]
    return vertices


def _write_lines(tmp_path, capsys, **kwargs) -> str:
    with LuxarZarrCompiler(tmp_path / "lint.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines("lines", widths=0.5, **kwargs)
    return capsys.readouterr().out


def test_exploded_continuous_segments_warns(tmp_path, capsys) -> None:
    points = np.cumsum(
        np.random.default_rng(0).uniform(0.1, 1.0, (40, 3)).astype(np.float32),
        axis=0,
    )
    out = _write_lines(
        tmp_path, capsys, vertices=_explode(points), line_type="segments"
    )
    assert WARNING_MARKER in out


def test_disconnected_segments_stay_silent(tmp_path, capsys) -> None:
    # Unrelated endpoint pairs (graph-edge soup): no shared coordinates.
    vertices = np.random.default_rng(1).uniform(-10, 10, (80, 3)).astype(np.float32)
    out = _write_lines(tmp_path, capsys, vertices=vertices, line_type="segments")
    assert WARNING_MARKER not in out


def test_indexed_authoring_stays_silent(tmp_path, capsys) -> None:
    points = np.cumsum(
        np.random.default_rng(2).uniform(0.1, 1.0, (40, 3)).astype(np.float32),
        axis=0,
    )
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
    # (E, 2) edge arrays with an ODD number of edges are valid: the
    # validator must count ELEMENTS (indices.size), not rows — len() on
    # an (E, 2) array wrongly rejected E = 3 as "odd length".
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)  # 3 edges
    out = _write_lines(
        tmp_path, capsys, vertices=points, indices=edges, line_type="indexed"
    )
    assert "Converted indexed to 3" in out


def test_tiny_segment_counts_stay_silent(tmp_path, capsys) -> None:
    # Below the 16-vertex floor the lint never fires — a handful of
    # touching segments is not evidence of exploded authoring.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
        dtype=np.float32,
    )
    out = _write_lines(
        tmp_path, capsys, vertices=_explode(points), line_type="segments"
    )
    assert WARNING_MARKER not in out
