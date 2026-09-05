"""Tests for the gsplats <-> tracksdata interop adapter.

The pure-NumPy mask/bbox kernel is fully tested here (runs in CI). The
graph-building path requires the optional ``tracksdata`` extra and is guarded
with ``importorskip`` — it runs only where that extra is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats import GSplatData
from luxar.gsplats.interop import gsplats_to_tracksdata_graph, splat_mask_and_bbox


def _identity_L(scale: float, d: int = 2) -> np.ndarray:
    return np.eye(d, dtype=np.float64) * scale


# --------------------------------------------------------------------------- #
# splat_mask_and_bbox — pure NumPy, no optional deps
# --------------------------------------------------------------------------- #


def test_bbox_is_analytic_and_clamped() -> None:
    # Isotropic splat: half-extent = n_sigma * scale on every axis.
    bbox, mask = splat_mask_and_bbox(
        np.array([32.0, 32.0]), _identity_L(5.0), (64, 64), n_sigma=2.0
    )
    start, stop = bbox[:2], bbox[2:]
    # half-extent = 2 * 5 = 10 -> [22, 22) .. [43, 43)
    assert start.tolist() == [22, 22]
    assert stop.tolist() == [43, 43]
    assert mask.shape == tuple(stop - start)


def test_mask_center_inside_corners_outside() -> None:
    center = np.array([32.0, 32.0])
    bbox, mask = splat_mask_and_bbox(center, _identity_L(5.0), (64, 64), n_sigma=2.0)
    start = bbox[:2]
    # the voxel containing the center is inside the support
    ci = tuple(int(round(c)) - int(s) for c, s in zip(center, start))
    assert mask[ci]
    # the local-box corners (far from center) are outside
    assert not mask[0, 0]
    assert not mask[-1, -1]


def test_mask_area_matches_sigma_disk() -> None:
    # For L = scale*I, the n_sigma support is a disk of radius n_sigma*scale,
    # so the mask voxel count should be close to pi * r^2.
    scale, n_sigma = 5.0, 2.0
    _, mask = splat_mask_and_bbox(
        np.array([40.0, 40.0]), _identity_L(scale), (96, 96), n_sigma=n_sigma
    )
    expected = np.pi * (n_sigma * scale) ** 2
    assert mask.sum() == pytest.approx(expected, rel=0.12)


def test_anisotropic_orientation() -> None:
    # A splat elongated along axis-0 should have a taller-than-wide bbox.
    L = np.array([[8.0, 0.0], [0.0, 2.0]])
    bbox, _ = splat_mask_and_bbox(np.array([48.0, 48.0]), L, (96, 96), n_sigma=2.0)
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    assert height > width


def test_splat_outside_frame_returns_empty_mask() -> None:
    # Center far outside the frame -> clamped, degenerate (empty) box.
    bbox, mask = splat_mask_and_bbox(
        np.array([-100.0, -100.0]), _identity_L(3.0), (32, 32), n_sigma=2.0
    )
    assert mask.size == 0
    assert mask.dtype == bool


def test_mismatched_dims_raise() -> None:
    with pytest.raises(ValueError):
        splat_mask_and_bbox(np.array([1.0, 2.0]), _identity_L(1.0, 2), (10, 10, 10))


# --------------------------------------------------------------------------- #
# gsplats_to_tracksdata_graph — requires the optional `tracksdata` extra
# --------------------------------------------------------------------------- #


def _make_2d_gsplats(n: int = 6, seed: int = 0) -> GSplatData:
    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=(rng.rand(n, 2) * 100).astype(np.float32),
        amplitudes=rng.rand(n).astype(np.float32),
        # packed lower-triangular [L00, L10, L11] -> identity per splat
        cholesky_factors=np.tile(np.array([3, 0, 3], dtype=np.float32), (n, 1)),
    )


def test_gsplats_to_tracksdata_graph() -> None:
    pytest.importorskip("tracksdata")
    gsplats = _make_2d_gsplats(n=6)
    graph = gsplats_to_tracksdata_graph(gsplats, frame_shape=(128, 128), t=0)

    assert graph.num_nodes() == 6
    keys = set(graph.node_attr_keys())
    assert {"amplitude", "y", "x"} <= keys
    assert "t" in keys


def test_gsplats_to_tracksdata_graph_dim_mismatch() -> None:
    pytest.importorskip("tracksdata")
    gsplats = _make_2d_gsplats(n=2)
    with pytest.raises(ValueError):
        gsplats_to_tracksdata_graph(gsplats, frame_shape=(64, 64, 64))


def test_gsplats_to_tracksdata_uses_effective_amplitude_for_rgba() -> None:
    # Imported classical splats carry amplitude = 1 with per-splat opacity in
    # the color alpha channel. The exported `amplitude` attribute (and the
    # paint-order sort) must rank by rendered energy A·α, not the constant
    # raw amplitude — otherwise the sort is a no-op for imported data.
    pytest.importorskip("tracksdata")
    n = 5
    alpha = np.array([0.9, 0.1, 0.5, 0.99, 0.2], dtype=np.float32)
    gsplats = GSplatData(
        centers=(np.random.RandomState(3).rand(n, 2) * 100).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),  # imported-style
        cholesky_factors=np.tile(np.array([3, 0, 3], dtype=np.float32), (n, 1)),
        colors=np.concatenate(
            [np.full((n, 3), 0.5, dtype=np.float32), alpha[:, None]], axis=1
        ),
    )
    graph = gsplats_to_tracksdata_graph(gsplats, frame_shape=(128, 128), t=0)
    amps = sorted(graph.node_attrs(attr_keys=["amplitude"])["amplitude"].to_list())
    # The exported amplitudes are A·α = α here (A = 1): the full alpha spread,
    # NOT a constant 1 (which the pre-fix raw-amplitude export would give).
    assert amps == pytest.approx(sorted(alpha.tolist()), abs=1e-6)
    assert max(amps) < 1.0  # not the constant raw amplitude


def test_gsplats_to_tracksdata_graph_accumulates_timepoints() -> None:
    pytest.importorskip("tracksdata")
    g = gsplats_to_tracksdata_graph(_make_2d_gsplats(3, seed=1), (128, 128), t=0)
    g = gsplats_to_tracksdata_graph(
        _make_2d_gsplats(4, seed=2), (128, 128), t=1, graph=g
    )
    assert g.num_nodes() == 7
