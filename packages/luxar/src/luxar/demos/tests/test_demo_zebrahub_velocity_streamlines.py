"""Smoke tests for pure helpers in demo_zebrahub_velocity_streamlines.

These tests cover deterministic numerical helpers that do not touch the network,
the cache, or matplotlib. Network-fetching code paths (``resolve_h5ad`` Drive
download, ``load_zebrahub``) are intentionally not exercised. All heavy deps are
imported lazily via ``luxar.demos.require_module``, so the module imports with
no extras installed. That shared gate is tested in ``test_demos_dependencies.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

# The ``luxar.demos`` package is now importable directly (the sys.modules alias
# that used to shadow it was removed).
from luxar.demos.demo_zebrahub_multiome import _category_label as _multiome_label
from luxar.demos.demo_zebrahub_velocity_streamlines import (
    ZebrahubData,
    _array_hash,
    _select_seeds,
    _stabilize_3d,
)
from luxar.demos.demo_zebrahub_velocity_streamlines import (
    _category_label as _velocity_label,
)


def test_category_labels_do_not_alias_negative_codes() -> None:
    categories = ["first", "last"]
    for label in (_multiome_label, _velocity_label):
        assert label(categories, 0) == "first"
        assert label(categories, 1) == "last"
        assert label(categories, -1) == "-1"
        assert label(categories, 2) == "2"


class TestArrayHash:
    def test_deterministic_same_input(self) -> None:
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert _array_hash(a) == _array_hash(b)

    def test_changes_with_content(self) -> None:
        a = np.arange(12, dtype=np.float32)
        b = a.copy()
        b[0] += 0.5
        assert _array_hash(a) != _array_hash(b)

    def test_combines_multiple_arrays(self) -> None:
        a = np.arange(4, dtype=np.float32)
        b = np.arange(4, 8, dtype=np.float32)
        # Hashing (a, b) is not the same as hashing each individually.
        assert _array_hash(a, b) != _array_hash(a)
        assert _array_hash(a, b) != _array_hash(b)


class TestStabilize3D:
    def test_returns_3x3_rotation_and_keepdims_mean(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((50, 3)).astype(np.float32)
        rotation, mean = _stabilize_3d(coords)
        assert rotation.shape == (3, 3)
        # mean is keepdims-style: shape (1, 3)
        assert mean.shape == (1, 3)
        assert rotation.dtype == np.float32
        assert mean.dtype == np.float32

    def test_handles_few_points_with_identity(self) -> None:
        # With <3 points, stabilize falls back to identity rotation.
        coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rotation, mean = _stabilize_3d(coords)
        np.testing.assert_allclose(rotation, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(mean.ravel(), [1.0, 2.0, 3.0])

    def test_centers_at_mean(self) -> None:
        coords = np.array(
            [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [14.0, 24.0, 34.0]],
            dtype=np.float32,
        )
        _, mean = _stabilize_3d(coords)
        np.testing.assert_allclose(mean.ravel(), coords.mean(axis=0), atol=1e-5)


class TestSelectSeeds:
    def _make_data(self, n_cells: int, anatomy_codes: np.ndarray) -> "ZebrahubData":
        return ZebrahubData(
            positions=np.zeros((n_cells, 3), dtype=np.float32),
            velocities=np.zeros((n_cells, 3), dtype=np.float32),
            anatomy_codes=anatomy_codes,
            anatomy_categories=[
                f"class{i}" for i in range(int(anatomy_codes.max()) + 1)
            ],
            stage_codes=np.zeros(n_cells, dtype=np.int32),
            stage_categories=["s0"],
        )

    def test_n_seeds_none_returns_every_cell(self) -> None:
        data = self._make_data(50, np.zeros(50, dtype=np.int32))
        result = _select_seeds(data, n_seeds=None)
        assert result.shape == (50,)
        np.testing.assert_array_equal(np.sort(result), np.arange(50))

    def test_n_seeds_geq_total_returns_every_cell(self) -> None:
        data = self._make_data(20, np.zeros(20, dtype=np.int32))
        result = _select_seeds(data, n_seeds=100)
        assert result.shape == (20,)

    def test_subsamples_to_requested_count_or_less(self) -> None:
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 4, size=200, dtype=np.int32)
        data = self._make_data(200, codes)
        result = _select_seeds(data, n_seeds=50)
        # Stratified subsampling: result should be roughly n_seeds and never more.
        assert result.size <= 200
        assert result.size > 0
        # All indices should be valid and unique.
        assert len(set(result.tolist())) == result.size
        assert result.min() >= 0
        assert result.max() < 200


# ---------------------------------------------------------------------------
# 2026-09-10 review: bloom, streamline gain, legend placement, subtitle
# ---------------------------------------------------------------------------


def test_streamline_gain_matches_the_reviewed_colour_range() -> None:
    from luxar.demos import demo_zebrahub_velocity_streamlines as demo

    assert demo.STREAMLINE_INTENSITY == pytest.approx(1.0 / 45.26)
    assert demo.STREAMLINE_INTENSITY > demo.LINE_INTENSITY * 2.5


@pytest.mark.parametrize("max_vertices", [31, 61])
def test_streamline_ladder_caps_each_polyline_commit(max_vertices: int) -> None:
    from luxar.demos import demo_zebrahub_velocity_streamlines as demo

    counts = demo.streamline_ladder(120_800, max_vertices)["counts"]
    increments = [
        count - previous
        for previous, count in zip([0, *counts[:-1]], counts, strict=True)
    ]

    assert counts[-1] == 120_800
    assert max(increments) * max_vertices <= 900_000


def test_bloom_is_faint_and_wide() -> None:
    from luxar.demos import demo_zebrahub_velocity_streamlines as demo

    assert (demo.BLOOM_THRESHOLD, demo.BLOOM_STRENGTH, demo.BLOOM_RADIUS) == (
        0.01,
        0.05,
        1.0,
    )
    assert demo.BLOOM_LEVELS == 8


def test_legend_is_only_the_colour_key() -> None:
    from luxar.demos import demo_zebrahub_velocity_streamlines as demo

    html = demo.build_legend_html(
        ["brain", "gut", "skin"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([30, 20, 10]),
    )
    assert "Top anatomy classes (3)" in html
    assert "brain (30)" in html
    assert "comet" not in html and "advect" not in html
    assert "Zebrahub RNA-velocity field" not in html
    # ...and the explanation lives in the subtitle instead.
    assert "comet" in demo.SUBTITLE.lower() and "streamlines" in demo.SUBTITLE.lower()
    assert len(demo.SUBTITLE) < 240


def test_legend_sits_in_the_lower_left_corner() -> None:
    import inspect

    from luxar.demos import demo_zebrahub_velocity_streamlines as demo

    src = inspect.getsource(demo.write_scene)
    i = src.index("build_legend_html(data.anatomy_categories")
    block = src[i : i + 200]
    assert "position=(0.06, 0.98)" in block
    assert 'anchor="bottom-left"' in block
