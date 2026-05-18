"""Tests for luxar.utils.fields helpers."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.utils.fields import (
    FlowField,
    add_reference_cube_to_scene,
    cubic_bounds,
    rk4_step,
    trilinear_vector,
    unit_flow,
)


def _make_constant_field(value: tuple[float, float, float], n: int = 5) -> FlowField:
    """Field with the same vector at every grid point, on [0, n-1]^3."""
    vectors = np.broadcast_to(np.array(value, dtype=np.float32), (n, n, n, 3)).copy()
    return FlowField(
        vectors=vectors,
        grid_min=np.zeros(3, dtype=np.float32),
        grid_max=np.full(3, float(n - 1), dtype=np.float32),
        spacing=1.0,
        cache_key="test-constant",
    )


class TestCubicBounds:
    def test_symmetric_around_centroid(self) -> None:
        coords = np.array([[-1, -2, -3], [3, 4, 5]], dtype=np.float32)
        lo, hi = cubic_bounds(coords, pad_fraction=0.0)
        center = (lo + hi) * 0.5
        np.testing.assert_allclose(center, [1.0, 1.0, 1.0], atol=1e-6)

    def test_cube_uses_longest_side(self) -> None:
        coords = np.array([[0, 0, 0], [10, 1, 1]], dtype=np.float32)
        lo, hi = cubic_bounds(coords, pad_fraction=0.0)
        sides = hi - lo
        assert np.allclose(sides, sides[0])

    def test_pad_fraction_applied(self) -> None:
        coords = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        lo_no_pad, hi_no_pad = cubic_bounds(coords, pad_fraction=0.0)
        lo_padded, hi_padded = cubic_bounds(coords, pad_fraction=0.1)
        # Padded cube should be larger by ~20% (10% pad on each side).
        assert (hi_padded[0] - lo_padded[0]) > (hi_no_pad[0] - lo_no_pad[0])

    def test_min_half_side_clamped(self) -> None:
        coords = np.array([[0.5, 0.5, 0.5], [0.51, 0.51, 0.51]], dtype=np.float32)
        lo, hi = cubic_bounds(coords, pad_fraction=0.0)
        # Half-side floor is 1.0, so total side >= 2.0
        assert hi[0] - lo[0] >= 2.0


class TestTrilinearVector:
    def test_grid_point_returns_exact_value(self) -> None:
        field = _make_constant_field((2.0, 3.0, -1.0))
        # Sample exactly on a grid vertex (well inside, away from upper edge).
        points = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        result = trilinear_vector(field, points)
        np.testing.assert_allclose(result, [[2.0, 3.0, -1.0], [2.0, 3.0, -1.0]])

    def test_off_grid_constant_field_returns_constant(self) -> None:
        field = _make_constant_field((1.0, -2.0, 0.5))
        points = np.array([[0.5, 1.7, 2.3]], dtype=np.float32)
        result = trilinear_vector(field, points)
        np.testing.assert_allclose(result, [[1.0, -2.0, 0.5]], atol=1e-6)

    def test_out_of_bounds_returns_nan(self) -> None:
        field = _make_constant_field((1.0, 0.0, 0.0))
        points = np.array(
            [
                [-0.1, 0.0, 0.0],  # below grid_min
                [10.0, 0.0, 0.0],  # above grid_max
                [0.0, 4.5, 0.0],  # within grid (n=5, so [0..4]; valid up to 3)
            ],
            dtype=np.float32,
        )
        result = trilinear_vector(field, points)
        assert np.all(np.isnan(result[0]))
        assert np.all(np.isnan(result[1]))
        assert np.all(np.isnan(result[2]))  # 4.5 is past last interior cell

    def test_empty_input(self) -> None:
        field = _make_constant_field((1.0, 0.0, 0.0))
        result = trilinear_vector(field, np.zeros((0, 3), dtype=np.float32))
        assert result.shape == (0, 3)


class TestUnitFlow:
    def test_constant_field_normalized(self) -> None:
        field = _make_constant_field((3.0, 0.0, 4.0))  # magnitude = 5
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        result = unit_flow(field, points)
        np.testing.assert_allclose(result, [[0.6, 0.0, 0.8]], atol=1e-6)

    def test_zero_field_returns_nan(self) -> None:
        field = _make_constant_field((0.0, 0.0, 0.0))
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        result = unit_flow(field, points)
        assert np.all(np.isnan(result))


class TestRk4Step:
    def test_advect_constant_unit_field(self) -> None:
        # Unit field along +x, integrate by step_size = 0.5
        field = _make_constant_field((1.0, 0.0, 0.0))
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        out = rk4_step(points, step_size=0.5, field=field)
        np.testing.assert_allclose(out, [[1.5, 1.0, 1.0], [2.5, 2.0, 2.0]], atol=1e-5)

    def test_out_of_bounds_seed_propagates_nan(self) -> None:
        field = _make_constant_field((1.0, 0.0, 0.0))
        # Below grid_min, so unit_flow(k1) returns NaN.
        points = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        out = rk4_step(points, step_size=0.5, field=field)
        assert np.all(np.isnan(out))

    def test_zero_field_seed_propagates_nan(self) -> None:
        field = _make_constant_field((0.0, 0.0, 0.0))
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        out = rk4_step(points, step_size=0.5, field=field)
        assert np.all(np.isnan(out))


class TestAddReferenceCube:
    def test_calls_scene_add_lines_with_12_edges(self) -> None:
        calls: list[dict] = []

        class _FakeScene:
            def add_lines(self, name, **kwargs):  # type: ignore[no-untyped-def]
                calls.append({"name": name, **kwargs})

        scene = _FakeScene()
        grid_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        grid_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        add_reference_cube_to_scene(scene, grid_min, grid_max, name="Test cube")

        assert len(calls) == 1
        call = calls[0]
        assert call["name"] == "Test cube"
        assert call["line_type"] == "indexed"
        assert call["vertices"].shape == (8, 3)
        # 12 edges × 2 indices/edge = 24 entries
        assert call["indices"].shape == (24,)
        assert call["visible"] is False

    @pytest.mark.parametrize(
        "kw,value",
        [
            ("widths", 0.01),
            ("opacity", 0.5),
            ("intensity", 1.0),
            ("color", (1.0, 0.0, 0.0)),
            ("sharpness", 0.5),
            ("blending_mode", "alpha"),
            ("layer", False),
            ("visible", True),
        ],
    )
    def test_kwargs_passed_through(self, kw: str, value: object) -> None:
        captured: dict = {}

        class _FakeScene:
            def add_lines(self, name, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)

        scene = _FakeScene()
        add_reference_cube_to_scene(
            scene,
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            **{kw: value},  # type: ignore[arg-type]
        )

        # Note that "color" is forwarded as the colors keyword.
        if kw == "color":
            assert captured["colors"] == value
        else:
            assert captured[kw] == value
