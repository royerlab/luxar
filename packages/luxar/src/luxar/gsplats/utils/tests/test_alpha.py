"""Tests for per-splat opacity (color alpha) conversions.

See :mod:`luxar.gsplats.utils.alpha` and VOLUMETRIC_BLENDING_SPEC.md §5.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.alpha import (
    ALPHA_CLAMP,
    alpha_to_optical_depth,
    apply_alpha_to_amplitudes,
    effective_amplitudes,
    optical_depth_to_alpha,
)


class TestAlphaOpticalDepthRoundTrip:
    def test_round_trip_interior(self) -> None:
        # a → w → a is identity well inside the clamp.
        a = np.array([0.0, 0.1, 0.5, 0.9, 0.99], dtype=np.float64)
        back = optical_depth_to_alpha(alpha_to_optical_depth(a))
        assert np.allclose(back, a, atol=1e-6)

    def test_zero_alpha_is_zero_depth(self) -> None:
        assert alpha_to_optical_depth(np.array([0.0]))[0] == 0.0

    def test_clamp_bounds_depth(self) -> None:
        # a = 1 would be infinite depth; the clamp keeps it finite and equal
        # to the shader's ALPHA_CLAMP (kept in lockstep with the GLSL/TSL
        # 0.998046875 literal = 1 − 1/512).
        assert ALPHA_CLAMP == pytest.approx(1.0 - 1.0 / 512.0)
        w = alpha_to_optical_depth(np.array([1.0, 2.0]))
        assert np.all(np.isfinite(w))
        assert w[0] == pytest.approx(-np.log1p(-ALPHA_CLAMP))

    def test_dilute_limit_matches_identity(self) -> None:
        # w ≈ a as a → 0 (why volumetric and additive agree in the dilute
        # limit — the κ→0 coherence carried to per-splat alpha).
        a = np.array([1e-4, 1e-3], dtype=np.float64)
        assert np.allclose(alpha_to_optical_depth(a), a, rtol=1e-2)


class TestEffectiveAmplitudes:
    def _leaf(self, amps: np.ndarray, colors: np.ndarray | None) -> GSplatData:
        n = amps.shape[0]
        return GSplatData(
            centers=np.zeros((n, 3), dtype=np.float32),
            amplitudes=amps.astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
            ),
            colors=colors,
        )

    def test_rgb_returns_raw_amplitudes(self) -> None:
        amps = np.array([0.5, 2.0, 3.0])
        data = self._leaf(amps, np.ones((3, 3), dtype=np.float32))
        assert np.allclose(effective_amplitudes(data), amps)

    def test_no_colors_returns_raw_amplitudes(self) -> None:
        amps = np.array([0.5, 2.0, 3.0])
        data = self._leaf(amps, None)
        assert np.allclose(effective_amplitudes(data), amps)

    def test_rgba_scales_by_alpha(self) -> None:
        amps = np.array([1.0, 1.0, 1.0])
        colors = np.array(
            [[1, 1, 1, 0.25], [1, 1, 1, 0.5], [1, 1, 1, 1.0]], dtype=np.float32
        )
        data = self._leaf(amps, colors)
        assert np.allclose(effective_amplitudes(data), [0.25, 0.5, 1.0])

    def test_imported_style_ordering(self) -> None:
        # Imported classical splats have amplitude 1 and per-splat opacity in
        # alpha — a raw-amplitude ranking would call them all equal; the
        # effective ranking recovers the opacity order.
        amps = np.ones(3)
        colors = np.array(
            [[1, 1, 1, 0.9], [1, 1, 1, 0.1], [1, 1, 1, 0.5]], dtype=np.float32
        )
        data = self._leaf(amps, colors)
        order = np.argsort(-effective_amplitudes(data))
        assert list(order) == [0, 2, 1]

    def test_integer_rgba_alpha_is_normalized_to_unit_range(self) -> None:
        # uint8 colors are a valid SDR storage form (see _merge_lod_colors):
        # opacity is stored at full scale (255 = opaque). The [0, 1] opacity
        # contract requires normalizing before weighting — pre-fix, a uint8
        # RGBA leaf misranked every splat by ~255x against float data.
        amps = np.ones(3)
        colors = np.array(
            [[255, 255, 255, 255], [255, 255, 255, 128], [255, 255, 255, 0]],
            dtype=np.uint8,
        )
        data = self._leaf(amps, colors)
        weighted = effective_amplitudes(data)
        assert np.allclose(weighted, [1.0, 128 / 255, 0.0], atol=1e-6)

    def test_uint16_rgba_alpha_full_scale_is_opaque(self) -> None:
        amps = np.array([2.0])
        colors = np.array([[65535, 0, 0, 65535]], dtype=np.uint16)
        data = self._leaf(amps, colors)
        assert np.allclose(effective_amplitudes(data), [2.0], atol=1e-6)


class TestApplyAlphaToAmplitudes:
    """Direct tests on the shared helper both the object path
    (``effective_amplitudes``) and the annotate store path delegate to — the
    single source of truth for the ``A·α`` convention (guards the extracted
    logic against silent drift)."""

    def test_none_colors_is_identity(self) -> None:
        amps = np.array([0.5, 2.0, 3.0])
        assert np.allclose(apply_alpha_to_amplitudes(amps, None), amps)

    def test_rgb_colors_is_identity(self) -> None:
        amps = np.array([0.5, 2.0, 3.0])
        colors = np.ones((3, 3), dtype=np.float32)
        assert np.allclose(apply_alpha_to_amplitudes(amps, colors), amps)

    def test_float_rgba_scales_by_alpha(self) -> None:
        amps = np.ones(3)
        colors = np.array(
            [[1, 1, 1, 0.25], [1, 1, 1, 0.5], [1, 1, 1, 1.0]], dtype=np.float32
        )
        assert np.allclose(apply_alpha_to_amplitudes(amps, colors), [0.25, 0.5, 1.0])

    def test_uint8_rgba_alpha_normalized_by_255(self) -> None:
        # The integer branch no current object-path fixture reaches through the
        # store: alpha byte 128 → ~0.502 factor, 255 → opaque, 0 → transparent.
        amps = np.ones(3)
        colors = np.array(
            [[255, 255, 255, 255], [255, 255, 255, 128], [255, 255, 255, 0]],
            dtype=np.uint8,
        )
        assert np.allclose(
            apply_alpha_to_amplitudes(amps, colors),
            [1.0, 128 / 255, 0.0],
            atol=1e-6,
        )

    def test_uint16_rgba_alpha_normalized_by_65535(self) -> None:
        amps = np.array([2.0, 2.0])
        colors = np.array([[0, 0, 0, 65535], [0, 0, 0, 0]], dtype=np.uint16)
        assert np.allclose(
            apply_alpha_to_amplitudes(amps, colors), [2.0, 0.0], atol=1e-6
        )
