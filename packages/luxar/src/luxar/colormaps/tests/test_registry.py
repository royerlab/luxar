"""Tests for colormap registry and resolution."""

import numpy as np
import pytest

from luxar.colormaps import BUILTIN_COLORMAP_NAMES, resolve_colormap
from luxar.colormaps.builtins import get_builtin_lut


class TestBuiltinColormaps:
    """Tests for built-in colormap data."""

    def test_builtin_names_not_empty(self) -> None:
        assert len(BUILTIN_COLORMAP_NAMES) >= 15

    @pytest.mark.parametrize("name", BUILTIN_COLORMAP_NAMES)
    def test_builtin_shape_and_dtype(self, name: str) -> None:
        lut = get_builtin_lut(name)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    @pytest.mark.parametrize("name", BUILTIN_COLORMAP_NAMES)
    def test_builtin_returns_copy(self, name: str) -> None:
        lut1 = get_builtin_lut(name)
        lut2 = get_builtin_lut(name)
        assert lut1 is not lut2
        np.testing.assert_array_equal(lut1, lut2)

    def test_linear_ramp_starts_black(self) -> None:
        """Linear ramps should start at (0, 0, 0)."""
        for name in ["green", "magenta", "cyan", "red", "blue", "yellow", "gray"]:
            lut = get_builtin_lut(name)
            np.testing.assert_array_equal(lut[0], [0, 0, 0])

    def test_green_ends_green(self) -> None:
        lut = get_builtin_lut("green")
        assert lut[255, 0] == 0  # no red
        assert lut[255, 1] == 255  # full green
        assert lut[255, 2] == 0  # no blue

    def test_gray_ends_white(self) -> None:
        lut = get_builtin_lut("gray")
        np.testing.assert_array_equal(lut[255], [255, 255, 255])

    # [Python-R2/D-W3] Pin EXACT uint8 boundary rounding at both ends of
    # every linear-ramp builtin. The float→uint8 rounding (round-to-
    # nearest-even at 0.5 inputs) can shift boundary values by ±1 under
    # a refactor that swapped `round().astype(uint8)` for, say,
    # `astype(uint8)` (truncation) — visible as off-by-one color at
    # the ramp ends. The general-shape test above covers the centre;
    # this one nails down the endpoints.
    @pytest.mark.parametrize(
        "name", ["green", "magenta", "cyan", "red", "blue", "yellow"]
    )
    def test_linear_ramp_endpoints_exact_uint8(self, name: str) -> None:
        lut = get_builtin_lut(name)
        # Index 0 is the all-black anchor for every linear ramp.
        np.testing.assert_array_equal(lut[0], [0, 0, 0])
        # Index 255 must be saturated on the active channel(s); no
        # channel that started at 0 may have crept above 0 by rounding,
        # and the active channel(s) must hit exactly 255.
        expected_top = {
            "green": [0, 255, 0],
            "magenta": [255, 0, 255],
            "cyan": [0, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
        }[name]
        np.testing.assert_array_equal(lut[255], expected_top)

    def test_unknown_builtin_raises(self) -> None:
        with pytest.raises(KeyError):
            get_builtin_lut("nonexistent_colormap")


class TestResolveColormap:
    """Tests for resolve_colormap() function."""

    def test_resolve_builtin_by_name(self) -> None:
        lut = resolve_colormap("viridis")
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_resolve_all_builtins(self) -> None:
        for name in BUILTIN_COLORMAP_NAMES:
            lut = resolve_colormap(name)
            assert lut.shape == (256, 3)

    def test_resolve_custom_float_array(self) -> None:
        """Custom float32 array in [0, 1] range."""
        arr = np.linspace(0, 1, 256 * 3).reshape(256, 3).astype(np.float32)
        lut = resolve_colormap(arr)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_resolve_custom_uint8_array(self) -> None:
        """Custom uint8 array, exact 256 entries."""
        arr = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        lut = resolve_colormap(arr)
        assert lut.shape == (256, 3)
        np.testing.assert_array_equal(lut, arr)

    def test_resolve_custom_array_resampled(self) -> None:
        """Custom array with non-256 entries gets resampled."""
        arr = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        lut = resolve_colormap(arr)
        assert lut.shape == (256, 3)
        # First and last should match
        np.testing.assert_array_equal(lut[0], [0, 0, 0])
        np.testing.assert_array_equal(lut[255], [255, 255, 255])
        # Middle should be interpolated
        assert 120 <= lut[128, 0] <= 135  # ~127.5

    def test_resolve_float64_array(self) -> None:
        arr = np.linspace(0, 1, 256 * 3).reshape(256, 3).astype(np.float64)
        lut = resolve_colormap(arr)
        assert lut.dtype == np.uint8

    def test_reject_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            resolve_colormap(np.zeros((256, 4), dtype=np.uint8))

    def test_reject_1d_array(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            resolve_colormap(np.zeros(256, dtype=np.uint8))

    def test_reject_too_few_entries(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            resolve_colormap(np.zeros((1, 3), dtype=np.uint8))

    def test_reject_float_out_of_range(self) -> None:
        arr = np.array([[0, 0, 1.5], [1, 1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            resolve_colormap(arr)

    def test_reject_negative_float(self) -> None:
        arr = np.array([[-0.1, 0, 0], [1, 1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            resolve_colormap(arr)

    # [Python-R1/D-C2] NaN values would silently pass the `arr < 0` /
    # `arr > 1` checks (NaN comparisons are always False) and produce
    # garbage uint8 via `(NaN * 255).astype(uint8)`. The finiteness
    # guard must fire BEFORE the range check; pin both NaN and ±Inf.
    def test_reject_nan_float(self) -> None:
        arr = np.array([[0, 0, 0], [np.nan, 1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            resolve_colormap(arr)

    def test_reject_positive_infinity_float(self) -> None:
        arr = np.array([[0, 0, 0], [np.inf, 1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            resolve_colormap(arr)

    def test_reject_negative_infinity_float(self) -> None:
        arr = np.array([[0, 0, 0], [-np.inf, 1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            resolve_colormap(arr)

    # [Python-R5 / D-G2] Custom float array round-trip: the
    # `_resolve_array` pipeline does (float [0,1]) → (uint8 [0,255]) →
    # (resample to 256). For a known input we can predict the
    # resampled output within the documented ±1 uint8 rounding
    # tolerance. Pin both endpoint anchoring AND the linear-ramp
    # midpoint so a regression that swapped (round → floor) or
    # dropped a column would show up.
    def test_custom_float_array_resample_preserves_endpoints_and_midpoint(self) -> None:
        # Linear ramp from black → pure red in 100 steps, requested
        # at standard 256-entry LUT resolution.
        n_in = 100
        custom = np.zeros((n_in, 3), dtype=np.float32)
        custom[:, 0] = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        resolved = resolve_colormap(custom)
        assert resolved.shape == (256, 3)
        assert resolved.dtype == np.uint8

        # Endpoint anchors (lossless after float→uint8 rounding).
        np.testing.assert_array_equal(resolved[0], [0, 0, 0])
        np.testing.assert_array_equal(resolved[-1], [255, 0, 0])

        # Linear midpoint should be close to (128, 0, 0); allow ±2
        # uint8 units for the combined rounding + resampling drift.
        # Green / blue channels remain 0 (no interpolation crosstalk).
        assert abs(int(resolved[128, 0]) - 128) <= 2
        assert resolved[128, 1] == 0
        assert resolved[128, 2] == 0

        # Monotonicity on the active channel: a regression that
        # reversed the LUT direction would fail this on a single
        # comparison.
        red_channel = resolved[:, 0]
        assert (np.diff(red_channel.astype(int)) >= 0).all(), (
            "red channel must be non-decreasing along the ramp"
        )

    def test_reject_wrong_dtype(self) -> None:
        with pytest.raises(TypeError, match="uint8"):
            resolve_colormap(np.zeros((256, 3), dtype=np.int32))

    def test_reject_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="string or numpy"):
            resolve_colormap(42)  # type: ignore[arg-type]

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown colormap"):
            resolve_colormap("definitely_not_a_real_colormap_name_xyz")

    def test_matplotlib_fallback(self) -> None:
        """If matplotlib is available, resolve non-builtin names."""
        pytest.importorskip("matplotlib")
        lut = resolve_colormap("cividis")
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_colorcet_fallback(self) -> None:
        """If colorcet is available, resolve colorcet names."""
        pytest.importorskip("colorcet")
        lut = resolve_colormap("bgy")
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
