"""Round-trip + policy tests for ``geolog_perchannel_u8/u16`` (HDR colors).

The per-channel member of the geolog family: each column quantized on a
min/max-anchored TRUE-log grid (uniform relative precision across the
column's whole dynamic range), code 0 reserved for exact zeros. Chosen by
the 2026-07 HDR-color spike, where it dominated linear and log1p per-channel
at every measured dynamic range (2..12.6 decades, 6 datasets).
"""

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _hdr_colors(n=8000, decades=6.0, seed=0):
    rng = np.random.default_rng(seed)
    b = np.exp(rng.uniform(np.log(10.0 * 10.0**-decades), np.log(10.0), n))
    hue = 0.3 + 0.7 * rng.random((n, 3))
    colors = (b[:, None] * hue).astype(np.float32)
    colors[::37] = 0.0  # whole-splat exact zeros
    colors[5, 1] = 0.0  # lone zero entry
    return colors


def _encode_color(colors, mode):
    g = memory_group()
    ArrayEncoder().encode(
        data=colors,
        zarr_group=g,
        name="colors",
        semantic_type=SemanticType.COLOR,
        mode=mode,
        color_mode="hdr",
    )
    enc = dict(g["colors"].attrs["encoding"])
    decoded = np.asarray(ArrayDecoder().decode(g["colors"], g), dtype=np.float64)
    return decoded, enc, np.asarray(g["colors"])


class TestHdrColorPolicy:
    @pytest.mark.parametrize(
        "mode,name,dtype",
        [
            (EncodingMode.PRECISION, "float32", np.float32),
            (EncodingMode.AUTO, "geolog_perchannel_u16", np.uint16),
            (EncodingMode.MEMORY, "geolog_perchannel_u8", np.uint8),
        ],
    )
    def test_mode_dispatch(self, mode, name, dtype):
        colors = _hdr_colors()
        decoded, enc, stored = _encode_color(colors, mode)
        assert enc["name"] == name
        assert stored.dtype == dtype
        if mode == EncodingMode.PRECISION:
            np.testing.assert_array_equal(decoded, colors)

    def test_non_2d_hdr_input_falls_back_to_float32(self):
        # The per-channel encoding needs (N, C); a 1-D HDR color-ish array
        # (defensive path) must fall back to plain float32, not crash.
        flat = np.linspace(0.0, 10.0, 30, dtype=np.float32)
        g = memory_group()
        ArrayEncoder().encode(
            data=flat,
            zarr_group=g,
            name="colors",
            semantic_type=SemanticType.COLOR,
            mode=EncodingMode.AUTO,
            color_mode="hdr",
        )
        assert g["colors"].attrs["encoding"]["name"] == "float32"
        np.testing.assert_array_equal(
            np.asarray(ArrayDecoder().decode(g["colors"], g)), flat
        )

    def test_sdr_colors_unaffected(self):
        rng = np.random.default_rng(1)
        sdr = rng.random((500, 3)).astype(np.float32)
        g = memory_group()
        ArrayEncoder().encode(
            data=sdr,
            zarr_group=g,
            name="colors",
            semantic_type=SemanticType.COLOR,
            mode=EncodingMode.AUTO,
            color_mode="sdr",
        )
        assert g["colors"].attrs["encoding"]["name"] == "rgb_uint8"


class TestGeologPerchannelRoundtrip:
    def test_uniform_relative_precision_across_decades(self):
        # The load-bearing property: rel err must NOT grow toward the faint
        # end (this is exactly where linear and log1p failed in the spike).
        colors = _hdr_colors(decades=6.0)
        decoded, enc, _ = _encode_color(colors, EncodingMode.AUTO)
        pos = colors > 0
        rel = np.abs(decoded[pos] - colors[pos].astype(np.float64)) / colors[pos]
        assert np.percentile(rel, 95) < 3e-4  # ~ln-range/65534 per column
        mags = colors[pos].astype(np.float64)
        faint = mags <= mags.min() * 10.0
        assert np.percentile(rel[faint], 95) < 3e-4  # faint == overall

    def test_exact_zeros_roundtrip_and_no_positive_collapses(self):
        colors = _hdr_colors()
        decoded, enc, stored = _encode_color(colors, EncodingMode.AUTO)
        zero_mask = colors == 0.0
        np.testing.assert_array_equal(stored[zero_mask], 0)
        np.testing.assert_array_equal(decoded[zero_mask], 0.0)
        assert np.all(decoded[~zero_mask] > 0)  # reserved level: no collapse

    def test_per_column_anchors_are_log_domain_nonzero_minmax(self):
        colors = _hdr_colors()
        _, enc, _ = _encode_color(colors, EncodingMode.AUTO)
        for c in range(3):
            pos = colors[colors[:, c] > 0, c].astype(np.float64)
            assert enc["col_lo"][c] == pytest.approx(np.log(pos.min()), rel=1e-6)
            assert enc["col_hi"][c] == pytest.approx(np.log(pos.max()), rel=1e-6)
        assert enc["zero_level"] is True
        assert enc["bits"] == 16

    def test_hand_computed_decode(self):
        # Lock the exact inverse: 0 -> 0; u -> exp(lo + (u-1)/(2^b-2) * rng).
        colors = _hdr_colors(n=300)
        decoded, enc, stored = _encode_color(colors, EncodingMode.MEMORY)
        lo = np.asarray(enc["col_lo"])
        hi = np.asarray(enc["col_hi"])
        denom = (1 << enc["bits"]) - 2
        y = lo + (stored.astype(np.float64) - 1.0) / denom * np.maximum(hi - lo, 1e-30)
        hand = np.where(stored == 0, 0.0, np.exp(y))
        # The decode contract returns original_dtype (float32 here), so the
        # f64 hand computation must be compared after the same final cast.
        np.testing.assert_array_equal(
            decoded.astype(np.float32), hand.astype(np.float32)
        )

    def test_idempotent_reencode(self):
        colors = _hdr_colors()
        once, _, _ = _encode_color(colors, EncodingMode.AUTO)
        twice, _, _ = _encode_color(once.astype(np.float32), EncodingMode.AUTO)
        np.testing.assert_array_equal(twice, once)

    def test_constant_column(self):
        colors = np.full((100, 3), 2.5, np.float32)
        colors[:, 1] = 7.0
        g = memory_group()
        ArrayEncoder()._encode_geolog_perchannel(g, "c", colors, 16)
        dec = np.asarray(ArrayDecoder().decode(g["c"], g))
        np.testing.assert_allclose(dec[:, 0], 2.5, rtol=1e-6)
        np.testing.assert_allclose(dec[:, 1], 7.0, rtol=1e-6)

    def test_all_zero_column(self):
        colors = _hdr_colors(n=200)
        colors[:, 2] = 0.0
        decoded, enc, _ = _encode_color(colors, EncodingMode.AUTO)
        np.testing.assert_array_equal(decoded[:, 2], 0.0)
        assert enc["col_lo"][2] == 0.0 and enc["col_hi"][2] == 0.0

    def test_metadata_ignores_libm_ulp(self, monkeypatch):
        colors = np.array(
            [[1e-5, 0.2, 3.0], [2e-3, 4.0, 900.0], [0.3, 25.0, 1e5]],
            dtype=np.float32,
        )
        baseline_group = memory_group()
        ArrayEncoder()._encode_geolog_perchannel(baseline_group, "c", colors, 16)
        baseline = dict(baseline_group["c"].attrs["encoding"])
        baseline_codes = np.asarray(baseline_group["c"])
        original_log = np.log

        def perturbed_log(values):
            return np.nextafter(original_log(values), np.inf)

        monkeypatch.setattr(np, "log", perturbed_log)
        perturbed_group = memory_group()
        ArrayEncoder()._encode_geolog_perchannel(perturbed_group, "c", colors, 16)
        perturbed = dict(perturbed_group["c"].attrs["encoding"])

        assert perturbed == baseline
        np.testing.assert_array_equal(np.asarray(perturbed_group["c"]), baseline_codes)

    def test_scales_are_canonical_float64_values(self):
        colors = np.array([[1e-5, 0.2], [3.0, 1e5]], dtype=np.float32)
        lo, hi = ArrayEncoder._perchannel_geolog_scales(colors)

        assert lo.dtype == hi.dtype == np.float64
        np.testing.assert_array_equal(lo, lo.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(hi, hi.astype(np.float32).astype(np.float64))


class TestDecoderValidation:
    def _corrupt(self, patch):
        colors = _hdr_colors(n=100)
        g = memory_group()
        ArrayEncoder()._encode_geolog_perchannel(g, "c", colors, 8)
        attrs = dict(g["c"].attrs["encoding"])
        attrs.update(patch)
        g["c"].attrs["encoding"] = attrs
        return g

    def test_rejects_non_finite_scales(self):
        g = self._corrupt({"col_hi": [1.0, float("inf"), 1.0]})
        with pytest.raises(ValueError, match="finite"):
            ArrayDecoder().decode(g["c"], g)

    def test_rejects_length_mismatch(self):
        g = self._corrupt({"col_lo": [0.0], "col_hi": [1.0]})
        with pytest.raises(ValueError, match="per-column scales"):
            ArrayDecoder().decode(g["c"], g)

    def test_rejects_missing_scales(self):
        colors = _hdr_colors(n=50)
        g = memory_group()
        ArrayEncoder()._encode_geolog_perchannel(g, "c", colors, 8)
        attrs = dict(g["c"].attrs["encoding"])
        del attrs["col_lo"]
        g["c"].attrs["encoding"] = attrs
        with pytest.raises(ValueError, match="col_lo"):
            ArrayDecoder().decode(g["c"], g)
