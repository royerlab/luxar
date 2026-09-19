"""Tests for the geometric-log scalar encoding (``geolog_scalar_uint8/uint16``).

The rescale-first encoding for wide-dynamic-range positive scalars: the grid
is anchored to the array's own nonzero ``[min, max]`` in true log space
(uniform RELATIVE precision), and level 0 is RESERVED for exact zeros — no
nonzero input can decode to zero, by construction.
"""

import numpy as np
import pytest

from luxar._zarr_compat import memory_group
from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _roundtrip(data, mode=EncodingMode.AUTO, **kw):
    enc = ArrayEncoder()
    g = memory_group()
    enc.encode(
        data=data,
        zarr_group=g,
        name="a",
        semantic_type=SemanticType.POSITIVE_SCALAR,
        mode=mode,
        **kw,
    )
    return ArrayDecoder().decode(g["a"], g), dict(g["a"].attrs["encoding"]), g


def _wide(n=50_000, lo=5e-4, hi=2e4, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)


class TestGeologRoundTrip:
    def test_auto_accepts_explicit_uint8_tier(self):
        data = np.geomspace(1.0, 1000.0, 10_000).astype(np.float32)
        _, default_enc, _ = _roundtrip(data)
        decoded, enc, group = _roundtrip(data, positive_scalar_bits=8)

        assert default_enc["name"] == "bounded_scalar_uint16"
        assert enc["name"] == "geolog_scalar_uint8"
        assert group["a"].dtype == np.uint8
        assert np.all(decoded > 0)

    def test_explicit_tier_does_not_weaken_precision_mode(self):
        data = _wide(n=1000)
        decoded, enc, group = _roundtrip(
            data,
            mode=EncodingMode.PRECISION,
            positive_scalar_bits=8,
        )

        assert enc["name"] == "float32"
        assert group["a"].dtype == np.float32
        np.testing.assert_array_equal(decoded, data)

    def test_rejects_invalid_explicit_tier_before_broadcasting(self):
        with pytest.raises(ValueError, match="positive_scalar_bits must be 8 or 16"):
            _roundtrip(
                np.ones(10, dtype=np.float32),
                positive_scalar_bits=12,
            )

    @pytest.mark.parametrize(
        "mode,name,bound",
        [
            (EncodingMode.AUTO, "geolog_scalar_uint16", 2e-4),
            (EncodingMode.MEMORY, "geolog_scalar_uint8", 4e-2),
        ],
    )
    def test_uniform_relative_error_across_seven_decades(self, mode, name, bound):
        # The signature of rescale-first log quantization: max relative error
        # ~= p95 ~= the analytic step bound, UNIFORM across the whole range.
        data = _wide()
        decoded, enc, _ = _roundtrip(data, mode)
        assert enc["name"] == name
        rel = np.abs(decoded - data) / data
        assert rel.max() < bound
        # uniformity: the worst decade is no worse than 3x the best decade
        decades = np.floor(np.log10(data))
        per_decade = [
            rel[decades == d].max()
            for d in np.unique(decades)
            if (decades == d).sum() > 100
        ]
        assert max(per_decade) < 3 * min(per_decade)

    def test_zeros_roundtrip_exactly_and_nonzero_never_zeroes(self):
        data = _wide()
        data[::13] = 0.0
        decoded, enc, _ = _roundtrip(data)
        nz = data > 0
        assert (decoded[~nz] == 0).all()  # exact zeros preserved
        assert not ((decoded == 0) & nz).any()  # the designed-out failure mode

    def test_old_log_scalar_zeroes_the_same_fixture(self):
        # Motivating contrast: the legacy 0-anchored log1p encoding at u8
        # collapses small values to zero on the same wide-range data.
        data = _wide()
        max_log = np.log1p(float(data.max()))
        codes = np.clip(np.round(np.log1p(data) / max_log * 255), 0, 255)
        zeroed = int(((codes == 0) & (data > 0)).sum())
        assert zeroed > 0  # legacy failure mode is real on this fixture

    def test_idempotent_reencode(self):
        data = _wide(n=5000)
        once, _, _ = _roundtrip(data)
        twice, _, _ = _roundtrip(once)
        np.testing.assert_array_equal(once, twice)

    def test_narrow_range_still_bounded_linear(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(0.5, 1.0, 1000).astype(np.float32)
        _, enc, _ = _roundtrip(data)
        assert enc["name"] == "bounded_scalar_uint8"

    def test_explicit_log_optin_selects_geolog(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(1.0, 100.0, 1000).astype(np.float32)
        _, enc, _ = _roundtrip(data, positive_scalar_encoding="log")
        assert enc["name"] == "geolog_scalar_uint16"
        _, enc8, _ = _roundtrip(
            data, mode=EncodingMode.MEMORY, positive_scalar_encoding="log"
        )
        assert enc8["name"] == "geolog_scalar_uint8"


class TestGeologEdgeCases:
    def test_constant_nonzero(self):
        data = np.full(200, 7.25, np.float32)
        # constant arrays take the broadcast priority path (exact)
        decoded, enc, _ = _roundtrip(data, positive_scalar_encoding="log")
        assert enc["name"] == "broadcasted"
        np.testing.assert_array_equal(decoded, data)

    def test_two_distinct_values_min_equals_grid_ends(self):
        data = np.array([1e-4, 1e-4, 1e4, 1e4, 0.0], np.float32)
        decoded, enc, _ = _roundtrip(data, positive_scalar_encoding="log")
        assert enc["name"] == "geolog_scalar_uint16"
        # grid endpoints are exact by construction (min/max anchoring)
        np.testing.assert_allclose(decoded[:2], 1e-4, rtol=1e-6)
        np.testing.assert_allclose(decoded[2:4], 1e4, rtol=1e-6)
        assert decoded[4] == 0.0

    def test_forward_inverse_helpers_are_exact_mirrors(self):
        data = _wide(n=3000).astype(np.float64)
        min_log = float(np.log(data.min()))
        max_log = float(np.log(data.max()))
        for bits in (8, 16):
            codes = ArrayEncoder._geolog_forward(data, bits, min_log, max_log)
            back = ArrayEncoder._geolog_inverse(codes, bits, min_log, max_log)
            codes2 = ArrayEncoder._geolog_forward(
                back.astype(np.float64), bits, min_log, max_log
            )
            np.testing.assert_array_equal(codes, codes2)


class TestGeologDecoderValidation:
    def _encode_then_corrupt(self, patch):
        data = _wide(n=500)
        enc = ArrayEncoder()
        g = memory_group()
        enc.encode(
            data=data,
            zarr_group=g,
            name="a",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.AUTO,
        )
        attrs = dict(g["a"].attrs["encoding"])
        attrs.update(patch)
        g["a"].attrs["encoding"] = attrs
        return g

    def test_rejects_non_finite_min_log(self):
        g = self._encode_then_corrupt({"min_log": float("nan")})
        with pytest.raises(ValueError, match="finite"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_max_below_min(self):
        g = self._encode_then_corrupt({"min_log": 5.0, "max_log": 1.0})
        with pytest.raises(ValueError, match="max_log >= min_log"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_missing_min_log(self):
        g = self._encode_then_corrupt({})
        attrs = dict(g["a"].attrs["encoding"])
        del attrs["min_log"]
        g["a"].attrs["encoding"] = attrs
        with pytest.raises(ValueError, match="min_log"):
            ArrayDecoder().decode(g["a"], g)

    def test_decode_matches_hand_computed(self):
        data = _wide(n=400)
        enc = ArrayEncoder()
        g = memory_group()
        enc.encode(
            data=data,
            zarr_group=g,
            name="a",
            semantic_type=SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.AUTO,
        )
        meta = dict(g["a"].attrs["encoding"])
        codes = np.asarray(g["a"])
        hand = np.zeros(codes.shape)
        nzc = codes > 0
        hand[nzc] = np.exp(
            meta["min_log"]
            + (codes[nzc].astype(np.float64) - 1)
            / 65534
            * (meta["max_log"] - meta["min_log"])
        )
        decoded = ArrayDecoder().decode(g["a"], g)
        # decoded is float32; compare relatively (values span 7 decades)
        np.testing.assert_allclose(decoded, hand, rtol=1e-6, atol=1e-9)
