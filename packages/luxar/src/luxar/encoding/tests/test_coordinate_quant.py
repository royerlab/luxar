"""Round-trip + edge-case tests for COORDINATE uint16 fixed-point quantization
(the generic ``linear_perchannel_*`` encoding).

COORDINATE positions/centers/vertices: PRECISION → float32 (exact); AUTO / MEMORY
→ ``linear_perchannel_u16`` (per-axis fixed-point), decoded back to float32.
Coordinates never use uint8 (too coarse) and never float16. An array-local extent
rail warns above 2**12 and falls back to float32 at/above 2**16.
"""

import warnings

import numpy as np
import pytest
import zarr

from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _roundtrip(data, mode, float16_allowed=False):
    """Encode `data` as COORDINATE then decode; return (decoded, encoding, stored)."""
    enc = ArrayEncoder(float16_allowed=float16_allowed)
    g = zarr.group(store=zarr.MemoryStore())
    enc.encode(
        data=data,
        zarr_group=g,
        name="c",
        semantic_type=SemanticType.COORDINATE,
        mode=mode,
    )
    encoding = dict(g["c"].attrs["encoding"])
    decoded = ArrayDecoder().decode(g["c"], g)
    return decoded, encoding, np.asarray(g["c"])


def _atol(data):
    """uint16 per-axis fixed-point bound: max per-axis extent / 65535, ×2 safety."""
    return float(np.ptp(np.asarray(data, np.float64), axis=0).max()) / 65535 * 2


class TestCoordinateModes:
    @pytest.mark.parametrize("mode", [EncodingMode.AUTO, EncodingMode.MEMORY])
    def test_uint16_roundtrip(self, mode):
        rng = np.random.default_rng(0)
        pos = (rng.random((4000, 3)) * [400, 1800, 2000] + [3, 100, 30]).astype(
            np.float32
        )
        decoded, enc, stored = _roundtrip(pos, mode)
        assert enc["name"] == "linear_perchannel_u16"
        assert stored.dtype == np.uint16
        assert decoded.dtype == np.float32  # always decodes to float32
        assert len(enc["col_lo"]) == 3 and len(enc["col_hi"]) == 3  # per-axis scales
        np.testing.assert_allclose(decoded, pos, atol=_atol(pos))

    def test_precision_exact(self):
        rng = np.random.default_rng(1)
        pos = (rng.standard_normal((500, 3)) * 100).astype(np.float32)
        decoded, enc, stored = _roundtrip(pos, EncodingMode.PRECISION)
        assert enc["name"] == "float32"
        assert stored.dtype == np.float32
        np.testing.assert_array_equal(decoded, pos)

    def test_negatives_and_sign(self):
        # coordinates can be negative — linear (not log) handles them
        rng = np.random.default_rng(2)
        pos = (rng.standard_normal((2000, 3)) * 50).astype(np.float32)  # centered at 0
        assert (pos < 0).any() and (pos > 0).any()
        decoded, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "linear_perchannel_u16"
        np.testing.assert_allclose(decoded, pos, atol=_atol(pos))
        nz = np.abs(pos) > _atol(pos)  # sign preserved away from zero
        assert np.array_equal(np.sign(decoded[nz]), np.sign(pos[nz]))

    def test_never_uint8_never_float16(self):
        rng = np.random.default_rng(3)
        pos = (rng.standard_normal((100, 3)) * 10).astype(np.float32)
        for float16_allowed in (False, True):
            _, enc, stored = _roundtrip(
                pos, EncodingMode.MEMORY, float16_allowed=float16_allowed
            )
            assert enc["name"] == "linear_perchannel_u16"  # never u8
            assert stored.dtype == np.uint16  # never float16


class TestCoordinateEdgeCases:
    def test_single_point_exact(self):
        pos = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        decoded, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        # every axis is a constant column → decodes exactly
        np.testing.assert_allclose(decoded, pos, atol=1e-4)

    def test_constant_axis(self):
        rng = np.random.default_rng(4)
        pos = rng.standard_normal((500, 3)).astype(np.float32)
        pos[:, 2] = 7.0  # planar slice: constant Z
        decoded, _, _ = _roundtrip(pos, EncodingMode.AUTO)
        np.testing.assert_allclose(decoded[:, 2], 7.0, atol=1e-4)
        np.testing.assert_allclose(decoded, pos, atol=_atol(pos) + 1e-4)

    def test_empty(self):
        decoded, _, _ = _roundtrip(np.zeros((0, 3), np.float32), EncodingMode.AUTO)
        assert decoded.shape == (0, 3)

    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
    def test_nd(self, d):
        rng = np.random.default_rng(5)
        pos = (rng.standard_normal((300, d)) * 20).astype(np.float32)
        decoded, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "linear_perchannel_u16"
        assert len(enc["col_lo"]) == d
        np.testing.assert_allclose(decoded, pos, atol=_atol(pos))

    def test_idempotent_resave_no_compounding(self):
        rng = np.random.default_rng(6)
        pos = (rng.standard_normal((2000, 3)) * 30).astype(np.float32)
        once, _, _ = _roundtrip(pos, EncodingMode.AUTO)
        twice, _, _ = _roundtrip(once, EncodingMode.AUTO)
        np.testing.assert_array_equal(twice, once)  # second cycle adds zero error

    def test_decode_matches_hand_computed(self):
        # lock the exact inverse formula: x = lo + u/levels·(hi-lo)
        rng = np.random.default_rng(7)
        pos = (rng.standard_normal((200, 3)) * 40).astype(np.float32)
        decoded, enc, stored = _roundtrip(pos, EncodingMode.AUTO)
        lo = np.asarray(enc["col_lo"])
        hi = np.asarray(enc["col_hi"])
        levels = (1 << enc["bits"]) - 1
        # decoder computes in float64 then casts to the original dtype (float32);
        # mirror that cast so the formula lock is bit-exact.
        hand = (lo + stored.astype(np.float64) / levels * np.maximum(hi - lo, 1e-30)).astype(
            np.float32
        )
        np.testing.assert_array_equal(decoded, hand)


class TestExtentRail:
    def test_warn_above_4096(self):
        rng = np.random.default_rng(8)
        pos = rng.standard_normal((1000, 3)).astype(np.float32)
        pos[0, 0] = 5000.0  # per-axis extent > 2**12
        with pytest.warns(UserWarning, match="shrinking"):
            _, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "linear_perchannel_u16"  # still u16 (just a warning)

    def test_fallback_float32_above_65536(self):
        rng = np.random.default_rng(9)
        pos = rng.standard_normal((1000, 3)).astype(np.float32)
        pos[0, 0] = 70000.0  # per-axis extent >= 2**16 → float32 fallback
        with pytest.warns(UserWarning, match="storing float32"):
            decoded, enc, stored = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "float32"
        assert stored.dtype == np.float32
        np.testing.assert_array_equal(decoded, pos)  # exact — no quantization

    def test_no_warn_small_extent(self):
        rng = np.random.default_rng(10)
        pos = (rng.standard_normal((1000, 3)) * 100).astype(np.float32)  # extent < 4096
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "linear_perchannel_u16"
        assert not any(
            ("shrinking" in str(w.message)) or ("storing float32" in str(w.message))
            for w in caught
        )


class TestGenericLinearPerchannel:
    """The encoding is GENERIC (reusable), not coordinate-specific: verify a direct
    ``linear_perchannel`` round-trip through the decoder at both bit depths, and the
    shared per-column scale validation (rejects malformed metadata)."""

    @pytest.mark.parametrize("bits,udtype", [(16, np.uint16), (8, np.uint8)])
    def test_direct_generic_roundtrip(self, bits, udtype):
        rng = np.random.default_rng(11)
        vals = (rng.standard_normal((300, 4)) * 5).astype(np.float32)
        lo = vals.min(0).astype(np.float64)
        hi = vals.max(0).astype(np.float64)
        levels = (1 << bits) - 1
        rngv = np.maximum(hi - lo, 1e-30)
        u = np.round((vals - lo) / rngv * levels).astype(udtype)
        g = zarr.group(store=zarr.MemoryStore())
        g.create_dataset("a", data=u)
        g["a"].attrs["encoding"] = {
            "name": f"linear_perchannel_u{bits}",
            "col_lo": lo.tolist(),
            "col_hi": hi.tolist(),
            "bits": bits,
            "original_dtype": "float32",
        }
        decoded = ArrayDecoder().decode(g["a"], g)
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, vals, atol=float(rngv.max()) / levels * 2)

    def test_decoder_rejects_non_finite_scales(self):
        rng = np.random.default_rng(12)
        vals = (rng.standard_normal((50, 3)) * 5).astype(np.float32)
        g = zarr.group(store=zarr.MemoryStore())
        ArrayEncoder().encode(
            data=vals,
            zarr_group=g,
            name="a",
            semantic_type=SemanticType.COORDINATE,
            mode=EncodingMode.AUTO,
        )
        attrs = dict(g["a"].attrs["encoding"])
        attrs["col_hi"] = [1.0, float("inf"), 1.0]  # corrupt
        g["a"].attrs["encoding"] = attrs
        with pytest.raises(ValueError, match="finite"):
            ArrayDecoder().decode(g["a"], g)

    def test_decoder_rejects_length_mismatch(self):
        rng = np.random.default_rng(13)
        vals = (rng.standard_normal((50, 3)) * 5).astype(np.float32)
        g = zarr.group(store=zarr.MemoryStore())
        ArrayEncoder().encode(
            data=vals,
            zarr_group=g,
            name="a",
            semantic_type=SemanticType.COORDINATE,
            mode=EncodingMode.AUTO,
        )
        attrs = dict(g["a"].attrs["encoding"])
        attrs["col_lo"] = [0.0, 0.0]  # 2 scales for a 3-column array
        attrs["col_hi"] = [1.0, 1.0]
        g["a"].attrs["encoding"] = attrs
        with pytest.raises(ValueError, match="per-column scales"):
            ArrayDecoder().decode(g["a"], g)
