"""Round-trip + edge-case tests for the differential Cholesky encodings.

CHOLESKY_DIAG (positive, per-column log) and CHOLESKY_OFFDIAG (signed, per-column
signed-log), each at float32 (PRECISION), uint16 (AUTO) and uint8 (MEMORY).
"""

import numpy as np
import pytest
import zarr

from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _roundtrip(data, semantic_type, mode):
    """Encode then decode `data`, returning (decoded, encoding_dict)."""
    enc = ArrayEncoder()  # fresh: avoid cross-call array_ref dedup in unit tests
    g = zarr.group(store=zarr.MemoryStore())
    enc.encode(
        data=data, zarr_group=g, name="a", semantic_type=semantic_type, mode=mode
    )
    encoding = dict(g["a"].attrs["encoding"])
    decoded = ArrayDecoder().decode(g["a"], g)
    return decoded, encoding, np.asarray(g["a"])


class TestCholeskyDiag:
    @pytest.mark.parametrize(
        "mode,name,dtype,bits",
        [
            (EncodingMode.PRECISION, "float32", np.float32, None),
            (EncodingMode.AUTO, "log_perchannel_u16", np.uint16, 16),
            (EncodingMode.MEMORY, "log_perchannel_u8", np.uint8, 8),
        ],
    )
    def test_roundtrip(self, mode, name, dtype, bits):
        rng = np.random.default_rng(0)
        diag = rng.uniform(0.4, 18.0, size=(4000, 3)).astype(np.float32)  # positive
        decoded, enc, stored = _roundtrip(diag, SemanticType.CHOLESKY_DIAG, mode)
        assert enc["name"] == name
        assert stored.dtype == dtype
        assert decoded.shape == diag.shape
        if mode == EncodingMode.PRECISION:
            np.testing.assert_array_equal(decoded, diag)
        else:
            assert len(enc["col_lo"]) == 3 and len(enc["col_hi"]) == 3  # per-column
            relmax = np.abs(decoded - diag).max() / np.ptp(diag)
            assert relmax < (1e-3 if bits == 16 else 1e-2)

    def test_per_column_scales_differ(self):
        # Anisotropic columns → per-column scales must be distinct (not global).
        rng = np.random.default_rng(1)
        diag = np.stack(
            [
                rng.uniform(0.4, 1.0, 500),  # fine column
                rng.uniform(5.0, 50.0, 500),  # coarse column
                rng.uniform(0.4, 1.0, 500),
            ],
            axis=1,
        ).astype(np.float32)
        decoded, enc, _ = _roundtrip(
            diag, SemanticType.CHOLESKY_DIAG, EncodingMode.MEMORY
        )
        assert enc["col_hi"][1] > enc["col_hi"][0]  # column 1 has a larger range
        # per-column scaling keeps the fine columns precise despite the coarse one
        col0_rel = np.abs(decoded[:, 0] - diag[:, 0]).max() / np.ptp(diag[:, 0])
        assert col0_rel < 0.02


class TestCholeskyOffdiag:
    @pytest.mark.parametrize(
        "mode,name,dtype",
        [
            (EncodingMode.PRECISION, "float32", np.float32),
            (EncodingMode.AUTO, "signed_log_perchannel_u16", np.uint16),
            (EncodingMode.MEMORY, "signed_log_perchannel_u8", np.uint8),
        ],
    )
    def test_roundtrip_signed(self, mode, name, dtype):
        rng = np.random.default_rng(2)
        # zero-peaked, signed, heavy tails (like real off-diagonals)
        off = (rng.standard_normal((4000, 3)) * 0.3).astype(np.float32)
        off[::50] *= 15.0  # tails
        decoded, enc, stored = _roundtrip(off, SemanticType.CHOLESKY_OFFDIAG, mode)
        assert enc["name"] == name
        assert stored.dtype == dtype
        if mode == EncodingMode.PRECISION:
            np.testing.assert_array_equal(decoded, off)
        else:
            assert len(enc["col_lo"]) == 3 and len(enc["col_hi"]) == 3
            # sign preserved everywhere it is non-trivial
            nz = np.abs(off) > 1e-3
            assert np.mean(np.sign(decoded[nz]) == np.sign(off[nz])) > 0.99


class TestEdgeCases:
    def test_constant_column(self):
        const = np.full((100, 2), 3.0, np.float32)
        decoded, _, _ = _roundtrip(
            const, SemanticType.CHOLESKY_DIAG, EncodingMode.MEMORY
        )
        np.testing.assert_allclose(decoded, const, atol=1e-4)

    def test_single_column_1d(self):
        rng = np.random.default_rng(3)
        oned = rng.uniform(0.5, 2.0, (30, 1)).astype(np.float32)
        decoded, enc, _ = _roundtrip(
            oned, SemanticType.CHOLESKY_DIAG, EncodingMode.AUTO
        )
        assert len(enc["col_lo"]) == 1
        np.testing.assert_allclose(decoded, oned, rtol=0, atol=1e-3)

    def test_all_zero_offdiag(self):
        z = np.zeros((50, 3), np.float32)
        decoded, _, _ = _roundtrip(
            z, SemanticType.CHOLESKY_OFFDIAG, EncodingMode.MEMORY
        )
        np.testing.assert_array_equal(decoded, z)

    def test_all_zero_diagonal(self):
        # Singular covariance (zero-scale splat). All rows identical → the
        # broadcast optimization takes over (the path the broadcasted fixtures
        # exercise); the round-trip must still recover exact zeros.
        z = np.zeros((50, 3), np.float32)
        decoded, enc, _ = _roundtrip(
            z, SemanticType.CHOLESKY_DIAG, EncodingMode.MEMORY
        )
        assert enc["name"] == "broadcasted"
        np.testing.assert_array_equal(decoded, z)

    def test_zero_diagonal_rows_quantized_path(self):
        # Mix of zero-scale and normal splats forces the per-channel log path
        # (not broadcast) and must keep the zero rows exactly zero.
        rng = np.random.default_rng(11)
        diag = rng.uniform(0.4, 5.0, size=(60, 3)).astype(np.float32)
        diag[::3] = 0.0  # singular splats interleaved
        decoded, enc, _ = _roundtrip(
            diag, SemanticType.CHOLESKY_DIAG, EncodingMode.MEMORY
        )
        assert enc["name"] == "log_perchannel_u8"
        np.testing.assert_allclose(decoded[::3], 0.0, atol=1e-4)

    def test_idempotent_resave_no_compounding(self):
        # AUTO→decode→AUTO must not compound error: the second cycle is exact.
        rng = np.random.default_rng(7)
        diag = rng.uniform(0.4, 18.0, size=(2000, 3)).astype(np.float32)
        once, _, _ = _roundtrip(diag, SemanticType.CHOLESKY_DIAG, EncodingMode.AUTO)
        twice, _, _ = _roundtrip(once, SemanticType.CHOLESKY_DIAG, EncodingMode.AUTO)
        np.testing.assert_array_equal(twice, once)  # zero additional error


class TestDecoderValidation:
    """The per-channel decoders must reject malformed scales (matching the
    scalar decoders and the viewer's makePerChannelDequant) rather than
    silently producing corrupt output."""

    def _encode_then_corrupt(self, enc_patch):
        rng = np.random.default_rng(9)
        diag = rng.uniform(0.4, 5.0, size=(100, 3)).astype(np.float32)
        enc = ArrayEncoder()
        g = zarr.group(store=zarr.MemoryStore())
        enc.encode(
            data=diag,
            zarr_group=g,
            name="a",
            semantic_type=SemanticType.CHOLESKY_DIAG,
            mode=EncodingMode.AUTO,
        )
        attrs = dict(g["a"].attrs["encoding"])
        attrs.update(enc_patch)
        g["a"].attrs["encoding"] = attrs
        return g

    def test_rejects_non_finite_scales(self):
        g = self._encode_then_corrupt({"col_hi": [1.0, float("inf"), 1.0]})
        with pytest.raises(ValueError, match="finite"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_length_mismatch(self):
        g = self._encode_then_corrupt({"col_lo": [0.0, 0.0], "col_hi": [1.0, 1.0]})
        with pytest.raises(ValueError, match="per-column scales"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_hi_below_lo(self):
        g = self._encode_then_corrupt({"col_lo": [5.0, 0.0, 0.0]})  # lo > hi col 0
        with pytest.raises(ValueError, match="col_hi >= col_lo"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_nonpositive_bits(self):
        g = self._encode_then_corrupt({"bits": 0})
        with pytest.raises(ValueError, match="bits > 0"):
            ArrayDecoder().decode(g["a"], g)

    def test_rejects_non_1d_scales(self):
        # col_lo/col_hi must be 1-D length-C lists, not nested.
        g = self._encode_then_corrupt(
            {"col_lo": [[0.0, 0.0, 0.0]], "col_hi": [[1.0, 1.0, 1.0]]}
        )
        with pytest.raises(ValueError, match="1-D col_lo/col_hi"):
            ArrayDecoder().decode(g["a"], g)

    def test_decode_matches_hand_computed(self):
        # Lock the exact inverse formula (per-column signed-log, uint8).
        rng = np.random.default_rng(4)
        off = (rng.standard_normal((200, 2)) * 0.5).astype(np.float32)
        decoded, enc, stored = _roundtrip(
            off, SemanticType.CHOLESKY_OFFDIAG, EncodingMode.MEMORY
        )
        lo = np.asarray(enc["col_lo"])
        hi = np.asarray(enc["col_hi"])
        levels = (1 << enc["bits"]) - 1
        y = lo + stored.astype(np.float64) / levels * np.maximum(hi - lo, 1e-30)
        hand = np.sign(y) * np.expm1(np.abs(y))
        np.testing.assert_allclose(decoded, hand, rtol=0, atol=1e-6)
