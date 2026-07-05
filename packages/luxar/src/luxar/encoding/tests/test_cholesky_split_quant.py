"""Round-trip + edge-case tests for the differential Cholesky encodings.

CHOLESKY_DIAG (positive, per-column log) and CHOLESKY_OFFDIAG (signed, per-column
signed-log), each at float32 (PRECISION) and uint8 (AUTO and MEMORY). AUTO's
escalation to uint16 is owned by ``encode_cholesky_split`` (the joint pair entry
point with the encode-time covariance certificate) — tested in
``TestEncodeCholeskySplit`` below.
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
            (EncodingMode.AUTO, "log_perchannel_u8", np.uint8, 8),
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
            (EncodingMode.AUTO, "signed_log_perchannel_u8", np.uint8),
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
        np.testing.assert_allclose(decoded, oned, rtol=0, atol=1e-2)

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
        decoded, enc, _ = _roundtrip(z, SemanticType.CHOLESKY_DIAG, EncodingMode.MEMORY)
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


class TestEncodeCholeskySplit:
    """The joint pair entry point: one owned policy, encode-time certificate,
    u8 -> u16 -> float32 escalation ladder (AUTO only)."""

    @staticmethod
    def _make(n=4000, seed=0):
        rng = np.random.default_rng(seed)
        diag = rng.uniform(0.4, 5.0, size=(n, 3)).astype(np.float32)
        off = (rng.standard_normal((n, 3)) * 0.3).astype(np.float32)
        return diag, off

    @staticmethod
    def _encode(diag, off, mode, ndim=3, **kw):
        g = zarr.group(store=zarr.MemoryStore())
        ArrayEncoder().encode_cholesky_split(g, diag, off, ndim, mode, **kw)
        return g

    def test_auto_clean_data_stays_u8_with_certificate(self):
        diag, off = self._make()
        g = self._encode(diag, off, EncodingMode.AUTO)
        for name, enc_name in (
            ("cholesky_factors_diag", "log_perchannel_u8"),
            ("cholesky_factors_offdiag", "signed_log_perchannel_u8"),
        ):
            enc = dict(g[name].attrs["encoding"])
            assert enc["name"] == enc_name
            cert = enc["certificate"]
            assert cert["metric"] == "cov_relf_p95"
            assert cert["tier"] == "u8"
            assert 0.0 <= cert["value"] <= cert["threshold"]
        # certificate is provenance only: decode needs nothing beyond the
        # array's own standard encoding fields (self-contained decode).
        decoded = ArrayDecoder().decode(g["cholesky_factors_diag"], g)
        assert np.abs(decoded - diag).max() / np.ptp(diag) < 1e-2

    def test_auto_escalates_to_u16_on_stretched_range(self):
        # A few huge-sigma outliers stretch every column's log range so the
        # normal splats' u8 reconstruction error blows past the threshold —
        # the merged-heterogeneous-stores failure mode.
        diag, off = self._make()
        diag[:10] = 1e8
        with pytest.warns(UserWarning, match="escalating to uint16"):
            g = self._encode(diag, off, EncodingMode.AUTO)
        for name, enc_name in (
            ("cholesky_factors_diag", "log_perchannel_u16"),
            ("cholesky_factors_offdiag", "signed_log_perchannel_u16"),
        ):
            enc = dict(g[name].attrs["encoding"])
            assert enc["name"] == enc_name
            cert = enc["certificate"]
            assert cert["tier"] == "u16"
            assert cert["value"] <= cert["threshold"]  # invariant holds at u16
        # escalated store still round-trips within the u16 tolerance
        decoded = ArrayDecoder().decode(g["cholesky_factors_diag"], g)
        assert np.abs(decoded - diag).max() / np.ptp(diag) < 1e-3

    def test_memory_never_escalates(self):
        diag, off = self._make()
        diag[:10] = 1e8  # same nasty data as the escalation test
        g = self._encode(diag, off, EncodingMode.MEMORY)
        enc = dict(g["cholesky_factors_diag"].attrs["encoding"])
        assert enc["name"] == "log_perchannel_u8"
        assert "certificate" not in enc  # explicit user choice: no certificate

    def test_precision_stays_float32(self):
        diag, off = self._make(n=200)
        g = self._encode(diag, off, EncodingMode.PRECISION)
        for name in ("cholesky_factors_diag", "cholesky_factors_offdiag"):
            enc = dict(g[name].attrs["encoding"])
            assert enc["name"] == "float32"
            assert "certificate" not in enc

    def test_both_halves_share_one_tier(self):
        # Nasty diag alone must drag the (well-behaved) offdiag up to u16 too:
        # mixed tiers within one pair are forbidden by design.
        diag, off = self._make()
        diag[:10] = 1e8
        with pytest.warns(UserWarning):
            g = self._encode(diag, off, EncodingMode.AUTO)
        assert (
            g["cholesky_factors_diag"].attrs["encoding"]["bits"]
            == g["cholesky_factors_offdiag"].attrs["encoding"]["bits"]
            == 16
        )

    def test_threshold_override_and_float32_rung(self):
        # An impossible threshold pushes AUTO through u8 and u16 down to the
        # float32 rung (exercises the ladder end even though real data never
        # reaches it).
        diag, off = self._make(n=500)
        with pytest.warns(UserWarning, match="storing float32"):
            g = self._encode(diag, off, EncodingMode.AUTO, certificate_threshold=0.0)
        enc = dict(g["cholesky_factors_diag"].attrs["encoding"])
        assert enc["name"] == "float32"
        assert enc["certificate"]["tier"] == "float32"
        decoded = ArrayDecoder().decode(g["cholesky_factors_diag"], g)
        np.testing.assert_array_equal(decoded, diag)

    def test_1d_skips_offdiag(self):
        diag, _ = self._make(n=100)
        g = self._encode(
            diag[:, :1], np.zeros((100, 0), np.float32), EncodingMode.AUTO, ndim=1
        )
        assert "cholesky_factors_offdiag" not in g
        enc = dict(g["cholesky_factors_diag"].attrs["encoding"])
        assert enc["name"] == "log_perchannel_u8"
        assert enc["certificate"]["tier"] == "u8"

    def test_rejects_mismatched_shapes(self):
        diag, off = self._make(n=100)
        with pytest.raises(ValueError, match="diag must have shape"):
            self._encode(diag[:, :2], off, EncodingMode.AUTO)
        with pytest.raises(ValueError, match="offdiag must have shape"):
            self._encode(diag, off[:, :2], EncodingMode.AUTO)

    @pytest.mark.parametrize(
        "mode", [EncodingMode.AUTO, EncodingMode.MEMORY, EncodingMode.PRECISION]
    )
    def test_empty_pair_writes_passthrough(self, mode):
        # Zero-splat pairs (empty BSP part / LOD tile / filtered-out result)
        # must fall through to encode()'s size==0 passthrough — never reach
        # _is_uniform (which indexes data[0]) or the certificate.
        g = self._encode(
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            mode,
        )
        for name in ("cholesky_factors_diag", "cholesky_factors_offdiag"):
            assert g[name].shape == (0, 3)
            assert "certificate" not in dict(g[name].attrs.get("encoding", {}))

    def test_sample_cap_bounds_certificate_and_preserves_decision(self, monkeypatch):
        # Above COV_CERT_SAMPLE_MAX rows, the certificate measures a bounded
        # evenly-spaced sample (records "sample") — but the quantization scales
        # come from the FULL columns, so the sampled value tracks the full one.
        import luxar.encoding.encoder as enc_mod

        diag, off = self._make(n=4000, seed=8)
        full = ArrayEncoder._cov_relf_p95(
            np.maximum(diag.astype(np.float64), 0.0),
            ArrayEncoder._perchannel_log_roundtrip(diag, 8, signed=False),
            off,
            ArrayEncoder._perchannel_log_roundtrip(off, 8, signed=True),
            3,
        )
        monkeypatch.setattr(enc_mod, "COV_CERT_SAMPLE_MAX", 512)
        g = self._encode(diag, off, EncodingMode.AUTO)
        cert = dict(g["cholesky_factors_diag"].attrs["encoding"])["certificate"]
        assert cert["sample"] == 512
        assert cert["tier"] == "u8"
        # sampled estimate within 25% of the full measurement (same scales)
        assert abs(cert["value"] - full) / full < 0.25

        # Escalation must still fire through the sample: outliers spread across
        # the array so evenly-spaced sampling sees the stretched range (the
        # scales are full-column regardless, which is what stretches the grid).
        diag2 = diag.copy()
        diag2[::16] = 1e8
        with pytest.warns(UserWarning, match="escalating to uint16"):
            g2 = self._encode(diag2, off, EncodingMode.AUTO)
        cert2 = dict(g2["cholesky_factors_diag"].attrs["encoding"])["certificate"]
        assert cert2["tier"] == "u16"
        assert cert2["sample"] == 512

        # Below the cap: no "sample" key (full measurement).
        monkeypatch.setattr(enc_mod, "COV_CERT_SAMPLE_MAX", 262_144)
        g3 = self._encode(diag, off, EncodingMode.AUTO)
        assert (
            "sample"
            not in dict(g3["cholesky_factors_diag"].attrs["encoding"])["certificate"]
        )

    def test_certificate_metric_matches_trils_convention(self):
        # The encoder-local Sigma rebuild must agree with the canonical
        # gsplats.utils.trils packing (row-major np.tril_indices). The test may
        # import both; production encoding code must not import gsplats.
        from luxar.gsplats.utils.trils import merge_tril, unpack_tril

        diag, off = self._make(n=300, seed=5)
        diag_q = diag * 1.01
        off_q = off * 0.99

        def sigma_via_trils(dg, od):
            packed = merge_tril(dg, od, 3)
            tri = unpack_tril(packed.astype(np.float64), 3)
            return np.einsum("nij,nkj->nik", tri, tri).reshape(len(dg), -1)

        s0 = sigma_via_trils(diag, off)
        sq = sigma_via_trils(diag_q, off_q)
        num = np.linalg.norm(sq - s0, axis=1)
        den = np.maximum(np.linalg.norm(s0, axis=1), 1e-30)
        expected = float(np.percentile(num / den, 95))

        got = ArrayEncoder._cov_relf_p95(diag, diag_q, off, off_q, 3)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_roundtrip_helper_matches_stored_encoding(self):
        # The certificate's round-trip helper and the real encode->decode path
        # must be the same transform (the "certificate can never lie" contract).
        diag, off = self._make(n=800, seed=6)
        g = self._encode(diag, off, EncodingMode.AUTO)
        decoded = ArrayDecoder().decode(g["cholesky_factors_diag"], g)
        helper = ArrayEncoder._perchannel_log_roundtrip(diag, 8, signed=False)
        np.testing.assert_allclose(decoded, helper, rtol=0, atol=1e-6)
        decoded_off = ArrayDecoder().decode(g["cholesky_factors_offdiag"], g)
        helper_off = ArrayEncoder._perchannel_log_roundtrip(off, 8, signed=True)
        np.testing.assert_allclose(decoded_off, helper_off, rtol=0, atol=1e-6)


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
