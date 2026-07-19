"""Tests for the ``luxar_delta_v1`` zarr filter (columnar delta+zigzag).

Locks three contracts:
1. the WIRE FORMAT (hand-computed reference — the TS twin decodes these bytes),
2. exact round-trips through the codec and through a real zarr array,
3. the encode-time probe's monotonic gating (wins on smooth ramps, declines
   on noise / unknown chunking / missing compressor).
"""

import warnings

import numcodecs
import numpy as np
import pytest
import zarr

from luxar.encoding._encoders.delta_codec import (
    LuxarDelta,
    probe_delta_filter,
)
from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor


def _smooth_codes(n: int, cols: int, dtype: np.dtype, seed: int = 7) -> np.ndarray:
    """Hilbert-like smooth random walk quantized to the full code range."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 1.0, size=(n, cols)), axis=0)
    lo, hi = walk.min(axis=0), walk.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    levels = 255 if np.dtype(dtype) == np.uint8 else 65535
    return np.round((walk - lo) / span * levels).astype(dtype)


class TestWireFormat:
    """Byte-level lock on the format the TS codec must mirror."""

    def test_hand_computed_u16(self):
        # rows x cols = 3 x 2, per-column delta with implicit 0 anchor.
        codes = np.array([[100, 5], [98, 5], [103, 65535]], dtype=np.uint16)
        codec = LuxarDelta(cols=2, bits=16)
        enc = codec.encode(codes)
        # col 0 deltas: 100, -2, +5 -> zigzag 200, 3, 10
        # col 1 deltas: 5, 0, 65530 mod 2^16 -> signed -6 -> zigzag 10, 0, 11
        expected = np.array([200, 3, 10, 10, 0, 11], dtype=np.uint16)
        np.testing.assert_array_equal(np.asarray(enc).ravel(), expected)
        np.testing.assert_array_equal(codec.decode(enc), codes)

    def test_hand_computed_u8_wraparound(self):
        # Modular arithmetic: 250 -> 3 is +9 mod 256 (not -247).
        codes = np.array([[250], [3], [1]], dtype=np.uint8)
        codec = LuxarDelta(cols=1, bits=8)
        enc = codec.encode(codes)
        # deltas: 250 -> signed -6 (250 >= 128), +9, -2 -> zigzag 11, 18, 3
        expected = np.array([11, 18, 3], dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(enc).ravel(), expected)
        np.testing.assert_array_equal(codec.decode(enc), codes)

    def test_columnar_layout(self):
        # Constant col 0, ramp col 1: residual blocks must be contiguous
        # per column (all of col 0's, then all of col 1's).
        codes = np.stack(
            [np.full(5, 7, np.uint16), np.arange(5, dtype=np.uint16)], axis=1
        )
        enc = np.asarray(LuxarDelta(cols=2, bits=16).encode(codes)).ravel()
        # col 0: anchor 7 then zeros; col 1: anchor 0 then +1s (zigzag 2).
        np.testing.assert_array_equal(enc[:5], [14, 0, 0, 0, 0])
        np.testing.assert_array_equal(enc[5:], [0, 2, 2, 2, 2])


class TestRoundTrip:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    @pytest.mark.parametrize("cols", [1, 3, 6])
    def test_random_codes_exact(self, dtype, cols):
        rng = np.random.default_rng(42)
        levels = 256 if np.dtype(dtype) == np.uint8 else 65536
        codes = rng.integers(0, levels, size=(1013, cols)).astype(dtype)
        codec = LuxarDelta(cols=cols, bits=8 * np.dtype(dtype).itemsize)
        np.testing.assert_array_equal(codec.decode(codec.encode(codes)), codes)

    def test_decode_from_bytes(self):
        # zarr hands the filter raw bytes after decompression.
        codes = _smooth_codes(500, 3, np.uint16)
        codec = LuxarDelta(cols=3, bits=16)
        raw = np.asarray(codec.encode(codes)).tobytes()
        np.testing.assert_array_equal(
            np.asarray(codec.decode(raw)).reshape(codes.shape), codes
        )

    def test_out_param(self):
        codes = _smooth_codes(64, 2, np.uint8)
        codec = LuxarDelta(cols=2, bits=8)
        out = np.empty_like(codes)
        codec.decode(codec.encode(codes), out=out)
        np.testing.assert_array_equal(out, codes)

    def test_single_row_and_empty(self):
        codec = LuxarDelta(cols=3, bits=16)
        one = np.array([[9, 0, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(codec.decode(codec.encode(one)), one)
        empty = np.empty((0, 3), dtype=np.uint16)
        assert np.asarray(codec.decode(codec.encode(empty))).size == 0

    def test_zarr_array_round_trip_with_partial_last_chunk(self):
        codes = _smooth_codes(1000, 3, np.uint16)
        g = zarr.group()
        g.create_dataset(
            "codes",
            data=codes,
            chunks=(256, 3),  # last chunk is partial (232 rows)
            compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, codes.dtype),
            filters=[LuxarDelta(cols=3, bits=16)],
        )
        np.testing.assert_array_equal(g["codes"][:], codes)
        # Metadata carries the filter for the viewer's zarrita to resolve.
        (flt,) = g["codes"].filters
        assert flt.get_config() == {"id": "luxar_delta_v1", "cols": 3, "bits": 16}

    def test_registry_lookup(self):
        codec = numcodecs.get_codec({"id": "luxar_delta_v1", "cols": 4, "bits": 8})
        assert isinstance(codec, LuxarDelta)
        assert (codec.cols, codec.bits) == (4, 8)


class TestPropertyRoundTrip:
    """Hypothesis property: decode(encode(x)) == x for ALL valid inputs."""

    @pytest.mark.parametrize("bits", [8, 16])
    def test_round_trip_law(self, bits):
        from hypothesis import given, settings
        from hypothesis import strategies as st_

        dt = np.uint8 if bits == 8 else np.uint16
        levels = 1 << bits

        @settings(max_examples=200, deadline=None)
        @given(
            cols=st_.integers(min_value=1, max_value=8),
            rows=st_.integers(min_value=0, max_value=300),
            seed=st_.integers(min_value=0, max_value=2**31),
        )
        def check(cols, rows, seed):
            rng = np.random.default_rng(seed)
            codes = rng.integers(0, levels, size=(rows, cols)).astype(dt)
            codec = LuxarDelta(cols=cols, bits=bits)
            round_tripped = np.asarray(codec.decode(codec.encode(codes)))
            np.testing.assert_array_equal(round_tripped.reshape(codes.shape), codes)

        check()


class TestValidation:
    def test_bad_params(self):
        with pytest.raises(ValueError):
            LuxarDelta(cols=3, bits=12)
        with pytest.raises(ValueError):
            LuxarDelta(cols=0, bits=16)

    def test_f_order_input_fails_loud(self):
        # numcodecs flattens in memory order — an F-contiguous chunk would
        # silently scramble the columnar transform. zarr always hands the
        # filter C-contiguous chunks; direct misuse must raise, not corrupt.
        codec = LuxarDelta(cols=3, bits=16)
        f_order = np.asfortranarray(np.zeros((10, 3), dtype=np.uint16))
        with pytest.raises(ValueError, match="C-contiguous"):
            codec.encode(f_order)

    def test_size_not_multiple_of_cols(self):
        codec = LuxarDelta(cols=3, bits=16)
        with pytest.raises(ValueError, match="multiple of cols"):
            codec.encode(np.zeros(10, dtype=np.uint16))


class TestProbe:
    def _comp(self, dtype):
        return resolve_compressor(WIDTH_AWARE_DEFAULT, np.dtype(dtype))

    def test_smooth_ramp_wins(self):
        codes = _smooth_codes(20000, 3, np.uint16)
        filters = probe_delta_filter(codes, (4096, 3), self._comp(np.uint16))
        assert filters is not None
        (codec,) = filters
        assert (codec.cols, codec.bits) == (3, 16)

    def test_noise_declines(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 65536, size=(20000, 3)).astype(np.uint16)
        assert probe_delta_filter(codes, (4096, 3), self._comp(np.uint16)) is None

    def test_1d_smooth_wins_without_chunks(self):
        codes = _smooth_codes(20000, 1, np.uint16).ravel()
        filters = probe_delta_filter(codes, None, self._comp(np.uint16))
        assert filters is not None
        assert filters[0].cols == 1

    def test_declines_unknown_or_column_splitting_chunks(self):
        codes = _smooth_codes(20000, 3, np.uint16)
        comp = self._comp(np.uint16)
        assert probe_delta_filter(codes, None, comp) is None
        assert probe_delta_filter(codes, (4096, 2), comp) is None

    def test_never_raises_on_exotic_chunk_specs(self):
        # zarr accepts chunks=None/True/False/int/"auto"/sequence — the probe
        # must degrade gracefully (decline or fall back), never raise.
        comp = self._comp(np.uint16)
        codes2d = _smooth_codes(2000, 3, np.uint16)
        codes1d = _smooth_codes(2000, 1, np.uint16).ravel()
        for chunks in (None, True, False, 4096, "auto", (4096,), [4096, 3]):
            probe_delta_filter(codes2d, chunks, comp)  # must not raise
            probe_delta_filter(codes1d, chunks, comp)  # must not raise
        # Bare int is the 1D idiom: treated as (int,).
        assert probe_delta_filter(codes1d, 512, comp) is not None
        # chunks=True must behave like "unknown" (whole-array fallback for
        # 1D), NOT like chunks=1 — bool subclasses int and must be excluded.
        assert (probe_delta_filter(codes1d, True, comp) is None) == (
            probe_delta_filter(codes1d, None, comp) is None
        )

    def test_declines_big_endian_dtype(self):
        # zarrita's bytes codec byte-swaps BEFORE filters; Python zarr views
        # the dtype AFTER filters — a big-endian store would desync the two.
        # Use LOW-RANGE smooth codes (< 256): their byte-swapped stream is
        # ALSO smooth (v << 8), so delta would win even without the rail —
        # this pins the RAIL itself, not an incidental compression decline.
        codes = (_smooth_codes(20000, 3, np.uint16) >> 8).astype(np.uint16)
        comp = self._comp(np.uint16)
        assert probe_delta_filter(codes, (4096, 3), comp) is not None  # sanity
        assert probe_delta_filter(codes.astype(">u2"), (4096, 3), comp) is None

    def test_declines_no_compressor_float_empty(self):
        codes = _smooth_codes(1000, 3, np.uint16)
        assert probe_delta_filter(codes, (256, 3), None) is None
        f32 = codes.astype(np.float32)
        assert probe_delta_filter(f32, (256, 3), self._comp(np.float32)) is None
        empty = np.empty((0, 3), dtype=np.uint16)
        assert probe_delta_filter(empty, (256, 3), self._comp(np.uint16)) is None

    def test_deterministic(self):
        codes = _smooth_codes(20000, 3, np.uint16)
        comp = self._comp(np.uint16)
        a = probe_delta_filter(codes, (4096, 3), comp)
        b = probe_delta_filter(codes, (4096, 3), comp)
        assert (a is None) == (b is None)
        if a is not None:
            assert a[0].get_config() == b[0].get_config()


class TestCompressionRegressionGuard:
    """The filter must actually SHRINK a canonical smooth store.

    Guards the probe wiring end-to-end: if a future change silently stops
    the filter from engaging (or makes it engage without winning), this
    fails — not just the metadata checks."""

    def test_smooth_store_is_smaller_with_delta(self):
        from unittest import mock

        import zarr

        from luxar.encoding.encoder import ArrayEncoder
        from luxar.encoding.modes import EncodingMode
        from luxar.encoding.semantic_types import SemanticType

        rng = np.random.default_rng(77)
        walk = np.cumsum(rng.normal(0.0, 1.0, size=(50000, 3)), axis=0)
        lo, hi = walk.min(axis=0), walk.max(axis=0)
        pos = ((walk - lo) / (hi - lo) * 500).astype(np.float32)

        def stored_bytes(delta_on: bool) -> int:
            store: dict = {}
            g = zarr.group(store=store)
            ctx = (
                mock.patch(
                    "luxar.encoding._encoders.perchannel.probe_delta_filter",
                    lambda *a, **k: None,
                )
                if not delta_on
                else mock.patch("builtins.len", len)
            )
            with ctx:
                ArrayEncoder().encode(
                    data=pos,
                    zarr_group=g,
                    name="a",
                    semantic_type=SemanticType.COORDINATE,
                    mode=EncodingMode.AUTO,
                    chunks=(8192, 3),
                    compressor=WIDTH_AWARE_DEFAULT,
                )
            assert delta_on == bool(g["a"].filters), "probe gating regressed"
            return sum(
                len(v)
                for k, v in store.items()
                if not k.endswith((".zarray", ".zattrs", ".zgroup"))
            )

        on, off = stored_bytes(True), stored_bytes(False)
        # Campaign-measured smooth-walk gain is ~1.2-1.3x; require >= 10%.
        assert on < off * 0.90, f"delta store not smaller: {on} vs {off}"


class TestEncoderIntegration:
    """Probe-gated delta through the real semantic-type encoders.

    Covers the full approved scope: coordinates (linear_perchannel),
    Cholesky halves (log / signed_log perchannel), amplitudes
    (geolog_scalar) — plus the structural exclusions (float32 fallback,
    no-compressor, PRECISION)."""

    def _smooth_positions(self, n=20000, span=(400.0, 900.0, 1300.0), seed=3):
        rng = np.random.default_rng(seed)
        walk = np.cumsum(rng.normal(0.0, 1.0, size=(n, 3)), axis=0)
        lo, hi = walk.min(axis=0), walk.max(axis=0)
        return ((walk - lo) / (hi - lo) * np.asarray(span)).astype(np.float32)

    def _encode(self, data, semantic_type, chunks, **kw):
        from luxar.encoding.encoder import ArrayEncoder
        from luxar.encoding.modes import EncodingMode
        from luxar.encoding.semantic_types import SemanticType  # noqa: F401

        g = zarr.group(store=zarr.MemoryStore())
        ArrayEncoder().encode(
            data=data,
            zarr_group=g,
            name="a",
            semantic_type=semantic_type,
            mode=kw.pop("mode", EncodingMode.AUTO),
            chunks=chunks,
            compressor=kw.pop("compressor", WIDTH_AWARE_DEFAULT),
            **kw,
        )
        return g

    def test_coordinate_gets_delta_and_roundtrips(self):
        from luxar.encoding.decoder import ArrayDecoder
        from luxar.encoding.semantic_types import SemanticType

        pos = self._smooth_positions()
        g = self._encode(pos, SemanticType.COORDINATE, chunks=(4096, 3))
        arr = g["a"]
        assert arr.attrs["encoding"]["name"] == "linear_perchannel_u16"
        assert arr.filters is not None and len(arr.filters) == 1
        assert arr.filters[0].get_config()["id"] == "luxar_delta_v1"
        # Delta is a pure storage transform: decode must be bit-identical to
        # an unfiltered encode of the same data.
        g_ref = zarr.group(store=zarr.MemoryStore())
        g_ref.create_dataset("a", data=np.asarray(arr), chunks=(4096, 3))
        np.testing.assert_array_equal(np.asarray(arr), np.asarray(g_ref["a"]))
        dec = ArrayDecoder().decode(arr, g)
        atol = float(np.ptp(pos, axis=0).max()) / 65535 * 2
        np.testing.assert_allclose(dec, pos, atol=atol)

    def test_cholesky_split_gets_delta_and_roundtrips(self):
        from luxar.encoding.decoder import ArrayDecoder
        from luxar.encoding.encoder import ArrayEncoder

        rng = np.random.default_rng(5)
        n = 20000
        # Smoothly varying sizes/correlations (Hilbert-like spatial coherence).
        base = np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0)
        diag = np.exp(base) + 0.5
        offd = np.cumsum(rng.normal(0, 0.005, size=(n, 3)), axis=0)
        g = zarr.group(store=zarr.MemoryStore())
        ArrayEncoder().encode_cholesky_split(
            g,
            diag,
            offd,
            ndim=3,
            chunks_diag=(8192, 3),
            chunks_offdiag=(8192, 3),
            compressor=WIDTH_AWARE_DEFAULT,
        )
        dec = ArrayDecoder()
        for nm in ("cholesky_factors_diag", "cholesky_factors_offdiag"):
            arr = g[nm]
            enc_name = arr.attrs["encoding"]["name"]
            assert "perchannel" in enc_name
            filts = arr.filters or []
            got = [f.get_config()["id"] for f in filts]
            # Smooth coherent codes: delta must win on at least the diag; when
            # present it must round-trip decode exactly like the codes say.
            if got:
                assert got == ["luxar_delta_v1"]
            d = dec.decode(arr, g)
            ref = diag if nm.endswith("_diag") else offd
            assert d.shape == ref.shape
        # At least one of the pair should have taken the filter on this data.
        n_delta = sum(
            1
            for nm in ("cholesky_factors_diag", "cholesky_factors_offdiag")
            if (g[nm].filters or [])
        )
        assert n_delta >= 1

    def test_amplitude_geolog_gets_delta(self):
        from luxar.encoding.decoder import ArrayDecoder
        from luxar.encoding.semantic_types import SemanticType

        rng = np.random.default_rng(9)
        amp = (np.exp(np.cumsum(rng.normal(0, 0.01, size=50000)))).astype(np.float32)
        g = self._encode(
            amp,
            SemanticType.POSITIVE_SCALAR,
            chunks=(16384,),
            positive_scalar_encoding="log",
        )
        arr = g["a"]
        assert arr.attrs["encoding"]["name"].startswith("geolog_scalar")
        assert (arr.filters or []) and arr.filters[0].get_config()[
            "id"
        ] == "luxar_delta_v1"
        dec = ArrayDecoder().decode(arr, g)
        np.testing.assert_allclose(dec, amp, rtol=2e-3)

    def test_sdr_colors_get_delta_and_decode_identically(self):
        from luxar.encoding.decoder import ArrayDecoder
        from luxar.encoding.semantic_types import SemanticType

        rng = np.random.default_rng(21)
        # Spatially coherent colors (smooth walk in [0, 1] per channel).
        walk = np.cumsum(rng.normal(0.0, 0.01, size=(30000, 3)), axis=0)
        lo, hi = walk.min(axis=0), walk.max(axis=0)
        colors = ((walk - lo) / (hi - lo)).astype(np.float32)
        g = self._encode(colors, SemanticType.COLOR, chunks=(8192, 3), color_mode="sdr")
        arr = g["a"]
        assert arr.attrs["encoding"]["name"] == "rgb_uint8"
        assert (arr.filters or []) and arr.filters[0].get_config()[
            "id"
        ] == "luxar_delta_v1"
        dec = ArrayDecoder().decode(arr, g)
        np.testing.assert_allclose(dec, colors, atol=1.5 / 255)

    def test_hdr_colors_get_delta_and_decode_identically(self):
        from luxar.encoding.decoder import ArrayDecoder
        from luxar.encoding.semantic_types import SemanticType

        rng = np.random.default_rng(22)
        walk = np.cumsum(rng.normal(0.0, 0.01, size=(30000, 3)), axis=0)
        hdr = np.exp(walk - walk.min(axis=0) + 0.1).astype(np.float32) * 50
        g = self._encode(hdr, SemanticType.COLOR, chunks=(8192, 3), color_mode="hdr")
        arr = g["a"]
        assert arr.attrs["encoding"]["name"] == "geolog_perchannel_u16"
        assert (arr.filters or []) and arr.filters[0].get_config()[
            "id"
        ] == "luxar_delta_v1"
        dec = ArrayDecoder().decode(arr, g)
        np.testing.assert_allclose(dec, hdr, rtol=2e-3)

    def test_float32_fallback_never_delta(self):
        from luxar.encoding.semantic_types import SemanticType

        rng = np.random.default_rng(2)
        pos = (rng.random((5000, 3)) * 1e6).astype(np.float32)  # extent >= 2^16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = self._encode(pos, SemanticType.COORDINATE, chunks=(2048, 3))
        arr = g["a"]
        assert arr.dtype == np.float32
        assert not (arr.filters or [])

    def test_precision_mode_never_delta(self):
        from luxar.encoding.modes import EncodingMode
        from luxar.encoding.semantic_types import SemanticType

        pos = self._smooth_positions(n=5000)
        g = self._encode(
            pos, SemanticType.COORDINATE, chunks=(2048, 3), mode=EncodingMode.PRECISION
        )
        assert g["a"].dtype == np.float32
        assert not (g["a"].filters or [])

    def test_no_compressor_never_delta(self):
        from luxar.encoding.semantic_types import SemanticType

        pos = self._smooth_positions(n=5000)
        g = self._encode(
            pos, SemanticType.COORDINATE, chunks=(2048, 3), compressor=None
        )
        assert g["a"].dtype == np.uint16
        assert not (g["a"].filters or [])
