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

from luxar._zarr_compat import create_array, memory_group
from luxar.encoding._encoders.perchannel import COORDINATE_LEVELS, gridded_axis_step
from luxar.encoding._encoders.structural import LUT_SCALAR_MAX_DISTINCT
from luxar.encoding.decoder import ArrayDecoder, decode_coordinate_columns
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


def _roundtrip(data, mode, float16_allowed=False):
    """Encode `data` as COORDINATE then decode; return (decoded, encoding, stored)."""
    enc = ArrayEncoder(float16_allowed=float16_allowed)
    g = memory_group()
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

    def test_int_input_decodes_float32_no_truncation(self):
        # Decode contract: COORDINATE always decodes to float32 regardless of the
        # input dtype. Regression: original_dtype used to echo the raw input dtype,
        # so int32 voxel coordinates were dequantized to floats then cast BACK to
        # int32 — truncating toward zero (99.9998 → 99) and breaking idempotency.
        rng = np.random.default_rng(20)
        pos = rng.integers(0, 2000, size=(1500, 3)).astype(np.int32)
        decoded, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert enc["name"] == "linear_perchannel_u16"
        assert enc["original_dtype"] == "float32"  # pinned, not the input dtype
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, pos.astype(np.float32), atol=_atol(pos))

    def test_float64_input_decodes_float32(self):
        # float64 input (e.g. un-cast np.random.randn) must not silently widen
        # every downstream consumer: decode returns float32, same as PRECISION.
        rng = np.random.default_rng(21)
        pos = rng.standard_normal((800, 3)) * 50  # float64
        decoded, enc, _ = _roundtrip(pos, EncodingMode.AUTO)
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, pos.astype(np.float32), atol=_atol(pos))

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
        hand = (
            lo + stored.astype(np.float64) / levels * np.maximum(hi - lo, 1e-30)
        ).astype(np.float32)
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
        g = memory_group()
        create_array(g, "a", data=u)
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

    def test_coordinate_column_fallback_resolves_array_reference_to_lut(self):
        group = memory_group()
        target = create_array(
            group,
            "target",
            data=np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8),
        )
        target.attrs["encoding"] = {
            "name": "lut_uint8",
            "lut": [10.0, 20.0],
            "original_dtype": "float32",
        }
        reference = create_array(
            group, "reference", data=np.empty((0, 4), dtype=np.uint8)
        )
        reference.attrs["encoding"] = {"name": "array_ref", "target": "target"}

        decoded = decode_coordinate_columns(reference, [3], group)

        np.testing.assert_array_equal(decoded, [[20.0], [10.0]])

    def test_decoder_rejects_non_finite_scales(self):
        rng = np.random.default_rng(12)
        vals = (rng.standard_normal((50, 3)) * 5).astype(np.float32)
        g = memory_group()
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
        g = memory_group()
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


class TestCoordinateRoundTripSlack:
    """``coordinate_round_trip_slack``: one test per exit, plus the invariant.

    The chunk-bounds writers pad their bounds by this answer, so an answer that
    is too small silently drops geometry the reader can no longer find
    (issue #1655). Every exit is pinned EXACTLY — the closed form
    ``extent / 131070``, not an approximation.
    """

    def test_precision_mode_is_exact(self):
        rng = np.random.default_rng(100)
        pos = (rng.random((500, 3)) * 1000.0).astype(np.float32)
        assert (
            ArrayEncoder().coordinate_round_trip_slack(pos, EncodingMode.PRECISION)
            is None
        )

    def test_empty_array_is_exact(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        assert (
            ArrayEncoder().coordinate_round_trip_slack(empty, EncodingMode.AUTO) is None
        )

    def test_non_2d_array_is_out_of_scope_not_exact(self):
        """``None`` here means "not the (N, d) shape this predicate covers".

        It is emphatically NOT a claim of exactness: a 1-D COORDINATE array is
        stored as ``linear_perchannel_u16`` like any other and moves by the full
        half-quantum. Pinned below so nobody reads the ``None`` as a guarantee.
        The bound builders all require ``(N, d)`` and never see this shape.
        """
        flat = np.linspace(0.0, 1000.0, 5000, dtype=np.float32)
        assert (
            ArrayEncoder().coordinate_round_trip_slack(flat, EncodingMode.AUTO) is None
        ), "the (N, d) shape guard must not answer for a 1-D array"
        decoded, encoding, _stored = _roundtrip(flat, EncodingMode.AUTO)
        assert encoding["name"] == "linear_perchannel_u16"
        moved = float(
            np.abs(decoded.astype(np.float64) - flat.astype(np.float64)).max()
        )
        assert moved > 1e-3, (
            "a 1-D COORDINATE array really is quantised — the None above is a "
            f"scope guard, not an exactness claim (max |Δ| = {moved:.4g})"
        )

    def test_non_finite_input_returns_none(self):
        """NaN/inf in, ``None`` out — never a NaN slack entry.

        A NaN entry would reach ``_normalise_coord_slack`` and be rejected with
        a "must be finite" message pointing at the CALLER rather than the data.
        Unreachable through the compiler (its fail-fast position gate rejects
        non-finite coordinates first), but this is public API.
        """
        enc = ArrayEncoder()
        for bad in (np.nan, np.inf, -np.inf):
            pos = np.array([[0.0, 1.0], [bad, 2.0]], dtype=np.float64)
            assert enc.coordinate_round_trip_slack(pos, EncodingMode.AUTO) is None

    def test_extent_at_or_above_the_u16_rail_is_exact(self):
        """At/above 2**16 the encoder falls back to float32 — nothing moves."""
        rng = np.random.default_rng(101)
        pos = (rng.random((500, 3)) * 70000.0).astype(np.float32)
        assert float(np.ptp(pos, axis=0).max()) >= 65536.0
        assert (
            ArrayEncoder().coordinate_round_trip_slack(pos, EncodingMode.AUTO) is None
        )

    def test_lut_eligible_array_is_exact(self):
        """A LUT stores the values verbatim, so nothing moves.

        The values are deliberately NOT on a regular grid, so the per-axis grid
        exit cannot be what answers here — it is the LUT exit or nothing.
        """
        palette = np.array([0.0, 0.37, 1.9, 5.5, 13.25, 61.0, 199.5, 800.0])
        rng = np.random.default_rng(102)
        pos = palette[rng.integers(0, palette.size, (600, 3))].astype(np.float32)
        enc = ArrayEncoder()
        assert enc.encodes_as_lut(pos, SemanticType.COORDINATE)
        arr = pos.astype(np.float64)
        for c in range(3):
            lo = float(arr[:, c].min())
            extent = float(arr[:, c].max()) - lo
            assert gridded_axis_step(arr[:, c], lo, extent, COORDINATE_LEVELS) is None
        assert enc.coordinate_round_trip_slack(pos, EncodingMode.AUTO) is None

    def test_constant_axis_is_zero_next_to_a_moving_sibling(self):
        rng = np.random.default_rng(103)
        pos = np.empty((5000, 3), dtype=np.float32)
        pos[:, 0] = 7.5  # constant: every value maps to level 0, decodes to lo
        pos[:, 1] = (rng.random(5000) * 1000.0).astype(np.float32)
        pos[:, 2] = 7.5
        slack = ArrayEncoder().coordinate_round_trip_slack(pos, EncodingMode.AUTO)
        assert slack is not None
        assert slack[0] == 0.0 and slack[2] == 0.0
        extent = float(pos[:, 1].max()) - float(pos[:, 1].min())
        assert slack[1] == extent / 131070.0

    def test_gridded_axis_is_zero(self):
        """A stacked integer time axis is grid-snapped, so it round-trips exactly."""
        rng = np.random.default_rng(104)
        pos = np.empty((5000, 2), dtype=np.float32)
        pos[:, 0] = rng.integers(0, 100, 5000)  # gridded
        pos[:, 1] = (rng.random(5000) * 1000.0).astype(np.float32)
        slack = ArrayEncoder().coordinate_round_trip_slack(pos, EncodingMode.AUTO)
        assert slack is not None
        assert slack[0] == 0.0
        assert slack[1] > 0.0

    @pytest.mark.parametrize("mode", [EncodingMode.AUTO, EncodingMode.MEMORY])
    def test_plain_random_axis_is_exactly_half_a_quantum(self, mode):
        rng = np.random.default_rng(105)
        pos = np.column_stack(
            [rng.random(5000) * 1000.0, rng.random(5000) * 4.0 - 2.0]
        ).astype(np.float32)
        slack = ArrayEncoder().coordinate_round_trip_slack(pos, mode)
        assert slack is not None
        assert slack.dtype == np.float64 and slack.shape == (2,)
        arr = pos.astype(np.float64)
        for c in range(2):
            extent = float(arr[:, c].max()) - float(arr[:, c].min())
            assert slack[c] == extent / 131070.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_slack_bounds_the_measured_round_trip_error(self, seed):
        """The invariant the whole chunk-bounds fix rests on.

        Encode and decode through the encoder's OWN path and check that no
        value moved further than the predicate promised, per axis.

        Two decodes are checked, because the COORDINATE decode contract is
        float32 while the quantization itself is float64:

        * The float64 dequantization is bounded by the slack up to float64
          rounding in the encode/decode chain. The closed form
          ``extent / 2·levels`` is the exact half-quantum, but the replay that
          produces the decoded value is a few float64 operations, so the
          measured ratio can exceed 1 in the last bits (worst observed over 300
          random configs: 1.0000000008). Hence the ``1 + 1e-9`` below rather
          than a bare ``<=``; it is what the chunk-bounds pad is derived from,
          and containment is still safe because the float32 outward store below
          absorbs a relative 1e-9 at any magnitude.
        * The stored float32 decode can exceed it by up to half a float32 ULP
          at the coordinate's own magnitude (measured ~1.07e-5 past a 7.62e-3
          slack at |x| ~ 500). That does NOT leak into the stored bound: the
          bound is accumulated in float64 as ``max_authored + slack``, so it is
          ``>=`` the float64 decode of every value in the chunk, and float32
          conversion is monotone — narrowing both ends OUTWARD (which the bound
          builders do, never inward) preserves the ordering. The end-to-end
          proof is
          ``io/tests/_compiler/test_spatial_ordering.py``'s decoded-containment
          tests, which read the real stored bounds.
        """
        rng = np.random.default_rng(seed)
        pos = np.column_stack(
            [
                rng.random(3000) * 1000.0 - 500.0,
                rng.random(3000) * 13.75 + 2.0,
                rng.standard_normal(3000) * 0.5,
            ]
        ).astype(np.float32)
        slack = ArrayEncoder().coordinate_round_trip_slack(pos, EncodingMode.AUTO)
        assert slack is not None
        decoded, encoding, stored = _roundtrip(pos, EncodingMode.AUTO)
        assert encoding["name"] == "linear_perchannel_u16"

        # (a) the float64 dequantization, replayed from the stored scales.
        lo = np.asarray(encoding["col_lo"], dtype=np.float64)
        hi = np.asarray(encoding["col_hi"], dtype=np.float64)
        decoded64 = lo + stored.astype(np.float64) / 65535.0 * (hi - lo)
        moved64 = np.abs(decoded64 - pos.astype(np.float64)).max(axis=0)
        assert np.all(moved64 <= slack * (1.0 + 1e-9)), (
            f"moved64={moved64} slack={slack}"
        )
        # And the bound is tight enough to be useful: every axis really does
        # get within a few percent of it.
        assert np.all(moved64 >= 0.9 * slack)

        # (b) the stored float32 decode: slack + at most half a float32 ULP.
        half_ulp = np.spacing(np.abs(pos)).astype(np.float64) / 2.0
        moved32 = np.abs(decoded.astype(np.float64) - pos.astype(np.float64))
        assert np.all(moved32 <= slack + half_ulp)


# --------------------------------------------------------------------------
# The predicate <-> encoder coupling corpus (issue #1655).
# --------------------------------------------------------------------------


def _corpus() -> dict[str, np.ndarray]:
    """COORDINATE arrays covering every exit of both models, and every path
    the #1655 review exercised.

    Keyed by name so a failure names the shape that broke rather than an index.
    """
    rng = np.random.default_rng(20260821)
    cases: dict[str, np.ndarray] = {}

    # --- continuous (the plain quantised case, at three widths) -------------
    cases["continuous_d2"] = np.column_stack(
        [rng.random(2000) * 1000.0 - 500.0, rng.random(2000) * 4.0 - 2.0]
    ).astype(np.float32)
    cases["continuous_d3"] = np.column_stack(
        [
            rng.random(3000) * 1000.0,
            rng.random(3000) * 13.75 + 2.0,
            rng.standard_normal(3000) * 0.5,
        ]
    ).astype(np.float32)
    cases["continuous_d8"] = (
        rng.random((1500, 8))
        * np.array([1000.0, 1.0, 0.01, 250.0, 7.0, 3e3, 0.5, 90.0])
    ).astype(np.float32)

    # --- LUT-eligible, IRREGULAR palette ------------------------------------
    # The load-bearing LUT case: the palette is deliberately NOT a regular
    # grid, so with `allow_lut=False` (the lines writer) NO exact exit applies
    # and the predicate MUST report slack. A linspace palette would grid-snap
    # and pass for the wrong reason -- which is exactly how the first attempt
    # at this fix shipped a wrong LUT exit for lines.
    palette = np.array([0.0, 0.37, 1.9, 5.5, 13.25, 61.0, 199.5, 800.0])
    cases["lut_irregular_palette_d3"] = palette[
        rng.integers(0, palette.size, (600, 3))
    ].astype(np.float32)
    # ... and its trap twin, kept so the difference stays visible: a regular
    # palette is exact through the GRID exit whether or not a LUT is allowed.
    regular = np.linspace(0.0, 700.0, 8)
    cases["lut_regular_palette_d3"] = regular[
        rng.integers(0, regular.size, (600, 3))
    ].astype(np.float32)

    # --- gridded / constant / mixed ----------------------------------------
    cases["gridded_int_axis_d3"] = np.column_stack(
        [
            rng.integers(0, 100, 4000),  # stacked time axis: grid-snapped
            rng.random(4000) * 1000.0,
            rng.random(4000) * 2.0,
        ]
    ).astype(np.float32)
    cases["all_gridded_d2"] = np.column_stack(
        [rng.integers(0, 100, 3000), rng.integers(-40, 41, 3000) * 0.25]
    ).astype(np.float32)
    const = np.empty((2500, 3), dtype=np.float32)
    const[:, 0] = 7.5
    const[:, 1] = (rng.random(2500) * 1000.0).astype(np.float32)
    const[:, 2] = -3.25
    cases["constant_axes_d3"] = const
    mixed = np.empty((2500, 4), dtype=np.float32)
    mixed[:, 0] = rng.integers(0, 64, 2500)  # gridded
    mixed[:, 1] = (rng.random(2500) * 300.0).astype(np.float32)  # continuous
    mixed[:, 2] = 42.0  # constant
    mixed[:, 3] = (rng.standard_normal(2500) * 0.001).astype(np.float32)  # tiny extent
    cases["mixed_gridded_continuous_constant_d4"] = mixed

    # --- the extent rail, from both sides -----------------------------------
    under = np.column_stack(
        [rng.random(3000) * 65000.0, rng.random(3000) * 5.0]
    ).astype(np.float32)
    under[0, 0] = 0.0
    under[1, 0] = 65000.0  # extent just UNDER COORDINATE_U16_MAX_EXTENT
    cases["extent_just_under_rail_d2"] = under
    at = under.copy()
    at[1, 0] = 65536.0  # extent exactly AT the rail -> float32 fallback
    cases["extent_at_rail_d2"] = at
    over = np.column_stack(
        [
            rng.random(2000) * 70000.0,
            rng.random(2000) * 12.0,
            rng.random(2000) * 0.3,
        ]
    ).astype(np.float32)
    cases["extent_over_rail_d3"] = over

    # --- large offsets -------------------------------------------------------
    # float32 at 1e6 has a 0.0625 ULP, at 2**23 exactly 1.0, so the AUTHORED
    # array is what decides whether the axis reads as gridded. Both dtypes are
    # here on purpose: the float32 forms land on the integer grid (exact), the
    # float64 forms do not (quantised, and the float32 decode cast is then
    # LARGER than the slack -- the case the ULP allowance exists for).
    cases["offset_1e6_f32_d3"] = (
        1e6 + rng.random((2500, 3)) * np.array([1000.0, 40.0, 3.0])
    ).astype(np.float32)
    cases["offset_1e6_f64_d3"] = 1e6 + rng.random((2500, 3)) * np.array(
        [1000.0, 40.0, 3.0]
    )
    cases["offset_2p23_f32_d3"] = (
        float(2**23) + rng.random((2500, 3)) * np.array([1000.0, 40.0, 3.0])
    ).astype(np.float32)
    cases["offset_2p23_f64_d3"] = float(2**23) + rng.random((2500, 3)) * np.array(
        [1000.0, 40.0, 3.0]
    )
    return cases


_CORPUS = _corpus()

#: The exact encodings a COORDINATE array can legitimately end up in. Recorded
#: (not just asserted on values) so a new encoder exit shows up as an unknown
#: name here rather than as a mystery numeric failure.
_EXACT_ENCODINGS = {"float32", "lut_uint8", "lut_uint16"}


class TestPredicateAgreesWithTheRealEncoder:
    """The MECHANISM keeping ``coordinate_round_trip_slack`` and
    ``_encode_coordinate`` in step (issue #1655).

    The chunk-bounds writers now carry a SECOND, independent model of what the
    COORDINATE encoder does — four replayed exits (mode, grid snap, extent
    rail, LUT). Prose in a docstring is not a mechanism: nothing fails if
    :meth:`~luxar.encoding._encoders.perchannel.PerChannelEncoderMixin._encode_coordinate`
    grows a fifth exit, and the failure mode is a chunk bound the decoded
    coordinates fall outside of — geometry the viewer never fetches, with
    nothing to notice. (This PR's own history is the evidence: the LUT exit was
    wrong for lines on the first attempt, because a LUT-eligible lines vertices
    array is written with ``allow_lut=False`` and quantised after all.)

    So for every corpus array × mode × ``allow_lut`` this asks the predicate
    AND performs a real encode into an in-memory zarr group followed by a real
    :class:`ArrayDecoder` decode, then holds the two against each other in BOTH
    directions:

    * ``slack[axis] > 0``  ⇒ nothing on that axis moved further than
      ``slack[axis]``, plus at most half a float32 ULP at the coordinate's own
      magnitude (the decode contract is float32; see
      :meth:`TestCoordinateRoundTripSlack.test_slack_bounds_the_measured_round_trip_error`
      for why that extra term cannot leak into a stored bound).
    * ``slack[axis] == 0`` ⇒ that axis is stored EXACTLY: the decode equals the
      authored value cast to the decode dtype, bit for bit. This is the
      direction that catches a new lossy exit the predicate does not know
      about, and it is per-AXIS rather than per-array, because the predicate's
      whole point is that one wide continuous axis must not be allowed to
      excuse a wrong answer on the snapped axis beside it.
    * ``slack is None``    ⇒ every axis exact, by the same standard.

    The exactness standard is "equals the authored value cast to the decode
    dtype", not "equals the authored float64 value", because the COORDINATE
    decode contract IS float32 (``original_dtype`` is pinned) on every exit but
    the LUT one, which returns the input dtype. A float64 author therefore
    cannot get its input back from any exact path, and demanding it would fail
    the float32 rail — the honest guarantee is "the store added nothing on top
    of the float32 cast the format specifies", which is also exactly what the
    chunk bounds need: they are narrowed to float32 OUTWARD, and float32
    conversion is monotone.

    ``deduplicate``/``allow_lut`` mirror the two production call sites:
    ``allow_lut=True`` is the points path (``dataset_writers/positions.py``,
    both defaults), ``allow_lut=False`` the lines path
    (``geometry_writers/lines.py``, which passes both as ``False``).
    """

    @pytest.mark.parametrize("name", sorted(_CORPUS))
    @pytest.mark.parametrize(
        "mode", [EncodingMode.AUTO, EncodingMode.MEMORY, EncodingMode.PRECISION]
    )
    @pytest.mark.parametrize("allow_lut", [True, False])
    def test_slack_matches_a_real_encode_decode(self, name, mode, allow_lut):
        authored = _CORPUS[name]
        enc = ArrayEncoder()
        slack = enc.coordinate_round_trip_slack(authored, mode, allow_lut=allow_lut)

        group = memory_group()
        with warnings.catch_warnings():
            # The extent rail warns by design; this test is about values.
            warnings.simplefilter("ignore", UserWarning)
            enc.encode(
                data=authored,
                zarr_group=group,
                name="c",
                semantic_type=SemanticType.COORDINATE,
                mode=mode,
                allow_lut=allow_lut,
                # Mirrors the production call sites: the lines writer passes
                # deduplicate=False alongside allow_lut=False.
                deduplicate=allow_lut,
            )
        encoding_name = dict(group["c"].attrs["encoding"])["name"]
        decoded = ArrayDecoder().decode(group["c"], group)

        assert decoded.shape == authored.shape
        exact_reference = authored.astype(decoded.dtype)
        moved = np.abs(decoded.astype(np.float64) - authored.astype(np.float64))
        # Half a float32 ULP at each coordinate's own magnitude. Computed from
        # the float32 view of the author, since that is the grid the decode
        # lands on.
        half_ulp = (
            np.spacing(np.abs(authored).astype(np.float32)).astype(np.float64) / 2.0
        )

        if slack is None:
            assert np.array_equal(decoded, exact_reference), (
                f"{name}/{mode}/allow_lut={allow_lut}: the predicate promised an "
                f"EXACT store but {encoding_name} moved values by up to "
                f"{moved.max():.6g}"
            )
            assert encoding_name in _EXACT_ENCODINGS or encoding_name.startswith(
                "linear_perchannel_"
            ), f"unrecognised encoding {encoding_name!r} — teach this test about it"
            return

        assert slack.shape == (authored.shape[1],) and slack.dtype == np.float64
        for axis in range(authored.shape[1]):
            if slack[axis] == 0.0:
                assert np.array_equal(decoded[:, axis], exact_reference[:, axis]), (
                    f"{name}/{mode}/allow_lut={allow_lut} axis {axis}: the "
                    f"predicate promised an EXACT axis but {encoding_name} moved "
                    f"it by up to {moved[:, axis].max():.6g}"
                )
            else:
                budget = slack[axis] + half_ulp[:, axis]
                assert np.all(moved[:, axis] <= budget), (
                    f"{name}/{mode}/allow_lut={allow_lut} axis {axis}: "
                    f"{encoding_name} moved a value by "
                    f"{moved[:, axis].max():.6g}, past the promised "
                    f"{slack[axis]:.6g} (+ half a float32 ULP)"
                )

    def test_the_corpus_actually_reaches_every_exit(self):
        """A corpus that quietly stopped covering an exit is not a mechanism.

        Pins that each of the four replayed exits, and the quantised
        non-exit, is genuinely represented — so a future edit that (say)
        makes every array grid-snap cannot leave the test passing vacuously.
        """
        enc = ArrayEncoder()
        seen = set()
        for name, authored in _CORPUS.items():
            arr = authored.astype(np.float64)
            extent = float(np.ptp(arr, axis=0).max())
            if extent >= 65536.0:
                seen.add("extent_rail")
            if enc.encodes_as_lut(authored, SemanticType.COORDINATE):
                seen.add("lut")
            if (np.ptp(arr, axis=0) == 0.0).any():
                seen.add("constant_axis")
            slack_lut = enc.coordinate_round_trip_slack(
                authored, EncodingMode.AUTO, allow_lut=True
            )
            slack_no_lut = enc.coordinate_round_trip_slack(
                authored, EncodingMode.AUTO, allow_lut=False
            )
            if slack_no_lut is not None and (slack_no_lut == 0.0).any():
                seen.add("gridded_axis")
            if slack_no_lut is not None and (slack_no_lut > 0.0).any():
                seen.add("quantised")
            if slack_lut is None and slack_no_lut is not None:
                # The exit that was wrong on the first attempt: LUT-eligible,
                # but NOT exact once the write forbids a LUT.
                seen.add("lut_only_when_allowed")
            assert (
                enc.coordinate_round_trip_slack(authored, EncodingMode.PRECISION)
                is None
            ), f"{name}: PRECISION must always be exact"
            seen.add("precision")
        assert seen == {
            "extent_rail",
            "lut",
            "constant_axis",
            "gridded_axis",
            "quantised",
            "lut_only_when_allowed",
            "precision",
        }, f"corpus coverage regressed: {sorted(seen)}"


class TestLutProbeShortCircuit:
    """The premise of ``coordinate_round_trip_slack``'s LUT-probe skip.

    The predicate skips the (whole-array, dominant-cost) ``encodes_as_lut``
    probe when a single COLUMN already holds more distinct values than
    :data:`LUT_SCALAR_MAX_DISTINCT`. That is sound only because a COORDINATE
    array always takes ``_lut_plan``'s SCALAR mode (row mode needs a
    ≤4-channel 2-D COLOR) and scalar mode is capped at the uint8 tier — the
    uint16 tier is row mode only. If either ever changes, the skip starts
    reporting slack for arrays that really are stored verbatim, i.e. bounds
    wider than they need to be — harmless, but the tests below exist so the
    change is noticed rather than absorbed.
    """

    @staticmethod
    def _palette_array(k: int, rows: int = 3000, seed: int = 7) -> np.ndarray:
        """(rows, 3) float32 drawn from exactly ``k`` distinct values."""
        rng = np.random.default_rng(seed)
        palette = np.sort(rng.random(k) * 1000.0)
        flat = rng.integers(0, k, rows * 3)
        flat[:k] = np.arange(k)  # every palette entry really appears
        return palette[flat.reshape(rows, 3)].astype(np.float32)

    def test_scalar_lut_cap_is_the_uint8_tier(self):
        """256 distinct values LUT-encode; 257 do not. The uint16 tier is
        unreachable for COORDINATE, so the cap is 256 and NOT 65,536."""
        enc = ArrayEncoder()
        assert LUT_SCALAR_MAX_DISTINCT == 256
        at_cap = self._palette_array(LUT_SCALAR_MAX_DISTINCT)
        over_cap = self._palette_array(LUT_SCALAR_MAX_DISTINCT + 1)
        assert np.unique(at_cap).size == LUT_SCALAR_MAX_DISTINCT
        assert np.unique(over_cap).size == LUT_SCALAR_MAX_DISTINCT + 1
        assert enc.encodes_as_lut(at_cap, SemanticType.COORDINATE)
        assert not enc.encodes_as_lut(over_cap, SemanticType.COORDINATE)

    def test_lut_eligible_array_still_answers_none(self):
        """The short-circuit must not break the LUT exit it guards.

        An irregular palette at the cap has NO other exact exit (no axis is
        gridded), so ``None`` here can only be the LUT exit still firing.
        """
        enc = ArrayEncoder()
        pos = self._palette_array(200)
        assert enc.encodes_as_lut(pos, SemanticType.COORDINATE)
        arr = pos.astype(np.float64)
        for c in range(3):
            lo = float(arr[:, c].min())
            assert (
                gridded_axis_step(
                    arr[:, c], lo, float(arr[:, c].max()) - lo, COORDINATE_LEVELS
                )
                is None
            )
        assert enc.coordinate_round_trip_slack(pos, EncodingMode.AUTO) is None
        # ...and with the write forbidding a LUT, it is quantised after all.
        assert (
            enc.coordinate_round_trip_slack(pos, EncodingMode.AUTO, allow_lut=False)
            is not None
        )

    def test_short_circuit_does_not_change_any_answer(self):
        """Over the whole corpus, skipping the probe answers identically.

        Replays the pre-short-circuit logic — always ask ``encodes_as_lut``
        when ``allow_lut`` — and requires the same result everywhere. This is
        the property the optimisation claims, checked rather than argued.
        """
        enc = ArrayEncoder()
        for name, authored in {
            **_CORPUS,
            "palette_at_cap": self._palette_array(256),
            "palette_over_cap": self._palette_array(257),
            # Per-axis counts under the cap, whole-array count over it: the
            # one shape the short-circuit cannot help with, and the one most
            # likely to be answered wrongly if the cap were per-axis-only.
            "disjoint_palettes_d3": np.column_stack(
                [self._palette_array(200, seed=s)[:, 0] for s in (1, 2, 3)]
            ),
        }.items():
            for mode in (EncodingMode.AUTO, EncodingMode.MEMORY):
                for allow_lut in (True, False):
                    got = enc.coordinate_round_trip_slack(
                        authored, mode, allow_lut=allow_lut
                    )
                    want = _slack_without_short_circuit(enc, authored, mode, allow_lut)
                    if want is None or got is None:
                        assert (got is None) == (want is None), (
                            f"{name}/{mode}/allow_lut={allow_lut}: "
                            f"short-circuit changed the answer"
                        )
                    else:
                        np.testing.assert_array_equal(got, want)

    def test_probe_is_skipped_for_continuous_data(self):
        """The optimisation actually fires (and is not silently dead).

        Counts ``_lut_plan`` calls rather than timing anything: a wall-clock
        assertion on a shared machine is a flake generator, and "the probe did
        not run" is the whole claim.
        """
        calls = []
        real_plan = ArrayEncoder._lut_plan

        class CountingEncoder(ArrayEncoder):
            def _lut_plan(self, data, semantic_type):
                calls.append(semantic_type)
                return real_plan(self, data, semantic_type)

        enc = CountingEncoder()
        rng = np.random.default_rng(11)
        continuous = (rng.random((5000, 3)) * 1000.0).astype(np.float32)
        assert (
            enc.coordinate_round_trip_slack(continuous, EncodingMode.AUTO) is not None
        )
        assert calls == [], "the whole-array LUT probe should have been skipped"

        # ...but a genuinely LUT-shaped array still pays for the probe.
        assert (
            enc.coordinate_round_trip_slack(self._palette_array(200), EncodingMode.AUTO)
            is None
        )
        assert calls == [SemanticType.COORDINATE]


def _slack_without_short_circuit(enc, data, mode, allow_lut):
    """The predicate as it read BEFORE the LUT-probe short-circuit.

    Kept as a local replay rather than a git archaeology exercise so
    :meth:`TestLutProbeShortCircuit.test_short_circuit_does_not_change_any_answer`
    can compare against something executable.
    """
    if mode not in (EncodingMode.AUTO, EncodingMode.MEMORY):
        return None
    arr = np.asarray(data).astype(np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    if float((hi - lo).max()) >= 65536.0:
        return None
    slack = np.zeros(arr.shape[1], dtype=np.float64)
    for axis in range(arr.shape[1]):
        extent = float(hi[axis] - lo[axis])
        if extent <= 0.0:
            continue
        if (
            gridded_axis_step(arr[:, axis], float(lo[axis]), extent, COORDINATE_LEVELS)
            is not None
        ):
            continue
        slack[axis] = extent / (2.0 * COORDINATE_LEVELS)
    if not slack.any():
        return None
    if allow_lut and enc.encodes_as_lut(data, SemanticType.COORDINATE):
        return None
    return slack
