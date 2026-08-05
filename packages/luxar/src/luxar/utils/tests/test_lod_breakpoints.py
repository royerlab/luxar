"""Tests for the shared streaming-ladder breakpoint math.

These cover the cut geometry that all three geometries (Points, Lines, GSplats)
now derive from an identical ``stream:<c>`` spec. The GSplats-side regression
guard lives in ``gsplats/lod/tests/test_additive.py`` and must keep passing
unmodified — that is what proves the delegation is behavior-preserving.
"""

from __future__ import annotations

import pytest

from luxar.utils.lod_breakpoints import (
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_STREAM_MAX_LEVELS,
    parse_stream_chunk,
    sibling_aware_stream_breakpoints,
    stream_cuts,
    streaming_chunk_splats,
    validate_element_breakpoints,
)


class TestStreamCuts:
    def test_geometric_doubling(self) -> None:
        # Increments are [8, 8, 16, 32, 36] — first paint costs one chunk, then
        # each refinement doubles the resident set.
        assert stream_cuts(100, 8) == [8, 16, 32, 64, 100]

    def test_n_at_or_below_chunk_is_a_single_level(self) -> None:
        assert stream_cuts(8, 8) == [8]
        assert stream_cuts(5, 8) == [5]

    @pytest.mark.parametrize("chunk", [1, 500])
    def test_empty_n_is_a_single_empty_level(self, chunk: int) -> None:
        # n == 0 falls in the `n <= chunk` branch, so the contract is a single
        # cut of 0 (one empty level) — NOT an empty list. A downstream ladder
        # builder can therefore always assume at least one cut.
        assert stream_cuts(0, chunk) == [0]

    def test_sliver_tail_folds_into_the_previous_cut(self) -> None:
        # Without the fold this would end [.., 64, 66] — a 2-element level.
        assert stream_cuts(66, 8) == [8, 16, 32, 66]

    def test_exact_power_of_two_multiple_needs_no_fold(self) -> None:
        assert stream_cuts(64, 8) == [8, 16, 32, 64]

    def test_level_cap_jumps_straight_to_n(self) -> None:
        cuts = stream_cuts(10**9, 1, max_levels=4)
        assert len(cuts) == 4
        assert cuts == [1, 2, 4, 10**9]

    def test_cuts_are_strictly_ascending_and_end_at_n(self) -> None:
        for n in (37, 1000, 9_751_955):
            cuts = stream_cuts(n, 40_000)
            assert cuts[-1] == n
            assert all(a < b for a, b in zip(cuts, cuts[1:]))


class TestParseStreamChunk:
    def test_parses_the_body(self) -> None:
        assert parse_stream_chunk("stream:14000") == 14000

    def test_rejects_zero_and_negative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_stream_chunk("stream:0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_stream_chunk("stream:-5")

    def test_rejects_non_integer_body(self) -> None:
        with pytest.raises(ValueError, match="stream:<c>"):
            parse_stream_chunk("stream:lots")


class TestValidateElementBreakpoints:
    def test_valid_specs_pass(self) -> None:
        validate_element_breakpoints("stream:14000")
        validate_element_breakpoints("energy:0.5,0.9,1.0")
        validate_element_breakpoints([100, 200, 400])

    def test_rejects_bad_stream_chunk(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_element_breakpoints("stream:0")
        with pytest.raises(ValueError, match="stream:<c>"):
            validate_element_breakpoints("stream:lots")

    def test_rejects_unknown_string(self) -> None:
        # 'equal-count' belongs to the GSplats vocabulary only; Points/Lines
        # express it via n_lods.
        with pytest.raises(ValueError, match="unrecognized breakpoints string"):
            validate_element_breakpoints("equal-count")

    def test_rejects_bad_energy_fractions(self) -> None:
        with pytest.raises(ValueError, match="must be numbers"):
            validate_element_breakpoints("energy:0.5,abc")
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_element_breakpoints("energy:")
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_element_breakpoints("energy: , ,")

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_element_breakpoints([])


class TestStreamingChunkSplats:
    def test_known_sizing(self) -> None:
        # 200 ms @ 25 Mbps @ 45 B/splat -> ~13.9 k splats (the gsplat default).
        assert streaming_chunk_splats(200.0, DEFAULT_BANDWIDTH_MBPS, 45.0) == 13889

    def test_scales_inversely_with_payload_size(self) -> None:
        # 4x the bytes per element -> a quarter of the elements, up to rounding.
        small = streaming_chunk_splats(200.0, 25.0, 16.0)
        large = streaming_chunk_splats(200.0, 25.0, 64.0)
        assert small == pytest.approx(4 * large, abs=4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(target_ms=0.0, bandwidth_mbps=25.0, bytes_per_splat=16.0),
            dict(target_ms=200.0, bandwidth_mbps=0.0, bytes_per_splat=16.0),
            dict(target_ms=200.0, bandwidth_mbps=25.0, bytes_per_splat=0.0),
        ],
    )
    def test_rejects_non_positive_inputs(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            streaming_chunk_splats(**kwargs)


class TestSiblingAwareStreamBreakpoints:
    def test_raises_the_base_to_half_the_sibling_size(self) -> None:
        # leaf_n / (2K) = 1_000_000 / 8 = 125_000 > the user's 39_062 base.
        assert (
            sibling_aware_stream_breakpoints("stream:39062", 1_000_000, 4)
            == "stream:125000"
        )

    def test_keeps_a_larger_user_base(self) -> None:
        assert sibling_aware_stream_breakpoints("stream:500000", 1_000_000, 4) == (
            "stream:500000"
        )

    def test_non_stream_specs_pass_through_untouched(self) -> None:
        assert (
            sibling_aware_stream_breakpoints("equal-count", 1_000, 4) == "equal-count"
        )
        assert sibling_aware_stream_breakpoints([10, 50], 1_000, 4) == [10, 50]
        assert sibling_aware_stream_breakpoints("energy:0.5,1.0", 1_000, 4) == (
            "energy:0.5,1.0"
        )

    def test_malformed_stream_spec_passes_through_rather_than_raising(self) -> None:
        assert sibling_aware_stream_breakpoints("stream:x", 1_000, 4) == "stream:x"

    def test_compression_factor_is_floored_at_two(self) -> None:
        # K=1 would make sibling_base == leaf_n/2 via the floor, not leaf_n.
        assert sibling_aware_stream_breakpoints("stream:1", 1_000, 1) == "stream:250"

    def test_default_level_cap_is_exposed(self) -> None:
        assert DEFAULT_STREAM_MAX_LEVELS >= 8
