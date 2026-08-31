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
    DEFAULT_CAPPED_FIRST_CHUNK,
    DEFAULT_MAX_ADDITIVE_COMMIT,
    DEFAULT_STREAM_MAX_LEVELS,
    capped_stream_cuts,
    hidden_coordinate_count,
    parse_stream_chunk,
    sibling_aware_stream_breakpoints,
    sliced_ladder_first_chunk,
    stream_cuts,
    streaming_chunk_splats,
    validate_element_breakpoints,
)


class TestHiddenCoordinateCount:
    def test_counts_occurring_combinations_not_cardinality_product(self) -> None:
        positions = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
        assert hidden_coordinate_count(positions, [0, 1]) == 3

    def test_empty_positions_or_columns_are_unsliced(self) -> None:
        assert hidden_coordinate_count([], [0]) == 1
        assert hidden_coordinate_count([[1.0, 2.0]], []) == 1

    def test_out_of_range_column_is_undeterminable(self) -> None:
        assert hidden_coordinate_count([[1.0, 2.0]], [2]) == 1


class TestSlicedLadderFirstChunk:
    def test_unsliced_preserves_the_download_budget(self) -> None:
        assert sliced_ladder_first_chunk(40, elements=800, slices=1) == 40

    def test_resident_share_floor_binds(self) -> None:
        assert sliced_ladder_first_chunk(40, elements=800, slices=2) == 100

    def test_download_budget_wins_above_the_floor(self) -> None:
        assert sliced_ladder_first_chunk(120, elements=800, slices=2) == 120

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"first_chunk": 0, "elements": 1, "slices": 1}, "first_chunk"),
            ({"first_chunk": 1, "elements": 0, "slices": 1}, "elements"),
            ({"first_chunk": 1, "elements": 1, "slices": 0}, "slices"),
            (
                {"first_chunk": 1, "elements": 1, "slices": 1, "max_depth": 0},
                "max_depth",
            ),
        ],
    )
    def test_rejects_non_positive_inputs(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            sliced_ladder_first_chunk(**kwargs)


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


class TestCappedStreamCuts:
    """The whole point of this schedule is the LARGEST INCREMENT, so that is
    what these assert — not the cut positions, which are incidental."""

    def test_largest_increment_is_capped_at_any_n(self) -> None:
        # The property `stream_cuts` cannot have: a pure doubling ladder's final
        # increment is always n/2, whatever the first chunk is.
        for n in (10_000, 1_153_506, 3_000_000, 6_248_730, 9_751_955, 82_000_000):
            cuts = capped_stream_cuts(n)
            increments = [b - a for a, b in zip([0, *cuts], cuts)]
            assert max(increments) <= DEFAULT_MAX_ADDITIVE_COMMIT, (
                f"n={n} largest increment {max(increments):,}"
            )

    def test_doubling_would_have_blown_the_cap(self) -> None:
        # Guards the motivation with the MEASURED figure, not the n/2 upper
        # bound: human_multiome's finest leaf commits 2,152,730 in one go under a
        # pure doubling ladder, twice the 1,000,000 check_demo_ladders fails at.
        n = 6_248_730
        plain = stream_cuts(n, DEFAULT_CAPPED_FIRST_CHUNK)
        plain_max = max(b - a for a, b in zip([0, *plain], plain))
        assert plain_max == 2_152_730
        capped = capped_stream_cuts(n)
        capped_max = max(b - a for a, b in zip([0, *capped], capped))
        assert capped_max <= DEFAULT_MAX_ADDITIVE_COMMIT
        assert capped_max * 2 < plain_max

    def test_doubling_tail_grows_with_n_but_capped_tail_does_not(self) -> None:
        # Why this is size-driven rather than a bad first chunk: shrinking the
        # chunk does not help the tail, but the cap bounds it at every n.
        for chunk in (500, 2_000, 40_000):
            assert (
                max(
                    b - a
                    for a, b in zip(
                        [0, *stream_cuts(20_000_000, chunk)],
                        stream_cuts(20_000_000, chunk),
                    )
                )
                > DEFAULT_MAX_ADDITIVE_COMMIT
            )
            capped = capped_stream_cuts(20_000_000, chunk=chunk)
            assert (
                max(b - a for a, b in zip([0, *capped], capped))
                <= DEFAULT_MAX_ADDITIVE_COMMIT
            )

    def test_geometric_head_is_preserved(self) -> None:
        # Cheap first paint is the other half of the contract: the early cuts
        # must still double, or time-to-first-pixel regresses.
        cuts = capped_stream_cuts(9_751_955)
        assert cuts[0] == DEFAULT_CAPPED_FIRST_CHUNK
        head = [c for c in cuts if c <= 1_024_000]
        assert head == [DEFAULT_CAPPED_FIRST_CHUNK * 2**i for i in range(len(head))]
        assert head[-1] == 1_024_000

    def test_cuts_are_strictly_ascending_and_end_at_n(self) -> None:
        for n in (1, 1_999, 2_000, 2_001, 500_000, 9_751_955):
            cuts = capped_stream_cuts(n)
            assert cuts[-1] == n
            assert all(a < b for a, b in zip(cuts, cuts[1:]))

    def test_n_at_or_below_chunk_is_a_single_level(self) -> None:
        assert capped_stream_cuts(2_000) == [2_000]
        assert capped_stream_cuts(5) == [5]

    def test_first_chunk_is_clamped_to_the_commit_ceiling(self) -> None:
        cuts = capped_stream_cuts(10_000_000, chunk=1_500_000, max_commit=900_000)
        increments = [b - a for a, b in zip([0, *cuts], cuts)]
        assert cuts[0] == 900_000
        assert max(increments) <= 900_000

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_n_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            capped_stream_cuts(bad)

    def test_a_small_max_commit_degenerates_to_equal_steps(self) -> None:
        # No geometric head survives when the cap is below the first chunk;
        # the result must still be a valid ascending ladder, not empty.
        cuts = capped_stream_cuts(10_000, chunk=100, max_commit=1_000)
        increments = [b - a for a, b in zip([0, *cuts], cuts)]
        assert max(increments) <= 1_000
        assert cuts[-1] == 10_000

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_knobs_raise(self, bad: int) -> None:
        with pytest.raises(ValueError, match="chunk must be >= 1"):
            capped_stream_cuts(1000, chunk=bad)
        with pytest.raises(ValueError, match="max_commit must be >= 1"):
            capped_stream_cuts(1000, max_commit=bad)

    def test_matches_the_desi_schedule_it_was_extracted_from(self) -> None:
        # The demo's wrapper must stay behaviour-identical, since its published
        # archive was built with it.
        from luxar.demos.demo_desi_galaxies import (
            SCENE_FIRST_CHUNK,
            SCENE_MAX_COMMIT,
            streaming_breakpoints,
        )

        for n in (2_000, 152_262, 1_218_970, 9_751_955):
            assert streaming_breakpoints(n) == capped_stream_cuts(
                n, SCENE_FIRST_CHUNK, SCENE_MAX_COMMIT
            )


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
