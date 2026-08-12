"""Unit tests for ``luxar.core.group.compositing`` — the pure helpers shared
by the partition-/lod-wrapping leaf adders.

B7/[P5]: ``slice_optional_array`` and ``position_bounds_from_array`` were only
exercised incidentally through the end-to-end partition tests, leaving the
scalar / list / 0-D / mismatched-shape branches (compositing.py:62-70) and
the empty-array guard (line 83) uncovered. These direct unit tests pin every
branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.compositing import (
    funnel_add_error,
    is_broadcast_color,
    position_bounds_from_array,
    slice_optional_array,
    unnest_add_error,
)


class TestSliceOptionalArray:
    def test_none_passes_through(self) -> None:
        idx = np.array([0, 2], dtype=np.intp)
        assert slice_optional_array(None, idx, n_elements=4) is None

    @pytest.mark.parametrize("scalar", [3, 1.5, True, "label"])
    def test_scalars_pass_through_unchanged(self, scalar) -> None:
        idx = np.array([0, 1], dtype=np.intp)
        assert slice_optional_array(scalar, idx, n_elements=4) == scalar

    def test_list_matching_length_is_sliced(self) -> None:
        """A per-element list (e.g. string labels) is reindexed."""
        value = ["a", "b", "c", "d"]
        idx = np.array([3, 0], dtype=np.intp)
        assert slice_optional_array(value, idx, n_elements=4) == ["d", "a"]

    def test_list_wrong_length_passes_through(self) -> None:
        """A list whose length != n_elements is left untouched."""
        value = ["only", "two"]
        idx = np.array([0, 1, 2], dtype=np.intp)
        assert slice_optional_array(value, idx, n_elements=4) == ["only", "two"]

    def test_zero_d_array_passes_through(self) -> None:
        value = np.array(5.0, dtype=np.float32)  # 0-D
        idx = np.array([0, 1], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        assert out is value  # returned unchanged (identity)

    def test_per_element_array_is_sliced(self) -> None:
        value = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        idx = np.array([2, 0], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, [30.0, 10.0])

    def test_2d_per_element_array_is_sliced_on_first_axis(self) -> None:
        value = np.arange(12, dtype=np.float32).reshape(4, 3)  # (N=4, 3)
        idx = np.array([1, 3], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, value[[1, 3]])

    def test_broadcast_vector_passes_through(self) -> None:
        """An RGB 3-vector (shape (3,)) with n_elements=4 must NOT be sliced."""
        value = np.array([255, 128, 0], dtype=np.uint8)
        idx = np.array([0, 1], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, value)
        assert out.shape == (3,)

    def test_non_ndarray_sequence_is_converted_then_sliced(self) -> None:
        """A tuple of length n_elements is asarray'd and first-axis sliced."""
        value = (1.0, 2.0, 3.0, 4.0)
        idx = np.array([3, 2], dtype=np.intp)
        out = slice_optional_array(value, idx, n_elements=4)
        np.testing.assert_array_equal(out, [4.0, 3.0])


class TestIsBroadcastColor:
    """The classifier that keeps a uniform RGB(A) out of the per-element slicer.

    Its whole job is a TYPE/SHAPE decision, so the edges worth pinning are the
    ones where "3 or 4 numbers" is ambiguous: a bool is an ``int`` subclass, a
    numpy scalar is not a Python one, and a 1-D ndarray colour looks the same
    length as a legal tuple.
    """

    @pytest.mark.parametrize(
        "colors",
        [
            (0.25, 0.5, 1.0),
            [0.25, 0.5, 1.0],
            (0.25, 0.5, 1.0, 0.5),
            [0, 128, 255],
            # bool IS an int subclass, so (True, False, True) reads as a colour.
            # Deliberate: the writer's broadcast validator accepts it too, so
            # classifying it here keeps the two paths agreeing.
            (True, False, True),
            (np.float32(0.25), np.float32(0.5), np.float32(1.0)),
            (np.uint8(64), np.uint8(128), np.uint8(255)),
        ],
    )
    def test_uniform_sequences_are_broadcast(self, colors) -> None:
        assert is_broadcast_color(colors) is True

    @pytest.mark.parametrize(
        "colors",
        [
            None,
            "viridis",
            0.5,
            (0.25, 0.5),  # length 2
            (0.1, 0.2, 0.3, 0.4, 0.5),  # length 5
            (),
            # Per-element data of every shape: entries that are themselves
            # sequences fail the component test.
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            # A numpy colour is never the broadcast form — the writer refuses a
            # 1-D ndarray outright and wants (1, c) instead.
            np.array([0.25, 0.5, 1.0], dtype=np.float32),
            np.array([[0.25, 0.5, 1.0]], dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        ],
    )
    def test_everything_else_is_not_broadcast(self, colors) -> None:
        assert is_broadcast_color(colors) is False


class TestPositionBoundsFromArray:
    def test_normal_nd_bounds(self) -> None:
        pos = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        bounds = position_bounds_from_array(pos)
        assert bounds["min"] == [1.0, -2.0, -6.0]
        assert bounds["max"] == [4.0, 5.0, 3.0]

    def test_single_row(self) -> None:
        """N=1 boundary: min == max == the single point."""
        pos = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
        bounds = position_bounds_from_array(pos)
        assert bounds["min"] == [7.0, 8.0, 9.0]
        assert bounds["max"] == [7.0, 8.0, 9.0]

    def test_empty_array_raises(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="empty array"):
            position_bounds_from_array(pos)

    def test_returns_python_floats(self) -> None:
        """Values are JSON-serialisable Python floats, not numpy scalars."""
        pos = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        bounds = position_bounds_from_array(pos)
        assert all(isinstance(v, float) for v in bounds["min"])
        assert all(isinstance(v, float) for v in bounds["max"])


class TestFunnelAddError:
    """Unit tests for ``funnel_add_error`` (#1491) — no end-to-end test above the
    adders exercises the raw string logic directly, so these pin every branch
    the regex/token-match decides between.
    """

    def test_same_geometry_inner_prefix_is_stripped(self) -> None:
        """The case the function exists for: a same-kind recursive child."""
        exc = ValueError("Could not add points 'child_3': Image labels length (2)")
        message = funnel_add_error("points", "p", exc)
        assert message == "Could not add points 'p': Image labels length (2)"
        assert "child_3" not in message
        assert message.count("Could not add") == 1

    def test_cross_geometry_inner_prefix_is_kept(self) -> None:
        """A DIFFERENT geometry's own failure — e.g. a lifted gsplats child under
        a Points ``substitutive_lod=`` ladder — must not be relabelled as the
        outer geometry's fault (the reserved-attr misattribution this fixes).
        """
        exc = ValueError(
            "Could not add gsplats 'child_0': Attribute(s) ['amplitude_range'] "
            "are reserved"
        )
        message = funnel_add_error("points", "p", exc)
        assert message == (
            "Could not add points 'p': Could not add gsplats 'child_0': "
            "Attribute(s) ['amplitude_range'] are reserved"
        )

    def test_group_inner_prefix_is_never_stripped(self) -> None:
        """``group`` is not a geometry word — a wrapper-creation failure must
        stay visibly a group failure, never be relabelled as this geometry's.
        """
        exc = ValueError("Could not add group 'g': boom")
        message = funnel_add_error("points", "P", exc)
        assert message == "Could not add points 'P': Could not add group 'g': boom"

    def test_plain_message_passes_through_byte_for_byte(self) -> None:
        exc = ValueError("positions must be finite")
        message = funnel_add_error("mesh", "surf", exc)
        assert message == "Could not add mesh 'surf': positions must be finite"

    def test_apostrophe_in_name_does_not_get_mangled(self) -> None:
        """``validate_node_name`` accepts an apostrophe in a node name.

        The regex's ``[^']*`` name group stops at the FIRST quote it meets, so
        a name like ``a'b`` breaks its own closing ``': `` and the message is
        never recognised as the nested-funnel shape — it is not stripped, but
        it must still come through intact (re-prefixed, not truncated or
        interleaved) rather than corrupted.
        """
        exc = ValueError("Could not add lines 'a'b': widths must be positive")
        message = funnel_add_error("lines", "a'b", exc)
        assert message == (
            "Could not add lines 'a'b': "
            "Could not add lines 'a'b': widths must be positive"
        )

    def test_idempotent_on_its_own_output(self) -> None:
        """Running the result back through the function reproduces it exactly.

        Not because the regex has nothing left to match — it matches this
        function's OWN output shape too — but because the strip and the
        re-prefix use the SAME geometry/name, so stripping what was just added
        back and re-adding it is a no-op.
        """
        exc = ValueError("Could not add points 'child_3': Image labels length (2)")
        once = funnel_add_error("points", "p", exc)
        twice = funnel_add_error("points", "p", ValueError(once))
        assert once == twice

    def test_wrong_length_full_shape_matches_the_issue_repro(self) -> None:
        """The exact shape measured in issue #1491's own repro."""
        exc = ValueError(
            "Could not add points 'child_3': Image labels length (200) must "
            "match element count (400)"
        )
        message = funnel_add_error("points", "p", exc)
        assert message == (
            "Could not add points 'p': Image labels length (200) must match "
            "element count (400)"
        )

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("Could not add points 'child_3': Image labels length (2)"),
            ValueError(
                "Could not add gsplats 'child_0': Attribute(s) "
                "['amplitude_range'] are reserved"
            ),
            ValueError("Could not add group 'g': boom"),
            ValueError("positions must be finite"),
        ],
    )
    def test_agrees_with_unnest_add_error(self, exc: ValueError) -> None:
        """``funnel_add_error`` is exactly the ``Could not add …`` re-prefix of
        ``unnest_add_error``'s own output — the dead re-derivation four adders
        used to do by hand (stripping their own ``Could not add <type>
        '<name>': `` prefix back off ``funnel_add_error``'s result) is provably
        the same string as calling ``unnest_add_error`` directly (#1491).
        """
        assert funnel_add_error("points", "p", exc) == (
            f"Could not add points 'p': {unnest_add_error('points', 'p', exc)}"
        )


class TestUnnestAddError:
    """Unit tests for ``unnest_add_error`` (#1491) — the un-nesting half
    ``funnel_add_error`` is built on, and the function each adder's
    ``except`` block now calls directly for its ``aprint`` line instead of
    stripping ``funnel_add_error``'s own re-prefix back off by hand.
    """

    def test_same_geometry_inner_prefix_is_stripped(self) -> None:
        exc = ValueError("Could not add points 'child_3': Image labels length (2)")
        assert unnest_add_error("points", "p", exc) == "Image labels length (2)"

    def test_cross_geometry_inner_prefix_is_kept(self) -> None:
        exc = ValueError(
            "Could not add gsplats 'child_0': Attribute(s) ['amplitude_range'] "
            "are reserved"
        )
        assert unnest_add_error("points", "p", exc) == (
            "Could not add gsplats 'child_0': Attribute(s) ['amplitude_range'] "
            "are reserved"
        )

    def test_group_inner_prefix_is_never_stripped(self) -> None:
        exc = ValueError("Could not add group 'g': boom")
        assert unnest_add_error("points", "P", exc) == "Could not add group 'g': boom"

    def test_plain_message_passes_through_byte_for_byte(self) -> None:
        exc = ValueError("positions must be finite")
        assert unnest_add_error("mesh", "surf", exc) == "positions must be finite"
