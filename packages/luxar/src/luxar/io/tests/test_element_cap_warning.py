"""Tests for the authoring-time element-texture cap warning (issue #1957).

The viewer CLAMPS a node that overflows its element texture — it drops the tail
with one console warning nobody reads. Geometry is stored in Hilbert order, so
the lost tail is one spatially CONTIGUOUS lobe: the symptom is a clean-edged
wedge of missing geometry that reads as a data or masking bug. #1957 lost the
entire North Atlantic to a 2.3% overflow. Warning at write time puts the
diagnosis where it is cheap to act on.
"""

from __future__ import annotations

import pytest

from luxar.io._compiler.node_common import warn_if_over_element_cap
from luxar.typing_utils.constants import (
    MAX_POINTS_PER_POINTS_NODE,
    MAX_SEGMENTS_PER_LINES_NODE,
    MAX_SPLATS_PER_GSPLATS_NODE,
    max_elements_per_node,
)

# ------------------------------------------------------------------------ caps


def test_caps_match_the_viewer_element_texture_layout() -> None:
    """Mirror of ``element-texture-layout.ts``: width rounds DOWN to whole elements.

    ``width = floor(4096 / texels) * texels``, then
    ``capacity = floor(width * 4096 / texels)``. Lines waste 4 texels of width
    (4096 is not a multiple of 6) and points waste 1 — which is why the caps are
    not simply ``4096 * 4096 / texels``.
    """
    assert MAX_SEGMENTS_PER_LINES_NODE == 682 * 4096 == 2_793_472
    assert MAX_POINTS_PER_POINTS_NODE == 1365 * 4096 == 5_591_040
    assert MAX_SPLATS_PER_GSPLATS_NODE == 1024 * 4096 == 4_194_304


def test_unknown_geometry_type_has_no_cap() -> None:
    with pytest.raises(KeyError):
        max_elements_per_node("mesh")


# -------------------------------------------------------------------- warnings


@pytest.mark.parametrize(
    "geometry_type,cap",
    [
        ("lines", MAX_SEGMENTS_PER_LINES_NODE),
        ("points", MAX_POINTS_PER_POINTS_NODE),
        ("gsplats", MAX_SPLATS_PER_GSPLATS_NODE),
    ],
)
def test_over_cap_warns_and_at_cap_does_not(geometry_type: str, cap: int) -> None:
    """The boundary is inclusive: exactly at the cap still renders whole."""
    assert warn_if_over_element_cap(geometry_type, cap + 1, "/node") is True
    assert warn_if_over_element_cap(geometry_type, cap, "/node") is False
    assert warn_if_over_element_cap(geometry_type, 1, "/node") is False


def test_a_geometry_type_without_an_element_texture_is_a_no_op() -> None:
    """Mesh has no per-element texture, so there is nothing to overflow."""
    assert warn_if_over_element_cap("mesh", 10**9, "/node") is False


def test_the_ocean_currents_overflow_would_have_warned() -> None:
    """The exact #1957 numbers: 11,440,000 segments in one un-partitioned node."""
    assert warn_if_over_element_cap("lines", 11_440_000, "/currents") is True


def test_a_partitioned_part_stays_quiet(capsys: pytest.CaptureFixture) -> None:
    """Splitting the same 11.44M segments across 8 parts silences the warning.

    This is the fix the message tells the author to apply, so it has to work.
    """
    per_part = 11_440_000 // 8
    for i in range(8):
        assert (
            warn_if_over_element_cap("lines", per_part, f"/currents/part_{i}") is False
        )
    assert "partition=" not in capsys.readouterr().out


def test_the_warning_names_the_count_the_cap_and_the_remedy(
    capsys: pytest.CaptureFixture,
) -> None:
    warn_if_over_element_cap("lines", 11_440_000, "/currents")
    out = capsys.readouterr().out
    assert "/currents" in out
    assert "11,440,000" in out
    assert "2,793,472" in out
    assert "partition=dict(max_elements=...)" in out
