"""Tests for the authoring-time element-texture cap warning (issue #1957).

The viewer CLAMPS a node that overflows its element texture — it drops the tail
with one console warning nobody reads. Geometry is stored in Hilbert order, so
the lost tail is one spatially CONTIGUOUS lobe: the symptom is a clean-edged
wedge of missing geometry that reads as a data or masking bug. #1957 lost the
entire North Atlantic to a 2.3% overflow. Warning at write time puts the
diagnosis where it is cheap to act on.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.conftest import viewer_source
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
from luxar.io import ElementCapacityWarning
from luxar.io._compiler.node_common import warn_if_over_element_cap
from luxar.typing_utils.constants import (
    ELEMENT_TEXELS_PER_ELEMENT,
    ELEMENT_TEXTURE_MAX_WIDTH,
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


def test_cap_inputs_match_the_viewer_source() -> None:
    source_path = viewer_source("src/rendering/element-texture-layout.ts")
    source = source_path.read_text(encoding="utf-8")
    width_match = re.search(r"ELEMENT_TEXTURE_MAX_WIDTH\s*=\s*(\d+)", source)
    assert width_match is not None
    assert int(width_match.group(1)) == ELEMENT_TEXTURE_MAX_WIDTH

    layout_names = {"points": "POINT", "lines": "LINE", "gsplats": "SPLAT"}
    for geometry_type, layout_name in layout_names.items():
        match = re.search(
            rf"{layout_name}_TEXTURE_LAYOUT[^=]*=\s*\{{.*?"
            rf"texelsPerElement:\s*(\d+)",
            source,
            re.DOTALL,
        )
        assert match is not None, f"cannot find {layout_name}_TEXTURE_LAYOUT"
        assert int(match.group(1)) == ELEMENT_TEXELS_PER_ELEMENT[geometry_type]


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
    with pytest.warns(ElementCapacityWarning):
        assert warn_if_over_element_cap(geometry_type, cap + 1, "/node") is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert warn_if_over_element_cap(geometry_type, cap, "/node") is False
        assert warn_if_over_element_cap(geometry_type, 1, "/node") is False
    assert caught == []


def test_element_cap_warning_can_be_promoted_to_error() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ElementCapacityWarning)
        with pytest.raises(ElementCapacityWarning, match="5,591,041 points"):
            warn_if_over_element_cap("points", MAX_POINTS_PER_POINTS_NODE + 1, "/node")


def test_a_geometry_type_without_an_element_texture_is_a_no_op() -> None:
    """Mesh has no per-element texture, so there is nothing to overflow."""
    assert warn_if_over_element_cap("mesh", 10**9, "/node") is False


def test_the_ocean_currents_overflow_would_have_warned() -> None:
    """The exact #1957 numbers: 11,440,000 segments in one un-partitioned node."""
    with pytest.warns(ElementCapacityWarning):
        assert warn_if_over_element_cap("lines", 11_440_000, "/currents") is True


def test_a_partitioned_part_stays_quiet() -> None:
    """Splitting the same 11.44M segments across 8 parts silences the warning.

    This is the fix the message tells the author to apply, so it has to work.
    """
    per_part = 11_440_000 // 8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(8):
            assert (
                warn_if_over_element_cap("lines", per_part, f"/currents/part_{i}")
                is False
            )
    assert caught == []


def test_the_warning_names_the_count_the_cap_and_the_remedy() -> None:
    with pytest.warns(ElementCapacityWarning) as caught:
        warn_if_over_element_cap("lines", 11_440_000, "/currents")
    message = str(caught[0].message)
    assert "/currents" in message
    assert "11,440,000" in message
    assert "2,793,472" in message
    assert "partition=dict(max_elements=...)" in message
    assert "whole node is committed at once" in message
    assert "current slice" in message


def test_gsplat_warning_names_both_supported_remedies() -> None:
    with pytest.warns(ElementCapacityWarning) as caught:
        warn_if_over_element_cap("gsplats", MAX_SPLATS_PER_GSPLATS_NODE + 1, "/gs")
    message = str(caught[0].message)
    assert "luxar gsplat lod --recipe tiles" in message
    assert "partition=dict(max_elements=...)" in message


@pytest.mark.parametrize("geometry_type", ["points", "lines"])
def test_a_flat_node_warns_on_its_total(
    geometry_type: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The leaf writers themselves must enforce the cap, not only the helper."""
    monkeypatch.setitem(ELEMENT_TEXELS_PER_ELEMENT, geometry_type, 4096)
    positions = np.zeros((5_001, 3), dtype=np.float32)

    with pytest.warns(ElementCapacityWarning) as caught:
        with LuxarZarrCompiler(tmp_path / f"{geometry_type}.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            if geometry_type == "points":
                scene.add_points("flat", positions[:5_000])
            else:
                scene.add_lines("flat", positions, widths=0.1)

    message = str(caught[0].message)
    assert "'/flat'" in message
    assert (
        "5,000 points" if geometry_type == "points" else "5,000 segments"
    ) in message


@pytest.mark.parametrize("geometry_type", ["points", "lines"])
def test_an_additive_ladder_warns_on_its_total(
    geometry_type: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Only the actionable parent path warns when the viewer concatenates levels."""
    monkeypatch.setitem(ELEMENT_TEXELS_PER_ELEMENT, geometry_type, 4096)
    positions = np.zeros((5_001, 3), dtype=np.float32)

    with pytest.warns(ElementCapacityWarning) as caught:
        with LuxarZarrCompiler(tmp_path / f"{geometry_type}.luxar.zarr") as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            if geometry_type == "points":
                compiler.write_points_multi_lod(
                    "ladder",
                    [
                        {"positions": positions[:5_000]},
                        {"positions": positions[:5_000]},
                    ],
                )
            else:
                compiler.write_lines_multi_lod(
                    "ladder",
                    [
                        {"vertices": positions, "widths": 0.1},
                        {"vertices": positions, "widths": 0.1},
                    ],
                )

    assert len(caught) == 1
    message = str(caught[0].message)
    expected = "10,000 points" if geometry_type == "points" else "10,000 segments"
    assert "'/ladder'" in message
    assert expected in message
    assert "'/ladder/additive_0'" not in message
    assert "'/ladder/additive_1'" not in message


def test_a_gsplat_leaf_warns_on_its_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(ELEMENT_TEXELS_PER_ELEMENT, "gsplats", 4096)
    centers = np.zeros((5_000, 3), dtype=np.float32)
    cholesky = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    with pytest.warns(ElementCapacityWarning) as caught:
        with LuxarZarrCompiler(tmp_path / "gsplats.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats("gs", centers, 1.0, cholesky)

    message = str(caught[0].message)
    assert "'/gs'" in message
    assert "5,000 splats" in message


def test_an_additive_gsplat_ladder_warns_on_its_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(ELEMENT_TEXELS_PER_ELEMENT, "gsplats", 4096)
    centers = np.zeros((3_000, 3), dtype=np.float32)
    amplitudes = np.ones(3_000, dtype=np.float32)
    cholesky = np.tile(
        np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (3_000, 1),
    )
    result = GSplatData.from_additive_sublods(
        [
            AdditiveSubLOD(centers, amplitudes, cholesky),
            AdditiveSubLOD(centers, amplitudes, cholesky),
        ]
    )

    with pytest.warns(ElementCapacityWarning) as caught:
        with LuxarZarrCompiler(tmp_path / "gsplat-ladder.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("gs", result)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "'/gs'" in message
    assert "6,000 splats" in message
