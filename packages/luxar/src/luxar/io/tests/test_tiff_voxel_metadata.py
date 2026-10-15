"""``load_volume`` reports the voxel spacing an OME-TIFF / ImageJ TIFF recorded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from luxar.io._tiff_metadata import normalize_unit
from luxar.io.volume import load_volume


def _ome(path: Path, **meta: object) -> Path:
    tifffile.imwrite(
        path, np.zeros((4, 5, 6), np.uint16), metadata={"axes": "ZYX", **meta}
    )
    return path


def _imagej(path: Path, *, unit: str | None = "um", spacing: float = 2.0) -> Path:
    metadata: dict = {"axes": "ZYX", "spacing": spacing}
    if unit is not None:
        metadata["unit"] = unit
    tifffile.imwrite(
        path,
        np.zeros((4, 5, 6), np.uint16),
        imagej=True,
        resolution=(1 / 0.25, 1 / 0.5),
        metadata=metadata,
    )
    return path


def test_ome_physical_sizes_in_array_order_with_units(tmp_path: Path) -> None:
    path = _ome(
        tmp_path / "s.ome.tif",
        PhysicalSizeX=0.2,
        PhysicalSizeY=0.3,
        PhysicalSizeZ=1.5,
        PhysicalSizeZUnit="nm",
    )
    info: dict = {}
    volume = load_volume(path, info=info)
    assert volume.shape == (4, 5, 6)
    assert info["axes"] == ["z", "y", "x"]
    # Normalized to ONE unit: the z step recorded in nm becomes 0.0015 um.
    assert info["voxel_size"] == pytest.approx([1.5e-3, 0.3, 0.2])
    assert info["voxel_unit"] == "um"
    # As recorded; an absent PhysicalSize?Unit means the OME default, um.
    assert info["source_voxel_size"] == pytest.approx([1.5, 0.3, 0.2])
    assert info["source_voxel_units"] == ["nm", "um", "um"]
    assert info["voxel_size_source"] == "ome"
    assert info["source_dtype"] == "uint16"


def test_imagej_spacing_and_resolution(tmp_path: Path) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif"), info=info)
    assert info["axes"] == ["z", "y", "x"]
    assert info["voxel_size"] == pytest.approx([2.0, 0.5, 0.25])
    assert info["voxel_unit"] == "um"
    assert info["source_voxel_units"] == ["um", "um", "um"]
    assert info["voxel_size_source"] == "imagej"


def test_imagej_micro_sign_unit(tmp_path: Path) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif", unit="\\u00B5m"), info=info)
    assert info["source_voxel_units"] == ["um", "um", "um"]
    assert info["voxel_size"] == pytest.approx([2.0, 0.5, 0.25])


@pytest.mark.parametrize(
    ("unit", "factor"),
    # ImageJ descriptions are ASCII: it writes the A-ring as a literal escape.
    [("mm", 1e3), ("nm", 1e-3), ("\\u00C5", 1e-4)],
)
def test_imagej_units_convert_to_micrometres(
    tmp_path: Path, unit: str, factor: float
) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif", unit=unit), info=info)
    assert info["voxel_size"] == pytest.approx(
        [2.0 * factor, 0.5 * factor, 0.25 * factor]
    )
    assert info["voxel_unit"] == "um"
    assert info["source_voxel_size"] == pytest.approx([2.0, 0.5, 0.25])


@pytest.mark.parametrize("unit", ["\u00c5", "\u212b"])
def test_ome_angstrom_converts_to_micrometres(tmp_path: Path, unit: str) -> None:
    path = _ome(
        tmp_path / "a.ome.tif",
        PhysicalSizeX=2.0,
        PhysicalSizeY=2.0,
        PhysicalSizeZ=5.0,
        PhysicalSizeXUnit=unit,
        PhysicalSizeYUnit=unit,
        PhysicalSizeZUnit=unit,
    )
    info: dict = {}
    load_volume(path, info=info)
    assert info["voxel_size"] == pytest.approx([5e-4, 2e-4, 2e-4])
    assert info["source_voxel_units"] == ["angstrom"] * 3


def test_unknown_unit_keeps_the_record_but_no_normalized_size(tmp_path: Path) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif", unit="furlong"), info=info)
    assert "voxel_size" not in info
    assert info["source_voxel_units"] == ["furlong"] * 3


def test_uncalibrated_imagej_reports_no_spacing(tmp_path: Path) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif", unit=None), info=info)
    assert "voxel_size" not in info


def test_plain_tiff_reports_no_spacing(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 5, 6), np.uint16), photometric="minisblack")
    info: dict = {}
    load_volume(path, info=info)
    assert "voxel_size" not in info
    assert info["source_dtype"] == "uint16"


def test_explicit_axes_labels_name_the_kept_axes(tmp_path: Path) -> None:
    info: dict = {}
    load_volume(_imagej(tmp_path / "ij.tif"), axes="depth,height,width", info=info)
    assert info["axes"] == ["z", "y", "x"]
    assert info["voxel_size"] == pytest.approx([2.0, 0.5, 0.25])


def test_squeezed_singleton_axis_keeps_alignment(tmp_path: Path) -> None:
    path = tmp_path / "plane.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((1, 5, 6), np.uint16),
        metadata={"axes": "ZYX", "PhysicalSizeX": 0.2, "PhysicalSizeY": 0.3},
    )
    info: dict = {}
    volume = load_volume(path, info=info)
    assert volume.shape == (5, 6)
    assert info["axes"] == ["y", "x"]
    assert info["voxel_size"] == pytest.approx([0.3, 0.2])
    assert info["voxel_unit"] == "um"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("µm", "um"),
        ("\\u00B5m", "um"),
        ("micron", "um"),
        ("nm", "nm"),
        ("furlong", "furlong"),
        ("\u00c5", "angstrom"),
        ("\\u00C5", "angstrom"),
        (None, None),
    ],
)
def test_normalize_unit(raw: str | None, expected: str | None) -> None:
    assert normalize_unit(raw) == expected
