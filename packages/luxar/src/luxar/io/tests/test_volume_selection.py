"""A channel/timepoint index that a non-zarr input cannot apply must fail loudly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.io.volume import VolumeSelectionError, load_volume


@pytest.fixture
def stack_npy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    stack = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    path = tmp_path / "stack.npy"
    np.save(path, stack)
    return path, stack


@pytest.mark.parametrize(
    "kwargs", [{"channel": 1}, {"timepoint": 0}, {"channel": 0, "timepoint": 1}]
)
def test_an_index_without_axes_is_refused(stack_npy, kwargs) -> None:
    path, _ = stack_npy
    with pytest.raises(ValueError, match="axes spec"):
        load_volume(path, **kwargs)


def test_tiff_index_without_axes_is_refused(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros((2, 4, 5, 6), np.float32))
    with pytest.raises(ValueError, match=r"channel=1 to stack\.tif"):
        load_volume(path, channel=1)


def test_with_axes_the_index_selects(stack_npy) -> None:
    path, stack = stack_npy
    np.testing.assert_array_equal(
        load_volume(path, channel=1, axes="c,z,y,x"), stack[1]
    )


def test_no_index_loads_the_whole_array(stack_npy) -> None:
    path, stack = stack_npy
    np.testing.assert_array_equal(load_volume(path), stack)


def test_selection_errors_are_volume_selection_errors(stack_npy) -> None:
    """The CLI turns this subclass into a usage error instead of a traceback."""
    path, _ = stack_npy
    with pytest.raises(VolumeSelectionError):
        load_volume(path, channel=1)
    with pytest.raises(VolumeSelectionError, match="out of range"):
        load_volume(path, channel=9, axes="c,z,y,x")


#: Writer signatures of microscope software. tifffile's own (``ImageJ=1.11a``,
#: ``Creator="tifffile.py ..."``) mark axes that may be writer defaults.
_FIJI = {"ImageJ": "1.54f"}
_BIO_FORMATS = {"Creator": "OME Bio-Formats 5.5.3"}


@pytest.fixture
def zcyx_stack() -> np.ndarray:
    return np.arange(4 * 3 * 5 * 6, dtype=np.uint16).reshape(4, 3, 5, 6)


def _bio_formats_xml(size_z: int, size_c: int, size_y: int, size_x: int) -> str:
    """OME-XML shaped like a Bio-Formats export (the Allen hiPSC FOV header)."""
    planes = "".join(
        f'<TiffData FirstC="{c}" FirstT="0" FirstZ="{z}" IFD="{z * size_c + c}" '
        'PlaneCount="1"/>'
        for z in range(size_z)
        for c in range(size_c)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
        'Creator="OME Bio-Formats 5.5.3" UUID="urn:uuid:316eaf8c-24ca-43e2-b504-'
        '922d8dab9882"><Image ID="Image:0" Name="fov"><Pixels BigEndian="false" '
        'DimensionOrder="XYCZT" ID="Pixels:0" Interleaved="false" '
        'PhysicalSizeX="0.108" PhysicalSizeXUnit="µm" PhysicalSizeY="0.108" '
        'PhysicalSizeYUnit="µm" PhysicalSizeZ="0.29" PhysicalSizeZUnit="µm" '
        f'SizeC="{size_c}" SizeT="1" SizeX="{size_x}" SizeY="{size_y}" '
        f'SizeZ="{size_z}" Type="uint16">'
        + "".join(
            f'<Channel ID="Channel:0:{c}" SamplesPerPixel="1"/>' for c in range(size_c)
        )
        + planes
        + "</Pixels></Image></OME>"
    )


def test_bio_formats_ome_axes_select_the_channel(
    tmp_path: Path, zcyx_stack: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-written Bio-Formats OME header (not tifffile's) is trusted."""
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.ome.tiff"
    with tifffile.TiffWriter(path) as writer:
        planes = zcyx_stack.reshape(-1, *zcyx_stack.shape[2:])
        # Bio-Formats writes UTF-8 (the micro sign); tifffile's str path is ASCII.
        xml = _bio_formats_xml(*zcyx_stack.shape).encode("utf-8")
        for i, plane in enumerate(planes):
            writer.write(
                plane,
                description=xml if i == 0 else None,
                metadata=None,
                contiguous=False,
            )
    info: dict = {}
    volume = load_volume(path, channel=2, info=info)
    np.testing.assert_array_equal(volume, zcyx_stack[:, 2])
    assert info["axes"] == ["z", "y", "x"]
    assert "(OME): z=4, c=3, y=5, x=6" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ome": True, "metadata": {"axes": "ZCYX", **_BIO_FORMATS}},
        {"imagej": True, "metadata": {"axes": "ZCYX", **_FIJI}},
    ],
    ids=["ome", "imagej"],
)
def test_declared_axes_select_the_channel(
    tmp_path: Path, zcyx_stack: np.ndarray, kwargs: dict
) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.tif"
    tifffile.imwrite(path, zcyx_stack, **kwargs)
    np.testing.assert_array_equal(load_volume(path, channel=2), zcyx_stack[:, 2])


def test_imagej_hyperstack_selects_time_and_channel(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    stack = np.arange(3 * 4 * 2 * 5 * 6, dtype=np.uint16).reshape(3, 4, 2, 5, 6)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, stack, imagej=True, metadata={"axes": "TZCYX", **_FIJI})
    np.testing.assert_array_equal(
        load_volume(path, channel=1, timepoint=2), stack[2, :, 1]
    )


@pytest.mark.parametrize(
    ("shape", "kwargs", "declared"),
    [
        # A CZYX array saved without axes: tifffile writes channels=10, slices=2.
        ((2, 10, 5, 6), {"imagej": True}, "z=2, c=10, y=5, x=6"),
        # A z-stack saved without axes comes back as channels.
        ((10, 5, 6), {"imagej": True}, "c=10, y=5, x=6"),
        ((2, 10, 5, 6), {"ome": True}, "z=2, c=10, y=5, x=6"),
        ((10, 5, 6), {"ome": True}, "c=10, y=5, x=6"),
        # Explicit axes are indistinguishable on disk, so they are refused too.
        ((10, 2, 5, 6), {"ome": True, "metadata": {"axes": "ZCYX"}}, "z=10, c=2"),
    ],
)
def test_tifffile_written_axes_are_not_trusted(
    tmp_path: Path, shape: tuple, kwargs: dict, declared: str
) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros(shape, np.uint16), **kwargs)
    with pytest.raises(VolumeSelectionError) as excinfo:
        load_volume(path, channel=1)
    message = str(excinfo.value)
    assert declared in message and "tifffile" in message and "--axes" in message


@pytest.mark.parametrize("creator", ["bioio_ome_tiff 1.4.0", "aicsimageio 4.14.0"])
def test_bioio_written_axes_are_not_trusted(tmp_path: Path, creator: str) -> None:
    """bioio's OmeTiffWriter.save(a4d) without dim_order writes CZYX defaults."""
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.ome.tiff"
    tifffile.imwrite(
        path,
        np.zeros((2, 10, 5, 6), np.uint16),
        ome=True,
        metadata={"axes": "CZYX", "Creator": creator},
    )
    with pytest.raises(VolumeSelectionError) as excinfo:
        load_volume(path, channel=1)
    message = str(excinfo.value)
    assert f"'{creator}' wrote it, so the axes may be writer defaults" in message
    assert "c=2, z=10, y=5, x=6" in message and "--axes" in message


def test_samples_axis_is_refused(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "rgb.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 5, 6, 3), np.uint8),
        ome=True,
        photometric="rgb",
        metadata={"axes": "ZYXS", **_BIO_FORMATS},
    )
    with pytest.raises(VolumeSelectionError, match=r"z=4, y=5, x=6, s=3.*not s"):
        load_volume(path, channel=1)


def test_no_index_ignores_the_declared_axes(
    tmp_path: Path, zcyx_stack: np.ndarray
) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.ome.tif"
    tifffile.imwrite(
        path, zcyx_stack, ome=True, metadata={"axes": "ZCYX", **_BIO_FORMATS}
    )
    np.testing.assert_array_equal(load_volume(path), zcyx_stack)


def test_index_for_an_undeclared_axis_is_refused(
    tmp_path: Path, zcyx_stack: np.ndarray
) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.ome.tif"
    tifffile.imwrite(
        path, zcyx_stack, ome=True, metadata={"axes": "ZCYX", **_BIO_FORMATS}
    )
    with pytest.raises(VolumeSelectionError, match="no time axis"):
        load_volume(path, timepoint=1)


def test_explicit_axes_override_the_declared_ones(
    tmp_path: Path, zcyx_stack: np.ndarray
) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "fov.ome.tif"
    tifffile.imwrite(
        path, zcyx_stack, ome=True, metadata={"axes": "ZCYX", **_BIO_FORMATS}
    )
    np.testing.assert_array_equal(
        load_volume(path, channel=1, axes="c,z,y,x"), zcyx_stack[1]
    )


def test_plain_tiff_error_names_the_missing_metadata(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros((2, 4, 5, 6), np.float32), photometric="minisblack")
    with pytest.raises(VolumeSelectionError, match="no OME or ImageJ metadata"):
        load_volume(path, channel=1)


def test_missing_tiff_reports_the_real_reason(tmp_path: Path) -> None:
    pytest.importorskip("tifffile")
    with pytest.raises(OSError):
        load_volume(tmp_path / "absent.tif", channel=1)


@pytest.mark.parametrize(
    "kwargs", [{"channel": 3}, {"timepoint": 2}, {"channel": 7, "timepoint": 0}]
)
def test_positional_zarr_index_out_of_range(tmp_path: Path, kwargs: dict) -> None:
    from luxar._zarr_compat import create_array, open_group

    store = tmp_path / "t5.zarr"
    root = open_group(store, mode="w")
    create_array(root, "0", data=np.zeros((2, 3, 4, 5, 6), np.float32), compressor=None)
    with pytest.raises(VolumeSelectionError, match="out of range"):
        load_volume(store, **kwargs)
