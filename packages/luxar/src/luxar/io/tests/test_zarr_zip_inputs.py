"""Reading a ``.zarr.zip`` volume — the suffix-dispatch the zarr 3 move broke.

``luxar gsplat fit data.zarr.zip`` and ``luxar gsplat cal data.zarr.zip`` are
documented entry points, and the whole ``batch-fit`` family takes zipped OME-Zarr
input. zarr 2 made that work by accident of ``normalize_store_arg``, which sniffed
a ``.zip`` suffix and substituted a ZipStore. zarr 3 removed that sniffing: a bare
``zarr.open("x.zarr.zip")`` treats the archive as a directory and raises
``GroupNotFoundError``.

Nothing in the suite covered a zipped input before, which is why the regression
was invisible — the readers were only ever exercised on directory stores.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import ZARR_FORMAT, open_store
from luxar.io.ome_zarr import discover_ome_zarr_shape
from luxar.io.volume import load_volume


def _write_zipped_zarr(
    tmp_path: Path, shape: tuple[int, ...]
) -> tuple[Path, np.ndarray]:
    """Build a real ``.zarr.zip``: a directory store, then zipped in place.

    Zipped by hand rather than through ZipStore so the archive layout matches
    what external producers ship (paths relative to the store root).
    """
    src = tmp_path / "vol.zarr"
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / np.prod(
        shape
    )
    arr = zarr.create_array(
        store=str(src),
        shape=shape,
        dtype=np.float32,
        zarr_format=ZARR_FORMAT,
        chunks=tuple(max(1, s // 2) for s in shape),
    )
    arr[...] = data

    zipped = tmp_path / "vol.zarr.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src))
    return zipped, data


def test_a_bare_zarr_open_cannot_read_a_zip_which_is_why_the_facade_exists(
    tmp_path: Path,
) -> None:
    """Pin the zarr-3 behaviour this module works around.

    If a future zarr restores suffix sniffing this test fails, which is the
    signal to simplify the readers rather than a bug to paper over.
    """
    zipped, _ = _write_zipped_zarr(tmp_path, (4, 4, 4))
    with pytest.raises(FileNotFoundError):
        zarr.open(str(zipped), mode="r")


def test_open_store_reads_a_zipped_store(tmp_path: Path) -> None:
    """The facade dispatches ZipStore on the suffix, so the archive opens."""
    zipped, data = _write_zipped_zarr(tmp_path, (4, 4, 4))
    store = open_store(zipped, mode="r")
    assert isinstance(store, zarr.storage.ZipStore)
    arr = zarr.open(store=store, mode="r")
    assert np.allclose(np.asarray(arr[...]), data)


def test_load_volume_reads_a_zipped_store(tmp_path: Path) -> None:
    """`load_volume` on a `.zarr.zip` — the `gsplat fit`/`cal` input path."""
    zipped, data = _write_zipped_zarr(tmp_path, (4, 6, 8))
    volume = load_volume(zipped)
    assert volume.shape == data.shape
    assert np.allclose(volume, data, atol=1e-6)


def test_discover_ome_zarr_shape_reads_a_zipped_store(tmp_path: Path) -> None:
    """`discover_ome_zarr_shape` on a `.zarr.zip` — the `batch-fit` planning path.

    Planning reads only metadata, so a failure here would abort a whole cluster
    submission before any tile was fitted.
    """
    zipped, data = _write_zipped_zarr(tmp_path, (4, 6, 8))
    info = discover_ome_zarr_shape(str(zipped))
    assert tuple(info.shape) == data.shape


def test_a_directory_store_still_works(tmp_path: Path) -> None:
    """The suffix dispatch must not regress the ordinary directory case."""
    zipped, data = _write_zipped_zarr(tmp_path, (4, 4, 4))
    directory = zipped.parent / "vol.zarr"
    assert directory.is_dir(), "fixture should have left the directory store behind"
    assert isinstance(open_store(directory, mode="r"), zarr.storage.LocalStore)
    assert np.allclose(load_volume(directory), data, atol=1e-6)
