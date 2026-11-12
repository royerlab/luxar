"""A compressed TIFF without imagecodecs fails with an install hint.

tifffile raises a missing-imagecodecs ``ValueError`` for LZW/JPEG or a missing
compression ``ImportError`` for zstd on Python < 3.14. ``load_volume`` must
name the extra and pip command, and leave unrelated errors alone.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from luxar.io.volume import load_volume


def _fake_tifffile(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    module = types.ModuleType("tifffile")

    def imread(_path: str) -> np.ndarray:
        raise exc

    module.imread = imread  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tifffile", module)


def test_missing_imagecodecs_names_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stack.tif"
    path.write_bytes(b"")
    _fake_tifffile(
        monkeypatch,
        ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"),
    )
    with pytest.raises(ImportError) as info:
        load_volume(path)
    message = str(info.value)
    assert "pip install 'luxar[io]'" in message
    assert "pip install imagecodecs" in message
    assert "stack.tif" in message


def test_missing_zstd_fallback_names_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "zstd.tif"
    path.write_bytes(b"")
    _fake_tifffile(monkeypatch, ModuleNotFoundError("No module named 'compression'"))
    with pytest.raises(ImportError) as info:
        load_volume(path)
    assert "pip install 'luxar[io]'" in str(info.value)
    assert "pip install imagecodecs" in str(info.value)
    assert "zstd.tif" in str(info.value)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_unrelated_value_error_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"")
    _fake_tifffile(monkeypatch, ValueError("not a TIFF file"))
    with pytest.raises(ValueError, match="not a TIFF file"):
        load_volume(path)


def test_unrelated_import_error_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"")
    error = ImportError("some other import failed")
    _fake_tifffile(monkeypatch, error)
    with pytest.raises(ImportError) as info:
        load_volume(path)
    assert info.value is error


def test_compressed_tiff_round_trips_when_imagecodecs_is_installed(
    tmp_path: Path,
) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    volume = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    path = tmp_path / "lzw.tif"
    tifffile.imwrite(str(path), volume, compression="lzw", photometric="minisblack")
    np.testing.assert_array_equal(load_volume(path), volume)
