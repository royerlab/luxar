"""A channel/timepoint index that a non-zarr input cannot apply must fail loudly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.io.volume import load_volume


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
