"""Lazy volume access: :func:`open_volume_lazy`, :func:`pin_volume_axes`, axis maps.

These three exist for one caller — a volume re-fit that only ever wants a slice
of a source that may be hundreds of gigabytes (see
:mod:`luxar.gsplats.lod.volume_regions`). What matters is therefore not just the
values they return but that nothing on the path materialises the whole array, and
that the array they pick is the SAME one the shape-discovery path picked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import create_array
from luxar.io.volume import (
    load_volume,
    open_volume_lazy,
    pin_volume_axes,
    volume_axes_from_spec,
)


def _write_zarr_array(path: Path, arr: np.ndarray) -> None:
    import zarr

    z = zarr.open(str(path), mode="w", shape=arr.shape, dtype=arr.dtype)
    z[:] = arr


@pytest.mark.parametrize(
    ("shape", "region", "expected_shape"),
    [
        ((1, 1, 1, 8, 8), (slice(0, 6), slice(2, 8)), (6, 6)),
        ((1, 8, 8, 8), (slice(0, 6), slice(1, 7), slice(2, 8)), (6, 6, 6)),
    ],
)
def test_load_volume_region_targets_post_squeeze_axes(
    tmp_path: Path,
    shape: tuple[int, ...],
    region: tuple[slice, ...],
    expected_shape: tuple[int, ...],
) -> None:
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = tmp_path / "vol.zarr"
    _write_zarr_array(path, arr)

    loaded = load_volume(path, region=region)

    expected = np.squeeze(arr)[region].astype(np.float32)
    assert loaded.shape == expected_shape
    np.testing.assert_array_equal(loaded, expected)


def test_load_volume_region_rank_must_match_surviving_axes(tmp_path: Path) -> None:
    path = tmp_path / "vol.zarr"
    _write_zarr_array(path, np.zeros((1, 8, 8, 8), dtype=np.uint16))

    with pytest.raises(ValueError, match="region has 2 axes.*keeps 3"):
        load_volume(path, region=(slice(0, 4), slice(0, 4)))


class TestOpenVolumeLazy:
    """A zarr store comes back as its array object, never as a numpy copy."""

    def test_plain_array_stays_lazy(self, tmp_path: Path) -> None:
        arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        path = tmp_path / "vol.zarr"
        _write_zarr_array(path, arr)

        node = open_volume_lazy(path)
        # Not a numpy array: the point is that reads are deferred.
        assert not isinstance(node, np.ndarray)
        assert tuple(node.shape) == (2, 3, 4)
        np.testing.assert_array_equal(np.asarray(node[1]), arr[1])

    def test_ome_group_resolves_level_zero(self, tmp_path: Path) -> None:
        """An OME-NGFF multiscale group has no ``shape``; level "0" is the source.

        ``load_volume`` and ``discover_ome_zarr_shape`` both resolve it that way,
        and a caller RE-opening a store those already read must land on the same
        array — otherwise a re-fit would silently target a downsampled level.
        """
        import zarr

        path = tmp_path / "ome.zarr"
        root = zarr.open_group(str(path), mode="w")
        full = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
        create_array(root, "0", data=full)
        create_array(root, "1", data=full[::2, ::2, ::2])

        node = open_volume_lazy(path)
        assert tuple(node.shape) == (8, 8, 8)
        np.testing.assert_array_equal(np.asarray(node[:]), full)

    def test_group_without_level_zero_picks_the_largest_array(
        self, tmp_path: Path
    ) -> None:
        import zarr

        path = tmp_path / "nested.zarr"
        root = zarr.open_group(str(path), mode="w")
        big = np.zeros((4, 5, 6), dtype=np.uint16)
        create_array(root.create_group("h2afva"), "fused", data=big)
        create_array(root, "thumb", data=np.zeros((2, 2, 2), np.uint16))

        assert tuple(open_volume_lazy(path).shape) == (4, 5, 6)
        # An explicit key still wins over the heuristic.
        assert tuple(open_volume_lazy(path, "thumb").shape) == (2, 2, 2)

    def test_missing_array_key_is_named(self, tmp_path: Path) -> None:
        import zarr

        path = tmp_path / "g.zarr"
        create_array(
            zarr.open_group(str(path), mode="w"),
            "a",
            data=np.zeros((2, 2, 2), np.uint16),
        )
        with pytest.raises(ValueError, match="nope"):
            open_volume_lazy(path, "nope")

    def test_non_zarr_falls_back_to_the_eager_loader(self, tmp_path: Path) -> None:
        """.npy/.tiff have no lazy reader; they are single volumes anyway."""
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        path = tmp_path / "v.npy"
        np.save(path, arr)
        np.testing.assert_allclose(open_volume_lazy(path), load_volume(path))


class TestPinVolumeAxes:
    """Pinning defers the fixed indices into every read, rather than pre-slicing.

    Pre-slicing a zarr array materialises it, which is exactly what the lazy open
    exists to avoid — so the view has to survive being indexed like the array it
    stands in for.
    """

    def test_nothing_to_pin_hands_the_store_straight_through(self) -> None:
        base = np.zeros((2, 2), np.float32)
        assert pin_volume_axes(base, {}) is base

    def test_pinned_axes_disappear_from_the_shape_and_from_reads(self) -> None:
        arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        # A t,c,z,y source with t and c selected away.
        view = pin_volume_axes(arr, {0: 1, 1: 2})
        assert view.shape == (4, 5)
        np.testing.assert_array_equal(np.asarray(view[:, :]), arr[1, 2])
        np.testing.assert_array_equal(np.asarray(view[1:3, 2:4]), arr[1, 2, 1:3, 2:4])
        # A partial index leaves the trailing axes whole, like numpy.
        np.testing.assert_array_equal(np.asarray(view[2]), arr[1, 2, 2])

    def test_lazy_base_is_only_read_through_the_pinned_index(
        self, tmp_path: Path
    ) -> None:
        arr = np.arange(3 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 4, 4)
        path = tmp_path / "tc.zarr"
        _write_zarr_array(path, arr)

        view = pin_volume_axes(open_volume_lazy(path), {1: 1})
        assert view.shape == (3, 4, 4)
        np.testing.assert_array_equal(np.asarray(view[2]), arr[2, 1])

    def test_out_of_range_pin_is_rejected_with_the_axis_named(self) -> None:
        arr = np.zeros((2, 3), np.float32)
        with pytest.raises(ValueError, match="axis 1"):
            pin_volume_axes(arr, {1: 5})
        with pytest.raises(ValueError, match="out of range"):
            pin_volume_axes(arr, {7: 0})


class TestVolumeAxesFromSpec:
    """Spatial-first with the stacked axes LAST — the fitted-splat convention."""

    def test_time_first_source_maps_to_spatial_first_centers(self) -> None:
        assert volume_axes_from_spec("t,z,y,x", 4) == (1, 2, 3, 0)

    def test_identity_for_a_purely_spatial_source(self) -> None:
        assert volume_axes_from_spec("z,y,x", 3) == (0, 1, 2)

    def test_label_count_must_match_the_splats(self) -> None:
        with pytest.raises(ValueError, match="4 labels but the splats are 3D"):
            volume_axes_from_spec("t,z,y,x", 3)

    def test_unrecognised_label_names_the_flag_the_user_typed(self) -> None:
        with pytest.raises(ValueError, match=r"--axes label 'q'"):
            volume_axes_from_spec("q,y,x", 3, flag="--axes")
