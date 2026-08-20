"""bioformats2raw layouts: the node at ``"0"`` is an image GROUP, not an array.

``bioformats2raw`` — the standard converter for most of the OME ecosystem — writes
one image GROUP per series (``0``, ``1``, …) with the pyramid levels one level
further down (``0/0``, ``0/1``, …), plus an ``OME`` metadata group at the root. All
three entry points that select an array out of a zarr store used to take ``"0"`` to
BE the full-resolution array, so they either crashed with ``AttributeError: 'Group'
object has no attribute 'shape'`` or refused a store whose ``0/0`` was right there
and findable. They must also agree with each other: a re-fit re-opens a store whose
shape another command already read, so a different choice would silently target a
different (e.g. downsampled) array.

Both on-disk formats are exercised — the failure was identical in zarr format 2 and
format 3, and the selection rule reads a store rather than writing one.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import create_array
from luxar.io.ome_zarr import discover_ome_zarr_shape
from luxar.io.volume import load_volume, open_volume_lazy

ZARR_FORMATS = (2, 3)


def _ramp(shape: Tuple[int, ...], start: int = 0) -> np.ndarray:
    """A distinguishable array — so a test can prove WHICH one was read."""
    return (start + np.arange(int(np.prod(shape)), dtype=np.uint16)).reshape(shape)


def _write_image_group(
    root: zarr.Group,
    name: str,
    levels: Sequence[np.ndarray],
    attrs: Optional[Dict[str, Any]] = None,
) -> zarr.Group:
    """One bioformats2raw image: a GROUP whose children are the pyramid levels."""
    image = root.create_group(name)
    for index, level in enumerate(levels):
        create_array(image, str(index), data=level)
    for key, value in (attrs or {}).items():
        image.attrs[key] = value
    return image


def _write_labels(image: zarr.Group, name: str, mask: np.ndarray) -> None:
    """Attach an NGFF segmentation mask to an image: ``labels/<name>/<level>``.

    That is exactly where the spec puts a mask FOR THIS IMAGE, which is why a
    recursive largest-array search inside the image group is unsafe: a mask can be
    as big as level 0 (a tie) or bigger (a deterministic win).
    """
    labels = image.create_group("labels")
    labels.attrs["labels"] = [name]
    create_array(labels.create_group(name), "0", data=mask)


def _bioformats2raw_store(
    path: Path,
    zarr_format: int,
    series: Sequence[Tuple[str, Sequence[np.ndarray]]],
    image_attrs: Optional[Dict[str, Any]] = None,
    root_attrs: Optional[Dict[str, Any]] = None,
    labels: Optional[np.ndarray] = None,
) -> Path:
    """Write a bioformats2raw-shaped store, ``image_attrs`` landing on series ``0``.

    ``labels``, when given, becomes a segmentation mask under series ``0``.
    """
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    # The metadata group bioformats2raw always writes beside the images: real
    # stores have a non-image sibling at the root, which is part of why the
    # largest-array search has to be scoped to the image group.
    root.create_group("OME")
    for name, levels in series:
        image = _write_image_group(
            root, name, levels, image_attrs if name == "0" else None
        )
        if labels is not None and name == "0":
            _write_labels(image, "seg", labels)
    root.attrs["bioformats2raw.layout"] = 3
    for key, value in (root_attrs or {}).items():
        root.attrs[key] = value
    return path


def _zip_store(store: Path) -> Path:
    """Zip a directory store in place, paths relative to its root.

    Zipped by hand rather than through ZipStore so the archive layout matches what
    external producers ship (the same helper `test_zarr_zip_inputs` uses).
    """
    archive = store.parent / f"{store.name}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in sorted(store.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(store))
    return archive


def _tzyx_multiscales(
    n_levels: int, paths: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """A valid NGFF ``multiscales`` block whose axes the heuristic gets WRONG.

    A 4D store falls back to ``CZYX`` in :func:`_heuristic_ome_info`; this declares
    ``TZYX``, so reading it swaps which axis is the timepoint one. ``paths``
    overrides the declared ``datasets[*].path`` values (default ``"0"``, ``"1"``,
    …) — a block sitting at the ROOT has to name its levels through the group that
    holds them.
    """
    level_paths = (
        list(paths) if paths is not None else [str(level) for level in range(n_levels)]
    )
    return [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": level_path,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0, 0.5, 0.5]}
                    ],
                }
                for level_path in level_paths
            ],
        }
    ]


class TestTheIssueRepro:
    """The exact store from #1777: one image group holding one level."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_discovery_reports_the_level_inside_the_image_group(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = _bioformats2raw_store(
            tmp_path / "bf.zarr",
            zarr_format,
            [("0", [_ramp((2, 8, 16, 16))])],
        )

        info = discover_ome_zarr_shape(path)

        assert info.shape == (2, 8, 16, 16)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_load_volume_returns_the_real_data(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        expected = _ramp((2, 8, 16, 16))
        path = _bioformats2raw_store(
            tmp_path / "bf.zarr", zarr_format, [("0", [expected])]
        )

        volume = load_volume(path)

        assert volume.shape == (2, 8, 16, 16)
        np.testing.assert_array_equal(volume, expected.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_lazy_open_hands_back_the_array_without_materialising_it(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        expected = _ramp((2, 8, 16, 16))
        path = _bioformats2raw_store(
            tmp_path / "bf.zarr", zarr_format, [("0", [expected])]
        )

        node = open_volume_lazy(path)

        # Still the store's own array: the whole point of this entry point is that
        # a caller slicing a 431 GB timelapse never pays for the rest of it.
        assert isinstance(node, zarr.Array)
        assert not isinstance(node, np.ndarray)
        assert tuple(node.shape) == (2, 8, 16, 16)
        np.testing.assert_array_equal(np.asarray(node[1]), expected[1])


class TestScoping:
    """The search descends into the image group ONLY, never across the store."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_larger_second_series_does_not_win(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Series ``1`` is bigger, and must still lose to series ``0``'s level 0.

        A whole-store largest-array search would answer with the wrong IMAGE —
        which is a plausible wrong answer rather than an error, so it is the
        failure this test exists for.
        """
        first = _ramp((2, 4, 4, 4))
        second = _ramp((8, 16, 16, 16), start=10_000)
        path = _bioformats2raw_store(
            tmp_path / "multiseries.zarr",
            zarr_format,
            [("0", [first]), ("1", [second])],
        )

        assert discover_ome_zarr_shape(path).shape == (2, 4, 4, 4)
        assert tuple(open_volume_lazy(path).shape) == (2, 4, 4, 4)
        np.testing.assert_array_equal(load_volume(path), first.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_largest_level_of_the_pyramid_is_chosen(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        full = _ramp((2, 8, 16, 16))
        path = _bioformats2raw_store(
            tmp_path / "pyramid.zarr",
            zarr_format,
            [("0", [full, full[:, ::2, ::2, ::2], full[:, ::4, ::4, ::4]])],
        )

        assert discover_ome_zarr_shape(path).shape == (2, 8, 16, 16)
        assert tuple(open_volume_lazy(path).shape) == (2, 8, 16, 16)


class TestMetadataOnTheImageGroup:
    """NGFF metadata sits on the image group, so that is where it is read from."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_multiscales_on_the_image_group_beats_the_shape_heuristic(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Without this, a 4D store is GUESSED as CZYX with no voxel size at all."""
        full = _ramp((2, 8, 16, 16))
        path = _bioformats2raw_store(
            tmp_path / "meta.zarr",
            zarr_format,
            [("0", [full, full[:, ::2, ::2, ::2]])],
            image_attrs={"multiscales": _tzyx_multiscales(2)},
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "z", "y", "x"]
        # The heuristic's 4D fallback would say n_timepoints=1, n_channels=2.
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.spatial_axes == ["z", "y", "x"]
        assert info.voxel_size == (2.0, 0.5, 0.5)
        assert info.unit == "micrometer"
        assert info.resolution_levels == 2

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_owning_group_wins_over_a_multiscales_at_the_root(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Both declared: the owner's block is the one describing the array read."""
        root_block = _tzyx_multiscales(1)
        root_block[0]["axes"][0] = {"name": "c", "type": "channel"}
        path = _bioformats2raw_store(
            tmp_path / "both.zarr",
            zarr_format,
            [("0", [_ramp((2, 8, 16, 16))])],
            image_attrs={"multiscales": _tzyx_multiscales(1)},
            root_attrs={"multiscales": root_block},
        )

        assert discover_ome_zarr_shape(path).axes == ["t", "z", "y", "x"]


class TestExplicitArrayKey:
    """An ``array_key`` may name a group; it must never crash on ``.shape``."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_key_naming_an_image_group_resolves_to_its_level(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        second = _ramp((3, 6, 6, 6), start=99)
        path = _bioformats2raw_store(
            tmp_path / "keyed.zarr",
            zarr_format,
            [("0", [_ramp((2, 4, 4, 4))]), ("1", [second])],
        )

        assert discover_ome_zarr_shape(path, array_key="1").shape == (3, 6, 6, 6)
        assert tuple(open_volume_lazy(path, "1").shape) == (3, 6, 6, 6)
        np.testing.assert_array_equal(
            load_volume(path, array_key="1"), second.astype(np.float32)
        )

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_key_naming_an_empty_group_is_a_clear_value_error(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "hollow.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        create_array(root.create_group("images"), "0", data=_ramp((2, 2, 2)))
        root.create_group("empty")

        for call in (
            lambda: discover_ome_zarr_shape(path, array_key="empty"),
            lambda: load_volume(path, array_key="empty"),
            lambda: open_volume_lazy(path, "empty"),
        ):
            with pytest.raises(ValueError, match="empty"):
                call()

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_missing_key_still_names_the_key_and_the_store(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "g.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        create_array(root, "a", data=_ramp((2, 2, 2)))

        for call in (
            lambda: discover_ome_zarr_shape(path, array_key="nope"),
            lambda: load_volume(path, array_key="nope"),
            lambda: open_volume_lazy(path, "nope"),
        ):
            with pytest.raises(ValueError, match="nope"):
                call()


class TestLabelsAreNeverSelected:
    """An image's own segmentation masks must never be mistaken for the image.

    ``0/labels/<name>/<level>`` is where NGFF puts a mask, so a recursive
    largest-array sweep of the image group reaches it. A mask that ties with level
    0 makes the answer flip; a mask one axis bigger makes it deterministically
    WRONG — ``luxar gsplat fit`` would then fit the mask, silently.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize(
        "mask_shape", [(2, 8, 16, 16), (4, 8, 16, 16)], ids=["tied", "bigger"]
    )
    @pytest.mark.parametrize("declared", [True, False], ids=["multiscales", "bare"])
    def test_the_image_wins_over_its_own_label_mask(
        self,
        tmp_path: Path,
        zarr_format: int,
        mask_shape: Tuple[int, ...],
        declared: bool,
    ) -> None:
        """Both branches: a declared pyramid, and direct-children-only."""
        image = _ramp((2, 8, 16, 16))
        path = _bioformats2raw_store(
            tmp_path / "labelled.zarr",
            zarr_format,
            [("0", [image])],
            image_attrs={"multiscales": _tzyx_multiscales(1)} if declared else None,
            labels=_ramp(mask_shape, start=50_000),
        )

        assert discover_ome_zarr_shape(path).shape == image.shape
        lazy = open_volume_lazy(path)
        assert tuple(lazy.shape) == image.shape
        # Data, not just shape: a tied mask has the image's shape exactly, so a
        # right-shape/wrong-array answer has to fail here.
        np.testing.assert_array_equal(np.asarray(lazy[...]), image)
        np.testing.assert_array_equal(load_volume(path), image.astype(np.float32))


class TestSelectionIsDeterministic:
    """A size TIE resolves the same way every time, and across processes.

    zarr does not stabilise ``Group.keys()`` order across opens, so a plain
    ``max()`` can flip between two equal-sized candidates. The batch planner
    reading a shape and a per-tile worker re-opening the store are different
    processes: disagreement there is the "plan says one shape, worker loads
    another" failure this whole rule exists to prevent. The tie-break is the
    LOWEST key path, which is a total order.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_tie_inside_an_image_group_resolves_to_the_lowest_key(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        # 2*8*16*16 == 4*4*16*16 — a genuine tie on element count.
        level_zero = _ramp((2, 8, 16, 16))
        level_one = _ramp((4, 4, 16, 16), start=20_000)
        path = _bioformats2raw_store(
            tmp_path / "tie.zarr", zarr_format, [("0", [level_zero, level_one])]
        )

        for _ in range(10):
            assert discover_ome_zarr_shape(path).shape == level_zero.shape
            assert tuple(open_volume_lazy(path).shape) == level_zero.shape
            np.testing.assert_array_equal(
                load_volume(path), level_zero.astype(np.float32)
            )

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_tie_in_the_whole_store_fallback_resolves_to_the_lowest_key(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """The recursive fallback — the path serving ``h2afva/fused`` layouts."""
        first = _ramp((2, 8, 16, 16))
        second = _ramp((4, 4, 16, 16), start=20_000)
        path = tmp_path / "tie_fallback.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        create_array(root.create_group("aaa"), "fused", data=first)
        create_array(root.create_group("bbb"), "fused", data=second)

        for _ in range(10):
            assert discover_ome_zarr_shape(path).shape == first.shape
            assert tuple(open_volume_lazy(path).shape) == first.shape
            np.testing.assert_array_equal(load_volume(path), first.astype(np.float32))


class TestTheRootBlockWinsUnlessTheOwnerDeclaresSomethingUsable:
    """Regression guards: adopting the owner's attributes must not lose the root's.

    ``axes`` / ``n_timepoints`` / ``n_channels`` / ``channel_shape`` drive
    ``batch-fit``'s whole T×C task fan-out, so a block adopted from the owning
    group because it merely EXISTS — rather than because it describes the array
    selected — produces a silently wrong plan, not an error.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_root_multiscales_survives_an_owner_axes_of_the_wrong_length(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "rootmeta.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        image = root.create_group("img")
        create_array(image, "0", data=_ramp((2, 8, 16, 16)))
        create_array(image, "1", data=_ramp((2, 4, 8, 8), start=40_000))
        # The owner declares SOMETHING, but nothing that could describe a 4D array.
        image.attrs["axes"] = ["a", "b", "c"]
        root.attrs["multiscales"] = _tzyx_multiscales(2, paths=["img/0", "img/1"])

        info = discover_ome_zarr_shape(path)

        assert info.shape == (2, 8, 16, 16)
        assert info.axes == ["t", "z", "y", "x"]
        # The 4D heuristic would answer CZYX: n_timepoints=1, n_channels=2.
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (2.0, 0.5, 0.5)
        assert info.unit == "micrometer"
        assert info.resolution_levels == 2

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_keller_lab_root_axes_survive_an_intermediate_groups_own_axes(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Root ``axes``, array at ``h2afva/fused``, owner with a short ``axes``."""
        path = tmp_path / "keller.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        group = root.create_group("h2afva")
        create_array(group, "fused", data=_ramp((2, 2, 3, 4, 4, 4)))
        group.attrs["axes"] = ["a", "b", "c"]
        root.attrs["axes"] = ["time", "camera", "channel", "z", "y", "x"]

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["time", "camera", "channel", "z", "y", "x"]
        assert info.n_timepoints == 2
        # camera x channel, folded into one flat channel task index.
        assert info.n_channels == 6
        assert info.channel_shape == (2, 3)
        assert info.spatial_shape == (4, 4, 4)


class TestABlankArrayKeyMeansNoKey:
    """``--array-key ""`` — typer hands back ``""``, not ``None``.

    ``node[""]`` is the ROOT group, so a blank key taken literally descends the
    whole store with an unscoped search and can answer with a different series
    entirely.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize("blank", ["", "/"])
    def test_a_blank_key_selects_exactly_what_no_key_selects(
        self, tmp_path: Path, zarr_format: int, blank: str
    ) -> None:
        first = _ramp((2, 4, 4, 4))
        second = _ramp((8, 16, 16, 16), start=10_000)
        path = _bioformats2raw_store(
            tmp_path / "multiseries.zarr",
            zarr_format,
            [("0", [first]), ("1", [second])],
        )

        assert discover_ome_zarr_shape(path, array_key=blank).shape == first.shape
        assert tuple(open_volume_lazy(path, blank).shape) == first.shape
        np.testing.assert_array_equal(
            load_volume(path, array_key=blank), first.astype(np.float32)
        )

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_non_string_key_is_a_value_error(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """A hand-edited manifest can carry one; it must not be an AttributeError."""
        path = _bioformats2raw_store(
            tmp_path / "bf.zarr", zarr_format, [("0", [_ramp((2, 4, 4, 4))])]
        )

        for call in (
            lambda: discover_ome_zarr_shape(path, array_key=0),  # type: ignore[arg-type]
            lambda: load_volume(path, array_key=0),  # type: ignore[arg-type]
            lambda: open_volume_lazy(path, 0),  # type: ignore[arg-type]
        ):
            with pytest.raises(ValueError, match="not a key path string"):
                call()


class TestAnArrayKeyNamingAnArray:
    """The other half of ``array_key``: it may name an array outright."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_nested_key_naming_a_pyramid_level_is_taken_verbatim(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """A DOWNSAMPLED level, so falling back to "largest" would fail."""
        full = _ramp((2, 8, 16, 16))
        half = _ramp((2, 4, 8, 8), start=40_000)
        path = _bioformats2raw_store(
            tmp_path / "levels.zarr", zarr_format, [("0", [full, half])]
        )

        assert discover_ome_zarr_shape(path, array_key="0/1").shape == half.shape
        assert tuple(open_volume_lazy(path, "0/1").shape) == half.shape
        np.testing.assert_array_equal(
            load_volume(path, array_key="0/1"), half.astype(np.float32)
        )


class TestAZippedArchive:
    """``.zarr.zip`` is a documented input and its ZipStore dispatch is a trap.

    This is the first change to route an archive through a nested group, and the
    store carries a bigger ``labels/`` mask too, so the selection rule that
    actually matters is exercised over the archive route.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_all_three_entry_points_read_the_level_inside_the_image_group(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        image = _ramp((2, 8, 16, 16))
        archive = _zip_store(
            _bioformats2raw_store(
                tmp_path / "bf.zarr",
                zarr_format,
                [("0", [image])],
                image_attrs={"multiscales": _tzyx_multiscales(1)},
                labels=_ramp((4, 8, 16, 16), start=50_000),
            )
        )

        info = discover_ome_zarr_shape(archive)
        assert info.shape == image.shape
        assert info.axes == ["t", "z", "y", "x"]
        assert tuple(open_volume_lazy(archive).shape) == image.shape
        np.testing.assert_array_equal(load_volume(archive), image.astype(np.float32))


class TestTheConsoleLine:
    """The loader names the array it landed on — the old line would be a lie."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_bioformats2raw_store_names_the_level_inside_the_group(
        self, tmp_path: Path, zarr_format: int, capsys: pytest.CaptureFixture
    ) -> None:
        path = _bioformats2raw_store(
            tmp_path / "bf.zarr", zarr_format, [("0", [_ramp((2, 4, 4, 4))])]
        )

        load_volume(path)

        out = capsys.readouterr().out
        assert "Using array '0/0'" in out
        assert "Detected OME-ZARR layout" not in out

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_plain_ome_ngff_store_still_reports_the_layout(
        self, tmp_path: Path, zarr_format: int, capsys: pytest.CaptureFixture
    ) -> None:
        path = tmp_path / "ngff.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        create_array(root, "0", data=_ramp((4, 4, 4)))

        load_volume(path)

        out = capsys.readouterr().out
        assert "Detected OME-ZARR layout" in out
        assert "Using array" not in out


class TestUnchangedBehaviour:
    """Regression guards: these already passed, and must keep passing."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_plain_ome_ngff_store_still_resolves_level_zero(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        full = _ramp((8, 8, 8))
        path = tmp_path / "ngff.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        create_array(root, "0", data=full)
        create_array(root, "1", data=full[::2, ::2, ::2])

        assert discover_ome_zarr_shape(path).shape == (8, 8, 8)
        node = open_volume_lazy(path)
        assert isinstance(node, zarr.Array)
        assert tuple(node.shape) == (8, 8, 8)
        np.testing.assert_array_equal(load_volume(path), full.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_store_with_no_arrays_at_all_still_raises(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "empty.zarr"
        zarr.open_group(str(path), mode="w", zarr_format=zarr_format)

        for call in (
            lambda: discover_ome_zarr_shape(path),
            lambda: load_volume(path),
            lambda: open_volume_lazy(path),
        ):
            with pytest.raises(ValueError, match="No arrays found"):
                call()
