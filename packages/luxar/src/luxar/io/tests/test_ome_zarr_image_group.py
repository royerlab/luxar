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

from luxar._zarr_compat import create_array, open_group
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
    root = open_group(path, mode="w", zarr_format=zarr_format)
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


_AXIS_TYPES = {"t": "time", "time": "time", "c": "channel", "channel": "channel"}


def _multiscales(
    axis_names: Sequence[str],
    paths: Sequence[str],
    scale: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """A valid NGFF ``multiscales`` block over ``axis_names``, declaring ``paths``.

    ``paths`` is what makes a block EVIDENCE about a particular array: the owning
    group's block only overrides the root's when one of these resolves to the
    array that was selected. A block can be perfectly well-formed and still be
    about something else.
    """
    names = list(axis_names)
    return [
        {
            "version": "0.4",
            "axes": [
                {
                    "name": name,
                    "type": _AXIS_TYPES.get(name, "space"),
                    **({"unit": "micrometer"} if name not in _AXIS_TYPES else {}),
                }
                for name in names
            ],
            "datasets": [
                {
                    "path": level_path,
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": list(scale)
                            if scale is not None
                            else [1.0] * len(names),
                        }
                    ],
                }
                for level_path in paths
            ],
        }
    ]


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
    return _multiscales(["t", "z", "y", "x"], level_paths, scale=[1.0, 2.0, 0.5, 0.5])


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
    def test_the_owner_wins_when_it_declares_the_selected_level(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """The feature itself: the owner DECLARES ``0/0``, the root names elsewhere.

        A bioformats2raw root carrying its own competing block over the same level
        is not a real layout; a root block about a DIFFERENT array is (a stale or
        hand-written attribute). The owner's declaration is the evidence that
        settles it, and this is the direction the evidence gate must not disable.
        """
        path = _bioformats2raw_store(
            tmp_path / "both.zarr",
            zarr_format,
            [("0", [_ramp((2, 8, 16, 16))])],
            image_attrs={"multiscales": _tzyx_multiscales(1)},
            root_attrs={"multiscales": _multiscales(["c", "z", "y", "x"], ["1/0"])},
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
        root = open_group(path, mode="w", zarr_format=zarr_format)
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
        root = open_group(path, mode="w", zarr_format=zarr_format)
        create_array(root, "a", data=_ramp((2, 2, 2)))

        for call in (
            lambda: discover_ome_zarr_shape(path, array_key="nope"),
            lambda: load_volume(path, array_key="nope"),
            lambda: open_volume_lazy(path, "nope"),
        ):
            with pytest.raises(ValueError, match="nope"):
                call()

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize(
        "bad_key", ["x" * 300, "../0", "a\x00b"], ids=["too_long", "dotdot", "nul"]
    )
    def test_a_key_zarr_itself_rejects_is_still_the_documented_value_error(
        self, tmp_path: Path, zarr_format: int, bad_key: str
    ) -> None:
        """Not every bad key is a ``KeyError``.

        zarr 3 raises ``ValueError`` for ``..`` segments and an embedded null byte,
        and the store raises ``OSError(ENAMETOOLONG)`` for an over-long segment. A
        ``except KeyError`` let those out raw, so a caller saw zarr's internals
        instead of the message that names the key and lists what is there.
        """
        path = tmp_path / "g.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        create_array(root, "a", data=_ramp((2, 2, 2)))

        for call in (
            lambda: discover_ome_zarr_shape(path, array_key=bad_key),
            lambda: load_volume(path, array_key=bad_key),
            lambda: open_volume_lazy(path, bad_key),
        ):
            with pytest.raises(ValueError) as excinfo:
                call()
            assert "Available keys" in str(excinfo.value)


class TestTheDeclaredLevelsBranch:
    """An image group's own ``multiscales`` block names its levels — branch 1."""

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_declared_level_beats_a_bigger_undeclared_sibling(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Without branch 1 the largest-direct-child rule would answer ``0/9``.

        A declaration is the principled answer even when something bigger sits
        beside it (a stray export, a working array): the block says which arrays
        are this image's pyramid.
        """
        declared = _ramp((2, 4, 4, 4))
        bigger = _ramp((4, 8, 8, 8), start=60_000)
        path = tmp_path / "declared.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image, "0", data=declared)
        create_array(image, "9", data=bigger)
        image.attrs["multiscales"] = _tzyx_multiscales(1)

        assert discover_ome_zarr_shape(path).shape == declared.shape
        np.testing.assert_array_equal(np.asarray(open_volume_lazy(path)[...]), declared)
        np.testing.assert_array_equal(load_volume(path), declared.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_numeric_path_and_its_string_alias_use_the_first_declaration(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Malformed aliases must not make selection and metadata disagree.

        The permissive path rule treats numeric ``0`` and string ``"./0"`` as
        the same level. The first declaration therefore remains authoritative
        for that level's scale, just as it is when both paths are strings.
        """
        data = _ramp((2, 4, 4, 4))
        block = _multiscales(["t", "z", "y", "x"], ["0", "./0"])
        datasets = block[0]["datasets"]
        datasets[0]["path"] = 0
        datasets[0]["coordinateTransformations"][0]["scale"] = [1, 7, 7, 7]
        datasets[1]["coordinateTransformations"][0]["scale"] = [1, 2, 0.5, 0.5]

        path = _bioformats2raw_store(
            tmp_path / "mixed_numeric_path.zarr",
            zarr_format,
            [("0", [data])],
            image_attrs={"multiscales": block},
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "z", "y", "x"]
        assert info.voxel_size == (7.0, 7.0, 7.0)
        np.testing.assert_array_equal(np.asarray(open_volume_lazy(path)[...]), data)
        np.testing.assert_array_equal(load_volume(path), data.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_all_numeric_pyramid_recovers_its_declared_metadata(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Numeric paths from a consistently malformed producer still resolve."""
        level_zero = _ramp((2, 8, 8, 8), start=10_000)
        level_one = _ramp((2, 4, 4, 4), start=20_000)
        block = _multiscales(["t", "z", "y", "x"], ["0", "1"], scale=[1, 2, 0.5, 0.5])
        for index, dataset in enumerate(block[0]["datasets"]):
            dataset["path"] = index

        path = _bioformats2raw_store(
            tmp_path / "numeric_pyramid.zarr",
            zarr_format,
            [("0", [level_zero, level_one])],
            image_attrs={"multiscales": block},
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "z", "y", "x"]
        assert info.voxel_size == (2.0, 0.5, 0.5)
        np.testing.assert_array_equal(
            np.asarray(open_volume_lazy(path)[...]), level_zero
        )
        np.testing.assert_array_equal(load_volume(path), level_zero.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_level_declared_one_group_deeper_keeps_its_declarers_metadata(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """``datasets[0].path == "res/0"``: the DECLARING group owns the metadata.

        Reporting the level's immediate parent (``0/res``) as the owner made
        discovery consult a group that declares nothing, so the array was selected
        correctly and then described by the 4D ``CZYX`` heuristic with no voxel
        size — the exact failure the owner override exists to avoid.
        """
        path = tmp_path / "deep_declared.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image.create_group("res"), "0", data=_ramp((2, 4, 4, 4)))
        image.attrs["multiscales"] = _tzyx_multiscales(1, paths=["res/0"])

        info = discover_ome_zarr_shape(path)

        assert info.shape == (2, 4, 4, 4)
        assert info.axes == ["t", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (2.0, 0.5, 0.5)
        assert info.unit == "micrometer"

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize("bad_path", ["../0/0", "nonexistent"])
    def test_a_declared_path_zarr_rejects_degrades_to_the_direct_children(
        self,
        tmp_path: Path,
        zarr_format: int,
        bad_path: str,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A ``..`` path is a ``ValueError`` from zarr 3, not a ``KeyError``.

        The docstring promises a half-written block degrades to the searches
        below; with ``except KeyError`` alone all three entry points raised
        instead, on a store whose ``0/0`` was right there.

        The block loses the evidence gate (it names nothing that resolves), so
        discovery ends on the heuristic — and the notice must say the store
        DECLARED something. "No OME-Zarr/NGFF metadata found" is a plain lie about
        a store carrying a well-formed TZYX block, and it hides the one detail
        that makes it fixable.
        """
        image_data = _ramp((2, 4, 4, 4))
        path = tmp_path / "badpath.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image, "0", data=image_data)
        image.attrs["multiscales"] = _tzyx_multiscales(1, paths=[bad_path])

        info = discover_ome_zarr_shape(path)
        out = capsys.readouterr().out
        assert info.shape == image_data.shape
        assert "no OME-Zarr/NGFF metadata found" not in out
        assert "declares a `multiscales` block that does not name it" in out

        np.testing.assert_array_equal(
            np.asarray(open_volume_lazy(path)[...]), image_data
        )
        np.testing.assert_array_equal(load_volume(path), image_data.astype(np.float32))


class TestLabelsAreNeverSelected:
    """Inside an IMAGE GROUP, a mask must never be mistaken for the image.

    ``0/labels/<name>/<level>`` is where NGFF puts a mask, so a recursive
    largest-array sweep of the image group reaches it. A mask that ties with level
    0 makes the answer flip; a mask one axis bigger makes it deterministically
    WRONG — ``luxar gsplat fit`` would then fit the mask, silently.

    Scoped deliberately: the guarantee is the image-group branch's (a ``"0"`` key,
    or an ``array_key`` naming a group). The whole-store fallback below — no
    ``"0"`` at all, the pre-existing ``h2afva/fused`` path — still sweeps
    ``labels/`` like any other group, unchanged.
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

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize(
        "declared_path",
        ["labels/seg/0", "aux/labels/seg/0"],
        ids=["direct-labels-group", "nested-labels-group"],
    )
    def test_a_declared_label_path_is_not_an_image_level(
        self, tmp_path: Path, zarr_format: int, declared_path: str
    ) -> None:
        """A stale declaration cannot override the image/mask boundary."""
        image = _ramp((2, 8, 16, 16))
        path = tmp_path / "declared_label.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image_group = root.create_group("0")
        create_array(image_group, "0", data=image)

        parent = image_group
        parts = declared_path.split("/")
        for part in parts[:-1]:
            parent = parent.create_group(part)
        create_array(parent, parts[-1], data=_ramp((4, 8, 16, 16), start=50_000))
        image_group.attrs["multiscales"] = _tzyx_multiscales(1, paths=[declared_path])

        for key in (None, "0"):
            assert discover_ome_zarr_shape(path, array_key=key).shape == image.shape
            np.testing.assert_array_equal(
                np.asarray(open_volume_lazy(path, key)[...]), image
            )
            np.testing.assert_array_equal(
                load_volume(path, array_key=key), image.astype(np.float32)
            )

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_declared_array_named_labels_is_still_a_level(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Only a traversed ``labels`` group is reserved, not an array leaf."""
        image = _ramp((2, 8, 16, 16))
        path = tmp_path / "labels_array.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image_group = root.create_group("0")
        create_array(image_group, "labels", data=image)
        image_group.attrs["multiscales"] = _tzyx_multiscales(1, paths=["labels"])

        for key in (None, "0"):
            assert discover_ome_zarr_shape(path, array_key=key).shape == image.shape
            np.testing.assert_array_equal(
                np.asarray(open_volume_lazy(path, key)[...]), image
            )
            np.testing.assert_array_equal(
                load_volume(path, array_key=key), image.astype(np.float32)
            )

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_mask_nested_one_group_deeper_is_skipped_too(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """The skip has to hold at EVERY depth, not just among direct children.

        Levels at ``0/sub/0`` reach the recursive branch, and that store's masks
        are one deeper too (``0/sub/labels/seg/0``). A first-level-only filter let
        the bigger mask win — for ``array_key=None`` and for ``array_key="0"``.
        """
        image = _ramp((2, 8, 16, 16))
        path = tmp_path / "nested_labels.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        sub = root.create_group("0").create_group("sub")
        create_array(sub, "0", data=image)
        create_array(
            sub.create_group("labels").create_group("seg"),
            "0",
            data=_ramp((4, 8, 16, 16), start=50_000),
        )

        for key in (None, "0"):
            assert discover_ome_zarr_shape(path, array_key=key).shape == image.shape
            np.testing.assert_array_equal(
                np.asarray(open_volume_lazy(path, key)[...]), image
            )
            np.testing.assert_array_equal(
                load_volume(path, array_key=key), image.astype(np.float32)
            )


class TestAnImageGroupWithNoArrayIsTerminal:
    """``"0"`` resolving to nothing must RAISE, not fall through to the sweep.

    The whole-store fallback is unscoped — it descends ``labels/`` at every depth
    and every other series — so falling into it from the image-group branch throws
    away the two guarantees that branch exists for. Both failures are plausible
    wrong ANSWERS rather than errors, while on the very same store an explicit
    ``--array-key 0`` crashed outright (``AttributeError: 'Group' object has no
    attribute 'shape'``) — so one store answered three different ways depending on
    how (or whether) the key was spelled. All three now raise the same clear
    ``ValueError``.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_image_group_holding_only_labels_does_not_select_the_mask(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """``0/labels/seg/0`` is a segmentation MASK; a fit would run against it."""
        path = tmp_path / "only_labels.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        _write_labels(root.create_group("0"), "seg", _ramp((2, 8, 16, 16), 50_000))

        for call in (
            lambda: discover_ome_zarr_shape(path),
            lambda: load_volume(path),
            lambda: open_volume_lazy(path),
            # The explicit spelling used to crash with an ``AttributeError``; it
            # must now raise, with the same message as the auto routes.
            lambda: discover_ome_zarr_shape(path, array_key="0"),
        ):
            with pytest.raises(ValueError, match="names a zarr group holding no array"):
                call()

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_empty_image_group_does_not_select_a_different_series(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Series ``1`` is a DIFFERENT image — selecting it defeats "first image"."""
        path = tmp_path / "empty_first.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        root.create_group("0")
        create_array(root.create_group("1"), "0", data=_ramp((3, 6, 6, 6), 99))

        for call in (
            lambda: discover_ome_zarr_shape(path),
            lambda: load_volume(path),
            lambda: open_volume_lazy(path),
            lambda: discover_ome_zarr_shape(path, array_key="0"),
        ):
            with pytest.raises(ValueError, match="names a zarr group holding no array"):
                call()

        # …and the caller can still reach the image it does hold.
        assert discover_ome_zarr_shape(path, array_key="1").shape == (3, 6, 6, 6)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_auto_route_does_not_blame_a_key_the_caller_never_passed(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """No ``--array-key`` was typed, so the message must not quote one.

        ``"0"`` was picked by the OME-NGFF convention, not by the caller. "Array
        key '0' names a zarr group holding no array" sends the reader hunting
        their own command line for a ``0`` that is not in it. The keyed spelling,
        where a key WAS typed, must keep quoting it.
        """
        path = tmp_path / "stub_first.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        root.create_group("0")
        create_array(root.create_group("1"), "0", data=_ramp((3, 6, 6, 6), 99))

        with pytest.raises(ValueError) as auto:
            discover_ome_zarr_shape(path)
        assert "OME-NGFF resolution level '0'" in str(auto.value)
        assert "Array key" not in str(auto.value)

        with pytest.raises(ValueError) as keyed:
            discover_ome_zarr_shape(path, array_key="0")
        assert "Array key '0'" in str(keyed.value)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_terminal_error_names_the_arrays_the_store_does_hold(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Unrecoverable without a key, so the message names keys that WORK.

        ``"0"`` is an empty stub and the real image is at ``"1"``. Listing the
        root's members (``['0', '1', 'OME']``) leaves the reader to guess which of
        those is an image and how deep its levels sit; the sweep answers with the
        key they can paste.
        """
        path = tmp_path / "stub_then_real.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        root.create_group("OME")
        root.create_group("0")
        create_array(root.create_group("1"), "0", data=_ramp((3, 6, 6, 6), 99))

        with pytest.raises(ValueError) as excinfo:
            discover_ome_zarr_shape(path)
        message = str(excinfo.value)
        assert "Arrays this store does hold: '1/0'" in message

        # And it is a key that actually resolves — the whole point of quoting it.
        assert discover_ome_zarr_shape(path, array_key="1/0").shape == (3, 6, 6, 6)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_store_with_no_array_anywhere_says_so_instead_of_suggesting_nothing(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Advice to pass an array_key is unfollowable when the store holds none."""
        path = tmp_path / "no_arrays.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        root.create_group("0").create_group("res")

        with pytest.raises(ValueError, match="holds no array anywhere"):
            discover_ome_zarr_shape(path)


class TestTheOwnerBlockIsResolvedOnce:
    """``owner_declares`` exists to spare the owner a SECOND resolution.

    ``_image_group_array`` reports ``declares=False`` only after running
    ``_declared_levels`` on that very group and finding nothing resolvable. Asking
    ``_declaring_owner`` to start the walk AT that group re-opens it and re-walks
    every ``datasets[*].path`` in its block — reintroducing exactly the doubling
    the plumbed-through fact was introduced to avoid.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_stale_block_is_walked_once_not_twice(
        self, tmp_path: Path, zarr_format: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "stale_block.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        for level in range(4):
            create_array(image, str(level), data=_ramp((2, 4, 4, 4)))
        # A stale hand-written block: four entries, none of which resolve.
        image.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], [f"gone/{level}" for level in range(4)]
        )

        lookups: List[str] = []
        original = zarr.Group.__getitem__

        def spy(self: zarr.Group, key: str) -> Any:
            lookups.append(f"{str(getattr(self, 'path', '') or '/')}::{key}")
            return original(self, key)

        monkeypatch.setattr(zarr.Group, "__getitem__", spy)
        info = discover_ome_zarr_shape(path)
        monkeypatch.undo()

        # The selection is unchanged — the point is the cost, not the answer.
        assert info.shape == (2, 4, 4, 4)
        for level in range(4):
            assert lookups.count(f"0::gone/{level}") == 1, lookups
        assert len(lookups) == len(set(lookups)), (
            f"{len(lookups) - len(set(lookups))} repeated lookups: {lookups}"
        )


class TestEveryKeySpellingReportsTheSameOwner:
    """One array, one owner — whatever the caller typed to reach it.

    ``_image_group_array`` reports the group it happened to SEARCH when that
    group's own block declared nothing, which is not necessarily the group that
    DECLARES the array. On a store whose image group ``"0"`` declares
    ``datasets[0].path == "res/0"``, ``--array-key 0/res`` searched ``res``,
    adopted it as owner, found no declaration there and fell through to the ndim
    heuristic — a different ``batch-fit`` T×C fan-out for the same array, decided
    by nothing but the spelling.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_all_four_spellings_of_a_deeply_declared_level_agree(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "spellings.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image.create_group("res"), "0", data=_ramp((2, 4, 4, 4)))
        image.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], ["res/0"], scale=[1.0, 2.0, 0.3, 0.3]
        )

        for key in (None, "0", "0/res", "0/res/0"):
            info = discover_ome_zarr_shape(path, array_key=key)

            # The DECLARED values, not merely the same values four times: the
            # heuristic answers ['c','z','y','x'], (1, 2) and no spacing at all.
            assert info.shape == (2, 4, 4, 4), key
            assert info.axes == ["t", "z", "y", "x"], key
            assert (info.n_timepoints, info.n_channels) == (2, 1), key
            assert info.voxel_size == (2.0, 0.3, 0.3), key

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_no_key_sweep_reads_the_declaration_a_key_would_have(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """No ``"0"`` at the root, so the WHOLE-STORE sweep decides.

        That sweep reports the level's IMMEDIATE PARENT (``img/res``), which
        declares nothing — the declaring group is ``img``, one further up. Every
        keyed spelling re-derived the owner and read the declaration; the no-key
        spelling adopted the sweep's report verbatim, so this one store answered
        ``['c','z','y','x']`` with no spacing when nothing was typed and
        ``['t','z','y','x']`` at (7.0, 0.25, 0.25) when anything was — the same
        selected array either way (``img/res/0``), fanned out over a different
        axis by ``batch-fit``.
        """
        path = tmp_path / "sweep_spellings.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("img")
        levels = image.create_group("res")
        create_array(levels, "0", data=_ramp((2, 4, 4, 4)))
        create_array(levels, "1", data=_ramp((2, 2, 2, 2)))
        image.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], ["res/0", "res/1"], scale=[1.0, 7.0, 0.25, 0.25]
        )

        for key in (None, "img", "img/res", "img/res/0"):
            info = discover_ome_zarr_shape(path, array_key=key)

            # Pinned against the DECLARED values, not against mutual agreement:
            # four spellings could agree on the heuristic's guess and still be
            # wrong about the store.
            assert info.shape == (2, 4, 4, 4), key
            assert info.axes == ["t", "z", "y", "x"], key
            assert (info.n_timepoints, info.n_channels) == (2, 1), key
            assert info.voxel_size == (7.0, 0.25, 0.25), key

        # …and the array that was selected is the one those values describe.
        assert tuple(open_volume_lazy(path).shape) == (2, 4, 4, 4)


class TestTheOwnerIsAskedWhenTheSelectionNeverDecided:
    """A block one group away from the selected array is still read.

    In both stores below the owner's ``multiscales`` is the ONLY metadata there
    is, and neither owner is the group the naive rules would consult: the first is
    never inspected by the selection at all, the second is a group BELOW the image
    group. Missing either drops the store onto the 4D ``CZYX`` heuristic, T and C
    the wrong way round and no spacing.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_whole_store_sweep_asks_the_owner_it_never_inspected(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """The ``h2afva/fused`` layout: no ``"0"`` key, so the sweep decides.

        The one route that reports ``owner_declares`` as UNDETERMINED — the sweep
        looked at no ``multiscales`` block — so reading that ``None`` as "does not
        declare it" instead of ASKING loses this store's metadata outright.
        """
        path = tmp_path / "sweep_owner.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        group = root.create_group("h2afva")
        create_array(group, "fused", data=_ramp((2, 4, 4, 4)))
        group.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], ["fused"], scale=[1.0, 2.0, 0.5, 0.5]
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (2.0, 0.5, 0.5)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_level_nested_below_the_image_group_asks_its_own_parent(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Levels at ``0/sub/0`` with the block on ``0/sub`` declaring ``"0"``.

        The image-group search reaches this array through its recursive branch and
        reports ``0/sub`` as the owner without ever having looked at its block;
        the owner-resolution rule then asks, and finds the declaration there.
        """
        path = tmp_path / "nested_owner.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        sub = root.create_group("0").create_group("sub")
        create_array(sub, "0", data=_ramp((2, 4, 4, 4)))
        sub.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], ["0"], scale=[1.0, 3.0, 0.25, 0.25]
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (3.0, 0.25, 0.25)


class TestTheNearestDeclaringAncestorWins:
    """Two ancestors declare the same array; the NEAREST one describes it.

    The walk stops at the first declaring ancestor, which is not merely a cost
    optimisation — on a bioformats2raw store the root's block habitually names
    every series' level 0 (``"0/0"``, ``"1/0"``, …) at the coarse whole-plate
    spacing while each image group's own block carries that series' real axes and
    scale. Preferring the farthest declaration (the store root) hands the image
    the wrong block: same arity, T and C transposed, and a voxel size off by 30x.
    Nothing pinned this, so "nearest" could be silently rewritten to "farthest".
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize("array_key", [None, "0", "0/0"])
    def test_the_image_groups_own_block_beats_the_roots(
        self, tmp_path: Path, zarr_format: int, array_key: Optional[str]
    ) -> None:
        path = tmp_path / "both_declare.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image, "0", data=_ramp((2, 4, 4, 4)))
        # The nearest declaration: the image group, naming its level relative to
        # itself.
        image.attrs["multiscales"] = _multiscales(
            ["t", "z", "y", "x"], ["0"], scale=[1.0, 2.0, 0.3, 0.3]
        )
        # The farthest one: the store ROOT, naming the very same array through
        # the image group, with T/C the other way round and a coarse spacing.
        root.attrs["multiscales"] = _multiscales(
            ["c", "z", "y", "x"], ["0/0"], scale=[1.0, 9.0, 9.0, 9.0]
        )

        info = discover_ome_zarr_shape(path, array_key=array_key)

        assert info.shape == (2, 4, 4, 4)
        assert info.axes == ["t", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (2.0, 0.3, 0.3)


class TestSelectionIsDeterministic:
    """A size TIE resolves the same way on every fresh open (one process here).

    zarr gives no guarantee that ``Group.keys()`` enumerates members in a stable
    order, so a plain ``max()`` could flip between two equal-sized candidates. The
    batch planner reading a shape and a per-tile worker re-opening the store are
    different processes: disagreement there is the "plan says one shape, worker
    loads another" failure this whole rule exists to prevent. The tie-break is the
    LOWEST key path, a total order, so it cannot depend on enumeration at all —
    which is why repeated fresh opens IN ONE PROCESS are a sufficient test and no
    subprocess is spawned.
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
        root = open_group(path, mode="w", zarr_format=zarr_format)
        create_array(root.create_group("aaa"), "fused", data=first)
        create_array(root.create_group("bbb"), "fused", data=second)

        for _ in range(10):
            assert discover_ome_zarr_shape(path).shape == first.shape
            assert tuple(open_volume_lazy(path).shape) == first.shape
            np.testing.assert_array_equal(load_volume(path), first.astype(np.float32))

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_tie_is_broken_on_the_key_path_not_on_the_walk_order(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """The tie-break DIRECTION, on a store where the two orders disagree.

        Sorted member order visits ``a`` before ``a-b`` (``"a"`` is a prefix), but
        as whole key paths ``"a-b/y" < "a/x"`` (``-`` sorts below ``/``). So a
        "largest, first one wins" rule answers ``a/x`` and the documented
        lowest-key-path rule answers ``a-b/y`` — the only shape of store that can
        tell the two apart.
        """
        walked_first = _ramp((4, 4, 4))
        lowest_path = _ramp((4, 4, 4), start=30_000)
        path = tmp_path / "tiebreak.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        create_array(root.create_group("a"), "x", data=walked_first)
        create_array(root.create_group("a-b"), "y", data=lowest_path)

        np.testing.assert_array_equal(
            np.asarray(open_volume_lazy(path)[...]), lowest_path
        )
        np.testing.assert_array_equal(load_volume(path), lowest_path.astype(np.float32))


class TestAFalsyOwnerBlockIsStillADeclaration:
    """Nothing-declared vs declared-but-unusable cannot depend on the NODE.

    The give-up notice's whole job is telling those two apart, and a PRESENT but
    empty ``multiscales`` is the second one. Reading the owner's block with a bare
    truthiness test collapsed it into the first, so the identical malformed
    declaration was reported one way sitting on the root and another sitting on
    the image group — "no OME-Zarr/NGFF metadata found" being a lie about a store
    that declared something.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize(
        "attrs",
        [{"multiscales": []}, {"ome": {"version": "0.5", "multiscales": []}}],
        ids=["v04-top-level", "v05-nested"],
    )
    def test_an_empty_owner_multiscales_reports_the_same_reason_as_the_root(
        self,
        tmp_path: Path,
        zarr_format: int,
        attrs: Dict[str, Any],
        capsys: pytest.CaptureFixture,
    ) -> None:
        reasons = []
        for where in ("owner", "root"):
            path = tmp_path / f"empty_block_{where}.zarr"
            root = open_group(path, mode="w", zarr_format=zarr_format)
            image = root.create_group("0")
            create_array(image, "0", data=_ramp((2, 4, 4, 4)))
            for key, value in attrs.items():
                (image if where == "owner" else root).attrs[key] = value

            info = discover_ome_zarr_shape(path)
            out = capsys.readouterr().out

            assert info.shape == (2, 4, 4, 4), where
            assert "no OME-Zarr/NGFF metadata found" not in out, where
            assert "empty or not a list of blocks" in out, where
            reasons.append(out[out.index("(") : out.index(")") + 1])

        assert reasons[0] == reasons[1]

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_explicit_null_multiscales_is_still_nothing_declared(
        self, tmp_path: Path, zarr_format: int, capsys: pytest.CaptureFixture
    ) -> None:
        """The one falsy value that is NOT a declaration — matching the root rule.

        ``_usable_multiscales`` reads a ``null`` as "the store declared no
        ``multiscales`` at all", so the owner has to as well or the two nodes
        disagree again, in the other direction.
        """
        path = tmp_path / "null_block.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image, "0", data=_ramp((2, 4, 4, 4)))
        image.attrs["multiscales"] = None

        discover_ome_zarr_shape(path)

        assert "no OME-Zarr/NGFF metadata found" in capsys.readouterr().out


class TestTheRootBlockWinsUnlessTheOwnerDeclaresTheSelectedArray:
    """Regression guards: adopting the owner's attributes must not lose the root's.

    ``axes`` / ``n_timepoints`` / ``n_channels`` / ``channel_shape`` drive
    ``batch-fit``'s whole T×C task fan-out, so a block adopted from the owning
    group because it merely EXISTS — rather than because it declares the array
    selected — produces a silently wrong plan, not an error. A matching axis COUNT
    is not evidence: a permutation has exactly the right length.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_permuted_owner_block_declaring_nothing_loses_to_the_root(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Right arity, wrong order, no declaration — the root's T/C must survive.

        With the gate on arity alone this store planned 10 timepoints x 3 channels
        instead of 3 x 10, so 21 of 30 ``batch-fit`` tasks died on a bounds check.
        """
        path = _bioformats2raw_store(
            tmp_path / "permuted.zarr",
            zarr_format,
            [("0", [_ramp((3, 10, 4, 5, 5))])],
            # A stale hand-written block: it names a level that is not there, so it
            # cannot be evidence about the array actually selected.
            image_attrs={
                "multiscales": _multiscales(["c", "t", "z", "y", "x"], ["stale"])
            },
            root_attrs={
                "multiscales": _multiscales(["t", "c", "z", "y", "x"], ["0/0"])
            },
        )

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["t", "c", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (3, 10)
        assert info.channel_shape == (10,)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_the_keller_lab_root_axes_survive_a_permuted_owner_axes(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """A bare ``axes`` list names nothing, so no evidence about it exists.

        Hence root-first for that key: an owner list of the same LENGTH would
        otherwise reorder the decomposition (T 2→4, C 6→16, spatial (4,4,4)→(2,2,3))
        on a store the root already had right.
        """
        path = tmp_path / "keller_permuted.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        group = root.create_group("h2afva")
        create_array(group, "fused", data=_ramp((2, 2, 3, 4, 4, 4)))
        group.attrs["axes"] = ["z", "y", "x", "time", "camera", "channel"]
        root.attrs["axes"] = ["time", "camera", "channel", "z", "y", "x"]

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["time", "camera", "channel", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 6)
        assert info.channel_shape == (2, 3)
        assert info.spatial_shape == (4, 4, 4)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_owner_axes_is_still_used_when_the_root_has_none(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Root-first is a PRECEDENCE, not a removal: the fallback still fires."""
        path = tmp_path / "owner_axes_only.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        group = root.create_group("h2afva")
        create_array(group, "fused", data=_ramp((2, 2, 3, 4, 4, 4)))
        group.attrs["axes"] = ["time", "camera", "channel", "z", "y", "x"]

        info = discover_ome_zarr_shape(path)

        assert info.axes == ["time", "camera", "channel", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 6)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    @pytest.mark.parametrize(
        "datasets",
        [{"0": {"path": "0"}}, ["0"]],
        ids=["mapping", "strings"],
    )
    def test_a_malformed_owner_datasets_falls_back_instead_of_raising(
        self,
        tmp_path: Path,
        zarr_format: int,
        datasets: Any,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """No entry of such a block can NAME an array, so it is never evidence.

        Neither a mapping nor a list of bare strings can declare a level, so the
        owner's block loses the evidence gate and discovery falls back (here: the
        4D ``CZYX`` heuristic), exactly as it did before the owner was consulted
        at all — but the notice must SAY that a block was declared, since "no
        metadata found" is a lie about this store.
        """
        block = [
            {
                "version": "0.4",
                "axes": [{"name": name} for name in ("t", "z", "y", "x")],
                "datasets": datasets,
            }
        ]
        path = _bioformats2raw_store(
            tmp_path / "malformed.zarr",
            zarr_format,
            [("0", [_ramp((2, 4, 4, 4))])],
            image_attrs={"multiscales": block},
        )

        info = discover_ome_zarr_shape(path)

        assert info.shape == (2, 4, 4, 4)
        assert info.axes == ["c", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (1, 2)
        out = capsys.readouterr().out
        assert "no OME-Zarr/NGFF metadata found" not in out
        assert "declares a `multiscales` block that does not name it" in out

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_mixed_owner_datasets_list_is_still_read_when_it_names_the_array(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """``["0", {"path": "0", …}]``: the dict entry is what selected the array.

        This is the ONE malformed shape that can reach the owner test with
        evidence in hand, and the parser reads it completely — axes, level count
        and spacing. A shape gate on ``datasets`` therefore only ever discarded a
        block that WAS about the selected array, dropping this 4D ``TZYX`` store
        onto the ``CZYX`` heuristic: the silently wrong ``batch-fit`` fan-out the
        gate was meant to prevent, caused by the gate.
        """
        block = [
            {
                "version": "0.4",
                "axes": [
                    {"name": name, "type": _AXIS_TYPES.get(name, "space")}
                    for name in ("t", "z", "y", "x")
                ],
                "datasets": [
                    "0",
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 2.0, 0.5, 0.5]}
                        ],
                    },
                ],
            }
        ]
        path = _bioformats2raw_store(
            tmp_path / "mixed.zarr",
            zarr_format,
            [("0", [_ramp((2, 4, 4, 4))])],
            image_attrs={"multiscales": block},
        )

        info = discover_ome_zarr_shape(path)

        assert info.shape == (2, 4, 4, 4)
        assert info.axes == ["t", "z", "y", "x"]
        assert (info.n_timepoints, info.n_channels) == (2, 1)
        assert info.voxel_size == (2.0, 0.5, 0.5)

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_root_multiscales_survives_an_owner_axes_of_the_wrong_length(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "rootmeta.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
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
    def test_the_keller_lab_root_axes_survive_an_owner_axes_of_the_wrong_length(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        """Root ``axes``, array at ``h2afva/fused``, owner with a short ``axes``.

        The unusable-length case, kept beside the same-length PERMUTATION above:
        those fail for different reasons (unusable vs no-evidence-obtainable) and
        only one of them was ever caught.
        """
        path = tmp_path / "keller.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
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


class TestAnExplicitKeyDescribesTheArrayTheWayNoKeyDoes:
    """Naming the array explicitly must not lose the metadata that describes it.

    ``_select_zarr_array``'s whole contract is that its three callers agree about
    one store, and ``discover_ome_zarr_shape`` reads the OWNER it reports. The
    auto-selection paths hand back the DECLARING group; the explicit-key branch
    handed back the array's immediate parent, which for a level declared one group
    deeper (``datasets[0].path == "res/0"``) declares nothing — so
    ``--array-key 0/res/0`` fell through to the ndim heuristic and planned a
    different ``batch-fit`` T×C fan-out than the same store planned with no key.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_a_deep_explicit_key_keeps_the_declaring_groups_metadata(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        path = tmp_path / "deep_key.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image.create_group("res"), "0", data=_ramp((3, 10, 4, 5, 5)))
        # A CTZYX block — the heuristic's 5D guess is TCZYX, so adopting the
        # declaration is what puts 10 timepoints and 3 channels the right way
        # round (the heuristic answers 3 x 10 and no voxel size at all).
        image.attrs["multiscales"] = _multiscales(
            ["c", "t", "z", "y", "x"], ["res/0"], scale=[1.0, 1.0, 9.0, 9.0, 9.0]
        )

        for key in (None, "0", "0/res/0"):
            info = discover_ome_zarr_shape(path, array_key=key)

            # Asserted against the DECLARED values, not just across the three
            # calls: agreeing on the heuristic's answer is not agreement.
            assert info.shape == (3, 10, 4, 5, 5), key
            assert info.axes == ["c", "t", "z", "y", "x"], key
            assert (info.n_timepoints, info.n_channels) == (10, 3), key
            assert info.channel_shape == (3,), key
            assert info.voxel_size == (9.0, 9.0, 9.0), key


class TestADroppedDeclaredLevelIsReported:
    """A declared level that will not resolve is skipped — and must SAY so.

    The survivors of a half-written block can perfectly well be a DOWNSAMPLED
    level, so a silent skip reports a plausible wrong shape: half the resolution
    the store actually holds, with no message anywhere to explain it.
    """

    @pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
    def test_an_unresolvable_declared_level_is_named_on_the_console(
        self, tmp_path: Path, zarr_format: int, capsys: pytest.CaptureFixture
    ) -> None:
        full = _ramp((2, 8, 16, 16))
        half = _ramp((2, 4, 8, 8), start=40_000)
        path = tmp_path / "half_written.zarr"
        root = open_group(path, mode="w", zarr_format=zarr_format)
        image = root.create_group("0")
        create_array(image, "0", data=full)
        create_array(image, "1", data=half)
        # Level 0 declared through a path zarr itself rejects — the portable
        # stand-in for a level whose metadata document is corrupt or momentarily
        # unreadable on a remote backend.
        image.attrs["multiscales"] = _tzyx_multiscales(2, paths=["../0/0", "1"])

        info = discover_ome_zarr_shape(path)

        out = capsys.readouterr().out
        assert "Skipping declared level" in out
        assert "../0/0" in out
        # Still lands on a declared level of this image — the half-resolution one,
        # which is exactly why staying quiet about it is not an option.
        assert info.shape == half.shape
        assert info.axes == ["t", "z", "y", "x"]


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
        root = open_group(path, mode="w", zarr_format=zarr_format)
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
        root = open_group(path, mode="w", zarr_format=zarr_format)
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
        open_group(path, mode="w", zarr_format=zarr_format)

        for call in (
            lambda: discover_ome_zarr_shape(path),
            lambda: load_volume(path),
            lambda: open_volume_lazy(path),
        ):
            with pytest.raises(ValueError, match="No arrays found"):
                call()
