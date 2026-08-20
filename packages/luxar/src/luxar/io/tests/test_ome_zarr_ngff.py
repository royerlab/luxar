"""Both OME-Zarr metadata layouts must parse — 0.4 top-level and 0.5 nested.

OME-Zarr 0.5 (what a zarr v3 store declares) moved the whole NGFF block one
level down, under an ``ome`` key. ``discover_ome_zarr_shape`` only read the 0.4
spelling, and the failure was SILENT: no ``multiscales`` was found, the custom
``axes`` fallback did not match either, and discovery landed on the shape
heuristic — axis roles guessed from ndim and no physical voxel size. On a 0.5
``TZYX`` store the 4D heuristic reads the TIME axis as a channel, which fans a
batch plan out over the wrong axis while reporting a perfectly ordinary-looking
``OMEZarrInfo``.

Both layouts are written at BOTH zarr formats here: attribute nesting is an
OME-Zarr version thing and is independent of the zarr format, so a zarr v3 store
written by a 0.4-era tool carries the v3 chunk layout with 0.4 attributes — and
0.4 attributes on a zarr **format 2** store are the commonest real store there
is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import typer

from luxar._zarr_compat import create_array, open_group
from luxar.io.ome_zarr import discover_ome_zarr_shape, resolve_ngff_attrs

_TCZYX = ("t", "c", "z", "y", "x")
_TZYX = ("t", "z", "y", "x")
_CZYX = ("c", "z", "y", "x")
_ZYX = ("z", "y", "x")


def _axes_meta(labels: Sequence[Any]) -> List[Dict[str, Any]]:
    """NGFF ``axes`` records for ``labels`` (spatial axes carry the unit).

    A plain string is given the type its NAME implies; a ``(name, type)`` tuple
    spells the two out separately, which is the whole point of several tests here
    — NGFF classifies by ``type``, and name and type routinely disagree.
    """
    out: List[Dict[str, Any]] = []
    for label in labels:
        if isinstance(label, tuple):
            name, atype = label
            record: Dict[str, Any] = {"name": name, "type": atype}
            if atype == "space":
                record["unit"] = "micrometer"
            out.append(record)
        elif label == "t":
            out.append({"name": "t", "type": "time", "unit": "second"})
        elif label == "c":
            out.append({"name": "c", "type": "channel"})
        else:
            out.append({"name": label, "type": "space", "unit": "micrometer"})
    return out


def _multiscales(
    labels: Sequence[Any],
    scale: Sequence[float],
    *,
    extra_levels: Optional[Sequence[Sequence[float]]] = None,
    ms_scale: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(s) for s in scale]}
            ],
        }
    ]
    for i, level_scale in enumerate(extra_levels or (), start=1):
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(s) for s in level_scale]}
                ],
            }
        )
    ms: Dict[str, Any] = {"axes": _axes_meta(labels), "datasets": datasets}
    if ms_scale is not None:
        ms["coordinateTransformations"] = [
            {"type": "scale", "scale": [float(s) for s in ms_scale]}
        ]
    return [ms]


def _write_store(
    path: Path,
    shape: Sequence[int],
    *,
    labels: Optional[Sequence[Any]] = None,
    scale: Optional[Sequence[float]] = None,
    nested: bool = True,
    zarr_format: int = 3,
    ome_attr: Any = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
    extra_levels: Optional[Sequence[Sequence[float]]] = None,
    extra_level_shapes: Optional[Sequence[Sequence[int]]] = None,
    ms_scale: Optional[Sequence[float]] = None,
) -> Path:
    """Write a single-level store, optionally carrying an NGFF block.

    ``nested=True`` writes the 0.5 spelling (everything under ``ome``);
    ``nested=False`` the 0.4 one (``multiscales`` at the top level).
    ``ome_attr`` writes a literal ``ome`` attribute instead, for the malformed
    cases; ``extra_attrs`` writes arbitrary top-level attributes alongside.
    """
    root = open_group(path, mode="w", zarr_format=zarr_format)
    create_array(
        root, "0", data=np.zeros(tuple(shape), dtype=np.float32), compressor="auto"
    )
    for i, level_shape in enumerate(extra_level_shapes or (), start=1):
        create_array(
            root,
            str(i),
            data=np.zeros(tuple(level_shape), dtype=np.float32),
            compressor="auto",
        )
    if labels is not None:
        assert scale is not None
        block = {
            "version": "0.5",
            "multiscales": _multiscales(
                labels, scale, extra_levels=extra_levels, ms_scale=ms_scale
            ),
        }
        if nested:
            root.attrs["ome"] = block
        else:
            root.attrs["multiscales"] = block["multiscales"]
    if ome_attr is not None:
        root.attrs["ome"] = ome_attr
    for key, value in (extra_attrs or {}).items():
        root.attrs[key] = value
    return path


# ---------------------------------------------------------------------------
# The resolver itself
# ---------------------------------------------------------------------------


def test_resolve_returns_the_nested_block_when_it_carries_multiscales() -> None:
    block = {"multiscales": [{"axes": []}], "version": "0.5"}
    assert resolve_ngff_attrs({"ome": block}) == block


@pytest.mark.parametrize(
    "attrs",
    [
        {"multiscales": ["top-level"]},  # 0.4: no `ome` key at all
        {"ome": "0.5", "multiscales": ["top-level"]},  # `ome` is not a mapping
        {"ome": {"version": "0.5"}, "multiscales": ["top-level"]},  # no NGFF key
        # The `omero` trap: a nested block carrying only rendering metadata must
        # NOT win over a perfectly good top-level pyramid.
        {
            "ome": {"version": "0.5", "omero": {"channels": []}},
            "multiscales": ["top-level"],
        },
    ],
)
def test_resolve_falls_back_to_the_top_level(attrs: Dict[str, Any]) -> None:
    assert resolve_ngff_attrs(attrs)["multiscales"] == ["top-level"]


# ---------------------------------------------------------------------------
# Discovery on real stores
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_nested_ome_zarr_05_metadata_is_recovered(
    tmp_path: Path, zarr_format: int
) -> None:
    """A 0.5 store: axes, T, C, spatial shape, voxel size and unit all read.

    5D TCZYX, where the heuristic happens to guess the same axes — so this pins
    the metadata PATH end to end, not the T/C flip. The discriminating cases are
    the 4D ``TZYX`` store below and the batch-planner test at the bottom.
    """
    path = _write_store(
        tmp_path / f"v{zarr_format}.zarr",
        (5, 2, 32, 64, 64),
        labels=_TCZYX,
        scale=(1.0, 1.0, 0.5, 0.3, 0.3),
        nested=True,
        zarr_format=zarr_format,
    )

    info = discover_ome_zarr_shape(path)

    assert info.axes == list(_TCZYX)
    assert info.n_timepoints == 5
    assert info.n_channels == 2
    assert info.channel_axes == ["c"]
    assert info.channel_shape == (2,)
    assert info.spatial_shape == (32, 64, 64)
    assert info.spatial_axes == ["z", "y", "x"]
    assert info.voxel_size == (0.5, 0.3, 0.3)
    assert info.unit == "micrometer"
    assert info.resolution_levels == 1


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_the_04_top_level_layout_still_parses_identically(
    tmp_path: Path, zarr_format: int
) -> None:
    """Regression guard: reading 0.5 must not have moved 0.4's ground.

    Both zarr formats, because 0.4 attributes on a zarr format 2 store are the
    commonest real OME-Zarr on disk and the nesting is orthogonal to the format.
    """
    shape = (5, 2, 32, 64, 64)
    scale = (1.0, 1.0, 0.5, 0.3, 0.3)
    flat = discover_ome_zarr_shape(
        _write_store(
            tmp_path / "v04.zarr",
            shape,
            labels=_TCZYX,
            scale=scale,
            nested=False,
            zarr_format=zarr_format,
        )
    )
    nested = discover_ome_zarr_shape(
        _write_store(
            tmp_path / "v05.zarr",
            shape,
            labels=_TCZYX,
            scale=scale,
            nested=True,
            zarr_format=zarr_format,
        )
    )

    assert flat.voxel_size == (0.5, 0.3, 0.3)
    assert flat.unit == "micrometer"
    # Everything but the store path is the same fact about the same data.
    for field in (
        "axes",
        "shape",
        "n_timepoints",
        "n_channels",
        "channel_axes",
        "channel_shape",
        "spatial_shape",
        "spatial_axes",
        "voxel_size",
        "unit",
        "resolution_levels",
    ):
        assert getattr(flat, field) == getattr(nested, field), field


def test_a_nested_tzyx_store_keeps_its_time_axis(tmp_path: Path) -> None:
    """The concrete scientific failure: 4D ``TZYX`` read as ``CZYX``.

    The heuristic labels any 4D store ``c,z,y,x``, so an unparsed 0.5 timelapse
    plans as T=1, C=7 — the whole batch fan-out over the wrong axis.
    """
    path = _write_store(
        tmp_path / "tzyx.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=True,
    )

    info = discover_ome_zarr_shape(path)

    assert info.axes == list(_TZYX)
    assert info.n_timepoints == 7
    assert info.n_channels == 1
    assert info.channel_axes == []
    assert info.spatial_shape == (16, 32, 32)
    assert info.voxel_size == (2.0, 0.325, 0.325)


def test_a_top_level_pyramid_survives_an_omero_only_ome_block(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A 0.4 pyramid must not be discarded because a 0.5-ish `ome` block exists.

    Selecting the nested block on the strength of ``omero`` alone reintroduces
    the very bug this module guards, from the other side: the top-level
    ``multiscales`` is thrown away and the 4D heuristic reads T=7 as C=7.
    """
    path = _write_store(
        tmp_path / "omero.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=False,
        extra_attrs={"ome": {"version": "0.5", "omero": {"channels": []}}},
    )

    info = discover_ome_zarr_shape(path)

    assert info.axes == list(_TZYX)
    assert info.n_timepoints == 7
    assert info.n_channels == 1
    assert info.voxel_size == (2.0, 0.325, 0.325)
    assert "GUESSED" not in capsys.readouterr().out


def test_the_demo_voxel_size_reader_survives_the_same_store(tmp_path: Path) -> None:
    """``voxel_size_of`` shares the resolver, so it shares the ``omero`` trap."""
    from luxar.demos.demo_gsplats_4d_cell_tracking_challenge import voxel_size_of

    path = _write_store(
        tmp_path / "omero_demo.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=False,
        extra_attrs={"ome": {"version": "0.5", "omero": {"channels": []}}},
    )

    assert voxel_size_of(path) == (2.0, 0.325, 0.325)


@pytest.mark.parametrize(
    "ome_attr",
    [
        {"version": "0.5"},  # an `ome` block with no multiscales
        "0.5",  # `ome` is not a mapping at all
        ["0.5"],
    ],
)
def test_a_useless_ome_attribute_falls_through_gracefully(
    tmp_path: Path, ome_attr: Any
) -> None:
    """No multiscales to read is a fallback, not an exception."""
    path = _write_store(tmp_path / "odd.zarr", (16, 32, 32), ome_attr=ome_attr)

    info = discover_ome_zarr_shape(path)

    assert info.spatial_shape == (16, 32, 32)
    assert info.n_timepoints == 1
    assert info.n_channels == 1


# ---------------------------------------------------------------------------
# `array_key` selects a DIFFERENT array than the multiscales block describes
# ---------------------------------------------------------------------------


def test_a_labels_array_beside_a_5d_image_does_not_crash(tmp_path: Path) -> None:
    """The classic image-with-labels store: 5D pyramid, 3D ``labels/cells``.

    The root block declares 5 TCZYX axes; the selected array is 3D. Indexing the
    shape by those axes raised a bare ``IndexError: tuple index out of range``
    (reachable from ``batch-fit --array-key labels/cells``). The block simply is
    not about this array — fall through to the heuristic.
    """
    path = tmp_path / "image_labels.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros((2, 1, 8, 16, 16), dtype=np.float32), compressor="auto"
    )
    labels = root.create_group("labels")
    create_array(
        labels, "cells", data=np.zeros((8, 16, 16), dtype=np.uint16), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_TCZYX, (1.0, 1.0, 2.0, 0.3, 0.3)),
    }

    info = discover_ome_zarr_shape(path, array_key="labels/cells")

    assert info.shape == (8, 16, 16)
    assert info.spatial_shape == (8, 16, 16)
    assert info.n_timepoints == 1
    assert info.n_channels == 1


def test_an_axes_ndim_mismatch_says_so_instead_of_claiming_nothing_was_found(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Same mismatch on a ≥4D selection: the notice must name the real reason."""
    path = tmp_path / "mismatch.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros((2, 1, 8, 16, 16), dtype=np.float32), compressor="auto"
    )
    create_array(
        root, "big", data=np.zeros((3, 8, 16, 16), dtype=np.uint16), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_TCZYX, (1.0, 1.0, 2.0, 0.3, 0.3)),
    }

    discover_ome_zarr_shape(path, array_key="big")
    out = capsys.readouterr().out

    assert "no OME-Zarr/NGFF metadata found" not in out
    assert "5 axes" in out and "4-D" in out


# ---------------------------------------------------------------------------
# Voxel size: the right dataset entry, and the multiscale-level transform
# ---------------------------------------------------------------------------


def test_a_coarser_pyramid_level_reports_its_own_voxel_size(tmp_path: Path) -> None:
    """``array_key="1"`` must not report level 0's spacing."""
    path = _write_store(
        tmp_path / "pyramid.zarr",
        (2, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        extra_levels=[(1.0, 4.0, 0.65, 0.65)],
        extra_level_shapes=[(2, 8, 16, 16)],
    )

    assert discover_ome_zarr_shape(path, array_key="0").voxel_size == (
        2.0,
        0.325,
        0.325,
    )
    assert discover_ome_zarr_shape(path, array_key="1").voxel_size == (
        4.0,
        0.65,
        0.65,
    )


def test_a_multiscale_level_transform_composes_with_the_dataset_one(
    tmp_path: Path,
) -> None:
    """Both 0.4 and 0.5 allow a transform on the multiscales entry itself."""
    path = _write_store(
        tmp_path / "composed.zarr",
        (2, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 1.0, 1.0, 1.0),  # identity per-dataset
        ms_scale=(1.0, 2.0, 0.325, 0.325),
    )

    assert discover_ome_zarr_shape(path).voxel_size == (2.0, 0.325, 0.325)


# ---------------------------------------------------------------------------
# The give-up is audible — and honest about WHY
# ---------------------------------------------------------------------------


def test_the_heuristic_fallback_announces_its_guess(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A ≥4D store with no readable metadata must not fail silently."""
    path = _write_store(tmp_path / "bare.zarr", (7, 16, 32, 32))

    info = discover_ome_zarr_shape(path)
    out = capsys.readouterr().out

    assert info.axes == ["c", "z", "y", "x"]  # the guess, unchanged
    assert "no OME-Zarr/NGFF metadata found" in out
    assert "c,z,y,x" in out
    assert "--axes" in out
    assert "axes_override" in out
    assert "voxel size" in out


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"ome": {"version": "0.5", "multiscales": []}}, "empty or not a list"),
        ({"multiscales": {"axes": []}}, "empty or not a list"),
        ({"axes": ["t", "z", "y"]}, "3 labels for a 4-D array"),
    ],
)
def test_present_but_unusable_metadata_is_not_reported_as_absent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    attrs: Dict[str, Any],
    expected: str,
) -> None:
    """ "No metadata found" is a lie when the store plainly declares something."""
    path = _write_store(tmp_path / "unusable.zarr", (7, 16, 32, 32), extra_attrs=attrs)

    discover_ome_zarr_shape(path)
    out = capsys.readouterr().out

    assert "no OME-Zarr/NGFF metadata found" not in out
    assert "present but unusable" in out
    assert expected in out


def test_a_parsed_store_says_nothing_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The notice is for GUESSES only — a successfully read ≥4D store is quiet.

    Without this, hoisting ``_announce_guessed_axes`` to the top of discovery
    leaves the whole module green while every well-formed store is slandered.
    """
    path = _write_store(
        tmp_path / "good.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
    )

    discover_ome_zarr_shape(path)

    assert capsys.readouterr().out == ""


def test_an_unambiguous_3d_store_stays_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """2D/3D are all-spatial — there is no T-vs-C guess to warn about."""
    path = _write_store(tmp_path / "plain.zarr", (16, 32, 32))

    discover_ome_zarr_shape(path)

    assert "--axes" not in capsys.readouterr().out


def test_a_bioformats2raw_layout_is_still_not_discovered(tmp_path: Path) -> None:
    """KNOWN NON-GOAL, pinned so the day it is fixed this test says so.

    bioformats2raw writes ``{"ome": {"bioformats2raw.layout": 3}}`` at the ROOT
    and puts ``multiscales`` on a CHILD image group. Discovery reads attributes
    from the root only, and auto-selection picks the child GROUP (not an array),
    so it dies on ``.shape``. Unchanged by this fix, before and after.
    """
    path = tmp_path / "b2r.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    image = root.create_group("0")
    create_array(
        image, "0", data=np.zeros((2, 8, 16, 16), dtype=np.float32), compressor="auto"
    )
    image.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_TZYX, (1.0, 2.0, 0.3, 0.3)),
    }
    root.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}

    with pytest.raises(AttributeError, match="no attribute 'shape'"):
        discover_ome_zarr_shape(path)


# ---------------------------------------------------------------------------
# The recovered axes reach the batch planner
# ---------------------------------------------------------------------------


def _plan(src: Path, out_dir: Path) -> Any:
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    return plan_batch(
        input_path=src,
        output_dir=out_dir,
        tiling="uniform",
        tile_size=64,  # above every spatial extent -> a single tile, no GPU profile
        tile_overlap=8,
        axes_list=None,  # the point: discovery, not an explicit --axes
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(floor=None),
        denoise=DenoiseConfig(),
        content=ContentKnobs(),
        merge=MergeConfig(),
    )


def test_the_batch_planner_fans_out_over_the_recovered_time_axis(
    tmp_path: Path,
) -> None:
    """End of the wire: a 0.5 ``TZYX`` store plans as T=7, C=1.

    ``plan_batch`` is where a wrong axis guess does its damage — it reads
    ``n_timepoints``/``n_channels``/``spatial_shape`` straight off discovery and
    fans one fit task out per (t, c). Pre-fix this planned 7 CHANNELS of a
    single timepoint.
    """
    src = _write_store(
        tmp_path / "movie.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=True,
    )

    plan = _plan(src, tmp_path / "out")

    assert plan.manifest.n_timepoints == 7
    assert plan.manifest.n_channels == 1


def _plan_axes(src: Path, out_dir: Path, axes_list: List[str]) -> Any:
    """``_plan`` with an explicit ``--axes`` — the guard's documented escape hatch."""
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    return plan_batch(
        input_path=src,
        output_dir=out_dir,
        tiling="uniform",
        tile_size=512,
        tile_overlap=8,
        axes_list=axes_list,
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(floor=None),
        denoise=DenoiseConfig(),
        content=ContentKnobs(),
        merge=MergeConfig(),
    )


def test_a_layout_the_workers_cannot_slice_is_refused(tmp_path: Path) -> None:
    """A ``(t, c, y, x)`` store: the plan and the workers would disagree.

    The manifest carries no ``--axes`` (the user gave none), so every worker
    slices POSITIONALLY: at 4D it prefers ``--channel`` and IGNORES
    ``--timepoint``. Planning 5 timepoints × 3 channels there yields 3 distinct
    volumes repeated 5 times, silently. It must fail loudly instead.
    """
    src = _write_store(
        tmp_path / "tcyx.zarr",
        (5, 3, 128, 128),
        labels=("t", "c", "y", "x"),
        scale=(1.0, 1.0, 0.325, 0.325),
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out")

    message = str(excinfo.value)
    assert "t,c,y,x" in message
    assert "--axes t,c,y,x" in message


@pytest.mark.parametrize(
    ("labels", "shape", "expected_t", "expected_c"),
    [
        (_TCZYX, (3, 2, 8, 16, 16), 3, 2),
        (_TZYX, (3, 8, 16, 16), 3, 1),
        (_CZYX, (2, 8, 16, 16), 1, 2),
        (_ZYX, (8, 16, 16), 1, 1),
    ],
)
def test_positionally_sliceable_layouts_still_plan(
    tmp_path: Path,
    labels: Sequence[str],
    shape: Sequence[int],
    expected_t: int,
    expected_c: int,
) -> None:
    """The guard must only refuse what the worker really cannot reproduce."""
    src = _write_store(
        tmp_path / f"{''.join(labels)}.zarr",
        shape,
        labels=labels,
        scale=tuple(1.0 for _ in labels),
    )

    plan = _plan(src, tmp_path / f"out_{''.join(labels)}")

    assert plan.manifest.n_timepoints == expected_t
    assert plan.manifest.n_channels == expected_c


# ---------------------------------------------------------------------------
# The guard asks discovery's OWN decomposition, not the axis labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "labels", "shape", "expected_t", "expected_c", "expected_spatial"),
    [
        # A canonical 5D store whose channel axis is NAMED for its stain. NGFF
        # classifies by `type`, so discovery is perfect and `arr[t, c]` matches
        # exactly — but a name-driven rule reads `stain` as spatial and refuses.
        (
            "stain",
            (("t", "time"), ("stain", "channel"), "z", "y", "x"),
            (2, 3, 8, 16, 16),
            2,
            3,
            (8, 16, 16),
        ),
        # Same, for a time axis TYPED time but named `timepoint`.
        (
            "timepoint",
            (("timepoint", "time"), ("c", "channel"), "z", "y", "x"),
            (2, 3, 8, 16, 16),
            2,
            3,
            (8, 16, 16),
        ),
        # Non-spatial axes that are all singletons: `load_volume` squeezes the
        # whole-array positional result, so these load exactly the planned shape
        # however they are ordered. 2D gsplats are a first-class authoring path.
        ("tcyx_singleton", ("t", "c", "y", "x"), (1, 1, 32, 32), 1, 1, (32, 32)),
        ("zyxc_singleton", ("z", "y", "x", "c"), (16, 32, 32, 1), 1, 1, (16, 32, 32)),
        ("tyx_singleton", ("t", "y", "x"), (1, 64, 64), 1, 1, (64, 64)),
    ],
)
def test_layouts_the_worker_does_reproduce_are_not_refused(
    tmp_path: Path,
    name: str,
    labels: Sequence[Any],
    shape: Sequence[int],
    expected_t: int,
    expected_c: int,
    expected_spatial: Sequence[int],
) -> None:
    """False refusals, all of which planned and loaded consistently before.

    Every one of these was rejected while the guard re-derived the layout from
    the axis NAMES instead of reading the decomposition discovery published.
    """
    src = _write_store(
        tmp_path / f"{name}.zarr",
        shape,
        labels=labels,
        scale=tuple(1.0 for _ in labels),
    )

    plan = _plan(src, tmp_path / f"out_{name}")

    assert plan.manifest.n_timepoints == expected_t
    assert plan.manifest.n_channels == expected_c
    assert tuple(plan.manifest.spatial_shape) == tuple(expected_spatial)


def test_a_view_typed_leading_axis_is_refused(tmp_path: Path) -> None:
    """The false PASS: `type: view` is SPATIAL to the parser, channel to a name.

    Discovery reports a 4-D "spatial" shape ``(2, 8, 16, 16)`` with T=1, C=3, so
    the plan tiles four dimensions the worker's ``arr[t, c]`` never produces and
    the ``view=1`` half of the store is never fitted. A name-driven guard calls
    ``view`` channel-like and waves it through.
    """
    src = _write_store(
        tmp_path / "view.zarr",
        (2, 3, 8, 16, 16),
        labels=(("view", "view"), ("c", "channel"), "z", "y", "x"),
        scale=(1.0, 1.0, 1.0, 1.0, 1.0),
    )

    assert discover_ome_zarr_shape(src).spatial_shape == (2, 8, 16, 16)

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_view")

    assert "view,c,z,y,x" in str(excinfo.value)


def test_a_6d_store_with_no_metadata_is_refused_without_faking_an_axes_string(
    tmp_path: Path,
) -> None:
    """The honest message: ``dim0…dim5`` is not an ``--axes`` spelling.

    ``volume._axis_kind`` rejects every one of those labels, so quoting them back
    prints a command that fails — and ``plan_batch``'s override path is lenient
    enough to ACCEPT it first, folding the unknown label into the tile grid so
    that every worker dies later instead.
    """
    src = _write_store(tmp_path / "bare6d.zarr", (2, 2, 3, 8, 16, 16))

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_bare6d")

    message = str(excinfo.value)
    assert "--axes dim0" not in message  # the fabricated suggestion
    assert "unrecognised" in message
    assert "'dim0'" in message


def test_the_suggested_axes_string_is_one_that_actually_works(tmp_path: Path) -> None:
    """The guard only quotes an ``--axes`` spec when running it would succeed.

    Pins the other half of the contract: for a store whose labels ARE in
    ``_axis_kind``'s vocabulary the suggestion is printed, and following it plans
    the same spatial shape the workers then load.
    """
    src = _write_store(
        tmp_path / "tcyx_advice.zarr",
        (5, 3, 32, 32),
        labels=("t", "c", "y", "x"),
        scale=(1.0, 1.0, 1.0, 1.0),
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_refused")
    assert "--axes t,c,y,x" in str(excinfo.value)

    plan = _plan_axes(src, tmp_path / "out_axes", ["t", "c", "y", "x"])

    assert plan.manifest.n_timepoints == 5
    assert plan.manifest.n_channels == 3
    assert tuple(plan.manifest.spatial_shape) == (32, 32)


# ---------------------------------------------------------------------------
# The decomposition discovery used is PUBLISHED, not left to be re-derived
# ---------------------------------------------------------------------------


def test_the_ngff_parser_publishes_its_axis_indices(tmp_path: Path) -> None:
    """`_parse_ngff_metadata` classifies by `type`; the indices say what it did."""
    src = _write_store(
        tmp_path / "indices_ngff.zarr",
        (2, 3, 8, 16, 16),
        labels=(("t", "time"), ("stain", "channel"), "z", "y", "x"),
        scale=(1.0, 1.0, 1.0, 1.0, 1.0),
    )

    info = discover_ome_zarr_shape(src)

    assert info.time_axis == 0
    assert info.channel_indices == (1,)
    assert info.spatial_indices == (2, 3, 4)


def test_the_custom_axes_parser_publishes_its_axis_indices(tmp_path: Path) -> None:
    """The Keller-lab ``axes`` attribute route: several channel-like axes fold."""
    src = _write_store(
        tmp_path / "indices_custom.zarr",
        (2, 2, 3, 8, 16, 16),
        extra_attrs={"axes": ["time", "camera", "channel", "z", "y", "x"]},
    )

    info = discover_ome_zarr_shape(src)

    assert info.time_axis == 0
    assert info.channel_indices == (1, 2)  # fold order == flat channel decode order
    assert info.spatial_indices == (3, 4, 5)
    assert info.n_channels == 6


@pytest.mark.parametrize(
    ("shape", "time_axis", "channel_indices", "spatial_indices"),
    [
        ((2, 3, 8, 16, 16), 0, (1,), (2, 3, 4)),  # 5D → TCZYX
        ((3, 8, 16, 16), None, (0,), (1, 2, 3)),  # 4D → CZYX
        ((8, 16, 16), None, (), (0, 1, 2)),  # 3D → ZYX
        ((16, 16), None, (), (0, 1)),  # 2D → YX
        ((2, 2, 3, 8, 16, 16), None, (), (0, 1, 2, 3, 4, 5)),  # generic nD
    ],
)
def test_the_shape_heuristic_publishes_its_axis_indices(
    tmp_path: Path,
    shape: Sequence[int],
    time_axis: Optional[int],
    channel_indices: Sequence[int],
    spatial_indices: Sequence[int],
) -> None:
    """Every heuristic branch too — a default of ``()`` would be a wrong answer."""
    src = _write_store(tmp_path / f"h{len(shape)}.zarr", shape)

    info = discover_ome_zarr_shape(src)

    assert info.time_axis == time_axis
    assert info.channel_indices == tuple(channel_indices)
    assert info.spatial_indices == tuple(spatial_indices)


# ---------------------------------------------------------------------------
# Malformed metadata DEGRADES — every other path in the module does
# ---------------------------------------------------------------------------


def _raw_ms_store(tmp_path: Path, shape: Sequence[int], ms: Any) -> Path:
    """A store carrying a literal (possibly malformed) multiscales block."""
    path = tmp_path / "raw.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros(tuple(shape), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {"version": "0.5", "multiscales": [ms]}
    return path


@pytest.mark.parametrize(
    "datasets",
    [
        ["0"],  # a list of PLAIN STRINGS — `datasets[0].get` is an AttributeError
        {
            "0": {"coordinateTransformations": []}
        },  # a dict — `datasets[0]` is a KeyError
        [],  # empty
        "0",  # not a container of entries at all
    ],
)
def test_a_malformed_datasets_list_costs_the_voxel_size_not_a_traceback(
    tmp_path: Path, datasets: Any
) -> None:
    """The axes are still perfectly readable; only the spacing is unknown."""
    path = _raw_ms_store(
        tmp_path, (8, 16, 16), {"axes": _axes_meta(_ZYX), "datasets": datasets}
    )

    info = discover_ome_zarr_shape(path)

    assert info.spatial_shape == (8, 16, 16)
    assert info.voxel_size is None


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        ([1.0, None, 2.0], None),  # a JSON null — TypeError on float()
        ([1.0, "many", 2.0], None),  # a non-numeric string — ValueError
        (["2.0", "0.5", "0.5"], (2.0, 0.5, 0.5)),  # numeric strings still convert
        ([2.0, 0.5, 0.5], (2.0, 0.5, 0.5)),
    ],
)
def test_a_malformed_scale_vector_degrades_to_no_voxel_size(
    tmp_path: Path, scale: Any, expected: Any
) -> None:
    """A scale that is not a vector of numbers is a missing spacing, not a crash."""
    path = _raw_ms_store(
        tmp_path,
        (8, 16, 16),
        {
            "axes": _axes_meta(_ZYX),
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": scale}],
                }
            ],
        },
    )

    assert discover_ome_zarr_shape(path).voxel_size == expected


@pytest.mark.parametrize(
    ("axes_raw", "expected_axes"),
    [
        # A dict axis with no `name` at all — `a["name"]` was a KeyError, while
        # the very next loop already used `a.get("name", "")`.
        ([{"type": "space"}, {"name": "y"}, {"name": "x"}], ["", "y", "x"]),
        # `type: null` — `a.get("type", "").lower()` was an AttributeError.
        (
            [{"name": "z", "type": None}, {"name": "y"}, {"name": "x"}],
            ["z", "y", "x"],
        ),
    ],
)
def test_a_malformed_axis_record_degrades_instead_of_raising(
    tmp_path: Path, axes_raw: Any, expected_axes: List[str]
) -> None:
    """An axis missing its name/type falls through to spatial, like an unknown one."""
    path = _raw_ms_store(tmp_path, (8, 16, 16), {"axes": axes_raw, "datasets": []})

    info = discover_ome_zarr_shape(path)

    assert info.axes == expected_axes
    assert info.spatial_shape == (8, 16, 16)


# ---------------------------------------------------------------------------
# Voxel size: no half-composed number, no other level's spacing
# ---------------------------------------------------------------------------


def test_an_uncomposable_multiscales_level_scale_yields_no_voxel_size(
    tmp_path: Path,
) -> None:
    """A length mismatch cannot be composed — and half of a product is not a fact.

    The dataset scale ``[1, 1, 3]`` with a 2-long multiscales-entry scale used to
    drop the latter silently and report ``(1.0, 1.0, 3.0)`` as the spacing, which
    is exactly the composition the docstring promises NOT to skip.
    """
    path = _write_store(
        tmp_path / "uncomposable.zarr",
        (8, 16, 16),
        labels=_ZYX,
        scale=(1.0, 1.0, 3.0),
        ms_scale=(2.0, 2.0),
    )

    assert discover_ome_zarr_shape(path).voxel_size is None


def test_a_nested_array_key_matches_its_dataset_by_the_trailing_segment(
    tmp_path: Path,
) -> None:
    """``datasets[].path`` is relative to the multiscales group, ``array_key`` is not.

    Comparing the whole ``labels/cells/1`` against the entry's ``"1"`` never
    matched, so a 2x-downsampled array was quoted level 0's spacing.
    """
    path = tmp_path / "nested_levels.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    cells = root.create_group("labels").create_group("cells")
    create_array(
        cells, "0", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
    create_array(
        cells, "1", data=np.zeros((4, 8, 8), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(
            _ZYX, (1.0, 0.5, 0.5), extra_levels=[(2.0, 1.0, 1.0)]
        ),
    }

    assert discover_ome_zarr_shape(path, array_key="labels/cells/1").voxel_size == (
        2.0,
        1.0,
        1.0,
    )
    assert discover_ome_zarr_shape(path, array_key="labels/cells/0").voxel_size == (
        1.0,
        0.5,
        0.5,
    )


def test_an_unmatched_array_key_in_a_pyramid_reports_no_voxel_size(
    tmp_path: Path,
) -> None:
    """The caller selected SOME array and this cannot say which of the levels.

    Quoting level 0's spacing as the selected array's is the plausible wrong
    answer; with more than one level on offer, ``None`` is the honest one.
    """
    path = tmp_path / "unmatched.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    fused = root.create_group("fused")
    for key in ("a", "b"):
        create_array(
            fused, key, data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
        )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(
            _ZYX, (1.0, 0.5, 0.5), extra_levels=[(2.0, 1.0, 1.0)]
        ),
    }

    assert discover_ome_zarr_shape(path, array_key="fused/b").voxel_size is None


def test_a_single_level_pyramid_still_answers_for_an_unmatched_key(
    tmp_path: Path,
) -> None:
    """With one level there is nothing to be wrong about — keep answering."""
    path = tmp_path / "single_level.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    fused = root.create_group("fused")
    create_array(
        fused, "b", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_ZYX, (1.0, 0.5, 0.5)),
    }

    assert discover_ome_zarr_shape(path, array_key="fused/b").voxel_size == (
        1.0,
        0.5,
        0.5,
    )


# ---------------------------------------------------------------------------
# The resolver keys on a USABLE nested block, not merely a present one
# ---------------------------------------------------------------------------


def test_an_empty_nested_multiscales_does_not_discard_the_top_level_pyramid() -> None:
    """Presence is not usability — the same silent mis-read as the ``omero`` trap."""
    attrs = {
        "multiscales": ["top-level"],
        "ome": {"version": "0.5", "multiscales": []},
    }

    assert resolve_ngff_attrs(attrs)["multiscales"] == ["top-level"]


def test_an_empty_nested_multiscales_is_still_returned_when_it_is_all_there_is() -> (
    None
):
    """Otherwise "declared but unusable" would be downgraded to "nothing declared"."""
    attrs: Dict[str, Any] = {"ome": {"version": "0.5", "multiscales": []}}

    assert resolve_ngff_attrs(attrs)["multiscales"] == []


def test_a_04_pyramid_survives_an_empty_nested_multiscales_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The store DOES have a pyramid; blaming an empty ``multiscales`` is false."""
    path = _write_store(
        tmp_path / "empty_nested.zarr",
        (7, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=False,
        extra_attrs={"ome": {"version": "0.5", "multiscales": []}},
    )

    info = discover_ome_zarr_shape(path)
    out = capsys.readouterr().out

    assert info.n_timepoints == 7
    assert info.n_channels == 1
    assert info.voxel_size == (2.0, 0.325, 0.325)
    assert "empty or not a list" not in out
    assert "GUESSED" not in out


def test_the_demo_voxel_size_reader_survives_a_leading_translation(
    tmp_path: Path,
) -> None:
    """``coordinateTransformations[0]`` is not necessarily the scale.

    The spec allows a ``translation`` first; the demo's hardcoded ``[0]["scale"]``
    raised a ``KeyError`` on such a store while the library helper searched for
    the ``type == "scale"`` entry all along.
    """
    from luxar.demos.demo_gsplats_4d_cell_tracking_challenge import voxel_size_of

    path = tmp_path / "translated.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros((2, 8, 16, 16), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": _axes_meta(_TZYX),
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "translation",
                                "translation": [0.0, 1.0, 2.0, 3.0],
                            },
                            {"type": "scale", "scale": [1.0, 2.0, 0.325, 0.325]},
                        ],
                    }
                ],
            }
        ],
    }

    assert voxel_size_of(path) == (2.0, 0.325, 0.325)
