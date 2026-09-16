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
from luxar.io.ome_zarr import (
    _dataset_path_matches,
    discover_ome_zarr_shape,
    resolve_ngff_attrs,
)

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
    data: Optional[np.ndarray] = None,
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
        root,
        "0",
        data=np.zeros(tuple(shape), dtype=np.float32) if data is None else data,
        compressor="auto",
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


def test_a_bioformats2raw_layout_resolves_its_level_with_no_key_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``"0"`` is an image GROUP here, and discovery descends INTO it (#1777).

    Auto-selection used to refuse such a store, naming ``--array-key 0/0`` as the
    workaround. It now resolves the level itself, and — the half that is easy to
    lose — reads the 0.5 ``multiscales`` off the IMAGE GROUP that declares it. The
    root carries only ``bioformats2raw.layout``, so a reader that consults the
    root alone lands on the 4D ``CZYX`` heuristic with no voxel size, silently.
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

    # Naming the level explicitly must describe it the same way, or the store
    # plans one way with a key and another without it.
    for key in (None, "0", "0/0"):
        info = discover_ome_zarr_shape(path, array_key=key)
        assert info.shape == (2, 8, 16, 16), key
        assert info.axes == list(_TZYX), key
        assert (info.n_timepoints, info.n_channels) == (2, 1), key
        assert info.voxel_size == (2.0, 0.3, 0.3), key

    assert "GUESSED" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The recovered axes reach the batch planner
# ---------------------------------------------------------------------------


def _plan(src: Path, out_dir: Path, **overrides: Any) -> Any:
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    options: Dict[str, Any] = dict(
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
    options.update(overrides)
    return plan_batch(**options)


def _worker_loaded_volume(manifest: Any) -> Any:
    """The volume a real worker's ``load_volume`` returns for the LAST task.

    The guard in ``planning`` is a hand-written mirror of
    :func:`luxar.io.volume._load_zarr_volume`, so a test that only re-asserts the
    plan proves nothing about the loader: flipping the loader's 4D
    channel-over-timepoint preference left the whole suite green. This drives the
    real pair — the task argv is built by
    :func:`~luxar.gsplats.batch.fit_command.build_task_fit_argv`, so the
    ``--channel`` / ``--timepoint`` flags are exactly the ones the guard reasons
    about, and the LAST job carries the largest indices (a wrongly-fanned axis
    goes out of range or lands on a duplicate there first).
    """
    from luxar.gsplats.batch.fit_command import build_task_fit_argv
    from luxar.io.volume import load_volume

    job = manifest.jobs[-1]
    argv = build_task_fit_argv(manifest, job, "unused.gsplats.zarr", argv0=["luxar"])
    kwargs: Dict[str, Any] = {}
    for flag, name in (("--channel", "channel"), ("--timepoint", "timepoint")):
        if flag in argv:
            kwargs[name] = int(argv[argv.index(flag) + 1])
    if "--array-key" in argv:
        kwargs["array_key"] = argv[argv.index("--array-key") + 1]
    if "--axes" in argv:
        kwargs["axes"] = argv[argv.index("--axes") + 1]
    return load_volume(Path(manifest.input_path), **kwargs)


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


def test_batch_physical_forwards_discovered_ngff_voxel_size(tmp_path: Path) -> None:
    """The opt-in moves worker output and the merge grid into physical space."""
    from luxar.cli.gsplat_ops.batch.planning import FitConfig
    from luxar.gsplats.batch.fit_command import build_task_fit_argv

    src = _write_store(
        tmp_path / "physical.zarr",
        (2, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
        nested=True,
    )

    plan = _plan(
        src,
        tmp_path / "out_physical",
        fit=FitConfig(floor=None, physical=True),
    )
    argv = build_task_fit_argv(
        plan.manifest,
        plan.manifest.jobs[0],
        "unused.gsplats.zarr",
        argv0=["luxar"],
    )

    assert tuple(
        float(value) for value in argv[argv.index("--voxel-size") + 1].split(",")
    ) == (2.0, 0.325, 0.325)
    assert plan.manifest.grid_scale == [2.0, 0.325, 0.325]


def test_batch_index_space_remains_the_default(tmp_path: Path) -> None:
    """NGFF spacing is inert unless the user explicitly requests physical output."""
    src = _write_store(
        tmp_path / "index.zarr",
        (2, 16, 32, 32),
        labels=_TZYX,
        scale=(1.0, 2.0, 0.325, 0.325),
    )

    plan = _plan(src, tmp_path / "out_index")

    assert "voxel-size" not in plan.manifest.fit_args
    assert plan.manifest.grid_scale is None


def test_batch_physical_respects_configured_voxel_size(tmp_path: Path) -> None:
    """An explicit fit config remains the higher-priority spacing source."""
    from luxar.cli.gsplat_ops.batch.planning import FitConfig

    src = _write_store(
        tmp_path / "configured.zarr",
        (16, 32, 32),
        labels=_ZYX,
        scale=(2.0, 0.325, 0.325),
    )
    config = tmp_path / "fit.yaml"
    config.write_text("voxel_size: [4.0, 1.0, 1.0]\n")

    plan = _plan(
        src,
        tmp_path / "out_configured",
        fit=FitConfig(floor=None, physical=True, config=config),
    )

    assert "voxel-size" not in plan.manifest.fit_args
    assert plan.manifest.grid_scale == [4.0, 1.0, 1.0]


def test_batch_physical_requires_a_spacing_source(tmp_path: Path) -> None:
    """The opt-in fails loudly instead of silently producing voxel coordinates."""
    import typer
    import zarr

    from luxar.cli.gsplat_ops.batch.planning import FitConfig

    src = tmp_path / "plain.zarr"
    create_array(
        zarr.open_group(src, mode="w"),
        "0",
        data=np.zeros((16, 32, 32), dtype=np.float32),
    )

    with pytest.raises(typer.BadParameter, match="no usable spatial scale"):
        _plan(
            src,
            tmp_path / "out_plain",
            fit=FitConfig(floor=None, physical=True),
        )


def test_batch_physical_drops_squeezed_singleton_spacing(tmp_path: Path) -> None:
    """Discovered spacing follows the exact dimensions positional workers fit."""
    from luxar.cli.gsplat_ops.batch.planning import FitConfig

    src = _write_store(
        tmp_path / "thin.zarr",
        (1, 16, 32),
        labels=_ZYX,
        scale=(5.0, 0.4, 0.2),
    )

    plan = _plan(
        src,
        tmp_path / "out_thin",
        fit=FitConfig(floor=None, physical=True),
    )

    assert plan.manifest.spatial_shape == (16, 32)
    assert plan.manifest.grid_scale == [0.4, 0.2]
    assert plan.manifest.fit_args["voxel-size"] == "0.4,0.2"


def test_batch_physical_rejects_voxel_output_space(tmp_path: Path) -> None:
    """An explicit request for voxel output cannot silently defeat --physical."""
    import typer

    from luxar.cli.gsplat_ops.batch.planning import FitConfig

    src = _write_store(
        tmp_path / "voxel-output.zarr",
        (16, 32, 32),
        labels=_ZYX,
        scale=(2.0, 0.4, 0.4),
    )
    config = tmp_path / "voxel.yaml"
    config.write_text("output_space: voxel\n")

    with pytest.raises(typer.BadParameter, match="requires output_space: real"):
        _plan(
            src,
            tmp_path / "out_voxel",
            fit=FitConfig(floor=None, physical=True, config=config),
        )


def test_batch_physical_rejects_volume_merge_refine(tmp_path: Path) -> None:
    """Volume refinement crops in voxels, so planning refuses a physical frame."""
    import typer

    from luxar.cli.gsplat_ops.batch.planning import FitConfig, MergeConfig

    src = _write_store(
        tmp_path / "refine.zarr",
        (16, 32, 32),
        labels=_ZYX,
        scale=(2.0, 0.4, 0.4),
    )

    with pytest.raises(typer.BadParameter, match="frame scaled by"):
        _plan(
            src,
            tmp_path / "out_refine",
            axes_list=["z", "y", "x"],
            fit=FitConfig(floor=None, physical=True),
            merge=MergeConfig(recipe="levels", levels=1, refine="volume"),
        )


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


# ---------------------------------------------------------------------------
# Every layout that PLANS is one the worker's own loader reproduces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "labels", "shape", "expected_t", "expected_c", "expected_spatial", "opts"),
    [
        # --- the canonical layouts ------------------------------------------
        ("tczyx", _TCZYX, (3, 2, 8, 16, 16), 3, 2, (8, 16, 16), {}),
        ("tzyx", _TZYX, (3, 8, 16, 16), 3, 1, (8, 16, 16), {}),
        ("czyx", _CZYX, (2, 8, 16, 16), 1, 2, (8, 16, 16), {}),
        ("zyx", _ZYX, (8, 16, 16), 1, 1, (8, 16, 16), {}),
        ("yx", ("y", "x"), (16, 16), 1, 1, (16, 16), {}),
        # --- NGFF classifies by `type`, so name and type may disagree --------
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
            {},
        ),
        # Same, for a time axis TYPED time but named `timepoint`.
        (
            "timepoint",
            (("timepoint", "time"), ("c", "channel"), "z", "y", "x"),
            (2, 3, 8, 16, 16),
            2,
            3,
            (8, 16, 16),
            {},
        ),
        # --- singleton non-spatial axes: `load_volume` squeezes them away ----
        # 2D gsplats are a first-class authoring path, so these must plan.
        ("tcyx_1_1", ("t", "c", "y", "x"), (1, 1, 32, 32), 1, 1, (32, 32), {}),
        ("zyxc_1", ("z", "y", "x", "c"), (16, 32, 32, 1), 1, 1, (16, 32, 32), {}),
        ("tyx_1", ("t", "y", "x"), (1, 64, 64), 1, 1, (64, 64), {}),
        # A 2D TIMELAPSE with a singleton channel. The worker's `arr[3]` on the
        # `--timepoint` branch IS the planned volume, but a per-ndim table that
        # demands an exact axis partition refused it — while admitting the T=1
        # sibling above, so its verdict flipped on the size of T alone.
        ("tcyx_5_1", ("t", "c", "y", "x"), (5, 1, 32, 32), 5, 1, (32, 32), {}),
        # A singleton CHANNEL axis explicitly sliced: `--timepoints 0` makes the
        # argv carry `--timepoint`, which the loader spends on axis 0 — harmless,
        # because a size-1 axis can only be indexed at 0 either way.
        (
            "czyx_1_tp0",
            _CZYX,
            (1, 16, 32, 32),
            1,
            1,
            (16, 32, 32),
            {"timepoints_slice": "0"},
        ),
    ],
)
def test_every_planned_layout_is_one_the_worker_actually_loads(
    tmp_path: Path,
    name: str,
    labels: Sequence[Any],
    shape: Sequence[int],
    expected_t: int,
    expected_c: int,
    expected_spatial: Sequence[int],
    opts: Dict[str, Any],
) -> None:
    """The guard must refuse only what the worker really cannot reproduce.

    Two assertions per row, and the second is the load-bearing one: the plan's
    ``spatial_shape`` is what every downstream stage (tile grid, BSP split
    planes, merge) is built on, so it has to be the shape ``load_volume``
    actually hands the worker for a real task.
    """
    src = _write_store(
        tmp_path / f"{name}.zarr",
        shape,
        labels=labels,
        scale=tuple(1.0 for _ in labels),
    )

    plan = _plan(src, tmp_path / f"out_{name}", **opts)

    assert plan.manifest.n_timepoints == expected_t
    assert plan.manifest.n_channels == expected_c
    assert tuple(plan.manifest.spatial_shape) == tuple(expected_spatial)
    assert _worker_loaded_volume(plan.manifest).shape == tuple(expected_spatial)


def test_a_6d_custom_axes_store_plans_and_loads(tmp_path: Path) -> None:
    """``time,camera,channel,z,y,x`` — a headline supported input, end to end.

    The Keller-lab ``axes`` attribute route, and the only layout that exercises
    the loader's ``ndim >= 6`` branch: axis 0 takes the timepoint and axes 1..2
    fold into the flat channel index, which is exactly the decomposition the
    custom-axes parser publishes. Task ``(t=1, c=5)`` is the far corner of that
    fold — a mis-ordered fold lands on the wrong camera there, and a wrongly
    guessed one goes out of range.
    """
    src = _write_store(
        tmp_path / "keller6d.zarr",
        (2, 2, 3, 8, 16, 16),
        extra_attrs={"axes": ["time", "camera", "channel", "z", "y", "x"]},
    )

    plan = _plan(src, tmp_path / "out_keller6d")

    assert plan.manifest.n_timepoints == 2
    assert plan.manifest.n_channels == 6  # camera x channel, folded
    assert tuple(plan.manifest.spatial_shape) == (8, 16, 16)
    assert _worker_loaded_volume(plan.manifest).shape == (8, 16, 16)


def test_the_worker_reads_the_slice_the_job_names_not_merely_a_right_shape(
    tmp_path: Path,
) -> None:
    """At 4D the loader consumes ONE axis, and it must be the one the plan fanned.

    ``--timepoints 0`` on a ``(c, z, y, x)`` store puts BOTH flags on the task
    argv, which is the only situation where the loader's channel-over-timepoint
    preference is observable — and every channel has the same shape, so only the
    CONTENT distinguishes "read channel 2" from "read index 0 of the same axis".
    Without this the guard is a mirror of a contract nothing pins: flipping that
    preference in ``_load_zarr_volume`` left the whole suite green.
    """
    values = np.arange(3, dtype=np.float32)[:, None, None, None]
    src = _write_store(
        tmp_path / "content_czyx.zarr",
        (3, 8, 16, 16),
        data=np.broadcast_to(values, (3, 8, 16, 16)).astype(np.float32),
        labels=_CZYX,
        scale=(1.0, 1.0, 1.0, 1.0),
    )

    plan = _plan(src, tmp_path / "out_content", timepoints_slice="0")
    job = plan.manifest.jobs[-1]
    assert (job.channel, job.timepoint) == (2, 0)

    volume = _worker_loaded_volume(plan.manifest)

    assert volume.shape == (8, 16, 16)
    assert volume.min() == volume.max() == 2.0  # channel 2, not timepoint 0


# ---------------------------------------------------------------------------
# ... and every layout it cannot is REFUSED, on discovery's own decomposition
# ---------------------------------------------------------------------------


def test_a_layout_the_workers_would_duplicate_is_refused(tmp_path: Path) -> None:
    """A 3D ``(c, y, x)`` store: the plan fans, the worker reads it whole.

    The loader's ``ndim <= 3`` branch is ``np.array(arr)`` — it IGNORES both
    flags — so planning 3 channels here yields the same ``(3, 64, 64)`` array
    three times, and each task then tiles a volume with one axis more than the
    plan's ``(64, 64)``.
    """
    src = _write_store(
        tmp_path / "cyx.zarr",
        (3, 64, 64),
        labels=("c", "y", "x"),
        scale=(1.0, 1.0, 1.0),
    )

    assert discover_ome_zarr_shape(src).spatial_shape == (64, 64)

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_cyx")

    assert "c,y,x" in str(excinfo.value)


@pytest.mark.parametrize(
    "slice_opts",
    [{}, {"channels_slice": ""}, {"timepoints_slice": ""}],
)
def test_a_4d_store_sliced_by_no_flag_at_all_is_refused(
    tmp_path: Path, slice_opts: Dict[str, Any]
) -> None:
    """The 4D whole-array branch: nothing is consumed, so nothing may be fanned.

    Two ``type: channel`` axes is a layout the NGFF parser cannot represent — it
    keeps the LAST as the channel and drops the other from every role — so the
    plan's ``(32, 32)`` omits an axis that is still there. With T=1 and C=1 the
    argv carries neither flag, the worker loads the whole array and squeezes to
    ``(2, 32, 32)``, and the tile grid is built on the wrong rank.
    """
    src = _write_store(
        tmp_path / "ccyx.zarr",
        (2, 1, 32, 32),
        labels=(("c0", "channel"), ("c1", "channel"), "y", "x"),
        scale=(1.0, 1.0, 1.0, 1.0),
    )

    info = discover_ome_zarr_shape(src)
    assert info.spatial_shape == (32, 32) and info.n_channels == 1

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_ccyx", **slice_opts)

    assert "c0,c1,y,x" in str(excinfo.value)


def test_empty_slice_strings_do_not_model_flags_the_worker_will_not_receive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's emission mirror must use the argv builder's truthiness rule."""
    from luxar.cli.gsplat_ops.batch import planning
    from luxar.gsplats.batch.fit_command import build_task_fit_argv

    src = _write_store(
        tmp_path / "zyx_empty_slices.zarr",
        (8, 16, 16),
        labels=_ZYX,
        scale=(1.0, 1.0, 1.0),
    )
    emitted: Dict[str, bool] = {}

    def capture_emission(
        axes_list: Optional[List[str]],
        info: Any,
        emits_channel: bool,
        emits_timepoint: bool,
    ) -> None:
        emitted.update(channel=emits_channel, timepoint=emits_timepoint)

    monkeypatch.setattr(
        planning, "_refuse_layout_the_workers_cannot_slice", capture_emission
    )
    plan = _plan(
        src,
        tmp_path / "out_empty_slices",
        channels_slice="",
        timepoints_slice="",
    )
    argv = build_task_fit_argv(
        plan.manifest,
        plan.manifest.jobs[0],
        "unused.gsplats.zarr",
        argv0=["luxar"],
    )

    assert emitted == {"channel": False, "timepoint": False}
    assert "--channel" not in argv
    assert "--timepoint" not in argv


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
    would print a command that fails. The planner preflights the same strict
    worker vocabulary, so it withholds that invalid suggestion.
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
    assert _worker_loaded_volume(plan.manifest).shape == (32, 32)


def test_axes_spec_is_quoted_for_folded_channel_like_axes(
    tmp_path: Path,
) -> None:
    """A flat task channel addresses every folded channel-like coordinate."""
    from luxar.io.volume import load_volume

    data = np.empty((2, 3, 2, 8, 16, 16), dtype=np.float32)
    for camera in range(2):
        for channel in range(2):
            data[camera, :, channel] = 10 * camera + channel
    src = _write_store(
        tmp_path / "camera_time_channel.zarr",
        data.shape,
        data=data,
        extra_attrs={"axes": ["camera", "time", "channel", "z", "y", "x"]},
    )
    spec = "camera,time,channel,z,y,x"

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_ctc")
    message = str(excinfo.value)

    assert f"Pass '--axes {spec}'" in message

    forced = _plan_axes(src, tmp_path / "out_ctc_axes", spec.split(","))
    manifest = forced.manifest
    assert manifest.n_channels == 4  # camera x channel
    assert len(manifest.jobs) == 12  # 3 timepoints x 4 folded channels
    for job in manifest.jobs:
        loaded = load_volume(
            src, channel=job.channel, timepoint=job.timepoint, axes=spec
        )
        camera, channel = job.channel_coords
        np.testing.assert_array_equal(loaded, data[camera, job.timepoint, channel])


def test_axes_spec_is_not_quoted_for_folded_time_axes(tmp_path: Path) -> None:
    """One flat ``--timepoint`` cannot address two discovered time axes."""
    src = _write_store(
        tmp_path / "folded_time.zarr",
        (3, 2, 4, 8, 8),
        extra_attrs={"axes": ["t", "time", "z", "y", "x"]},
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out_folded_time")

    message = str(excinfo.value)
    assert "more than one time axis ('t', 'time')" in message
    assert "one --timepoint cannot address them independently" in message
    assert "Pass '--axes t,time,z,y,x'" not in message


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
        5,  # not even sized — `len(datasets)` for the level count
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


@pytest.mark.parametrize("axes", [None, 5, "zyx", {"0": "z"}])
def test_a_malformed_axes_entry_degrades_instead_of_raising(
    tmp_path: Path, axes: Any, capsys: pytest.CaptureFixture
) -> None:
    """``{"axes": null}`` used to be a ``TypeError`` straight out of discovery.

    The sibling of the ``"type": null`` record below: ``len(ms.get("axes", []))``
    measured whatever was there. Discovery must fall through to the heuristic
    with a reason, like every other unusable block.
    """
    path = _raw_ms_store(tmp_path, (2, 3, 8, 16, 16), {"axes": axes, "datasets": []})

    info = discover_ome_zarr_shape(path)

    assert info.spatial_shape == (8, 16, 16)  # the 5D heuristic
    out = capsys.readouterr().out
    assert "no OME-Zarr/NGFF metadata found" not in out
    assert "no `axes` list" in out


@pytest.mark.parametrize("axes", [5, {"0": "z"}, "zyx"])
def test_a_custom_axes_attribute_that_is_not_a_list_degrades_instead_of_raising(
    tmp_path: Path, axes: Any, capsys: pytest.CaptureFixture
) -> None:
    """The bare-``axes`` sibling of the ``multiscales.axes`` case above.

    ``len(custom_axes)`` measures whatever is there, so a root attribute
    ``{"axes": 5}`` is a ``TypeError`` straight out of ``discover_ome_zarr_shape``
    — the one failure mode this module promises never to produce. (``"zyx"`` IS
    sized, so it pins the other half: a string must never be mistaken for three
    labels.)
    """
    path = _write_store(
        tmp_path / "custom_axes.zarr", (2, 3, 8, 16, 16), extra_attrs={"axes": axes}
    )

    info = discover_ome_zarr_shape(path)

    assert info.spatial_shape == (8, 16, 16)  # the 5D heuristic
    out = capsys.readouterr().out
    assert "no OME-Zarr/NGFF metadata found" not in out
    assert "not a list of labels" in out


def test_an_owner_axes_is_used_when_the_roots_own_list_is_unusable(
    tmp_path: Path,
) -> None:
    """Root-first is a PRECEDENCE, and an UNUSABLE root list still yields to it.

    The absent-root case is covered next door; this is the other branch of the
    same fallback — the root declares an ``axes`` list of the wrong length, so it
    cannot be used, and the owner's six labels must still be read rather than the
    6-D generic ``dim0…dim5`` heuristic (which reports T=1, C=1).
    """
    path = tmp_path / "root_axes_unusable.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    group = root.create_group("h2afva")
    create_array(
        group,
        "fused",
        data=np.zeros((2, 2, 3, 4, 4, 4), dtype=np.float32),
        compressor="auto",
    )
    root.attrs["axes"] = ["t", "z", "y"]  # 3 labels for a 6-D array
    group.attrs["axes"] = ["time", "camera", "channel", "z", "y", "x"]

    info = discover_ome_zarr_shape(path)

    assert info.axes == ["time", "camera", "channel", "z", "y", "x"]
    assert (info.n_timepoints, info.n_channels) == (2, 6)
    assert info.spatial_shape == (4, 4, 4)


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


def test_a_nested_array_key_uses_the_pyramid_which_owns_it(
    tmp_path: Path,
) -> None:
    """A nested pyramid must not borrow a same-named level from the root pyramid."""
    path = tmp_path / "nested_levels.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros((16, 32, 32), dtype=np.float32), compressor="auto"
    )
    create_array(
        root, "1", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
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
    cells.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(
            _ZYX, (9.0, 9.0, 9.0), extra_levels=[(18.0, 18.0, 18.0)]
        ),
    }

    assert discover_ome_zarr_shape(path, array_key="labels/cells/1").voxel_size == (
        18.0,
        18.0,
        18.0,
    )
    assert discover_ome_zarr_shape(path, array_key="labels/cells/0").voxel_size == (
        9.0,
        9.0,
        9.0,
    )


@pytest.mark.parametrize(
    "bad_multiscales",
    [
        ["not a block"],
        [{"axes": None, "datasets": [{"path": "0"}]}],
        [{"axes": _axes_meta(_ZYX), "datasets": [{"path": "0"}]}],
    ],
)
def test_an_unusable_nested_owner_falls_back_to_matching_root_metadata(
    tmp_path: Path, bad_multiscales: Any
) -> None:
    """A malformed parent block must not hide valid root metadata for the array."""
    path = tmp_path / "nested_owner_fallback.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    sub = root.create_group("sub")
    create_array(
        sub,
        "0",
        data=np.zeros((2, 3, 8, 16, 16), dtype=np.float32),
        compressor="auto",
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": _axes_meta(_TCZYX),
                "datasets": [
                    {
                        "path": "sub/0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1, 1, 4, 0.5, 0.5]}
                        ],
                    }
                ],
            }
        ],
    }
    sub.attrs["ome"] = {"version": "0.5", "multiscales": bad_multiscales}

    info = discover_ome_zarr_shape(path, array_key="sub/0")

    assert info.axes == list(_TCZYX)
    assert info.voxel_size == (4.0, 0.5, 0.5)


def test_an_implicit_nested_parent_does_not_break_discovery(tmp_path: Path) -> None:
    """Zarr v2 permits arrays below groups with no explicit metadata node."""
    path = tmp_path / "implicit_groups.zarr"
    root = open_group(path, mode="w", zarr_format=2)
    create_array(
        root,
        "labels/cells/1",
        data=np.zeros((4, 8, 8), dtype=np.uint16),
        compressor="auto",
    )
    (path / "labels" / ".zgroup").unlink()
    (path / "labels" / "cells" / ".zgroup").unlink()

    info = discover_ome_zarr_shape(path, array_key="labels/cells/1")

    assert info.shape == (4, 8, 8)
    assert info.spatial_shape == (4, 8, 8)


@pytest.mark.parametrize("malformed_side", ["dataset", "multiscales"])
def test_a_malformed_component_of_a_composed_scale_yields_no_voxel_size(
    tmp_path: Path, malformed_side: str
) -> None:
    """A declared but malformed scale is not an absent identity transform."""
    dataset_scale: Sequence[Any] = (1.0, None, 1.0)
    multiscales_scale: Sequence[Any] = (2.0, 2.0, 2.0)
    if malformed_side == "multiscales":
        dataset_scale, multiscales_scale = multiscales_scale, dataset_scale
    path = _raw_ms_store(
        tmp_path,
        (8, 16, 16),
        {
            "axes": _axes_meta(_ZYX),
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": list(dataset_scale)}
                    ],
                }
            ],
            "coordinateTransformations": [
                {"type": "scale", "scale": list(multiscales_scale)}
            ],
        },
    )

    assert discover_ome_zarr_shape(path).voxel_size is None


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


@pytest.mark.parametrize("zarr_format", (2, 3))
def test_an_explicitly_relative_dataset_path_still_names_its_level(
    tmp_path: Path, zarr_format: int
) -> None:
    """``"./0"`` and ``"0"`` are the same child, so the match must see through it.

    Writers do emit the explicitly relative spelling. Comparing the raw strings
    made every entry of such a pyramid unmatchable, and a multi-LEVEL pyramid has
    no single-level fall-back to ``datasets[0]``, so the store silently lost its
    voxel size — silently because the block itself parses, so not even the
    "present but unusable" notice fires. Each level carries its OWN spacing here,
    so this pins that the right ENTRY matched, not merely that some entry did.
    """
    path = tmp_path / "relative_paths.zarr"
    root = open_group(path, mode="w", zarr_format=zarr_format)
    create_array(
        root, "0", data=np.zeros((2, 8, 16, 16), dtype=np.float32), compressor="auto"
    )
    create_array(
        root, "1", data=np.zeros((2, 4, 8, 8), dtype=np.float32), compressor="auto"
    )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": _axes_meta(_TZYX),
            "datasets": [
                {
                    "path": "./0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0, 0.5, 0.5]}
                    ],
                },
                {
                    "path": "./1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 4.0, 1.0, 1.0]}
                    ],
                },
            ],
        }
    ]

    assert discover_ome_zarr_shape(path).voxel_size == (2.0, 0.5, 0.5)
    assert discover_ome_zarr_shape(path, array_key="0").voxel_size == (2.0, 0.5, 0.5)
    assert discover_ome_zarr_shape(path, array_key="1").voxel_size == (4.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("declared", "matches"),
    [
        ("0", True),
        ("/0", True),
        ("./0", True),
        # A bare current-directory path names the OWNER, not a level inside it,
        # and `a/./b` is a path zarr refuses outright — neither may be collapsed
        # into a match for whichever level happens to be selected.
        (".", False),
        ("./", False),
        ("0/./0", False),
        ("", False),
    ],
)
def test_which_dataset_path_spellings_name_the_level_zero_array(
    declared: str, matches: bool
) -> None:
    assert _dataset_path_matches({"path": declared}, "0") is matches


def test_a_single_level_pyramid_still_answers_for_an_unmatched_key(
    tmp_path: Path,
) -> None:
    """With one level there is nothing to be wrong about — keep answering."""
    path = tmp_path / "single_level.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "s0", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_ZYX, (1.0, 0.5, 0.5)),
    }

    assert discover_ome_zarr_shape(path, array_key="s0").voxel_size == (
        1.0,
        0.5,
        0.5,
    )


def test_a_single_level_pyramid_does_not_describe_an_array_in_another_group(
    tmp_path: Path,
) -> None:
    """A nested independent array must not inherit the root level's spacing."""
    path = tmp_path / "independent_nested_array.zarr"
    root = open_group(path, mode="w", zarr_format=3)
    create_array(
        root, "0", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
    cells = root.create_group("labels").create_group("cells")
    create_array(
        cells, "0", data=np.zeros((8, 16, 16), dtype=np.float32), compressor="auto"
    )
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": _multiscales(_ZYX, (1.0, 0.5, 0.5)),
    }

    assert discover_ome_zarr_shape(path, array_key="labels/cells/0").voxel_size is None


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
