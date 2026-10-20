"""Tests for demo_gsplats_3d_easifish_hypothalamus.

No network, no CZI, no GPU fit: the stitch runs on a synthetic two-view
acquisition served through a stand-in ``czifile`` module, the fits are checked
at the ``batch-fit run`` argument level, and the scene is built from tiny
on-disk archives. The demo is loaded by file path (see
test_demo_gsplats_3d_flylight_mcfo_63x_brain).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from luxar.gsplats.io.save_gsplats import save_gsplats

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_easifish_hypothalamus.py"
)


def _load_demo_module():
    name = "_luxar_demo_easifish_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


# =============================================================================
# A synthetic two-view acquisition
# =============================================================================
_Z, _H, _W = 4, 64, 128
_DX, _DY = 80, 6  # where view 1 lands relative to view 0
_EMISSIONS = {0: 498.0, 1: 648.0, 2: 415.0, 3: 571.0}


def _scene_volume(seed: int = 0) -> np.ndarray:
    """A (C, Z, H, W+DX) specimen wider than one view, with a textured DAPI."""
    rng = np.random.default_rng(seed)
    shape = (4, _Z, _H + _DY, _W + _DX)
    volume = rng.integers(100, 140, size=shape).astype(np.uint16)
    # Distinct blobs so the max-projection has structure to register on.
    for _ in range(60):
        y = int(rng.integers(0, shape[2]))
        x = int(rng.integers(0, shape[3]))
        volume[:, :, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] += 2000
    return volume


class _FakeView:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def asarray(self) -> np.ndarray:
        return self._data


class _FakeImage:
    """The slice of ``czifile.CziImage`` the demo touches."""

    def __init__(self, specimen: np.ndarray) -> None:
        # View 0 covers columns [0, W); view 1 covers [DX, DX+W), DY rows down.
        self._views = {
            0: specimen[:, :, :_H, :_W],
            1: specimen[:, :, _DY : _DY + _H, _DX : _DX + _W],
        }
        self.sizes = {"V": 2, "C": 4, "Z": _Z, "Y": _H, "X": _W}
        self.attrs = {
            "coord_scales": {"Z": 0.42e-6, "Y": 0.23e-6, "X": 0.23e-6},
            "channels": {
                f"ch{c}": {"EmissionWavelength": em} for c, em in _EMISSIONS.items()
            },
        }

    def __call__(self, *, V: int, C: int) -> _FakeView:
        return _FakeView(np.ascontiguousarray(self._views[V][C]))


def _fake_czifile(specimen: np.ndarray, positions_um: tuple[float, float]) -> Any:
    image = _FakeImage(specimen)
    metadata = "".join(f"<PositionX>{p}</PositionX>" for p in positions_um)

    class _CziFile:
        def __init__(self, _path: str) -> None:
            self.scenes = [image]

        def metadata(self) -> str:
            return metadata

        def __enter__(self) -> "_CziFile":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    return SimpleNamespace(CziFile=_CziFile)


# =============================================================================
# Stitching
# =============================================================================
def test_find_tile_offset_recovers_the_true_offset() -> None:
    specimen = _scene_volume()
    image = _FakeImage(specimen)
    mips = [image(V=v, C=2).asarray().max(axis=0).astype(np.float32) for v in (0, 1)]
    # A nominal offset 20 px off, as the stage positions are on the real file.
    offset = _demo.find_tile_offset(mips, nominal_dx=_DX + 20)
    assert (offset.order, offset.dx, offset.dy) == ((0, 1), _DX, _DY)
    assert offset.ncc > 0.9


def test_find_tile_offset_searches_the_view_order() -> None:
    image = _FakeImage(_scene_volume(seed=1))
    mips = [image(V=v, C=2).asarray().max(axis=0).astype(np.float32) for v in (1, 0)]
    offset = _demo.find_tile_offset(mips, nominal_dx=_DX)
    assert offset.order == (1, 0)
    assert (offset.dx, offset.dy) == (_DX, _DY)


def test_blend_views_reassembles_the_specimen() -> None:
    specimen = _scene_volume(seed=2)
    image = _FakeImage(specimen)
    offset = _demo.TileOffset(order=(0, 1), dx=_DX, dy=_DY, ncc=1.0)
    out = _demo._blend_views(
        image(V=0, C=0).asarray(), image(V=1, C=0).asarray(), offset
    )
    # Rows present in both views; every column of the union.
    expected = specimen[0, :, _DY:_H, : _DX + _W]
    assert out.shape == expected.shape
    # Identical overlapping content blends back to itself (up to rounding).
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1


def test_blend_views_ramps_across_the_overlap() -> None:
    left = np.zeros((1, 8, 100), dtype=np.uint16)
    right = np.full((1, 8, 100), 1000, dtype=np.uint16)
    offset = _demo.TileOffset(order=(0, 1), dx=60, dy=0, ncc=1.0)
    out = _demo._blend_views(left, right, offset)
    assert out.shape == (1, 8, 160)
    assert out[0, 0, 59] == 0 and out[0, 0, 100] == 1000
    ramp = out[0, 0, 60:100].astype(int)
    assert ramp[0] == 0 and ramp[-1] == 1000
    assert np.all(np.diff(ramp) >= 0)


def test_downsample_is_a_block_mean_that_crops_remainders() -> None:
    volume = np.arange(3 * 5 * 6, dtype=np.uint16).reshape(3, 5, 6)
    out = _demo._downsample(volume, (2, 2, 2))
    assert out.shape == (1, 2, 3)
    assert out[0, 0, 0] == np.rint(volume[:2, :2, :2].mean())


@pytest.mark.parametrize(
    ("level", "expected"),
    [("1", (0.0, 0.115, 0.115)), ("2", (0.21, 0.345, 0.345))],
)
def test_level_translation_matches_block_centres(
    level: str, expected: tuple[float, float, float]
) -> None:
    assert _demo._level_translation_um(
        (0.42, 0.23, 0.23), _demo._LEVELS[level]
    ) == pytest.approx(expected)


def test_nominal_dx_reads_the_stage_positions() -> None:
    meta = "<PositionX>-398.811</PositionX><PositionX>4.609</PositionX>"
    assert _demo._nominal_dx(meta, 2, 1920, 0.22851993924221545) == 1765


def test_nominal_dx_falls_back_to_the_acquisition_overlap() -> None:
    assert _demo._nominal_dx("", 2, 1920, 0.2285) == round(1920 * 0.92)


def test_channel_order_is_checked_against_the_emission_wavelengths() -> None:
    image = _FakeImage(_scene_volume())
    _demo._check_channels(image)
    swapped = dict(image.attrs["channels"])
    swapped["ch0"], swapped["ch1"] = swapped["ch1"], swapped["ch0"]
    image.attrs = {**image.attrs, "channels": swapped}
    with pytest.raises(RuntimeError, match="Wrong file or channel order"):
        _demo._check_channels(image)


def test_stitch_writes_a_two_level_ome_zarr_with_per_channel_floors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from luxar._zarr_compat import open_group

    specimen = _scene_volume(seed=3)
    pixel_um = 0.23
    fake = _fake_czifile(specimen, (0.0, (_DX + 10) * pixel_um))
    monkeypatch.setattr(_demo, "require_module", lambda name: fake)
    out = tmp_path / "stitched.ome.zarr"

    _demo.stitch_czi(tmp_path / "unused.czi", out)

    record = _demo._stitch_record(out)
    assert record is not None and record["complete"] is True
    assert (record["dx"], record["dy"]) == (_DX, _DY)
    group = open_group(str(out), mode="r")
    stitched_hw = (_H - _DY, _W + _DX)
    assert group["1"].shape == (4, _Z, stitched_hw[0] // 2, stitched_hw[1] // 2)
    assert group["2"].shape == (4, _Z // 2, stitched_hw[0] // 4, stitched_hw[1] // 4)
    datasets = dict(group.attrs)["multiscales"][0]["datasets"]
    assert [d["path"] for d in datasets] == ["1", "2"]
    scale_1 = datasets[0]["coordinateTransformations"][0]["scale"]
    assert scale_1 == pytest.approx([1.0, 0.42, 0.46, 0.46])
    for dataset, factors in zip(datasets, ((1, 2, 2), (2, 4, 4)), strict=True):
        assert dataset["coordinateTransformations"][1] == {
            "type": "translation",
            "translation": pytest.approx(
                [
                    0.0,
                    *(
                        ((f - 1) / 2) * v
                        for f, v in zip(factors, (0.42, 0.23, 0.23), strict=True)
                    ),
                ]
            ),
        }
    # One floor per channel per level, each resolved on its own array.
    assert all(len(v) == 4 for v in record["floors"].values())


def test_a_partial_stitch_is_not_reused(tmp_path: Path) -> None:
    from luxar._zarr_compat import open_group

    out = tmp_path / "partial.ome.zarr"
    group = open_group(str(out), mode="w")
    group.attrs["easifish_stitch"] = {"dx": 1, "dy": 0}  # no "complete"
    assert _demo._stitch_record(out) is None
    assert _demo._stitch_record(tmp_path / "absent.ome.zarr") is None


# =============================================================================
# Fitting
# =============================================================================
def _flag_value(args: tuple[str, ...], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_every_mrna_channel_pins_min_extent_zero() -> None:
    """mRNA channels keep every splat for spot-level fidelity (~1% recall)."""
    for ch in _demo.CHANNELS:
        if ch.key == "dapi":
            assert "--min-extent" not in ch.fit_args
        else:
            assert _flag_value(ch.fit_args, "--min-extent") == "0"
            assert _flag_value(ch.fit_args, "--tiling") == "content"


def test_channels_cover_the_four_czi_channels_once() -> None:
    assert sorted(ch.index for ch in _demo.CHANNELS) == [0, 1, 2, 3]
    assert {ch.colormap for ch in _demo.CHANNELS} == {
        "gray",
        "magenta",
        "green",
        "yellow",
    }


def test_fit_channel_passes_its_own_floor_and_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_demo, "RECOMPUTE_DIR", tmp_path)
    monkeypatch.delenv("LUXAR_AUTO_WORKER_HARD_CAP", raising=False)
    calls: list[tuple[str, ...]] = []
    caps: list[str | None] = []

    def fake_cli(*args: str) -> None:
        import os

        calls.append(args)
        caps.append(os.environ.get("LUXAR_AUTO_WORKER_HARD_CAP"))
        final = Path(args[4]) / "merged" / "final.gsplats.zarr"
        final.mkdir(parents=True)

    monkeypatch.setattr(_demo, "run_luxar_cli", fake_cli)
    bdnf = next(ch for ch in _demo.CHANNELS if ch.key == "bdnf")
    path = _demo.fit_channel(bdnf, 236.951)
    assert path == tmp_path / "fit_bdnf" / "merged" / "final.gsplats.zarr"
    (args,) = calls
    assert args[:3] == ("gsplat", "batch-fit", "run")
    assert _flag_value(args, "--channels") == "3:4"
    assert _flag_value(args, "--array-key") == "1"
    assert _flag_value(args, "--floor") == "236.951"
    assert _flag_value(args, "--merge-recipe") == "levels"
    assert "--physical" in args
    # Auto-sized workers, capped through the environment only while fitting.
    assert _flag_value(args, "--jobs-per-gpu") == "auto"
    assert caps == [str(_demo.WORKER_CAP)]
    import os

    assert "LUXAR_AUTO_WORKER_HARD_CAP" not in os.environ
    # A second call finds the merged archive and does not refit.
    assert _demo.fit_channel(bdnf, 236.951) == path
    assert len(calls) == 1


def _finish_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from luxar._zarr_compat import open_group

    monkeypatch.setattr(_demo, "RECOMPUTE_DIR", tmp_path)
    monkeypatch.setattr(_demo, "OME_ZARR", tmp_path / "stitched.ome.zarr")
    group = open_group(str(_demo.OME_ZARR), mode="w")
    group.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "path": "1",
                    "coordinateTransformations": [{"scale": [1.0, 0.42, 0.46, 0.46]}],
                },
                {
                    "path": "2",
                    "coordinateTransformations": [{"scale": [1.0, 0.84, 0.92, 0.92]}],
                },
            ]
        }
    ]


def test_finish_channel_reladders_then_scales_to_biological_um(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _finish_source(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fake_cli(*args: str) -> None:
        calls.append(args)
        Path(args[3]).mkdir(parents=True)

    monkeypatch.setattr(_demo, "run_luxar_cli", fake_cli)
    gpr83 = next(ch for ch in _demo.CHANNELS if ch.key == "gpr83")
    merged = _demo.merged_path(gpr83)
    merged.mkdir(parents=True)
    final = _demo.finish_channel(gpr83, merged)
    assert final == _demo.archive_path(gpr83)
    additive, transform = calls
    assert additive[:3] == ("gsplat", "additive", str(merged))
    assert _flag_value(additive, "-b") == "equi-energy:4"
    assert transform[:2] == ("gsplat", "transform")
    assert _flag_value(transform, "--scale") == "0.5,0.5,0.5"
    assert _flag_value(transform, "--translate") == "0,0.0575,0.0575"
    # Newer than the merge: reused, nothing re-run.
    assert _demo.finish_channel(gpr83, merged) == final
    assert len(calls) == 2
    monkeypatch.setattr(_demo, "EXPANSION_FACTOR", 4.0)
    assert _demo.finish_channel(gpr83, merged) == final
    assert len(calls) == 4
    assert _flag_value(calls[-1], "--translate") == "0,0.02875,0.02875"
    monkeypatch.setattr(
        _demo, "LADDER_ARGS", ("-b", "equi-energy:5", "-m", "self_energy")
    )
    assert _demo.finish_channel(gpr83, merged) == final
    assert len(calls) == 6
    assert _flag_value(calls[-2], "-b") == "equi-energy:5"
    monkeypatch.setattr(_demo, "EXPANSION_FACTOR", 2.0)
    dapi = next(ch for ch in _demo.CHANNELS if ch.key == "dapi")
    dapi_merged = _demo.merged_path(dapi)
    dapi_merged.mkdir(parents=True)
    _demo.finish_channel(dapi, dapi_merged)
    assert _flag_value(calls[-1], "--translate") == "0.105,0.1725,0.1725"


def test_finish_channel_rebuilds_after_interrupted_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _finish_source(tmp_path, monkeypatch)
    ch = next(ch for ch in _demo.CHANNELS if ch.level == "1")
    merged = _demo.merged_path(ch)
    merged.mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def interrupted_cli(*args: str) -> None:
        calls.append(args)
        Path(args[3]).mkdir(parents=True, exist_ok=True)
        if args[1] == "transform" and len(calls) == 2:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(_demo, "run_luxar_cli", interrupted_cli)
    with pytest.raises(RuntimeError, match="interrupted"):
        _demo.finish_channel(ch, merged)
    assert _demo.finish_channel(ch, merged) == _demo.archive_path(ch)
    assert [call[1] for call in calls] == [
        "additive",
        "transform",
        "additive",
        "transform",
    ]


def test_each_channel_is_fitted_with_its_own_level_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor is looked up by the channel's LEVEL, then its CZI INDEX."""
    floors = {"1": [101.0, 111.0, 121.0, 131.0], "2": [202.0, 212.0, 222.0, 232.0]}
    monkeypatch.setattr(
        _demo, "_stitch_record", lambda _p: {"complete": True, "floors": floors}
    )
    monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)
    seen: dict[str, float | None] = {}

    def fake_fit(ch, floor):
        seen[ch.key] = floor
        return tmp_path / ch.key

    monkeypatch.setattr(_demo, "fit_channel", fake_fit)
    monkeypatch.setattr(_demo, "finish_channel", lambda ch, merged: merged)
    paths = _demo.recompute_channel_paths()
    assert seen == {"dapi": 222.0, "slc17a6": 111.0, "gpr83": 101.0, "bdnf": 131.0}
    assert paths == {ch.key: tmp_path / ch.key for ch in _demo.CHANNELS}


def test_a_missing_czi_reader_fails_before_the_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_demo, "_stitch_record", lambda _p: None)
    monkeypatch.setattr(_demo, "warn_if_no_cuda_gpu", lambda: None)

    def missing(name: str):
        raise ImportError(f"no {name}")

    def fetch():
        raise AssertionError("the download started before the reader was checked")

    monkeypatch.setattr(_demo, "require_module", missing)
    monkeypatch.setattr(_demo, "fetch_czi", fetch)
    with pytest.raises(ImportError, match="czifile"):
        _demo.recompute_channel_paths()


def test_a_user_czi_is_refused_not_deleted_on_a_hash_mismatch(tmp_path: Path) -> None:
    wrong = tmp_path / "LHA3_R3_small.czi"
    wrong.write_bytes(b"not the medium acquisition")
    with pytest.raises(SystemExit, match="pinned"):
        _demo.user_czi(wrong)
    assert wrong.exists()
    with pytest.raises(SystemExit, match="no such file"):
        _demo.user_czi(tmp_path / "absent.czi")


def test_stitch_refuses_a_bad_layout_or_registration() -> None:
    image = _FakeImage(_scene_volume())
    with pytest.raises(RuntimeError, match="2 tile views"):
        _demo._check_layout({"V": 3, "C": 4}, image)
    with pytest.raises(RuntimeError, match="uniquely named"):
        _demo._check_layout({"V": 2, "C": 5}, image)
    good = _demo.TileOffset(order=(0, 1), dx=80, dy=0, ncc=0.93)
    _demo._check_offset(good, 128)
    with pytest.raises(RuntimeError, match="NCC"):
        _demo._check_offset(_demo.TileOffset((0, 1), 80, 0, 0.2), 128)
    with pytest.raises(RuntimeError, match="outside"):
        _demo._check_offset(_demo.TileOffset((0, 1), 0, 0, 0.93), 128)


def test_resolution_is_hosting_agnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever resolves the fits, the scene gets one archive per channel."""
    sentinel = {ch.key: Path(ch.key) for ch in _demo.CHANNELS}
    monkeypatch.setattr(_demo, "recompute_channel_paths", lambda: sentinel)
    assert set(_demo.resolve_channel_paths()) == {ch.key for ch in _demo.CHANNELS}


def test_a_scene_older_than_a_fit_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    scene = tmp_path / "scene.luxar.zarr"
    scene.mkdir()
    fit = tmp_path / "fit.gsplats.zarr"
    fit.mkdir()
    monkeypatch.setattr(_demo, "scene_is_current", lambda *a, **k: True)
    os.utime(fit, (1_000, 1_000))
    os.utime(scene, (2_000, 2_000))
    assert _demo.scene_is_fresh(scene, {"dapi": fit})
    os.utime(fit, (3_000, 3_000))
    assert not _demo.scene_is_fresh(scene, {"dapi": fit})
    monkeypatch.setattr(_demo, "scene_is_current", lambda *a, **k: False)
    os.utime(fit, (1_000, 1_000))
    assert not _demo.scene_is_fresh(scene, {"dapi": fit})


def test_the_manifest_declares_the_dataset_as_redistributable_cc_by() -> None:
    manifest_path = _DEMO_PATH.parent / "data_manifest.json"
    spec = json.loads(manifest_path.read_text())["datasets"][_demo.DATASET]
    assert spec["license"] == "cc-by-4.0"
    assert spec["redistribute"] is True
    assert spec["bucket"] == "local-compute" and spec["files"] == []
    assert _demo.DATASET in _demo.DEMO_META["caches"]


# =============================================================================
# Scene
# =============================================================================
def _tiny_store(path: Path, offset: float) -> Path:
    rng = np.random.default_rng(int(offset))
    n = 32
    save_gsplats(
        path,
        centers=(rng.uniform(0.0, 50.0, (n, 3)) + offset).astype(np.float32),
        amplitudes=rng.uniform(1.0, 10.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )
    return path


def test_scene_has_four_additive_layers_with_authored_windows(
    tmp_path: Path,
) -> None:
    from luxar._zarr_compat import open_group

    paths = {
        ch.key: _tiny_store(tmp_path / f"{ch.key}.gsplats.zarr", float(i))
        for i, ch in enumerate(_demo.CHANNELS)
    }
    out = tmp_path / "scene.luxar.zarr"
    _demo.create_luxar_scene(paths, out, 110_718_328)

    root = open_group(str(out), mode="r")
    attrs = dict(root.attrs)
    assert attrs["citation"]["short"] == _demo.DEMO_META["citation"]["short"]
    layers = {}
    for ch in _demo.CHANNELS:
        node = dict(root[ch.label].attrs)
        layers[ch.key] = node
        assert node["layer"] is True
        assert node["blending_mode"] == "additive"
        assert node["colormap"] == ch.colormap
        # The window is authored relative to the 99.9th percentile, which the
        # stored amplitudes put at `brightness`.
        lo, hi = (v * ch.brightness for v in ch.window)
        assert node["intensity"] == pytest.approx(1.0 / (hi - lo))
        assert node["offset"] == pytest.approx(-lo / (hi - lo))
        assert node["amplitude_normalization_factor"] > 0
    assert len(layers) == 4


def _scene_texts(scene: Path) -> tuple[str, list[str]]:
    from luxar._zarr_compat import group_keys, open_group

    root = open_group(str(scene), mode="r")
    overlays = root["overlays"]
    return dict(root.attrs)["description"], [
        dict(overlays[name].attrs)["text"] for name in group_keys(overlays)
    ]


def test_scene_text_states_its_size_against_the_raw_czi(tmp_path: Path) -> None:
    paths = {
        ch.key: _tiny_store(tmp_path / f"{ch.key}.gsplats.zarr", float(i))
        for i, ch in enumerate(_demo.CHANNELS)
    }
    out = tmp_path / "scene.luxar.zarr"
    _demo.create_luxar_scene(paths, out, 110_718_328)
    description, texts = _scene_texts(out)
    assert description.endswith(
        "Scene size: 111 MB on disk, from the 34.9 GB raw CZI (≈315x smaller)."
    )
    assert any(
        "splats • 111 MB, ≈315x smaller than the raw CZI • Wang et al. 2021" in t
        for t in texts
    )


def test_build_scene_restates_the_measured_packed_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final text states the size of the store it ends up in."""
    import shutil

    paths = {
        ch.key: _tiny_store(tmp_path / f"{ch.key}.gsplats.zarr", float(i))
        for i, ch in enumerate(_demo.CHANNELS)
    }
    packs: list[Path] = []

    def fake_pack(built: Path, output: Path) -> None:
        # A stand-in for `optimize --pack` that makes the scene 3x larger than
        # its splats, so the first (splat-sized) estimate is wrong.
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(built, output)
        (output / "padding.bin").write_bytes(b"\0" * 2 * _demo.dir_size_bytes(built))
        packs.append(output)

    monkeypatch.setattr(_demo, "local_fit_path", lambda _ds, name: tmp_path / name)
    monkeypatch.setattr(_demo, "pack_scene", fake_pack)
    out = tmp_path / "out.luxar.zarr"
    measured = _demo.build_scene(paths, out)

    assert measured == _demo.dir_size_bytes(out)
    assert len(packs) == 2  # the estimate was off; one restatement converged
    description, texts = _scene_texts(out)
    note, caption = _demo._size_texts(measured)
    assert note in description
    assert any(caption in t for t in texts)
    assert not (tmp_path / "scene_unpacked.luxar.zarr").exists()
    # Every pass packed into the scratch sibling, never into the served path.
    assert set(packs) == {tmp_path / ".out.luxar.zarr.packing"}
    assert not packs[0].exists()


def test_an_interrupted_pass_leaves_output_path_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    paths = {
        ch.key: _tiny_store(tmp_path / f"{ch.key}.gsplats.zarr", float(i))
        for i, ch in enumerate(_demo.CHANNELS)
    }
    out = tmp_path / "out.luxar.zarr"
    out.mkdir()
    (out / "previous.txt").write_text("the scene served before this build")
    calls = 0

    def crashing_pack(built: Path, output: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:  # the restating pass dies (Ctrl-C, OOM, ...)
            raise KeyboardInterrupt
        if output.exists():  # `optimize --overwrite`
            shutil.rmtree(output)
        shutil.copytree(built, output)
        (output / "padding.bin").write_bytes(b"\0" * 2 * _demo.dir_size_bytes(built))

    monkeypatch.setattr(_demo, "local_fit_path", lambda _ds, name: tmp_path / name)
    monkeypatch.setattr(_demo, "pack_scene", crashing_pack)
    with pytest.raises(KeyboardInterrupt):
        _demo.build_scene(paths, out)
    assert [p.name for p in out.iterdir()] == ["previous.txt"]

    # The next run cleans the interrupted scratch and replaces the old scene.
    calls = 10
    _demo.build_scene(paths, out)
    assert not (out / "previous.txt").exists()
    assert not (tmp_path / ".out.luxar.zarr.packing").exists()
    assert not (tmp_path / ".out.luxar.zarr.old").exists()
    note, _caption = _demo._size_texts(_demo.dir_size_bytes(out))
    assert note in _scene_texts(out)[0]


@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_camera_looks_at_the_slab_with_image_rows_down(
    tilt: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_demo, "CAMERA_TILT_DEG", tilt)
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([250.0, 435.0, 835.0])  # (z, y, x) in um
    camera = _demo._camera(bmin, bmax)
    assert camera.up == (0.0, -1.0, 0.0)
    assert camera.target == pytest.approx(tuple((bmin + bmax) / 2))
    assert camera.position[0] > bmax[0]
    eye_to_target = np.asarray(camera.position) - np.asarray(camera.target)
    screen_right = np.cross(np.asarray(camera.up), eye_to_target)
    screen_right /= np.linalg.norm(screen_right)
    assert screen_right == pytest.approx([0.0, 0.0, 1.0])


def test_pack_scene_writes_chunk_packs_into_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(_demo, "run_luxar_cli", lambda *a: calls.append(a))
    _demo.pack_scene(tmp_path / "built", tmp_path / "out")
    assert calls == [
        (
            "optimize",
            str(tmp_path / "built"),
            str(tmp_path / "out"),
            "--pack",
            "--overwrite",
        )
    ]
