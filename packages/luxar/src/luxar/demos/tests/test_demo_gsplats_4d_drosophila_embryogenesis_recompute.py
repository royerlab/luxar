"""Tests for the Drosophila embryogenesis archive recompute recipe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from luxar.demos import demo_gsplats_4d_drosophila_embryogenesis as demo


def _write_source(
    path: Path,
    *,
    shape: tuple[int, ...] = (500, 108, 1352, 532),
    dtype: str = "uint16",
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_array(
        demo.SOURCE_ARRAY_KEY,
        shape=shape,
        chunks=(1,) * len(shape),
        dtype=dtype,
    )


def _write_zip_source(
    path: Path,
    *,
    shape: tuple[int, ...] = (500, 108, 1352, 532),
    dtype: str = "uint16",
) -> None:
    with ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store)
        root.create_array(
            demo.SOURCE_ARRAY_KEY,
            shape=shape,
            chunks=(1,) * len(shape),
            dtype=dtype,
        )


def test_recorded_recipe_constants_match_the_source_and_published_run() -> None:
    assert demo.SOURCE_FILENAME == "DrosophilaHistone.zarr.zip"
    assert demo.SOURCE_ARRAY_KEY == "data"
    assert demo.SOURCE_SHAPE == (500, 108, 1352, 532)
    assert demo.SOURCE_DTYPE == np.dtype("uint16")
    assert demo.SOURCE_AXES == "time,z,y,x"
    assert demo.TILE_SIZE >= max(demo.SOURCE_SHAPE[1:])
    assert demo.EXPECTED_FITTED_SPLATS == 128_000_000
    assert demo.EXPECTED_SPLATS == 83_221_420
    assert demo.EXPECTED_RUNGS == 14


def test_validate_source_accepts_directory_and_zip_without_writing(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "recording.zarr"
    _write_source(directory)
    directory_files = sorted(
        path.relative_to(directory) for path in directory.rglob("*")
    )

    archive = tmp_path / demo.SOURCE_FILENAME
    _write_zip_source(archive)
    archive_bytes = archive.read_bytes()

    demo._validate_source(directory)
    demo._validate_source(archive)

    assert (
        sorted(path.relative_to(directory) for path in directory.rglob("*"))
        == directory_files
    )
    assert archive.read_bytes() == archive_bytes


def test_open_source_closes_its_zip_store(tmp_path: Path) -> None:
    archive = tmp_path / demo.SOURCE_FILENAME
    _write_zip_source(archive)

    with demo._open_source(archive) as root:
        store = root.store
        assert store._is_open
    assert not store._is_open


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        ((500, 108, 1352), "uint16", "shape"),
        ((499, 108, 1352, 532), "uint16", "shape"),
        ((500, 108, 1352, 532), "float32", "dtype"),
    ],
)
def test_validate_source_rejects_wrong_array_contract(
    tmp_path: Path,
    shape: tuple[int, ...],
    dtype: str,
    message: str,
) -> None:
    source = tmp_path / "wrong.zarr"
    _write_source(source, shape=shape, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        demo._validate_source(source)


def test_validate_source_rejects_missing_array_and_invalid_stores(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.zarr"
    zarr.open_group(str(missing), mode="w")
    with pytest.raises(ValueError, match="has no 'data' array"):
        demo._validate_source(missing)

    bare = tmp_path / "bare.zarr"
    zarr.open_array(str(bare), mode="w", shape=demo.SOURCE_SHAPE, dtype="uint16")
    with pytest.raises(ValueError, match="bare array, not a group"):
        demo._validate_source(bare)

    invalid_directory = tmp_path / "not-zarr"
    invalid_directory.mkdir()
    (invalid_directory / "README.txt").write_text("not zarr", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected the DrosophilaHistone recording"):
        demo._validate_source(invalid_directory)

    invalid_zip = tmp_path / "not-zarr.zip"
    invalid_zip.write_bytes(b"not a zip store")
    with pytest.raises(ValueError, match="Expected the DrosophilaHistone recording"):
        demo._validate_source(invalid_zip)


def test_rebuilt_archive_requires_one_leaf_and_the_recorded_rungs(monkeypatch) -> None:
    valid = SimpleNamespace(
        n_splats=demo.EXPECTED_SPLATS,
        n_additive_sublods=demo.EXPECTED_RUNGS,
    )
    monkeypatch.setattr(demo, "iter_leaves", lambda _node: [valid])
    assert demo._validate_rebuilt_archive(object()) == demo.EXPECTED_SPLATS

    monkeypatch.setattr(demo, "iter_leaves", lambda _node: [valid, valid])
    with pytest.raises(RuntimeError, match="2 leaves, expected one"):
        demo._validate_rebuilt_archive(object())

    invalid_rungs = SimpleNamespace(
        n_splats=demo.EXPECTED_SPLATS,
        n_additive_sublods=demo.EXPECTED_RUNGS - 1,
    )
    monkeypatch.setattr(demo, "iter_leaves", lambda _node: [invalid_rungs])
    with pytest.raises(RuntimeError, match="progressive rungs"):
        demo._validate_rebuilt_archive(object())


def test_splat_count_uses_a_one_percent_tolerance() -> None:
    demo._validate_splat_count(round(demo.EXPECTED_SPLATS * 0.991))

    with pytest.raises(RuntimeError, match="Over 1% means the recipe changed"):
        demo._validate_splat_count(round(demo.EXPECTED_SPLATS * 0.989))


def test_recompute_requires_source_and_invokes_exact_cli_paths(
    monkeypatch, tmp_path: Path
) -> None:
    work_dir = tmp_path / "recompute"
    monkeypatch.setattr(demo, "SOURCE_ARG", None)
    with pytest.raises(SystemExit, match="--recompute needs --source PATH"):
        demo.recompute_archive(work_dir)

    monkeypatch.setattr(demo, "SOURCE_ARG", tmp_path / "missing.zarr")
    with pytest.raises(FileNotFoundError, match="--source does not exist"):
        demo.recompute_archive(work_dir)

    source = tmp_path / demo.SOURCE_FILENAME
    source.touch()
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(demo, "SOURCE_ARG", source)
    monkeypatch.setattr(demo, "_validate_source", lambda _path: None)
    monkeypatch.setattr(demo, "run_luxar_cli", lambda *args: calls.append(args))
    monkeypatch.setattr(demo, "load_gsplat_node", lambda _path: (object(), {}))
    monkeypatch.setattr(
        demo,
        "iter_leaves",
        lambda _node: [
            SimpleNamespace(
                n_splats=demo.EXPECTED_SPLATS,
                n_additive_sublods=demo.EXPECTED_RUNGS,
            )
        ],
    )

    result = demo.recompute_archive(work_dir)

    fit = work_dir / "fit"
    culled = work_dir / "culled.gsplats.zarr"
    scaled = work_dir / "um.gsplats.zarr"
    final = work_dir / "drosophila_embryogenesis_500tp.gsplats.zarr.zip"
    assert result == final
    assert calls == [
        (
            "gsplat",
            "batch-fit",
            "run",
            str(source),
            str(fit),
            "--array-key",
            "data",
            "--axes",
            "time,z,y,x",
            "--tiling",
            "uniform",
            "--tile-size",
            "1400",
            "--overlap",
            "0",
            "--preset",
            "standard",
            "--iters",
            "1500",
            "--seeds",
            "256000",
            "--floor",
            "8",
            "--gpus",
            "0",
            "--jobs-per-gpu",
            "2",
            "--merge-recipe",
            "stream",
            "--merge-target-ms",
            "200",
        ),
        (
            "gsplat",
            "cull",
            str(fit / "merged" / "final.gsplats.zarr"),
            str(culled),
            "--method",
            "cumulative",
            "--retention",
            "0.960",
        ),
        (
            "gsplat",
            "transform",
            str(culled),
            str(scaled),
            "--scale",
            "1.93,0.40625,0.40625,1",
        ),
        (
            "optimise",
            str(scaled),
            str(final),
            "--profile",
            "archive",
        ),
    ]


def test_main_routes_recompute_output_into_the_scene(
    monkeypatch, tmp_path: Path
) -> None:
    rebuilt = tmp_path / "rebuilt.gsplats.zarr.zip"
    scene = tmp_path / "scene.luxar.zarr"
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(demo, "RECOMPUTE", True)
    monkeypatch.setattr(demo, "SERVE_ONLY", False)
    monkeypatch.setattr(demo, "NO_SERVE", True)
    monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
    monkeypatch.setattr(demo, "recompute_archive", lambda _work_dir: rebuilt)
    monkeypatch.setattr(
        demo,
        "resolve_data",
        lambda: pytest.fail("the hosted archive must not resolve during --recompute"),
    )

    def create(source: Path, output: Path) -> Path:
        calls.append((source, output))
        return scene

    monkeypatch.setattr(demo, "create_luxar_scene", create)

    demo.main()

    assert calls == [(rebuilt, tmp_path / demo.SCENE_NAME)]
