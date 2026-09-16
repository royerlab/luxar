# test_batch_run.py
"""CLI smoke tests for `luxar gsplat batch-fit run` (the local runner command)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat
from luxar.cli.tests._testing import normalized_cli_output

runner = CliRunner()


def test_local_cpu_profile_preserves_auto_sizing(monkeypatch) -> None:
    """CPU execution still uses the legacy default profile for tile sizing."""
    import luxar.gsplats.gpu_profile as gpu_profile
    import luxar.gsplats.utils.device as device
    from luxar.cli.gsplat_ops.batch.run_orchestration import (
        _resolve_local_gpu_profile,
    )

    summary = {"oom_boundaries": {"3d": {"max_successful_shape": [192, 192, 192]}}}
    throughput = [{"voxels": 192**3, "throughput": 1.0}]
    monkeypatch.setattr(device, "resolve_gpu_selection", lambda spec: [])
    monkeypatch.setattr(gpu_profile, "get_gpu_summary", lambda: summary)
    monkeypatch.setattr(gpu_profile, "get_gpu_throughput_table", lambda: throughput)

    selected, manifest_name, max_shape, table = _resolve_local_gpu_profile("cpu")

    assert selected == []
    assert manifest_name == "CPU"
    assert max_shape == [192, 192, 192]
    assert table == throughput


def test_local_profile_rejects_invalid_gpu_spec_cleanly(monkeypatch) -> None:
    """Invalid local GPU selections surface as CLI parameter errors."""
    import luxar.gsplats.utils.device as device
    from luxar.cli.gsplat_ops.batch.run_orchestration import (
        _resolve_local_gpu_profile,
    )

    def _reject(spec: str) -> list[int]:
        raise ValueError(f"invalid GPU selection: {spec}")

    monkeypatch.setattr(device, "resolve_gpu_selection", _reject)

    with pytest.raises(typer.BadParameter, match="invalid GPU selection: bogus"):
        _resolve_local_gpu_profile("bogus")


def test_local_profile_uses_selected_devices(monkeypatch) -> None:
    """Local planning records and profiles the GPUs that will run the tasks."""
    import torch

    import luxar.gsplats.gpu_profile as gpu_profile
    import luxar.gsplats.utils.device as device
    from luxar.cli.gsplat_ops.batch.run_orchestration import (
        _resolve_local_gpu_profile,
    )

    class _Properties:
        def __init__(self, name: str, total_memory: int) -> None:
            self.name = name
            self.total_memory = total_memory

    properties = {
        0: _Properties("Large GPU", 48 * 1024**3),
        1: _Properties("Small GPU", 8 * 1024**3),
    }
    summaries = {
        "Large GPU": {
            "recommendations": {"peak_throughput_3d": {"shape": [512, 512, 512]}}
        },
        "Small GPU": {
            "oom_boundaries": {"3d": {"max_successful_shape": [256, 256, 256]}}
        },
    }
    tables = {"Large GPU": [{"voxels": 2}], "Small GPU": [{"voxels": 1}]}

    monkeypatch.setattr(device, "resolve_gpu_selection", lambda spec: [0, 1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties.__getitem__)
    monkeypatch.setattr(
        gpu_profile, "get_gpu_summary", lambda gpu_name: summaries.get(gpu_name)
    )
    monkeypatch.setattr(
        gpu_profile,
        "get_gpu_throughput_table",
        lambda gpu_name: tables.get(gpu_name),
    )

    selected, manifest_name, max_shape, throughput = _resolve_local_gpu_profile("0,1")

    assert selected == [0, 1]
    assert manifest_name == "Large GPU, Small GPU"
    assert max_shape == [256, 256, 256]
    assert throughput == [{"voxels": 1}]


def test_local_profile_does_not_size_from_only_part_of_selected_set(
    monkeypatch,
) -> None:
    """An unprofiled selected model prevents unsafe heterogeneous auto-sizing."""
    import torch

    import luxar.cli.gsplat_ops.batch.run_orchestration as orchestration
    import luxar.gsplats.gpu_profile as gpu_profile
    import luxar.gsplats.utils.device as device

    class _Properties:
        def __init__(self, name: str, total_memory: int) -> None:
            self.name = name
            self.total_memory = total_memory

    properties = {
        0: _Properties("Profiled GPU", 48 * 1024**3),
        1: _Properties("Unprofiled GPU", 8 * 1024**3),
    }
    output: list[str] = []
    monkeypatch.setattr(device, "resolve_gpu_selection", lambda spec: [0, 1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties.__getitem__)
    monkeypatch.setattr(
        gpu_profile,
        "get_gpu_summary",
        lambda gpu_name: (
            {"recommendations": {}} if gpu_name == "Profiled GPU" else None
        ),
    )
    monkeypatch.setattr(orchestration, "aprint", lambda message: output.append(message))

    selected, manifest_name, max_shape, throughput = (
        orchestration._resolve_local_gpu_profile("0,1")
    )

    assert selected == [0, 1]
    assert manifest_name == "Profiled GPU, Unprofiled GPU"
    assert max_shape is None
    assert throughput is None
    assert output == [
        "no benchmark profile for Unprofiled GPU — tile auto-sizing off; "
        "pass --tile-size or run 'luxar gsplat benchmark'"
    ]


def _make_zarr(path: Path) -> None:
    import zarr

    rng = np.random.default_rng(0)
    V = np.zeros((48, 48, 48), np.float32)
    zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
    for _ in range(15):
        cz, cy, cx = rng.integers(6, 42, 3)
        V += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)).astype(
            np.float32
        )
    z = zarr.open_array(
        str(path), mode="w", shape=V.shape, chunks=(24, 24, 24), dtype="f4"
    )
    z[:] = np.clip(V, 0, 1)


def test_batch_fit_group_renamed_from_slurm_fit() -> None:
    """The group is `batch-fit` now; `slurm-fit` no longer resolves."""
    assert runner.invoke(app_gsplat, ["batch-fit", "--help"]).exit_code == 0
    assert runner.invoke(app_gsplat, ["slurm-fit", "--help"]).exit_code != 0


def test_run_dry_run_reports_plan_without_fitting(tmp_path: Path) -> None:
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    out = tmp_path / "out"
    res = runner.invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(out),
            "--axes",
            "z,y,x",
            "--tile-size",
            "64",  # > 48 -> single tile, no GPU profile needed
            "--gpus",
            "cpu",
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "LOCAL BATCH FIT" in res.output
    assert "Dry run" in res.output
    # Dry run must NOT fit: no tiles written.
    assert not (out / "tiles").exists() or not list((out / "tiles").glob("*.zarr"))


def test_run_dry_run_announces_whole_volume_seed_split(tmp_path: Path) -> None:
    """The plan announces how an integer ``--seeds`` divides across tiles (#1556).

    Every task is a ``--tile k/M`` fit that divides the budget itself, but the
    task pool captures worker output (``task_pool.py``), so the worker's own
    notice never reaches the console. Plan time is the one place the tile count
    is known before anything runs, so the notice belongs there — which also
    makes it visible under ``--dry-run``.
    """
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "out"),
            "--axes",
            "z,y,x",
            "--tile-size",
            "32",
            "--overlap",
            "8",  # 48^3 / 32 / 8 -> a 2x2x2 grid = 8 tiles
            "--gpus",
            "cpu",
            "--seeds",
            "800",
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "whole-volume budget" in res.output
    assert "at least 100 per non-empty tile" in res.output
    assert "8 grid tiles" in res.output
    assert "exact non-empty count" in res.output


def test_run_dry_run_no_seed_notice_without_seeds(tmp_path: Path) -> None:
    """Guard: no ``--seeds`` means no split notice (auto is sized per tile)."""
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        # fmt: off
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "out"),
            "--axes",
            "z,y,x",
            "--tile-size",
            "32",
            "--overlap",
            "8",
            "--gpus",
            "cpu",
            "--dry-run",
        ],
        # fmt: on
    )
    assert res.exit_code == 0, res.output
    assert "whole-volume budget" not in res.output


def test_run_rejects_npy_input_with_clear_message(tmp_path: Path) -> None:
    """batch-fit needs OME-Zarr; a .npy input must fail fast with a clear pointer
    (not an opaque zarr 'not a directory' error)."""
    npy = tmp_path / "vol.npy"
    np.save(npy, np.zeros((16, 16, 16), np.float32))
    res = runner.invoke(
        app_gsplat,
        ["batch-fit", "run", str(npy), str(tmp_path / "o"), "--gpus", "cpu"],
    )
    assert res.exit_code != 0
    output = normalized_cli_output(res)
    assert "OME-Zarr" in output and "gsplat fit" in output
    # Clean usage-error rendering (typer.BadParameter), not a raw traceback.
    assert "Traceback" not in output


def test_run_rejects_bad_tiling(tmp_path: Path) -> None:
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        ["batch-fit", "run", str(src), str(tmp_path / "o"), "--tiling", "nope"],
    )
    assert res.exit_code != 0


def test_run_uniform_auto_tile_without_profile_errors(tmp_path: Path) -> None:
    """A multi-tile uniform fit with no --tile-size and no GPU profile must fail
    with a clear message (auto-size needs a benchmark profile)."""
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "o"),
            "--axes",
            "z,y,x",
            "--gpus",
            "cpu",
        ],
    )
    # Either a clear tile-size error (no GPU profile -> typer.BadParameter, which
    # Typer renders cleanly as a usage error, exit 2) or a successful single-tile
    # plan if the volume fits and a profile is present (exit 0). Never an uncaught
    # crash, and any error must name --tile-size / the profile.
    assert res.exit_code in (0, 1, 2)
    if res.exit_code != 0:
        output = normalized_cli_output(res).lower()
        assert "tile-size" in output or "profile" in output


def _blob_field(shape: "tuple[int, int, int]" = (16, 24, 24), seed: int = 0):
    """A small 3D field of Gaussian blobs (the 'signal' the floor sits under)."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    blobs = np.zeros(shape, np.float32)
    for _ in range(8):
        cz, cy, cx = (rng.integers(4, s - 4) for s in shape)
        blobs += 10.0 * np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
        ).astype(np.float32)
    return blobs


def _write_timelapse(
    path: Path, frames: "np.ndarray", axes_attr: "list[str] | None" = None
) -> "np.ndarray":
    """Write a (T, Z, Y, X) zarr, optionally DECLARING its axes in ``.zattrs``.

    ``axes_attr`` is what lets a test exercise discovery WITHOUT passing
    ``--axes``: a store that declares ``["time", "z", "y", "x"]`` is 4D
    time-first, which the positional heuristic would otherwise read as CZYX.
    """
    import zarr

    z = zarr.open_array(
        str(path),
        mode="w",
        shape=frames.shape,
        chunks=(1, *frames.shape[1:]),
        dtype="f4",
    )
    z[:] = frames
    if axes_attr is not None:
        z.attrs["axes"] = list(axes_attr)
    return frames


def _make_timelapse_zarr(
    path: Path,
    n_t: int = 4,
    axes_attr: "list[str] | None" = None,
    pedestals: "list[float] | None" = None,
) -> "np.ndarray":
    """A (T, Z, Y, X) zarr whose background pedestal DRIFTS with time.

    A drifting pedestal is what makes per-timepoint floor estimation visible:
    each timepoint's own 'auto' level differs, so forwarding the spec instead of
    one resolved level yields a time-varying pedestal (brightness flicker).
    ``pedestals`` overrides the default ``5 + 20t`` ramp (e.g. a NON-monotone
    drift, where the store's minimum pedestal is not at ``t=0``).
    Returns the full array so a test can compute per-timepoint levels itself.
    """
    blobs = _blob_field()
    if pedestals is None:
        pedestals = [5.0 + 20.0 * t for t in range(n_t)]
    full = np.stack([blobs + p for p in pedestals]).astype(np.float32)
    return _write_timelapse(path, full, axes_attr=axes_attr)


def _auto_levels(full: "np.ndarray", spec: str = "auto") -> "list[float]":
    """Per-timepoint resolved levels of a (T, ...) array (the sampler's own rule)."""
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

    levels = [resolve_volume_floor(frame, spec, guard_numeric=True) for frame in full]
    assert all(level is not None for level in levels), levels
    return [float(level) for level in levels if level is not None]


_TZYX = ["t", "z", "y", "x"]


def _plan(
    src: Path,
    out: Path,
    *,
    tiling: str = "uniform",
    axes_list: "list[str] | None" = None,
    timepoints_slice: "str | None" = None,
    channels_slice: "str | None" = None,
    tile_size: int = 64,
    tile_overlap: int = 8,
    denoise_kwargs: "dict[str, object] | None" = None,
    **fit_kwargs: object,
):
    """Plan a batch over a (T, Z, Y, X) store — one uniform tile by default.

    ``axes_list`` defaults to the explicit ``t,z,y,x`` labels; pass ``None``
    explicitly via ``axes_list=[]`` — or rely on the store's own ``axes`` attr —
    to exercise the no-``--axes`` discovery path.

    ``tile_size`` defaults to 64, which is above every spatial extent of the
    ``(16, 24, 24)`` store and so gives a SINGLE tile (no GPU profile needed).
    Pass a smaller one for a genuinely multi-tile plan.
    """
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    return plan_batch(
        input_path=src,
        output_dir=out,
        tiling=tiling,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes_list=(_TZYX if axes_list is None else (axes_list or None)),
        array_key=None,
        timepoints_slice=timepoints_slice,
        channels_slice=channels_slice,
        fit=FitConfig(**fit_kwargs),  # type: ignore[arg-type]
        denoise=DenoiseConfig(**(denoise_kwargs or {})),  # type: ignore[arg-type]
        content=ContentKnobs(),
        merge=MergeConfig(),
    )


def test_batch_plan_records_ngff_dimension_metadata_with_axes_override(
    tmp_path: Path,
) -> None:
    import zarr

    src = tmp_path / "recording.ome.zarr"
    root = zarr.open_group(str(src), mode="w", zarr_format=2)
    root.create_array(
        "0",
        data=np.zeros((4, 16, 24, 24), dtype=np.float32),
        chunks=(1, 16, 24, 24),
    )
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 2.0, 0.75, 0.75]}
                    ],
                }
            ],
        }
    ]

    manifest = _plan(
        src,
        tmp_path / "out",
        axes_list=["time", "z", "y", "x"],
        floor="none",
    ).manifest

    assert manifest.dimension_metadata == [
        {"name": "z", "unit": "micrometer", "scale": 2.0},
        {"name": "y", "unit": "micrometer", "scale": 0.75},
        {"name": "x", "unit": "micrometer", "scale": 0.75},
        {"name": "time", "unit": "second", "scale": 0.5},
    ]


@pytest.mark.parametrize(
    ("fit_kwargs", "denoise_kwargs", "expected_flag"),
    [
        ({"progressive": True}, None, "--progressive"),
        ({}, {"denoise": True}, "--denoise"),
    ],
)
def test_batch_plan_rejects_unsupported_content_fit_before_discovery(
    tmp_path: Path,
    fit_kwargs: dict[str, object],
    denoise_kwargs: "dict[str, object] | None",
    expected_flag: str,
) -> None:
    """Content workers must not receive fit flags they silently ignore."""
    with pytest.raises(
        typer.BadParameter,
        match=rf"--tiling content.*{expected_flag}.*not supported",
    ):
        _plan(
            tmp_path / "missing.zarr",
            tmp_path / "out",
            tiling="content",
            denoise_kwargs=denoise_kwargs,
            **fit_kwargs,
        )


def test_batch_plan_allows_preprocessed_content_denoising(tmp_path: Path) -> None:
    """Preprocess mode retargets content workers to the denoised store."""
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    out = tmp_path / "out"
    result = plan_batch(
        input_path=src,
        output_dir=out,
        tiling="content",
        tile_size=None,
        tile_overlap=8,
        axes_list=_TZYX,
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(),
        denoise=DenoiseConfig(denoise=True, preprocess=True),
        content=ContentKnobs(
            k_star_ref=60000,
            n_features_ref=5000,
            cell=4,
            min_leaf=8,
            max_leaf=16,
        ),
        merge=MergeConfig(),
    )

    assert result.manifest.denoise_mode == "preprocess"
    assert result.manifest.denoised_zarr_path == str(out.resolve() / "denoised.zarr")
    assert "denoise" not in result.manifest.fit_args


@pytest.mark.parametrize(
    ("extra_args", "expected_flag"),
    [
        (["--denoise"], "--denoise"),
        (["--progressive"], "--progressive"),
        (["--downscale", "2"], "--downscale"),
    ],
)
def test_content_worker_warns_on_unsupported_fit_flags(
    tmp_path: Path,
    extra_args: list[str],
    expected_flag: str,
) -> None:
    """A direct content worker names every fit option it will ignore."""
    from luxar.gsplats.planner import FitPlan, PlanBox

    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    plan = tmp_path / "plan.json"
    FitPlan(
        volume_shape=[48, 48, 48],
        boxes=[PlanBox(box=[0, 48, 0, 48, 0, 48], n_features=0, budget=0)],
        overlap=0,
        feature_method="peaks",
        min_leaf=48,
        max_leaf=48,
    ).to_json(plan)
    output = tmp_path / "box.gsplats.zarr"

    result = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(src),
            str(output),
            "--tiling",
            "content",
            "--plan",
            str(plan),
            "--plan-box",
            "0",
            *extra_args,
        ],
    )

    assert result.exit_code == 0, result.output
    assert expected_flag in result.output
    assert "not supported" in result.output.lower()
    assert Path(f"{output}.empty").exists()


def test_batch_plan_resolves_one_global_floor_level(tmp_path: Path) -> None:
    """Every (t, c) task carries the manifest's ONE resolved level (#1174).

    Pre-fix the manifest forwarded the SPEC ('auto') and every task re-estimated
    the floor on its own timepoint — a time-varying pedestal across the merged
    partition. The level must now be a concrete number, identical in every task's
    argv, and equal to the MINIMUM of the sampled slices' levels (see
    ``resolve_batch_floor``: a minimum cannot erase a sampled slice).
    """
    from luxar.gsplats.batch.fit_command import build_task_fit_argv

    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)
    plan = _plan(src, tmp_path / "out", floor="auto")
    manifest = plan.manifest

    # T=4 -> all four timepoints are sampled, and each fits the per-slice budget
    # whole, so the plan's per-slice levels are exactly these.
    levels_per_t = _auto_levels(full)
    expected = min(levels_per_t)
    assert manifest.floor_level == pytest.approx(expected, rel=1e-6)
    assert float(manifest.fit_args["floor"]) == pytest.approx(expected, rel=1e-6)

    # The drift is real: the per-timepoint levels differ, so the single global
    # level above genuinely pins what spec-forwarding left free.
    assert max(levels_per_t) - expected > 1.0

    levels = set()
    for job in manifest.jobs:
        argv = build_task_fit_argv(manifest, job, tmp_path / "t.gsplats.zarr", argv0=[])
        levels.add(argv[argv.index("--floor") + 1])
    assert len(manifest.jobs) == 4
    assert len(levels) == 1
    assert float(levels.pop()) == pytest.approx(expected, rel=1e-6)


def test_batch_plan_resolves_one_global_normalization_range(tmp_path: Path) -> None:
    """Every timepoint gets one raw-input range resolved across the run."""
    from luxar.gsplats.batch.fit_command import build_task_fit_argv

    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)
    manifest = _plan(src, tmp_path / "out", floor="none").manifest
    expected = (float(full.min()), float(full.max()))

    assert manifest.norm_range == pytest.approx(expected)
    assert manifest.fit_args["norm_range"] == f"{expected[0]:.17g},{expected[1]:.17g}"
    ranges = set()
    for job in manifest.jobs:
        argv = build_task_fit_argv(manifest, job, tmp_path / "t.gsplats.zarr", argv0=[])
        ranges.add(argv[argv.index("--norm-range") + 1])
    assert ranges == {manifest.fit_args["norm_range"]}


def test_batch_denoise_does_not_pin_a_raw_sampled_normalization_range(
    tmp_path: Path,
) -> None:
    """Denoising tasks resolve ranges from the data each task fits."""
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)

    manifest = _plan(
        src,
        tmp_path / "out",
        floor="none",
        denoise_kwargs={"denoise": True, "denoise_h": 0.04},
    ).manifest

    assert manifest.norm_range is None
    assert "norm_range" not in manifest.fit_args


def test_batch_declines_a_degenerate_shared_normalization_range(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli.gsplat_ops.batch import planning

    sample = np.full(64, 50.0, np.float32)
    monkeypatch.setattr(
        planning, "_sample_batch_slices", lambda *args, **kwargs: [(0, 0, sample)]
    )
    assert planning.resolve_batch_norm_range(tmp_path / "unused.zarr", 0.0) is None


def test_batch_plan_preserves_an_explicit_config_normalization_range(
    tmp_path: Path, monkeypatch
) -> None:
    """A deliberate YAML range already is global and must not be re-measured."""
    from luxar.cli.gsplat_ops.batch import planning

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    config = tmp_path / "fit.yaml"
    config.write_text("norm_range: [2.5, 90.0]\nfloor: none\n")

    def _no_sample(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("concrete floor + range need no voxel sampling")

    monkeypatch.setattr(planning, "_sample_batch_slices", _no_sample)
    manifest = _plan(
        src, tmp_path / "out", floor=None, config=config, preset="standard"
    ).manifest

    assert manifest.norm_range == pytest.approx((2.5, 90.0))
    assert manifest.fit_args["norm_range"] == "2.5,90"


def test_batch_denoise_preserves_an_explicit_config_normalization_range(
    tmp_path: Path,
) -> None:
    """A configured range remains an explicit override under denoising."""
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    config = tmp_path / "fit.yaml"
    config.write_text("norm_range: [2.5, 90.0]\nfloor: none\n")

    manifest = _plan(
        src,
        tmp_path / "out",
        floor=None,
        config=config,
        preset="standard",
        denoise_kwargs={"denoise": True, "denoise_h": 0.04},
    ).manifest

    assert manifest.norm_range == pytest.approx((2.5, 90.0))
    assert manifest.fit_args["norm_range"] == "2.5,90"


@pytest.mark.parametrize(
    ("floor", "preprocess"),
    [("p10", False), ("auto", True)],
)
def test_batch_plan_defers_volume_floor_until_denoise_basis_exists(
    tmp_path: Path, floor: str, preprocess: bool
) -> None:
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)

    manifest = _plan(
        src,
        tmp_path / "out",
        floor=floor,
        denoise_kwargs={"denoise": True, "denoise_h": 0.04, "preprocess": preprocess},
    ).manifest

    assert manifest.floor_deferred is True
    assert manifest.floor_spec == floor
    assert manifest.floor_level is None
    assert "floor" not in manifest.fit_args


def test_batch_plan_does_not_defer_on_the_fly_auto_floor(tmp_path: Path) -> None:
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)

    manifest = _plan(
        src,
        tmp_path / "out",
        floor="auto",
        denoise_kwargs={"denoise": True, "denoise_h": 0.04, "preprocess": False},
    ).manifest

    assert manifest.floor_deferred is False
    assert manifest.floor_spec is None
    assert manifest.floor_level is not None


def test_batch_plan_global_level_cannot_erase_a_sampled_slice(tmp_path: Path) -> None:
    """The ONE global level stays below EVERY sampled (t, c)'s maximum.

    (Here T=4, so every timepoint is sampled and "sampled" == "every". A dimmer
    slice the bounded sampler skips is NOT covered by this bound — see
    ``_floor_sample_pairs``' residual-risk note.)

    FAILS against the previous version of this fix, which measured the level on a
    single representative timepoint (the middle of the selection). The uniform
    tile worker applies the forwarded number unguarded and batch always passes
    ``--allow-empty-tile``, so a level above a timepoint's maximum clips that
    whole sub-volume to zero: 0 splats, an ``.empty`` marker, a task that exits 0
    and a merge that skips it — the slice is silently MISSING while `status`
    reports success. On this fixture the middle timepoint's level (~45) is above
    the maxima of t=0 (~23) and t=1 (~43): half the timelapse would vanish.
    """
    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)
    manifest = _plan(src, tmp_path / "out", floor="auto").manifest

    level = manifest.floor_level
    assert level is not None
    for t, frame in enumerate(full):
        assert level < float(frame.max()), (
            f"level {level} would clip t={t} (max {float(frame.max())}) to zero"
        )
    # The hazard is real, not hypothetical: the middle timepoint's own level is
    # above the earlier timepoints' maxima.
    middle = _auto_levels(full)[len(full) // 2]
    assert middle > float(full[0].max()) and middle > float(full[1].max())


def test_batch_plan_level_is_safe_across_channels(tmp_path: Path) -> None:
    """A dim CHANNEL must survive the global level too, not just a dim timepoint.

    FAILS against the previous version of this fix: it measured the level on the
    first selected channel only, so a (3, 2, Z, Y, X) store whose c0 sits on a
    pedestal of 400 handed ~400 to BOTH channels — erasing c1 entirely, whose
    maximum is far below that pedestal.
    """
    src = tmp_path / "movie.zarr"
    blobs = _blob_field()
    frames = np.stack(
        [np.stack([blobs + 400.0, blobs + 39.0]) for _ in range(3)]
    ).astype(np.float32)
    _write_timelapse(src, frames, axes_attr=["time", "channel", "z", "y", "x"])

    manifest = _plan(src, tmp_path / "out", axes_list=[], floor="auto").manifest

    level = manifest.floor_level
    assert level is not None
    c1_max = float(frames[:, 1].max())
    # A c0-only level (what the previous version resolved) is above c1's whole
    # dynamic range: that is the erasure this test forbids.
    c0_only_level = _auto_levels(frames[:, 0])[0]
    assert c0_only_level > c1_max
    assert level < c1_max, f"level {level} erases c1 (max {c1_max})"


def test_batch_plan_level_ignores_the_timepoint_selection(tmp_path: Path) -> None:
    """The level is a property of the STORE, not of --timepoints/--channels.

    FAILS against the previous version of this fix, which resolved against the
    middle of the SELECTION: two runs over the same store (say a first pass over
    ``0:2`` and a resumed/extended pass over the whole movie) then subtracted
    different pedestals, so tiles already on disk no longer matched the new ones.
    """
    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)

    whole = _plan(src, tmp_path / "o-all", floor="auto").manifest
    tail = _plan(
        src, tmp_path / "o-tail", timepoints_slice="3:4", floor="auto"
    ).manifest

    assert whole.floor_level is not None
    assert tail.floor_level == pytest.approx(whole.floor_level, rel=1e-9)
    # And it really is the store-wide minimum, not the selected timepoint's own
    # level (which is much higher and would erase t=0..2).
    assert whole.floor_level == pytest.approx(min(_auto_levels(full)), rel=1e-6)


def test_batch_plan_floor_none_and_explicit_numeric_round_trip(tmp_path: Path) -> None:
    """`--floor none` and a user numeric behave exactly as before the fix."""
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)

    m_none = _plan(src, tmp_path / "o1", floor="none").manifest
    assert m_none.fit_args["floor"] == "none"
    assert m_none.floor_level is None

    m_num = _plan(src, tmp_path / "o2", floor="7.5").manifest
    assert float(m_num.fit_args["floor"]) == pytest.approx(7.5)
    assert m_num.floor_level == pytest.approx(7.5)


def test_resolve_batch_floor_needs_no_volume_for_concrete_specs(tmp_path: Path) -> None:
    """A concrete spec must not read the store (no volume load at plan time)."""
    from luxar.cli.gsplat_ops.batch.planning import resolve_batch_floor

    missing = tmp_path / "does-not-exist.zarr"
    assert resolve_batch_floor(missing, "none") == (None, "none")
    level, forward = resolve_batch_floor(missing, "12")
    assert level == pytest.approx(12.0) and forward == pytest.approx(12.0)
    # An explicitly disabled spec emits no --floor at all.
    assert resolve_batch_floor(missing, None) == (None, None)


def _spy_on_floor_sampling(monkeypatch) -> "list":
    """Record every object the bounded floor SAMPLER is handed, calling through.

    Seam: ``_sample_volume_for_floor`` is the one function that actually reads
    voxels, and it is imported inside the functions that use it, so patching the
    attribute on ``preprocessing`` catches every caller. Note the plan resolves
    each slice's level from an already-drawn sample, so a lazy view is followed by
    the flat ``np.ndarray`` sample it produced.
    """
    from luxar.gsplats.fitting import preprocessing

    seen: list = []
    original = preprocessing._sample_volume_for_floor

    def _spy(volume, budget):
        seen.append(volume)
        return original(volume, budget)

    monkeypatch.setattr(preprocessing, "_sample_volume_for_floor", _spy)
    return seen


def test_batch_plan_samples_a_lazy_view_not_a_materialized_volume(
    tmp_path: Path, monkeypatch
) -> None:
    """Plan time must not materialize a whole timepoint for the default 'auto'.

    FAILS pre-fix: the plan called ``load_volume``, which ends in
    ``np.asarray(..., float32)``, so the sampler was handed a NumPy array — a
    full multi-GB timepoint decoded on a login node (and again on every resume
    re-plan), defeating the bounded lazy sampling. The poison below is set on
    ``luxar.io.volume.load_volume``, which is the binding production resolves
    (the pinned-view helper imports it from there at call time).
    """
    from luxar.io import volume as volume_mod

    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)
    seen = _spy_on_floor_sampling(monkeypatch)

    def _no_eager_load(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("plan time must not materialize the volume")

    monkeypatch.setattr(volume_mod, "load_volume", _no_eager_load)

    manifest = _plan(src, tmp_path / "out", floor="auto").manifest

    # One lazy view per sampled (t, c) slice (T=4 here), each followed by the
    # flat sample it yielded.
    views = [item for item in seen if not isinstance(item, np.ndarray)]
    assert len(views) == 4, [type(item) for item in seen]
    for view in views:
        # The pinned view drops the time axis, so it looks like a plain volume to
        # the sampler while reading lazily.
        assert tuple(view.shape) == full.shape[1:]
    assert manifest.floor_level is not None


def test_batch_plan_pins_each_sampled_timepoint_by_label(
    tmp_path: Path, capsys
) -> None:
    """A 4D time-first store with NO --axes must read the timepoints it names.

    FAILS pre-fix: with ``axes=None`` the 4D positional heuristic takes the
    ``channel is not None`` branch ("Slicing 4D (CZYX)") and IGNORES
    ``timepoint``, so every sample was measured on ``arr[0]``. The pedestal here
    is NON-monotone, so ``arr[0]``'s level is NOT the minimum: a positional read
    resolves t=0's 40, while pinning by label finds t=1's 5.
    """
    src = tmp_path / "movie.zarr"
    # The store DECLARES its axes, so discovery knows it is time-first even
    # though the caller passes no --axes (which is what hid the bug).
    full = _make_timelapse_zarr(
        src, axes_attr=["time", "z", "y", "x"], pedestals=[40.0, 5.0, 30.0, 20.0]
    )
    manifest = _plan(src, tmp_path / "out", axes_list=[], floor="auto").manifest

    levels = _auto_levels(full)
    assert levels[0] > min(levels) + 1.0  # t=0 is NOT the minimum
    assert manifest.floor_level == pytest.approx(min(levels), rel=1e-6)
    # Every sampled slice is logged with the timepoint it was read from.
    out = capsys.readouterr().out
    for t in range(4):
        assert f"t={t}, c=0" in out


def test_batch_plan_percentile_spec_resolves_to_a_number(tmp_path: Path) -> None:
    """A `pNN` spec resolves end to end into one concrete level.

    FAILS pre-fix: the manifest carried the string 'p10' and every task ran its
    own percentile over its own timepoint.
    """
    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src)
    manifest = _plan(src, tmp_path / "out", floor="p10").manifest

    expected = min(_auto_levels(full, "p10"))
    assert manifest.floor_level == pytest.approx(expected, rel=1e-6)
    assert float(manifest.fit_args["floor"]) == pytest.approx(expected, rel=1e-6)


def test_batch_plan_guard_refusal_is_loud_and_disables_for_the_whole_run(
    tmp_path: Path, capsys
) -> None:
    """A refused level disables suppression for EVERY task — say so out loud.

    Pre-fix each task guarded itself, so healthy timepoints still suppressed;
    one global level means one refusal (on ANY sampled slice) turns it off
    everywhere, which must not be silent. FAILS pre-fix: no level was resolved at
    plan time at all (the spec was forwarded), so neither the recorded ``None``
    nor the warning existed.
    """
    src = tmp_path / "movie.zarr"
    frames = np.stack(
        [
            _blob_field() + 5.0,
            np.zeros((16, 24, 24), np.float32),  # degenerate: a blank timepoint
            _blob_field() + 45.0,
        ]
    ).astype(np.float32)
    _write_timelapse(src, frames)

    manifest = _plan(src, tmp_path / "out", floor="auto").manifest

    # One sampled timepoint is blank, so its 'auto' level is its own max (0) ->
    # the "would erase all signal" guard refuses it -> None everywhere.
    assert manifest.floor_level is None
    assert manifest.fit_args["floor"] == "none"
    out = capsys.readouterr().out
    assert "DISABLED for the" in out and "WHOLE run" in out


def test_batch_plan_level_survives_an_extreme_dynamic_range_spread(
    tmp_path: Path,
) -> None:
    """Even a 2000x dimmer timepoint keeps signal above the global level.

    The pathological shape for one global level: a nearly-black timepoint sitting
    entirely below the other timepoints' PEDESTAL. FAILS against the previous
    version of this fix, whose representative-timepoint level (100) was 250x that
    slice's maximum, erasing it outright. (This is the invariant
    ``resolve_batch_floor`` relies on instead of a redundant re-check: a minimum of
    individually-guarded per-slice levels is already below every sampled maximum.
    Asserted here on real data.)
    """
    src = tmp_path / "movie.zarr"
    blobs = _blob_field()
    frames = np.stack(
        [blobs + 100.0, blobs * 0.02 + 0.05, blobs + 100.0],
    ).astype(np.float32)
    _write_timelapse(src, frames)

    manifest = _plan(src, tmp_path / "out", floor="auto").manifest

    level = manifest.floor_level
    assert level is not None
    for t, frame in enumerate(frames):
        assert level < float(frame.max()), f"level {level} erases t={t}"
    # The previous rule (the middle selected timepoint) would have erased t=1.
    assert _auto_levels(frames)[1] < float(frames[1].max())
    assert _auto_levels(frames)[0] > float(frames[1].max())


def test_batch_plan_warns_and_forwards_the_spec_for_a_negative_level(
    tmp_path: Path, capsys
) -> None:
    """A negative level cannot be forwarded as a number — warn, don't abort.

    A negative level (dark-frame-corrected / deconvolved data) is legitimate
    input that planned and fit fine before #1174, and there is no way to express
    it as a concrete ``--floor`` (``_validate_floor`` rejects ``< 0``, and
    ``--floor -5.0`` is misparsed by click as an option). So the SPEC is
    forwarded, each task resolves it itself, and the manifest records no level —
    said out loud, because pedestals may then differ across the run. FAILS
    against the previous version of this fix, which raised ``typer.BadParameter``
    and made such a store un-batch-fittable (even on ``--dry-run``).
    """
    src = tmp_path / "movie.zarr"
    frames = np.stack([_blob_field() - (8.0 + 2.0 * t) for t in range(3)]).astype(
        np.float32
    )
    _write_timelapse(src, frames)

    manifest = _plan(src, tmp_path / "out", floor="auto").manifest

    assert manifest.floor_level is None  # "not pinned"
    assert manifest.fit_args["floor"] == "auto"  # the SPEC, so each task resolves
    out = capsys.readouterr().out
    assert "⚠" in out and "NEGATIVE" in out
    assert "floor_level=None" in out


def test_batch_plan_rejects_a_bad_floor_spec_before_reading_anything(
    tmp_path: Path, monkeypatch
) -> None:
    """An invalid spec is a usage error, paid for with zero volume reads.

    FAILS pre-fix: the spec was never validated at plan time (it was forwarded
    verbatim), and the first version of this fix validated only INSIDE
    ``resolve_shared_floor``, i.e. after the read, raising a bare ValueError.
    """
    from luxar.gsplats.fitting import preprocessing
    from luxar.io import volume as volume_mod

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)

    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an invalid --floor must not read the volume")

    monkeypatch.setattr(volume_mod, "load_volume", _boom)
    monkeypatch.setattr(preprocessing, "_sample_volume_for_floor", _boom)

    for bad in ("p150", "potato", "-5"):
        with pytest.raises(typer.BadParameter):
            _plan(src, tmp_path / f"out-{bad}", floor=bad)


def test_run_bad_floor_spec_is_a_clean_cli_error(tmp_path: Path) -> None:
    """The CLI renders it as a usage error, never a traceback."""
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        # fmt: off
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "o"),
            "--axes",
            "z,y,x",
            "--tile-size",
            "64",
            "--gpus",
            "cpu",
            "--floor",
            "p150",
            "--dry-run",
        ],
        # fmt: on
    )
    assert res.exit_code != 0
    output = normalized_cli_output(res)
    assert "Traceback" not in output
    assert "floor" in output.lower()


def test_manifest_predating_shared_normalization_loads_and_keeps_its_specs(
    tmp_path: Path,
) -> None:
    """A resumed pre-#1174 manifest behaves exactly as it was planned.

    FAILS pre-fix: ``floor_level`` did not exist, so the round-trip this pins
    (unknown-field-free load + spec preserved in the task argv) had nothing to
    assert against.
    """
    import json

    from luxar.gsplats.batch.fit_command import build_task_fit_argv
    from luxar.gsplats.batch.manifest import load_manifest, save_manifest

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    out = tmp_path / "out"
    manifest = _plan(src, out, floor="auto").manifest
    # Rewrite it the way a pre-#1174 plan would have: no floor_level, and the
    # SPEC in fit_args.
    manifest.fit_args = dict(manifest.fit_args, floor="auto")
    save_manifest(manifest, out)
    data = json.loads((out / "manifest.json").read_text())
    data.pop("floor_level")
    data.pop("norm_range")
    data["fit_args"].pop("norm_range")
    (out / "manifest.json").write_text(json.dumps(data))

    loaded = load_manifest(out)
    assert loaded.floor_level is None
    assert loaded.norm_range is None
    assert loaded.fit_args["floor"] == "auto"
    argv = build_task_fit_argv(
        loaded, loaded.jobs[0], tmp_path / "t.gsplats.zarr", argv0=[]
    )
    assert argv[argv.index("--floor") + 1] == "auto"
    assert "--norm-range" not in argv


def test_manifest_round_trips_normalization_range_as_a_tuple(tmp_path: Path) -> None:
    from luxar.gsplats.batch.manifest import BatchManifest, load_manifest, save_manifest

    save_manifest(BatchManifest(norm_range=(2.5, 90.0)), tmp_path)

    loaded = load_manifest(tmp_path)
    assert loaded.norm_range == (2.5, 90.0)
    assert isinstance(loaded.norm_range, tuple)


def test_effective_floor_spec_reads_the_config_chain(tmp_path: Path) -> None:
    """Unset --floor resolves the spec a task would use, not a blind 'auto'."""
    from luxar.cli.gsplat_ops.batch.planning import FitConfig, effective_floor_spec

    assert effective_floor_spec(FitConfig(floor="p10")) == "p10"
    assert effective_floor_spec(FitConfig(floor=None)) == "auto"
    cfg = tmp_path / "fit.yaml"
    cfg.write_text("floor: p25\n")
    assert effective_floor_spec(FitConfig(floor=None, config=cfg)) == "p25"
    # An explicit --floor still wins over the config.
    assert effective_floor_spec(FitConfig(floor="none", config=cfg)) == "none"
    # A MALFORMED config must not escape planning as a raw yaml traceback
    # (yaml.YAMLError is not a ValueError); the task's own load reports it, as
    # before, so planning falls back to the default spec.
    broken = tmp_path / "broken.yaml"
    broken.write_text("floor: [unclosed\n")
    assert effective_floor_spec(FitConfig(floor=None, config=broken)) == "auto"


def test_batch_plan_rejects_a_negative_config_floor(tmp_path: Path) -> None:
    """A `floor: -5.0` in a YAML config is rejected, not emitted as an argv flag.

    Guards a hazard THIS change introduces, not a pre-existing one: pre-#1174 the
    plan never looked inside ``--config``, so an invalid ``floor:`` there was the
    task's own problem (it read the config itself and raised at fit time). Now
    ``effective_floor_spec`` reads that chain at plan time and forwards the value
    into every task's argv, where ``--floor -5.0`` would be parsed by click as an
    option rather than a value — so plan time has to reject it.
    """
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    cfg = tmp_path / "fit.yaml"
    cfg.write_text("floor: -5.0\n")

    # floor=None = "--floor unset", which is what lets the config's floor apply.
    with pytest.raises(typer.BadParameter):
        _plan(src, tmp_path / "out", floor=None, config=cfg)


def test_batch_plan_refuses_a_downscale_whose_worker_grid_disagrees(
    tmp_path: Path,
) -> None:
    """A config `downscale:` that moves the workers off the planned grid (#1624).

    `--config` is handed to every array task VERBATIM, and a uniform task is a
    `fit --tile k/M` run that decimates its own volume and then re-tiles the
    DECIMATED shape, while the planner tiled the full-resolution one. Measured on
    this `(16, 24, 24)` store with `--tile-size 12 --overlap 4`: 18 planned tiles
    per slot against a worker that sees 4 (its volume is `(8, 12, 12)`), so each
    slot's tiles 4..17 exit 1 with "tile index out of range" and no merge ever
    happens.

    FAILS pre-fix: the plan succeeded, submitted every doomed task, and the
    failure only surfaced once the array job had started writing tiles. The
    message has to carry BOTH tile counts, because that difference is the whole
    diagnosis and neither number is visible from the config alone.
    """
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    cfg = tmp_path / "ds.yaml"
    cfg.write_text("downscale: 2\n")

    with pytest.raises(typer.BadParameter) as excinfo:
        _plan(src, tmp_path / "out", config=cfg, tile_size=12, tile_overlap=4)
    message = str(excinfo.value)
    assert "downscale" in message
    assert "18 tiles" in message  # what the plan built
    assert "only 4" in message  # what a worker would see
    assert "(8, 12, 12)" in message  # the decimated shape (ceil division)
    assert "gsplat fit -j N --downscale" in message  # the working alternative


def test_batch_plan_records_no_frame_for_a_single_tile_downscale(
    tmp_path: Path,
) -> None:
    """A SINGLE-tile uniform plan completes with a `downscale:`, so it is not refused.

    Measured: with one tile the worker's decimated volume still yields exactly
    one tile, so `--tile 0/1` passes its bounds check, and `rescale_and_save`
    hands the splats back at FULL resolution — the frame the planner tiled.

    That is also why nothing is recorded as `grid_scale`: composing the
    downscale into that factor (as this planner did before #1624) changed no
    split plane, but falsely refused `--merge-refine volume`, whose per-part
    crops are taken in voxels off the very grid the splats are already in.

    The `grid_scale is None` half is a REGRESSION TEST against the base, which
    recorded `[2.0, 2.0, 2.0]` here. The "plan does not raise" half is not — the
    base had no gate to raise — it guards this gate's own first version, which
    refused every plan carrying a decimating value.
    """
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    cfg = tmp_path / "ds.yaml"
    cfg.write_text("downscale: 2\n")

    manifest = _plan(src, tmp_path / "out", config=cfg).manifest

    assert manifest.n_tiles == 1
    assert manifest.grid_scale is None
    assert manifest.total_tasks == 4  # T=4, C=1, one tile


def test_batch_plan_accepts_a_downscale_confined_to_single_tile_axes(
    tmp_path: Path,
) -> None:
    """A MULTI-tile uniform plan whose decimated grid is the PLANNED one (#1624).

    `n_tiles > 1` is not the real condition — the worker's grid differing from
    the planner's is — and an anisotropic factor that decimates only axes holding
    a single tile at full resolution leaves the grid alone. Measured with
    `compute_tile_specs` on this `(16, 24, 24)` store, `--tile-size 20
    --overlap 4` (stride 16): the plan builds 4 tiles (z: 16 <= 20, one tile;
    y/x: origins 0 and 16, two each), and a `downscale: [2, 1, 1]` worker tiling
    its decimated `(8, 24, 24)` builds 4 as well, at origins
    `(0,0,0) (0,0,16) (0,16,0) (0,16,16)` — identical to the planner's, because
    the only rescaled axis has origin 0 in every tile. Every task then rescales
    its splats back to full resolution, so the run completes.

    The `grid_scale is None` half IS a regression test against the base, which
    recorded `[2.0, 1.0, 1.0]` here and so mis-scaled this plan's split planes
    (and falsely refused `--merge-refine volume`). The "plan does not raise" half
    is not — the base had no gate to raise — it guards this gate's own first
    version, whose `n_tiles <= 1` proxy refused this plan, with a message whose
    tile range degenerated to the empty "tiles 4..3".
    """
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    cfg = tmp_path / "ds-z.yaml"
    cfg.write_text("downscale: [2, 1, 1]\n")

    manifest = _plan(
        src, tmp_path / "out", config=cfg, tile_size=20, tile_overlap=4
    ).manifest

    assert manifest.n_tiles == 4  # multi-tile: not the single-tile trivial pass
    assert manifest.grid_scale is None
    assert manifest.total_tasks == 4 * 4  # T=4, C=1, four tiles


def test_batch_plan_accepts_a_downscale_in_content_mode(tmp_path: Path) -> None:
    """A content-mode plan completes with a `downscale:`, so it is not refused.

    A content task is `fit --tiling content --plan plan.json --plan-box K`, which
    never re-tiles — it bounds-checks `K` against the same plan JSON the planner
    wrote — and `_fit_one_box` crops the FULL-resolution volume, after which
    `finalize_results` rescales the crop's splats back to full resolution before
    the box origin is added. Measured: with `downscale: 2` leaked in, the centers
    land in the same global full-resolution range as without it.

    Deliberately a MULTI-box plan (4 boxes here), so the exemption is not
    passing on the strength of a degenerate single-box decomposition.

    NOT a regression test against the base: there was no gate there, and content
    mode never resolved a `grid_scale` either, so both assertions passed. It
    guards this gate's own first version, which sat ahead of the mode split and
    refused a content plan too.
    """
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    cfg = tmp_path / "ds.yaml"
    cfg.write_text("downscale: 2\n")

    manifest = plan_batch(
        input_path=src,
        output_dir=tmp_path / "out-content",
        tiling="content",
        tile_size=None,
        tile_overlap=8,
        axes_list=_TZYX,
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(config=cfg),
        denoise=DenoiseConfig(),
        content=ContentKnobs(
            k_star_ref=60000, n_features_ref=5000, cell=4, min_leaf=8, max_leaf=16
        ),
        merge=MergeConfig(),
    ).manifest

    assert manifest.mode == "content"
    assert manifest.n_tiles > 1  # a real multi-box plan, not a degenerate one
    assert manifest.grid_scale is None


def test_batch_plan_accepts_a_no_op_config_downscale(tmp_path: Path) -> None:
    """Non-vacuity control: the gate reads the VALUE, not the key's presence.

    `downscale: 1` (and its per-axis spelling) decimates nothing, so the
    planner's grid and the worker's agree even for the multi-tile plan that a
    decimating value is refused for — a blanket "any `downscale:` is fatal" gate
    would break a config that is a documented no-op.
    """
    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    for i, spelling in enumerate(("downscale: 1\n", "downscale: [1, 1, 1]\n")):
        cfg = tmp_path / f"noop-{i}.yaml"
        cfg.write_text(spelling)
        manifest = _plan(
            src, tmp_path / f"out-{i}", config=cfg, tile_size=12, tile_overlap=4
        ).manifest
        assert manifest.grid_scale is None
        assert manifest.n_tiles == 18  # the multi-tile plan, planned in full
        assert manifest.total_tasks == 4 * 18


def test_batch_plan_rejects_a_malformed_config_downscale(tmp_path: Path) -> None:
    """A malformed `downscale:` is a plan-time usage error in EVERY mode.

    The value is validated through `normalize_downscale` before the mode/tile
    exemptions are consulted, because nothing else on the batch path looks at
    this key any more: since #1624 the grid-scale resolver — which used to raise
    on it — is no longer passed it. Without the check here a `downscale: 0` or
    `2.5` would reach every task as a raw traceback at fit time (`2.5` is neither
    an int nor iterable, so it raises TypeError, not ValueError).

    Both exempt shapes are covered: a single-tile uniform plan and a content
    plan must reject it too, not wave it through on their exemption.

    The content half also pins WHEN: `validate_config_downscale` runs before the
    box plan is scanned, so the output directory must not exist afterwards.
    Measured against the first version of this fix, which validated alongside the
    decomposition: the "Plan: 6 boxes ... -> out-c/plan.json" line printed and the
    file existed before the refusal — after a max-projection over up to
    `--plan-samples` timepoints, tens of GB on a real timelapse.
    """
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    for i, spelling in enumerate(
        ("downscale: 0\n", "downscale: 2.5\n", "downscale: [2, 2]\n")
    ):
        cfg = tmp_path / f"bad-{i}.yaml"
        cfg.write_text(spelling)
        # Single-tile uniform (exempt from the refusal, not from validation).
        with pytest.raises(typer.BadParameter, match="invalid"):
            _plan(src, tmp_path / f"out-u-{i}", config=cfg)
        # Content (likewise exempt from the refusal only).
        with pytest.raises(typer.BadParameter, match="invalid"):
            plan_batch(
                input_path=src,
                output_dir=tmp_path / f"out-c-{i}",
                tiling="content",
                tile_size=None,
                tile_overlap=8,
                axes_list=_TZYX,
                array_key=None,
                timepoints_slice=None,
                channels_slice=None,
                fit=FitConfig(config=cfg),
                denoise=DenoiseConfig(),
                content=ContentKnobs(
                    k_star_ref=60000,
                    n_features_ref=5000,
                    cell=4,
                    min_leaf=8,
                    max_leaf=16,
                ),
                merge=MergeConfig(),
            )
        # Fail-fast: refused before the box plan was scanned or written.
        assert not (tmp_path / f"out-c-{i}").exists()


def test_run_and_submit_reject_a_config_downscale(tmp_path: Path) -> None:
    """Both front doors refuse it as a clean usage error, before any submission.

    `--dry-run` is the strongest place to pin this: the refusal has to happen
    during PLANNING (which dry-run does in full), not in the fitting/submission
    step that dry-run skips.

    `--tile-size 24 --overlap 8` on this 48^3 store is deliberately MULTI-tile
    (27 tiles; a `downscale: 2` worker sees 8): the refusal is scoped to that
    shape, so a single-tile `--tile-size 64` would now plan fine and this test
    would pass vacuously.
    """
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    cfg = tmp_path / "ds.yaml"
    cfg.write_text("downscale: 2\n")

    common = [
        "--axes",
        "z,y,x",
        "--tile-size",
        "24",
        "--overlap",
        "8",
        "--config",
        str(cfg),
        "--dry-run",
    ]
    invocations = [
        ["run", str(src), str(tmp_path / "o-run"), *common, "--gpus", "cpu"],
        ["submit", str(src), str(tmp_path / "o-sub"), *common, "-p", "gpu"],
    ]
    for argv in invocations:
        res = runner.invoke(app_gsplat, ["batch-fit", *argv])
        output = normalized_cli_output(res)
        assert res.exit_code != 0, output
        assert "downscale" in output, output
        assert "Traceback" not in output
        # Nothing was planned into existence, let alone submitted.
        assert not (Path(argv[2]) / "manifest.json").exists()


def test_floor_sample_pairs_are_bounded_and_deterministic() -> None:
    """The sampling rule: a few evenly spaced (t, c) slices over the FULL extent."""
    from luxar.cli.gsplat_ops.batch.planning import (
        FLOOR_SAMPLE_MAX_SLICES,
        _floor_sample_pairs,
    )

    # A long movie is sampled at a handful of evenly spaced timepoints, endpoints
    # included (never one arbitrary slice, never all 1000).
    assert _floor_sample_pairs(1000, 1) == [(0, 0), (333, 0), (666, 0), (999, 0)]
    assert _floor_sample_pairs(4, 1) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert _floor_sample_pairs(1, 1) == [(0, 0)]
    # Channel-like coordinates are covered too (a dim channel is the other way a
    # single global level erases data).
    assert _floor_sample_pairs(3, 2) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    ]
    for n_t, n_c in ((1000, 1), (3, 2), (50, 16), (7, 5), (1, 64), (4, 5), (200, 8)):
        pairs = _floor_sample_pairs(n_t, n_c)
        assert 1 <= len(pairs) <= FLOOR_SAMPLE_MAX_SLICES, (n_t, n_c, pairs)
        assert pairs == _floor_sample_pairs(n_t, n_c)  # pure function, no RNG
        assert all(0 <= t < n_t and 0 <= c < n_c for t, c in pairs)


def test_floor_sample_pairs_cover_both_axes_including_many_channels() -> None:
    """TIME is allocated FIRST: a many-channel store still spans the whole movie.

    FAILS against the previous version of this fix, which spent a shared 8-slice
    budget on channels first: a store with >= 5 channel-like coordinates then
    sampled exactly ONE (middle) timepoint, reinstating #1174's erase bug in the
    TIME direction. Measured on a real ``(T=4, C=5, 16, 24, 24)`` store with a
    pedestal of ``5 + 20t``, it sampled only ``[(1, 0) ... (1, 4)]`` -> level 25.0
    while ``max(t=0)`` is ~23, so every ``(t=0, c)`` task clipped to all zeros, fit
    0 splats, wrote ``.empty`` and was silently skipped by the merge while
    ``status`` reported success. A 4-camera x 2-channel store (n_c = 8) is in the
    same broken regime.
    """
    from luxar.cli.gsplat_ops.batch.planning import (
        FLOOR_SAMPLE_MAX_CHANNELS,
        FLOOR_SAMPLE_MAX_SLICES,
        FLOOR_SAMPLE_MAX_TIMEPOINTS,
        _floor_sample_pairs,
    )

    # The exact regime that was broken: 4 timepoints, 5 channel-like coordinates.
    pairs = _floor_sample_pairs(4, 5)
    assert sorted({t for t, _ in pairs}) == [0, 1, 2, 3]
    assert len(pairs) <= FLOOR_SAMPLE_MAX_SLICES

    for n_t, n_c in ((4, 5), (4, 8), (200, 8), (2, 64), (1000, 5), (3, 2), (7, 4)):
        pairs = _floor_sample_pairs(n_t, n_c)
        times = sorted({t for t, _ in pairs})
        chans = sorted({c for _, c in pairs})
        # Both ENDPOINTS on both axes whenever the axis is non-degenerate: for a
        # monotone pedestal drift the run's dimmest slice is at an end, which is
        # what makes the MINIMUM over samples a lower bound for the whole run.
        if n_t > 1:
            assert len(times) >= 2 and times[0] == 0 and times[-1] == n_t - 1
        if n_c > 1:
            assert len(chans) >= 2 and chans[0] == 0 and chans[-1] == n_c - 1
        assert len(times) <= FLOOR_SAMPLE_MAX_TIMEPOINTS
        assert len(chans) <= FLOOR_SAMPLE_MAX_CHANNELS
        # A full grid: every sampled timepoint is measured on every sampled channel.
        assert len(pairs) == len(times) * len(chans) <= FLOOR_SAMPLE_MAX_SLICES


def test_floor_axis_pins_decode_5d_and_flat_channel_indices() -> None:
    """The pins agree with the flat channel index each task's argv carries.

    The only nontrivial new algorithm here: a task is identified by ONE flat
    channel index, which has to be decoded back into a coordinate per
    channel-like axis (``camera`` x ``channel``) and combined with the time axis
    — by LABEL, at whatever positions the store puts them.
    """
    from luxar.cli.gsplat_ops.batch.planning import _floor_axis_pins
    from luxar.io.volume import decode_flat_channel_index

    # 5D TCZYX.
    assert _floor_axis_pins(
        ["t", "c", "z", "y", "x"], (3,), timepoint=7, channel=2
    ) == {0: 7, 1: 2}
    # 6D (time, camera, channel, z, y, x): flat index 4 of (2, 3) -> (1, 1).
    assert decode_flat_channel_index(4, (2, 3)) == (1, 1)
    assert _floor_axis_pins(
        ["time", "camera", "channel", "z", "y", "x"],
        (2, 3),
        timepoint=5,
        channel=4,
    ) == {0: 5, 1: 1, 2: 1}
    # Channel-first, time in the middle: position-independent.
    assert _floor_axis_pins(
        ["c", "z", "t", "y", "x"], (2,), timepoint=1, channel=1
    ) == {0: 1, 2: 1}
    # A label set that does not describe the store is a ValueError (the caller
    # falls back to an eager load).
    with pytest.raises(ValueError):
        _floor_axis_pins(["t", "z", "y", "x"], (2,), timepoint=0, channel=0)


def test_pinned_slice_reads_the_same_slice_the_task_argv_will(tmp_path: Path) -> None:
    """A 6D (t, camera, channel, z, y, x) store: pins == the task's own decode.

    FAILS pre-fix: nothing decoded the flat channel index at plan time at all
    (the floor was never resolved there), and the positional heuristic on a 6D
    store folds the leading axes itself — so this pins the one place where the
    planner's read and the worker's read must agree.
    """
    from luxar.cli.gsplat_ops.batch.planning import _materialize, _pinned_slice_volume
    from luxar.io.volume import decode_flat_channel_index

    src = tmp_path / "multi.zarr"
    rng = np.random.default_rng(3)
    full = rng.random((3, 2, 2, 6, 8, 8)).astype(np.float32)
    labels = ["time", "camera", "channel", "z", "y", "x"]
    _write_timelapse(src, full, axes_attr=labels)

    flat_channel = 3  # (camera, channel) = (1, 1)
    assert decode_flat_channel_index(flat_channel, (2, 2)) == (1, 1)
    view = _pinned_slice_volume(
        src,
        channel=flat_channel,
        timepoint=2,
        array_key=None,
        axes=None,
        axes_labels=labels,
        channel_shape=(2, 2),
        spatial_shape=(6, 8, 8),
    )
    assert not isinstance(view, np.ndarray)  # still lazy
    np.testing.assert_allclose(_materialize(view), full[2, 1, 1])


def test_pinned_slice_falls_back_eagerly_but_keeps_the_timepoint(
    tmp_path: Path, capsys
) -> None:
    """When a lazy pinned view can't be built, the eager fallback still slices right.

    The fallback exists for stores whose discovered labels do not map onto the
    array the way planning assumed (NGFF classifies by axis *type*, not name). It
    must still read the timepoint it was asked for: it forwards the discovered
    labels to ``load_volume`` rather than dropping to the positional heuristic,
    which on a 4D store would read ``arr[0]``.
    """
    from luxar.cli.gsplat_ops.batch.planning import _pinned_slice_volume

    src = tmp_path / "movie.zarr"
    full = _make_timelapse_zarr(src, axes_attr=["time", "z", "y", "x"])
    # A channel_shape that the labels do not describe -> _floor_axis_pins raises.
    view = _pinned_slice_volume(
        src,
        channel=0,
        timepoint=3,
        array_key=None,
        axes=None,
        axes_labels=["t", "z", "y", "x"],
        channel_shape=(2,),
        spatial_shape=(16, 24, 24),
    )
    assert isinstance(view, np.ndarray)  # eager
    np.testing.assert_allclose(view, full[3])
    assert "eagerly" in capsys.readouterr().out


def test_content_plan_scan_sees_a_later_timepoint_without_axes(
    tmp_path: Path, monkeypatch
) -> None:
    """--plan-samples must actually reach later timepoints (#1174, same class).

    FAILS pre-fix: the scan called ``load_volume(..., channel=rep_c,
    timepoint=t, axes=None)``, and on a 4D store whose DISCOVERED axes are
    ``[time, z, y, x]`` the positional heuristic takes the ``channel is not
    None`` -> "Slicing 4D (CZYX)" branch and reads ``arr[0]`` for EVERY sampled
    timepoint. The box plan was then built from one timepoint, leaving holes
    wherever content moved — while the tasks DID read the right timepoint.
    """
    from luxar.gsplats import planner as planner_mod
    from luxar.gsplats.planner import FitPlan, PlanBox

    src = tmp_path / "movie.zarr"
    blobs = _blob_field()
    frames = np.stack([blobs + 5.0 for _ in range(4)]).astype(np.float32)
    frames[3, 8, 12, 12] = 500.0  # content ONLY at the last timepoint
    _write_timelapse(src, frames, axes_attr=["time", "z", "y", "x"])

    scanned: list = []

    def _fake_plan_volume(volume, density, **kwargs):
        scanned.append(np.asarray(volume))
        z, y, x = volume.shape
        return FitPlan(
            volume_shape=[z, y, x],
            boxes=[PlanBox(box=[0, z, 0, y, 0, x], n_features=10, budget=50)],
            overlap=int(kwargs.get("overlap", 8)),
            feature_method="peaks",
            min_leaf=8,
            max_leaf=32,
        )

    monkeypatch.setattr(planner_mod, "plan_volume", _fake_plan_volume)

    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    plan_batch(
        input_path=src,
        output_dir=tmp_path / "out",
        tiling="content",
        tile_size=None,
        tile_overlap=8,
        axes_list=None,  # NO --axes: the discovered labels must carry the day
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(floor="none"),  # keep the scan un-subtracted for the assert
        denoise=DenoiseConfig(),
        content=ContentKnobs(k_star_ref=4000, n_features_ref=200),
        merge=MergeConfig(),
    )

    assert len(scanned) == 1
    assert float(scanned[0].max()) == pytest.approx(500.0, rel=1e-5)


def test_uniform_tile_worker_does_not_re_guard_a_concrete_level(
    tmp_path: Path, monkeypatch
) -> None:
    """The batch task must apply the plan's number even on a dim timepoint.

    FAILS pre-fix (this fix's own hazard): ``fit_single_tile`` — the path every
    uniform batch ``(t, c)`` task runs with ``--floor <level>`` — resolved with
    ``guard_numeric=True``, so a timepoint whose sampled max is below the level
    got ``floor=None`` (hard-min normalization) while every sibling subtracted the
    level: exactly the per-timepoint pedestal difference #1174 removes.
    """
    from types import SimpleNamespace

    from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile
    from luxar.gsplats import fit_tiled_gsplats

    seen: dict = {}

    def _fake_fit_tile(volume, spec, **kwargs):
        seen["floor"] = kwargs.get("floor")
        return "fitted"

    monkeypatch.setattr(fit_tiled_gsplats, "fit_tile", _fake_fit_tile)
    ctx = SimpleNamespace(
        tile="0/1",
        tile_size=64,
        tile_overlap=8,
        progressive=False,
        max_splats_per_pass=None,
        psnr_patience=None,
        max_passes=None,
    )
    dim_volume = _blob_field() * 0.1 + 1.0  # max ~2, far below the level below

    # The concrete level the parent resolved and already guarded, as it arrives
    # from argv: a STRING that happens to be a number.
    fit_single_tile(ctx, dim_volume, {"floor": "45.0"}, None)
    assert seen["floor"] == pytest.approx(45.0)

    # A volume-derived SPEC is still resolved (and guarded) here, since it
    # becomes a level for the first time: the worker must end up with the level
    # `resolve_volume_floor` produces for THIS volume, never the string 'auto'.
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

    expected = resolve_volume_floor(dim_volume, "auto", guard_numeric=True)
    fit_single_tile(ctx, dim_volume, {"floor": "auto"}, None)
    if expected is None:  # guard refused / nothing to subtract
        assert seen["floor"] == "none"
    else:
        assert seen["floor"] == pytest.approx(expected)


def test_uniform_tile_worker_warns_loudly_when_the_level_erases_its_volume(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A user numeric >= this worker's own max is APPLIED, but announced loudly.

    Behaviour change vs main on `fit --tile k/M --floor 110` and `fit -j N
    --floor 110` (whose `-j` parent forwards the spec verbatim): main guarded the
    numeric here and reported-and-IGNORED it, so the fit went ahead; the level is
    now applied unvetoed, so the tile windows to zero and yields an ``.empty``
    marker. FAILS against the current tree, which strips the guard silently — with
    nothing downstream of ``.empty`` mentioning the floor, the user gets "writer
    rejects empty stores" and no clue why.
    """
    from types import SimpleNamespace

    from luxar.cli.gsplat_ops.fitting.fit_utils import fit_single_tile
    from luxar.gsplats import fit_tiled_gsplats

    seen: dict = {}

    def _fake_fit_tile(volume, spec, **kwargs):
        seen["floor"] = kwargs.get("floor")
        return "fitted"

    monkeypatch.setattr(fit_tiled_gsplats, "fit_tile", _fake_fit_tile)
    ctx = SimpleNamespace(
        tile="0/1",
        tile_size=64,
        tile_overlap=8,
        progressive=False,
        max_splats_per_pass=None,
        psnr_patience=None,
        max_passes=None,
        timepoint=7,
        channel=1,
    )
    dim_volume = _blob_field() * 0.1 + 1.0  # max ~2, far below 110

    fit_single_tile(ctx, dim_volume, {"floor": "110"}, None)

    # Still APPLIED: no worker may disagree with its siblings about the pedestal.
    assert seen["floor"] == pytest.approx(110.0)
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "110" in out  # the level
    assert "t=7, c=1" in out  # which sub-volume
    assert "0 SPLATS" in out and "skipped by the merge" in out  # the consequence

    # A level comfortably below the volume's max says nothing.
    fit_single_tile(ctx, dim_volume, {"floor": "0.5"}, None)
    assert seen["floor"] == pytest.approx(0.5)
    assert "⚠" not in capsys.readouterr().out


def test_submit_sbatch_carries_the_numeric_level(tmp_path: Path) -> None:
    """`batch-fit submit`'s generated task command carries the number, not the spec."""
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    src = tmp_path / "movie.zarr"
    _make_timelapse_zarr(src)
    manifest = _plan(src, tmp_path / "out", floor="auto").manifest
    assert manifest.floor_level is not None

    script = generate_fit_sbatch(manifest, env_preamble="")
    assert f"--floor {manifest.floor_level}" in script
    assert "--floor auto" not in script


def test_resolve_tiling_small_volume_fits_whole() -> None:
    """Companion fix: a small/medium stack no longer tiles (avoids seams)."""
    from luxar.cli.gsplat_ops.fitting.fit_utils import resolve_tiling as _resolve_tiling

    # Neuromast-shaped stack (dims > 256 but only ~28M voxels) → whole-volume.
    assert _resolve_tiling("auto", (84, 580, 576), 256, False) == "none"


def test_resolve_tiling_large_volume_tiles() -> None:
    from luxar.cli.gsplat_ops.fitting.fit_utils import resolve_tiling as _resolve_tiling

    # Genuinely large gigavoxel volume still tiles.
    assert _resolve_tiling("auto", (1024, 1024, 1024), 256, False) == "uniform"
    assert _resolve_tiling("auto", (1024, 1024, 1024), 256, True) == "content"
