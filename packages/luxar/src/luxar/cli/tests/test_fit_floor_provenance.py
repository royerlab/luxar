"""``fit`` floor provenance: consuming a calibration's floor, recording a
content fit's floor (issue #1175).

Two write-only records used to meet here. ``gsplat cal`` stamped the level it
subtracted into ``fit_config.floor_subtracted`` and nothing read it, so a
``fit --cal`` re-derived its own floor and fitted on a different intensity
scale than the density's ``feature_threshold`` was calibrated on. And a
``--tiling content`` fit saved no record of the one level every box removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer

from luxar.cli.gsplat_ops.fitting.fit_utils import resolve_floor_with_calibration
from luxar.gsplats.calibration import CalibrationResult, HeldOutPeak, NoiseFloor
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import tril_size


def _write_cal(path: Path, fit_config: dict) -> Path:
    """A minimal but real ``cal.json`` carrying ``fit_config``."""
    CalibrationResult(
        k_values_requested=[1000, 4000],
        k_values_effective=[1000, 4000],
        held_out_psnr_db=[30.0, 32.0],
        train_psnr_db=[31.0, 33.0],
        held_out_mse=[1e-4, 1e-4],
        full_psnr_db=[30.0, 32.0],
        full_ssim=[0.8, 0.9],
        held_out_peak=HeldOutPeak(k_star=4000, type="peak", confidence_db=1.0),
        noise_floor=NoiseFloor(
            sigma_hat=0.01,
            sigma_laplacian=0.01,
            sigma_haar=0.01,
            sigma_background=0.01,
            psnr_max_db=45.0,
        ),
        fit_times_seconds=[1.0, 2.0],
        splat_paths=None,
        mask_seed=0,
        mask_fraction=0.05,
        donut_radius=2,
        fit_config=fit_config,
        volume_shape=[32, 32, 32],
        volume_dtype="float32",
        timestamp="2026-01-01T00:00:00",
    ).to_json(path)
    return path


class TestFloorFromCalibration:
    def test_recorded_level_is_adopted_when_floor_is_unset(
        self, tmp_path: Path, capsys
    ) -> None:
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})
        assert (
            resolve_floor_with_calibration(cal, None, None, tiling="content") == "110.0"
        )
        # Never silent: the fit says where its floor came from.
        assert "Floor from calibration" in capsys.readouterr().out

    def test_explicit_floor_wins_even_when_it_is_auto(self, tmp_path: Path) -> None:
        """``--floor`` defaults to None, not "auto" — so an explicit ``auto`` is
        a user decision and must not be silently replaced by the cal's level."""
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})
        assert (
            resolve_floor_with_calibration(cal, "auto", None, tiling="content")
            == "auto"
        )
        assert (
            resolve_floor_with_calibration(cal, "p20", None, tiling="content") == "p20"
        )

    def test_recorded_null_is_not_adopted_as_none(self, tmp_path: Path) -> None:
        """``cal`` writes ``null`` both when the user asked for ``none`` AND
        when its own too-high guard REFUSED the floor — indistinguishable in the
        file. Adopting it would silently disable the ``auto`` default on the
        strength of a guard firing, so say nothing instead."""
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": None})
        assert resolve_floor_with_calibration(cal, None, None, tiling="content") is None

    def test_a_non_numeric_recorded_level_does_not_traceback(
        self, tmp_path: Path
    ) -> None:
        """A hand-edited ``floor_subtracted: "auto"`` must fall through quietly,
        not raise a bare ValueError out of the CLI."""
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": "auto"})
        assert resolve_floor_with_calibration(cal, None, None, tiling="content") is None

    def test_a_non_finite_recorded_level_is_ignored(self, tmp_path: Path) -> None:
        """``json`` round-trips ``NaN``/``Infinity``, but ``--floor`` refuses
        them — forwarding one would abort the fit with an error naming a flag
        the user never passed."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": bad})
            assert (
                resolve_floor_with_calibration(cal, None, None, tiling="content")
                is None
            )

    def test_absent_key_leaves_the_caller_alone(self, tmp_path: Path) -> None:
        cal = _write_cal(tmp_path / "cal.json", {})
        assert (
            resolve_floor_with_calibration(cal, None, None, tiling="content") is None
        )  # unset stays unset

    def test_no_cal_is_a_no_op(self) -> None:
        assert (
            resolve_floor_with_calibration(None, None, None, tiling="content") is None
        )

    def test_only_content_tiling_adopts_it(self, tmp_path: Path) -> None:
        """Every other mode announces --cal as IGNORED two lines later."""
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})
        for tiling in ("none", "uniform"):
            assert (
                resolve_floor_with_calibration(cal, None, None, tiling=tiling) is None
            )
        assert (
            resolve_floor_with_calibration(cal, None, None, tiling="content") == "110.0"
        )

    def test_a_yaml_config_floor_wins(self, tmp_path: Path) -> None:
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})
        config = tmp_path / "fit.yaml"
        config.write_text('floor: "p20"\nn_iters: 100\n')
        # Left unset, so `load_fit_config` layers the YAML floor as it always did.
        assert (
            resolve_floor_with_calibration(cal, None, config, tiling="content") is None
        )

    def test_a_yaml_config_without_a_floor_does_not_block(self, tmp_path: Path) -> None:
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})
        config = tmp_path / "fit.yaml"
        config.write_text("n_iters: 100\n")
        assert (
            resolve_floor_with_calibration(cal, None, config, tiling="content")
            == "110.0"
        )

    def test_a_negative_recorded_level_is_declined_not_forwarded(
        self, tmp_path: Path, capsys
    ) -> None:
        """``--floor`` rejects a negative level, so forwarding one would abort
        the fit; decline it out loud instead."""
        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": -3.0})
        assert resolve_floor_with_calibration(cal, None, None, tiling="content") is None
        assert "negative floor" in capsys.readouterr().out

    def test_the_adopted_spec_is_a_valid_floor(self, tmp_path: Path) -> None:
        """Asserted through the PUBLIC wrapper the CLI itself calls, not the
        private validator behind it."""
        from luxar.cli.gsplat_ops.fitting.fit_utils import validate_floor_spec

        cal = _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.5})
        validate_floor_spec(
            resolve_floor_with_calibration(cal, None, None, tiling="content")
        )


def _stub_leaf(ndim: int = 3, n: int = 4) -> GSplatData:
    chol = np.zeros((n, tril_size(ndim)), dtype=np.float32)
    k = 0
    for i in range(ndim):
        for j in range(i + 1):
            if i == j:
                chol[:, k] = 1.0
            k += 1
    return GSplatData(
        centers=np.ones((n, ndim), dtype=np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
        stats={"time_seconds": 0.0},
    )


@pytest.mark.parametrize("flat", [False, True])
def test_content_fit_records_the_level_every_box_subtracted(
    monkeypatch, tmp_path: Path, flat: bool
) -> None:
    """A ``--tiling content`` result — partition or ``--flat`` leaf — persists
    ``pipeline/floor``. It used to persist nothing at all."""
    pytest.importorskip("torch")
    import luxar.gsplats.planner as gplanner
    from luxar.cli.gsplat_ops import planner
    from luxar.gsplats.io.load_gsplats import load_gsplat_node, load_gsplats

    def _fake_fit_planned(vol: Any, fitplan: Any, **kwargs: Any) -> Any:
        regions = [_stub_leaf(), _stub_leaf()]
        if kwargs.get("partition", True):
            result = GSplatData.partition_from_regions(regions)
            result.meta["fit_stats"] = {
                "planned_fit": True,
                "n_splats": result.n_splats,
                "psnr_db": 33.3,
            }
            return result
        result = GSplatData.concatenate(regions)
        result.stats.update({"planned_fit": True, "psnr_db": 33.3})
        return result

    # `run_content_fit` re-imports `fit_planned` from its source module on every
    # call, so the stub has to replace it THERE.
    monkeypatch.setattr(gplanner, "fit_planned", _fake_fit_planned)

    rng = np.random.RandomState(5)
    volume = rng.normal(100.0, 1.0, size=(48, 48, 48)).astype(np.float32)
    volume[16:32, 16:32, 16:32] += 200.0

    output = tmp_path / f"content_{flat}.gsplats.zarr"
    planner.run_content_fit(
        tmp_path / "unused.npy",
        output,
        volume=volume,
        k_star_ref=1000,
        n_features_ref=100,
        floor="100.0",
        flat=flat,
        min_leaf=16,
        max_leaf=32,
        cell=8,
        overlap=4,
        verbose=False,
    )

    if flat:
        stats = load_gsplats(output, include_stats=True).stats
    else:
        _, stats = load_gsplat_node(output, include_stats=True)
    assert stats["floor"] == pytest.approx(100.0)
    assert stats["psnr_db"] == pytest.approx(33.3)


def test_parallel_content_fit_hands_the_loaded_volume_to_the_merge(
    monkeypatch, tmp_path: Path
) -> None:
    """The parent still owns the reference after its box workers finish.

    A builder-level test cannot catch this handoff: the merged score happens in
    the parent process, after every worker store has been reloaded.
    """
    import importlib

    from luxar.cli.gsplat_ops import planner
    from luxar.gsplats.planner import FitPlan, PlanBox

    parallel_module = importlib.import_module(
        "luxar.gsplats.planner.fit_planned_parallel"
    )

    volume = np.ones((8, 8, 16), np.float32)
    plan = FitPlan(
        volume_shape=list(volume.shape),
        boxes=[
            PlanBox(box=[0, 8, 0, 8, 0, 8], n_features=10, budget=10),
            PlanBox(box=[0, 8, 0, 8, 8, 16], n_features=10, budget=10),
        ],
        overlap=1,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=8,
        density={"saturation_cap": 100},
    )
    plan_path = tmp_path / "plan.json"
    plan.to_json(plan_path)
    seen: dict[str, Any] = {}

    def _fake_parallel(fitplan: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return _stub_leaf()

    monkeypatch.setattr(parallel_module, "fit_planned_parallel", _fake_parallel)
    monkeypatch.setattr(planner, "_save_fit_result", lambda *args, **kwargs: None)

    planner.run_content_fit(
        tmp_path / "unused.npy",
        tmp_path / "out.gsplats.zarr",
        volume=volume,
        plan=plan_path,
        jobs="2",
        k_star_ref=100,
        n_features_ref=10,
        floor="none",
        device="cpu",
        flat=True,
        verbose=False,
    )

    assert seen["volume"] is volume
    assert seen["device"] == "cpu"


def test_parallel_content_fit_rejects_an_external_plan_for_another_grid(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    from luxar.cli.gsplat_ops import planner
    from luxar.gsplats.planner import FitPlan, PlanBox

    parallel_module = importlib.import_module(
        "luxar.gsplats.planner.fit_planned_parallel"
    )
    plan = FitPlan(
        volume_shape=[8, 8, 8],
        boxes=[PlanBox(box=[0, 8, 0, 8, 0, 8], n_features=10, budget=10)],
        overlap=1,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=8,
        density={"saturation_cap": 100},
    )
    plan_path = tmp_path / "plan.json"
    plan.to_json(plan_path)
    called = False

    def _fake_parallel(fitplan: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _stub_leaf()

    def _unexpected_floor_scan(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("grid mismatch must be rejected before floor preprocessing")

    monkeypatch.setattr(parallel_module, "fit_planned_parallel", _fake_parallel)
    monkeypatch.setattr(planner, "resolve_shared_floor", _unexpected_floor_scan)

    with pytest.raises(typer.BadParameter, match="does not match the plan grid"):
        planner.run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "out.gsplats.zarr",
            volume=np.zeros((16, 16, 16), np.float32),
            plan=plan_path,
            jobs="2",
            k_star_ref=100,
            n_features_ref=10,
            floor="none",
            flat=True,
            verbose=False,
        )

    assert called is False


def test_content_plan_box_rejects_a_volume_from_another_grid(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    from luxar.cli.gsplat_ops import planner
    from luxar.gsplats.planner import FitPlan, PlanBox

    plan = FitPlan(
        volume_shape=[8, 8, 8],
        boxes=[PlanBox(box=[0, 8, 0, 8, 0, 8], n_features=10, budget=10)],
        overlap=1,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=8,
        density={"saturation_cap": 100},
    )
    plan_path = tmp_path / "plan.json"
    plan.to_json(plan_path)
    fit_planned_module = importlib.import_module("luxar.gsplats.planner.fit_planned")
    called = False

    def _fake_box(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _stub_leaf()

    monkeypatch.setattr(fit_planned_module, "_fit_one_box", _fake_box)

    with pytest.raises(typer.BadParameter, match="does not match the plan grid"):
        planner.run_content_fit(
            tmp_path / "unused.npy",
            tmp_path / "box.gsplats.zarr",
            volume=np.zeros((16, 16, 16), np.float32),
            plan=plan_path,
            plan_box=0,
            k_star_ref=100,
            n_features_ref=10,
            floor="none",
            verbose=False,
        )

    assert called is False


def test_the_content_stamp_never_contradicts_the_boxes(tmp_path: Path) -> None:
    """A level the boxes already recorded wins over the planned one:
    ``_stamp_content_floor`` must not overwrite a box-recorded ``floor`` with a
    different asked level, which would ship a store whose ``floor`` and
    ``image_min`` contradict each other."""
    from luxar.cli.gsplat_ops.planner import _stamp_content_floor
    from luxar.gsplats.planner.fit_planned import _stamp_planned_normalization

    leaf = _stub_leaf()
    leaf.stats.update({"floor": 100.0, "image_min": 100.0, "intensity_range": 150.0})
    _stamp_content_floor(leaf, 5.0, 5.0)

    assert leaf.stats["floor"] == pytest.approx(100.0)
    assert leaf.stats["image_min"] == pytest.approx(100.0)

    regions = [_stub_leaf(), _stub_leaf()]
    for region in regions:
        region.stats.update(
            {
                "floor": 100.0,
                "image_min": 100.0,
                "image_max": 250.0,
                "intensity_range": 150.0,
            }
        )
    node = GSplatData.partition_from_regions(regions)
    _stamp_planned_normalization(node.meta, regions)
    node.meta["fit_stats"] = {"planned_fit": True, "psnr_db": 33.3}
    _stamp_content_floor(node, 5.0, 5.0)

    assert node.meta["floor"] == pytest.approx(100.0)
    assert node.meta["image_min"] == pytest.approx(100.0)
    assert "floor" not in node.meta["fit_stats"]


def test_info_lists_the_normalization_block_not_just_the_generic_dump(
    tmp_path: Path,
) -> None:
    """Amplitudes are background-RELATIVE, so ``floor`` belongs in the headline
    metadata, above the "Additional Metadata" catch-all."""
    from typer.testing import CliRunner

    from luxar.cli import app

    leaf = _stub_leaf()
    leaf.stats.update(
        {
            "floor": 110.0,
            "image_min": 110.0,
            "image_max": 4095.0,
            "intensity_range": 3985.0,
        }
    )
    out = tmp_path / "floored.gsplats.zarr"
    leaf.save(out)

    result = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert result.exit_code == 0, result.output
    head = result.output.split("Additional Metadata:")[0]
    for key in ("floor", "image_min", "image_max", "intensity_range"):
        assert f"{key}:" in head, result.output


def test_info_lists_the_block_for_a_partition_too(tmp_path: Path) -> None:
    """A ``kind=partition`` is the tiled CLI's DEFAULT output, and ``info``
    reports it through a different branch — one with no ``stats`` dict and no
    "Additional Metadata" dump, so the block was simply absent there."""
    from typer.testing import CliRunner

    from luxar.cli import app
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    node = GSplatData.partition_from_regions([_stub_leaf(), _stub_leaf()])
    node.meta.update(
        {
            "floor": 110.0,
            "image_min": 110.0,
            "image_max": 4095.0,
            "intensity_range": 3985.0,
        }
    )
    out = tmp_path / "part.gsplats.zarr"
    write_gsplats_tree(out, node)

    result = CliRunner().invoke(app, ["gsplat", "info", str(out), "--no-histograms"])
    assert result.exit_code == 0, result.output
    assert "Root kind: partition" in result.output
    for key in ("floor", "image_min", "image_max", "intensity_range"):
        assert f"{key}:" in result.output, result.output


# ────────────────────────────────────────────────────────────────────────
# Wiring: `run_fit_volume` really consumes the resolver, and only where
# `--cal` is honoured
# ────────────────────────────────────────────────────────────────────────


def _capture_ctx_floor(monkeypatch) -> dict:
    """Intercept the pipeline ctx right after ``run_fit_volume`` builds it.

    ``warn_ignored_density_flags`` is called once, immediately after
    construction, on every tiling branch — so it is the one seam that reports
    the resolved ``--floor`` without running a fit.
    """
    import typer

    from luxar.cli.gsplat_ops.fitting import fit as fit_mod

    seen: dict = {}

    def _capture(ctx: Any) -> None:
        seen["floor"] = ctx.floor
        seen["tiling"] = ctx.resolved_tiling
        raise typer.Exit(0)

    monkeypatch.setattr(fit_mod, "warn_ignored_density_flags", _capture)
    return seen


def _fit_inputs(tmp_path: Path) -> "tuple[Path, Path]":
    volume = np.zeros((24, 24, 24), np.float32)
    volume[8:16, 8:16, 8:16] = 1.0
    vol_path = tmp_path / "vol.npy"
    np.save(vol_path, volume)
    return vol_path, _write_cal(tmp_path / "cal.json", {"floor_subtracted": 110.0})


def test_a_content_fit_really_receives_the_calibration_floor(
    monkeypatch, tmp_path: Path
) -> None:
    """Not just the resolver in isolation: the level has to reach the ctx the
    content fit is driven from."""
    from typer.testing import CliRunner

    from luxar.cli import app

    seen = _capture_ctx_floor(monkeypatch)
    vol_path, cal = _fit_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol_path),
            str(tmp_path / "out.gsplats.zarr"),
            "--tiling",
            "content",
            "--cal",
            str(cal),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"floor": "110.0", "tiling": "content"}


def test_a_uniform_fit_does_not_adopt_the_calibration_floor(
    monkeypatch, tmp_path: Path
) -> None:
    """``--cal`` is announced as IGNORED outside ``--tiling content``; adopting
    its floor there would make the CLI contradict its own message."""
    from typer.testing import CliRunner

    from luxar.cli import app

    seen = _capture_ctx_floor(monkeypatch)
    vol_path, cal = _fit_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol_path),
            str(tmp_path / "out.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--cal",
            str(cal),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"floor": None, "tiling": "uniform"}
