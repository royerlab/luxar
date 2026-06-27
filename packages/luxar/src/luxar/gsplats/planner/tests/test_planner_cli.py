"""End-to-end CLI tests for `luxar gsplat plan` (H7)."""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat

runner = CliRunner()


def _vol(tmp_path, shape=(48, 48, 48), n=10):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    V = np.zeros(shape, np.float32)
    rng = np.random.default_rng(0)
    for _ in range(n):
        cz, cy, cx = rng.integers(6, min(shape) - 6, 3)
        V += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)).astype(
            np.float32
        )
    V = np.clip(V, 0, 1)
    p = tmp_path / "vol.npy"
    np.save(p, V)
    return p


_DENSITY_FLAGS = [
    "--k-star-ref",
    "4000",
    "--n-features-ref",
    "200",
    "--feature-threshold",
    "0.1",
    "--feature-metric",
    "peaks",
    "--cell",
    "8",
    "--min-leaf",
    "16",
    "--max-leaf",
    "32",
]


def test_plan_writes_json(tmp_path):
    vol = _vol(tmp_path)
    out = tmp_path / "plan.json"
    res = runner.invoke(app_gsplat, ["plan", str(vol), str(out), *_DENSITY_FLAGS])
    assert res.exit_code == 0, res.output
    from luxar.gsplats.planner import FitPlan

    plan = FitPlan.from_json(out)
    assert plan.n_boxes >= 1 and plan.total_budget > 0


def test_plan_fit_end_to_end(tmp_path):
    vol = _vol(tmp_path)
    out_json = tmp_path / "plan.json"
    out_z = tmp_path / "planned.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "plan",
            str(vol),
            str(out_json),
            *_DENSITY_FLAGS,
            "--fit",
            "-o",
            str(out_z),
            "--preset",
            "draft",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_z.exists()
    from luxar.gsplats.io import load_gsplats

    assert load_gsplats(out_z).n_splats > 0


def test_missing_density_flags_errors(tmp_path):
    vol = _vol(tmp_path)
    res = runner.invoke(app_gsplat, ["plan", str(vol), str(tmp_path / "p.json")])
    assert res.exit_code != 0  # neither --cal nor --k-star-ref/--n-features-ref


def test_fit_without_output_errors(tmp_path):
    vol = _vol(tmp_path)
    res = runner.invoke(
        app_gsplat,
        ["plan", str(vol), str(tmp_path / "p.json"), *_DENSITY_FLAGS, "--fit"],
    )
    assert res.exit_code != 0  # --fit requires -o


def test_fit_box_worker_mode(tmp_path):
    # The hidden --fit-box worker: write a plan, then fit just box 0 from it.
    vol = _vol(tmp_path)
    plan_json = tmp_path / "plan.json"
    res = runner.invoke(app_gsplat, ["plan", str(vol), str(plan_json), *_DENSITY_FLAGS])
    assert res.exit_code == 0, res.output

    out = tmp_path / "box0.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "plan",
            str(vol),
            str(plan_json),
            "--fit-box",
            "0",
            "-o",
            str(out),
            "--preset",
            "draft",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output
    # box 0 either produced a store or a 0-splat .empty marker — never both, never neither
    produced = out.exists()
    empty = (tmp_path / "box0.gsplats.zarr.empty").exists()
    assert produced ^ empty
    if produced:
        from luxar.gsplats.io import load_gsplats

        assert load_gsplats(out).n_splats > 0


def test_fit_box_out_of_range_errors(tmp_path):
    vol = _vol(tmp_path)
    plan_json = tmp_path / "plan.json"
    assert (
        runner.invoke(
            app_gsplat, ["plan", str(vol), str(plan_json), *_DENSITY_FLAGS]
        ).exit_code
        == 0
    )
    res = runner.invoke(
        app_gsplat,
        [
            "plan",
            str(vol),
            str(plan_json),
            "--fit-box",
            "9999",
            "-o",
            str(tmp_path / "x.gsplats.zarr"),
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code != 0  # box index out of range


def test_plan_fit_parallel_jobs(tmp_path):
    # End-to-end concurrent path: -j 2 spawns per-box workers, merges to one file.
    vol = _vol(tmp_path)
    out = tmp_path / "par.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "plan",
            str(vol),
            str(tmp_path / "plan.json"),
            *_DENSITY_FLAGS,
            "--fit",
            "-o",
            str(out),
            "-j",
            "2",
            "--preset",
            "draft",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    from luxar.gsplats.io import load_gsplats

    assert load_gsplats(out).n_splats > 0


def test_bad_jobs_value_errors(tmp_path):
    vol = _vol(tmp_path)
    res = runner.invoke(
        app_gsplat,
        [
            "plan",
            str(vol),
            str(tmp_path / "plan.json"),
            *_DENSITY_FLAGS,
            "--fit",
            "-o",
            str(tmp_path / "o.gsplats.zarr"),
            "-j",
            "notanint",
            "--preset",
            "draft",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code != 0  # --jobs must be int or 'auto'
