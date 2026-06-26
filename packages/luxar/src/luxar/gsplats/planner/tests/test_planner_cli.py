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
        V += np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
        ).astype(np.float32)
    V = np.clip(V, 0, 1)
    p = tmp_path / "vol.npy"
    np.save(p, V)
    return p


_DENSITY_FLAGS = [
    "--k-star-ref", "4000", "--n-features-ref", "200",
    "--feature-threshold", "0.1", "--feature-metric", "peaks",
    "--cell", "8", "--min-leaf", "16", "--max-leaf", "32",
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
        ["plan", str(vol), str(out_json), *_DENSITY_FLAGS,
         "--fit", "-o", str(out_z), "--preset", "draft", "--device", "cpu"],
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
