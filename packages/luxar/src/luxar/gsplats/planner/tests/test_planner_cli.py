"""End-to-end CLI tests for content-aware fitting (`gsplat fit --tiling content`).

The standalone `gsplat plan` command was folded into `fit --tiling content`
(+ `--plan-only` / `--plan` / hidden `--plan-box`); these exercise that surface.
"""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat
from luxar.gsplats.io.load_gsplats import load_gsplat_node

runner = CliRunner()


def _vol(tmp_path, shape=(48, 48, 48), n=12):
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


# content-tiling density + small leaf bounds so a 48^3 volume yields >=2 boxes
_CONTENT = [
    "--tiling",
    "content",
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
_CPU = ["--preset", "draft", "--device", "cpu"]


def _kind(path):
    node, _ = load_gsplat_node(path)
    return type(node).__name__


def test_plan_only_writes_json(tmp_path):
    vol = _vol(tmp_path)
    out = tmp_path / "plan.json"
    res = runner.invoke(
        app_gsplat, ["fit", str(vol), str(out), *_CONTENT, "--plan-only"]
    )
    assert res.exit_code == 0, res.output
    from luxar.gsplats.planner import FitPlan

    plan = FitPlan.from_json(out)
    assert plan.n_boxes >= 1 and plan.total_budget > 0


def test_content_fit_partition_by_default(tmp_path):
    vol = _vol(tmp_path)
    out = tmp_path / "c.gsplats.zarr"
    res = runner.invoke(app_gsplat, ["fit", str(vol), str(out), *_CONTENT, *_CPU])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert _kind(out) == "GSplatPartition"  # one part per content box


def test_content_fit_flat_is_leaf(tmp_path):
    vol = _vol(tmp_path)
    out = tmp_path / "cf.gsplats.zarr"
    res = runner.invoke(
        app_gsplat, ["fit", str(vol), str(out), *_CONTENT, *_CPU, "--flat"]
    )
    assert res.exit_code == 0, res.output
    assert _kind(out) == "GSplatLeaf"  # --flat collapses to a single leaf


def test_content_missing_density_errors(tmp_path):
    vol = _vol(tmp_path)
    res = runner.invoke(
        app_gsplat,
        ["fit", str(vol), str(tmp_path / "o.gsplats.zarr"), "--tiling", "content"],
    )
    assert res.exit_code != 0  # no --cal / --k-star-ref


def test_plan_box_worker_mode(tmp_path):
    # write a plan, then fit just box 0 from it (the -j parallel worker entry).
    vol = _vol(tmp_path)
    plan_json = tmp_path / "plan.json"
    assert (
        runner.invoke(
            app_gsplat, ["fit", str(vol), str(plan_json), *_CONTENT, "--plan-only"]
        ).exit_code
        == 0
    )
    out = tmp_path / "box0.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "content",
            "--plan",
            str(plan_json),
            "--plan-box",
            "0",
            *_CPU,
        ],
    )
    assert res.exit_code == 0, res.output
    produced = out.exists()
    empty = (tmp_path / "box0.gsplats.zarr.empty").exists()
    assert produced ^ empty  # exactly one of a store or an .empty marker


def test_plan_box_out_of_range_errors(tmp_path):
    vol = _vol(tmp_path)
    plan_json = tmp_path / "plan.json"
    assert (
        runner.invoke(
            app_gsplat, ["fit", str(vol), str(plan_json), *_CONTENT, "--plan-only"]
        ).exit_code
        == 0
    )
    res = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(tmp_path / "x.gsplats.zarr"),
            "--tiling",
            "content",
            "--plan",
            str(plan_json),
            "--plan-box",
            "9999",
            *_CPU,
        ],
    )
    assert res.exit_code != 0  # box index out of range


def test_content_fit_parallel_jobs(tmp_path):
    # concurrent path: -j 2 spawns per-box `fit --plan-box` workers, merges.
    vol = _vol(tmp_path)
    out = tmp_path / "par.gsplats.zarr"
    res = runner.invoke(
        app_gsplat, ["fit", str(vol), str(out), *_CONTENT, *_CPU, "-j", "2"]
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert _kind(out) == "GSplatPartition"


def test_content_fit_honors_compress(tmp_path):
    # --compress must thread through the content save path (was silently dropped):
    # a compressed store is a single archive FILE, not a .gsplats.zarr directory.
    vol = _vol(tmp_path)
    out = tmp_path / "cz.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        ["fit", str(vol), str(out), *_CONTENT, *_CPU, "--compress", "zip"],
    )
    assert res.exit_code == 0, res.output
    assert (
        out.is_file()
    )  # compressed → archive file; pre-fix it was an uncompressed dir


def test_bad_jobs_value_errors(tmp_path):
    vol = _vol(tmp_path)
    res = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(tmp_path / "o.gsplats.zarr"),
            *_CONTENT,
            *_CPU,
            "-j",
            "notanint",
        ],
    )
    assert res.exit_code != 0  # --jobs must be int or 'auto'
