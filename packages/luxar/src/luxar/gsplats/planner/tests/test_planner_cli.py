"""End-to-end CLI tests for content-aware fitting (`gsplat fit --tiling content`).

The standalone `gsplat plan` command was folded into `fit --tiling content`
(+ `--plan-only` / `--plan` / hidden `--plan-box`); these exercise that surface.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat
from luxar.gsplats.io.load_gsplats import load_gsplat_node

# Heavy end-to-end content-mode CPU fitting (~15 min for this module — the
# dominant cost of the serial suite). Marked slow so CI's `-m "not slow"` pass
# stays under the runner timeout; the full suite (incl. slow) runs locally
# before every push.
pytestmark = pytest.mark.slow

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


def test_content_fit_recipe_stream_gives_partition_of_ladders(tmp_path):
    """`fit --tiling content --recipe stream` → a kind=partition whose every
    box-part carries its own additive ladder (per-part LOD at fit time)."""
    from luxar.gsplats.tree import GSplatLeaf, iter_leaves

    vol = _vol(tmp_path)
    out = tmp_path / "clod.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(out),
            *_CONTENT,
            *_CPU,
            "--recipe",
            "stream",
            "--n-lods",
            "2",
        ],
    )
    assert res.exit_code == 0, res.output
    node, _ = load_gsplat_node(out)
    assert type(node).__name__ == "GSplatPartition"
    leaves = list(iter_leaves(node))
    assert all(isinstance(leaf, GSplatLeaf) for leaf in leaves)
    assert any(leaf.n_additive_sublods > 1 for leaf in leaves), (
        f"no additive ladder built: {[leaf.n_additive_sublods for leaf in leaves]}"
    )


def test_content_fit_recipe_rejects_flat(tmp_path):
    """--recipe is incompatible with --flat (needs a partition output)."""
    vol = _vol(tmp_path)
    out = tmp_path / "x.gsplats.zarr"
    res = runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(out),
            *_CONTENT,
            *_CPU,
            "--recipe",
            "stream",
            "--flat",
        ],
    )
    assert res.exit_code != 0
    assert "flat" in res.output.lower() and "partition" in res.output.lower()


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


def test_content_fit_warns_on_unsupported_flags(tmp_path):
    """--denoise/--downscale/--progressive are not implemented for content; the
    CLI must warn (not silently ignore them)."""
    vol = _vol(tmp_path)
    out = tmp_path / "w.gsplats.zarr"
    res = runner.invoke(
        app_gsplat, ["fit", str(vol), str(out), *_CONTENT, *_CPU, "--denoise"]
    )
    assert res.exit_code == 0, res.output
    assert "not supported" in res.output.lower()


def test_content_fit_threads_iters_into_fit_config(tmp_path, monkeypatch):
    """--iters must reach the fit config in content mode (pre-fix it was dropped:
    run_content_fit forwarded only preset/device). Spy on load_fit_config and
    abort before the real fit — we only assert the override was threaded."""
    import luxar.cli.gsplat_config as cfg

    captured: dict = {}

    class _Stop(Exception):
        pass

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(cfg, "load_fit_config", _spy)
    vol = _vol(tmp_path)
    runner.invoke(
        app_gsplat,
        [
            "fit",
            str(vol),
            str(tmp_path / "o.gsplats.zarr"),
            *_CONTENT,
            *_CPU,
            "--iters",
            "7",
        ],
    )
    assert captured.get("cli_overrides", {}).get("n_iters") == 7
    assert captured.get("config_path") is None  # no --config passed


def test_content_fit_warns_on_ignored_seeds(tmp_path):
    """--seeds is superseded by the content plan; the CLI must say so (not drop
    it silently). Uses --plan-only so no fit runs."""
    vol = _vol(tmp_path)
    out = tmp_path / "plan.json"
    res = runner.invoke(
        app_gsplat,
        ["fit", str(vol), str(out), *_CONTENT, "--plan-only", "--seeds", "5000"],
    )
    assert res.exit_code == 0, res.output
    assert "--seeds is ignored" in res.output
