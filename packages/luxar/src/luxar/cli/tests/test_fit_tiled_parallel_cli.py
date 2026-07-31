"""CLI integration test for parallel tiled fitting (``fit --tiling uniform -j N``).

Drives the real ``luxar gsplat fit`` command end-to-end: the parallel branch
spawns ``fit --tile i/M`` worker subprocesses, then merges.  Gated on torch
(fitting requires it).  Kept small (2D, few iters) to stay fast.

The merge-parity tests use ``--flat`` (a single comparable leaf); a separate
test covers the partition-by-default output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# End-to-end CLI tiled fitting (seconds per test). Slow → CI runs `-m "not
# slow"`; the full suite runs locally pre-push.
pytestmark = pytest.mark.slow

runner = CliRunner()


def _make_volume(path: Path) -> None:
    """A small 2D image with a few Gaussian blobs."""
    v = np.zeros((48, 48), np.float32)
    yy, xx = np.ogrid[:48, :48]
    for cy, cx in [(12, 12), (36, 36), (12, 36), (36, 12)]:
        v += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0).astype(np.float32)
    np.save(path, v)


def _make_sparse_volume(path: Path) -> None:
    """A 2D image with a single corner blob — most tiles window to 0 splats."""
    v = np.zeros((48, 48), np.float32)
    yy, xx = np.ogrid[:48, :48]
    v += np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 20.0).astype(np.float32)
    np.save(path, v)


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tiled_parallel_cli_end_to_end(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "par.gsplats.zarr"

    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "-j",
            "2",
            "--flat",
            "--seeds",
            "10",
            "-n",
            "15",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    data = GSplatData.load(out)
    assert data.n_splats > 0
    assert data.ndim == 2
    # the parallel temp dir (now token-suffixed: .tiles.<host>-<pid>-<rand>)
    # must be cleaned up after a successful merge
    assert list(tmp_path.glob(".par.gsplats.zarr.tiles*")) == []


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tiled_parallel_partition_by_default(tmp_path: Path) -> None:
    """Without --flat, a uniform tiled fit emits a kind=partition (one part/tile)."""
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "par.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "-j",
            "2",
            "--seeds",
            "10",
            "-n",
            "15",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    node, _ = load_gsplat_node(out)
    assert type(node).__name__ == "GSplatPartition"


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tiled_parallel_handles_empty_tiles(tmp_path: Path) -> None:
    """Sparse input → many 0-splat tiles must NOT crash the parallel fit.

    Pre-fix, a windowed-to-zero tile's worker save() raised
    'Cannot write empty points', failing the whole run; the empty-tile marker
    path must let the parent skip it and match the sequential result.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    vol = tmp_path / "sparse.npy"
    _make_sparse_volume(vol)

    par = tmp_path / "par.gsplats.zarr"
    seq = tmp_path / "seq.gsplats.zarr"
    common = [
        "--tiling",
        "uniform",
        "--tile-size",
        "16",
        "--overlap",
        "4",
        "--flat",
        "--seeds",
        "10",
        "-n",
        "15",
        "--device",
        "cpu",
    ]
    rp = runner.invoke(app, ["gsplat", "fit", str(vol), str(par), *common, "-j", "2"])
    rs = runner.invoke(app, ["gsplat", "fit", str(vol), str(seq), *common])

    assert rp.exit_code == 0, rp.output
    assert rs.exit_code == 0, rs.output
    # empty tiles skipped in both paths → identical splat count
    assert GSplatData.load(par).n_splats == GSplatData.load(seq).n_splats


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tiled_parallel_matches_sequential(tmp_path: Path) -> None:
    """`-j 2` and the default `-j 1` produce the same splat count (flat merge)."""
    from luxar.gsplats.gsplat_data import GSplatData

    vol = tmp_path / "vol.npy"
    _make_volume(vol)

    base = [
        "gsplat",
        "fit",
        str(vol),
        "",
        "--tiling",
        "uniform",
        "--tile-size",
        "24",
        "--overlap",
        "4",
        "--flat",
        "--seeds",
        "10",
        "-n",
        "15",
        "--device",
        "cpu",
    ]

    seq_out = tmp_path / "seq.gsplats.zarr"
    args = base.copy()
    args[3] = str(seq_out)
    r1 = runner.invoke(app, args)
    assert r1.exit_code == 0, r1.output

    par_out = tmp_path / "par.gsplats.zarr"
    args = base.copy()
    args[3] = str(par_out)
    args += ["-j", "2"]
    r2 = runner.invoke(app, args)
    assert r2.exit_code == 0, r2.output

    assert GSplatData.load(seq_out).n_splats == GSplatData.load(par_out).n_splats


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tiled_recipe_stream_gives_partition_of_ladders(tmp_path: Path) -> None:
    """`fit --tiling uniform --recipe stream` → a kind=partition whose parts
    each carry their own additive ladder (the 'tiles' topology), built at
    fit time without a separate `gsplat lod` pass."""
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import GSplatLeaf, iter_leaves

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "lod.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "-j",
            "2",
            "--recipe",
            "stream",
            "--n-lods",
            "2",
            "--seeds",
            "12",
            "-n",
            "15",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    node, _ = load_gsplat_node(out)
    assert type(node).__name__ == "GSplatPartition"
    leaves = list(iter_leaves(node))
    assert all(isinstance(leaf, GSplatLeaf) for leaf in leaves)
    # the additive recipe ran: at least one part has a >1-entry ladder
    assert any(leaf.n_additive_sublods > 1 for leaf in leaves), (
        f"no additive ladder built: {[leaf.n_additive_sublods for leaf in leaves]}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_recipe_rejects_flat(tmp_path: Path) -> None:
    """--recipe needs a partition output, so it is rejected with --flat."""
    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "x.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--recipe",
            "stream",
            "--flat",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "flat" in result.output.lower() and "partition" in result.output.lower()
    assert not out.exists()


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_recipe_rejects_tiling_none(tmp_path: Path) -> None:
    """--recipe needs a tiled fit; a whole-volume (--tiling none) fit is rejected."""
    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "x.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "none",
            "--recipe",
            "stream",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "tiled" in result.output.lower()
    assert not out.exists()


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_recipe_rejects_cross_recipe_knobs(tmp_path: Path) -> None:
    """--recipe stream must reject substitutive-only knobs (and vice-versa) —
    mirroring `gsplat lod`, which errors on irrelevant options rather than
    silently dropping them. No fit is performed (validation fails first)."""
    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "x.gsplats.zarr"

    # stream recipe + a substitutive-only knob → rejected
    res = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--recipe",
            "stream",
            "--compression-factor",
            "8",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code != 0
    assert "--compression-factor" in res.output and "not used" in res.output.lower()
    assert not out.exists()

    # substitutive recipe + an additive-only knob → rejected
    res2 = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--recipe",
            "levels",
            "--n-lods",
            "5",
            "--device",
            "cpu",
        ],
    )
    assert res2.exit_code != 0
    assert "--n-lods" in res2.output and "not used" in res2.output.lower()


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_recipe_short_flags_parse(tmp_path: Path) -> None:
    """fit --recipe gains lod's short flags (-r/-K/-L/-m/-b). Drive them into the
    cross-recipe rejection (validation runs before any fit): `-r additive -K 8`
    proves -r->recipe and -K->compression_factor without a fit."""
    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "x.gsplats.zarr"
    res = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "-r",
            "stream",
            "-K",
            "8",
            "--device",
            "cpu",
        ],
    )
    # -r=additive + -K (a substitutive-only knob) → cross-recipe rejection
    assert res.exit_code != 0
    assert "--compression-factor" in res.output and "not used" in res.output.lower()
