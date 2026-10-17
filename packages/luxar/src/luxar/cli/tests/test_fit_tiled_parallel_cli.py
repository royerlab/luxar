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
from luxar.cli.tests._testing import normalized_cli_output

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_tile_amplitude_floor_matches_sequential_parallel_and_worker(
    tmp_path: Path,
) -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    source = tmp_path / "image.npy"
    np.save(source, image)
    common = [
        "gsplat",
        "fit",
        str(source),
        "",
        "--tiling",
        "uniform",
        "--tile-size",
        "32",
        "--overlap",
        "8",
        "--flat",
        "--seeds",
        "256",
        "--iters",
        "20",
        "--device",
        "cpu",
        "--cull-retention",
        "0",
        "--tile-amplitude-floor",
        "1e-5",
    ]
    pruned_counts = []
    for scale in (1.0, 1000.0):
        np.save(source, image * scale)
        outputs = []
        for jobs in ("1", "2"):
            output = tmp_path / f"fit_{scale}_{jobs}.gsplats.zarr"
            argv = common.copy()
            argv[3] = str(output)
            result = runner.invoke(app, [*argv, "-j", jobs])
            assert result.exit_code == 0, result.output
            data = GSplatData.load(output, include_stats=True)
            span = data.stats["image_max"] - data.stats["image_min"]
            assert np.all(data.amplitudes >= 1e-5 * span)
            assert data.stats["tile_amplitude_floor"] == 1e-5
            assert data.stats["tile_amplitude_pruned"] > 0
            outputs.append(data)
        assert outputs[0].n_splats == outputs[1].n_splats
        assert (
            outputs[0].stats["tile_amplitude_pruned"]
            == outputs[1].stats["tile_amplitude_pruned"]
        )
        pruned_counts.append(outputs[0].stats["tile_amplitude_pruned"])
    assert pruned_counts[0] == pruned_counts[1]


def test_tile_amplitude_floor_refuses_nonuniform_fit(tmp_path: Path) -> None:
    source = tmp_path / "image.npy"
    np.save(source, np.ones((8, 8), dtype=np.float32))
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(source),
            str(tmp_path / "out.gsplats.zarr"),
            "--tiling",
            "content",
            "--tile-amplitude-floor",
            "1e-5",
        ],
    )
    assert result.exit_code == 2
    assert "requires uniform tiling" in normalized_cli_output(result)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    output = normalized_cli_output(result).lower()
    assert "flat" in output and "partition" in output
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
    assert "tiled" in normalized_cli_output(result).lower()
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
    output = normalized_cli_output(res)
    assert "--compression-factor" in output and "not used" in output.lower()
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
    output = normalized_cli_output(res2)
    assert "--n-lods" in output and "not used" in output.lower()


def test_fit_help_does_not_advertise_unsupported_stream_sizing() -> None:
    result = runner.invoke(app, ["gsplat", "fit", "--help"])

    assert result.exit_code == 0
    output = normalized_cli_output(result)
    assert "--target-ms" not in output
    assert "--bandwidth-mbps" not in output
    assert "--bytes-per-splat" not in output


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_refine_volume_rejects_downscale(tmp_path: Path) -> None:
    """`--refine volume` crops the volume to each tile, which only holds while the
    tile grid and the splats share a coordinate frame.

    Under ``--downscale`` they do not: the grid is computed on the DOWNSCALED
    shape (so the parent and the workers agree on the tile count) while every
    worker rescales its splats back to full resolution. Each tile's crop would
    then be a factor too small and in the wrong place, so the re-fit would target
    the wrong data — silently, since the never-worse guard compares against that
    same wrong crop. Rejected before any fitting. (The sequential tiled path
    refuses ``--recipe`` outright under ``--downscale``, so ``-j`` is what makes
    this combination otherwise reachable.)
    """
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
            "--downscale",
            "2",
            "--recipe",
            "levels",
            "--refine",
            "volume",
            "-j",
            "2",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code != 0
    output = normalized_cli_output(res)
    assert "--downscale" in output and "--refine volume" in output
    assert not out.exists()


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
    output = normalized_cli_output(res)
    assert "--compression-factor" in output and "not used" in output.lower()


# ---------------------------------------------------------------------------
# Content-tiled fits resume from per-box checkpoints
# ---------------------------------------------------------------------------

_CONTENT_ARGS = [
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
    "--preset",
    "draft",
    "--device",
    "cpu",
]


def _make_blob_volume_3d(path: Path) -> None:
    zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
    v = np.zeros((48, 48, 48), np.float32)
    rng = np.random.default_rng(0)
    for _ in range(12):
        cz, cy, cx = rng.integers(6, 42, 3)
        v += np.exp(-((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
    np.save(path, np.clip(v, 0, 1).astype(np.float32))


@pytest.mark.skipif(not HAS_TORCH, reason="fitting requires torch")
def test_content_fit_resumes_finished_boxes_after_a_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill after the first box; the rerun reloads it and fits only the rest.

    The per-box fit is stubbed (a real CPU fit per box is far too slow for a
    unit test); the interrupt is the ``KeyboardInterrupt`` SIGTERM is routed
    into for the duration of the fit.
    """
    import importlib
    import signal

    from luxar._process import _raise_keyboard_interrupt
    from luxar.gsplats.gsplat_data import GSplatData

    fit_planned_mod = importlib.import_module("luxar.gsplats.planner.fit_planned")
    vol = tmp_path / "vol.npy"
    _make_blob_volume_3d(vol)
    out = tmp_path / "content.gsplats.zarr"
    fitted: list[tuple[int, ...]] = []
    state = {"interrupt_after": 1, "sigterm": None}

    def fake_fit_one_box(V, b, pad, cap, **kwargs):
        state["sigterm"] = signal.getsignal(signal.SIGTERM)
        if state["interrupt_after"] is not None and len(fitted) >= 1:
            raise KeyboardInterrupt
        fitted.append(tuple(b.box))
        z0, z1, y0, y1, x0, x1 = b.box
        centers = np.array(
            [[(z0 + z1) / 2, (y0 + y1) / 2, (x0 + x1) / 2]] * 3, np.float32
        )
        centers[:, 2] += np.array([-1.0, 0.0, 1.0], np.float32)
        chol = np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (3, 1))
        return GSplatData(
            centers=centers,
            amplitudes=np.full(3, 0.5, np.float32),
            cholesky_factors=chol,
            stats={
                "floor": 0.0,
                "image_min": 0.0,
                "image_max": 1.0,
                "intensity_range": 1.0,
            },
        )

    monkeypatch.setattr(fit_planned_mod, "_fit_one_box", fake_fit_one_box)
    args = ["gsplat", "fit", str(vol), str(out), *_CONTENT_ARGS]
    try:
        first = runner.invoke(app, args)
        assert first.exit_code != 0
    except KeyboardInterrupt:
        pass
    boxes_dir = tmp_path / "content.gsplats.zarr.boxes"
    assert len(fitted) == 1
    assert state["sigterm"] is _raise_keyboard_interrupt  # SIGTERM -> KI
    assert signal.getsignal(signal.SIGTERM) is not _raise_keyboard_interrupt
    assert len(list(boxes_dir.glob("box_*.done.json"))) == 1
    assert not out.exists()

    state["interrupt_after"] = None
    first_box = fitted[0]
    fitted.clear()
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert first_box not in fitted  # resumed from the checkpoint, not refit
    assert len(fitted) >= 1
    assert out.exists()
    assert not boxes_dir.exists()  # removed only after the final save
