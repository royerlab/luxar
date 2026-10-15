"""``luxar gsplat render`` puts the splats on the right voxel grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from luxar.cli.main import app
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData


def _model(offset: float, n: int = 20, seed: int = 0) -> GSplatData:
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=(rng.uniform(0.0, 10.0, (n, 3)) + offset).astype(np.float32),
        amplitudes=rng.uniform(0.3, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1.0, 0, 1.0, 0, 0, 1.0], (n, 1)).astype(np.float32),
    )


def _render(store: Path, out: Path, *extra: str):
    result = CliRunner().invoke(
        app, ["gsplat", "render", str(store), str(out), "--device", "cpu", *extra]
    )
    assert result.exit_code == 0, result.output
    return result, np.load(out)


def test_bounding_box_render_starts_at_the_lowest_center(tmp_path: Path) -> None:
    """An offset store used to be clipped: extent-sized grid anchored at 0."""
    data = _model(offset=40.0)
    store = tmp_path / "offset.gsplats.zarr"
    data.save(store, encoding_mode=EncodingMode.PRECISION)

    result, volume = _render(store, tmp_path / "out.npy")
    lo = np.floor(data.centers.min(axis=0)).astype(int)
    hi = np.floor(data.centers.max(axis=0)).astype(int)
    assert volume.shape == tuple(hi - lo + 1)
    assert f"origin {tuple(int(v) for v in lo)}" in result.output

    # The same splats rendered on a grid that holds them all, cropped to the box:
    # identical, i.e. nothing was clipped off the top of the data.
    _, big = _render(store, tmp_path / "big.npy", "--shape", "60,60,60")
    crop = big[tuple(slice(a, b + 1) for a, b in zip(lo, hi, strict=True))]
    np.testing.assert_allclose(volume, crop, rtol=1e-4, atol=1e-6)


def test_a_fit_renders_on_its_source_grid(tmp_path: Path) -> None:
    data = _model(offset=2.0)
    data.stats.update({"source_shape": [16, 18, 20]})
    store = tmp_path / "fit.gsplats.zarr"
    data.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)

    result, volume = _render(store, tmp_path / "out.npy")
    assert volume.shape == (16, 18, 20)
    assert "source grid" in result.output


def test_a_physical_fit_renders_back_on_voxels(tmp_path: Path) -> None:
    vox = _model(offset=2.0)
    vs = np.array([2.0, 0.5, 0.5], np.float32)
    tril = np.concatenate([[vs[i]] * (i + 1) for i in range(3)])
    phys = GSplatData(
        centers=vox.centers * vs,
        amplitudes=vox.amplitudes,
        cholesky_factors=vox.cholesky_factors * tril,
        stats={"source_shape": [16, 16, 16], "voxel_size": vs.tolist()},
    )
    store = tmp_path / "phys.gsplats.zarr"
    phys.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)

    result, volume = _render(store, tmp_path / "out.npy")
    expected = vox.render_to_volume(shape=(16, 16, 16), device="cpu")
    assert "voxel_size=[2.0, 0.5, 0.5]" in result.output
    np.testing.assert_allclose(volume, expected, rtol=1e-4, atol=1e-5)

    # '1' renders in the store's own (physical) units instead: the source grid
    # no longer holds the centers, so it falls back to their bounding box.
    raw_result, raw = _render(store, tmp_path / "raw.npy", "--voxel-size", "1")
    assert "bounding box" in raw_result.output
    assert raw.shape != expected.shape


def test_a_geometric_transform_drops_the_voxel_size_stamp(tmp_path: Path) -> None:
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    data = _model(offset=2.0)
    data.stats.update({"voxel_size": [2.0, 0.5, 0.5]})
    store = tmp_path / "phys.gsplats.zarr"
    data.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)
    out = tmp_path / "moved.gsplats.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "transform", str(store), str(out), "--scale", "2,2,2"]
    )
    assert result.exit_code == 0, result.output
    assert "frame stamp (voxel_size)" in result.output
    _, stats = load_gsplat_node(out, include_stats=True)
    assert "voxel_size" not in stats


def _crop_fit_store(tmp_path: Path, physical: bool) -> tuple[Path, GSplatData]:
    """A `--region` crop fit as WP3b CLI fits stamp it: full-volume centers,
    crop-shaped `source_shape`, `source_region`, `physical_coordinates`."""
    crop_model = _model(offset=1.0)  # centers on the crop's own grid
    start = np.array([5, 7, 3], np.float32)
    vs = np.array([2.0, 0.5, 0.5], np.float32) if physical else np.ones(3, np.float32)
    tril = np.concatenate([[vs[i]] * (i + 1) for i in range(3)])
    stats = {
        "source_shape": [14, 14, 14],
        "source_region": [[5, 19], [7, 21], [3, 17]],
        "physical_coordinates": physical,
        "voxel_size": [2.0, 0.5, 0.5],
        "image_min": 0.0,
    }
    stored = GSplatData(
        centers=((crop_model.centers + start) * vs).astype(np.float32),
        amplitudes=crop_model.amplitudes,
        cholesky_factors=(crop_model.cholesky_factors * tril).astype(np.float32),
        stats=stats,
    )
    store = tmp_path / f"crop_{physical}.gsplats.zarr"
    stored.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)
    return store, crop_model


def test_a_crop_fit_renders_and_scores_on_the_crop_grid(tmp_path: Path) -> None:
    import json

    for physical in (True, False):
        store, crop_model = _crop_fit_store(tmp_path, physical)
        expected = crop_model.render_to_volume(shape=(14, 14, 14), device="cpu")

        result, volume = _render(store, tmp_path / f"r_{physical}.npy")
        assert "Crop fit" in result.output
        np.testing.assert_allclose(volume, expected, rtol=1e-4, atol=1e-5)

        # compare against the crop: a perfect reconstruction (render == reference).
        ref = tmp_path / f"crop_ref_{physical}.npy"
        np.save(ref, expected)
        out = tmp_path / f"m_{physical}.json"
        cmp = CliRunner().invoke(
            app,
            [
                "gsplat",
                "compare",
                str(store),
                str(ref),
                "--device",
                "cpu",
                "-j",
                str(out),
            ],
        )
        assert cmp.exit_code == 0, cmp.output
        assert json.loads(out.read_text())["max_abs_error"] < 1e-3


def _stacked_store(tmp_path: Path, physical: bool) -> tuple[Path, GSplatData]:
    """A batch-fit-style physical timelapse: spatial columns first, t LAST, a
    T-first `source_shape`, a 3-value spatial `voxel_size` stamp."""
    vox = _model(offset=2.0, n=40)
    t = np.repeat([0.0, 1.0], 20)[:, None].astype(np.float32)
    vs = np.array([2.0, 0.5, 0.5], np.float32) if physical else np.ones(3, np.float32)
    chol3 = vox.cholesky_factors * np.concatenate([[vs[i]] * (i + 1) for i in range(3)])
    # 4-D packed Cholesky: the 3-D block plus a tiny time row (sigma 0.1).
    chol4 = np.concatenate(
        [chol3, np.zeros((40, 3), np.float32), np.full((40, 1), 0.1, np.float32)],
        axis=1,
    )
    stats = {"source_shape": [2, 16, 16, 16], "image_min": 0.0}
    if physical:
        stats["voxel_size"] = vs.tolist()
    data = GSplatData(
        centers=np.concatenate([vox.centers * vs, t], axis=1).astype(np.float32),
        amplitudes=vox.amplitudes,
        cholesky_factors=chol4.astype(np.float32),
        stats=stats,
    )
    store = tmp_path / f"stacked_{physical}.gsplats.zarr"
    names = [{"name": n} for n in ("z", "y", "x", "t")]
    data.save(
        store,
        encoding_mode=EncodingMode.PRECISION,
        include_fitting_info=True,
        root_attrs={"dimension_metadata": names},
    )
    return store, vox


def test_a_stacked_store_does_not_render_on_its_t_first_source_shape(
    tmp_path: Path,
) -> None:
    """batch-fit T>1 records source_shape (t, z, y, x) while the centers are
    (z, y, x, t): rendering on it put 3% of the centers inside a wrong grid."""
    for physical in (False, True):
        store, vox = _stacked_store(tmp_path, physical)
        result, volume = _render(store, tmp_path / f"s_{physical}.npy")
        assert "bounding box" in result.output, result.output
        assert "source grid" not in result.output
        assert "WARNING" not in result.output
        assert volume.shape[3] == 2  # z, y, x, then the two timepoints
        if physical:
            assert "stacked" in result.output  # the 3-value stamp was padded
            lo = np.floor(vox.centers.min(axis=0)).astype(int)
            hi = np.floor(vox.centers.max(axis=0)).astype(int)
            assert volume.shape[:3] == tuple(hi - lo + 1)


def test_compare_skips_the_source_grid_check_on_a_stacked_store(
    tmp_path: Path,
) -> None:
    store, _ = _stacked_store(tmp_path, physical=True)
    ref = tmp_path / "ref4d.npy"
    np.save(ref, np.zeros((16, 16, 16, 2), np.float32) + 0.01)
    result = CliRunner().invoke(
        app, ["gsplat", "compare", str(store), str(ref), "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert "differs from the grid" not in result.output
    assert "voxel_size=[2.0, 0.5, 0.5, 1.0]" in result.output


def test_sub_unit_spacing_without_a_stamp_warns(tmp_path: Path) -> None:
    """0.25-um voxels, no stamp: every center inside, crammed into a corner."""
    rng = np.random.default_rng(0)
    vox = GSplatData(
        centers=rng.uniform(2.0, 60.0, (60, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 1.0, 60).astype(np.float32),
        cholesky_factors=np.tile([1.0, 0, 1.0, 0, 0, 1.0], (60, 1)).astype(np.float32),
    )
    phys = GSplatData(
        centers=vox.centers * 0.25,
        amplitudes=vox.amplitudes,
        cholesky_factors=vox.cholesky_factors * 0.25,
        stats={"source_shape": [64, 64, 64], "image_min": 0.0},
    )
    store = tmp_path / "quarter.gsplats.zarr"
    phys.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)

    result, _ = _render(store, tmp_path / "q.npy")
    assert "crammed into a corner" in result.output
    assert "--voxel-size" in result.output

    ref = tmp_path / "ref.npy"
    np.save(ref, np.zeros((64, 64, 64), np.float32))
    cmp = CliRunner().invoke(
        app, ["gsplat", "compare", str(store), str(ref), "--device", "cpu"]
    )
    assert "crammed into a corner" in cmp.output
