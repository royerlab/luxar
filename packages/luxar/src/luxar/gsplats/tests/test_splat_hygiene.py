"""Closing speck pruning (``min_extent``) shared by every fitter.

``min_extent`` drops splats whose geometric-mean SPATIAL sigma is below a
number of fitted voxels; the stored quality must describe the pruned set and
every fitter (single-pass, tiled, content, progressive) must apply it once. A
mass guard keeps the splats when pruning would remove more than
``min_extent_max_mass_share`` of the brightness. ``relocation_reseed_sigma_vox``
reaches ``DynamicOpsConfig``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from luxar.gsplats.fitting.splat_hygiene import (
    DEFAULT_MIN_EXTENT_MAX_MASS_SHARE,
    DEFAULT_MIN_EXTENT_VOX,
    aggregate_min_extent_stats,
    fitted_voxel_unit,
    integrated_mass,
    min_extent_keep_mask,
    spatial_extent_vox,
    validate_max_mass_share,
    validate_min_extent,
    validate_reseed_sigma,
)
from luxar.gsplats.gsplat_data import GSplatData


def _diag_splats(sigmas: list[list[float]], amps: list[float]) -> GSplatData:
    """Axis-aligned splats with the given per-axis sigmas (packed Cholesky)."""
    sig = np.asarray(sigmas, dtype=np.float32)
    n, d = sig.shape
    chol = []
    for row in sig:
        packed = []
        for i in range(d):
            packed += [0.0] * i + [float(row[i])]
        chol.append(packed)
    return GSplatData(
        centers=np.zeros((n, d), np.float32) + 5.0,
        amplitudes=np.asarray(amps, np.float32),
        cholesky_factors=np.asarray(chol, np.float32),
    )


# --------------------------------------------------------------------------
# Pruning math
# --------------------------------------------------------------------------


def test_extent_is_geometric_mean_of_spatial_sigmas() -> None:
    data = _diag_splats([[0.5, 0.5, 0.5], [0.25, 1.0, 1.0], [2.0, 2.0, 2.0]], [1, 1, 1])
    ext = spatial_extent_vox(data, np.ones(3))
    np.testing.assert_allclose(ext, [0.5, 0.25 ** (1 / 3), 2.0], rtol=1e-6)
    keep = min_extent_keep_mask(data, 0.6, np.ones(3))
    assert keep.tolist() == [False, True, True]


def test_stacked_zero_variance_axis_is_ignored() -> None:
    # 3 spatial axes + a stacked time axis with sigma 0 for every splat: the
    # extent must be the spatial mean, not collapse to ~0 (which would prune all).
    data = _diag_splats([[0.5, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 0.0]], [1, 1])
    ext = spatial_extent_vox(data, np.ones(4))
    np.testing.assert_allclose(ext, [0.5, 1.0], rtol=1e-6)
    assert min_extent_keep_mask(data, 0.6, np.ones(4)).tolist() == [False, True]


def test_physical_units_and_downscale_are_undone() -> None:
    # A fit written in physical units (voxel_size z=2, y=x=0.5): sigma
    # (1.0, 0.25, 0.25) physical is (0.5, 0.5, 0.5) voxels — a speck.
    data = _diag_splats([[1.0, 0.25, 0.25], [2.0, 0.5, 0.5]], [1, 1])
    unit = fitted_voxel_unit(3, voxel_size=(2.0, 0.5, 0.5), output_space="real")
    np.testing.assert_allclose(spatial_extent_vox(data, unit), [0.5, 1.0], rtol=1e-6)
    assert min_extent_keep_mask(data, 0.6, unit).tolist() == [False, True]
    # Voxel output ignores voxel_size; downscale multiplies the unit.
    np.testing.assert_allclose(
        fitted_voxel_unit(3, voxel_size=(2.0, 1, 1), output_space="voxel"), [1, 1, 1]
    )
    np.testing.assert_allclose(
        fitted_voxel_unit(3, voxel_size=2.0, output_space="real", downscale=(1, 2, 2)),
        [2.0, 4.0, 4.0],
    )


def test_validation() -> None:
    assert validate_min_extent(None) == 0.0
    assert validate_min_extent(0) == 0.0
    assert validate_min_extent(0.6) == 0.6
    for bad in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            validate_min_extent(bad)
    assert validate_reseed_sigma(None) is None
    assert validate_reseed_sigma(0.9) == 0.9
    for bad in (0.0, -1.0, float("nan"), False):
        with pytest.raises(ValueError):
            validate_reseed_sigma(bad)


def test_prune_reports_count_and_mass_share() -> None:
    from luxar.gsplats.fit_gsplats import prune_min_extent

    data = _diag_splats([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]], [8, 1, 4])
    mass = integrated_mass(data)  # 8*0.125=1, 1, 4*0.064=0.256
    data.stats["psnr_db"] = 30.0
    out, n = prune_min_extent(data, 0.6, np.ones(3), max_mass_share=1.0)
    assert n == 2 and out.n_splats == 1
    assert out.stats["n_splats"] == 1
    assert out.stats["min_extent_removed"] == 2
    assert out.stats["min_extent_vox"] == 0.6
    assert out.stats["min_extent_removed_mass_share"] == pytest.approx(
        (mass[0] + mass[2]) / mass.sum()
    )
    assert out.stats["min_extent_input_mass"] == pytest.approx(mass.sum())
    # The fit's measured score rides across (the caller re-scores).
    assert out.stats["psnr_db"] == 30.0
    assert out.stats["min_extent_skipped"] is False


def test_guard_keeps_splats_carrying_real_mass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Above the mass share the splats are kept, with one note and a flag."""
    from luxar.gsplats.fit_gsplats import prune_min_extent

    # Small splats carrying 1.256 / 2.256 = 56% of the brightness.
    data = _diag_splats([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.4, 0.4, 0.4]], [8, 1, 4])
    out, n = prune_min_extent(data, 0.6, np.ones(3))  # default guard 2%
    assert n == 0 and out.n_splats == 3
    assert out.stats["min_extent_skipped"] is True
    assert out.stats["min_extent_removed"] == 0
    assert out.stats["min_extent_removed_mass_share"] == 0.0
    note = capsys.readouterr().out
    assert "would remove 2 splats" in note and "kept them" in note

    # Just under the guard: pruned. 1 speck of mass 0.125 among 100 of mass 1.
    many = _diag_splats([[0.5, 0.5, 0.5]] + [[1.0, 1.0, 1.0]] * 100, [1.0] * 101)
    out, n = prune_min_extent(many, 0.6, np.ones(3))
    assert n == 1 and out.stats["min_extent_skipped"] is False
    assert (
        out.stats["min_extent_removed_mass_share"] < DEFAULT_MIN_EXTENT_MAX_MASS_SHARE
    )


def test_guard_threshold_validation() -> None:
    assert DEFAULT_MIN_EXTENT_MAX_MASS_SHARE == 0.02
    assert validate_max_mass_share(None) == 0.02
    assert validate_max_mass_share(0.5) == 0.5
    for bad in (0.0, -0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            validate_max_mass_share(bad)


def test_prune_records_zero_when_nothing_is_small() -> None:
    from luxar.gsplats.fit_gsplats import prune_min_extent

    data = _diag_splats([[1.0, 1.0, 1.0]], [1])
    out, n = prune_min_extent(data, 0.6, np.ones(3))
    assert n == 0 and out.n_splats == 1
    assert out.stats["min_extent_removed"] == 0
    assert out.stats["min_extent_removed_mass_share"] == 0.0


def test_aggregate_combines_shares_through_input_mass() -> None:
    target: dict = {}
    aggregate_min_extent_stats(
        [
            {
                "min_extent_vox": 0.6,
                "min_extent_removed": 3,
                "min_extent_removed_mass_share": 0.5,
                "min_extent_input_mass": 2.0,
            },
            {
                "min_extent_vox": 0.6,
                "min_extent_removed": 1,
                "min_extent_removed_mass_share": 0.0,
                "min_extent_input_mass": 8.0,
            },
            {},  # a tile without a record (e.g. empty) contributes nothing
        ],
        target,
    )
    assert target["min_extent_skipped"] is False
    assert target["min_extent_removed"] == 4
    assert target["min_extent_input_mass"] == pytest.approx(10.0)
    assert target["min_extent_removed_mass_share"] == pytest.approx(0.1)
    assert target["min_extent_vox"] == 0.6

    skipped: dict = {}
    aggregate_min_extent_stats(
        [
            {
                "min_extent_vox": 0.6,
                "min_extent_removed": 0,
                "min_extent_skipped": True,
            },
            {
                "min_extent_vox": 0.6,
                "min_extent_removed": 2,
                "min_extent_skipped": False,
            },
        ],
        skipped,
    )
    assert skipped["min_extent_skipped"] is True and skipped["min_extent_removed"] == 2


# --------------------------------------------------------------------------
# fit_gaussian_splats wiring
# --------------------------------------------------------------------------


def _blob_volume(seed: int = 0, shape=(16, 32, 32)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    V = np.zeros(shape, np.float32)
    for _ in range(6):
        c = rng.uniform([3, 5, 5], [shape[0] - 3, shape[1] - 5, shape[2] - 5])
        s = rng.uniform(1.2, 2.0)
        V += np.exp(
            -((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * s * s)
        )
    return V


_FAST = dict(n_iters=30, device="cpu", use_cuda=False, use_metal=False, verbose=False)


def test_default_is_on_and_zero_disables() -> None:
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    sig = inspect.signature(fit_gaussian_splats)
    assert sig.parameters["min_extent"].default == DEFAULT_MIN_EXTENT_VOX == 0.6
    assert sig.parameters["relocation_reseed_sigma_vox"].default is None

    V = _blob_volume()
    on = fit_gaussian_splats(V, seeds=120, floor="none", **_FAST)
    assert on.stats["min_extent_vox"] == 0.6
    assert on.stats["min_extent_removed"] >= 0
    off = fit_gaussian_splats(V, seeds=120, floor="none", min_extent=0, **_FAST)
    assert "min_extent_removed" not in off.stats


def test_fit_prunes_and_rescores_the_saved_model() -> None:
    """A huge threshold forces pruning; the stored PSNR is the pruned set's."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.rendering.volume_rendering import render_to_volume

    V = _blob_volume()
    out = fit_gaussian_splats(
        V,
        seeds=120,
        floor="none",
        cull_retention=None,
        min_extent=1.6,
        min_extent_max_mass_share=1.0,
        **_FAST,
    )
    assert out.stats["min_extent_removed"] > 0
    assert out.stats["n_splats"] == out.n_splats
    assert np.all(spatial_extent_vox(out, np.ones(3)) >= 1.6)
    # Independently render the delivered splats and score them: the stored
    # PSNR must match (it would not if it were the pre-pruning measurement).
    import torch

    from luxar.gsplats.metrics import compute_quality_metrics

    rendered = torch.from_numpy(
        np.asarray(
            render_to_volume(
                out, shape=V.shape, device="cpu", truncate=out.truncation_radius
            )
        )
    )
    ref = torch.from_numpy(V - float(out.stats["image_min"]))
    q = compute_quality_metrics(rendered.float(), ref.float())
    assert out.stats["psnr_db"] == pytest.approx(q["psnr_db"], abs=0.05)


def test_one_rescore_covers_pruning_and_cull(monkeypatch: pytest.MonkeyPatch) -> None:
    import luxar.gsplats.fit_gsplats as fg

    calls: list[int] = []
    monkeypatch.setattr(
        fg,
        "_rescore_saved_model",
        lambda result, V, **kw: calls.append(result.n_splats),
    )
    data = _diag_splats([[0.5, 0.5, 0.5]] + [[1.0, 1.0, 1.0]] * 3, [1.0, 1.0, 1.0, 1.0])
    data.stats["psnr_db"] = 20.0
    out = fg._post_fit_reductions(
        data,
        np.zeros((8, 8, 8), np.float32),
        cull_retention=None,
        min_extent_vox=0.6,
        max_mass_share=1.0,
        downscale=None,
        verbose=False,
        device="cpu",
        voxel_size=None,
        output_space="real",
    )
    # Pruning alone (no cull) still triggers the re-score, exactly once.
    assert calls == [out.n_splats] == [3]


def test_reseed_sigma_plumbs_into_dynamic_config() -> None:
    from luxar.gsplats.fit_gsplats import _resolve_hygiene_options, fit_gaussian_splats
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

    _, cfg = _resolve_hygiene_options(0.6, 0.9, None)
    assert cfg is not None and cfg.relocation_reseed_sigma_vox == 0.9
    base = DynamicOpsConfig(step_every=7)
    _, cfg = _resolve_hygiene_options(0.6, 1.0, base)
    assert cfg.relocation_reseed_sigma_vox == 1.0 and cfg.step_every == 7
    assert base.relocation_reseed_sigma_vox is None  # caller's config untouched
    _, same = _resolve_hygiene_options(0.6, None, base)
    assert same is base

    out = fit_gaussian_splats(
        _blob_volume(), seeds=80, floor="none", relocation_reseed_sigma_vox=0.9, **_FAST
    )
    assert out.stats["relocation_reseed_sigma_vox_effective"] == 0.9


# --------------------------------------------------------------------------
# Tiled / progressive wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("partition", [False, True])
def test_tiled_fit_prunes_per_tile_and_sums_the_record(partition: bool) -> None:
    from luxar.gsplats.fit_tiled_gsplats import fit_tiled
    from luxar.gsplats.tree import total_splats

    V = _blob_volume(shape=(16, 40, 40))
    result = fit_tiled(
        V,
        tile_size=24,
        overlap=8,
        partition=partition,
        cull_retention=None,
        seeds=200,
        floor="none",
        min_extent=1.6,
        min_extent_max_mass_share=1.0,
        **_FAST,
    )
    stats = result.meta["fit_stats"] if partition else result.stats
    assert stats["num_tiles"] > 1
    assert stats["min_extent_removed"] > 0
    assert 0 < stats["min_extent_removed_mass_share"] < 1
    if partition:
        parts = _leaves(result)
        assert sum(p.n_splats for p in parts) == total_splats(result)
        for part in parts:
            assert np.all(spatial_extent_vox(part, np.ones(3)) >= 1.6)
    else:
        assert np.all(spatial_extent_vox(result, np.ones(3)) >= 1.6)


def _leaves(node) -> "list[GSplatData]":  # noqa: ANN001 - tiny test helper
    from luxar.gsplats.tree import iter_leaves

    out = []
    for leaf in iter_leaves(node):
        subs = leaf.additive_sublods
        out.append(
            GSplatData(
                centers=np.concatenate([s.centers for s in subs]),
                amplitudes=np.concatenate([s.amplitudes for s in subs]),
                cholesky_factors=np.concatenate([s.cholesky_factors for s in subs]),
            )
        )
    return out


def test_progressive_prunes_once_on_the_accumulated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.fit_gsplats as fg
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    seen: list[float] = []
    real = fg.fit_gaussian_splats

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(kwargs.get("min_extent"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fg, "fit_gaussian_splats", spy)
    out = fit_progressive_gaussian_splats(
        _blob_volume(),
        max_splats=200,
        max_splats_per_pass=100,
        iters_per_pass=20,
        residual_pass_min_iters=10,
        max_passes=2,
        cull_retention=None,
        min_extent=1.6,
        min_extent_max_mass_share=1.0,
        device="cpu",
        verbose=False,
        floor="none",
    )
    assert seen and all(m == 0.0 for m in seen)
    assert out.stats["min_extent_removed"] > 0
    assert np.all(spatial_extent_vox(out, np.ones(3)) >= 1.6)


def test_fit_guard_skips_a_threshold_that_bites_into_structure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A threshold that would strip real structure leaves the fit untouched."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    V = _blob_volume()
    kw = dict(_FAST, seeds=120, floor="none", cull_retention=None)
    base = fit_gaussian_splats(V, min_extent=0, **kw)
    guarded = fit_gaussian_splats(V, min_extent=5.0, **kw)
    assert guarded.stats["min_extent_skipped"] is True
    assert guarded.stats["min_extent_removed"] == 0
    assert "kept them" in capsys.readouterr().out
    # Same seed/config: the guard returned the unpruned splat set.
    assert guarded.n_splats == base.n_splats


def test_reseed_sigma_warns_when_nothing_relocates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from luxar.gsplats.fit_gsplats import _warn_reseed_without_relocation
    from luxar.gsplats.fit_progressive_gsplats import _drop_inert_reseed_sigma

    _warn_reseed_without_relocation(0.9, enable_dynamic_ops=False)
    assert "no effect" in capsys.readouterr().out
    _warn_reseed_without_relocation(0.9, enable_dynamic_ops=True)
    _warn_reseed_without_relocation(None, enable_dynamic_ops=False)
    assert capsys.readouterr().out == ""

    kwargs = {"relocation_reseed_sigma_vox": 0.9, "lr": 0.01}
    _drop_inert_reseed_sigma(kwargs, enable_dynamic_ops=False)
    assert "no effect" in capsys.readouterr().out
    assert kwargs == {"lr": 0.01}  # so the passes do not warn again
    kept = {"relocation_reseed_sigma_vox": 0.9}
    _drop_inert_reseed_sigma(kept, enable_dynamic_ops=True)
    assert kept == {"relocation_reseed_sigma_vox": 0.9}


def test_cal_measures_the_unpruned_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cal` pins min_extent=0 whatever the inherited fit config says."""
    import luxar.gsplats.fit_gsplats as fg
    from luxar.gsplats.calibration.driver import calibrate

    seen: list[object] = []
    real = fg.fit_gaussian_splats

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(kwargs.get("min_extent"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fg, "fit_gaussian_splats", spy)
    calibrate(
        _blob_volume(),
        k_grid=[20, 40],
        fit_kwargs=dict(_FAST, n_iters=10, min_extent=0.6),
    )
    assert seen and all(m == 0.0 for m in seen)


def test_hygiene_stats_are_saved(tmp_path: Path) -> None:
    data = _diag_splats([[1.0, 1.0, 1.0]], [1.0])
    data.stats.update(
        min_extent_vox=0.6,
        min_extent_removed=3,
        min_extent_removed_mass_share=0.01,
        min_extent_input_mass=5.0,
        min_extent_skipped=False,
    )
    path = tmp_path / "h.gsplats.zarr"
    data.save(path)
    loaded = GSplatData.load(path, include_stats=True)
    assert loaded.stats["min_extent_skipped"] is False
    assert loaded.stats["min_extent_removed"] == 3
    assert loaded.stats["min_extent_removed_mass_share"] == pytest.approx(0.01)


# --------------------------------------------------------------------------
# batch-fit wiring
# --------------------------------------------------------------------------


def test_batch_plan_forwards_speck_flags_to_workers() -> None:
    from luxar.cli.gsplat_ops.batch.planning import (
        DenoiseConfig,
        FitConfig,
        _assemble_fit_args,
    )

    fit_args, _, _ = _assemble_fit_args(
        FitConfig(min_extent=0.0, reseed_sigma=0.9), DenoiseConfig()
    )
    assert fit_args["min-extent"] == "0.0"
    assert fit_args["reseed-sigma"] == "0.9"
    bare, _, _ = _assemble_fit_args(FitConfig(), DenoiseConfig())
    assert "min-extent" not in bare and "reseed-sigma" not in bare


def test_batch_merge_sums_the_pruning_record() -> None:
    from luxar.gsplats.batch.merge_orchestrator import _merge_fitting_info

    parts = [
        {
            "n_splats": 10,
            "min_extent_vox": 0.6,
            "min_extent_removed": 4,
            "min_extent_removed_mass_share": 0.01,
            "min_extent_input_mass": 100.0,
            "min_extent_skipped": False,
        },
        {
            "n_splats": 5,
            "min_extent_vox": 0.6,
            "min_extent_removed": 0,
            "min_extent_removed_mass_share": 0.0,
            "min_extent_input_mass": 100.0,
            "min_extent_skipped": True,
        },
    ]
    info = _merge_fitting_info([], None, parts)
    assert info["min_extent_removed"] == 4
    assert info["min_extent_removed_mass_share"] == pytest.approx(0.005)
    assert info["min_extent_skipped"] is True
