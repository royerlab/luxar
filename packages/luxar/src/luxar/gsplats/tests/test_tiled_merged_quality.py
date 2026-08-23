"""A tiled fit scores the archive it actually ships.

Every tile scores itself, but tiles overlap and are Hann-apodized, so their
errors do not compose into the merged one — and the merged result is what gets
written. Before this, a tiled archive carried no ``psnr_db`` at all, which is
the one number a published dataset is expected to state.

The load-bearing test here is the coordinate-frame equivalence: with
``output_space="real"`` the merged splats live in physical coordinates and have
to be mapped back onto the tile grid before they can be rendered against it. An
inverse that is wrong per-axis produces a plausible-looking but badly wrong
number, so it is checked against the same fit scored with no conversion at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from luxar.gsplats import merged_quality
from luxar.gsplats.fit_tiled_gsplats import fit_tiled
from luxar.gsplats.merged_quality import (
    _QUALITY_BUDGET_GB,
    _QUALITY_PEAK_VOLUMES,
    _quality_budget_gb,
)

#: Anisotropic on purpose: an isotropic spacing would hide a per-axis error in
#: the Cholesky row scaling, which is applied row by row.
VOXEL_SIZE = (3.0, 1.0, 1.0)

_QUALITY_KEYS = ("psnr_db", "mse", "ssim", "foreground_psnr_db", "foreground_fraction")


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    """Small structured volume — big enough that tiling actually tiles."""
    shape = (24, 48, 48)
    zz, yy, xx = np.meshgrid(
        *(np.arange(s, dtype=np.float32) for s in shape), indexing="ij"
    )
    vol = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in ((7, 12, 12), (16, 34, 30), (12, 24, 24)):
        vol += np.exp(
            -(((zz - cz) / 2.0) ** 2 + ((yy - cy) / 3.0) ** 2 + ((xx - cx) / 3.0) ** 2)
        )
    return vol


def _fit(volume: np.ndarray, **kwargs: Any) -> Any:
    opts: dict[str, Any] = dict(
        tile_size=24,
        overlap=6,
        seeds=60,
        n_iters=25,
        device="cpu",
        verbose=False,
    )
    opts.update(kwargs)
    return fit_tiled(volume, **opts)


def test_a_tiled_fit_records_merged_quality(
    volume: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    """The regression: the merged archive used to carry no PSNR at all."""
    stats = _fit(volume).stats
    missing = [k for k in _QUALITY_KEYS if k not in stats]
    assert not missing, f"the merged tiled result lost {missing}"
    assert np.isfinite(stats["psnr_db"]) and stats["psnr_db"] > 0
    # The unscored notice is unconditional on its own branch, so nothing else
    # would catch a change that printed it next to a store carrying `psnr_db`.
    assert "No merged quality metrics" not in capsys.readouterr().out


def test_partition_merge_records_the_same_whole_volume_score(
    volume: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    """The default partition path scores the parts as one reconstruction."""
    flat = _fit(volume, partition=False, cull_retention=None)
    partition = _fit(volume, partition=True, cull_retention=None)

    stats = partition.meta["fit_stats"]
    missing = [key for key in _QUALITY_KEYS if key not in stats]
    assert not missing, f"the partition merge lost {missing}"
    assert stats["psnr_db"] == pytest.approx(flat.stats["psnr_db"], abs=0.5)
    assert "No merged quality metrics" not in capsys.readouterr().out


@pytest.mark.parametrize("partition", [False, True])
def test_tiled_quality_is_independent_of_the_removed_pedestal(
    volume: np.ndarray, partition: bool
) -> None:
    """A merged fit and its reference must be scored on the same basis."""
    low_result = _fit(volume + 50.0, partition=partition, cull_retention=None)
    high_result = _fit(volume + 4000.0, partition=partition, cull_retention=None)
    low = low_result.meta["fit_stats"] if partition else low_result.stats
    high = high_result.meta["fit_stats"] if partition else high_result.stats
    low_basis = low_result.meta if partition else low
    high_basis = high_result.meta if partition else high

    assert low_basis["image_min"] == pytest.approx(50.0, abs=0.1)
    assert high_basis["image_min"] == pytest.approx(4000.0, abs=0.1)
    assert high["psnr_db"] == pytest.approx(low["psnr_db"], abs=0.5)


def test_partition_quality_reaches_the_archive_and_info(
    volume: np.ndarray,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Root fitting metrics are persisted and visible through ``gsplat info``."""
    from luxar.cli.gsplat_ops.fitting.fit_utils import save_fit_output
    from luxar.cli.gsplat_ops.inspect_commands import _print_gsplat_tree_summary

    result = _fit(volume, partition=True)
    output = tmp_path / "partition.gsplats.zarr"
    save_fit_output(result, output, compress=None, verbose=False)

    import zarr

    root = zarr.open_group(str(output), mode="r")
    fitting = dict(root["fitting"].attrs)
    assert fitting["psnr_db"] == pytest.approx(result.meta["fit_stats"]["psnr_db"])
    assert fitting["n_splats"] == result.n_splats

    _print_gsplat_tree_summary(output)
    info = capsys.readouterr().out
    assert "Fitting (fitting/):" in info
    assert f"psnr_db: {fitting['psnr_db']:.6f}" in info


def test_partition_lod_density_uses_the_persisted_total_splat_count() -> None:
    """Root ``n_splats`` and ``voxels_per_splat`` describe the same artifact."""
    from luxar.gsplats.fit_tiled_gsplats import merge_tile_results
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import total_splats

    def _region(offset: float) -> GSplatData:
        centers = np.stack(
            [np.linspace(offset, offset + 3, 12, dtype=np.float32)] * 3, axis=1
        )
        return GSplatData(
            centers=centers,
            amplitudes=np.linspace(1.0, 0.2, 12, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]], np.float32),
                (12, 1),
            ),
        )

    node = merge_tile_results(
        [_region(0.0), _region(8.0)],
        volume_shape=(8, 8, 8),
        tile_size=4,
        overlap=0,
        num_tiles=2,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
        partition=True,
        recipe="levels",
        recipe_params=RecipeParams(levels=1, additive_ladders=False),
    )

    stats = node.meta["fit_stats"]
    persisted_count = total_splats(node)
    assert persisted_count > node.n_splats
    assert stats["n_splats"] == persisted_count
    assert stats["voxels_per_splat"] == pytest.approx(8**3 / persisted_count)


def test_the_merged_score_survives_the_physical_coordinate_round_trip(
    volume: np.ndarray,
) -> None:
    """A real-space tiled fit must score the same as one that never converted.

    This is what catches a wrong inverse in ``_to_voxel_frame``: rendering
    physical centers on a voxel grid, or undoing the Cholesky scaling on the
    wrong axis, would collapse the PSNR rather than merely nudge it.
    """
    for partition in (False, True):
        plain_result = _fit(volume, partition=partition)
        real_result = _fit(
            volume,
            partition=partition,
            voxel_size=VOXEL_SIZE,
            output_space="real",
        )
        plain = plain_result.meta["fit_stats"] if partition else plain_result.stats
        real = real_result.meta["fit_stats"] if partition else real_result.stats
        assert real["psnr_db"] == pytest.approx(plain["psnr_db"], abs=0.5), (
            f"physical-coordinate scoring diverged: {real['psnr_db']:.2f} dB vs "
            f"{plain['psnr_db']:.2f} dB — the voxel-frame inverse is wrong"
        )
        assert real["foreground_fraction"] == pytest.approx(
            plain["foreground_fraction"], abs=1e-6
        )


def test_partition_scoring_without_a_stats_target_keeps_the_finished_fit(
    volume: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    """A caller mistake after fitting must decline scoring, not raise."""
    from luxar.gsplats.merged_quality import stamp_merged_quality

    part = _fit(volume, partition=False)
    stamp_merged_quality(
        [part],
        volume,
        volume_shape=volume.shape,
        grid_scale=None,
        device="cpu",
        verbose=False,
        image_min=None,
    )
    assert "not given a stats target" in capsys.readouterr().out


def test_merged_scoring_warns_when_the_basis_is_unknown(
    volume: np.ndarray,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    part = GSplatData(
        centers=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(merged_quality, "_quality_budget_gb", lambda: 0.0)

    merged_quality.stamp_merged_quality(
        part,
        volume,
        volume_shape=volume.shape,
        grid_scale=None,
        device="cpu",
        verbose=False,
        image_min=None,
        stats={},
    )

    output = capsys.readouterr().out
    assert "records no normalization basis" in output
    assert "Merged quality metrics skipped" in output


def test_the_returned_splats_stay_in_physical_coordinates(
    volume: np.ndarray,
) -> None:
    """Scoring copies the mixture; it must not rewrite what the caller gets."""
    real = _fit(volume, voxel_size=VOXEL_SIZE, output_space="real")
    plain = _fit(volume)
    # z is scaled x3, so the physical extent must clearly exceed the voxel one.
    assert real.centers[:, 0].max() > plain.centers[:, 0].max() * 2


def test_the_memory_budget_skips_rather_than_thrashes(
    volume: np.ndarray, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Above the budget the score is skipped — and says so even when quiet.

    ``verbose=False`` here on purpose: a scripted fit is exactly where an
    unscored archive would otherwise appear with no explanation anywhere, since
    nothing about the skip reaches the store.
    """
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "0.0000001")
    stats = _fit(volume).stats
    assert "psnr_db" not in stats
    out = capsys.readouterr().out
    assert "Merged quality metrics skipped" in out
    assert "gsplat compare" in out


def test_a_partition_budget_skip_includes_the_required_flatten_step(
    volume: np.ndarray, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The fallback command must accept the tree-shaped archive it describes."""
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "0.0000001")
    node = _fit(volume, partition=True)
    assert "psnr_db" not in node.meta["fit_stats"]
    out = capsys.readouterr().out
    assert "Merged quality metrics skipped" in out
    assert "gsplat flatten" in out
    assert "gsplat compare" in out


def test_the_peak_estimate_covers_ssims_intermediates() -> None:
    """The budget must count the SSIM peak, not just the two volumes it scores.

    SSIM holds several convolution intermediates live on top of the
    reconstruction and the reference, and its own tiled fallback only engages on
    CUDA. Since scoring runs before the archive is written, an under-count is not
    a missing metric — it is a thrash or an OOM kill that loses the finished fit.
    """
    from luxar.gsplats.metrics import _SSIM_PEAK_TENSOR_COUNT

    assert _QUALITY_PEAK_VOLUMES >= _SSIM_PEAK_TENSOR_COUNT


def test_the_default_budget_is_held_under_the_memory_actually_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling written on one machine must not license a peak on another.

    Scoring materializes the whole volume, and the OOM kill lands before the
    archive is written — so on a host smaller than the ceiling the budget has to
    follow the host, not the constant.
    """
    monkeypatch.delenv("LUXAR_TILED_QUALITY_MAX_GB", raising=False)

    monkeypatch.setattr(merged_quality, "_available_ram_gb", lambda: 4.0)
    assert _quality_budget_gb() == pytest.approx(2.0)

    # A machine with room to spare gets the ceiling, not a multiple of its RAM.
    monkeypatch.setattr(merged_quality, "_available_ram_gb", lambda: 1024.0)
    assert _quality_budget_gb() == pytest.approx(_QUALITY_BUDGET_GB)

    # Unmeasurable (a platform without SC_AVPHYS_PAGES) falls back to the ceiling
    # rather than declining to score at all.
    monkeypatch.setattr(merged_quality, "_available_ram_gb", lambda: None)
    assert _quality_budget_gb() == pytest.approx(_QUALITY_BUDGET_GB)


def test_an_explicit_override_still_wins_over_the_memory_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator knows what the machine can take; the cap is for the default."""
    monkeypatch.setattr(merged_quality, "_available_ram_gb", lambda: 4.0)
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "64")
    assert _quality_budget_gb() == pytest.approx(64.0)


def test_a_nan_override_cannot_switch_the_guard_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``NaN`` parses as a float but compares False against everything.

    It is the one malformed override that would DISABLE the budget rather than
    trip it — every volume would look in-budget — so it has to be refused like
    any other unusable value, not accepted because ``float()`` took it.
    """
    monkeypatch.setattr(merged_quality, "_available_ram_gb", lambda: 1024.0)
    for raw in ("nan", "NaN", "-nan"):
        monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", raw)
        budget = _quality_budget_gb()
        assert np.isfinite(budget) and budget == pytest.approx(_QUALITY_BUDGET_GB)
    assert "LUXAR_TILED_QUALITY_MAX_GB" in capsys.readouterr().out


def test_an_infinite_override_is_taken_at_face_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``inf`` is a coherent "score it whatever the size", unlike ``NaN``."""
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "inf")
    assert _quality_budget_gb() == float("inf")


def test_an_unparseable_budget_override_does_not_lose_the_fit(
    volume: np.ndarray, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A typo in the override falls back to the default instead of raising.

    This runs after every tile has been fitted, so a bad environment variable
    must not be what takes the run down.
    """
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "24 GiB")
    stats = _fit(volume).stats
    assert np.isfinite(stats["psnr_db"])
    assert "LUXAR_TILED_QUALITY_MAX_GB" in capsys.readouterr().out
