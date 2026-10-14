"""The relocation diagnostics must survive the tiled/partition merge (#2744).

``GSplatData.concatenate`` drops per-tile stats, so before #2734 the merged
partition root reported the relocation diagnostic keys as ABSENT. ``fit_tiled``
now aggregates them across the per-tile results. This is the regression test.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from luxar.gsplats.fit_tiled_gsplats import fit_tiled
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.tree import total_splats


def _structured_volume() -> np.ndarray:
    volume = np.zeros((40, 40), dtype=np.float32)
    volume[8:16, 8:16] = 1.0
    volume[24:32, 24:32] = 0.8
    volume[8:16, 24:32] = 0.6
    return volume


def test_partition_merge_carries_relocation_diagnostics() -> None:
    cfg = DynamicOpsConfig(step_every=5)
    node = fit_tiled(
        _structured_volume(),
        tile_size=24,
        overlap=8,
        partition=True,
        cull_retention=None,  # baseline without a closing cull
        enable_dynamic_ops=True,
        dynamic_config=cfg,
        seeds=120,
        n_iters=20,
        device="cpu",
        floor="none",
        verbose=False,
        napari_movie=False,
    )

    fit_stats = node.meta["fit_stats"]
    assert fit_stats.get("num_tiles", 0) > 1, "need a real multi-tile merge"

    # The #2744 drop: these keys must be PRESENT on the merged root, not absent.
    for key in (
        "splats_near_relocation_init_sigma_count",
        "splats_near_relocation_init_sigma_relocated_count",
        "splats_near_relocation_init_sigma_unrelocated_count",
        "dynamic_ops_relocation_events",
        "dynamic_ops_unique_splats_relocated",
        "relocation_init_sigma_vox",
        "relocation_reseed_sigma_vox_effective",
    ):
        assert key in fit_stats, f"merged root dropped {key!r} (#2744)"

    # The split adds up to the near-sigma count, summed across tiles.
    assert (
        fit_stats["splats_near_relocation_init_sigma_relocated_count"]
        + fit_stats["splats_near_relocation_init_sigma_unrelocated_count"]
        == fit_stats["splats_near_relocation_init_sigma_count"]
    )

    # The fraction was recomputed against the delivered splat count.
    delivered = total_splats(node)
    assert delivered > 0
    assert fit_stats["splats_near_relocation_init_sigma_fraction"] == pytest.approx(
        fit_stats["splats_near_relocation_init_sigma_count"] / delivered
    )
    assert fit_stats["relocation_reseed_sigma_vox_effective"] == 0.5


@pytest.mark.parametrize("partition", [False, True])
def test_culled_merge_keeps_delivered_population_and_history(partition: bool) -> None:
    result = fit_tiled(
        _structured_volume(),
        tile_size=24,
        overlap=8,
        partition=partition,
        cull_retention=0.9,
        enable_dynamic_ops=True,
        dynamic_config=DynamicOpsConfig(step_every=5),
        seeds=120,
        n_iters=20,
        device="cpu",
        floor="none",
        verbose=False,
        napari_movie=False,
    )
    stats = result.meta["fit_stats"] if partition else result.stats
    assert stats["dynamic_ops_relocation_events"] > 0
    assert stats["dynamic_ops_unique_splats_relocated"] > 0
    assert stats["relocation_reseed_sigma_vox_effective"] == 0.5
    delivered = total_splats(result) if partition else result.n_splats
    assert delivered > 0
    count = stats["splats_near_relocation_init_sigma_count"]
    assert 0 <= count <= delivered
    assert stats["splats_near_relocation_init_sigma_fraction"] == pytest.approx(
        count / delivered
    )
    assert (
        stats["splats_near_relocation_init_sigma_relocated_count"]
        + stats["splats_near_relocation_init_sigma_unrelocated_count"]
        == count
    )
