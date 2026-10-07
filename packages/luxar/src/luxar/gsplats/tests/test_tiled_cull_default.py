"""The tiled-fit default preserves both Hann-tapered overlap contributions."""

import numpy as np
import pytest

from luxar.gsplats.fit_tiled_gsplats import fit_tiled
from luxar.gsplats.gsplat_data import GSplatData


@pytest.mark.parametrize("partition", [False, True])
def test_default_cull_preserves_tiled_overlap(
    partition: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    options = dict(
        tile_size=32,
        overlap=8,
        seeds=128,
        n_iters=20,
        device="cpu",
        output_space="voxel",
        floor="none",
        partition=partition,
        verbose=not partition,
    )

    default = fit_tiled(image, **options)
    if not partition:
        assert "retained 99.9% of amplitude" in capsys.readouterr().out
    unculled = fit_tiled(image, cull_retention=None, **options)

    def overlap_mse(result) -> float:
        splats = GSplatData.from_default_selection(result) if partition else result
        reconstruction = splats.render_to_volume(shape=image.shape, device="cpu")
        return float(np.mean((reconstruction[:, 24:32] - image[:, 24:32]) ** 2))

    # Historical CPU measurements: the old 0.95 default yielded MSEs of
    # 0.0051 flat and 0.0047 partitioned, versus 0.0019 unculled. The
    # tolerance below is the contract; exact MSEs may drift across CPUs.
    assert overlap_mse(default) <= overlap_mse(unculled) + 0.0005
    assert default.n_splats >= 0.9 * unculled.n_splats
