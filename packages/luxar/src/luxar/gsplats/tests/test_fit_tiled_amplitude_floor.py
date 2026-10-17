"""An opt-in tile amplitude floor bounds large tiled fits without dark seams."""

import numpy as np

from luxar.gsplats import fit_tiled


def test_tile_amplitude_floor_removes_negligible_splats_before_merge() -> None:
    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    kwargs = dict(
        tile_size=32,
        overlap=8,
        seeds=128,
        n_iters=20,
        device="cpu",
        verbose=False,
        cull_retention=None,
    )

    original = fit_tiled(image, **kwargs)
    trimmed = fit_tiled(image, tile_amplitude_floor=1e-5, **kwargs)

    near_zero = int(np.count_nonzero(original.amplitudes < 1e-5))
    assert near_zero > 0
    assert trimmed.n_splats == original.n_splats - near_zero
    assert np.all(trimmed.amplitudes >= 1e-5)
    reconstruction = trimmed.render_to_volume(shape=image.shape, device="cpu")
    seam_mse = np.mean((reconstruction[:, 24:32] - image[:, 24:32]) ** 2)
    neighbour_mse = np.mean((reconstruction[:, 16:24] - image[:, 16:24]) ** 2)
    assert seam_mse <= neighbour_mse
