from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from luxar.demos.demo_storm_3d_microtubules import (
    LABEL_LINKAGE_SIGMA_NM,
    WIDEFIELD_PSF_SIGMA_NM,
    WIDEFIELD_VOXEL_SIZE_UM,
    create_storm_scene,
    extract_centers_and_amplitudes,
    parse_storm_localizations,
    rasterize_widefield_volume,
)
from luxar.gsplats.gsplat_data import GSplatData


def test_parser_converts_precision_with_its_coordinate_axis(tmp_path: Path) -> None:
    csv_path = tmp_path / "localizations.csv"
    pd.DataFrame(
        {
            "x_pix": [1.0, 2.0],
            "y_pix": [3.0, 4.0],
            "z_nm": [500.0, 600.0],
            "crlb_x": [0.1, 0.2],
            "crlb_y": [0.3, 0.4],
            "crlb_z": [7.0, 11.0],
            "photons": [1000.0, 2000.0],
        }
    ).to_csv(csv_path, index=False)

    parsed = parse_storm_localizations(csv_path)

    np.testing.assert_allclose(parsed["x"], [106.0, 212.0])
    np.testing.assert_allclose(parsed["y"], [318.0, 424.0])
    np.testing.assert_allclose(parsed["z"], [500.0, 600.0])
    np.testing.assert_allclose(parsed["precision_x"], [10.6, 21.2])
    np.testing.assert_allclose(parsed["precision_y"], [31.8, 42.4])
    np.testing.assert_allclose(parsed["precision_z"], [7.0, 11.0])


def test_superresolution_sigma_combines_crlb_and_label_linkage() -> None:
    localizations = {
        "x": np.array([0.0, 1000.0]),
        "y": np.array([0.0, 2000.0]),
        "z": np.array([0.0, 3000.0]),
        "photons": np.array([1000.0, 2000.0]),
        "precision_x": np.array([0.0, 8.0]),
        "precision_y": np.array([15.0, 0.0]),
        "precision_z": np.array([0.0, 150.0]),
    }

    _, _, precision_um = extract_centers_and_amplitudes(localizations)

    assert precision_um is not None
    expected_nm = np.array(
        [
            [
                LABEL_LINKAGE_SIGMA_NM,
                np.hypot(15.0, LABEL_LINKAGE_SIGMA_NM),
                LABEL_LINKAGE_SIGMA_NM,
            ],
            [
                np.hypot(8.0, LABEL_LINKAGE_SIGMA_NM),
                LABEL_LINKAGE_SIGMA_NM,
                150.0,
            ],
        ]
    )
    np.testing.assert_allclose(precision_um * 1000.0, expected_nm, rtol=1e-6)
    assert np.all(precision_um * 1000.0 >= LABEL_LINKAGE_SIGMA_NM)


def test_widefield_raster_preserves_photons_and_applies_axial_psf() -> None:
    centers_um = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    photons = np.array([37.0], dtype=np.float32)

    volume, origin_zyx = rasterize_widefield_volume(centers_um, photons)

    assert volume.dtype == np.float32
    assert origin_zyx.shape == (3,)
    np.testing.assert_allclose(volume.sum(), photons.sum(), rtol=1e-5)

    coordinates = [
        origin_zyx[axis]
        + np.arange(volume.shape[axis], dtype=np.float64) * WIDEFIELD_VOXEL_SIZE_UM
        for axis in range(3)
    ]
    marginals = [
        volume.sum(axis=tuple(other for other in range(3) if other != axis))
        for axis in range(3)
    ]
    variances = []
    for coordinate, marginal in zip(coordinates, marginals):
        mean = np.average(coordinate, weights=marginal)
        variances.append(np.average((coordinate - mean) ** 2, weights=marginal))

    assert np.sqrt(variances[0]) > 1.8 * np.sqrt(variances[2])
    np.testing.assert_allclose(
        np.sqrt(variances[2]), WIDEFIELD_PSF_SIGMA_NM / 1000.0, rtol=0.15
    )


def test_widefield_raster_accumulates_localization_photon_weights() -> None:
    centers_um = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        dtype=np.float32,
    )
    photons = np.array([2.0, 5.0, 11.0], dtype=np.float32)

    volume, _ = rasterize_widefield_volume(centers_um, photons)

    np.testing.assert_allclose(volume.sum(), 18.0, rtol=1e-5)


def test_scene_keeps_fitted_and_measured_view_counts_independent(
    tmp_path: Path,
) -> None:
    widefield = GSplatData(
        centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        amplitudes=np.array([1.0], dtype=np.float32),
        cholesky_factors=np.array([[0.2, 0.0, 0.1, 0.0, 0.0, 0.1]], dtype=np.float32),
    )
    centers_um = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    amplitudes = np.array([0.5, 1.0], dtype=np.float32)
    precision_um = np.full((2, 3), 0.02, dtype=np.float32)
    output_path = tmp_path / "storm.luxar.zarr"

    create_storm_scene(
        centers_um,
        amplitudes,
        widefield,
        precision_um,
        output_path,
    )

    root = zarr.open_group(output_path, mode="r")
    assert root["widefield_fit"]["centers"].shape == (1, 4)
    assert root["superresolution_localizations"]["centers"].shape == (2, 4)
    widefield_bounds = root["widefield_fit"]["chunk_bounds"][:]
    superresolution_bounds = root["superresolution_localizations"]["chunk_bounds"][:]
    np.testing.assert_allclose(widefield_bounds[:, 0].mean(axis=1), 0.0)
    np.testing.assert_allclose(superresolution_bounds[:, 0].mean(axis=1), 1.0)
    assert widefield_bounds[:, 0, 1].max() < 0.5
    assert superresolution_bounds[:, 0, 0].min() > 0.5
    (create_storm_scene,)
