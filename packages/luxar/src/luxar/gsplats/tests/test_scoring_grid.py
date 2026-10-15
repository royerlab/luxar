"""Mapping stored splats back onto the voxel grid they are scored on."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.scoring_grid import (
    fit_downscale_factors,
    fraction_inside_grid,
    map_to_grid,
    parse_voxel_size,
    physical_voxel_size_stamp,
    stored_voxel_size,
)


def _voxel_model(n: int = 12, seed: int = 0) -> GSplatData:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.6, 1.4, (n, 1)).astype(np.float32)
    chol = np.concatenate(
        [a, rng.uniform(-0.2, 0.2, (n, 1)), a, rng.uniform(-0.2, 0.2, (n, 2)), a],
        axis=1,
    ).astype(np.float32)
    return GSplatData(
        centers=rng.uniform(2.0, 14.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=chol,
    )


def _to_physical(data: GSplatData, vs: tuple[float, ...]) -> GSplatData:
    """The fitter's own conversion (fitting/results.py), applied to a voxel model."""
    tril = np.concatenate([[vs[i]] * (i + 1) for i in range(3)])
    return GSplatData(
        centers=(data.centers * np.asarray(vs)).astype(np.float32),
        amplitudes=data.amplitudes,
        cholesky_factors=(data.cholesky_factors * tril).astype(np.float32),
    )


def test_map_to_grid_inverts_the_fitters_physical_conversion() -> None:
    vox = _voxel_model()
    vs = (2.0, 0.4, 0.25)
    back = map_to_grid(_to_physical(vox, vs), scale=vs)
    np.testing.assert_allclose(back.centers, vox.centers, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(
        back.cholesky_factors, vox.cholesky_factors, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(back.amplitudes, vox.amplitudes)


def test_map_to_grid_renders_identically_to_the_voxel_model() -> None:
    """The point of the mapping: the SAME volume, voxel for voxel."""
    vox = _voxel_model()
    vs = (2.0, 0.4, 0.25)
    shape = (16, 16, 16)
    expected = vox.render_to_volume(shape=shape, device="cpu")
    got = map_to_grid(_to_physical(vox, vs), scale=vs).render_to_volume(
        shape=shape, device="cpu"
    )
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_origin_translates_in_grid_units() -> None:
    vox = _voxel_model()
    shifted = map_to_grid(vox, origin=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(shifted.centers, vox.centers - [1.0, 2.0, 3.0])
    assert map_to_grid(vox) is vox


def test_stamp_only_for_physical_output() -> None:
    assert physical_voxel_size_stamp((2, 0.5, 0.5), "real", 3) == {
        "voxel_size": [2.0, 0.5, 0.5]
    }
    assert physical_voxel_size_stamp(0.5, "real", 2) == {"voxel_size": [0.5, 0.5]}
    assert physical_voxel_size_stamp((2, 0.5, 0.5), "voxel", 3) == {}
    assert physical_voxel_size_stamp(None, "real", 3) == {}
    # Unit spacing changes nothing, so it claims nothing.
    assert physical_voxel_size_stamp((1, 1, 1), "real", 3) == {}


def test_stored_voxel_size_reads_the_stamp_then_a_config_block() -> None:
    assert stored_voxel_size({"voxel_size": [2.0, 1.0, 1.0]}, 3) == (2.0, 1.0, 1.0)
    config = {"config": {"voxel_size": [3.0, 1.0, 1.0], "output_space": "real"}}
    assert stored_voxel_size(config, 3) == (3.0, 1.0, 1.0)
    voxel_out = {"config": {"voxel_size": [3.0, 1.0, 1.0], "output_space": "voxel"}}
    assert stored_voxel_size(voxel_out, 3) is None
    # Too long, non-positive, boolean: refused rather than half-applied.
    assert stored_voxel_size({"voxel_size": [2.0, 1.0, 1.0, 1.0]}, 3) is None
    assert stored_voxel_size({"voxel_size": [2.0, 0.0, 1.0]}, 3) is None
    assert stored_voxel_size({"voxel_size": True}, 3) is None
    assert stored_voxel_size(None, 3) is None


def test_parse_voxel_size() -> None:
    assert parse_voxel_size("0.5", 3) == (0.5, 0.5, 0.5)
    assert parse_voxel_size("2,0.4,0.4", 3) == (2.0, 0.4, 0.4)
    with pytest.raises(ValueError, match="expected 1 or 3"):
        parse_voxel_size("2,0.4", 3)
    with pytest.raises(ValueError):
        parse_voxel_size("a,b,c", 3)


@pytest.mark.parametrize(
    ("source", "fitted", "expected"),
    [
        ([64, 100, 101], [32, 50, 51], (2, 2, 2)),
        ([64, 99, 100], [16, 33, 25], (4, 3, 4)),
        ([10, 10, 10], [10, 10, 10], None),  # not downscaled
        ([10, 10], [5, 5], None),  # wrong rank for 3-D centers
        ([10, 10, 10], [20, 5, 5], None),  # fitted larger than source
    ],
)
def test_fit_downscale_factors(source, fitted, expected) -> None:
    stats = {"source_shape": source, "fitted_shape": fitted}
    assert fit_downscale_factors(stats, 3) == expected


def test_fraction_inside_grid() -> None:
    pts = np.array([[0, 0, 0], [9.4, 9.4, 9.4], [9.6, 0, 0], [-0.6, 0, 0]], float)
    assert fraction_inside_grid(pts, (10, 10, 10)) == pytest.approx(0.5)
    assert fraction_inside_grid(np.zeros((0, 3)), (10, 10, 10)) == 0.0


def test_physical_coordinates_false_means_the_spacing_was_not_applied() -> None:
    """CLI fits record `physical_coordinates`; False = provenance-only spacing."""
    stats = {"voxel_size": [2.0, 0.5, 0.5], "physical_coordinates": False}
    assert stored_voxel_size(stats, 3) is None
    stats["physical_coordinates"] = True
    assert stored_voxel_size(stats, 3) == (2.0, 0.5, 0.5)


def test_source_region_start() -> None:
    from luxar.gsplats.scoring_grid import source_region_start

    assert source_region_start({"source_region": [[4, 20], [0, 32], [8, 40]]}, 3) == (
        4,
        0,
        8,
    )
    assert source_region_start({"source_region": [[0, 5], [0, 5], [0, 5]]}, 3) is None
    assert source_region_start({}, 3) is None
    assert source_region_start({"source_region": [[4, 20], [0, 32]]}, 3) is None
    assert source_region_start({"source_region": [4, 0, 8]}, 3) is None


def test_a_spatial_stamp_on_a_stacked_store_pads_the_trailing_columns(
    capsys,
) -> None:
    """A physical batch-fit timelapse: 3 spatial values, 4 center columns."""
    assert stored_voxel_size({"voxel_size": [2.0, 0.5, 0.5]}, 4) == (
        2.0,
        0.5,
        0.5,
        1.0,
    )
    assert "stacked" in capsys.readouterr().out
    assert stored_voxel_size({"voxel_size": [2.0, 0.5, 0.5, 0.5, 9.0]}, 4) is None
    assert "WARNING" in capsys.readouterr().out


def test_agreed_voxel_size() -> None:
    from luxar.gsplats.scoring_grid import agreed_voxel_size

    vs = [2.0, 0.5, 0.5]
    assert agreed_voxel_size([{"voxel_size": vs}, {}, {"voxel_size": vs}]) == vs
    assert agreed_voxel_size([{"voxel_size": vs}, {"voxel_size": [1, 1, 1]}]) is None
    assert agreed_voxel_size([{}, None]) is None


def test_is_stacked_store() -> None:
    from luxar.gsplats.scoring_grid import is_stacked_store

    names = [{"name": n} for n in ("z", "y", "x", "t")]
    assert is_stacked_store({"source_shape": [2, 24, 40]}, 4, names)
    assert is_stacked_store({"source_shape": [2, 24, 40, 56]}, 4, names)
    assert is_stacked_store({"source_shape": [24, 40]}, 3)  # rank differs
    zyx = [{"name": n} for n in ("z", "y", "x")]
    assert not is_stacked_store({"source_shape": [24, 40, 56]}, 3, zyx)


def test_spatial_extent_fraction_sees_a_crammed_corner() -> None:
    """Sub-unit spacing with no stamp: everything inside, all in one corner."""
    from luxar.gsplats.scoring_grid import spatial_extent_fraction

    rng = np.random.default_rng(0)
    vox = rng.uniform(0, 63, (200, 3))
    assert spatial_extent_fraction(vox, (64, 64, 64)) > 0.9
    crammed = vox * 0.25
    assert fraction_inside_grid(crammed, (64, 64, 64)) == 1.0
    assert spatial_extent_fraction(crammed, (64, 64, 64)) < 0.3


def test_python_transform_keeps_the_stamp_only_for_translations() -> None:
    vox = _voxel_model()
    vox.stats.update(
        {
            "voxel_size": [2.0, 0.5, 0.5],
            "voxel_unit": "um",
            "physical_coordinates": True,
        }
    )
    moved = vox.translate(np.array([1.0, 2.0, 3.0]))
    assert moved.stats["voxel_size"] == [2.0, 0.5, 0.5]
    assert vox.center_at_centroid().stats["voxel_size"] == [2.0, 0.5, 0.5]
    affine = np.eye(4)
    affine[:3, 3] = [5.0, 0.0, 0.0]
    assert vox.transform(affine).stats["voxel_size"] == [2.0, 0.5, 0.5]

    scaled = vox.transform(np.diag([0.5, 2.0, 2.0]))
    for key in ("voxel_size", "voxel_unit", "physical_coordinates"):
        assert key not in scaled.stats
    assert vox.stats["voxel_size"] == [2.0, 0.5, 0.5]  # input untouched
