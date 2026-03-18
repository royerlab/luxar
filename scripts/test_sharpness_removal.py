#!/usr/bin/env python3
"""Smoke test: verify sharpness removal from GSplats and preservation for Points/Lines."""

import numpy as np
import tempfile
from pathlib import Path

from luxar import LuxarZarrCompiler, LuxarScene, Dimensions
from luxar.gsplats import GSplatData


def test_gsplats_without_sharpness():
    """GSplats should work without sharpness parameter."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "test",
                centers=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                amplitudes=np.array([1.0, 2.0], dtype=np.float32),
                cholesky_factors=np.array(
                    [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
                ),
                colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_gsplats("test")
        assert len(data["centers"]) == 2
        assert data.get("sharpness") is None, "GSplats should NOT have sharpness"
        print("PASS: GSplats round-trip without sharpness works")


def test_gsplats_from_data():
    """GSplatData should not require sharpnesses field."""
    result = GSplatData(
        centers=np.array([[0, 0, 0]], dtype=np.float32),
        amplitudes=np.array([1.0], dtype=np.float32),
        cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
    )
    assert not hasattr(result, "sharpnesses") or not hasattr(
        result.__dataclass_fields__, "sharpnesses"
    )
    print("PASS: GSplatData works without sharpnesses field")


def test_gsplats_add_no_sharpness_param():
    """add_gsplats() signature should not include sharpness."""
    import inspect
    from luxar.core.group import Group

    sig = inspect.signature(Group.add_gsplats)
    params = list(sig.parameters.keys())
    assert "sharpness" not in params, f"sharpness should not be in add_gsplats params: {params}"
    print("PASS: add_gsplats signature does not include sharpness")


def test_points_still_have_sharpness():
    """Points should still support sharpness parameter."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                positions=np.array([[1, 2, 3]], dtype=np.float32),
                radii=np.array([0.5], dtype=np.float32),
                sharpness=np.array([3.0], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_points("pts")
        assert data["sharpness"] is not None
        assert np.allclose(data["sharpness"], [3.0])
        print("PASS: Points with sharpness still work")


def test_lines_still_have_sharpness():
    """Lines should still support sharpness parameter."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                widths=np.array([0.1, 0.1], dtype=np.float32),
                sharpness=np.array([1.5, 1.5], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_lines("ln")
        assert data["sharpness"] is not None
        print("PASS: Lines with sharpness still work")


def test_gsplat_fitting_no_sharpness():
    """fit_gaussian_splats should work and not return sharpness."""
    from luxar.gsplats import fit_gaussian_splats

    # Simple 2D test image
    img = np.random.rand(16, 16).astype(np.float32) * 0.5
    img[5:10, 5:10] = 1.0  # Bright region

    result = fit_gaussian_splats(
        img, seeds=5, max_iterations=10, verbose=False
    )
    assert isinstance(result, GSplatData)
    assert not hasattr(result, "sharpnesses")
    assert result.centers.shape[1] == 2  # 2D
    print(f"PASS: fit_gaussian_splats returns GSplatData without sharpness ({len(result.centers)} splats)")


if __name__ == "__main__":
    test_gsplats_without_sharpness()
    test_gsplats_from_data()
    test_gsplats_add_no_sharpness_param()
    test_points_still_have_sharpness()
    test_lines_still_have_sharpness()
    test_gsplat_fitting_no_sharpness()
    print("\nAll smoke tests passed!")
