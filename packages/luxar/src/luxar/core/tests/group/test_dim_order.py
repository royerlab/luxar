"""Tests for dim_order dimension mapping on add_points / add_lines / add_gsplats."""

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.gsplats import GSplats
from luxar.gsplats.utils.trils import pack_tril, unpack_tril
from luxar.io.compiler import LuxarZarrCompiler


def _make_4d_dims() -> Dimensions:
    """Scene: [X, Y, Z, Time] — 3 displayed + 1 hidden."""
    return Dimensions(
        [
            Dimension("X", display=True, range=(0, 10)),
            Dimension("Y", display=True, range=(0, 10)),
            Dimension("Z", display=True, range=(0, 10)),
            Dimension("Time", display=False, discrete=True, range=(0, 5), step=1.0),
        ]
    )


class TestDimOrderPoints:
    """Test dim_order on add_points."""

    def test_3d_points_to_4d_scene(self, tmp_path) -> None:
        """3D point data added to 4D scene with dim_order."""
        output = tmp_path / "test.zarr"
        # Data: (N, 3) with columns [Z, Y, X]
        positions_3d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            pts = scene.add_points(
                "pts",
                positions_3d,
                dim_order=["Z", "Y", "X"],
                fill={"Time": 2.0},
            )
            assert pts.n_elements == 2

        # Verify the stored positions are 4D
        store = zarr.open(str(output), mode="r")
        stored = store["pts"]["positions"][:]
        assert stored.shape == (2, 4)
        # X column (scene dim 0) should have original Z values (data col 0 → "Z" → scene col 2)
        # Actually: dim_order=["Z","Y","X"] means data col 0→Z (scene idx 2), col 1→Y (idx 1), col 2→X (idx 0)
        # Scene order is [X, Y, Z, Time]
        # So: stored[:,0]=X=data[:,2], stored[:,1]=Y=data[:,1], stored[:,2]=Z=data[:,0], stored[:,3]=Time=2.0
        np.testing.assert_allclose(stored[:, 0], [3, 6])  # X
        np.testing.assert_allclose(stored[:, 1], [2, 5])  # Y
        np.testing.assert_allclose(stored[:, 2], [1, 4])  # Z
        np.testing.assert_allclose(stored[:, 3], [2, 2])  # Time (filled)

    def test_auto_extend_unmapped_dims(self, tmp_path) -> None:
        """Unmapped dims should be auto-added to extend_to_all."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            scene.add_points("pts", positions, dim_order=["X", "Y", "Z"])

        store = zarr.open(str(output), mode="r")
        assert store["pts"].attrs.get("extend_to_all") == ["Time"]

    def test_explicit_extend_to_all_overrides_auto(self, tmp_path) -> None:
        """If user sets extend_to_all explicitly, don't auto-extend."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            scene.add_points(
                "pts",
                positions,
                dim_order=["X", "Y", "Z"],
                extend_to_all=[],  # explicit: no extension
            )

        store = zarr.open(str(output), mode="r")
        # extend_to_all should NOT be set (empty list means no extension)
        assert "extend_to_all" not in store["pts"].attrs

    def test_2d_points_to_3d_scene(self, tmp_path) -> None:
        """2D data in a 3D scene with fill for Z."""
        output = tmp_path / "test.zarr"
        positions_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)

        dims = Dimensions.default_3d()
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "pts",
                positions_2d,
                dim_order=["x", "y"],
                fill={"z": 5.0},
            )

        store = zarr.open(str(output), mode="r")
        stored = store["pts"]["positions"][:]
        assert stored.shape == (2, 3)
        np.testing.assert_allclose(stored[:, 2], [5, 5])  # z filled

    def test_no_dim_order_unchanged(self, tmp_path) -> None:
        """Without dim_order, behavior is unchanged (positional mapping)."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", positions)
            assert pts.n_elements == 1


class TestDimOrderLines:
    """Test dim_order on add_lines."""

    def test_2d_lines_to_3d_scene(self, tmp_path) -> None:
        """2D line vertices in a 3D scene."""
        output = tmp_path / "test.zarr"
        vertices_2d = np.array([[0, 0], [1, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lines = scene.add_lines(
                "lines",
                vertices_2d,
                widths=0.1,
                dim_order=["x", "y"],
                fill={"z": 0.0},
            )
            assert lines.n_elements == 2


class TestDimOrderGSplats:
    """Test dim_order on add_gsplats with Cholesky embedding."""

    def test_3d_gsplats_to_4d_scene(self, tmp_path) -> None:
        """3D gsplats embedded into 4D scene with Cholesky transformation."""
        output = tmp_path / "test.zarr"

        # Create 3D splat data
        centers_3d = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        # Simple isotropic 3D Cholesky: L = diag(1, 1, 1)
        L_3d = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)
        cholesky_3d = pack_tril(L_3d)  # (1, 6)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            gsplats = scene.add_gsplats(
                "splats",
                centers_3d,
                amplitudes,
                cholesky_3d,
                dim_order=["Z", "Y", "X"],
                fill={"Time": 0.0},
                fill_sigma={"Time": 0.5},
            )
            assert gsplats.n_splats == 1

        # Verify stored data
        store = zarr.open(str(output), mode="r")
        stored_centers = store["splats"]["centers"][:]
        assert stored_centers.shape == (1, 4)

        stored_chol = store["splats"]["cholesky_factors"][:]
        assert stored_chol.shape == (1, 10)  # 4D: k = 4*5/2 = 10

        # Verify the embedded covariance
        L_4d = unpack_tril(stored_chol, 4)
        Sigma_4d = L_4d @ L_4d.transpose(0, 2, 1)

        # Original 3D isotropic covariance was I_3
        # Mapped: Z→scene[2], Y→scene[1], X→scene[0]
        # So the XYZ block at [0:3, 0:3] should be I_3 (permuted but still identity)
        np.testing.assert_allclose(Sigma_4d[0, :3, :3], np.eye(3), atol=1e-5)
        # Time dimension (idx 3) should have variance = 0.5^2 = 0.25
        np.testing.assert_allclose(Sigma_4d[0, 3, 3], 0.25, atol=1e-5)

    def test_uniform_1d_cholesky_embedded(self, tmp_path) -> None:
        """B5/[P5]: a *uniform* Cholesky (a single 1-D packed vector shared by
        every splat) takes the ``cholesky_factors.ndim == 1`` branch of
        ``apply_dim_order_cholesky`` (dim_order.py:90-96): reshape to (1, k),
        embed, reshape back. The 2-D per-row tests above never reach it."""
        output = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        amplitudes = np.array([1.0, 1.0], dtype=np.float32)
        # 1-D uniform packed Cholesky for an isotropic 3D Gaussian.
        uniform_chol = pack_tril(np.eye(3, dtype=np.float32).reshape(1, 3, 3)).reshape(
            -1
        )
        assert uniform_chol.ndim == 1  # precondition for the branch

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            gsplats = scene.add_gsplats(
                "splats",
                centers,
                amplitudes,
                uniform_chol,
                dim_order=["Z", "Y", "X"],
                fill={"Time": 0.0},
                fill_sigma={"Time": 0.5},
            )
            assert gsplats.n_splats == 2

        store = zarr.open(str(output), mode="r")
        stored_chol = store["splats"]["cholesky_factors"][:]
        L_4d = unpack_tril(
            stored_chol.reshape(1, -1) if stored_chol.ndim == 1 else stored_chol, 4
        )
        Sigma_4d = L_4d @ L_4d.transpose(0, 2, 1)
        # Permuted-identity XYZ block + the filled Time variance (0.5² = 0.25).
        np.testing.assert_allclose(Sigma_4d[0, :3, :3], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(Sigma_4d[0, 3, 3], 0.25, atol=1e-5)

    def test_gsplats_cholesky_reorder_preserves_covariance(self, tmp_path) -> None:
        """Anisotropic 3D splats: verify covariance is correctly permuted."""
        output = tmp_path / "test.zarr"

        # Create anisotropic 3D Cholesky
        L_3d = np.array([[[2, 0, 0], [0.5, 1.5, 0], [0.3, 0.2, 1]]], dtype=np.float32)
        Sigma_3d = L_3d @ L_3d.transpose(0, 2, 1)
        cholesky_3d = pack_tril(L_3d)

        centers = np.array([[5, 6, 7]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)

        # dim_order=["Z","Y","X"] means: data col 0→Z, col 1→Y, col 2→X
        # Scene dims: [X(0), Y(1), Z(2), Time(3)]
        # So: data col 2→X→scene 0, data col 1→Y→scene 1, data col 0→Z→scene 2
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            scene.add_gsplats(
                "splats",
                centers,
                amplitudes,
                cholesky_3d,
                dim_order=["Z", "Y", "X"],
                fill={"Time": 0.0},
            )

        store = zarr.open(str(output), mode="r")
        stored_chol = store["splats"]["cholesky_factors"][:]
        L_4d = unpack_tril(stored_chol, 4)
        Sigma_4d = L_4d @ L_4d.transpose(0, 2, 1)

        # The XYZ block should be the permuted Sigma_3d
        # perm: data[0]→Z(2), data[1]→Y(1), data[2]→X(0) in scene
        # So Sigma_4d[scene_i, scene_j] = Sigma_3d[data_i, data_j] where
        # X(scene 0) = data 2, Y(scene 1) = data 1, Z(scene 2) = data 0
        perm = [
            2,
            1,
            0,
        ]  # scene_to_data: scene[0]→data[2], scene[1]→data[1], scene[2]→data[0]
        for si in range(3):
            for sj in range(3):
                np.testing.assert_allclose(
                    Sigma_4d[0, si, sj],
                    Sigma_3d[0, perm[si], perm[sj]],
                    atol=1e-5,
                    err_msg=f"Sigma mismatch at scene[{si},{sj}]",
                )

    def test_gsplats_from_data_with_dim_order(self, tmp_path) -> None:
        """add_gsplats_from_data passes through dim_order."""
        from luxar.gsplats.gsplat_data import GSplatData

        output = tmp_path / "test.zarr"
        result = GSplatData(
            centers=np.array([[1, 2, 3]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=pack_tril(np.eye(3, dtype=np.float32).reshape(1, 3, 3)),
        )

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            gsplats = scene.add_gsplats_from_data(
                "splats",
                result,
                dim_order=["Z", "Y", "X"],
                fill={"Time": 3.0},
            )
            # Single-substitutive path returns GSplats, not LODGroup.
            assert isinstance(gsplats, GSplats)
            assert gsplats.n_splats == 1

        store = zarr.open(str(output), mode="r")
        assert store["splats"]["centers"][:].shape == (1, 4)
        assert store["splats"]["cholesky_factors"][:].shape == (1, 10)


class TestDimOrderValidation:
    """Test dim_order validation errors."""

    def test_unknown_dimension_name(self, tmp_path) -> None:
        """dim_order with a name not in scene dimensions."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="not found in scene dimensions"):
                scene.add_points("pts", positions, dim_order=["x", "BOGUS"])

    def test_duplicate_dim_names(self, tmp_path) -> None:
        """dim_order with repeated dimension name."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="duplicate"):
                scene.add_points("pts", positions, dim_order=["x", "x"])

    def test_wrong_length_too_few(self, tmp_path) -> None:
        """dim_order shorter than data columns."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(
                ValueError, match="dim_order has 2 names but data has 3"
            ):
                scene.add_points("pts", positions, dim_order=["x", "y"])

    def test_wrong_length_too_many(self, tmp_path) -> None:
        """dim_order longer than data columns."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(
                ValueError, match="dim_order has 3 names but data has 2"
            ):
                scene.add_points("pts", positions, dim_order=["x", "y", "z"])

    def test_fill_key_also_in_dim_order(self, tmp_path) -> None:
        """fill key that conflicts with a dim_order name."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="already in dim_order"):
                scene.add_points(
                    "pts",
                    positions,
                    dim_order=["x", "y"],
                    fill={"x": 5.0},
                )

    def test_fill_key_unknown_dimension(self, tmp_path) -> None:
        """fill key with a name not in scene dimensions."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="fill key.*not found"):
                scene.add_points(
                    "pts",
                    positions,
                    dim_order=["x", "y"],
                    fill={"BOGUS": 0.0},
                )

    def test_dim_order_more_names_than_scene_dims(self, tmp_path) -> None:
        """dim_order cannot have more names than scene dimensions (implicitly
        caught by data having more columns than scene dims after transform)."""
        output = tmp_path / "test.zarr"
        # 4 data columns, 3D scene, dim_order with 4 names
        positions = np.array([[1, 2, 3, 4]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # "w" doesn't exist in 3D scene
            with pytest.raises(ValueError, match="not found in scene dimensions"):
                scene.add_points("pts", positions, dim_order=["x", "y", "z", "w"])

    def test_pure_reorder_no_padding(self, tmp_path) -> None:
        """dim_order covering all scene dims = pure reorder, no padding."""
        output = tmp_path / "test.zarr"
        # Data columns are [z, y, x], scene dims are [x, y, z]
        positions = np.array([[10, 20, 30]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", positions, dim_order=["z", "y", "x"])
            assert pts.n_elements == 1

        store = zarr.open(str(output), mode="r")
        stored = store["pts"]["positions"][:]
        # x=30, y=20, z=10
        np.testing.assert_allclose(stored[0], [30, 20, 10])

    def test_validation_on_add_lines(self, tmp_path) -> None:
        """dim_order validation also works on add_lines."""
        output = tmp_path / "test.zarr"
        vertices = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="not found in scene dimensions"):
                scene.add_lines("lines", vertices, widths=0.1, dim_order=["x", "NOPE"])

    def test_validation_on_add_gsplats(self, tmp_path) -> None:
        """dim_order validation also works on add_gsplats."""
        output = tmp_path / "test.zarr"
        centers = np.array([[1, 2]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="not found in scene dimensions"):
                scene.add_gsplats(
                    "splats",
                    centers,
                    amplitudes,
                    cholesky,
                    dim_order=["x", "NOPE"],
                )

    def test_empty_dim_order_with_mismatched_data(self, tmp_path) -> None:
        """Empty dim_order but data has columns — length mismatch."""
        output = tmp_path / "test.zarr"
        positions = np.array([[1, 2]], dtype=np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(
                ValueError, match="dim_order has 0 names but data has 2"
            ):
                scene.add_points("pts", positions, dim_order=[])

    def test_fill_sigma_unknown_dimension(self, tmp_path) -> None:
        """fill_sigma with a name not in scene dimensions."""
        output = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        L = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        cholesky = pack_tril(L)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            with pytest.raises(ValueError, match="fill_sigma key.*not found"):
                scene.add_gsplats(
                    "splats",
                    centers,
                    amplitudes,
                    cholesky,
                    dim_order=["X", "Y", "Z"],
                    fill_sigma={"BOGUS": 0.5},
                )

    def test_fill_sigma_on_mapped_dimension(self, tmp_path) -> None:
        """fill_sigma key that is already in dim_order (a mapped dimension)."""
        output = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        L = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        cholesky = pack_tril(L)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=_make_4d_dims())
            with pytest.raises(
                ValueError, match="fill_sigma key.*already in dim_order"
            ):
                scene.add_gsplats(
                    "splats",
                    centers,
                    amplitudes,
                    cholesky,
                    dim_order=["X", "Y", "Z"],
                    fill_sigma={"X": 0.5},  # X is mapped, not unmapped
                )
