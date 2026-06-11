"""Tests for v3.0 save and load (node-tree ``.gsplats.zarr``).

A single splat set is a leaf node at the file root (arrays directly under root);
an additive ladder writes ``additive_<i>/`` subgroups under the root leaf. Color
SDR/HDR is auto-detected (no explicit ``color_mode`` knob).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats import GSplatData
from luxar.gsplats.io import (
    format_gsplats_info,
    inspect_gsplats_zarr,
    load_gsplats,
    save_gsplats,
)


def create_test_splats_3d(n_splats: int = 100) -> dict:
    """Create test 3D Gaussian splats."""
    rng = np.random.default_rng(0)
    return {
        "centers": rng.random((n_splats, 3)).astype(np.float32) * 10,
        "amplitudes": rng.random(n_splats).astype(np.float32) * 2,
        "cholesky_factors": rng.random((n_splats, 6)).astype(np.float32),
    }


class TestSaveGsplats:
    """Test save_gsplats function (v3.0 leaf root)."""

    def test_save_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")

            assert path.exists()
            root = zarr.open_group(str(path), mode="r")
            assert root.attrs["format_type"] == "gsplats_zarr"
            assert root.attrs["format_version"] == "3.0"
            # v3.0: leaf at root, no "splats" group.
            assert "splats" not in root
            assert root.attrs["type"] == "gsplats"
            assert root.attrs["n_splats"] == 100
            assert root.attrs["ndim"] == 3

    def test_save_with_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(1).random((50, 3)).astype(np.float32)
            save_gsplats(
                path=path, **create_test_splats_3d(50), colors=colors, ordering="none"
            )
            root = zarr.open_group(str(path), mode="r")
            assert "colors" in root
            assert root.attrs["has_colors"] is True

    def test_save_with_morton_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            attrs = zarr.open_group(str(path), mode="r").attrs
            assert attrs["ordering"] == "morton"
            assert "ordering_min" in attrs
            assert "ordering_max" in attrs
            assert "ordering_bits_per_dim" in attrs

    def test_save_with_hilbert_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            try:
                save_gsplats(
                    path=path, **create_test_splats_3d(100), ordering="hilbert"
                )
            except ImportError:
                pytest.skip("hilbertcurve package not installed")
            assert zarr.open_group(str(path), mode="r").attrs["ordering"] == "hilbert"

    def test_save_with_encoding_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            splats = create_test_splats_3d(100)
            p_prec = Path(tmpdir) / "precision.gsplats.zarr"
            save_gsplats(
                path=p_prec,
                **splats,
                encoding_mode=EncodingMode.PRECISION,
                ordering="none",
            )
            assert zarr.open_group(str(p_prec), mode="r")["centers"].dtype == np.float32

            # MEMORY mode must keep COORDINATE centers float32 (float16 is
            # intentionally disabled — WebGL has no native float16 and float16
            # on coordinates is a precision footgun).
            p_mem = Path(tmpdir) / "memory.gsplats.zarr"
            save_gsplats(
                path=p_mem, **splats, encoding_mode=EncodingMode.MEMORY, ordering="none"
            )
            enc = zarr.open_group(str(p_mem), mode="r")["centers"].attrs.get(
                "encoding", {}
            )
            assert enc["name"] == "float32"

    def test_save_with_fitting_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={
                    "time_seconds": 45.3,
                    "iterations": 850,
                    "converged": True,
                    "fitter_name": "t",
                },
                fitting_config={"n_iters": 1000, "lr": 0.05},
                ordering="none",
            )
            root = zarr.open_group(str(path), mode="r")
            assert root["fitting"].attrs["time_seconds"] == 45.3
            assert "fitting/config" in root
            assert root["fitting/config"].attrs["n_iters"] == 1000

    def test_save_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            # Mismatched amplitudes shape
            with pytest.raises(ValueError, match="[Aa]mplitudes shape"):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=np.random.rand(50).astype(np.float32),
                    cholesky_factors=splats["cholesky_factors"],
                )
            # Mismatched cholesky shape
            with pytest.raises(ValueError):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=splats["amplitudes"],
                    cholesky_factors=np.random.rand(100, 3).astype(
                        np.float32
                    ),  # wrong k
                )


class TestLoadGsplats:
    def test_load_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            result = load_gsplats(path)
            assert result.centers.shape == (100, 3)
            assert result.amplitudes.shape == (100,)
            assert result.cholesky_factors.shape == (100, 6)
            assert result.centers.dtype == np.float32
            assert result.amplitudes.dtype in (np.float32, np.float16)

    def test_load_with_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.MEMORY,
                ordering="none",
            )
            result = load_gsplats(path)
            assert result.centers.dtype in (np.float32, np.float16)
            assert result.amplitudes.dtype in (np.float32, np.float16)

    def test_load_with_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={"time_seconds": 45.3, "iterations": 850},
                description="Test dataset",
                ordering="none",
            )
            result = load_gsplats(path, include_stats=True)
            assert result.stats["time_seconds"] == 45.3
            assert result.stats["description"] == "Test dataset"

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_gsplats("/nonexistent/path.gsplats.zarr")

    def test_load_invalid_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            root = zarr.open_group(str(path), mode="w")
            root.attrs["format_type"] = "wrong_format"
            with pytest.raises(ValueError, match="Invalid format_type"):
                load_gsplats(path)

    def test_load_partition_file_raises(self) -> None:
        """A standalone partition/nested tree has no flat GSplatData equivalent;
        load_gsplats() must raise clearly rather than silently mislead (TC-3)."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

        def _leaf(n, seed):
            rng = np.random.default_rng(seed)
            chol = np.zeros((n, 6), dtype=np.float32)
            chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
                        amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_gsplats_tree(
                path,
                GSplatPartition(children=[_leaf(20, 0), _leaf(20, 1)]),
                ordering="none",
            )
            with pytest.raises(ValueError, match="(?i)matrix|partition|tree"):
                load_gsplats(path)

    def test_load_rejects_legacy_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old.gsplats.zarr"
            root = zarr.open_group(str(path), mode="w")
            root.attrs["format_type"] = "gsplats_zarr"
            root.attrs["format_version"] = "2.0"
            with pytest.raises(ValueError, match="migrate-format"):
                load_gsplats(path)


class TestRoundTrip:
    def test_roundtrip_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(
                path=path,
                **splats,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert np.allclose(result.centers, splats["centers"])
            assert np.allclose(result.amplitudes, splats["amplitudes"])
            assert np.allclose(result.cholesky_factors, splats["cholesky_factors"])

    def test_roundtrip_with_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(path=path, **splats, ordering="morton")
            result = load_gsplats(path)
            assert len(result.centers) == len(splats["centers"])

    def test_roundtrip_with_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(
                path=path, **splats, encoding_mode=EncodingMode.MEMORY, ordering="none"
            )
            result = load_gsplats(path)
            assert np.allclose(result.centers, splats["centers"], atol=0.01)
            assert np.allclose(result.amplitudes, splats["amplitudes"], atol=0.05)


class TestGSplatDataMethods:
    def test_result_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            result = GSplatData(
                **splats, stats={"time_seconds": 10.5, "iterations": 500}
            )
            result.save(path, include_fitting_info=True)
            assert path.exists()
            root = zarr.open_group(str(path), mode="r")
            assert root["fitting"].attrs["time_seconds"] == 10.5

    def test_result_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            result = GSplatData.load(path)
            assert result.centers.shape == (50, 3)
            assert isinstance(result, GSplatData)

    def test_result_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            original = GSplatData(**create_test_splats_3d(100), stats={})
            original.save(path, ordering="none")
            loaded = GSplatData.load(path)
            assert np.allclose(loaded.centers, original.centers, atol=0.01)
            assert np.allclose(loaded.amplitudes, original.amplitudes, atol=0.05)


class TestColorRoundtrip:
    def test_roundtrip_with_sdr_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(1).random((100, 3)).astype(np.float32)
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                colors=colors,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert result.colors.shape == (100, 3)
            assert np.allclose(result.colors, colors, atol=0.01)

    def test_roundtrip_with_uint8_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(2).integers(
                0, 256, size=(50, 3), dtype=np.uint8
            )
            save_gsplats(
                path=path, **create_test_splats_3d(50), colors=colors, ordering="none"
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert result.colors.shape == (50, 3)
            assert result.colors.dtype == np.uint8
            assert np.array_equal(result.colors, colors)

    def test_roundtrip_with_hdr_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(3).random((75, 3)).astype(np.float32) * 5.0
            save_gsplats(
                path=path,
                **create_test_splats_3d(75),
                colors=colors,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert np.allclose(result.colors, colors, atol=0.05)

    def test_roundtrip_without_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            assert load_gsplats(path).colors is None

    def test_result_method_roundtrip_with_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(4).random((100, 3)).astype(np.float32)
            original = GSplatData(
                **create_test_splats_3d(100), colors=colors, stats={"test": "value"}
            )
            original.save(path, ordering="none")
            loaded = GSplatData.load(path)
            assert loaded.colors is not None
            assert loaded.colors.shape == colors.shape
            assert np.allclose(loaded.colors, colors, atol=0.01)


class TestInspectGsplats:
    def test_inspect_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            info = inspect_gsplats_zarr(path)
            assert info["n_splats"] == 100
            assert info["ndim"] == 3
            assert info["ordering"] == "morton"
            assert info["has_colors"] is False

    def test_inspect_with_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={"time_seconds": 45.3, "iterations": 850},
                ordering="none",
            )
            info = inspect_gsplats_zarr(path)
            assert info["fitting"]["time_seconds"] == 45.3

    def test_inspect_multi_level_pyramid_reports_finest_stats(self) -> None:
        # For a kind=lod pyramid, inspect must descend into the FINEST leaf so
        # its headline stats match the data-model default (finest, index 0) and
        # the `gsplat info` CLI (which loads via GSplatData). On disk child_0 is
        # the COARSEST, so reading default_level would report the wrong (coarsest)
        # count — the data-default vs viewer-hint conflation.
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        def _level(n: int, seed: int) -> SubstitutiveLevel:
            rng = np.random.RandomState(seed)
            chol = np.zeros((n, 6), dtype=np.float32)
            chol[:, [0, 3, 5]] = 1.0
            return SubstitutiveLevel(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.rand(n, 3).astype(np.float32) * 50,
                        amplitudes=np.ones(n, dtype=np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pyr.gsplats.zarr"
            # finest=100 at index 0, coarsest=10 at index 2
            data = GSplatData.from_substitutive_levels(
                [_level(100, 0), _level(30, 1), _level(10, 2)]
            )
            data.save(path, ordering="none")

            info = inspect_gsplats_zarr(path)
            assert info["kind"] == "lod"
            assert info["n_substitutive"] == 3
            assert info["default_lod_level"] == 0  # on-disk viewer hint = coarsest
            # Headline n_splats is the FINEST level, matching GSplatData.load.
            assert info["n_splats"] == 100
            assert info["n_splats"] == GSplatData.load(path).n_splats

    def test_inspect_format_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            try:
                save_gsplats(
                    path=path,
                    **create_test_splats_3d(100),
                    fitting_info={
                        "time_seconds": 45.3,
                        "iterations": 850,
                        "converged": True,
                    },
                    ordering="hilbert",
                )
            except ImportError:
                pytest.skip("hilbertcurve package not installed")
            formatted = format_gsplats_info(inspect_gsplats_zarr(path))
            assert "100" in formatted
            assert "3D" in formatted
            assert "hilbert" in formatted
            assert "45.3" in formatted


class TestCompression:
    """Blosc compression + chunk sizing (v3.0 leaf-root paths)."""

    def test_default_compression_applied(self):
        from numcodecs import Blosc

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100))
            centers = zarr.open(str(path), "r")["centers"]
            assert isinstance(centers.compressor, Blosc)
            assert centers.compressor.cname == "zstd"

    def test_compression_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, compressor=None, **create_test_splats_3d(100))
            assert zarr.open(str(path), "r")["centers"].compressor is None

    def test_chunk_capping_small_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50))
            root = zarr.open(str(path), "r")
            assert root["centers"].chunks[0] <= 50
            assert root["amplitudes"].chunks[0] <= 50
            assert root["cholesky_factors"].chunks[0] <= 50

    def test_gsplatdata_save_default_compression(self):
        from numcodecs import Blosc

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            GSplatData(**create_test_splats_3d(100)).save(path)
            assert isinstance(zarr.open(str(path), "r")["centers"].compressor, Blosc)

    def test_multi_lod_compression(self):
        from numcodecs import Blosc

        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=rng.standard_normal((n, 6)).astype(np.float32),
            )
            for n in (50, 80)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            GSplatData(additive_sublods=lods).save(path)
            root = zarr.open(str(path), "r")
            # Additive ladder → additive_<i>/ subgroups under the leaf root.
            assert isinstance(root["additive_0/centers"].compressor, Blosc)
            assert isinstance(root["additive_1/centers"].compressor, Blosc)
            assert root["additive_0/centers"].chunks[0] <= 50
            assert root["additive_1/centers"].chunks[0] <= 80

    def test_roundtrip_with_compression(self):
        rng = np.random.default_rng(42)
        splats = {
            "centers": rng.random((200, 3)).astype(np.float32) * 10,
            "amplitudes": rng.random(200).astype(np.float32) * 2,
            "cholesky_factors": rng.random((200, 6)).astype(np.float32),
        }
        g = GSplatData(**splats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            np.testing.assert_allclose(g.centers, g2.centers, atol=1e-6)
            np.testing.assert_allclose(g.amplitudes, g2.amplitudes, atol=1e-6)
            np.testing.assert_allclose(
                g.cholesky_factors, g2.cholesky_factors, atol=1e-6
            )


class TestTruncationRadiusRoundtrip:
    def test_default_truncation_radius(self):
        g = GSplatData(**create_test_splats_3d(50))
        assert g.truncation_radius == 3.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            assert GSplatData.load(path).truncation_radius == 3.0

    def test_custom_truncation_radius_single_lod(self):
        g = GSplatData(**create_test_splats_3d(50), truncation_radius=2.75)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            assert GSplatData.load(path).truncation_radius == 2.75

    def test_custom_truncation_radius_multi_lod(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=rng.standard_normal((n, 6)).astype(np.float32),
                truncation_radius=2.5,
            )
            for n in (30, 50)
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 2.5
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 2.5

    def test_per_level_truncation_radius_roundtrip(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=rng.standard_normal((n, 6)).astype(np.float32),
                truncation_radius=tr,
            )
            for n, tr in ((30, 2.5), (50, 4.0))
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 4.0

    def test_level_stats_read_without_include_stats(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        rng = np.random.default_rng(0)

        def _lod(n: int) -> AdditiveSubLOD:
            return AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=rng.standard_normal((n, 6)).astype(np.float32),
            )

        levels = [
            SubstitutiveLevel(
                additive_sublods=[_lod(8)],
                compression_factor=1,
                stats={"n_splats_total": 8},
            ),
            SubstitutiveLevel(
                additive_sublods=[_lod(2)],
                compression_factor=4,
                level_index=1,
                stats={"n_splats_total": 2},
            ),
        ]
        g = GSplatData.from_substitutive_levels(levels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)  # default: include_stats not set
            assert g2.substitutive_levels[1].stats.get("n_splats_total") == 2

    def test_truncation_radius_in_zarr_metadata(self):
        g = GSplatData(**create_test_splats_3d(50), truncation_radius=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            # v3.0: single leaf at root → truncation_radius on the root attrs.
            assert zarr.open(str(path), "r").attrs["truncation_radius"] == 2.0

    def test_backward_compat_missing_truncation_radius(self):
        g = GSplatData(**create_test_splats_3d(50))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            root = zarr.open(str(path), "r+")
            attrs = dict(root.attrs)
            del attrs["truncation_radius"]
            root.attrs.put(attrs)
            assert GSplatData.load(path).truncation_radius == 3.0


def test_save_explicit_none_compressor_disables_compression():
    """GSplatData.save(compressor=None) must write UNCOMPRESSED arrays so a
    cross-language (zarrita) reader can decode them. A plain None default used
    to be coerced to Blosc — making uncompressed output impossible and producing
    blosc-bitshuffle fixtures zarrita can't read (review full-suite finding)."""
    n = 16
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=np.random.rand(n, 3).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )
    with tempfile.TemporaryDirectory() as tmp:
        # Explicit None → no compression.
        raw = Path(tmp) / "raw.gsplats.zarr"
        data.save(raw, ordering="none", compressor=None)
        assert zarr.open_group(str(raw), mode="r")["centers"].compressor is None

        # Unspecified → default Blosc (compression still on by default).
        comp = Path(tmp) / "comp.gsplats.zarr"
        data.save(comp, ordering="none")
        assert zarr.open_group(str(comp), mode="r")["centers"].compressor is not None
