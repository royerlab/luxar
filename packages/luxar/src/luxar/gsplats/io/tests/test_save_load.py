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
            # v3.2 renames the lod selector attrs (coverage_fraction); leaf
            # layout is unchanged from v3.1.
            assert root.attrs["format_version"] == "3.2"
            # v3.1+: leaf at root, no "splats" group; Cholesky factors split.
            assert "splats" not in root
            assert "cholesky_factors_diag" in root
            assert "cholesky_factors" not in root
            assert root.attrs["type"] == "gsplats"
            assert root.attrs["n_splats"] == 100
            assert root.attrs["ndim"] == 3

    def test_save_stamps_content_hash(self) -> None:
        # The web viewer's persistent cache invalidates on the root
        # ``content_hash`` — without it, a regenerated file at the same URL
        # serves stale data indefinitely.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            root = zarr.open_group(str(path), mode="r")
            content_hash = root.attrs["content_hash"]
            assert isinstance(content_hash, str) and len(content_hash) > 0
            # The hash must also land in consolidated metadata (the viewer
            # reads .zmetadata for structure and .zattrs for validation).
            import json

            zmeta = json.loads((path / ".zmetadata").read_text())
            assert zmeta["metadata"][".zattrs"]["content_hash"] == content_hash

    def test_resave_changes_content_hash(self) -> None:
        # Identical data re-saved must yield a DIFFERENT hash (the timestamp
        # attr folds in) so the viewer cache invalidates on regeneration.
        splats = create_test_splats_3d(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.gsplats.zarr"
            path_b = Path(tmpdir) / "b.gsplats.zarr"
            save_gsplats(path=path_a, **splats, ordering="none")
            save_gsplats(path=path_b, **splats, ordering="none")
            hash_a = zarr.open_group(str(path_a), mode="r").attrs["content_hash"]
            hash_b = zarr.open_group(str(path_b), mode="r").attrs["content_hash"]
            assert hash_a != hash_b

    def test_streaming_partition_stamps_content_hash(self) -> None:
        # The streaming-partition writer path must stamp too (it is the merge
        # path for tiled fits — the largest, most re-generated artifacts).
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_partition_streaming
        from luxar.gsplats.tree import GSplatLeaf

        def make_leaf(seed: int) -> GSplatLeaf:
            rng = np.random.default_rng(seed)
            chol = np.zeros((20, 6), dtype=np.float32)
            chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(20, 3))
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.uniform(0, 50, (20, 3)).astype(np.float32),
                        amplitudes=rng.uniform(0.1, 1, (20,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_partition_streaming(
                path, lambda: iter([make_leaf(0), make_leaf(1)]), ordering="none"
            )
            root = zarr.open_group(str(path), mode="r")
            assert isinstance(root.attrs["content_hash"], str)

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

            # MEMORY mode stores COORDINATE centers as uint16 per-axis fixed-point
            # (linear_perchannel_u16). float16 is intentionally disabled (WebGL has no
            # native float16 and float16 on coordinates is a precision footgun);
            # coordinates never use uint8 (too coarse). Decodes back to float32.
            p_mem = Path(tmpdir) / "memory.gsplats.zarr"
            save_gsplats(
                path=p_mem, **splats, encoding_mode=EncodingMode.MEMORY, ordering="none"
            )
            enc = zarr.open_group(str(p_mem), mode="r")["centers"].attrs.get(
                "encoding", {}
            )
            assert enc["name"] == "linear_perchannel_u16"

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


class TestCholeskySplitRoundTrip:
    """v3.1 splits Cholesky factors into diag + offdiag on disk and recombines
    them on read. Verify the packed (N, k) form survives the round-trip across
    dimensionalities, encoding modes, and the uniform/broadcast case."""

    @staticmethod
    def _splats(n: int, d: int, rng: np.random.Generator) -> dict:
        k = d * (d + 1) // 2
        # Realistic Cholesky factors: POSITIVE diagonal (a real L has L[i,i] > 0),
        # signed off-diagonal. (Random unconstrained vectors would give negative
        # diagonals the log encoder legitimately clamps — not representative.)
        chol = rng.standard_normal((n, k)).astype(np.float32)
        diag_idx = np.cumsum(np.arange(1, d + 1)) - 1
        chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.5
        return {
            "centers": (rng.random((n, d)).astype(np.float32) * 10),
            "amplitudes": rng.random(n).astype(np.float32) * 2,
            "cholesky_factors": chol,
        }

    @staticmethod
    def _cov_relF_p95(ref: np.ndarray, got: np.ndarray, d: int) -> float:
        """p95 relative Frobenius error of Σ=LLᵀ between two packed sets."""
        from luxar.gsplats.utils.trils import unpack_tril

        Lr = unpack_tril(ref.astype(np.float64), d)
        Lg = unpack_tril(got.astype(np.float64), d)
        Sr = Lr @ Lr.transpose(0, 2, 1)
        Sg = Lg @ Lg.transpose(0, 2, 1)
        rel = np.linalg.norm(Sg - Sr, axis=(1, 2)) / (
            np.linalg.norm(Sr, axis=(1, 2)) + 1e-30
        )
        return float(np.percentile(rel, 95))

    # ndim=1 included: it is the degenerate case where the off-diagonal array is
    # intentionally omitted (k - d == 0), so it exercises a distinct write/read path.
    # Per-mode precision: PRECISION=float32 (exact); AUTO=uint8 with the encode-time
    # covariance certificate, whose escalation threshold (COV_CERT_RELF_P95_MAX)
    # makes the AUTO bound a hard invariant, not an observation; MEMORY=uint8
    # unconditionally (no certificate).
    _COV_P95_BOUND = {
        EncodingMode.PRECISION: 0.0,
        EncodingMode.AUTO: 0.05,
        EncodingMode.MEMORY: 0.1,
    }
    _DIAG_ENCODING = {
        EncodingMode.PRECISION: "float32",
        EncodingMode.AUTO: "log_perchannel_u8",
        EncodingMode.MEMORY: "log_perchannel_u8",
    }

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "mode", [EncodingMode.PRECISION, EncodingMode.AUTO, EncodingMode.MEMORY]
    )
    def test_roundtrip_dims_and_modes(self, ndim: int, mode: EncodingMode) -> None:
        rng = np.random.default_rng(ndim)
        splats = self._splats(64, ndim, rng)
        k = ndim * (ndim + 1) // 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.gsplats.zarr"
            save_gsplats(path=path, **splats, encoding_mode=mode, ordering="none")

            # On disk: split arrays, no single packed array. The off-diagonal
            # array is present iff there are off-diagonal elements (d > 1).
            root = zarr.open_group(str(path), mode="r")
            assert "cholesky_factors_diag" in root
            assert "cholesky_factors" not in root
            assert root["cholesky_factors_diag"].shape[1] == ndim
            assert (
                root["cholesky_factors_diag"].attrs["encoding"]["name"]
                == self._DIAG_ENCODING[mode]
            )
            # AUTO carries the covariance certificate as provenance; MEMORY and
            # PRECISION are unconditional tiers and must not.
            diag_enc = dict(root["cholesky_factors_diag"].attrs["encoding"])
            if mode == EncodingMode.AUTO:
                cert = diag_enc["certificate"]
                assert cert["metric"] == "cov_relf_p95"
                assert cert["tier"] == "u8"
                assert cert["value"] <= cert["threshold"]
            else:
                assert "certificate" not in diag_enc
            if k - ndim > 0:
                assert "cholesky_factors_offdiag" in root
                assert root["cholesky_factors_offdiag"].shape[1] == k - ndim
            else:
                assert "cholesky_factors_offdiag" not in root  # d == 1

            # Recombined on read into the packed (N, k) form.
            result = load_gsplats(path)
            assert result.cholesky_factors.shape == (64, k)
            if mode == EncodingMode.PRECISION:
                np.testing.assert_array_equal(
                    result.cholesky_factors, splats["cholesky_factors"]
                )
            else:
                cov_p95 = self._cov_relF_p95(
                    splats["cholesky_factors"], result.cholesky_factors, ndim
                )
                assert cov_p95 <= self._COV_P95_BOUND[mode], (
                    f"{mode} cov relF p95 {cov_p95:.2e} exceeds "
                    f"{self._COV_P95_BOUND[mode]:.0e}"
                )

    def test_roundtrip_uniform_cholesky(self) -> None:
        """Broadcast/uniform Cholesky (shape (1, k)) splits and recombines."""
        rng = np.random.default_rng(7)
        n, d = 50, 3
        k = d * (d + 1) // 2
        uniform = rng.standard_normal(k).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "u.gsplats.zarr"
            save_gsplats(
                path=path,
                centers=(rng.random((n, d)).astype(np.float32) * 10),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=np.tile(uniform, (n, 1)),
                ordering="none",
            )
            result = load_gsplats(path)
            for row in result.cholesky_factors:
                np.testing.assert_allclose(row, uniform, rtol=0, atol=1e-5)

    def test_roundtrip_2d_offdiag_single_column(self) -> None:
        """2D has exactly one off-diagonal element (k-d = 1)."""
        rng = np.random.default_rng(2)
        splats = self._splats(40, 2, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t2.gsplats.zarr"
            # default mode = AUTO → uint8 (certified; escalation not triggered here)
            save_gsplats(path=path, **splats, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            assert root["cholesky_factors_offdiag"].shape[1] == 1
            enc = dict(root["cholesky_factors_offdiag"].attrs["encoding"])
            assert enc["name"] == "signed_log_perchannel_u8"
            # the funnel routed through encode_cholesky_split → certificate present
            assert enc["certificate"]["tier"] == "u8"
            result = load_gsplats(path)
            assert (
                self._cov_relF_p95(
                    splats["cholesky_factors"], result.cholesky_factors, 2
                )
                <= 0.05
            )

    def test_corrupt_missing_offdiag_for_dgt1_raises(self) -> None:
        """A d>1 store with the diagonal but no off-diagonal array is corrupt;
        the reader must fail loud rather than silently drop off-diagonals."""
        import shutil

        rng = np.random.default_rng(3)
        splats = self._splats(32, 3, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "c.gsplats.zarr"
            save_gsplats(path=path, **splats, ordering="none")
            # Simulate a partial write: delete the off-diagonal array.
            shutil.rmtree(path / "cholesky_factors_offdiag")
            with pytest.raises(ValueError, match="offdiag.*missing|missing.*offdiag"):
                load_gsplats(path)


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
            # Both Cholesky halves share the same row-chunk size as centers.
            assert root["cholesky_factors_diag"].chunks[0] <= 50
            assert root["cholesky_factors_offdiag"].chunks[0] <= 50
            assert (
                root["cholesky_factors_diag"].chunks[0]
                == root["cholesky_factors_offdiag"].chunks[0]
            )

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


class TestCompressedLoadSecurity:
    """Path-safety of the compressed-archive loader (``_extract_compressed_zarr``)."""

    def test_targz_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """A tar.gz with a symlink escaping the extraction dir must be rejected
        BEFORE any file is written (CVE-2007-4559-style symlink escape).

        Pre-fix: the loader validated only ``member.name`` (which resolves inside
        the temp dir) and then ``extractall`` recreated the symlink, so a file
        member written "through" it landed outside the extraction dir.
        """
        import io
        import tarfile

        from luxar.gsplats.io.load_gsplats import _extract_compressed_zarr

        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "PWNED.txt"
        assert not target.exists()

        evil = tmp_path / "evil.gsplats.zarr.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            link = tarfile.TarInfo("d")  # symlink 'd' -> the outside dir
            link.type = tarfile.SYMTYPE
            link.linkname = str(outside)
            tar.addfile(link)
            payload = b"arbitrary write outside extraction dir"
            f = tarfile.TarInfo("d/PWNED.txt")  # writes through the symlink
            f.size = len(payload)
            tar.addfile(f, io.BytesIO(payload))

        with pytest.raises(ValueError, match="link|escape"):
            _extract_compressed_zarr(evil)
        # The decisive assertion: nothing was written outside the extraction dir.
        assert not target.exists(), "symlink escape wrote a file outside the temp dir"


def test_write_gsplats_tree_stamps_child_index_on_children() -> None:
    """Bare-root .gsplats.zarr trees stamp ``child_index`` (insertion order) on
    every ``child_<i>`` / ``part_<i>`` so the viewer restores napari-style order
    instead of zarr's alphabetical enumeration. Pre-fix the children carried no
    ``child_index``; a >=10-child group then reordered (child_10 before child_2)
    in the viewer's sibling sort.
    """
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    def _leaf(n: int, seed: int) -> GSplatLeaf:
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
        # kind=partition with 3 parts.
        ppath = Path(tmpdir) / "part.gsplats.zarr"
        write_gsplats_tree(
            ppath,
            GSplatPartition(children=[_leaf(20, 0), _leaf(20, 1), _leaf(20, 2)]),
            ordering="none",
        )
        proot = zarr.open_group(str(ppath), mode="r")
        for i in range(3):
            assert dict(proot[f"part_{i}"].attrs)["child_index"] == i

        # kind=lod: in-memory coarsest→finest, same as on disk (child_0 = coarsest),
        # and child_index must match the child_<i> numbering (0=coarsest..N=finest).
        lpath = Path(tmpdir) / "lod.gsplats.zarr"
        write_gsplats_tree(
            lpath,
            GSplatLodGroup(children=[_leaf(10, 4), _leaf(40, 3)]),
            ordering="none",
        )
        lroot = zarr.open_group(str(lpath), mode="r")
        for i in range(2):
            assert dict(lroot[f"child_{i}"].attrs)["child_index"] == i


def test_writer_derives_coverage_fractions_for_meta_less_lod_group() -> None:
    """#4: a meta-less (hand-built) kind=lod group gets viewport-relative
    ``coverage_fraction`` values from the writer fallback (``sqrt(N_i/N_finest)``:
    coarsest 0.0, finest 1.0). The builders normally stamp these into child meta —
    this exercises the fallback for a tree written without it."""
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    def _leaf(n: int, scale: float, seed: int) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = scale  # isotropic; radius ∝ scale
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol,
                )
            ]
        )

    # coarsest-first: 50 large (scale 4) then 800 small (scale 1). No authored meta.
    grp = GSplatLodGroup(children=[_leaf(50, 4.0, 0), _leaf(800, 1.0, 1)])
    assert "coverage_fraction" not in grp.children[0].meta
    assert "coverage_fraction" not in grp.children[1].meta

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "m.gsplats.zarr"
        write_gsplats_tree(path, grp, ordering="none")
        root = zarr.open_group(str(path), mode="r")
        assert root["child_0"].attrs["coverage_fraction"] == 0.0  # coarsest floor
        # sqrt(N_i/N_finest): coarsest-first counts [50, 800] → finest fills screen.
        assert root["child_1"].attrs["coverage_fraction"] == pytest.approx(1.0)
