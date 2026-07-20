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
from luxar.typing_utils._format_contract import GSPLATS_FORMAT_VERSION


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
            assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
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

    @pytest.mark.parametrize("mode", [EncodingMode.PRECISION, EncodingMode.AUTO])
    def test_rgba_colors_round_trip(self, mode: EncodingMode) -> None:
        # RGBA colors (per-splat opacity in the 4th column) survive
        # write→read through both the SDR (rgb_uint8/AUTO) and lossless
        # (PRECISION/float32) encoders. Alpha ∈ [0, 1] must never trip HDR.
        rng = np.random.default_rng(3)
        colors = rng.random((50, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgba.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                colors=colors,
                encoding_mode=mode,
                ordering="none",
            )
            root = zarr.open_group(str(path), mode="r")
            assert root.attrs["has_colors"] is True
            loaded = load_gsplats(path)
            assert loaded.colors is not None
            assert loaded.colors.shape == (50, 4)
            atol = 1e-6 if mode == EncodingMode.PRECISION else 2.0 / 255.0
            assert np.allclose(loaded.colors, colors, atol=atol)

    def test_rgba_hdr_rgb_keeps_alpha_bounded(self) -> None:
        # HDR RGB (values > 1) routes through geolog per-channel; the alpha
        # column rides along and must round-trip within [0, 1].
        rng = np.random.default_rng(4)
        colors = np.empty((40, 4), dtype=np.float32)
        colors[:, :3] = rng.random((40, 3)).astype(np.float32) * 8.0  # HDR RGB
        colors[:, 3] = rng.random(40).astype(np.float32)  # opacity in [0, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgba_hdr.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(40),
                colors=colors,
                encoding_mode=EncodingMode.PRECISION,
                ordering="none",
            )
            loaded = load_gsplats(path)
            assert loaded.colors is not None and loaded.colors.shape == (40, 4)
            assert np.allclose(loaded.colors, colors, atol=1e-5)

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
    """Path-safety of the shared compressed-archive extractor
    (``luxar.gsplats.io._archive.extract_compressed_zarr``), exercised through
    every entry point that consumes it (loader + format migrator)."""

    def _make_symlink_bomb(self, dest_dir: Path, outside: Path) -> Path:
        """Build a malicious .tar.gz: a symlink 'd' -> ``outside`` then a file
        'd/PWNED.txt' written through it (CVE-2007-4559-style symlink escape)."""
        import io
        import tarfile

        evil = dest_dir / "evil.gsplats.zarr.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            link = tarfile.TarInfo("d")
            link.type = tarfile.SYMTYPE
            link.linkname = str(outside)
            tar.addfile(link)
            payload = b"arbitrary write outside extraction dir"
            f = tarfile.TarInfo("d/PWNED.txt")
            f.size = len(payload)
            tar.addfile(f, io.BytesIO(payload))
        return evil

    def test_targz_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """The shared extractor must reject a symlink-escape tar.gz BEFORE any
        file is written, and leave no temp dir behind."""
        from luxar.gsplats.io._archive import extract_compressed_zarr

        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "PWNED.txt"

        evil = self._make_symlink_bomb(tmp_path, outside)
        with pytest.raises(ValueError, match="link|escape|device"):
            extract_compressed_zarr(evil)
        assert not target.exists(), "symlink escape wrote a file outside the temp dir"

    def test_targz_hardlink_rejected(self, tmp_path: Path) -> None:
        """Hardlink members are rejected too (not just symlinks)."""
        import io
        import tarfile

        from luxar.gsplats.io._archive import extract_compressed_zarr

        evil = tmp_path / "hard.gsplats.zarr.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            payload = b"real"
            real = tarfile.TarInfo("real.txt")
            real.size = len(payload)
            tar.addfile(real, io.BytesIO(payload))
            link = tarfile.TarInfo("link.txt")
            link.type = tarfile.LNKTYPE
            link.linkname = "real.txt"
            tar.addfile(link)

        with pytest.raises(ValueError, match="link|device"):
            extract_compressed_zarr(evil)

    def test_targz_member_count_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive with too many members is rejected (archive-bomb guard)."""
        import io
        import tarfile

        from luxar.gsplats.io import _archive

        monkeypatch.setattr(_archive, "_MAX_MEMBERS", 3)
        bomb = tmp_path / "bomb.gsplats.zarr.tar.gz"
        with tarfile.open(bomb, "w:gz") as tar:
            for i in range(5):
                info = tarfile.TarInfo(f"f{i}.txt")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))

        with pytest.raises(ValueError, match="members"):
            _archive.extract_compressed_zarr(bomb)

    def test_migrate_path_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """The migration entry point uses the same safe extractor: a malicious
        archive passed to ``detect_legacy_format`` must not escape the temp dir."""
        from luxar.gsplats.io.migrate import detect_legacy_format

        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "PWNED.txt"

        evil = self._make_symlink_bomb(tmp_path, outside)
        with pytest.raises(ValueError, match="link|escape|device"):
            detect_legacy_format(evil)
        assert not target.exists(), "migrate path allowed a symlink escape"


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


class TestBarrierAwareOrdering:
    """End-to-end: a 4D leaf with a time barrier writes single-timepoint chunks
    (the fix for per-timepoint viewer-load locality)."""

    @staticmethod
    def _make_4d_leaf(n: int, n_tps: int, seed: int) -> "GSplatData":
        """Random 4D splats spread over n_tps integer timepoints in column 3."""
        rng = np.random.default_rng(seed)
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100.0
        centers[:, 3] = rng.integers(0, n_tps, size=n).astype(np.float32)
        chol = np.zeros((n, 10), dtype=np.float32)
        diag_idx = [d * (d + 1) // 2 + d for d in range(4)]
        chol[:, diag_idx] = 2.0  # isotropic σ=2 in all 4 dims
        amps = (rng.random(n) + 0.5).astype(np.float32)
        return GSplatData(centers=centers, amplitudes=amps, cholesky_factors=chol)

    @staticmethod
    def _finest_chunk_bounds(path: Path) -> np.ndarray:
        """chunk_bounds of the (only) leaf's finest splat set."""
        root = zarr.open_group(str(path), mode="r")
        # single-leaf root: chunk_bounds directly under root
        if "chunk_bounds" in root:
            return np.asarray(root["chunk_bounds"])
        raise AssertionError("no chunk_bounds at leaf root")

    def test_explicit_barrier_yields_single_timepoint_chunks(self) -> None:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        # Many splats per timepoint (12k / 3 = 4k ≫ chunk_size ~1024) so a chunk
        # spans at most 2 adjacent timepoints (a boundary chunk), never all 3 —
        # this is the regime the real timelapse is in (~2.5M splats/timepoint).
        data = self._make_4d_leaf(12000, n_tps=3, seed=1)
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            bpath = Path(tmpdir) / "barrier.gsplats.zarr"
            npath = Path(tmpdir) / "nobarrier.gsplats.zarr"
            write_gsplats_tree(bpath, leaf, barrier_dims=[3])
            write_gsplats_tree(npath, leaf, barrier_dims=[])  # pure spatial

            bt = (
                self._finest_chunk_bounds(bpath)[:, 3, 1]
                - self._finest_chunk_bounds(bpath)[:, 3, 0]
            )
            nt = (
                self._finest_chunk_bounds(npath)[:, 3, 1]
                - self._finest_chunk_bounds(npath)[:, 3, 0]
            )

            # WITH barrier: most chunks single-timepoint. Since the write-side
            # padding shrank from ±0.5 step to the tiny float-boundary epsilon
            # (_BARRIER_BOUND_EPS = 1e-3), a single-timepoint chunk's extent is
            # ~2*eps (~0.002, no longer ~1.0), and a boundary chunk spanning 2
            # timepoints is ~1.0 + 2*eps. Never the whole-span-plus-sigma smear
            # the no-barrier ordering produces. The tight thresholds FAIL under
            # the legacy +/-0.5 padding — they pin the over-fetch fix's write side.
            assert np.median(bt) <= 0.1
            assert np.all(bt <= 1.0 + 0.1)
            # WITHOUT barrier: chunks smear across timepoints (σ-expanded too),
            # so the barrier version is decisively tighter — the fix's payoff.
            assert np.median(nt) > np.median(bt)
            assert nt.max() > bt.max()

            # ordering attrs advertise the barrier (mirrors Points/Lines).
            root = zarr.open_group(str(bpath), mode="r")
            assert list(root.attrs["slice_dims"]) == [3]
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2]

    def test_barrier_derived_from_coarsen_dims_provenance(self) -> None:
        """When barrier_dims is not passed, it is derived from
        pipeline_info['coarsen_dims'] (barrier = complement)."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        data = self._make_4d_leaf(12000, n_tps=3, seed=2)
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prov.gsplats.zarr"
            # coarsen spatial dims 0,1,2 → barrier = [3] (time). No barrier_dims arg.
            write_gsplats_tree(path, leaf, pipeline_info={"coarsen_dims": [0, 1, 2]})
            bt = (
                self._finest_chunk_bounds(path)[:, 3, 1]
                - self._finest_chunk_bounds(path)[:, 3, 0]
            )
            assert np.median(bt) <= 1.5  # barrier honored via provenance
            root = zarr.open_group(str(path), mode="r")
            assert list(root.attrs["slice_dims"]) == [3]

    def test_explicit_full_coarsen_list_yields_no_barrier(self) -> None:
        """A DIRECT caller passing an explicit full coarsen_dims list (barrier =
        empty complement) → pure spatial, NOT auto-detected [3].

        NOTE: this is the direct-caller contract for _barrier_from_coarsen_dims's
        empty-complement branch. The LOD reducer itself never persists a full
        list — _normalise_coarsen_dims collapses coarsen-all to `coarsen_dims=
        None`, which is indistinguishable from 'no provenance' and correctly
        falls through to auto-detect (a degenerate, rarely-used config)."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        data = self._make_4d_leaf(12000, n_tps=3, seed=4)  # integer time axis
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coarsen_all.gsplats.zarr"
            write_gsplats_tree(path, leaf, pipeline_info={"coarsen_dims": [0, 1, 2, 3]})
            root = zarr.open_group(str(path), mode="r")
            assert list(root.attrs["slice_dims"]) == []
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2, 3]

    def test_streaming_partition_explicit_barrier_on_sparse_time(self) -> None:
        """REGRESSION (deep-double-check, findings 1/6): the batch-merge streaming
        partition path passes an EXPLICIT barrier (the stacked-time axis) rather
        than relying on auto-detect, which false-negatives on sparse tiles (few
        splats per timepoint trips the n_unique*4<=n guard). Verify each part's
        finest chunk bounds are time-tight when barrier_dims is passed, and that
        auto-detect alone would NOT barrier this sparse part."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_partition_streaming
        from luxar.gsplats.tree import GSplatLeaf
        from luxar.io.ordering import detect_barrier_dims

        def make_sparse_4d_leaf(seed: int) -> GSplatLeaf:
            rng = np.random.default_rng(seed)
            n = 40  # sparse: 40 splats over 14 timepoints (40 < 14*4=56)
            centers = np.empty((n, 4), dtype=np.float32)
            centers[:, :3] = rng.uniform(0, 50, (n, 3))
            centers[:, 3] = rng.integers(0, 14, size=n).astype(np.float32)
            chol = np.zeros((n, 10), dtype=np.float32)
            chol[:, [0, 2, 5, 9]] = 2.0
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        # Auto-detect MISSES the sparse time axis (the finding's failure mode),
        # so relying on it (barrier_dims=None default) leaves the axis σ-smeared.
        assert (
            detect_barrier_dims(make_sparse_4d_leaf(0).additive_sublods[0].centers)
            == []
        )

        def max_time_extent(path: Path, part: str) -> float:
            cb = np.asarray(zarr.open_group(str(path), mode="r")[part]["chunk_bounds"])
            return float((cb[:, 3, 1] - cb[:, 3, 0]).max())

        with tempfile.TemporaryDirectory() as tmpdir:
            bpath = Path(tmpdir) / "barrier.gsplats.zarr"
            npath = Path(tmpdir) / "auto.gsplats.zarr"
            # Explicit barrier=[3] — what _merge_partition now passes.
            write_partition_streaming(
                bpath,
                lambda: iter([make_sparse_4d_leaf(0), make_sparse_4d_leaf(1)]),
                barrier_dims=[3],
            )
            # Auto-detect fallback (the pre-fix batch behavior): no barrier found.
            write_partition_streaming(
                npath,
                lambda: iter([make_sparse_4d_leaf(0), make_sparse_4d_leaf(1)]),
            )
            for part in ("part_0", "part_1"):
                broot = zarr.open_group(str(bpath), mode="r")
                assert list(broot[part].attrs["slice_dims"]) == [3]
                assert (
                    list(
                        zarr.open_group(str(npath), mode="r")[part].attrs["slice_dims"]
                    )
                    == []
                )
                # Barrier removes the σ (coverage 3·2=6) expansion on the time
                # axis → strictly tighter time bounds than the auto-detect miss.
                assert max_time_extent(bpath, part) < max_time_extent(npath, part)

    def test_scene_barrier_from_dimension_discrete_beats_autodetect(self) -> None:
        """REGRESSION (deep-double-check, finding 5): a scene-embedded gsplat with
        a NON-INTEGER discrete axis (e.g. physical-time seconds {0.0,0.5,1.0})
        must get its barrier from the scene's authoritative Dimension.discrete
        metadata — value-based auto-detect would reject 0.5 as non-integer and
        leave the axis smeared across chunks. Mirrors the Points/Lines scene path."""
        import zarr

        from luxar import LuxarZarrCompiler
        from luxar.core.dimensions import Dimension, Dimensions

        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, discrete=True, range=(0.0, 1.0)),
            ]
        )
        rng = np.random.default_rng(12)
        n = 3000
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100
        centers[:, 3] = rng.choice([0.0, 0.5, 1.0], size=n)  # NON-integer time
        chol = np.zeros((n, 10), dtype=np.float32)
        chol[:, [0, 2, 5, 9]] = 2.0
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "scene.luxar.zarr"
            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_gsplats(
                    "gsplats", centers, amplitudes=1.0, cholesky_factors=chol
                )
            store = zarr.open_group(str(store_path), mode="r")
            attrs = dict(store["gsplats"].attrs)
            # Barrier came from Dimension.discrete (index 3), NOT auto-detect
            # (which would return [] because 0.5 is not near-integer).
            assert list(attrs["slice_dims"]) == [3]
            from luxar.io.ordering import detect_barrier_dims

            assert detect_barrier_dims(centers) == []  # proves auto-detect misses it

    def test_no_barrier_3d_unaffected(self) -> None:
        """A 3D leaf gets NO barrier (auto-detect returns [] for continuous
        floats), pure spatial ordering, σ-expanded bounds on every axis.

        Asserts the barrier machinery specifically (not just shape/count): would
        fail if detect_barrier_dims wrongly flagged a float axis (→ slice_dims
        non-empty and tight ±_BARRIER_BOUND_EPS ≈ ±1e-3 bounds — extent ~0.002
        — instead of σ-expanded)."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        rng = np.random.default_rng(3)
        centers = (rng.random((2000, 3)) * 100).astype(np.float32)
        chol = np.zeros((2000, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 2.0  # isotropic σ=2 in all 3 dims
        data = GSplatData(
            centers=centers,
            amplitudes=(rng.random(2000) + 0.5).astype(np.float32),
            cholesky_factors=chol,
        )
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat3d.gsplats.zarr"
            write_gsplats_tree(path, leaf)  # auto-detect → no barrier for floats
            root = zarr.open_group(str(path), mode="r")
            # No axis flagged as a barrier (the load-bearing assertion).
            assert list(root.attrs["slice_dims"]) == []
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2]
            bounds = self._finest_chunk_bounds(path)
            assert bounds.shape[1] == 3
            # Every axis is σ-expanded — extents dwarf the ~2·eps (≈0.002,
            # _BARRIER_BOUND_EPS) a wrongly-flagged barrier axis would get
            # (proves NO axis received tight barrier bounds). σ=2,
            # coverage 3σ → ~6 extent, well over 1.5.
            extents = bounds[:, :, 1] - bounds[:, :, 0]
            assert np.all(extents.max(axis=0) > 1.5)
            got = GSplatData.load(path)
            assert got.n_splats == 2000


class TestRobustDisplayRange:
    """amplitude_data_range (the viewer's colormap display window) must use a
    robust upper (p99.9), not the raw max — and land on the SAME node as the
    'gray' colormap so a colormapped gsplat doesn't render near-black."""

    @staticmethod
    def _skewed_gsplat(n: int = 40000):
        rng = np.random.default_rng(0)
        amp = rng.exponential(0.003, n).astype(np.float32)  # heavy right skew
        amp[rng.integers(0, n, 40)] = rng.uniform(0.1, 0.3, 40)  # bright outliers
        c = (rng.random((n, 3)) * 100).astype(np.float32)
        chol = np.zeros((n, 6), np.float32)
        chol[:, [0, 2, 5]] = 2.0
        return GSplatData(centers=c, amplitudes=amp, cholesky_factors=chol), amp

    def test_leaf_display_range_is_robust_not_max(self) -> None:
        gd, amp = self._skewed_gsplat()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skew.gsplats.zarr"
            gd.save(path, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            adr = root.attrs.get("amplitude_data_range")
            assert adr is not None, "leaf must carry a display range"
            hi = adr[1]
            # Robust: the upper is near p99.9, well below the outlier max.
            assert hi < float(amp.max()) * 0.5, (hi, float(amp.max()))
            assert abs(hi - float(np.percentile(amp, 99.9))) < 1e-4
            # Colocated with the default 'gray' colormap on the SAME node.
            assert root.attrs.get("colormap") == "gray"
            assert "amplitude_data_range" in dict(root.attrs)

    def test_lod_level_carries_display_range_with_colormap(self) -> None:
        """An additive-laddered LOD level (colormap on the level, arrays on its
        sub-LODs) must still carry a display range on the level node itself."""
        from luxar.gsplats.lod import make_substitutive_lod

        gd, _ = self._skewed_gsplat()
        pyr = make_substitutive_lod(
            gd,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pyr.gsplats.zarr"
            pyr.save(path, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            # Every colormapped level node must ALSO have amplitude_data_range.
            for lvl in [k for k in root.group_keys() if k.startswith("child_")]:
                a = dict(root[lvl].attrs)
                if a.get("colormap"):
                    assert "amplitude_data_range" in a, f"{lvl} colormap without range"
                    assert a["amplitude_data_range"][1] > a["amplitude_data_range"][0]
