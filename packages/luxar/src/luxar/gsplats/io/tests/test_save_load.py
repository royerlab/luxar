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
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


def _positive_diag(chol: np.ndarray, d: int = 3) -> np.ndarray:
    """Force the packed-Cholesky diagonal strictly positive, in place.

    A real Cholesky factor has ``L[i, i] > 0``; unconstrained random vectors
    give ~50% negative diagonals that the writer's positive-diagonal gate
    legitimately rejects. Off-diagonals are left untouched (kept signed) so the
    fixture still exercises negative off-diagonals through the round-trip.
    """
    diag_idx = np.cumsum(np.arange(1, d + 1)) - 1
    chol[..., diag_idx] = np.abs(chol[..., diag_idx]) + 0.1
    return chol


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
        uniform = _positive_diag(rng.standard_normal(k).astype(np.float32), d)
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
            # Packed lower-tri diagonal slots for d=3 are [0, 2, 5].
            chol[:, [0, 2, 5]] = 1.0
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
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
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
        assert g.truncation_radius == DEFAULT_TRUNCATION_RADIUS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            assert GSplatData.load(path).truncation_radius == DEFAULT_TRUNCATION_RADIUS

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
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
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
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
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
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
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
            assert GSplatData.load(path).truncation_radius == DEFAULT_TRUNCATION_RADIUS


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


class TestArchiveRootAttrsPeek:
    """Reading a compressed store's ROOT ``.zattrs`` without extracting it
    (``luxar.gsplats.io._archive.read_archive_root_attrs``).

    Regression for #1604: a ``.gsplats.zarr.zip`` / ``.tar.gz`` is a first-class
    input to the rebuild commands, so the authored-appearance carry has to be
    able to see inside one — and the old "peeking would extract GBs again"
    reasoning was simply wrong (it is one member). The peek must find the ROOT
    ``.zattrs`` *specifically*: a child group's attrs is a different object, and
    handing it to the writer would author an appearance nobody asked for.
    """

    @staticmethod
    def _write_zip(path: Path, members: list[tuple[str, str]]) -> None:
        """Write a zip with ``(name, text)`` members in exactly the given order."""
        import zipfile

        with zipfile.ZipFile(path, "w") as zip_ref:
            for name, text in members:
                zip_ref.writestr(name, text)

    @staticmethod
    def _write_targz(path: Path, members: list[tuple[str, str]]) -> None:
        """Write a tar.gz with ``(name, text)`` members in exactly the given order."""
        import io
        import tarfile

        with tarfile.open(path, "w:gz") as tar_ref:
            for name, text in members:
                payload = text.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar_ref.addfile(info, io.BytesIO(payload))

    def _write(self, path: Path, fmt: str, members: list[tuple[str, str]]) -> None:
        if fmt == "zip":
            self._write_zip(path, members)
        else:
            self._write_targz(path, members)

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_reads_a_real_archived_store(self, tmp_path: Path, fmt: str) -> None:
        """The attrs a real ``save(compress=...)`` wrote come back off the archive.

        Covers the production nesting (``<name>.gsplats.zarr/.zattrs``) that
        ``_compress_zarr`` produces for both formats.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        data = GSplatData(**create_test_splats_3d(16))
        archive = tmp_path / f"peek.gsplats.zarr.{fmt}"
        data.save(archive, compress=fmt, root_attrs={"opacity": 0.75, "gamma": 1.3})

        attrs = read_archive_root_attrs(archive)
        assert attrs["format_type"] == "gsplats_zarr"
        assert attrs["opacity"] == 0.75
        assert attrs["gamma"] == 1.3

    #: The two store layouts ``extract_compressed_zarr`` accepts, as the archive
    #: member path of each one's ROOT ``.zattrs``: the ``*.gsplats.zarr``-named
    #: top-level directory ``_compress_zarr`` writes, and the extractor's fallback
    #: — the sole top-level directory, whatever it is called (what you get from
    #: ``tar czf x.gsplats.zarr.tar.gz mystore``).
    ROOT_LAYOUTS = ["x.gsplats.zarr/.zattrs", "mystore/.zattrs"]

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    @pytest.mark.parametrize("root", ROOT_LAYOUTS)
    def test_a_deeper_zattrs_never_wins(
        self, tmp_path: Path, fmt: str, root: str
    ) -> None:
        """A child group's ``.zattrs`` is not read as the root's, even listed first.

        Run for both store layouts the extractor accepts. The child member is
        deliberately written BEFORE the root so a first-match-wins implementation
        fails here.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        parent = root[: -len(".zattrs")]
        archive = tmp_path / f"nested.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (f"{parent}lod_0/.zattrs", '{"whose": "child"}'),
                (root, '{"whose": "root"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "root"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    @pytest.mark.parametrize("root", ROOT_LAYOUTS)
    def test_the_stores_root_wins_over_a_shallower_stray(
        self, tmp_path: Path, fmt: str, root: str
    ) -> None:
        """A stray top-level ``.zattrs`` does not outrank the store's own root.

        ``extract_compressed_zarr`` defines the store as a top-level DIRECTORY, so
        that directory's ``.zattrs`` is the root even though a loose member sits
        one level shallower; reading the stray instead would author an appearance
        from something that is not the dataset. Both layouts matter, and the
        unnamed fallback is the sharp one: a ranking that merely preferred the
        ``*.gsplats.zarr`` name and then went shallowest-first hands back the
        stray's attrs there, while the real load succeeds with the store's. The
        stray is written FIRST so first-match-wins fails too.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"stray.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (".zattrs", '{"whose": "stray"}'),
                (root, '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_depth_zero_only_archive_is_empty(self, tmp_path: Path, fmt: str) -> None:
        """A flat archive whose only ``.zattrs`` is at the root yields ``{}``.

        Not a regression in disguise: such an archive is unreachable through the
        loader — ``extract_compressed_zarr`` raises "No .gsplats.zarr directory
        found" for it, and the peek only ever runs alongside a load that
        succeeded. Treating a depth-0 member as a store root is what would let a
        stray outrank a real one (see the test above), so it is refused outright.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"flat.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [(".zattrs", '{"whose": "stray"}')])
        assert read_archive_root_attrs(archive) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_junk_sibling_directory_makes_the_unnamed_fallback_ambiguous(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """Two top-level directories and no ``*.gsplats.zarr`` name → ``{}``.

        The extractor's fallback is the SOLE top-level directory; with two of
        them it picks by ``iterdir()`` order, which no archive index can predict.
        A junk sibling listed first (``tar czf b.gsplats.zarr.tar.gz notes
        mystore``) would otherwise hand back ``notes``'s attrs while the load
        really read ``mystore`` — authoring an appearance from a node that is not
        the dataset. Carrying nothing is the status quo; guessing is not.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"siblings.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                ("notes/.zattrs", '{"whose": "junk"}'),
                ("mystore/.zattrs", '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {}

    @staticmethod
    def _write_with_empty_dir(
        path: Path, fmt: str, dir_name: str, files: list[tuple[str, str]]
    ) -> None:
        """Write an archive with an EMPTY explicit directory entry, plus files.

        A tar spells a directory as its own ``DIRTYPE`` header; a zip spells one
        as a member name ending in ``/``. Both are written here so the rule is
        exercised through each spelling.
        """
        import io
        import tarfile
        import zipfile

        bare = dir_name.rstrip("/")
        if fmt == "zip":
            with zipfile.ZipFile(path, "w") as zip_ref:
                zip_ref.writestr(zipfile.ZipInfo(f"{bare}/"), "")
                for name, text in files:
                    zip_ref.writestr(name, text)
        else:
            with tarfile.open(path, "w:gz") as tar_ref:
                dir_info = tarfile.TarInfo(bare)
                dir_info.type = tarfile.DIRTYPE
                tar_ref.addfile(dir_info)
                for name, text in files:
                    payload = text.encode("utf-8")
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    tar_ref.addfile(member, io.BytesIO(payload))

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_an_empty_stray_directory_does_not_suppress_the_carry(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """An EMPTY top-level directory beside the store is not a rival store.

        ``zip -r out.gsplats.zarr.zip mystore notes`` with an empty ``notes/`` has
        exactly one store in it. An empty directory can never be the node a
        SUCCESSFUL load read — ``zarr.open_group`` on one raises, so if the
        extractor lands there the whole load fails and no rebuild (hence no carry)
        happens at all. Counting it as a top-level directory could therefore never
        prevent a wrong carry, only manufacture false ambiguity and silently drop
        a correct one that the directory-store path keeps.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"empty_stray.gsplats.zarr.{fmt}"
        self._write_with_empty_dir(
            archive, fmt, "notes", [("mystore/.zattrs", '{"whose": "store"}')]
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_named_store_still_wins_over_a_junk_sibling(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """The uniqueness rule constrains the FALLBACK tier only.

        A top-level ``*.gsplats.zarr`` directory is the extractor's own first
        preference, so peek and extraction agree on it no matter what else the
        archive contains — the junk sibling must not suppress it.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"named_siblings.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                ("notes/.zattrs", '{"whose": "junk"}'),
                ("x.gsplats.zarr/.zattrs", '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_flat_dump_of_a_tree_stores_contents_is_empty(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """An archive of a tree store's CONTENTS never yields a child's attrs.

        ``tar czf b.gsplats.zarr.tar.gz -C store .`` puts the store root's own
        ``.zattrs``/``.zgroup`` at depth 0 (where the extractor cannot see a store
        at all) and its ``child_<i>/`` groups at depth 1 — so every depth-1
        candidate is a CHILD group, exactly what the ranking exists to keep out of
        the authored appearance. What refuses it is the several-top-level-
        directories rule, which is why a real tree shape (a ``kind=lod`` /
        ``kind=partition`` group always has at least two children) is used here.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"contents.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (".zattrs", '{"whose": "root"}'),
                (".zgroup", '{"zarr_format": 2}'),
                ("child_0/.zattrs", '{"whose": "child_0"}'),
                ("child_1/.zattrs", '{"whose": "child_1"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {}

    @staticmethod
    def _write_with_symlink(
        path: Path,
        fmt: str,
        link: tuple[str, str],
        extra: list[tuple[str, str]],
    ) -> None:
        """Write an archive whose ``link`` member is a real SYMLINK, plus files.

        Both formats spell a symlink differently — a unix mode in the zip's
        external attrs, a ``SYMTYPE`` header in the tar — so the guard has to be
        exercised through each spelling separately.
        """
        import io
        import stat
        import tarfile
        import zipfile

        name, target = link
        if fmt == "zip":
            with zipfile.ZipFile(path, "w") as zip_ref:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zip_ref.writestr(info, target)
                for member_name, text in extra:
                    zip_ref.writestr(member_name, text)
        else:
            with tarfile.open(path, "w:gz") as tar_ref:
                info_tar = tarfile.TarInfo(name)
                info_tar.type = tarfile.SYMTYPE
                info_tar.linkname = target
                tar_ref.addfile(info_tar)
                for member_name, text in extra:
                    payload = text.encode("utf-8")
                    member = tarfile.TarInfo(member_name)
                    member.size = len(payload)
                    tar_ref.addfile(member, io.BytesIO(payload))

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_symlinked_zattrs_member_is_never_read(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A ``.zattrs`` member that is a SYMLINK is skipped, not followed.

        The module's whole threat model is that no link is ever followed, for
        BOTH formats. An in-archive link is the sharp case: ``extractfile``
        happily resolves ``x.gsplats.zarr/.zattrs -> child/.zattrs`` inside the
        tar, so dropping the regular-file check silently promotes a CHILD group's
        attrs to the root's. An out-of-archive link is the other half: a symlink
        member's payload is its TARGET PATH, so following one leaks an arbitrary
        file's location into the attrs and hands the reader a path where a JSON
        object belongs. Third archive: a real ``*.gsplats.zarr`` root still wins
        even with a link listed before it.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        shadows_child = tmp_path / f"shadow.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            shadows_child,
            fmt,
            ("x.gsplats.zarr/.zattrs", "child/.zattrs"),
            [("x.gsplats.zarr/child/.zattrs", '{"whose": "child"}')],
        )
        assert read_archive_root_attrs(shadows_child) == {}

        only_link = tmp_path / f"link.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            only_link, fmt, ("x.gsplats.zarr/.zattrs", "/etc/passwd"), []
        )
        assert read_archive_root_attrs(only_link) == {}

        with_real = tmp_path / f"link_plus_real.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            with_real,
            fmt,
            ("notes/.zattrs", "/etc/passwd"),
            [("x.gsplats.zarr/.zattrs", '{"whose": "store"}')],
        )
        assert read_archive_root_attrs(with_real) == {"whose": "store"}

    def test_non_archive_and_missing_paths_are_empty(self, tmp_path: Path) -> None:
        """A plain file, a directory and a missing path all yield ``{}``.

        ``read_authored_appearance`` funnels every non-directory input here, so
        "not an archive" has to be a quiet empty answer rather than a raise.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs
        from luxar.gsplats.io.load_gsplats import read_authored_appearance

        plain = tmp_path / "notes.txt"
        plain.write_text("not an archive")
        missing = tmp_path / "gone.gsplats.zarr.zip"

        assert read_archive_root_attrs(plain) == {}
        assert read_archive_root_attrs(tmp_path) == {}
        assert read_archive_root_attrs(missing) == {}
        assert read_authored_appearance(plain) == {}
        assert read_authored_appearance(missing) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_oversized_attrs_member_refused(
        self, tmp_path: Path, fmt: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.zattrs`` far too big for attrs is refused, not read into memory."""
        from luxar.gsplats.io import _archive

        monkeypatch.setattr(_archive, "_MAX_ATTRS_BYTES", 8)
        archive = tmp_path / f"fat.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/.zattrs", '{"opacity": 0.75}')])
        assert _archive.read_archive_root_attrs(archive) == {}


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
    ``coverage_fraction`` values from the writer fallback (screen-occupancy
    AREA halving: coarsest 0.0, finest at the half-screen-area anchor
    0.5). The builders normally stamp these into child meta — this
    exercises the fallback for a tree written without it."""
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
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
        # Screen-occupancy AREA halving: any 2-level ladder → finest 0.5.
        assert root["child_1"].attrs["coverage_fraction"] == pytest.approx(
            WHOLE_OBJECT_FINEST_ANCHOR
        )


def test_writer_selector_threshold_consistency_gate() -> None:
    """A group's meta ``selector`` describes its AUTHORED thresholds, so the
    writer may preserve it only when EVERY child carries one. Three arms:

    * PARTIALLY-authored + ``selector="coverage"`` — without the gate, the
      writer's fallback derivation (screen-area units) filled the gaps under
      the legacy stamp: a mixed-units store. The gate scrubs the authored
      remnant, re-derives the whole ladder, and stamps ``screen-area`` so the
      written pair agrees.
    * FULLY authored + ``selector="coverage"`` — preserved verbatim (this is
      the legacy round-trip; the values must NOT be re-derived).
    * Unknown meta selector — refused before anything is written (the reader
      whitelists stale spellings away on LOAD; one arriving here is a
      hand-built tree that would otherwise write an out-of-vocabulary
      selector into a store claiming v3.4 compliance).
    """
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    def _leaf(n: int, seed: int, cov: float | None) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        meta = {} if cov is None else {"coverage_fraction": cov}
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol,
                )
            ],
            meta=meta,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Arm 1: partially authored (legacy value on child_0 only) under a
        # "coverage" stamp → uniform re-derivation + screen-area stamp.
        mixed = GSplatLodGroup(
            children=[_leaf(50, 0, cov=0.0), _leaf(800, 1, cov=None)],
            meta={"selector": "coverage"},
        )
        p1 = Path(tmpdir) / "mixed.gsplats.zarr"
        write_gsplats_tree(p1, mixed, ordering="none")
        r1 = zarr.open_group(str(p1), mode="r")
        assert r1.attrs["selector"] == "screen-area"
        assert r1["child_0"].attrs["coverage_fraction"] == 0.0
        assert r1["child_1"].attrs["coverage_fraction"] == pytest.approx(
            WHOLE_OBJECT_FINEST_ANCHOR
        )

        # Arm 2: fully authored legacy ladder → selector AND values preserved.
        legacy = GSplatLodGroup(
            children=[_leaf(50, 2, cov=0.0), _leaf(800, 3, cov=2.0)],
            meta={"selector": "coverage"},
        )
        p2 = Path(tmpdir) / "legacy.gsplats.zarr"
        write_gsplats_tree(p2, legacy, ordering="none")
        r2 = zarr.open_group(str(p2), mode="r")
        assert r2.attrs["selector"] == "coverage"
        assert r2["child_1"].attrs["coverage_fraction"] == 2.0  # NOT re-derived

        # Arm 3: out-of-vocabulary selector → refused, nothing written.
        bogus = GSplatLodGroup(
            children=[_leaf(50, 4, cov=0.0), _leaf(800, 5, cov=1.0)],
            meta={"selector": "pixel_size"},
        )
        p3 = Path(tmpdir) / "bogus.gsplats.zarr"
        with pytest.raises(ValueError, match="must be one of"):
            write_gsplats_tree(p3, bogus, ordering="none")

        # Arm 4: fully authored but SELECTOR-LESS → stamped LEGACY "coverage",
        # values preserved. Authored values with no stated units are exactly
        # what the viewer's loader treats as legacy (its missing/unknown-
        # selector fallback), so relabeling them "screen-area" would silently
        # reinterpret them — authored = legacy is the library convention.
        selectorless = GSplatLodGroup(
            children=[_leaf(50, 6, cov=0.0), _leaf(800, 7, cov=2.0)],
        )
        p4 = Path(tmpdir) / "selectorless.gsplats.zarr"
        write_gsplats_tree(p4, selectorless, ordering="none")
        r4 = zarr.open_group(str(p4), mode="r")
        assert r4.attrs["selector"] == "coverage"
        assert r4["child_1"].attrs["coverage_fraction"] == 2.0

        # Arm 5: authored thresholds must honor the selector's contract. 2.0 is
        # legal legacy-diagonal (arm 2/4) but OUT OF RANGE for screen-area
        # (the clipped area metric tops out at 1.0), and a non-monotonic
        # ladder is invalid under either — both refused before writing.
        over_range = GSplatLodGroup(
            children=[_leaf(50, 8, cov=0.0), _leaf(800, 9, cov=2.0)],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            write_gsplats_tree(
                Path(tmpdir) / "over.gsplats.zarr", over_range, ordering="none"
            )
        non_monotonic = GSplatLodGroup(
            children=[
                _leaf(50, 10, cov=0.0),
                _leaf(200, 14, cov=0.5),
                _leaf(800, 11, cov=0.25),
            ],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match="strictly greater"):
            write_gsplats_tree(
                Path(tmpdir) / "nonmono.gsplats.zarr", non_monotonic, ordering="none"
            )
        # The coarsest child must be exactly 0.0 (the always-eligible floor the
        # format requires) — [0.25, 0.5] is strictly ascending and in range but
        # leaves no eligible child below 0.25 occupancy.
        no_floor = GSplatLodGroup(
            children=[_leaf(50, 12, cov=0.25), _leaf(800, 13, cov=0.5)],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match="must be exactly 0.0"):
            write_gsplats_tree(
                Path(tmpdir) / "nofloor.gsplats.zarr", no_floor, ordering="none"
            )


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


class TestAtomicWrites:
    """Crash-safety: writers must never destroy a prior good store or leave a
    partial one — write-to-temp-sibling + atomic swap (see _atomic_finalize)."""

    @staticmethod
    def _leaf(seed: int):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.tree import GSplatLeaf

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

    @staticmethod
    def _no_tmp_siblings(directory: Path) -> bool:
        return not any(directory.glob(".*tmp-*"))

    def test_failed_tree_write_preserves_prior_store(self, monkeypatch) -> None:
        import importlib

        # NOT `import ... as sg`: the io package re-exports the FUNCTION
        # `save_gsplats`, which shadows the submodule on attribute lookup.
        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            # A good prior store.
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            before = zarr.open_group(str(path), mode="r").attrs["content_hash"]

            # A rewrite that crashes mid-write (inside the node walker).
            def boom(*a, **k):
                raise RuntimeError("simulated mid-write crash")

            monkeypatch.setattr(sg, "write_gsplat_node", boom)
            with pytest.raises(RuntimeError, match="simulated mid-write crash"):
                sg.write_gsplats_tree(path, self._leaf(1), ordering="none")

            # Prior good store untouched; no temp sibling left behind.
            after = zarr.open_group(str(path), mode="r").attrs["content_hash"]
            assert after == before
            assert self._no_tmp_siblings(path.parent)

    def test_failed_streaming_write_preserves_prior_store(self, monkeypatch) -> None:
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_partition_streaming(
                path, lambda: iter([self._leaf(0), self._leaf(1)]), ordering="none"
            )
            before = zarr.open_group(str(path), mode="r").attrs["content_hash"]

            def parts_then_boom():
                yield self._leaf(2)
                raise RuntimeError("simulated producer crash")

            with pytest.raises(RuntimeError, match="simulated producer crash"):
                write_partition_streaming(path, parts_then_boom, ordering="none")

            after = zarr.open_group(str(path), mode="r").attrs["content_hash"]
            assert after == before
            assert self._no_tmp_siblings(path.parent)

    def test_streaming_zero_parts_leaves_nothing(self) -> None:
        # The n_written == 0 ValueError used to leave a partial root at the
        # destination; now neither the destination nor a temp sibling exists.
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.gsplats.zarr"
            with pytest.raises(ValueError, match="no non-empty parts"):
                write_partition_streaming(path, lambda: iter([]), ordering="none")
            assert not path.exists()
            assert self._no_tmp_siblings(path.parent)

    def test_failed_compression_leaves_no_partial_archive(self, monkeypatch) -> None:
        import importlib

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr.zip"

            def boom(*a, **k):
                raise RuntimeError("simulated compression crash")

            monkeypatch.setattr(sg, "_atomic_finalize", boom)
            with pytest.raises(RuntimeError, match="simulated compression crash"):
                save_gsplats(
                    path=path,
                    **create_test_splats_3d(30),
                    ordering="none",
                    compress="zip",
                )
            assert not path.exists()
            assert self._no_tmp_siblings(path.parent)

    def test_success_roundtrip_unchanged(self) -> None:
        # The atomic swap must not change what a successful save produces.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt.gsplats.zarr"
            splats = create_test_splats_3d(40)
            save_gsplats(path=path, **splats, ordering="none")
            data = load_gsplats(path)
            assert data.n_splats == 40
            # Spatial ordering off + AUTO encoding: match the tolerance the
            # existing roundtrip tests use for quantized amplitudes.
            assert np.allclose(
                np.sort(data.amplitudes), np.sort(splats["amplitudes"]), atol=0.05
            )
            assert self._no_tmp_siblings(path.parent)


class TestAtomicFinalizeTrashFirst:
    """The directory swap is trash-first: dest never absent while a prior
    good store existed, and a failed swap-in restores the original."""

    def test_failed_swap_in_restores_prior_store(self, tmp_path, monkeypatch) -> None:
        import importlib
        import os as _os

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")
        dest = tmp_path / "store.gsplats.zarr"
        dest.mkdir()
        (dest / "old.txt").write_text("prior good store")
        tmp = tmp_path / ".store.tmp"
        tmp.mkdir()
        (tmp / "new.txt").write_text("new store")

        real_replace = _os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # the tmp→dest swap-in
                raise OSError("simulated swap-in failure")
            return real_replace(src, dst)

        monkeypatch.setattr(sg.os, "replace", flaky_replace)
        with pytest.raises(OSError, match="simulated swap-in failure"):
            sg._atomic_finalize(tmp, dest)
        # Prior store restored under its original name; tmp untouched.
        assert (dest / "old.txt").read_text() == "prior good store"
        assert (tmp / "new.txt").exists()

    def test_successful_swap_leaves_no_trash(self, tmp_path) -> None:
        import importlib

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")
        dest = tmp_path / "store.gsplats.zarr"
        dest.mkdir()
        (dest / "old.txt").write_text("x")
        tmp = tmp_path / ".store.tmp"
        tmp.mkdir()
        (tmp / "new.txt").write_text("y")
        sg._atomic_finalize(tmp, dest)
        assert (dest / "new.txt").read_text() == "y"
        assert not (dest / "old.txt").exists()
        assert not list(tmp_path.glob(".*trash-*"))


# ────────────────────────────────────────────────────────────────────────
# Topology-aware coverage_fraction FALLBACK (both writers)
#
# The stamped anchors are what recipes emit, but every writer also DERIVES a
# fallback for a node that carries no ``coverage_fraction`` in its ``meta`` —
# which is exactly the state ``luxar gsplat transform`` leaves the tree in after
# its scrub, and the state a legacy pre-v3.2 store loads in. The CHANGELOG's
# "scrub-and-re-derive stays a no-op" claim rests on that fallback picking the
# PARTITION-BOUND anchor for a partition-bound ladder, so pin it here for both
# ``write_gsplats_tree`` and its streaming sibling.
# ────────────────────────────────────────────────────────────────────────


def _strip_coverage(node):
    """The post-scrub state: no ``coverage_fraction`` anywhere in ``meta``."""
    from luxar.gsplats.tree import without_meta_key

    return without_meta_key(node, "coverage_fraction")


def _recipe_tree(recipe: str):
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

    rng = np.random.default_rng(0)
    n = 400
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
        cholesky_factors=chol,
    )
    params = RecipeParams(
        max_elements=120, compression_factor=4, levels=2, device="cpu", seed=0
    )
    return build_recipe(data, recipe, params)


@pytest.mark.parametrize("recipe", ["overview", "adaptive"])
def test_meta_less_partition_bound_tree_rederives_the_partitioned_anchor(
    recipe: str, tmp_path: Path
) -> None:
    """``write_gsplats_tree`` on a scrubbed tree must re-derive 0.0 → 1.0
    (``PARTITION_FINEST_AREA`` — the tile alone fills the screen)."""
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    node = _strip_coverage(_recipe_tree(recipe))
    out = tmp_path / f"{recipe}.gsplats.zarr"
    write_gsplats_tree(out, node)
    root = zarr.open_group(str(out), mode="r")
    # overview: the lod group IS the root. adaptive: one lod group per part.
    lod_group = root if recipe == "overview" else root["part_0"]
    covs = [
        float(lod_group[k].attrs["coverage_fraction"])
        for k in sorted(lod_group.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"{recipe}: meta-less re-derivation gave {covs}, expected the "
        "partition-bound anchor (finest = PARTITION_FINEST_AREA)"
    )
    assert all(covs[i] > covs[i - 1] for i in range(1, len(covs)))


def test_scrub_and_rederive_is_a_no_op_for_partition_bound_ladders(
    tmp_path: Path,
) -> None:
    """The exact ``gsplat transform`` round trip: stamped == re-derived."""
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    for recipe in ("overview", "adaptive"):
        node = _recipe_tree(recipe)
        stamped = tmp_path / f"{recipe}_stamped.gsplats.zarr"
        scrubbed = tmp_path / f"{recipe}_scrubbed.gsplats.zarr"
        write_gsplats_tree(stamped, node)
        write_gsplats_tree(scrubbed, _strip_coverage(node))

        def _covs(path):
            root = zarr.open_group(str(path), mode="r")
            g = root if recipe == "overview" else root["part_0"]
            return [
                float(g[k].attrs["coverage_fraction"])
                for k in sorted(g.group_keys(), key=lambda s: int(s.split("_")[1]))
            ]

        assert _covs(stamped) == pytest.approx(_covs(scrubbed)), recipe


def test_streaming_partition_writer_uses_the_partitioned_anchor(
    tmp_path: Path,
) -> None:
    """``write_partition_streaming``'s root IS a kind=partition, so every part is
    partition-bound by construction — its recursion must say so.

    Regression: the streaming writer started the walk at the default
    ``under_partition=False``, so a meta-less per-part ladder came out with the
    whole-object anchor (0.0/0.25/0.5) here while ``write_gsplats_tree`` on the
    same tree gave the tile anchor (0.0/0.5/1.0). Latent only because the batch
    merge stamps ``meta`` that wins over the fallback.
    """
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
    from luxar.gsplats.io.save_gsplats import write_partition_streaming
    from luxar.gsplats.tree import GSplatPartition

    partitioned = _strip_coverage(_recipe_tree("adaptive"))
    assert isinstance(partitioned, GSplatPartition)
    parts = list(partitioned.children)

    out = tmp_path / "streamed.gsplats.zarr"
    write_partition_streaming(out, lambda: iter(parts), max_elements=120)

    root = zarr.open_group(str(out), mode="r")
    part0 = root["part_0"]
    covs = [
        float(part0[k].attrs["coverage_fraction"])
        for k in sorted(part0.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"streaming writer gave {covs}; every part_<i> is under a kind=partition "
        "root, so the fallback must use the partition-bound anchor"
    )


def test_meta_less_one_part_partition_rederives_the_whole_object_anchor(
    tmp_path: Path,
) -> None:
    """A ONE-part partition is not a tiling — its part covers the whole object,
    so the fallback must NOT hand the fills-screen anchor down.

    ``build_adaptive`` emits exactly this shape whenever the dataset fits
    ``max_elements``, and it stamps the whole-object anchor (0.5). If the
    writer's topology fallback disagreed, a ``gsplat transform``
    scrub-and-re-derive would silently re-coarsen the store back to the #1361
    behaviour.
    """
    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
    from luxar.gsplats.tree import GSplatPartition

    rng = np.random.default_rng(0)
    n = 120
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
        cholesky_factors=chol,
    )
    node = build_recipe(
        data,
        "adaptive",
        # max_elements unset → the default 1,000,000, so the BSP never splits.
        RecipeParams(compression_factor=4, levels=2, device="cpu", seed=0),
    )
    assert isinstance(node, GSplatPartition) and len(node.children) == 1

    stamped = tmp_path / "one_part_stamped.gsplats.zarr"
    scrubbed = tmp_path / "one_part_scrubbed.gsplats.zarr"
    write_gsplats_tree(stamped, node)
    write_gsplats_tree(scrubbed, _strip_coverage(node))

    def _covs(path: Path) -> list[float]:
        g = zarr.open_group(str(path), mode="r")["part_0"]
        return [
            float(g[k].attrs["coverage_fraction"])
            for k in sorted(g.group_keys(), key=lambda s: int(s.split("_")[1]))
        ]

    assert _covs(scrubbed)[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR), (
        f"one-part re-derivation gave {_covs(scrubbed)}, expected the "
        "whole-object half-screen-area anchor (finest = 0.5)"
    )
    # And the round trip is still a no-op, as it is for real tilings.
    assert _covs(stamped) == pytest.approx(_covs(scrubbed))


def test_one_part_partition_nested_in_a_tiling_keeps_the_tile_anchor(
    tmp_path: Path,
) -> None:
    """A one-part partition INSIDE a real tiling must not drop the outer binding.

    The one-part exclusion only ever ADDS a binding, never removes one: ``part_1``
    below is a lone-part wrapper, but it still sits inside ONE tile of a >=2-part
    partition, so the meta-less ladder underneath it keeps the fills-screen
    anchor. Overwriting the incoming flag instead of OR-ing it in would hand that
    ladder the whole-object 0.5 — and would put the writer out of step with
    ``graft_gsplat_node``, its scene-side mirror.
    """
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
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

    # Outer = a genuine 2-part tiling. part_1 = a one-part wrapper holding a
    # meta-less coarse→fine ladder, so nothing authored beats the fallback.
    tree = GSplatPartition(
        children=[
            _leaf(40, 0),
            GSplatPartition(
                children=[GSplatLodGroup(children=[_leaf(25, 1), _leaf(100, 2)])]
            ),
        ]
    )

    out = tmp_path / "nested_one_part.gsplats.zarr"
    write_gsplats_tree(out, tree, ordering="none")

    ladder = zarr.open_group(str(out), mode="r")["part_1"]["part_0"]
    covs = [
        float(ladder[k].attrs["coverage_fraction"])
        for k in sorted(ladder.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"nested one-part wrapper gave {covs}; the outer >=2-part tiling still "
        "binds this ladder, so the finest must be PARTITION_FINEST_AREA"
    )
