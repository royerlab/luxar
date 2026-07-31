"""Tests for the v3.3 ``.gsplats.zarr`` node-tree format compliance.

A single splat set saved via :func:`save_gsplats` is a **leaf node at the file
root**: the arrays (``centers`` / ``amplitudes`` / ``cholesky_factors_diag`` /
``cholesky_factors_offdiag`` / ``chunk_bounds``) live directly under the root
group, and the root ``.zattrs`` carry both the self-identifying header
(``format_version`` = ``"3.3"``) and the leaf's own attrs (``type`` =
``"gsplats"``, ``n_splats``, ordering metadata, ``center_bounds``,
``position_bounds``, render defaults). v3.1 splits the Cholesky factors into a
diagonal and an off-diagonal array (v3.0 stored a single ``cholesky_factors``);
v3.2 renames the ``kind=lod`` selector attrs (``coverage_fraction``).
"""

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats.io import save_gsplats
from luxar.typing_utils._format_contract import GSPLATS_FORMAT_VERSION


def create_test_splats_3d(n_splats: int = 100) -> dict:
    """Create test 3D Gaussian splats."""
    rng = np.random.default_rng(0)
    return {
        "centers": (rng.random((n_splats, 3)).astype(np.float32) * 10),
        "amplitudes": (rng.random(n_splats).astype(np.float32) * 2),
        "cholesky_factors": rng.random((n_splats, 6)).astype(np.float32),
    }


class TestFormatCompliance:
    """Test current gsplats format specification compliance."""

    def test_root_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path, **create_test_splats_3d(100), description="Test dataset"
            )
            root = zarr.open_group(str(path), mode="r")

            assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
            assert root.attrs["format_type"] == "gsplats_zarr"
            assert "timestamp" in root.attrs
            assert "luxar_gsplats_version" in root.attrs
            assert root.attrs["description"] == "Test dataset"

    def test_leaf_root_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            root = zarr.open_group(str(path), mode="r")

            # v3.0: a single splat set is a leaf node AT the root — arrays live
            # directly under root, not under splats/substitutive_0/additive_0.
            assert root.attrs["type"] == "gsplats"
            assert "splats" not in root
            for arr in (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
                "chunk_bounds",
            ):
                assert arr in root, arr
            # v3.1 split: the single packed array is gone.
            assert "cholesky_factors" not in root

    def test_leaf_root_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            attrs = zarr.open_group(str(path), mode="r").attrs

            assert attrs["n_splats"] == 100
            assert attrs["ndim"] == 3
            assert "has_colors" in attrs
            assert attrs["ordering"] == "morton"
            assert "chunk_size" in attrs
            assert "ordering_min" in attrs
            assert "ordering_max" in attrs
            assert "ordering_bits_per_dim" in attrs
            assert "amplitude_range" in attrs and {"min", "max"} <= set(
                attrs["amplitude_range"]
            )
            cb = attrs["center_bounds"]
            assert len(cb["min"]) == 3 and len(cb["max"]) == 3
            # Blocker 2: every node carries position_bounds for framing-on-load.
            assert "position_bounds" in attrs

    def test_array_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.PRECISION,
            )
            root = zarr.open_group(str(path), mode="r")

            assert root["centers"].shape[1] == 3
            assert len(root["amplitudes"].shape) == 1
            # v3.1 split: diagonal has d=3 columns, off-diagonal has k-d=3.
            assert root["cholesky_factors_diag"].shape[1] == 3
            assert root["cholesky_factors_offdiag"].shape[1] == 3
            cb = root["chunk_bounds"]
            assert cb.shape[1] == 3 and cb.shape[2] == 2

    def test_encoding_metadata_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.MEMORY,
            )
            root = zarr.open_group(str(path), mode="r")
            for array_name in (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
            ):
                enc = root[array_name].attrs.get("encoding")
                assert enc is not None and "name" in enc, array_name

    def test_fitting_group_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={
                    "fitter_name": "luxar.gsplats",
                    "fitter_version": "0.1.0",
                    "time_seconds": 45.3,
                    "iterations": 850,
                    "converged": True,
                },
                fitting_config={"n_iters": 1000, "lr": 0.05},
            )
            root = zarr.open_group(str(path), mode="r")
            assert "fitting" in root
            fg = root["fitting"]
            assert fg.attrs["fitter_name"] == "luxar.gsplats"
            assert fg.attrs["iterations"] == 850
            assert fg.attrs["converged"] is True
            assert "config" in fg
            assert fg["config"].attrs["n_iters"] == 1000

    def test_provenance_group_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                provenance_info={
                    "source_file": "/path/to/image.tif",
                    "shape": [128, 256, 256],
                    "dtype": "uint16",
                    "normalization": {"method": "percentile", "low": 0.1, "high": 99.9},
                },
            )
            root = zarr.open_group(str(path), mode="r")
            assert "provenance" in root
            pg = root["provenance"]
            assert pg.attrs["source_file"] == "/path/to/image.tif"
            assert pg.attrs["shape"] == [128, 256, 256]
            assert pg.attrs["normalization"]["method"] == "percentile"

    def test_chunk_bounds_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            chunk_bounds = zarr.open_group(str(path), mode="r")["chunk_bounds"]
            assert chunk_bounds.ndim == 3
            assert chunk_bounds.shape[1] == 3 and chunk_bounds.shape[2] == 2
            assert chunk_bounds.dtype == np.float32
            data = chunk_bounds[:]
            assert np.all(data[:, :, 0] <= data[:, :, 1])

    def test_consolidated_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50))
            assert (path / ".zmetadata").exists()

    def test_3d_specific_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.PRECISION,
            )
            root = zarr.open_group(str(path), mode="r")
            assert root["cholesky_factors_diag"].shape[1] == 3
            assert root["cholesky_factors_offdiag"].shape[1] == 3
            assert root.attrs["ndim"] == 3
            cb = root.attrs["center_bounds"]
            assert len(cb["min"]) == 3 and len(cb["max"]) == 3

    def test_semantic_type_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.MEMORY,
            )
            root = zarr.open_group(str(path), mode="r")
            # COORDINATE → uint16 per-axis fixed-point in AUTO/MEMORY
            # (float16 disabled; coordinates never quantize to uint8).
            assert (
                root["centers"].attrs.get("encoding", {})["name"]
                == "linear_perchannel_u16"
            )

    def test_colors_auto_detected_sdr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = (
                np.random.default_rng(1).random((50, 3)).astype(np.float32)
            )  # in [0,1]
            save_gsplats(path=path, **create_test_splats_3d(50), colors=colors)
            root = zarr.open_group(str(path), mode="r")
            assert "colors" in root
            enc = root["colors"].attrs.get("encoding", {})
            # color_mode is auto-detected; [0,1] values → SDR.
            if "color_mode" in enc:
                assert enc["color_mode"] == "sdr"

    def test_ordering_metadata_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            attrs = zarr.open_group(str(path), mode="r").attrs
            assert attrs["ordering"] == "morton"
            assert len(attrs["ordering_min"]) == 3
            assert len(attrs["ordering_max"]) == 3
            assert isinstance(attrs["ordering_bits_per_dim"], int)
            assert attrs["ordering_bits_per_dim"] <= 21

    def test_no_ordering_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            attrs = zarr.open_group(str(path), mode="r").attrs
            assert attrs["ordering"] == "none"
            assert "ordering_min" not in attrs
            assert "ordering_max" not in attrs

    def test_chunk_bounds_count_matches_centers_chunks(self) -> None:
        """§8 regression: the spatial index partition count and the centers
        zarr chunk count must derive from the same chunk_size (one formula)."""
        import math

        n = 20_000  # large enough to span multiple chunks at the 64KB target
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(n), ordering="morton")
            root = zarr.open_group(str(path), mode="r")
            chunk_size = root.attrs["chunk_size"]
            expected = math.ceil(n / chunk_size)
            assert root["chunk_bounds"].shape[0] == expected
            assert root["centers"].nchunks == expected
