"""Tests for the `migrate_format` legacy → v2.0 conversion tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.io import load_gsplats
from luxar.gsplats.io.migrate import (
    _read_substitutive_directory,
    _read_v1_x_root,
    detect_legacy_format,
    migrate_format,
)

# ---------------------------------------------------------------------------
# Helpers to build legacy-format fixtures
# ---------------------------------------------------------------------------


def _identity_chol(n: int) -> np.ndarray:
    """Return n rows of the cholesky factor for an isotropic 3D Gaussian."""
    row = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.tile(row, (n, 1))


def _make_v1_0(path: Path, n: int, *, with_fitting: bool = False) -> None:
    """Build a v1.0 `.gsplats.zarr` (single flat splat set)."""
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(
        {
            "format_version": "1.0",
            "format_type": "gsplats_zarr",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "luxar_gsplats_version": "test",
            "description": "v1.0 test fixture",
        }
    )
    splats = root.create_group("splats")
    splats.attrs.update(
        {
            "type": "gsplats",
            "n_splats": n,
            "ndim": 3,
            "has_colors": False,
            "ordering": "none",
            "truncation_radius": 3.0,
        }
    )
    rng = np.random.default_rng(0)
    splats.create_dataset("centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
    splats.create_dataset("amplitudes", data=rng.random(n).astype(np.float32))
    splats.create_dataset("cholesky_factors", data=_identity_chol(n))
    splats.create_dataset("chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
    if with_fitting:
        fitting = root.create_group("fitting")
        fitting.attrs.update(
            {
                "fitter_name": "test-fitter",
                "fitter_version": "0.0.1",
                "time_seconds": 1.5,
                "iterations": 100,
                "converged": True,
            }
        )
        config = fitting.create_group("config")
        config.attrs.update({"lr": 0.05, "preset": "draft"})
        prov = root.create_group("provenance")
        prov.attrs.update(
            {
                "source_file": "test.tiff",
                "source_shape": [32, 32, 32],
                "source_dtype": "float32",
            }
        )
    zarr.consolidate_metadata(store)


def _make_v1_1(path: Path, lod_sizes: list[int]) -> None:
    """Build a v1.1 `.gsplats.zarr` (multi-LOD additive)."""
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    n_lods = len(lod_sizes)
    root.attrs.update(
        {
            "format_version": "1.1",
            "format_type": "gsplats_zarr",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "luxar_gsplats_version": "test",
            "n_lods": n_lods,
        }
    )
    splats = root.create_group("splats")
    splats.attrs.update(
        {
            "type": "gsplats",
            "n_lods": n_lods,
            "truncation_radius": 3.0,
        }
    )
    rng = np.random.default_rng(0)
    for i, n in enumerate(lod_sizes):
        lod = splats.create_group(f"lod_{i}")
        lod.attrs.update(
            {
                "n_splats": n,
                "ndim": 3,
                "ordering": "none",
                "lod_stats": {
                    "method": "greedy",
                    "lod_index": i,
                    "n_splats_in_level": n,
                },
            }
        )
        lod.create_dataset("centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
        lod.create_dataset("amplitudes", data=rng.random(n).astype(np.float32))
        lod.create_dataset("cholesky_factors", data=_identity_chol(n))
        lod.create_dataset("chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
    zarr.consolidate_metadata(store)


def _make_substitutive_dir(dir_path: Path, level_sizes: list[int]) -> None:
    """Build a pre-v2.0 substitutive directory layout (level_<i>.gsplats.zarr + manifest.json)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    levels_data = []
    for i, n in enumerate(level_sizes):
        file_name = f"level_{i}.gsplats.zarr"
        _make_v1_0(dir_path / file_name, n)
        levels_data.append({"level": i, "file": file_name, "n_splats": n})
    manifest = {
        "lod_kind": "substitutive",
        "compression_factor": 4,
        "levels": len(level_sizes) - 1,
        "method": "kmeans_lloyd",
        "lloyd_iterations": 5,
        "candidate_bins_k": 12,
        "seed": None,
        "input_n_splats": level_sizes[0],
        "input_ndim": 3,
        "levels_data": levels_data,
    }
    (dir_path / "manifest.json").write_text(json.dumps(manifest))


# ---------------------------------------------------------------------------
# detect_legacy_format
# ---------------------------------------------------------------------------


class TestDetectLegacyFormat:
    def test_detects_v1_0(self, tmp_path: Path) -> None:
        p = tmp_path / "v1_0.gsplats.zarr"
        _make_v1_0(p, n=5)
        assert detect_legacy_format(p) == "v1.0"

    def test_detects_v1_1(self, tmp_path: Path) -> None:
        p = tmp_path / "v1_1.gsplats.zarr"
        _make_v1_1(p, lod_sizes=[8, 4, 2])
        assert detect_legacy_format(p) == "v1.1"

    def test_detects_substitutive_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "sub_pyr"
        _make_substitutive_dir(d, level_sizes=[16, 4, 1])
        assert detect_legacy_format(d) == "substitutive_dir"

    def test_detects_v2_0(self, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        p = tmp_path / "v2.gsplats.zarr"
        rng = np.random.default_rng(0)
        n = 5
        data = GSplatData(
            centers=(rng.random((n, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            cholesky_factors=_identity_chol(n),
        )
        data.save(p)
        assert detect_legacy_format(p) == "v2.0"

    def test_raises_on_unknown_input(self, tmp_path: Path) -> None:
        d = tmp_path / "garbage"
        d.mkdir()
        with pytest.raises(ValueError, match="Unrecognised input layout"):
            detect_legacy_format(d)


# ---------------------------------------------------------------------------
# _read_v1_x_root and _read_substitutive_directory (internals)
# ---------------------------------------------------------------------------


class TestReadLegacyRoots:
    def test_read_v1_0_root(self, tmp_path: Path) -> None:
        p = tmp_path / "v1_0.gsplats.zarr"
        _make_v1_0(p, n=7)
        root = zarr.open_group(str(p), mode="r")
        data, fitting, config, prov = _read_v1_x_root(root, include_stats=True)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        assert data.n_splats == 7
        assert data.ndim == 3
        assert fitting == {} and config == {} and prov == {}

    def test_read_v1_0_root_with_fitting(self, tmp_path: Path) -> None:
        p = tmp_path / "v1_0_fit.gsplats.zarr"
        _make_v1_0(p, n=4, with_fitting=True)
        root = zarr.open_group(str(p), mode="r")
        _, fitting, config, prov = _read_v1_x_root(root, include_stats=True)
        assert fitting["fitter_name"] == "test-fitter"
        assert config["preset"] == "draft"
        assert prov["source_file"] == "test.tiff"

    def test_read_v1_1_root(self, tmp_path: Path) -> None:
        p = tmp_path / "v1_1.gsplats.zarr"
        _make_v1_1(p, lod_sizes=[6, 4, 2])
        root = zarr.open_group(str(p), mode="r")
        data, *_ = _read_v1_x_root(root)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 3
        assert [s.n_splats for s in data.additive_sublods] == [6, 4, 2]

    def test_read_v1_x_rejects_v2_0(self, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        p = tmp_path / "v2.gsplats.zarr"
        GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=_identity_chol(3),
        ).save(p)
        root = zarr.open_group(str(p), mode="r")
        with pytest.raises(ValueError, match="format_version 1.0 or 1.1"):
            _read_v1_x_root(root)

    def test_read_substitutive_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "pyr"
        _make_substitutive_dir(d, level_sizes=[16, 4, 1])
        data = _read_substitutive_directory(d)
        assert data.n_substitutive == 3
        # Each substitutive level has exactly 1 additive sub-LOD
        assert all(s.n_additive_lods == 1 for s in data.substitutive_levels)
        # compression_factor is K^level_idx with K=4
        assert [s.compression_factor for s in data.substitutive_levels] == [1, 4, 16]
        # n_splats per level matches manifest
        assert [s.n_splats_total for s in data.substitutive_levels] == [16, 4, 1]
        # parent_method is None on finest, "kmeans_lloyd" on coarser levels
        assert data.substitutive_levels[0].parent_method is None
        assert data.substitutive_levels[1].parent_method == "kmeans_lloyd"
        assert data.substitutive_levels[2].parent_method == "kmeans_lloyd"

    def test_read_substitutive_directory_missing_file(self, tmp_path: Path) -> None:
        d = tmp_path / "pyr"
        _make_substitutive_dir(d, level_sizes=[8, 2])
        (d / "level_1.gsplats.zarr").rename(d / "level_1_renamed.gsplats.zarr")
        with pytest.raises(FileNotFoundError, match="level_1.gsplats.zarr"):
            _read_substitutive_directory(d)

    def test_read_substitutive_directory_scrambled_manifest_order(
        self, tmp_path: Path
    ) -> None:
        """H4: a manifest that lists levels out of order must still produce a
        finest-first hierarchy (substitutive_levels[0] == compression 1)."""
        d = tmp_path / "pyr"
        _make_substitutive_dir(d, level_sizes=[16, 4, 1])
        # Rewrite the manifest with levels_data scrambled: [coarsest, finest, mid].
        manifest = json.loads((d / "manifest.json").read_text())
        ld = {int(e["level"]): e for e in manifest["levels_data"]}
        manifest["levels_data"] = [ld[2], ld[0], ld[1]]
        (d / "manifest.json").write_text(json.dumps(manifest))

        data = _read_substitutive_directory(d)
        # Finest (level 0, 16 splats, compression 1) is at index 0.
        assert [s.n_splats_total for s in data.substitutive_levels] == [16, 4, 1]
        assert [s.compression_factor for s in data.substitutive_levels] == [1, 4, 16]
        assert data.n_splats == 16  # default substitutive view = finest

    def test_read_substitutive_directory_noncontiguous_levels_raise(
        self, tmp_path: Path
    ) -> None:
        """H4: non-contiguous level indices are rejected, not mis-ordered."""
        d = tmp_path / "pyr"
        _make_substitutive_dir(d, level_sizes=[16, 4])
        manifest = json.loads((d / "manifest.json").read_text())
        manifest["levels_data"][1]["level"] = 5  # gap: levels {0, 5}
        (d / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="contiguous"):
            _read_substitutive_directory(d)


# ---------------------------------------------------------------------------
# End-to-end migrate_format
# ---------------------------------------------------------------------------


class TestMigrateFormat:
    def test_migrate_v1_0_to_v2(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=10)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v1.0"
        # Out is loadable as v2.0
        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["format_version"] == "2.0"
        # Shape is [1, 1]
        data = load_gsplats(out)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        assert data.n_splats == 10

    def test_migrate_v1_0_preserves_fitting_and_provenance(
        self, tmp_path: Path
    ) -> None:
        legacy = tmp_path / "legacy_fit.gsplats.zarr"
        _make_v1_0(legacy, n=4, with_fitting=True)
        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)
        root = zarr.open_group(str(out), mode="r")
        assert "fitting" in root
        assert root["fitting"].attrs["fitter_name"] == "test-fitter"
        assert "config" in root["fitting"]
        assert root["fitting"]["config"].attrs["preset"] == "draft"
        assert "provenance" in root
        assert root["provenance"].attrs["source_file"] == "test.tiff"

    def test_migrate_v1_1_to_v2(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_1(legacy, lod_sizes=[8, 4, 2])
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v1.1"
        data = load_gsplats(out)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 3
        # Splats per additive sub-LOD preserved
        assert [s.n_splats for s in data.additive_sublods] == [8, 4, 2]

    def test_migrate_substitutive_directory_to_v2(self, tmp_path: Path) -> None:
        legacy = tmp_path / "pyr"
        _make_substitutive_dir(legacy, level_sizes=[16, 4, 1])
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "substitutive_dir"
        data = load_gsplats(out)
        assert data.n_substitutive == 3
        # Each substitutive level has 1 additive sub-LOD
        for level in data.substitutive_levels:
            assert level.n_additive_lods == 1
        # Splats per level preserved
        assert [s.n_splats_total for s in data.substitutive_levels] == [16, 4, 1]
        # compression_factor metadata round-trips
        assert [s.compression_factor for s in data.substitutive_levels] == [1, 4, 16]

    def test_migrate_refuses_v2_0_input(self, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        legacy = tmp_path / "v2.gsplats.zarr"
        GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=_identity_chol(3),
        ).save(legacy)
        out = tmp_path / "out.gsplats.zarr"
        with pytest.raises(ValueError, match="already format v2.0"):
            migrate_format(legacy, out)

    def test_migrate_refuses_existing_output(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=3)
        out = tmp_path / "out.gsplats.zarr"
        out.mkdir()
        with pytest.raises(ValueError, match="exists"):
            migrate_format(legacy, out)

    def test_migrate_overwrite(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=3)
        out = tmp_path / "out.gsplats.zarr"
        out.mkdir()
        # No exception with overwrite=True
        migrate_format(legacy, out, overwrite=True)
        # Output is valid
        assert load_gsplats(out).n_splats == 3

    def test_migrate_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            migrate_format(
                tmp_path / "does_not_exist.gsplats.zarr",
                tmp_path / "out.gsplats.zarr",
            )

    def test_migrated_v1_1_numerical_equivalence(self, tmp_path: Path) -> None:
        """Splat arrays survive the round-trip with bitwise equality."""
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_1(legacy, lod_sizes=[5, 3])

        # Read source arrays directly
        src_root = zarr.open_group(str(legacy), mode="r")
        src_centers_0 = np.asarray(src_root["splats"]["lod_0"]["centers"])
        src_centers_1 = np.asarray(src_root["splats"]["lod_1"]["centers"])

        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)
        data = load_gsplats(out)
        np.testing.assert_array_equal(data.additive_sublods[0].centers, src_centers_0)
        np.testing.assert_array_equal(data.additive_sublods[1].centers, src_centers_1)

    # [P5][P1] boundary: v1.0 with the smallest non-degenerate splat count (1)
    def test_migrate_v1_0_single_splat_roundtrip(self, tmp_path: Path) -> None:
        """v1.0 with a single splat migrates to a v2.0 file holding that splat."""
        legacy = tmp_path / "single_v1_0.gsplats.zarr"
        _make_v1_0(legacy, n=1)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v1.0"
        data = load_gsplats(out)
        assert data.n_splats == 1
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        sub = data.additive_sublods[0]
        assert sub.centers.shape == (1, 3)
        assert sub.amplitudes.shape == (1,)
        assert sub.cholesky_factors.shape == (1, 6)

    # [P5] boundary: empty (n=0) was historically blocked by a
    # ZeroDivisionError in compute_chunk_bounds_gsplats; both that crash
    # and the empty migration round-trip are now exercised end-to-end.
    def test_migrate_v1_0_empty_splats_roundtrip(self, tmp_path: Path) -> None:
        """v1.0 with n=0 splats migrates to v2.0 preserving emptiness."""
        legacy = tmp_path / "empty_v1_0.gsplats.zarr"
        _make_v1_0(legacy, n=0)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v1.0"
        data = load_gsplats(out)
        assert data.n_splats == 0
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        sub = data.additive_sublods[0]
        assert sub.centers.shape == (0, 3)
        assert sub.amplitudes.shape == (0,)
        assert sub.cholesky_factors.shape == (0, 6)

    # [P1][P8] full numeric roundtrip for v1.0 (parallels v1.1 test above)
    def test_migrate_v1_0_numerical_equivalence(self, tmp_path: Path) -> None:
        """v1.0 centers survive migration bit-identically; amplitudes and
        cholesky factors survive within their respective quantization
        tolerances (the v2.0 saver applies log/scalar quantization to
        amplitudes and cholesky factors)."""
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=7)
        src_root = zarr.open_group(str(legacy), mode="r")
        src_centers = np.asarray(src_root["splats"]["centers"])
        src_amps = np.asarray(src_root["splats"]["amplitudes"])
        src_chol = np.asarray(src_root["splats"]["cholesky_factors"])

        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)
        data = load_gsplats(out)
        sub = data.additive_sublods[0]
        # Centers are stored as float32 directly — bit-identical
        np.testing.assert_array_equal(sub.centers, src_centers)
        # Amplitudes go through log-scalar quantization — within ~1% of value
        np.testing.assert_allclose(sub.amplitudes, src_amps, rtol=1e-2)
        # Cholesky factors quantized but should remain close
        np.testing.assert_allclose(sub.cholesky_factors, src_chol, atol=1e-2)

    # [P8] colors preservation (metadata roundtrip)
    def test_migrate_v1_0_with_colors_roundtrip(self, tmp_path: Path) -> None:
        """v1.0 with uint8 colors preserves the colors array through migration."""
        legacy = tmp_path / "with_colors_v1_0.gsplats.zarr"
        n = 6
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)

        store = zarr.DirectoryStore(str(legacy))
        root = zarr.group(store=store, overwrite=True)
        root.attrs.update(
            {
                "format_version": "1.0",
                "format_type": "gsplats_zarr",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "luxar_gsplats_version": "test",
            }
        )
        splats = root.create_group("splats")
        splats.attrs.update(
            {
                "type": "gsplats",
                "n_splats": n,
                "ndim": 3,
                "has_colors": True,
                "ordering": "none",
                "truncation_radius": 3.0,
            }
        )
        splats.create_dataset(
            "centers", data=(rng.random((n, 3)) * 10).astype(np.float32)
        )
        splats.create_dataset("amplitudes", data=rng.random(n).astype(np.float32))
        splats.create_dataset("cholesky_factors", data=_identity_chol(n))
        splats.create_dataset(
            "chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32)
        )
        splats.create_dataset("colors", data=colors)
        zarr.consolidate_metadata(store)

        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)
        data = load_gsplats(out)
        loaded_colors = data.additive_sublods[0].colors
        assert loaded_colors is not None
        np.testing.assert_array_equal(loaded_colors, colors)
