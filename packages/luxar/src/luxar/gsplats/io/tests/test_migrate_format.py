"""Tests for the `migrate_format` legacy → current-format conversion tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import create_array
from luxar._zarr_compat import open_group as zc_open_group
from luxar.gsplats.io import load_gsplats
from luxar.gsplats.io.migrate import (
    _read_substitutive_directory,
    _read_v1_x_root,
    _read_v2_0_root,
    detect_legacy_format,
    migrate_format,
)
from luxar.typing_utils._format_contract import GSPLATS_FORMAT_VERSION

# ---------------------------------------------------------------------------
# Helpers to build legacy-format fixtures
# ---------------------------------------------------------------------------


def _identity_chol(n: int) -> np.ndarray:
    """Return n rows of the cholesky factor for an isotropic 3D Gaussian."""
    row = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.tile(row, (n, 1))


def _make_v1_0(path: Path, n: int, *, with_fitting: bool = False) -> None:
    """Build a v1.0 `.gsplats.zarr` (single flat splat set)."""
    store = zarr.storage.LocalStore(str(path))
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
    create_array(splats, "centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
    create_array(splats, "amplitudes", data=rng.random(n).astype(np.float32))
    create_array(splats, "cholesky_factors", data=_identity_chol(n))
    create_array(splats, "chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
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
    zc_consolidate(store)


def _make_v1_1(path: Path, lod_sizes: list[int]) -> None:
    """Build a v1.1 `.gsplats.zarr` (multi-LOD additive)."""
    store = zarr.storage.LocalStore(str(path))
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
        create_array(lod, "centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
        create_array(lod, "amplitudes", data=rng.random(n).astype(np.float32))
        create_array(lod, "cholesky_factors", data=_identity_chol(n))
        create_array(lod, "chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
    zc_consolidate(store)


def _make_v2_0(path: Path, n: int) -> None:
    """Build a v2.0 `.gsplats.zarr` (substitutive_0/additive_0 matrix, [1, 1])."""
    store = zarr.storage.LocalStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(
        {
            "format_version": "2.0",
            "format_type": "gsplats_zarr",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "luxar_gsplats_version": "test",
            "n_substitutive": 1,
            "default_substitutive": 0,
        }
    )
    splats = root.create_group("splats")
    splats.attrs.update(
        {
            "type": "gsplats",
            "n_substitutive": 1,
            "default_substitutive": 0,
            "truncation_radius": 3.0,
        }
    )
    sub = splats.create_group("substitutive_0")
    sub.attrs.update(
        {
            "n_additive_sublods": 1,
            "compression_factor": 1,
            "parent_method": "",
            "level_index": 0,
        }
    )
    add = sub.create_group("additive_0")
    add.attrs.update(
        {"n_splats": n, "ndim": 3, "has_colors": False, "ordering": "none"}
    )
    rng = np.random.default_rng(0)
    create_array(add, "centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
    create_array(add, "amplitudes", data=rng.random(n).astype(np.float32))
    create_array(add, "cholesky_factors", data=_identity_chol(n))
    create_array(add, "chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
    zc_consolidate(store)


def _make_v2_0_multi(path: Path, level_sizes: list[int]) -> None:
    """Build a multi-substitutive v2.0 matrix (substitutive_<s>/additive_0).

    Level 0 is finest (largest); coarser levels follow — the v2.0 convention.
    """
    store = zarr.storage.LocalStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    n_sub = len(level_sizes)
    root.attrs.update(
        {
            "format_version": "2.0",
            "format_type": "gsplats_zarr",
            "n_substitutive": n_sub,
            "default_substitutive": 0,
        }
    )
    splats = root.create_group("splats")
    splats.attrs.update(
        {
            "type": "gsplats",
            "n_substitutive": n_sub,
            "default_substitutive": 0,
            "truncation_radius": 3.0,
        }
    )
    rng = np.random.default_rng(0)
    for s, n in enumerate(level_sizes):
        sub = splats.create_group(f"substitutive_{s}")
        sub.attrs.update(
            {
                "n_additive_sublods": 1,
                "compression_factor": 4**s,
                "parent_method": "" if s == 0 else "kmeans_lloyd",
                "level_index": s,
            }
        )
        add = sub.create_group("additive_0")
        add.attrs.update(
            {"n_splats": n, "ndim": 3, "has_colors": False, "ordering": "none"}
        )
        create_array(add, "centers", data=(rng.random((n, 3)) * 10).astype(np.float32))
        create_array(add, "amplitudes", data=rng.random(n).astype(np.float32))
        create_array(add, "cholesky_factors", data=_identity_chol(n))
    zc_consolidate(store)


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
        p = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(p, n=5)
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
        p = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(p, n=3)
        root = zarr.open_group(str(p), mode="r")
        with pytest.raises(ValueError, match="format_version 1.0 or 1.1"):
            _read_v1_x_root(root)

    def test_read_v2_0_root(self, tmp_path: Path) -> None:
        p = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(p, n=9)
        root = zarr.open_group(str(p), mode="r")
        data, fitting, config, prov = _read_v2_0_root(root, include_stats=True)
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1
        assert data.n_splats == 9
        assert data.ndim == 3

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
    def test_migrate_v1_0_to_v3(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=10)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v1.0"
        # Out is a current node-tree leaf with the split Cholesky layout.
        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
        assert "cholesky_factors_diag" in root
        assert "cholesky_factors" not in root
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

    def test_migrate_v2_0_to_v3(self, tmp_path: Path) -> None:
        legacy = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(legacy, n=12)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v2.0"
        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
        data = load_gsplats(out)
        assert data.n_splats == 12
        assert data.n_substitutive == 1
        assert data.n_additive_sublods == 1

    def test_migrate_preserves_zip_container(self, tmp_path: Path) -> None:
        """A ``.zip`` output is written as a compressed v3 archive (not a bare
        directory) — so migrating a compressed legacy file stays compressed.
        STORED by default; loadable as v3."""
        import zipfile

        legacy = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(legacy, n=10)
        out = tmp_path / "migrated.gsplats.zarr.zip"
        detected = migrate_format(legacy, out)
        assert detected == "v2.0"
        assert out.is_file()  # a zip FILE, not a directory
        with zipfile.ZipFile(out) as z:
            # default is STORED (matches the bundled demo archives)
            assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        data = load_gsplats(out)  # loader extracts + reads v3
        assert data.n_splats == 10

    def test_migrate_zip_deflate_flag(self, tmp_path: Path) -> None:
        import zipfile

        legacy = tmp_path / "v2.gsplats.zarr"
        _make_v2_0(legacy, n=8)
        out = tmp_path / "deflated.gsplats.zarr.zip"
        migrate_format(legacy, out, zip_deflate=True)
        with zipfile.ZipFile(out) as z:
            assert any(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())
        assert load_gsplats(out).n_splats == 8

    def test_migrate_v2_0_multi_preserves_orientation(self, tmp_path: Path) -> None:
        """Multi-substitutive v2.0 (finest=level0) migrates to v3.0 with the
        finest level still at substitutive_levels[0] — the reversal trap must
        round-trip through migrate → write (coarsest-first child_<i>) → read."""
        legacy = tmp_path / "v2multi.gsplats.zarr"
        _make_v2_0_multi(legacy, level_sizes=[64, 16, 4])  # finest..coarsest
        out = tmp_path / "out.gsplats.zarr"
        assert migrate_format(legacy, out) == "v2.0"
        data = load_gsplats(out)
        assert data.n_substitutive == 3
        # Finest level (64 splats, compression 1) restored at index 0.
        assert [s.n_splats_total for s in data.substitutive_levels] == [64, 16, 4]
        assert [s.compression_factor for s in data.substitutive_levels] == [1, 4, 16]
        assert data.n_splats == 64  # default view = finest

    def test_migrate_compressed_zip_archive(self, tmp_path: Path) -> None:
        """A .gsplats.zarr.zip legacy archive (the committed-demo shape) migrates
        end-to-end (M3)."""
        import zipfile

        src_dir = tmp_path / "legacy.gsplats.zarr"
        _make_v1_1(src_dir, lod_sizes=[8, 4])
        archive = tmp_path / "legacy.gsplats.zarr.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for f in src_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(src_dir.parent))

        out = tmp_path / "out.gsplats.zarr"
        assert migrate_format(archive, out) == "v1.1"
        data = load_gsplats(out)
        assert data.n_additive_sublods == 2
        assert [s.n_splats for s in data.additive_sublods] == [8, 4]

    def test_migrate_refuses_v3_input(self, tmp_path: Path) -> None:
        from luxar.gsplats import GSplatData

        current = tmp_path / "v3.gsplats.zarr"
        GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=_identity_chol(3),
        ).save(current)  # writes the current (v3.4) node-tree
        out = tmp_path / "out.gsplats.zarr"
        with pytest.raises(ValueError, match="already format v3"):
            migrate_format(current, out)

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
        # Centers migrate through AUTO (uint16 per-axis fixed-point) — near-lossless.
        np.testing.assert_allclose(
            data.additive_sublods[0].centers,
            src_centers_0,
            atol=float(np.ptp(src_centers_0, axis=0).max()) / 65535 * 2,
        )
        np.testing.assert_allclose(
            data.additive_sublods[1].centers,
            src_centers_1,
            atol=float(np.ptp(src_centers_1, axis=0).max()) / 65535 * 2,
        )

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

    def test_migrate_v1_0_empty_splats_rejected(self, tmp_path: Path) -> None:
        """v1.0 with n=0 splats is rejected by the unified writer.

        The v3.0 writer shares the scene's array validation, which refuses an
        empty splat set (``fit`` always produces ≥1 splat). This aligns the
        standalone format with the scene format's no-empty policy.
        """
        legacy = tmp_path / "empty_v1_0.gsplats.zarr"
        _make_v1_0(legacy, n=0)
        out = tmp_path / "out.gsplats.zarr"
        with pytest.raises(Exception, match="(?i)empty"):
            migrate_format(legacy, out)

    # [P1][P8] full numeric roundtrip for v1.0 (parallels v1.1 test above)
    def test_migrate_v1_0_numerical_equivalence(self, tmp_path: Path) -> None:
        """v1.0 centers survive migration within uint16 fixed-point tolerance; amplitudes and
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
        # Centers migrate through AUTO (uint16 per-axis fixed-point) — near-lossless.
        np.testing.assert_allclose(
            sub.centers,
            src_centers,
            atol=float(np.ptp(src_centers, axis=0).max()) / 65535 * 2,
        )
        # Amplitudes go through log-scalar quantization — within ~1% of value
        np.testing.assert_allclose(sub.amplitudes, src_amps, rtol=1e-2)
        # Cholesky factors quantized but should remain close. The default AUTO
        # policy re-encodes float32 Cholesky as uint16 per-column log
        # (near-lossless ~0.1% rel.); this contrived fixture has a narrow range
        # so atol=1e-2 is comfortably loose for it.
        np.testing.assert_allclose(sub.cholesky_factors, src_chol, atol=1e-2)

    def test_migrate_lossless_preserves_cholesky_float32(self, tmp_path: Path) -> None:
        """encoding_mode=PRECISION (`--lossless`) threads through migration and
        stores Cholesky factors as float32 (bit-identical), not quantized uint.
        (AUTO's near-lossless quantization on varied data is covered by
        encoding/tests/test_cholesky_split_quant.py.)"""
        from luxar.encoding import EncodingMode

        legacy = tmp_path / "legacy.gsplats.zarr"
        _make_v1_0(legacy, n=9)
        src_chol = np.asarray(
            zarr.open_group(str(legacy), mode="r")["splats"]["cholesky_factors"]
        )

        out_lossless = tmp_path / "lossless.gsplats.zarr"
        migrate_format(legacy, out_lossless, encoding_mode=EncodingMode.PRECISION)
        lossless = load_gsplats(out_lossless).additive_sublods[0]
        np.testing.assert_array_equal(lossless.cholesky_factors, src_chol)

    # [P8] colors preservation (metadata roundtrip)
    def test_migrate_v1_0_with_colors_roundtrip(self, tmp_path: Path) -> None:
        """v1.0 with uint8 colors preserves the colors array through migration."""
        legacy = tmp_path / "with_colors_v1_0.gsplats.zarr"
        n = 6
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)

        store = zarr.storage.LocalStore(str(legacy))
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
        create_array(
            splats, "centers", data=(rng.random((n, 3)) * 10).astype(np.float32)
        )
        create_array(splats, "amplitudes", data=rng.random(n).astype(np.float32))
        create_array(splats, "cholesky_factors", data=_identity_chol(n))
        create_array(splats, "chunk_bounds", data=np.zeros((1, 3, 2), dtype=np.float32))
        create_array(splats, "colors", data=colors)
        zc_consolidate(store)

        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)
        data = load_gsplats(out)
        loaded_colors = data.additive_sublods[0].colors
        assert loaded_colors is not None
        np.testing.assert_array_equal(loaded_colors, colors)


# ---------------------------------------------------------------------------
# v3.0 / v3.1 stores with pre-v3.2 lod selector attrs (pixel_size → coverage)
# ---------------------------------------------------------------------------


def _make_v3_lod_pixel_size(
    path: Path,
    level_sizes: list[int],
    *,
    format_version: str = "3.1",
    with_fitting: bool = False,
) -> None:
    """Build a v3.x node-tree ``kind=lod`` store carrying the pre-v3.2 selector
    attrs: ``selector='pixel_size'`` on the group and per-child
    ``min_pixel_size`` (no ``coverage_fraction``).

    Written with the current writer, then attr-rewritten into the legacy form
    (the array layout is identical across v3.0→v3.2; only the selector attrs
    were renamed).
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    rng = np.random.default_rng(0)
    leaves = []
    for n in level_sizes:  # coarsest→finest (the on-disk child_<i> order)
        leaves.append(
            GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=(rng.random((n, 3)) * 10).astype(np.float32),
                        amplitudes=rng.random(n).astype(np.float32),
                        cholesky_factors=_identity_chol(n),
                    )
                ]
            )
        )
    write_gsplats_tree(
        path,
        GSplatLodGroup(children=leaves),
        fitting_info={"fitter_name": "test-fitter", "n_splats": level_sizes[-1]}
        if with_fitting
        else None,
        pipeline_info={"lod_kind": "substitutive"} if with_fitting else None,
    )
    # Aged through the facade — see the note in
    # `test_detect_nested_legacy_lod_inside_partition`: a plain re-open leaves a
    # stale NESTED consolidated index that later reads prefer over the correct
    # per-node documents, so the deletions below would not be observed.
    root = zc_open_group(str(path), mode="r+")
    root.attrs["format_version"] = format_version
    root.attrs["selector"] = "pixel_size"
    for i, n in enumerate(level_sizes):
        child = root[f"child_{i}"]
        del child.attrs["coverage_fraction"]
        # The legacy count-anchored ladder: base(100px)·sqrt(N_i/N_0).
        child.attrs["min_pixel_size"] = 100.0 * float(np.sqrt(n / level_sizes[0]))
    zc_consolidate(root)


class TestMigrateV3LegacyLodAttrs:
    """v3.0/v3.1 stores whose kind=lod groups still carry the pre-v3.2
    'pixel_size' selector attrs are detected and rewritten to the current format
    (selector='screen-area' + re-derived screen-area coverage_fraction
    thresholds — occupancy halving)."""

    def test_detect_v3_1_legacy_lod(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy_lod.gsplats.zarr"
        _make_v3_lod_pixel_size(legacy, [2, 8])
        assert detect_legacy_format(legacy) == "v3.1-lod-pixel-size"

    def test_detect_v3_0_legacy_lod(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy_lod.gsplats.zarr"
        _make_v3_lod_pixel_size(legacy, [2, 8], format_version="3.0")
        assert detect_legacy_format(legacy) == "v3.0-lod-pixel-size"

    def test_detect_refuses_current_lod_store(self, tmp_path: Path) -> None:
        """A lod store with current coverage attrs is NOT migratable."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

        rng = np.random.default_rng(0)
        leaves = [
            GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=(rng.random((n, 3)) * 10).astype(np.float32),
                        amplitudes=rng.random(n).astype(np.float32),
                        cholesky_factors=_identity_chol(n),
                    )
                ]
            )
            for n in (2, 8)
        ]
        current = tmp_path / "current_lod.gsplats.zarr"
        write_gsplats_tree(current, GSplatLodGroup(children=leaves))
        with pytest.raises(ValueError, match="already format v3"):
            detect_legacy_format(current)

    def test_migrate_v3_1_lod_pixel_size(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy_lod.gsplats.zarr"
        _make_v3_lod_pixel_size(legacy, [2, 4, 16], with_fitting=True)
        out = tmp_path / "out.gsplats.zarr"
        detected = migrate_format(legacy, out)
        assert detected == "v3.1-lod-pixel-size"

        root = zarr.open_group(str(out), mode="r")
        assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
        assert root.attrs["kind"] == "lod"
        # Migration re-derives SCREEN-AREA thresholds (occupancy halving).
        assert root.attrs["selector"] == "screen-area"
        fractions = []
        for i in range(3):
            child_attrs = dict(root[f"child_{i}"].attrs)
            assert "min_pixel_size" not in child_attrs
            fractions.append(float(child_attrs["coverage_fraction"]))
        # Derived occupancy halving (whole-object): [0, 0.25, 0.5] — strictly
        # ascending, finest at the half-screen-area anchor.
        from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR

        assert fractions == sorted(fractions)
        assert all(a < b for a, b in zip(fractions, fractions[1:]))
        assert fractions == pytest.approx([0.0, 0.25, 0.5])
        assert fractions[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
        # Carry-along groups survive the rewrite.
        assert root["fitting"].attrs["fitter_name"] == "test-fitter"
        assert root["pipeline"].attrs["lod_kind"] == "substitutive"

        # Loadable as a substitutive matrix with counts preserved.
        data = load_gsplats(out)
        assert data.n_substitutive == 3
        assert data.n_splats == 16  # default view = finest

    def test_migrate_keeps_the_authored_appearance(self, tmp_path: Path) -> None:
        """A migration rewrites the LAYOUT and must leave the look alone (#1600).

        The rewrite goes through the same node-tree writer as the rest of the
        rewriting family (`lod`, `flatten`, `reencode`, ...), so without the
        carry the writer's defaults take over: ``blending_mode`` disappears and
        the multiplicative attrs snap back to 1.0. Every authored value here is
        non-default on purpose — an identity would coincide with the stamped
        default and hide the drop.
        """
        legacy = tmp_path / "legacy_lod.gsplats.zarr"
        _make_v3_lod_pixel_size(legacy, [2, 8])
        authored = {"blending_mode": "volumetric", "opacity": 0.75, "gamma": 1.3}
        src = zarr.open_group(str(legacy), mode="r+")
        for key, value in authored.items():
            src.attrs[key] = value
        zarr.consolidate_metadata(src.store)

        out = tmp_path / "out.gsplats.zarr"
        migrate_format(legacy, out)

        got = dict(zarr.open_group(str(out), mode="r").attrs)
        for key, want in authored.items():
            assert got.get(key) == want, f"dropped {key!r} (had {want!r})"
        # The layout upgrade itself still happened — the carry rides the
        # writer's lowest-precedence channel, so it cannot shadow structure.
        assert got["selector"] == "screen-area"
        assert got["format_version"] == GSPLATS_FORMAT_VERSION

    def test_detect_nested_legacy_lod_inside_partition(self, tmp_path: Path) -> None:
        """The legacy-attr scan recurses: a kind=partition root whose part is a
        legacy-attr lod group is detected (and migrates) too."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

        rng = np.random.default_rng(0)
        label_vocabulary = {index: f"class-{index}" for index in range(8)}

        def leaf(n: int) -> GSplatLeaf:
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=(rng.random((n, 3)) * 10).astype(np.float32),
                        amplitudes=rng.random(n).astype(np.float32),
                        cholesky_factors=_identity_chol(n),
                        label_ids=np.arange(n, dtype=np.uint8),
                        label_vocabulary=label_vocabulary,
                    )
                ]
            )

        legacy = tmp_path / "nested.gsplats.zarr"
        write_gsplats_tree(
            legacy,
            GSplatPartition(
                children=[GSplatLodGroup(children=[leaf(2), leaf(8)])],
                max_elements=0,
            ),
        )
        # Age the store into a v3.1 legacy shape THROUGH THE FACADE, which is
        # how Luxar itself re-opens a store to edit attributes in place.
        #
        # Not cosmetic. `zarr.open_group` trusts consolidated metadata, so the
        # nodes it hands back are built from the root index; re-consolidating
        # from that tree writes a SECOND, nested consolidated index into
        # `part_0/zarr.json` carrying the pre-edit attributes. At format 3 a
        # nested index is honoured even when the root one is bypassed, so the
        # migration then re-derives from `coverage_fraction` values this
        # function had just deleted and stamps the legacy `selector="coverage"`.
        #
        # Format 2 grows the nested index too, but reads there are unaffected:
        # `use_consolidated=False` skips a `.zmetadata` at EVERY level, while at
        # format 3 the index lives inside each `zarr.json` and only the root's
        # is bypassed. That asymmetry is the whole bug, not the extra document.
        # A real v3.0/v3.1 store carries exactly ONE index, at its root — what
        # the facade produces here.
        root = zc_open_group(str(legacy), mode="r+")
        root.attrs["format_version"] = "3.1"
        lod = root["part_0"]
        lod.attrs["selector"] = "pixel_size"
        for i, n in enumerate((2, 8)):
            del lod[f"child_{i}"].attrs["coverage_fraction"]
            lod[f"child_{i}"].attrs["min_pixel_size"] = 100.0 * float(np.sqrt(n / 2))
        zc_consolidate(root)

        assert detect_legacy_format(legacy) == "v3.1-lod-pixel-size"
        out = tmp_path / "out.gsplats.zarr"
        assert migrate_format(legacy, out) == "v3.1-lod-pixel-size"
        out_root = zarr.open_group(str(out), mode="r")
        assert out_root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
        # The re-derived nested ladder carries the screen-area selector too.
        assert out_root["part_0"].attrs["selector"] == "screen-area"
        assert "coverage_fraction" in out_root["part_0"]["child_0"].attrs
        assert "min_pixel_size" not in out_root["part_0"]["child_0"].attrs
        for index, count in enumerate((2, 8)):
            child = out_root["part_0"][f"child_{index}"]
            np.testing.assert_array_equal(
                np.sort(child["label_ids"][:]), np.arange(count, dtype=np.uint8)
            )
            assert {
                int(label_id): name
                for label_id, name in child.attrs["label_vocabulary"].items()
            } == label_vocabulary
