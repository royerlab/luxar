"""Tests for demo scene generators.

Tests cover:
- create_lorenz_attractor: Generates Lorenz attractor visualization
- create_random_spheres: Generates random colored spheres
- create_time_series_demo: Generates 4D time series demo
"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos.demo_lorenz import lorenz_trajectory
from luxar.encoding import ArrayDecoder
from luxar.utils.demos import (
    _safe_extract_zip_member,
    _validate_zip_member_path,
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)


def _max_nearest_neighbor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a point of ``a`` to its nearest point in ``b``.

    Comparing two point sets row-by-row needs them in the same order, and the
    compiler reorders points along a Hilbert curve. Matching whole points to
    their nearest counterpart recovers the correspondence without relying on
    order — and unlike a per-axis comparison it keeps each point's three
    coordinates tied together.
    """
    d = np.linalg.norm(
        a[:, None, :].astype(np.float64) - b[None, :, :].astype(np.float64), axis=-1
    )
    return float(d.min(axis=1).max())


class TestCreateLorenzAttractor:
    """Tests for create_lorenz_attractor function."""

    def test_basic_creation(self) -> None:
        """Test basic Lorenz attractor creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=100)

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure
            store = zarr.open(store_path, mode="r")
            assert "LorenzAttractor" in store
            assert "positions" in store["LorenzAttractor"]

            # Check positions shape
            positions = store["LorenzAttractor"]["positions"][:]
            assert positions.shape[0] == 100
            assert positions.shape[1] == 3  # 3D

    def test_with_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "lorenz1.luxar.zarr"
            store_path2 = Path(tmpdir) / "lorenz2.luxar.zarr"

            create_lorenz_attractor(store_path1, n_points=50, seed=42)
            create_lorenz_attractor(store_path2, n_points=50, seed=42)

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            # Positions should be identical
            pos1 = store1["LorenzAttractor"]["positions"][:]
            pos2 = store2["LorenzAttractor"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_positions_come_from_the_demo_integrator(self) -> None:
        """The fixture and ``luxar demo run lorenz`` share one integrator.

        Guards the dedup: if either side re-inlines or re-parameterises the
        Lorenz integration, the two stop agreeing and this fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=500, seed=42)

            store = zarr.open(store_path, mode="r")
            written = ArrayDecoder().decode(
                store["LorenzAttractor"]["positions"], store
            )

            # Same trajectory, scaled up for visibility exactly as the builder
            # does. Matched point-to-point (see _max_nearest_neighbor_distance);
            # the tolerance covers the uint16 position quantization, whose step
            # is ~0.006 over this extent. Both directions, so neither set may
            # contain a point the other lacks.
            expected = lorenz_trajectory(500, seed=42) * 100.0 - 50.0
            assert written.shape == expected.shape
            assert _max_nearest_neighbor_distance(written, expected) < 0.05
            assert _max_nearest_neighbor_distance(expected, written) < 0.05

    def test_has_colors_and_radii(self) -> None:
        """Test that colors and radii are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=100)

            store = zarr.open(store_path, mode="r")
            lorenz_group = store["LorenzAttractor"]

            assert "colors" in lorenz_group
            assert "radii" in lorenz_group

            colors = lorenz_group["colors"][:]
            # Colors are SDR RGB
            assert colors.shape[1] == 3

    def test_different_point_counts(self) -> None:
        """Test creation with different point counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in [10, 100, 1000]:
                store_path = Path(tmpdir) / f"lorenz_{n}.luxar.zarr"
                create_lorenz_attractor(store_path, n_points=n)

                store = zarr.open(store_path, mode="r")
                positions = store["LorenzAttractor"]["positions"][:]
                assert positions.shape[0] == n


class TestCreateRandomSpheres:
    """Tests for create_random_spheres function."""

    def test_basic_creation(self) -> None:
        """Test basic random spheres creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=5, points_per_sphere=50, seed=42
            )

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure - should have sphere_000, sphere_001, etc.
            store = zarr.open(store_path, mode="r")
            for i in range(5):
                sphere_name = f"sphere_{i:03d}"
                assert sphere_name in store, f"Missing {sphere_name}"

    def test_sphere_point_counts(self) -> None:
        """Test that each sphere has correct number of points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            n_spheres = 3
            points_per_sphere = 100

            create_random_spheres(
                store_path,
                n_spheres=n_spheres,
                points_per_sphere=points_per_sphere,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            for i in range(n_spheres):
                sphere_name = f"sphere_{i:03d}"
                positions = store[sphere_name]["positions"][:]
                assert positions.shape[0] == points_per_sphere

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible spheres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "spheres1.luxar.zarr"
            store_path2 = Path(tmpdir) / "spheres2.luxar.zarr"

            create_random_spheres(
                store_path1, n_spheres=3, points_per_sphere=50, seed=123
            )
            create_random_spheres(
                store_path2, n_spheres=3, points_per_sphere=50, seed=123
            )

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            # First sphere positions should be identical
            pos1 = store1["sphere_000"]["positions"][:]
            pos2 = store2["sphere_000"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_3d_positions(self) -> None:
        """Test that sphere positions are 3D."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=1, points_per_sphere=100, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["sphere_000"]["positions"][:]
            assert positions.shape[1] == 3  # 3D


class TestCreateTimeSeriesDemo:
    """Tests for create_time_series_demo function."""

    def test_basic_creation(self) -> None:
        """Test basic time series demo creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure
            store = zarr.open(store_path, mode="r")
            assert "time_series" in store

    def test_4d_positions(self) -> None:
        """Test that positions are 4D (x, y, z, time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            assert positions.shape[1] == 4  # 4D (x, y, z, time)

    def test_total_point_count(self) -> None:
        """Test that total points = n_timepoints * n_points_per_time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            n_timepoints = 5
            n_points_per_time = 100

            create_time_series_demo(
                store_path,
                n_timepoints=n_timepoints,
                n_points_per_time=n_points_per_time,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            expected_total = n_timepoints * n_points_per_time
            assert positions.shape[0] == expected_total

    def test_time_dimension_values(self) -> None:
        """Test that time dimension has correct integer values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            n_timepoints = 5
            n_points_per_time = 50

            create_time_series_demo(
                store_path,
                n_timepoints=n_timepoints,
                n_points_per_time=n_points_per_time,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            # Positions are uint16 per-axis fixed-point under the default AUTO
            # mode — decode (raw levels are meaningless integers) before reading
            # the time column.
            positions = ArrayDecoder().decode(store["time_series"]["positions"], store)

            # Time is the 4th dimension (index 3)
            time_values = positions[:, 3]
            unique_times = np.unique(time_values)

            # Should have exactly n_timepoints unique time values (each distinct
            # input value maps to one quantization code, so the count survives)
            assert len(unique_times) == n_timepoints
            # Time values should be integers 0 to n_timepoints-1, within one
            # quantization step (extent/65535)
            np.testing.assert_allclose(
                sorted(unique_times), list(range(n_timepoints)), atol=1e-3
            )

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible time series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "ts1.luxar.zarr"
            store_path2 = Path(tmpdir) / "ts2.luxar.zarr"

            create_time_series_demo(
                store_path1, n_timepoints=3, n_points_per_time=50, seed=456
            )
            create_time_series_demo(
                store_path2, n_timepoints=3, n_points_per_time=50, seed=456
            )

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            pos1 = store1["time_series"]["positions"][:]
            pos2 = store2["time_series"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_has_colors(self) -> None:
        """Test that colors are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            assert "colors" in store["time_series"]

            # Colors are encoded, verify encoding metadata exists
            colors_arr = store["time_series"]["colors"]
            # Should have some encoding - could be LUT, broadcasted, or direct
            # The raw shape depends on encoding - just verify presence
            assert colors_arr.shape[0] > 0

    def test_scene_dimensions(self) -> None:
        """Test that scene has 4D dimensions defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            # Check scene dimensions are defined
            assert "scene_dimensions" in store.attrs


class TestEdgeCases:
    """Edge case tests for demo functions."""

    def test_single_point_lorenz(self) -> None:
        """Test Lorenz with single point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=1)

            store = zarr.open(store_path, mode="r")
            positions = store["LorenzAttractor"]["positions"][:]
            assert positions.shape[0] == 1

    def test_single_sphere(self) -> None:
        """Test random spheres with single sphere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=1, points_per_sphere=10, seed=42
            )

            store = zarr.open(store_path, mode="r")
            assert "sphere_000" in store

    def test_single_timepoint_becomes_two(self) -> None:
        """Test time series with single timepoint gets adjusted to 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            # Requesting 1 timepoint should be adjusted to 2
            create_time_series_demo(
                store_path, n_timepoints=1, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            # Should have 2 timepoints * 50 points = 100 points
            assert positions.shape[0] == 100


class TestSafeZipExtraction:
    """Security tests for demo bundle zip extraction helpers."""

    @pytest.mark.parametrize(
        "member",
        ["../evil.zarr.zip", "/absolute/evil.zarr.zip", "dir/../../evil", "dir\\evil"],
    )
    def test_validate_zip_member_rejects_traversal(self, member: str) -> None:
        """Unsafe archive member paths are rejected before extraction."""
        with pytest.raises(ValueError, match="Unsafe|Invalid"):
            _validate_zip_member_path(member)

    def test_safe_extract_zip_member_flattens_safe_member(self, tmp_path: Path) -> None:
        """Safe members can be copied into a controlled cache filename."""
        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("nested/data.gsplats.zarr.zip", b"payload")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with zipfile.ZipFile(bundle, "r") as zf:
            extracted = _safe_extract_zip_member(
                zf,
                "nested/data.gsplats.zarr.zip",
                cache_dir,
                target_name="data.gsplats.zarr.zip",
            )

        assert extracted == cache_dir / "data.gsplats.zarr.zip"
        assert extracted.read_bytes() == b"payload"

    def test_safe_extract_zip_member_rejects_unsafe_target(
        self, tmp_path: Path
    ) -> None:
        """Even a safe archive member cannot be written outside the cache root."""
        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("data.gsplats.zarr.zip", b"payload")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with zipfile.ZipFile(bundle, "r") as zf:
            with pytest.raises(ValueError, match="Unsafe extraction target"):
                _safe_extract_zip_member(
                    zf,
                    "data.gsplats.zarr.zip",
                    cache_dir,
                    target_name="../escape.gsplats.zarr.zip",
                )


class TestCacheStaleness:
    """The demo cache must self-heal when packaged data is re-migrated
    (e.g. the gsplats v2.0 -> v3.0 cutover), not pin the first-seen copy."""

    def test_cache_is_stale_detects_refresh_conditions(self, tmp_path: Path) -> None:
        import os

        from luxar.utils.demos import _cache_is_stale

        cache = tmp_path / "cache.bin"
        source = tmp_path / "source.bin"

        # Missing cache → stale.
        source.write_bytes(b"x" * 100)
        assert _cache_is_stale(cache, source) is True

        # Identical size + same mtime → fresh.
        cache.write_bytes(b"x" * 100)
        os.utime(cache, (source.stat().st_atime, source.stat().st_mtime))
        assert _cache_is_stale(cache, source) is False

        # Source larger (re-migration changed content) → stale.
        source.write_bytes(b"x" * 250)
        os.utime(cache, (source.stat().st_atime, source.stat().st_mtime))
        assert _cache_is_stale(cache, source) is True

        # Same size but source newer (in-place rewrite) → stale.
        cache.write_bytes(b"y" * 250)
        old = source.stat().st_mtime - 100
        os.utime(cache, (old, old))
        assert _cache_is_stale(cache, source) is True

        # Source absent (unpulled LFS) → keep the cached copy.
        source.unlink()
        assert _cache_is_stale(cache, source) is False

    def test_load_precomputed_refreshes_a_stale_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A stale cached copy is replaced by the (changed) packaged source."""
        import os

        import luxar.utils.demos as demos
        from luxar.gsplats.gsplat_data import GSplatData

        data_root = tmp_path / "data"
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(demos, "_DEMOS_DATA_DIR", data_root)
        monkeypatch.setattr(demos, "_DEFAULT_CACHE_ROOT", cache_root)

        n = 8
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        src_dir = data_root / "gsplats_x"
        src_dir.mkdir(parents=True)
        src = src_dir / "x.gsplats.zarr.zip"
        GSplatData(
            centers=np.random.rand(n, 3).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        ).save(src, ordering="none", compress="zip")

        # Seed the cache with a STALE, unreadable copy (old mtime).
        (cache_root / "gsplats_x").mkdir(parents=True)
        stale = cache_root / "gsplats_x" / "x.gsplats.zarr.zip"
        stale.write_bytes(b"stale-not-a-zarr")
        os.utime(stale, (0, 0))  # far in the past → source is newer

        out = demos.load_precomputed_gsplats("gsplats_x", ["x.gsplats.zarr.zip"])
        # Refreshed from source and loaded (would raise on the stale bytes).
        assert out is not None and out[0].n_splats == n


class TestUnshippableData:
    """Data we may not redistribute is absent ON PURPOSE.

    Those datasets carry no in-repo copy, so the caller must be routed to its
    own rebuild path instead of being told to run ``git lfs pull`` for a file
    that does not exist in the repository and never will.
    """

    def test_local_compute_dataset_returns_none_instead_of_raising(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.utils.demos as demos

        # Neither an in-repo copy nor a cached one — the post-removal state of
        # every `local-compute` dataset on a fresh clone.
        monkeypatch.setattr(demos, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(demos, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        out = demos.load_precomputed_gsplats(
            "gsplats_tribolium", ["tribolium.gsplats.zarr.zip"]
        )
        assert out is None

    def test_shippable_dataset_still_raises_the_lfs_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A `zenodo` dataset that is merely unpulled must NOT be excused."""
        import luxar.utils.demos as demos

        monkeypatch.setattr(demos, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(demos, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        with pytest.raises(FileNotFoundError):
            demos.load_precomputed_gsplats("gsplats_dapi", ["dapi.gsplats.zarr.zip"])

    def test_unknown_dataset_is_treated_as_shippable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """No manifest entry → no excuse; the ordinary missing-file error wins."""
        import luxar.utils.demos as demos

        monkeypatch.setattr(demos, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(demos, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        assert demos._unshippable_reason("not_a_dataset") is None
        with pytest.raises(FileNotFoundError):
            demos.load_precomputed_gsplats("not_a_dataset", ["x.gsplats.zarr.zip"])


# ───────────────────── derived demo ports (launch_viewer) ─────────────────────
class TestDemoPorts:
    """Stable per-dataset ports so demos never contend for 8000/5173."""

    def test_deterministic_and_in_range(self) -> None:
        from luxar.utils.demos import demo_ports

        data, viewer = demo_ports("global_rivers_earth.luxar.zarr")
        assert (data, viewer) == demo_ports("global_rivers_earth.luxar.zarr")
        # Path components don't matter — only the dataset name does, so the
        # same demo maps to the same URL from any output directory.
        assert (data, viewer) == demo_ports(
            Path("/somewhere/else/global_rivers_earth.luxar.zarr")
        )
        assert 8001 <= data <= 8499
        assert 5200 <= viewer <= 5698
        # Never the bare `luxar serve` defaults.
        assert data != 8000 and viewer != 5173

    def test_different_demos_spread(self) -> None:
        from luxar.utils.demos import demo_ports

        names = [f"demo_{i}.luxar.zarr" for i in range(24)]
        assert len({demo_ports(n) for n in names}) > 20

    def test_serve_command_appends_derived_ports(self) -> None:
        from luxar.utils.demos import _serve_command, demo_ports

        data, viewer = demo_ports("x.luxar.zarr")
        cmd = _serve_command("x.luxar.zarr", open_browser=True, serve_args=None)
        assert cmd[-1] == "--open"
        assert ["--port", str(data)] == cmd[cmd.index("--port") :][:2]
        assert ["--viewer-port", str(viewer)] == cmd[cmd.index("--viewer-port") :][:2]

    def test_serve_command_respects_pinned_ports(self) -> None:
        from luxar.utils.demos import _serve_command

        cmd = _serve_command(
            "x.luxar.zarr",
            open_browser=False,
            serve_args=["--port", "9000"],
        )
        # The explicit pin survives and no second --port is appended.
        assert cmd.count("--port") == 1
        assert cmd[cmd.index("--port") + 1] == "9000"
        assert "--viewer-port" in cmd  # unpinned half still derived

        # The short spelling pins too — Click keeps the LAST occurrence of a
        # repeated option, so a missed pin would silently override the demo.
        for pinned in (["-p", "9100"], ["-p9100"]):
            cmd = _serve_command("x.luxar.zarr", open_browser=False, serve_args=pinned)
            assert "--port" not in cmd
            assert "--viewer-port" in cmd

        cmd = _serve_command(
            "x.luxar.zarr",
            open_browser=False,
            serve_args=["--viewer-port=6000"],
        )
        assert not any(a == "--viewer-port" for a in cmd[cmd.index("--viewer") :][1:])
        assert "--viewer-port=6000" in cmd
        assert "--port" in cmd  # unpinned half still derived

    def test_bundled_demo_table_has_no_full_pair_collisions(self) -> None:
        """No two bundled demo outputs share a full (data, viewer) pair.

        An identical pair reproduces the same-URL stale-tab trap this
        derivation exists to prevent. If adding a demo trips this, rename the
        output or widen the slot ranges.
        """
        from luxar.demos import registry
        from luxar.utils.demos import demo_ports

        # Resolve names through the registry's own rule (a stem already ending
        # in `.zarr` is used verbatim) so the guard keeps checking the names
        # demos actually serve.
        pairs: dict[tuple[int, int], list[str]] = {}
        for d in registry.iter_demos():
            for path in registry.demo_output_paths(d, demos_dir=Path("demos")):
                pairs.setdefault(demo_ports(path), []).append(path.name)
        collisions = {k: v for k, v in pairs.items() if len(v) > 1}
        assert not collisions, f"port-pair collisions: {collisions}"


class TestExtractBundleAndLoad:
    """The extraction body shared by both bundle loaders.

    Split out of ``load_precomputed_bundle`` so the manifest-driven
    ``load_dataset_bundle`` reuses it rather than growing a lookalike. The member
    matching is security-sensitive and the staleness stamp is what stops a
    re-migrated bundle serving stale frames, so both behaviours are pinned here.
    """

    @staticmethod
    def _bundle(path: Path, members: dict[str, bytes]) -> None:
        import zipfile

        with zipfile.ZipFile(path, "w") as zf:
            for name, blob in members.items():
                zf.writestr(name, blob)

    @staticmethod
    def _load_stub(monkeypatch):
        """Return per-file payloads instead of parsing real gsplat stores."""
        from luxar.gsplats import gsplat_data as gd

        class _Stub:
            def __init__(self, blob: bytes) -> None:
                self.blob = blob
                self.amplitudes = blob  # the loader logs len(amplitudes)

            def __eq__(self, other: object) -> bool:
                return self.blob == other

        monkeypatch.setattr(
            gd.GSplatData,
            "load",
            classmethod(lambda cls, p, **kw: _Stub(p.read_bytes())),
        )

    def test_extracts_members_flattened_into_the_cache_dir(self, tmp_path, monkeypatch):
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        # A member nested in a directory must still land flat, by basename.
        self._bundle(b, {"inner/f0.zip": b"zero", "f1.zip": b"one"})
        cache = tmp_path / "cache"
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip", "f1.zip"], validate_lfs=False
        )
        assert out == [b"zero", b"one"]
        assert (cache / "f0.zip").read_bytes() == b"zero"
        assert not (cache / "inner").exists(), "member was not flattened"

    def test_second_call_reuses_the_extraction(self, tmp_path, monkeypatch):
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        cache = tmp_path / "cache"
        du._extract_bundle_and_load(b, "b.zip", cache, ["f0.zip"], validate_lfs=False)
        # Corrupt the extracted copy: an honoured stamp means it is NOT re-extracted.
        (cache / "f0.zip").write_bytes(b"stale-but-present")
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False
        )
        assert out == [b"stale-but-present"]

    def test_a_changed_bundle_forces_re_extraction(self, tmp_path, monkeypatch):
        """The self-healing property: a re-migrated bundle must refresh the cache.

        Without it a format re-migration leaves demos loading frames from the old
        bundle, which then fail against the newer reader.
        """
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        cache = tmp_path / "cache"
        self._bundle(b, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(b, "b.zip", cache, ["f0.zip"], validate_lfs=False)
        self._bundle(b, {"f0.zip": b"v2-longer-payload"})  # new size => new stamp
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False
        )
        assert out == [b"v2-longer-payload"]

    def test_an_explicit_stamp_sees_a_swap_that_size_and_mtime_cannot(
        self, tmp_path, monkeypatch
    ):
        """The manifest-driven path keys staleness on the verified digest.

        A replacement bundle of the same byte length, written back with the
        previous mtime, is indistinguishable to the ``(size, mtime)`` fallback —
        it would keep serving the earlier extraction. The digest the manifest
        already carries settles it exactly.
        """
        import os

        from luxar.utils import demos as du

        self._load_stub(monkeypatch)

        def _swap_in_place(path: Path, payload: bytes) -> None:
            """Rewrite the bundle with an equal-length payload, mtime restored."""
            before = path.stat()
            self._bundle(path, {"f0.zip": payload})
            assert path.stat().st_size == before.st_size
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        # Control: with the (size, mtime) key the swap is invisible.
        ctl, ctl_cache = tmp_path / "ctl.zip", tmp_path / "ctl-cache"
        self._bundle(ctl, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(
            ctl, "ctl.zip", ctl_cache, ["f0.zip"], validate_lfs=False
        )
        _swap_in_place(ctl, b"v2")
        assert du._extract_bundle_and_load(
            ctl, "ctl.zip", ctl_cache, ["f0.zip"], validate_lfs=False
        ) == [b"v1"]

        # With the digest it is not.
        b, cache = tmp_path / "b.zip", tmp_path / "cache"
        self._bundle(b, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False, stamp="sha256:aaa"
        )
        _swap_in_place(b, b"v2")
        assert du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False, stamp="sha256:bbb"
        ) == [b"v2"]

    def test_a_member_absent_from_the_bundle_raises(self, tmp_path, monkeypatch):
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        with pytest.raises(FileNotFoundError, match="not found in bundle"):
            du._extract_bundle_and_load(
                b, "b.zip", tmp_path / "cache", ["nope.zip"], validate_lfs=False
            )

    def test_a_traversal_member_name_is_refused(self, tmp_path, monkeypatch):
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        with pytest.raises(ValueError, match="Unsafe zip path"):
            du._extract_bundle_and_load(
                b, "b.zip", tmp_path / "cache", ["../escape.zip"], validate_lfs=False
            )

    def test_a_member_whose_name_merely_contains_the_request_is_not_matched(
        self, tmp_path, monkeypatch
    ):
        """Members match by exact basename, not by substring.

        A bundle holding both ``decoy_f0.zip`` and ``f0.zip`` must yield the
        latter. Substring matching would take whichever the archive happens to
        list first and silently load the wrong frame -- and it passed every other
        test in this class, so it needs its own.
        """
        from luxar.utils import demos as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        # Decoy first: `matching[0]` picks it if the comparison is not exact.
        self._bundle(b, {"decoy_f0.zip": b"WRONG", "f0.zip": b"RIGHT"})
        out = du._extract_bundle_and_load(
            b, "b.zip", tmp_path / "cache", ["f0.zip"], validate_lfs=False
        )
        assert out == [b"RIGHT"]
