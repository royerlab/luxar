"""Tests for parallel tiled fitting (``fit --tiling uniform --jobs N``).

The orchestrator (``fit_tiled_parallel``) is exercised with a *fake* worker
command builder that writes deterministic tiny ``.gsplats.zarr`` files via a
short ``python -c`` script — no torch, no real fitting.  This keeps the tests
fast and focused on the orchestration / merge / error-handling logic.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.fit_tiled_parallel import (
    build_worker_cmd,
    fit_tiled_parallel,
    resolve_jobs,
)
from luxar.gsplats.tiling import compute_tile_specs

# Subprocess-pool tiled fitting (seconds per test). Slow → CI runs `-m "not
# slow"`; the full suite runs locally pre-push.
pytestmark = pytest.mark.slow

# ── A fake worker: writes a deterministic tiny gsplats per tile ──────────────


def _fake_worker_builder(n_per_tile: int = 5, n_lods: int = 1):
    """Return a worker_cmd_builder writing ``n_per_tile`` splats (x ``n_lods``)."""

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        script = textwrap.dedent(
            f"""
            import numpy as np
            from luxar.gsplats.gsplat_data import GSplatData, AdditiveSubLOD

            def lod(seed, k):
                rng = np.random.default_rng(seed)
                centers = rng.random((k, 3)).astype(np.float32) * 10.0
                amps = (rng.random(k).astype(np.float32) + 0.1)
                chol = np.tile(
                    np.array([1, 0, 1, 0, 0, 1], np.float32), (k, 1)
                )
                return AdditiveSubLOD(centers=centers, amplitudes=amps,
                                      cholesky_factors=chol)

            lods = [lod({i} * 100 + j, {n_per_tile}) for j in range({n_lods})]
            data = GSplatData.from_additive_sublods(lods)
            data.save(r"{out_path}")
            """
        )
        return [sys.executable, "-c", script]

    return builder


def _failing_worker_builder(fail_on: int):
    """Worker builder where tile ``fail_on`` exits non-zero with a marker."""
    ok = _fake_worker_builder()

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        if i == fail_on:
            return [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('BOOM tile failure\\n'); sys.exit(3)",
            ]
        return ok(i, m, out_path)

    return builder


def _exit0_nofile_builder(silent_on: int):
    """Worker builder where tile ``silent_on`` exits 0 but writes no file."""
    ok = _fake_worker_builder()

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        if i == silent_on:
            return [sys.executable, "-c", "import sys; sys.exit(0)"]
        return ok(i, m, out_path)

    return builder


def _empty_marker_builder(empty_on):
    """Worker builder where tiles in ``empty_on`` write only an .empty marker.

    Mimics the single-tile worker's --allow-empty-tile path: a 0-splat tile
    drops a sibling ``<out>.empty`` marker (no .gsplats.zarr) and exits 0.
    """
    empty = set(empty_on)
    ok = _fake_worker_builder()

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        if i in empty:
            marker = str(out_path) + ".empty"
            return [
                sys.executable,
                "-c",
                "open(r'" + marker + "', 'w').write('0 splats\\n')",
            ]
        return ok(i, m, out_path)

    return builder


def _raising_builder():
    """A worker_cmd_builder that itself raises (not a subprocess failure)."""

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        raise RuntimeError("builder blew up for tile " + str(i))

    return builder


def _corrupt_store_builder(corrupt_on: int):
    """Worker builder where tile ``corrupt_on`` writes a present-but-unreadable dir."""
    ok = _fake_worker_builder()

    def builder(i: int, m: int, out_path: Path) -> list[str]:
        if i == corrupt_on:
            # Create the .gsplats.zarr path as a dir with junk — exists() true,
            # GSplatData.load() fails.
            return [
                sys.executable,
                "-c",
                "import os; os.makedirs(r'" + str(out_path) + "', exist_ok=True); "
                "open(os.path.join(r'" + str(out_path) + "', 'junk'), 'w').write('x')",
            ]
        return ok(i, m, out_path)

    return builder


# ── resolve_jobs ─────────────────────────────────────────────────────────────


class TestResolveJobs:
    def test_explicit_passthrough(self) -> None:
        assert resolve_jobs(4, tile_voxels=1000, num_tiles=10) == 4

    def test_explicit_clamped_to_num_tiles(self) -> None:
        assert resolve_jobs(8, tile_voxels=1000, num_tiles=3) == 3

    def test_explicit_floor_one(self) -> None:
        assert resolve_jobs(0, tile_voxels=1000, num_tiles=5) == 1

    def test_bad_string_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_jobs("banana", tile_voxels=1000, num_tiles=5)

    def test_auto_uses_vram_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # free=300 MB, per_tile = 2 * 10e6 * 4 = 80 MB  -> 3 workers
        import luxar.gsplats.metrics as metrics
        import luxar.gsplats.utils.device as device_mod

        monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: 300_000_000)
        monkeypatch.setattr(device_mod, "resolve_torch_device", lambda *a, **k: None)
        n = resolve_jobs("auto", tile_voxels=10_000_000, num_tiles=10, device="cuda")
        assert n == 3

    def test_auto_clamped_to_num_tiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import luxar.gsplats.metrics as metrics
        import luxar.gsplats.utils.device as device_mod

        monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: 10_000_000_000)
        monkeypatch.setattr(device_mod, "resolve_torch_device", lambda *a, **k: None)
        n = resolve_jobs("auto", tile_voxels=1000, num_tiles=4, device="cuda")
        assert n == 4

    def test_auto_cpu_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No VRAM info -> CPU fallback = cpu_count()//2 (pinned to 8 cores here
        # so the assertion tests the formula, not the num_tiles clamp), then
        # clamped to num_tiles.
        import os

        import luxar.gsplats.metrics as metrics
        import luxar.gsplats.utils.device as device_mod

        monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: None)
        monkeypatch.setattr(device_mod, "resolve_torch_device", lambda *a, **k: None)
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        # num_tiles high enough to not clamp -> returns 8//2 = 4
        assert resolve_jobs("auto", tile_voxels=1000, num_tiles=100, device="cpu") == 4
        # and clamped when num_tiles is small
        assert resolve_jobs("auto", tile_voxels=1000, num_tiles=2, device="cpu") == 2


# ── build_worker_cmd ─────────────────────────────────────────────────────────


class TestBuildWorkerCmd:
    def _cmd(self, **kw) -> list[str]:
        return build_worker_cmd(
            ["luxar"], "in.zarr", "out.gsplats.zarr", 3, 16, 256, 32, **kw
        )

    def test_core_flags(self) -> None:
        cmd = self._cmd()
        assert cmd[:4] == ["luxar", "gsplat", "fit", "in.zarr"]
        assert "out.gsplats.zarr" in cmd
        # single-tile selector with the right index/count
        assert "--tile" in cmd and "3/16" in cmd
        assert "--tile-size" in cmd and "256" in cmd
        assert "--overlap" in cmd and "32" in cmd

    def test_never_forwards_driver_flags(self) -> None:
        cmd = self._cmd(downscale="1,4,4", seeds="8000")
        assert "--tiling" not in cmd
        assert "--jobs" not in cmd
        assert "-j" not in cmd
        assert "--compress" not in cmd

    def test_forwards_values_not_just_flags(self) -> None:
        # Guard against a mutation that drops/garbles a passthrough VALUE.
        cmd = self._cmd(
            seeds="8000",
            iters=5000,
            device="cuda",
            preset="hifi",
            lr=0.02,
            loss="mse",
            seed_method="edges",
            downscale="1,4,4",
            channel=2,
            timepoint=7,
            array_key="h2afva/fused",
        )

        def val(flag):
            return cmd[cmd.index(flag) + 1]

        assert val("--seeds") == "8000"
        assert val("--iters") == "5000"
        assert val("--device") == "cuda"
        assert val("--preset") == "hifi"
        assert val("--lr") == "0.02"
        assert val("--loss") == "mse"
        assert val("--seed-method") == "edges"
        assert val("--downscale") == "1,4,4"
        assert val("--channel") == "2"
        assert val("--timepoint") == "7"
        assert val("--array-key") == "h2afva/fused"

    def test_forwards_floor(self) -> None:
        # An explicit --floor must reach each parallel tile worker (else -j>1
        # silently drops the override and every tile falls back to 'auto').
        cmd = self._cmd(floor="none")
        assert "--floor" in cmd and cmd[cmd.index("--floor") + 1] == "none"
        cmd_p = self._cmd(floor="p10")
        assert cmd_p[cmd_p.index("--floor") + 1] == "p10"
        # Not emitted when unset (the worker then resolves its own
        # --config/--preset merge, defaulting to 'auto').
        assert "--floor" not in self._cmd()

    def test_forwards_config_and_progressive_and_denoise_values(self) -> None:
        # Close the rest of the mutation-survivor class (config / progressive /
        # denoise VALUES, not just flag presence).
        cmd = self._cmd(
            config="cfg.yaml",
            progressive=True,
            max_splats_per_pass=1234,
            psnr_patience=0.3,
            max_passes=7,
            denoise=True,
            denoise_h=0.041,
            denoise_patch_size=5,
            denoise_search_distance=11,
            denoise_backend="cuda",
        )

        def val(flag):
            return cmd[cmd.index(flag) + 1]

        assert val("--config") == "cfg.yaml"
        assert val("--splats-per-pass") == "1234"
        assert val("--psnr-patience") == "0.3"
        assert val("--max-passes") == "7"
        assert val("--denoise-h") == "0.041"
        assert val("--denoise-patch-size") == "5"
        assert val("--denoise-search-distance") == "11"
        assert val("--denoise-backend") == "cuda"

    def test_allow_empty_tile_flag(self) -> None:
        assert "--allow-empty-tile" not in self._cmd()
        assert "--allow-empty-tile" in self._cmd(allow_empty_tile=True)

    def test_quiet_and_cull_disabled(self) -> None:
        cmd = self._cmd()
        assert "--quiet" in cmd
        # parent culls once; workers must not cull per-tile
        assert "--cull-retention" in cmd
        assert cmd[cmd.index("--cull-retention") + 1] == "0"

    def test_passthrough_only_when_set(self) -> None:
        cmd = self._cmd()  # nothing optional set
        for flag in (
            "--seeds",
            "--iters",
            "--preset",
            "--config",
            "--downscale",
            "--channel",
            "--timepoint",
            "--array-key",
            "--progressive",
            "--denoise",
        ):
            assert flag not in cmd

    def test_denoise_injects_calibrated_h(self) -> None:
        cmd = self._cmd(denoise=True, denoise_h=0.037)
        assert "--denoise" in cmd
        assert "--denoise-h" in cmd
        assert cmd[cmd.index("--denoise-h") + 1] == "0.037"

    def test_progressive_flags(self) -> None:
        cmd = self._cmd(
            progressive=True, max_splats_per_pass=1000, psnr_patience=0.3, max_passes=4
        )
        assert "--progressive" in cmd
        assert "--splats-per-pass" in cmd and "1000" in cmd
        assert "--max-passes" in cmd and "4" in cmd


# ── M-under-downscale invariant ──────────────────────────────────────────────


class TestDownscaleGridMath:
    @pytest.mark.parametrize(
        "shape,factors",
        [
            ((64, 64, 64), (2, 2, 2)),
            ((100, 80, 120), (1, 4, 4)),
            ((33, 17), (2, 3)),
            ((50, 50, 50), (3, 3, 3)),
        ],
    )
    def test_parent_grid_matches_worker(self, shape, factors) -> None:
        """The parent's shape-only decimation must match real downscaling."""
        from luxar.gsplats.fitting.downscale import downscale_volume

        # Parent: shape math only (volume[::f]).
        parent_shape = tuple(len(range(0, s, f)) for s, f in zip(shape, factors))

        # Worker: actually downscales then tiles.
        vol = np.zeros(shape, dtype=np.float32)
        worker_shape = downscale_volume(vol, factors).shape

        assert parent_shape == worker_shape
        m_parent = len(compute_tile_specs(parent_shape, 16, 4))
        m_worker = len(compute_tile_specs(worker_shape, 16, 4))
        assert m_parent == m_worker


# ── Orchestrator ─────────────────────────────────────────────────────────────


class TestFitTiledParallel:
    def _specs_count(self, shape=(40, 40), tile=16, overlap=4) -> int:
        return len(compute_tile_specs(shape, tile, overlap))

    def test_merges_all_tiles(self, tmp_path: Path) -> None:
        m = self._specs_count()
        merged = fit_tiled_parallel(
            num_tiles=m,
            jobs=2,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=_fake_worker_builder(n_per_tile=5),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=False,
            cull_retention=None,
        )
        assert merged.n_splats == 5 * m
        assert merged.stats["num_tiles"] == m
        # temp dir cleaned up by default
        assert not (tmp_path / "tiles").exists()

    def test_keep_tiles(self, tmp_path: Path) -> None:
        m = self._specs_count()
        fit_tiled_parallel(
            num_tiles=m,
            jobs=1,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=_fake_worker_builder(n_per_tile=3),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=False,
            cull_retention=None,
            keep_tiles=True,
        )
        kept = list((tmp_path / "tiles").glob("tile_*.gsplats.zarr"))
        assert len(kept) == m

    def test_jobs1_equals_jobs2(self, tmp_path: Path) -> None:
        m = self._specs_count()
        kw = dict(
            num_tiles=m,
            worker_cmd_builder=_fake_worker_builder(n_per_tile=4),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=False,
            cull_retention=None,
        )
        a = fit_tiled_parallel(jobs=1, tmp_dir=tmp_path / "a", **kw)
        b = fit_tiled_parallel(jobs=2, tmp_dir=tmp_path / "b", **kw)
        assert a.n_splats == b.n_splats == 4 * m

    def test_lod_survives_disk_roundtrip(self, tmp_path: Path) -> None:
        """Progressive (multi-sublod) tiles must merge LOD-aware after reload."""
        m = self._specs_count()
        merged = fit_tiled_parallel(
            num_tiles=m,
            jobs=2,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=_fake_worker_builder(n_per_tile=4, n_lods=2),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=True,
            cull_retention=None,
        )
        assert merged.n_additive_sublods == 2

    def test_failure_raises_and_keeps_tiles(self, tmp_path: Path) -> None:
        m = self._specs_count()
        tiles = tmp_path / "tiles"
        with pytest.raises(RuntimeError) as exc:
            fit_tiled_parallel(
                num_tiles=m,
                jobs=2,
                tmp_dir=tiles,
                worker_cmd_builder=_failing_worker_builder(fail_on=2),
                volume_shape=(40, 40),
                tile_size=16,
                overlap=4,
                progressive=False,
                cull_retention=None,
            )
        msg = str(exc.value)
        assert "tile" in msg.lower()
        # the failing tile INDEX is named (not coincidentally matched by the
        # "N of M" count — assert the explicit "tiles: 2" listing)
        assert "tiles: 2" in msg
        assert "BOOM" in msg  # stderr tail is surfaced
        # temp dir retained for inspection on failure
        assert tiles.exists()

    def test_launch_failure_is_reported_not_raw(self, tmp_path: Path) -> None:
        """A worker that can't even launch (bogus argv) → curated RuntimeError."""

        def bogus_builder(i: int, m: int, out_path: Path) -> list[str]:
            return ["this-binary-does-not-exist-xyz", "--tile", f"{i}/{m}"]

        tiles = tmp_path / "tiles"
        with pytest.raises(RuntimeError) as exc:
            fit_tiled_parallel(
                num_tiles=self._specs_count(),
                jobs=2,
                tmp_dir=tiles,
                worker_cmd_builder=bogus_builder,
                volume_shape=(40, 40),
                tile_size=16,
                overlap=4,
                progressive=False,
                cull_retention=None,
            )
        # curated message, not a raw FileNotFoundError traceback
        assert "failed to build/launch" in str(exc.value).lower()
        assert tiles.exists()

    def test_builder_exception_is_curated(self, tmp_path: Path) -> None:
        """A worker_cmd_builder that raises is funneled into the failure path."""
        tiles = tmp_path / "tiles"
        with pytest.raises(RuntimeError) as exc:
            fit_tiled_parallel(
                num_tiles=self._specs_count(),
                jobs=2,
                tmp_dir=tiles,
                worker_cmd_builder=_raising_builder(),
                volume_shape=(40, 40),
                tile_size=16,
                overlap=4,
                progressive=False,
                cull_retention=None,
            )
        # not a raw traceback escaping the pool
        assert "failed to build/launch" in str(exc.value).lower()
        assert tiles.exists()

    def test_corrupt_store_is_curated(self, tmp_path: Path) -> None:
        """A present-but-unreadable tile store → curated error naming the tile."""
        tiles = tmp_path / "tiles"
        with pytest.raises(RuntimeError) as exc:
            fit_tiled_parallel(
                num_tiles=self._specs_count(),
                jobs=2,
                tmp_dir=tiles,
                worker_cmd_builder=_corrupt_store_builder(corrupt_on=1),
                volume_shape=(40, 40),
                tile_size=16,
                overlap=4,
                progressive=False,
                cull_retention=None,
            )
        msg = str(exc.value).lower()
        assert "unreadable" in msg
        assert tiles.exists()

    def test_empty_marker_tile_is_skipped(self, tmp_path: Path) -> None:
        """A tile that writes only an .empty marker contributes nothing (no raise)."""
        m = self._specs_count()
        merged = fit_tiled_parallel(
            num_tiles=m,
            jobs=2,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=_empty_marker_builder(empty_on=[0, 3]),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=False,
            cull_retention=None,
        )
        # two empty tiles skipped; the rest (m-2) contribute 5 splats each
        assert merged.n_splats == 5 * (m - 2)
        # splats_per_tile stays positionally aligned with the sequential path:
        # one entry per tile (full grid), with 0 for the two empty tiles.
        spt = merged.stats["splats_per_tile"]
        assert len(spt) == m
        assert spt[0] == 0 and spt[3] == 0
        assert sum(spt) == 5 * (m - 2)

    def test_exit0_without_output_raises(self, tmp_path: Path) -> None:
        """A worker exiting 0 but writing no file must not be silently dropped."""
        m = self._specs_count()
        tiles = tmp_path / "tiles"
        with pytest.raises(RuntimeError) as exc:
            fit_tiled_parallel(
                num_tiles=m,
                jobs=2,
                tmp_dir=tiles,
                worker_cmd_builder=_exit0_nofile_builder(silent_on=2),
                volume_shape=(40, 40),
                tile_size=16,
                overlap=4,
                progressive=False,
                cull_retention=None,
            )
        msg = str(exc.value)
        assert "no" in msg.lower() and "output" in msg.lower()
        assert "2" in msg  # the offending tile index
        assert tiles.exists()

    def test_stale_tile_dir_is_cleared(self, tmp_path: Path) -> None:
        """A pre-existing tile dir from a prior run is wiped at start."""
        tiles = tmp_path / "tiles"
        tiles.mkdir()
        # Sentinel from a hypothetical larger prior run (index outside this
        # run's grid). The reload never references it, so only a start-of-run
        # clear can remove it — a clean pre-fix failure independent of save().
        stale = tiles / "tile_99.gsplats.zarr"
        stale.mkdir()
        (stale / "STALE_MARKER").write_text("garbage")

        m = self._specs_count()
        merged = fit_tiled_parallel(
            num_tiles=m,
            jobs=2,
            tmp_dir=tiles,
            worker_cmd_builder=_fake_worker_builder(n_per_tile=5),
            volume_shape=(40, 40),
            tile_size=16,
            overlap=4,
            progressive=False,
            cull_retention=None,
            keep_tiles=True,
        )
        assert merged.n_splats == 5 * m
        assert not stale.exists()  # tmp_dir was cleared before the run
