"""Tests for the `luxar demo` sub-app (list / info / run / run-all / cache)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.demos.registry import iter_demos


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestListAndTable:
    def test_bare_demo_shows_table(self, runner) -> None:
        result = runner.invoke(app, ["demo"])
        assert result.exit_code == 0
        assert "lorenz" in result.stdout
        # One row per demo (plus header/footer lines).
        assert str(len(iter_demos())) in result.stdout

    def test_list_filter_by_category(self, runner) -> None:
        result = runner.invoke(app, ["demo", "list", "--category", "synthetic"])
        assert result.exit_code == 0
        assert "lorenz" in result.stdout
        # NO non-synthetic demo key may appear in a synthetic-filtered table.
        # Token-level match (ANSI stripped): substring checks false-positive
        # on key collisions like "galaxy" ⊂ "spiral_galaxy".
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        tokens = set(re.split(r"\s+", plain))
        for d in iter_demos():
            if d.category != "synthetic":
                assert d.key not in tokens, f"{d.key} leaked into filter"

    def test_list_empty_filter(self, runner) -> None:
        result = runner.invoke(app, ["demo", "list", "--category", "nonexistent"])
        assert result.exit_code == 0
        assert "No demos match" in result.stdout


class TestInfo:
    def test_info_by_key(self, runner) -> None:
        result = runner.invoke(app, ["demo", "info", "lorenz"])
        assert result.exit_code == 0
        assert "Lorenz" in result.stdout
        assert "luxar demo run lorenz" in result.stdout

    def test_info_by_index(self, runner) -> None:
        first = iter_demos()[0]
        result = runner.invoke(app, ["demo", "info", "1"])
        assert result.exit_code == 0
        assert first.key in result.stdout

    def test_info_unknown_key_suggests(self, runner) -> None:
        result = runner.invoke(app, ["demo", "info", "lorentz"])
        assert result.exit_code != 0
        assert "unknown demo" in result.stdout.lower()
        assert "lorenz" in result.stdout  # difflib suggestion


class TestRun:
    def test_run_dispatches_module_with_forwarded_args(self, runner) -> None:
        with patch("luxar.cli.demo_commands.run_child_process") as mock_run:
            mock_run.return_value = 0
            result = runner.invoke(
                app, ["demo", "run", "lorenz", "--no-serve", "--points=10"]
            )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        assert argv == [
            sys.executable,
            "-m",
            "luxar.demos.demo_lorenz",
            "--no-serve",
            "--points=10",
        ]
        assert "Forwarding args" in result.stdout

    def test_run_propagates_nonzero_exit(self, runner) -> None:
        with patch("luxar.cli.demo_commands.run_child_process") as mock_run:
            mock_run.return_value = 3
            result = runner.invoke(app, ["demo", "run", "lorenz"])
        assert result.exit_code == 3

    def test_run_propagates_shell_convention_exit(self, runner) -> None:
        # run_child_process already maps signal death to 128+N (e.g. SIGKILL
        # -> 137; the mapping itself is covered in test_process.py). demo_run
        # must propagate that code verbatim, not truncate it.
        with patch("luxar.cli.demo_commands.run_child_process") as mock_run:
            mock_run.return_value = 137
            result = runner.invoke(app, ["demo", "run", "lorenz"])
        assert result.exit_code == 137

    def test_run_by_index(self, runner) -> None:
        target = iter_demos()[0]
        with patch("luxar.cli.demo_commands.run_child_process") as mock_run:
            mock_run.return_value = 0
            result = runner.invoke(app, ["demo", "run", "1"])
        assert result.exit_code == 0
        argv = mock_run.call_args.args[0]
        assert argv[1] == "-m" and argv[2] == target.module

    def test_run_unknown_key_errors(self, runner) -> None:
        result = runner.invoke(app, ["demo", "run", "no-such-demo"])
        assert result.exit_code != 0
        assert "unknown demo" in result.stdout.lower()


class TestRunAll:
    """`demo run-all` batch semantics (mocked child; no demo executes)."""

    @pytest.fixture(autouse=True)
    def _no_existing_outputs(self, monkeypatch) -> None:
        # Make every demo look not-yet-generated so --skip-existing never
        # skips (host machines may have real outputs under datasets/demos).
        monkeypatch.setattr(
            "luxar.cli.demo_commands.registry.demo_output_paths", lambda d: []
        )

    @staticmethod
    def _is_runnable(d, *, max_download_mb: int = 200) -> bool:
        """Mirror run-all's DEFAULT skip criteria (no --include-gpu)."""
        return (
            d.local_data not in ("manual-file", "kaggle-auth")
            and d.gpu != "required"
            and d.download_mb <= max_download_mb
        )

    @classmethod
    def _runnable_count(cls) -> int:
        return sum(1 for d in iter_demos() if cls._is_runnable(d))

    def test_keep_going_aggregates_failures(self, runner) -> None:
        failing = next(d for d in iter_demos() if self._is_runnable(d))

        def fake_run(argv, **kw):
            return 2 if argv[2] == failing.module else 0

        with patch(
            "luxar.cli.demo_commands.run_child_process", side_effect=fake_run
        ) as m:
            result = runner.invoke(app, ["demo", "run-all"])
        assert result.exit_code == 1  # a failure surfaces at the end
        assert m.call_count == self._runnable_count()  # ...but nothing aborted
        assert failing.key in result.stdout
        assert "failed 1" in result.stdout

    def test_fail_fast_stops_at_first_failure(self, runner) -> None:
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 5
            result = runner.invoke(app, ["demo", "run-all", "--fail-fast"])
        assert result.exit_code == 1
        assert m.call_count == 1  # stopped at the first failing demo

    def test_all_green_exits_zero(self, runner) -> None:
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 0
            result = runner.invoke(app, ["demo", "run-all"])
        assert result.exit_code == 0
        assert m.call_count == self._runnable_count()
        assert "failed 0" in result.stdout

    def test_interrupt_stops_batch(self, runner) -> None:
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 130  # helper reports Ctrl-C
            result = runner.invoke(app, ["demo", "run-all"])
        assert result.exit_code == 130
        assert m.call_count == 1  # stopped immediately on interrupt

    def test_skip_gpu_by_default(self, runner) -> None:
        gpu_demo = next((d for d in iter_demos() if d.gpu == "required"), None)
        if gpu_demo is None:
            pytest.skip("no GPU-required demo in the registry")
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 0
            result = runner.invoke(app, ["demo", "run-all"])
        ran_modules = [c.args[0][2] for c in m.call_args_list]
        assert gpu_demo.module not in ran_modules
        assert f"{gpu_demo.key}: needs GPU" in result.stdout

    def test_include_gpu_runs_gpu_demo(self, runner) -> None:
        gpu_demo = next((d for d in iter_demos() if d.gpu == "required"), None)
        if gpu_demo is None:
            pytest.skip("no GPU-required demo in the registry")
        # --include-gpu + no download limit so the ONLY skip axis is GPU.
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 0
            result = runner.invoke(
                app, ["demo", "run-all", "--include-gpu", "--max-download-mb", "0"]
            )
        assert result.exit_code == 0
        ran_modules = [c.args[0][2] for c in m.call_args_list]
        # A GPU demo with no other skip reason must now run.
        if gpu_demo.local_data not in ("manual-file", "kaggle-auth"):
            assert gpu_demo.module in ran_modules

    def test_max_download_mb_filters(self, runner) -> None:
        big = next(
            (
                d
                for d in iter_demos()
                if d.download_mb > 200
                and d.local_data not in ("manual-file", "kaggle-auth")
                and d.gpu != "required"
            ),
            None,
        )
        if big is None:
            pytest.skip("no large-download CPU demo in the registry")
        # Default 200MB limit skips it...
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 0
            result = runner.invoke(app, ["demo", "run-all"])
        assert big.module not in [c.args[0][2] for c in m.call_args_list]
        assert f"{big.key}: download" in result.stdout
        # ...but --max-download-mb 0 (no limit) runs it.
        with patch("luxar.cli.demo_commands.run_child_process") as m:
            m.return_value = 0
            runner.invoke(app, ["demo", "run-all", "--max-download-mb", "0"])
        assert big.module in [c.args[0][2] for c in m.call_args_list]


class TestCache:
    def test_cache_list_empty(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path / "nope")
        result = runner.invoke(app, ["demo", "cache", "list"])
        assert result.exit_code == 0
        assert "No demo cache" in result.stdout

    def test_cache_list_reports_sizes_and_orphans(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        # A dir claimed by some demo + an orphan dir.
        claimed = next((d.caches[0] for d in iter_demos() if d.caches), None)
        assert claimed is not None
        (tmp_path / claimed).mkdir()
        (tmp_path / claimed / "blob.bin").write_bytes(b"x" * 4096)
        (tmp_path / "orphan-xyz").mkdir()
        result = runner.invoke(app, ["demo", "cache", "list"])
        assert result.exit_code == 0
        assert claimed in result.stdout
        assert "ORPHAN" in result.stdout

    def test_cache_clear_requires_selection(self, runner) -> None:
        result = runner.invoke(app, ["demo", "cache", "clear"])
        assert result.exit_code == 1
        assert "Specify" in result.stdout

    def test_cache_clear_dry_run_deletes_nothing(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        cdir.mkdir()
        blob = cdir / "data.bin"
        blob.write_bytes(b"y" * 1024)
        result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout
        assert blob.exists()  # nothing deleted

    def test_cache_clear_yes_deletes_downloads(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        cdir.mkdir()
        download = cdir / "archive.zip"
        download.write_bytes(b"z" * 2048)
        pkl = cdir / "cache_v1.pkl"
        pkl.write_bytes(b"p" * 512)
        # Clear only downloads (not computed pickles).
        result = runner.invoke(
            app,
            ["demo", "cache", "clear", demo.key, "--no-computed", "--yes"],
        )
        assert result.exit_code == 0
        assert not download.exists()
        assert pkl.exists()  # --no-computed preserved the pickle

    def test_cache_clear_orphans(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        orphan = tmp_path / "orphan-dir"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 100)
        result = runner.invoke(app, ["demo", "cache", "clear", "--orphans", "--yes"])
        assert result.exit_code == 0
        assert not orphan.exists()


class TestBrokenMeta:
    """A malformed DEMO_META must produce a clean error, not a traceback."""

    def test_list_reports_broken_meta_cleanly(self, runner, monkeypatch) -> None:
        from luxar.demos.registry import DemoMetaError

        def boom(*a, **k):
            raise DemoMetaError("demo_x.py: bad meta")

        monkeypatch.setattr("luxar.cli.demo_commands.registry.iter_demos", boom)
        result = runner.invoke(app, ["demo", "list"])
        assert result.exit_code == 1
        assert "Broken demo metadata" in result.stdout

    def test_run_reports_broken_meta_cleanly(self, runner, monkeypatch) -> None:
        from luxar.demos.registry import DemoMetaError

        def boom(*a, **k):
            raise DemoMetaError("demo_x.py: bad meta")

        monkeypatch.setattr("luxar.cli.demo_commands.registry.get_demo", boom)
        result = runner.invoke(app, ["demo", "run", "lorenz"])
        assert result.exit_code == 1
        assert "Broken demo metadata" in result.stdout


class TestCacheClearClassification:
    def test_corrupt_download_is_a_download_not_computed(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        cdir.mkdir()
        corrupt_dl = cdir / "archive.zip.corrupt"
        corrupt_dl.write_bytes(b"z" * 2048)
        corrupt_pkl = cdir / "cache_v1.pkl.corrupt"
        corrupt_pkl.write_bytes(b"p" * 512)
        # --no-computed must preserve the corrupt PICKLE but drop the corrupt DOWNLOAD.
        result = runner.invoke(
            app, ["demo", "cache", "clear", demo.key, "--no-computed", "--yes"]
        )
        assert result.exit_code == 0
        assert not corrupt_dl.exists()  # corrupt download cleared with downloads
        assert corrupt_pkl.exists()  # corrupt pickle preserved by --no-computed


@pytest.mark.slow
class TestRunRealSubprocess:
    def test_run_lorenz_no_serve_produces_zarr(self) -> None:
        """End-to-end: the child interpreter actually generates the scene.

        Runs the module the way `demo run` does, then verifies the demo wrote
        its scene to get_demos_output_dir() (repo datasets/demos, gitignored).
        """
        from luxar.utils.paths import get_demos_output_dir

        out = get_demos_output_dir() / "lorenz.luxar.zarr"
        pre_existing = out.exists()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "luxar.demos.demo_lorenz",
                "--no-serve",
                "--points=400",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        import zarr

        root = zarr.open_group(out, mode="r")
        assert len(list(root.group_keys())) > 0
        if not pre_existing:
            import shutil

            shutil.rmtree(out, ignore_errors=True)
