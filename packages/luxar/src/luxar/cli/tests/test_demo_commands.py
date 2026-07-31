"""Tests for the `luxar demo` sub-app (list / info / run / run-all / cache)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.demos._dependencies import DependencySpec
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


class TestDeps:
    """``luxar demo deps`` reports (and installs) the optional demo packages."""

    def test_deps_lists_every_known_dependency(self, runner) -> None:
        from luxar.demos import INSTALL_SPECS

        result = runner.invoke(app, ["demo", "deps"])
        # Exit code is 1 when anything is missing, 0 when all present — both are
        # valid here (it depends on the test machine), so only the table matters.
        assert result.exit_code in (0, 1)
        for module in INSTALL_SPECS:
            assert module in result.stdout, f"{module} missing from the table"

    def test_deps_shows_the_constrained_requirement_not_a_bare_name(
        self, runner
    ) -> None:
        """The whole point of INSTALL_SPECS: never advertise an unbounded pin."""
        result = runner.invoke(app, ["demo", "deps"])
        assert "anndata>=0.10,<0.13" in result.stdout

    def test_deps_rejects_an_unknown_extra(self, runner) -> None:
        result = runner.invoke(app, ["demo", "deps", "--extra", "nope"])
        assert result.exit_code == 1
        assert "No known dependencies" in result.stdout

    def test_deps_filters_to_one_extra(self, runner) -> None:
        result = runner.invoke(app, ["demo", "deps", "--extra", "io"])
        assert result.exit_code in (0, 1)
        assert "imageio" in result.stdout
        # A demos-only package must not appear in an io-filtered report.
        assert "anndata" not in result.stdout

    def test_deps_exits_nonzero_when_something_is_missing(self, runner) -> None:
        """A CI gate can rely on the exit code, so it must track missing-ness."""
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 1
        assert "1 missing: phony_xyz" in result.stdout

    def test_deps_exits_zero_when_all_present(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus("phony_xyz", DependencySpec("phony-xyz>=1", "demos"), True)
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 0
        assert "all 1 optional dependency installed." in result.stdout

    def test_deps_columns_are_never_narrower_than_their_headers(self, runner) -> None:
        """A one-row report (module shorter than "MODULE") must not go ragged."""
        import re

        from luxar.demos._dependencies import DependencyStatus

        fake = [DependencyStatus("ab", DependencySpec("ab>=1", "demos"), True)]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        plain = [re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in result.stdout.splitlines()]
        header = next(ln for ln in plain if "MODULE" in ln)
        row = next(ln for ln in plain if "ab>=1" in ln)
        # The EXTRA column must start at the same offset in both lines.
        assert header.index("EXTRA") == row.index("demos"), f"{header!r} vs {row!r}"
        # ...and the rule must be exactly as wide as the header it underlines
        # (a hand-counted constant overshot by one glyph).
        sep = next(ln for ln in plain if "─" * 10 in ln)
        gutter = header.index("MODULE") - 2  # arbol prefix + the 2-space indent
        assert len(sep.rstrip()) - gutter == len(header.rstrip()) - gutter, (
            f"rule {len(sep.rstrip())} != header {len(header.rstrip())}"
        )

    def test_deps_dry_run_install_runs_no_pip(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process") as proc:
                result = runner.invoke(app, ["demo", "deps", "--install", "--dry-run"])
        assert result.exit_code == 0
        proc.assert_not_called()
        assert "pip install" in result.stdout

    def test_deps_install_invokes_pip_with_the_missing_extras(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "gsplats"), False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch(
                "luxar.cli.demo_commands.run_child_process", return_value=0
            ) as proc:
                runner.invoke(app, ["demo", "deps", "--install"])
        proc.assert_called_once()
        cmd = proc.call_args[0][0]
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
        # The extra named by the missing spec must be the one installed.
        assert any("[gsplats]" in part for part in cmd), cmd

    def test_pip_cmd_names_distribution_for_a_foreign_project_root(
        self, tmp_path
    ) -> None:
        """A foreign pyproject.toml root must not trigger an editable install.

        For a non-editable Luxar in a project-local venv, ``get_project_root``
        returns the *user's own* project. Since it does not own the imported
        ``luxar``, the command must name the distribution, never ``-e <root>``.
        """
        from luxar.cli.demo_commands import _pip_install_cmd

        (tmp_path / "pyproject.toml").write_text("[project]\nname='foreign'\n")
        with patch("luxar.utils.paths.get_project_root", return_value=tmp_path):
            cmd = _pip_install_cmd(["demos"])
        assert "-e" not in cmd, cmd
        assert cmd[-1] == "luxar[demos]", cmd

    def test_pip_cmd_names_distribution_when_no_project_root(self) -> None:
        """No pyproject.toml anywhere → distribution form, not editable."""
        from luxar.cli.demo_commands import _luxar_checkout_root, _pip_install_cmd

        with patch(
            "luxar.utils.paths.get_project_root",
            side_effect=RuntimeError("no root"),
        ):
            cmd = _pip_install_cmd(["demos"])
            assert _luxar_checkout_root() is None
        assert "-e" not in cmd, cmd
        assert cmd[-1] == "luxar[demos]", cmd

    def test_pip_cmd_uses_editable_for_a_root_that_owns_luxar(self, tmp_path) -> None:
        """A root that genuinely owns the imported ``luxar`` → editable install.

        This pins the positive branch of ``_luxar_checkout_root``: when the
        discovered project root's ``packages/luxar/src/luxar`` IS the imported
        package, the command must be the editable ``-e <root>[extras]`` form so a
        dev checkout is not shadowed by a stale wheel.
        """
        import luxar
        from luxar.cli.demo_commands import _pip_install_cmd

        owned_pkg = tmp_path / "packages" / "luxar" / "src" / "luxar"
        owned_pkg.mkdir(parents=True)
        (owned_pkg / "__init__.py").write_text("")
        with (
            patch("luxar.utils.paths.get_project_root", return_value=tmp_path),
            patch.object(luxar, "__file__", str(owned_pkg / "__init__.py")),
        ):
            cmd = _pip_install_cmd(["demos"])
        assert "-e" in cmd, cmd
        assert cmd[cmd.index("-e") + 1] == f"{tmp_path}[demos]", cmd

    def test_deps_names_specs_that_belong_to_no_extra(self, runner) -> None:
        """gdown is installable only by name, so --install can't cover it."""
        from luxar.demos._dependencies import DependencyStatus

        fake = [DependencyStatus("gdown", DependencySpec("gdown", ""), False)]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 1
        assert "Not in any extra" in result.stdout
        assert "gdown" in result.stdout

    def test_deps_unknown_extra_lists_the_valid_ones(self, runner) -> None:
        """A bare "no such extra" leaves the user guessing what to type."""
        result = runner.invoke(app, ["demo", "deps", "--extra", "nope"])
        assert result.exit_code == 1
        assert "Valid extras:" in result.stdout
        assert "demos" in result.stdout

    def test_deps_extra_is_case_and_space_insensitive(self, runner) -> None:
        """Extra names are lowercase per PEP 685; accept what the user types."""
        loud = runner.invoke(app, ["demo", "deps", "--extra", "  IO  "])
        quiet = runner.invoke(app, ["demo", "deps", "--extra", "io"])
        assert loud.exit_code == quiet.exit_code
        assert "imageio" in loud.stdout

    def test_deps_blank_extra_is_treated_as_unset(self, runner) -> None:
        """A blank `--extra` surveys everything, not the no-extra (gdown) row.

        `survey` treats only None as "everything", so without normalization an
        empty-string extra was an active filter matching just the specs in no
        extra — a gdown-only table.
        """
        blank = runner.invoke(app, ["demo", "deps", "--extra", "   "])
        full = runner.invoke(app, ["demo", "deps"])
        assert blank.exit_code == full.exit_code
        # The full table lists demos-extra packages; the gdown-only path would not.
        assert "anndata" in blank.stdout

    def test_deps_dry_run_without_install_says_it_is_inert(self, runner) -> None:
        result = runner.invoke(app, ["demo", "deps", "--dry-run"])
        assert "--dry-run only applies with --install" in result.stdout

    def test_deps_does_not_blame_the_install_for_an_orphan_spec(self, runner) -> None:
        """gdown is in no extra, so --install never attempts it.

        Reporting it as "still missing after install" made a successful install
        look like a failure, and exited 1 on work that fully succeeded.
        """
        from luxar.demos._dependencies import DependencyStatus

        orphan = DependencyStatus("gdown", DependencySpec("gdown", ""), False)
        before = [orphan, DependencyStatus("m", DependencySpec("m>=1", "demos"), False)]
        after = [orphan, DependencyStatus("m", DependencySpec("m>=1", "demos"), True)]
        calls = {"n": 0}

        def fake_survey(extra=None):
            calls["n"] += 1
            return before if calls["n"] == 1 else after

        with patch("luxar.demos.survey", side_effect=fake_survey):
            with patch("luxar.cli.demo_commands.run_child_process", return_value=0):
                result = runner.invoke(app, ["demo", "deps", "--install"])

        assert result.exit_code == 0, "the extra installed fine; must not exit 1"
        assert "Still missing after install" not in result.stdout
        # ...but it must not claim completeness either.
        assert "Still to install by hand" in result.stdout
        assert "gdown" in result.stdout
