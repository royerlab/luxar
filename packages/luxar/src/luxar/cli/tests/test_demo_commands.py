"""Tests for the `luxar demo` sub-app (list / info / run / run-all / cache)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.demo_commands import _status
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

    def test_cache_clear_removes_a_leaked_staging_dir(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # A download killed mid-flight (SIGKILL/OOM/power loss) leaks the private
        # staging dir it streams into. Unlinking the staged file is not enough:
        # the empty dir keeps the cache dir alive, so the demo reports as cached
        # forever and a second clear finds nothing to do.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        staging = cdir / ".archive.zip.ab12cd"
        staging.mkdir(parents=True)
        (staging / "part").write_bytes(b"z" * 128)

        result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--yes"])

        assert result.exit_code == 0
        assert not cdir.exists()
        assert _status(demo) != "cached"

    def test_cache_clear_removes_an_already_empty_leaked_dir(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # Same leak, but killed before a byte was staged: no file to list, so the
        # command must still act rather than report "nothing to clear".
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        (cdir / ".archive.zip.ab12cd").mkdir(parents=True)

        result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--yes"])

        assert result.exit_code == 0
        assert "Nothing to clear" not in result.stdout
        assert not cdir.exists()
        assert _status(demo) != "cached"

    @pytest.mark.skipif(
        getattr(os, "geteuid", lambda: 1)() == 0,
        reason="root ignores the permission bits that make rmdir fail",
    )
    def test_cache_clear_survives_a_directory_it_cannot_remove(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The sweep runs *after* the files are gone. An rmdir that raises there
        # (read-only parent here; a Windows cwd lock or a concurrent staging
        # mkdtemp in the field) must be reported like any other failed removal,
        # not abort the command over the summary of what was already cleared.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        blob = cdir / "data.bin"
        locked = cdir / "locked"
        (locked / "staging").mkdir(parents=True)
        blob.write_bytes(b"y" * 1024)
        locked.chmod(0o555)  # cannot unlink `staging` out of `locked`
        try:
            result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--yes"])
        finally:
            locked.chmod(0o755)  # never poison the tmp_path teardown

        assert result.exit_code == 0
        assert not blob.exists()  # the deletions still happened
        assert "Cleared 1.0 KB" in result.stdout  # ...and were still reported
        assert "Could not remove" in result.stdout
        assert f"{demo.caches[0]}/locked/staging" in result.stdout

    def test_cache_clear_sweeps_a_shared_cache_dir_once(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # Several demos declare the same cache name, which puts that one path in
        # the sweep list once per demo. The count must be of directories, not of
        # demos claiming them.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        shared = next(
            name
            for name, n in Counter(
                name for d in iter_demos() for name in d.caches
            ).most_common()
            if n > 1
        )
        claimants = [d.key for d in iter_demos() if shared in d.caches]
        assert len(claimants) > 1
        cdir = tmp_path / shared
        (cdir / ".archive.zip.ab12cd").mkdir(parents=True)

        result = runner.invoke(app, ["demo", "cache", "clear", *claimants, "--dry-run"])

        assert result.exit_code == 0
        # The leaked staging dir + the cache dir itself — once each, however
        # many demos claim the name.
        assert "2 dir(s) removed" in result.stdout

    def test_cache_clear_dry_run_previews_the_emptied_cache_dir(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # Nothing under the cache dir is empty *yet*, but clearing its file
        # leaves it empty and the sweep then removes it — so the preview has to
        # look ahead at the deletions it is previewing.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        cdir.mkdir()
        blob = cdir / "data.bin"
        blob.write_bytes(b"y" * 1024)

        result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--dry-run"])

        assert result.exit_code == 0
        assert "1 dir(s) removed" in result.stdout
        assert blob.exists()  # still a preview

    def test_cache_clear_counts_a_shared_cache_file_once(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # Same shared cache name, but with a file in it: the directory was
        # walked once per claimant, so the file was listed — and its bytes
        # counted, and its "freed" bytes tallied — once per demo.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        shared = next(
            name
            for name, n in Counter(
                name for d in iter_demos() for name in d.caches
            ).most_common()
            if n > 1
        )
        claimants = [d.key for d in iter_demos() if shared in d.caches]
        assert len(claimants) > 1
        cdir = tmp_path / shared
        cdir.mkdir()
        (cdir / "data.bin").write_bytes(b"y" * 1024)

        result = runner.invoke(app, ["demo", "cache", "clear", *claimants, "--yes"])

        assert result.exit_code == 0
        assert "1 item(s), 1.0 KB" in result.stdout
        assert "Cleared 1.0 KB" in result.stdout
        assert not cdir.exists()

    def test_cache_clear_reports_the_cache_dir_it_will_remove(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The summary is printed before anything is unlinked in *both* modes, so
        # a real run has to look ahead at its own deletions exactly as --dry-run
        # does; otherwise it removes a directory it never mentioned, and the
        # same state previews differently from how it clears.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        demo = next(d for d in iter_demos() if d.caches)
        cdir = tmp_path / demo.caches[0]
        cdir.mkdir()
        (cdir / "data.bin").write_bytes(b"y" * 1024)

        result = runner.invoke(app, ["demo", "cache", "clear", demo.key, "--yes"])

        assert result.exit_code == 0
        assert "1 dir(s) removed" in result.stdout
        assert not cdir.exists()

    def test_cache_clear_orphans(self, runner, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        orphan = tmp_path / "orphan-dir"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 100)
        result = runner.invoke(app, ["demo", "cache", "clear", "--orphans", "--yes"])
        assert result.exit_code == 0
        assert not orphan.exists()


class TestProtectedInputDirs:
    """A hand-placed demo input is inventoried, but never an orphan and never cleared.

    ``~/.cache/luxar/milky_way_gaia_3m/`` holds a CC BY-NC catalog the user put
    there by hand; there is no download to get it back. Neither DEMO_META state
    protected it before: unclaimed it read as an orphan (`clear --orphans`
    rmtree'd it), claimed it read as a download (`clear <key>`/`--all` did).
    """

    @staticmethod
    def _protected_name() -> str:
        """A REAL protected name — used only where the claim state is irrelevant."""
        from luxar.demos.registry import PROTECTED_INPUT_DIRS

        assert PROTECTED_INPUT_DIRS, "nothing to protect: these tests would be vacuous"
        return sorted(PROTECTED_INPUT_DIRS)[0]

    @staticmethod
    def _unclaimed_protected(monkeypatch) -> str:
        """A protected name that is protected AND claimed by no demo.

        Deliberately SYNTHETIC rather than derived from today's demo set: the
        orphan route only ever sees unclaimed dirs, and if the Gaia demo one day
        declares ``caches: ["milky_way_gaia_3m"]`` (a state the design must
        survive, see the class docstring) a real-name search would find nothing
        and red these cases while the behaviour was perfectly fine.
        """
        name = "synthetic_hand_placed_input"
        monkeypatch.setattr(
            "luxar.demos.registry.PROTECTED_INPUT_DIRS", frozenset({name})
        )
        return name

    def _make_protected(self, root: Path, name: str | None = None) -> Path:
        """A protected input directory holding one irreplaceable file."""
        cdir = root / (name or self._protected_name())
        cdir.mkdir()
        payload = cdir / "catalog.zarr.zip"
        payload.write_bytes(b"g" * 4096)
        return payload

    def test_inventory_flags_it_and_never_calls_it_an_orphan(self, tmp_path) -> None:
        from luxar.demos import registry as demo_registry

        payload = self._make_protected(tmp_path)
        orphan = tmp_path / "orphan-xyz"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 8)
        # A near-miss name: protection is by EXACT directory name, so a leftover
        # copy is still an ordinary (deletable) orphan — a substring match here
        # would make `<name>_old` permanently unclearable.
        nearmiss = tmp_path / f"{self._protected_name()}_old"
        nearmiss.mkdir()
        (nearmiss / "junk.bin").write_bytes(b"q" * 8)

        entries = {
            e.path.name: e for e in demo_registry.inventory_caches(cache_root=tmp_path)
        }

        # Inventoried (the user should see the bytes they are holding)…
        assert set(entries) == {self._protected_name(), "orphan-xyz", nearmiss.name}
        assert entries[self._protected_name()].size_bytes == 4096
        # …and flagged independently of DEMO_META, so an empty `demo_keys` here
        # no longer means "orphan" — while a real orphan beside it still is one.
        assert entries[self._protected_name()].protected
        assert not entries["orphan-xyz"].protected
        assert entries["orphan-xyz"].demo_keys == ()
        assert not entries[nearmiss.name].protected
        assert payload.exists()

    def test_a_near_miss_name_is_still_a_clearable_orphan(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The behavioural half of the exact-name rule: `<protected>_old` must go.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        name = self._unclaimed_protected(monkeypatch)
        payload = self._make_protected(tmp_path, name)
        nearmiss = tmp_path / f"{name}_old"
        nearmiss.mkdir()
        (nearmiss / "junk.bin").write_bytes(b"q" * 8)

        result = runner.invoke(app, ["demo", "cache", "clear", "--orphans", "--yes"])

        assert result.exit_code == 0
        assert not nearmiss.exists()
        assert payload.exists()

    @staticmethod
    def _row(stdout: str, name: str) -> str:
        """The one `cache list` row for ``name`` (ANSI stripped)."""
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
        return next(line for line in plain.splitlines() if name in line)

    def test_cache_list_shows_it_without_the_orphan_warning(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # Asserted per ROW, not over the whole output: a global "no ORPHAN
        # anywhere" only holds while the fixture root happens to have no orphan.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        self._make_protected(tmp_path)
        orphan = tmp_path / "orphan-xyz"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 8)

        result = runner.invoke(app, ["demo", "cache", "list"])

        assert result.exit_code == 0
        protected_row = self._row(result.stdout, self._protected_name())
        assert "ORPHAN" not in protected_row
        assert "hand-placed input" in protected_row
        assert "ORPHAN" in self._row(result.stdout, "orphan-xyz")

    def test_cache_list_marks_a_claimed_protected_dir_too(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The post-#1558 state the design has to survive: a demo declares the
        # name (so the row has an owner) and it must STILL read as a kept input,
        # or `clear --all` looks legitimate to the reader.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, _other, name = self._claiming_pair(monkeypatch)
        (tmp_path / name).mkdir()
        (tmp_path / name / "catalog.zarr.zip").write_bytes(b"g" * 4096)

        result = runner.invoke(app, ["demo", "cache", "list"])

        assert result.exit_code == 0
        row = self._row(result.stdout, name)
        assert protected_demo.key in row  # the claim is still reported…
        assert "hand-placed input" in row  # …and so is the protection
        assert "ORPHAN" not in row

    def test_clear_orphans_spares_it_and_still_clears_a_real_orphan(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        # Synthetic name: the notice below is the UNCLAIMED-protected behaviour,
        # which a real name would stop modelling the day a demo claims it.
        payload = self._make_protected(tmp_path, self._unclaimed_protected(monkeypatch))
        orphan = tmp_path / "orphan-dir"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 100)

        result = runner.invoke(app, ["demo", "cache", "clear", "--orphans", "--yes"])

        assert result.exit_code == 0
        assert not orphan.exists()  # the sweep still does its job…
        assert payload.exists()  # …and the irreplaceable file survives it
        assert payload.parent.exists()
        assert "hand-placed" in result.stdout

    @staticmethod
    def _claiming_pair(monkeypatch) -> tuple[object, object, str]:
        """Make some demo's cache name protected; return (protected, other, name).

        No bundled demo declares ``milky_way_gaia_3m`` today (that is the whole
        problem — it reads as an orphan), so the by-key and ``--all`` routes are
        exercised by protecting the cache name of a demo that DOES declare one.
        The second demo keeps the test honest: the command must still clear
        everything else it was asked to.
        """
        demos = iter_demos()
        protected_demo = next(d for d in demos if d.caches)
        name = protected_demo.caches[0]
        other = next(d for d in demos if d.caches and name not in d.caches)
        monkeypatch.setattr(
            "luxar.demos.registry.PROTECTED_INPUT_DIRS", frozenset({name})
        )
        return protected_demo, other, name

    @pytest.mark.parametrize("route", ["by-key", "all"])
    def test_clear_keeps_it_but_clears_the_rest_of_the_selection(
        self, runner, tmp_path, monkeypatch, route
    ) -> None:
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, other, name = self._claiming_pair(monkeypatch)
        kept_dir = tmp_path / name
        kept_dir.mkdir()
        kept = kept_dir / "catalog.zarr.zip"
        kept.write_bytes(b"g" * 4096)
        doomed_dir = tmp_path / other.caches[0]
        doomed_dir.mkdir()
        doomed = doomed_dir / "archive.zip"
        doomed.write_bytes(b"z" * 1024)

        argv = ["--all"] if route == "all" else [protected_demo.key, other.key]
        result = runner.invoke(app, ["demo", "cache", "clear", *argv, "--yes"])

        assert result.exit_code == 0
        assert kept.exists(), "a hand-placed input must survive a cache clear"
        # The directory too: the empty-dir sweep must not rmdir it either.
        assert kept_dir.exists()
        assert not doomed.exists(), "the rest of the selection must still clear"
        assert "hand-placed" in result.stdout

    def test_an_empty_protected_dir_survives_the_empty_dir_sweep(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # The directory is skipped before it reaches `cache_dirs`, not merely
        # before its files are collected: an input directory the user has created
        # but not yet filled holds no bytes, and the sweep removes exactly those.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, _other, name = self._claiming_pair(monkeypatch)
        kept_dir = tmp_path / name
        kept_dir.mkdir()

        result = runner.invoke(
            app, ["demo", "cache", "clear", protected_demo.key, "--yes"]
        )

        assert result.exit_code == 0
        assert kept_dir.exists()
        assert "dir(s) removed" not in result.stdout

    @pytest.mark.parametrize("mode", ["--dry-run", "--yes"])
    def test_the_keep_notice_prints_in_both_preview_and_real_run(
        self, runner, tmp_path, monkeypatch, mode
    ) -> None:
        # The preview has to match the run, including what it refuses to touch.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, _other, name = self._claiming_pair(monkeypatch)
        kept_dir = tmp_path / name
        kept_dir.mkdir()
        kept = kept_dir / "catalog.zarr.zip"
        kept.write_bytes(b"g" * 4096)

        result = runner.invoke(
            app, ["demo", "cache", "clear", protected_demo.key, mode]
        )

        assert result.exit_code == 0
        assert f"Keeping {name}/" in result.stdout
        assert kept.exists()
        # Nothing else was selected, so the usual empty-selection line still
        # comes out — the guard must not turn that path into a traceback.
        assert "Nothing to clear" in result.stdout

    @pytest.mark.parametrize("flags", [["--dry-run"], ["--dry-run", "--yes"]])
    def test_a_non_empty_dry_run_previews_the_keep_notice(
        self, runner, tmp_path, monkeypatch, flags
    ) -> None:
        # The other notice case selects ONLY the protected demo, so it exits at
        # "Nothing to clear" and never reaches the `if dry_run:` branch — a preview
        # with real work to show is where the notice could go missing. `--dry-run
        # --yes` is included because --dry-run has to win: it is the flag people
        # reach for precisely when they are unsure.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, other, name = self._claiming_pair(monkeypatch)
        kept_dir = tmp_path / name
        kept_dir.mkdir()
        kept = kept_dir / "catalog.zarr.zip"
        kept.write_bytes(b"g" * 4096)
        other_dir = tmp_path / other.caches[0]
        other_dir.mkdir()
        spared = other_dir / "archive.zip"
        spared.write_bytes(b"z" * 1024)
        # An EMPTY dir inside it: the empty-directory sweep is a second deletion
        # path that the preview must also only preview (a leaked staging dir is
        # exactly the shape a real cache has).
        staging = other_dir / ".archive.zip.ab12cd"
        staging.mkdir()

        result = runner.invoke(
            app,
            ["demo", "cache", "clear", protected_demo.key, other.key, *flags],
        )

        assert result.exit_code == 0
        assert f"Keeping {name}/" in result.stdout
        assert "(--dry-run: nothing deleted)" in result.stdout
        assert kept.exists()
        assert spared.exists(), "--dry-run must delete nothing, even with --yes"
        assert staging.exists(), "--dry-run swept an empty directory for real"

    def test_no_keep_notice_when_the_input_dir_does_not_exist(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # A demo can declare the name on a machine that never received the file.
        # Announcing a directory that isn't there is pure noise — and on the real
        # `--all` route it would fire on every clear, everywhere.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        protected_demo, _other, _name = self._claiming_pair(monkeypatch)

        result = runner.invoke(
            app, ["demo", "cache", "clear", protected_demo.key, "--yes"]
        )

        assert result.exit_code == 0
        assert "Keeping" not in result.stdout
        assert "Nothing to clear" in result.stdout

    def test_the_keep_notice_is_printed_once_per_directory(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # One cache name can be claimed by several demos, which walks that one
        # directory once per claimant. The notice is about the directory, so a
        # selection covering both claimants still earns exactly one line.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        shared = next(
            name
            for name, n in Counter(
                name for d in iter_demos() for name in d.caches
            ).most_common()
            if n > 1
        )
        claimants = [d.key for d in iter_demos() if shared in d.caches]
        assert len(claimants) > 1
        monkeypatch.setattr(
            "luxar.demos.registry.PROTECTED_INPUT_DIRS", frozenset({shared})
        )
        (tmp_path / shared).mkdir()
        (tmp_path / shared / "catalog.zarr.zip").write_bytes(b"g" * 4096)

        result = runner.invoke(app, ["demo", "cache", "clear", *claimants, "--yes"])

        assert result.exit_code == 0
        assert result.stdout.count("Keeping") == 1

    def test_orphans_says_nothing_about_a_claimed_protected_dir(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        # `--orphans` only ever targets UNCLAIMED dirs, so a claimed protected one
        # was never a candidate: notifying about it would mean volunteering advice
        # ("delete it by hand if you really mean to") about a directory the user
        # neither selected nor endangered.
        monkeypatch.setattr("luxar.demos.registry.DEMO_CACHE_ROOT", tmp_path)
        _protected_demo, other, name = self._claiming_pair(monkeypatch)
        kept_dir = tmp_path / name
        kept_dir.mkdir()
        kept = kept_dir / "catalog.zarr.zip"
        kept.write_bytes(b"g" * 4096)
        orphan = tmp_path / "orphan-dir"
        orphan.mkdir()
        (orphan / "junk.bin").write_bytes(b"q" * 100)

        result = runner.invoke(
            app, ["demo", "cache", "clear", other.key, "--orphans", "--yes"]
        )

        assert result.exit_code == 0
        assert not orphan.exists()  # the real orphan still goes
        assert kept.exists()  # the claimed input is still spared…
        assert "Keeping" not in result.stdout  # …silently, as it was never at risk


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

    def test_deps_only_filters_to_one_module_case_insensitively(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "anndata", DependencySpec("anndata>=0.10,<0.13", "demos"), True, True
            ),
            DependencyStatus(
                "scipy", DependencySpec("scipy>=1.15,<2", "demos"), True, True
            ),
        ]
        with patch("luxar.demos.survey", return_value=fake):
            loud = runner.invoke(app, ["demo", "deps", "--only", "  SCIPY  "])
            quiet = runner.invoke(app, ["demo", "deps", "--only", "scipy"])
        assert loud.exit_code == quiet.exit_code == 0
        assert loud.stdout == quiet.stdout
        assert "scipy" in loud.stdout
        assert "anndata" not in loud.stdout
        assert "1 optional demo dependency" in loud.stdout

    def test_deps_only_install_is_a_noop_when_target_is_satisfied(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "scipy", DependencySpec("scipy>=1.15,<2", "demos"), True, True
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process") as proc:
                result = runner.invoke(
                    app, ["demo", "deps", "--only", "scipy", "--install"]
                )
        assert result.exit_code == 0
        proc.assert_not_called()
        assert "Nothing missing" in result.stdout

    def test_deps_only_rejects_an_unknown_module(self, runner) -> None:
        result = runner.invoke(app, ["demo", "deps", "--only", "not_a_module"])
        assert result.exit_code == 1
        assert "Unknown optional dependency module 'not_a_module'" in result.stdout
        assert "Valid modules:" in result.stdout
        assert "scipy" in result.stdout

    def test_deps_only_and_extra_are_mutually_exclusive(self, runner) -> None:
        result = runner.invoke(
            app,
            ["demo", "deps", "--extra", "demos", "--only", "scipy"],
        )
        assert result.exit_code == 2
        assert "--extra and --only cannot be combined" in result.stdout

    def test_deps_extra_report_hint_keeps_the_filter(self, runner) -> None:
        """The hinted CLI command must match the pip command shown beside it.

        With `--extra gsplats` the direct pip line installs only that extra, so
        a bare `luxar demo deps --install` hint (every unmet extra) would not be
        the equivalent alternative it claims to be.
        """
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "torch", DependencySpec("torch>=2,<3", "gsplats"), False, False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps", "--extra", "gsplats"])
        assert result.exit_code == 1
        assert "luxar demo deps --extra gsplats --install" in result.stdout

    def test_deps_exits_nonzero_when_something_is_missing(self, runner) -> None:
        """A CI gate can rely on the exit code, so it must track missing-ness."""
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False, False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 1
        assert "1 missing or outdated: phony_xyz" in result.stdout

    def test_deps_flags_an_installed_but_outdated_package(self, runner) -> None:
        """Importable but below its pin: OUTDATED, and it fails the exit gate."""
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=2", "demos"), True, False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 1
        assert "OUTDATED" in result.stdout
        assert "1 missing or outdated: phony_xyz" in result.stdout

    def test_deps_exits_zero_when_all_present(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), True, True
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 0
        assert "all 1 optional dependency installed." in result.stdout

    def test_deps_columns_are_never_narrower_than_their_headers(self, runner) -> None:
        """A one-row report (module shorter than "MODULE") must not go ragged."""
        import re

        from luxar.demos._dependencies import DependencyStatus

        fake = [DependencyStatus("ab", DependencySpec("ab>=1", "demos"), True, True)]
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
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False, False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process") as proc:
                result = runner.invoke(app, ["demo", "deps", "--install", "--dry-run"])
        assert result.exit_code == 0
        proc.assert_not_called()
        assert "pip install" in result.stdout

    def test_deps_only_dry_run_shows_the_exact_requirement(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus("gdown", DependencySpec("gdown>=5,<6", ""), False, False)
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process") as proc:
                result = runner.invoke(
                    app,
                    [
                        "demo",
                        "deps",
                        "--only",
                        "gdown",
                        "--install",
                        "--dry-run",
                    ],
                )
        assert result.exit_code == 0
        proc.assert_not_called()
        assert "pip install 'gdown>=5,<6'" in result.stdout
        assert "--dry-run: nothing installed" in result.stdout

    def test_deps_install_propagates_pip_failure(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False, False
            )
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process", return_value=17):
                result = runner.invoke(app, ["demo", "deps", "--install"])
        assert result.exit_code == 17
        assert "pip exited 17" in result.stdout

    def test_deps_install_invokes_pip_with_the_missing_extras(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus(
                "phony_xyz", DependencySpec("phony-xyz>=1", "gsplats"), False, False
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

    @pytest.mark.parametrize(
        ("module", "requirement", "extra"),
        [
            ("gdown", "gdown>=5,<6", ""),
            ("scipy", "scipy>=1.15,<2", "demos"),
        ],
    )
    def test_deps_only_installs_the_exact_requirement(
        self, runner, module: str, requirement: str, extra: str
    ) -> None:
        """Targeting one row never pulls its whole extra.

        The two cases pin both sides: an orphan gains a managed install path,
        while an extra-backed row installs only itself instead of every package
        in that extra.
        """
        from luxar.demos._dependencies import DependencyStatus

        missing = DependencyStatus(
            module, DependencySpec(requirement, extra), False, False
        )
        present = DependencyStatus(
            module, DependencySpec(requirement, extra), True, True
        )
        with patch("luxar.demos.survey", side_effect=[[missing], [present]]):
            with patch(
                "luxar.cli.demo_commands.run_child_process", return_value=0
            ) as proc:
                result = runner.invoke(
                    app, ["demo", "deps", "--only", module, "--install"]
                )

        assert result.exit_code == 0, result.stdout
        proc.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", requirement],
            label="pip install",
        )
        assert f"Installed: '{requirement}'" in result.stdout
        assert "luxar[" not in result.stdout

    def test_deps_only_fails_when_the_requested_row_remains_unmet(self, runner) -> None:
        from luxar.demos._dependencies import DependencyStatus

        missing = DependencyStatus(
            "phony_xyz", DependencySpec("phony-xyz>=1", "demos"), False, False
        )
        with patch("luxar.demos.survey", side_effect=[[missing], [missing]]):
            with patch("luxar.cli.demo_commands.run_child_process", return_value=0):
                result = runner.invoke(
                    app, ["demo", "deps", "--only", "phony_xyz", "--install"]
                )

        assert result.exit_code == 1
        assert "Still missing or outdated after install: phony_xyz" in result.stdout

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
        """gdown is installable only by name, so generic --install can't cover it."""
        from luxar.demos._dependencies import DependencyStatus

        fake = [DependencyStatus("gdown", DependencySpec("gdown", ""), False, False)]
        with patch("luxar.demos.survey", return_value=fake):
            result = runner.invoke(app, ["demo", "deps"])
        assert result.exit_code == 1
        assert "Not in any extra" in result.stdout
        assert "luxar demo deps --only gdown --install" in result.stdout

    def test_deps_install_with_only_orphans_is_a_successful_noop(self, runner) -> None:
        """Generic --install cannot cover orphans and must behave consistently.

        The same missing orphan already exits 0 when an extra was successfully
        installed beside it. With no extra to invoke, --install should likewise
        report every untouched row without turning the no-op into a false failure.
        """
        from luxar.demos._dependencies import DependencyStatus

        fake = [
            DependencyStatus("gdown", DependencySpec("gdown", ""), False, False),
            DependencyStatus("other", DependencySpec("other>=2", ""), False, False),
        ]
        with patch("luxar.demos.survey", return_value=fake):
            with patch("luxar.cli.demo_commands.run_child_process") as proc:
                result = runner.invoke(app, ["demo", "deps", "--install"])

        assert result.exit_code == 0
        proc.assert_not_called()
        assert "No Luxar extra provides the unmet requirements" in result.stdout
        assert "luxar demo deps --only gdown --install" in result.stdout
        assert "luxar demo deps --only other --install" in result.stdout

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
        assert blank.stdout == full.stdout

    def test_deps_blank_only_is_treated_as_unset(self, runner) -> None:
        blank = runner.invoke(app, ["demo", "deps", "--only", "   "])
        full = runner.invoke(app, ["demo", "deps"])
        assert blank.exit_code == full.exit_code
        assert blank.stdout == full.stdout

    def test_deps_dry_run_without_install_says_it_is_inert(self, runner) -> None:
        result = runner.invoke(app, ["demo", "deps", "--dry-run"])
        assert "--dry-run only applies with --install" in result.stdout

    def test_deps_does_not_blame_the_install_for_an_orphan_spec(self, runner) -> None:
        """gdown is in no extra, so --install never attempts it.

        Reporting it as "still missing after install" made a successful install
        look like a failure, and exited 1 on work that fully succeeded.
        """
        from luxar.demos._dependencies import DependencyStatus

        orphan = DependencyStatus("gdown", DependencySpec("gdown", ""), False, False)
        before = [
            orphan,
            DependencyStatus("m", DependencySpec("m>=1", "demos"), False, False),
        ]
        after = [
            orphan,
            DependencyStatus("m", DependencySpec("m>=1", "demos"), True, True),
        ]
        calls = {"n": 0}

        def fake_survey(extra=None):
            calls["n"] += 1
            return before if calls["n"] == 1 else after

        with patch("luxar.demos.survey", side_effect=fake_survey):
            with patch("luxar.cli.demo_commands.run_child_process", return_value=0):
                result = runner.invoke(app, ["demo", "deps", "--install"])

        assert result.exit_code == 0, "the extra installed fine; must not exit 1"
        assert "Still missing or outdated after install" not in result.stdout
        # ...but it must not claim completeness either.
        assert "Still to install by hand" in result.stdout
        assert "gdown" in result.stdout


def test_demo_runs_dir_matches_demo_cache_root() -> None:
    """demo_runs deliberately duplicates DEMO_CACHE_ROOT (import-weight); pin it.

    If the demo cache root ever moves, the running-demo registry must move
    with it — this is the guard for that silent divergence.
    """
    from luxar.demos.registry import DEMO_CACHE_ROOT, NON_CACHE_DIRS
    from luxar.utils.demo_runs import DEMO_RUNS_DIR

    assert DEMO_RUNS_DIR.parent == DEMO_CACHE_ROOT
    # …and it is not a cache: `cache clear --orphans` must never delete the
    # record of demos that are still running.
    assert DEMO_RUNS_DIR.name in NON_CACHE_DIRS


def test_running_registry_is_not_inventoried_as_a_cache(tmp_path: Path) -> None:
    """The pidfile registry is live state, so it is neither listed nor cleared."""
    from luxar.demos import registry as demo_registry

    (tmp_path / "running").mkdir()
    (tmp_path / "running" / "4242.json").write_text('{"key": "lorenz", "pgid": 4242}')
    (tmp_path / "some_cache").mkdir()
    (tmp_path / "some_cache" / "blob.bin").write_bytes(b"x" * 8)

    entries = demo_registry.inventory_caches(cache_root=tmp_path)
    assert [e.path.name for e in entries] == ["some_cache"]


_MAKEFILE = Path(__file__).resolve().parents[6] / "Makefile"


_CACHE_ROOT_IN_MAKEFILE = "~/.cache/luxar"


def _strip_make_prefixes(line: str) -> str:
    """Drop make's per-line prefixes (``@`` quiet, ``-`` ignore-errors, ``+``).

    They are make syntax, not shell syntax — and stripping them from anything but
    the HEAD of a logical line would eat a leading ``-mindepth`` / ``-exec``.
    """
    while line[:1] in ("@", "-", "+"):
        line = line[1:]
    return line


def _join_continuations(physical: list[str]) -> list[str]:
    """Join trailing-backslash lines, so each result is one line make would shell."""
    logical: list[str] = []
    pending: list[str] = []
    for text in physical:
        if text.endswith("\\"):
            pending.append(text[:-1])
            continue
        logical.append("".join([*pending, text]))
        pending = []
    if pending:
        logical.append("".join(pending))
    return logical


def _recipe_lines(target: str) -> list[str]:
    """The LOGICAL recipe lines of ``target``: continuations joined, prefixes gone.

    Make gives each *logical* line its own shell, so grading the recipe means
    reproducing that split — a single joined script hides both a failure that
    should abort the target and state (a ``cd``) that make deliberately drops
    between lines.

    Parsing details that are all load-bearing:

    * continuations are joined FIRST and the make prefixes (``@`` quiet, ``-``
      ignore-errors, ``+`` always-run) stripped only from the head of each
      logical line — otherwise a natural reformat that starts a continuation with
      ``-mindepth`` loses its leading dash and the recipe is mis-graded;
    * a ``#`` comment is not part of the recipe (a target that merely *mentions*
      the protected name while wiping everything must not look guarded);
    * a blank or column-0 comment line inside the block is skipped rather than
      treated as the end of the recipe, so an interleaved comment (or
      ``.RECIPEPREFIX``) cannot silently truncate the parse to nothing.
    """
    lines = _MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith(f"{target}:")),
        None,
    )
    if start is None:
        pytest.skip(f"Makefile has no {target} target")

    physical: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or (not line.startswith("\t") and line.startswith("#")):
            continue  # interleaved blank/comment, not the end of the recipe
        if not line.startswith("\t"):
            break  # the next target (or a variable): recipe over
        physical.append(line[1:])

    commands = []
    for line in _join_continuations(physical):
        head = _strip_make_prefixes(line)
        if head.lstrip().startswith("#") or not head.strip():
            continue
        commands.append(head)
    return commands


def _clean_cache_commands() -> str:
    """The ``clean-cache`` recipe as runnable shell (see :func:`_recipe_lines`)."""
    return "\n".join(_recipe_lines("clean-cache"))


def _run_clean_cache(cache_root: Path, sandbox_home: Path) -> None:
    """Run the real recipe against ``cache_root``, with ``HOME`` sandboxed.

    Substitution-then-execute is the only way to grade what the target DOES: a
    substring assertion passes on a recipe that drops ``-mindepth 1`` (which makes
    ``find`` match the root itself, whose name is not the excluded one, so
    ``rm -rf`` takes the whole tree).

    That makes this helper the most dangerous code in the file, so its gate is
    POSITIVE (prove the substitution took) rather than a denylist of spellings —
    a hoisted `$(CACHE_DIR)` would sail through a denylist, then expand to the
    empty string in ``sh`` and turn the removal into ``find / …``. ``HOME`` and
    ``XDG_CACHE_HOME`` are redirected into a sandbox as well, so a recipe that
    reaches for the user's cache by some other spelling (``cd ~/.cache && find
    luxar/ …``) can only reach the sandbox. Each logical line runs under the
    Makefile's own ``.SHELLFLAGS`` (``bash -e -o pipefail``): without ``-e`` a
    recipe whose removal fails would still grade green.
    """
    script = _clean_cache_commands().replace(_CACHE_ROOT_IN_MAKEFILE, str(cache_root))
    assert script.strip(), "clean-cache recipe parsed to nothing"
    assert str(cache_root) in script, "substitution did not take"
    # No `$` and no `~` left: either would be expanded by the shell against the
    # environment instead of pointing at the throwaway root.
    assert "$" not in script and "~" not in script, (
        f"unexpanded expansion left in recipe: {script}"
    )
    env = {
        **os.environ,
        "HOME": str(sandbox_home),
        "XDG_CACHE_HOME": str(sandbox_home / ".cache"),
    }
    for line in script.splitlines():
        proc = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", line],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, f"{line}\n{proc.stderr}"
        # The recipe's last line is an `echo`, so rc alone is nearly vacuous.
        assert proc.stderr == "", f"{line}\n{proc.stderr}"


def _populate_cache(root: Path) -> tuple[list[Path], Path]:
    """A protected payload per protected name + one ordinary cache dir."""
    from luxar.demos.registry import PROTECTED_INPUT_DIRS

    assert PROTECTED_INPUT_DIRS, "nothing to protect: these tests would be vacuous"
    payloads = []
    for name in sorted(PROTECTED_INPUT_DIRS):
        (root / name).mkdir(parents=True)
        payload = root / name / "catalog.zarr.zip"
        payload.write_bytes(b"g" * 64)
        payloads.append(payload)
    sibling = root / "ordinary_cache"
    sibling.mkdir(parents=True)
    (sibling / "blob.bin").write_bytes(b"z" * 8)
    return payloads, sibling


def _sandbox_home(tmp: Path) -> Path:
    """A decoy HOME beside the cache root, with a cache dir that must survive."""
    home = tmp / "sandbox-home"
    (home / ".cache" / "luxar" / "decoy").mkdir(parents=True)
    (home / ".cache" / "luxar" / "decoy" / "keep.bin").write_bytes(b"k" * 4)
    return home


@pytest.mark.skipif(
    not _MAKEFILE.exists(), reason="no Makefile (packaged install without repo root)"
)
def test_make_clean_cache_recipe_keeps_the_protected_dirs_when_run() -> None:
    """`make clean-cache` must not delete a hand-placed input.

    This is the most casual deletion route of all — a bare ``rm -rf
    ~/.cache/luxar`` there undoes the whole guard. The recipe cannot read
    PROTECTED_INPUT_DIRS (a clean target has to work with a broken env), so the
    names are duplicated in the Makefile and this test is what pins the copies
    together — by RUNNING the recipe against a throwaway root.
    (``clean-all`` reaches the same recipe; that delegation is pinned separately
    by :func:`test_no_other_make_recipe_removes_the_cache_root`.)
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "luxar"
        payloads, sibling = _populate_cache(root)
        home = _sandbox_home(Path(tmp))

        _run_clean_cache(root, home)

        for payload in payloads:
            assert payload.exists(), f"clean-cache destroyed {payload.parent.name}/"
        assert not sibling.exists(), "clean-cache stopped clearing ordinary caches"
        assert (home / ".cache" / "luxar" / "decoy" / "keep.bin").exists(), (
            "the recipe reached outside the root it was given"
        )


@pytest.mark.skipif(
    not _MAKEFILE.exists(), reason="no Makefile (packaged install without repo root)"
)
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() == 0,
    reason="root ignores the permission bits that make the removal fail",
)
def test_make_clean_cache_fails_loudly_when_it_cannot_remove() -> None:
    """A clear that cannot clear must not report success.

    This is what the Makefile's own ``.SHELLFLAGS`` (``bash -e -o pipefail``) buy,
    and therefore what the runner has to reproduce: joined into one prefix-less
    ``sh`` script the recipe ends in ``echo``, so a failed removal is swallowed and
    a target that cleared nothing grades green.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "luxar"
        _populate_cache(root)
        home = _sandbox_home(Path(tmp))
        root.chmod(0o555)  # nothing under the root can be unlinked
        try:
            with pytest.raises(AssertionError):
                _run_clean_cache(root, home)
        finally:
            root.chmod(0o755)  # never poison the tmp teardown


@pytest.mark.skipif(
    not _MAKEFILE.exists(), reason="no Makefile (packaged install without repo root)"
)
def test_make_clean_cache_works_through_a_symlinked_cache_root() -> None:
    """A cache root symlinked onto another disk must still be cleared.

    ``find`` defaults to ``-P``, so without ``-H`` the link itself is the only
    thing matched and the target is a silent no-op that still prints success.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "elsewhere"
        payloads, sibling = _populate_cache(real)
        link = Path(tmp) / "luxar"
        link.symlink_to(real, target_is_directory=True)
        home = _sandbox_home(Path(tmp))

        _run_clean_cache(link, home)

        for payload in payloads:
            assert payload.exists()
        assert not sibling.exists(), "a symlinked cache root was never cleared"


@pytest.mark.skipif(
    not _MAKEFILE.exists(), reason="no Makefile (packaged install without repo root)"
)
def test_make_clean_cache_recipe_shape() -> None:
    """Belt-and-braces on the recipe text, beside the executed behaviour tests."""
    import re

    from luxar.demos.registry import PROTECTED_INPUT_DIRS

    assert PROTECTED_INPUT_DIRS, "nothing to protect: this test would be vacuous"
    commands = _clean_cache_commands()
    for name in PROTECTED_INPUT_DIRS:
        # Tolerant of quoting (`! -name 'x'` is the same behaviour, better hygiene).
        assert re.search(rf"!\s+-name\s+['\"]?{re.escape(name)}['\"]?", commands), (
            f"clean-cache does not spare {name}/ — see PROTECTED_INPUT_DIRS"
        )
    # A `find` that removes must be depth-bounded: unbounded, it matches the root
    # itself (whose name is not the excluded one) and takes the whole tree.
    removals = [ln for ln in commands.splitlines() if "rm -rf" in ln]
    assert removals
    for line in removals:
        assert "find" in line, f"unguarded removal in clean-cache: {line}"
        assert "-mindepth 1" in line and "-maxdepth 1" in line, line
    assert not re.search(
        r"rm\s+-rf\s+(~|\$[({]?HOME[)}]?|\$\{?XDG_CACHE_HOME\}?)/?\.?[a-z/]*"
        r"\.?cache/luxar/?\*?(\s|$)",
        commands,
    ), "a blanket removal of the cache root is back in clean-cache"


@pytest.mark.skipif(
    not _MAKEFILE.exists(), reason="no Makefile (packaged install without repo root)"
)
def test_no_other_make_recipe_removes_the_cache_root() -> None:
    """The guard is only as wide as the Makefile: no OTHER recipe may wipe the cache.

    ``clean-all`` is the target people actually type, and it is one line away
    from bypassing ``clean-cache`` entirely — so the invariant is checked over the
    whole file, not over one recipe.
    """
    import re

    guarded = set(_recipe_lines("clean-cache"))
    offenders = []
    for raw in _MAKEFILE.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("\t"):
            continue
        line = _strip_make_prefixes(raw[1:])
        if line in guarded or line.lstrip().startswith("#"):
            continue
        if re.search(r"rm\s+-rf?.*\.cache/luxar", line):
            offenders.append(line.strip())
    assert not offenders, f"cache-root removal outside clean-cache: {offenders}"
    # …and `clean-all` must still reach the guarded recipe rather than its own.
    assert any("clean-cache" in line for line in _recipe_lines("clean-all")), (
        "clean-all no longer delegates to clean-cache"
    )


def test_protected_and_non_cache_dirs_are_disjoint() -> None:
    """A name cannot be both, and the NON_CACHE_DIRS skip runs first.

    An overlapping name would be skipped outright by ``inventory_caches`` — so it
    would vanish from the very listing the protection promises it appears in,
    silently.
    """
    from luxar.demos.registry import NON_CACHE_DIRS, PROTECTED_INPUT_DIRS

    assert PROTECTED_INPUT_DIRS & NON_CACHE_DIRS == frozenset()


def test_gaia_hand_placed_catalog_dir_is_protected() -> None:
    """PROTECTED_INPUT_DIRS names a directory the Gaia demo really reads.

    The constant is a hard-coded name (deliberately independent of DEMO_META, so
    no demo edit can disarm it), which makes it drift-prone the moment the demo
    moves its cache file — this is the guard for that.
    """
    from luxar.demos.demo_gaia_milky_way_3m import CACHE_FILE
    from luxar.demos.registry import DEMO_CACHE_ROOT, PROTECTED_INPUT_DIRS

    assert CACHE_FILE.parent.name in PROTECTED_INPUT_DIRS
    assert CACHE_FILE.parent.parent == DEMO_CACHE_ROOT


class TestDemoStop:
    """`demo stop` — discovery listing, filtering, confirmation, exit codes."""

    @staticmethod
    def _fake_runs():
        from luxar.utils.demo_runs import DemoRun

        return [
            DemoRun(key="lorenz", pgid=111, pid=0, started=0.0, source="registry"),
            # Sweep entries carry the MODULE suffix ("4d_fractals"), which the
            # command must translate to the real key ("fractals_4d").
            DemoRun(key="4d_fractals", pgid=222, pid=0, started=0.0, source="sweep"),
        ]

    def test_no_running_demos_exits_zero(self, runner, monkeypatch) -> None:
        from luxar.cli import demo_commands

        monkeypatch.setattr(demo_commands.demo_runs, "discover_runs", lambda **kw: [])
        result = runner.invoke(app, ["demo", "stop"])
        assert result.exit_code == 0
        assert "No running demos" in result.stdout

    def test_dry_run_lists_translated_keys_and_kills_nothing(
        self, runner, monkeypatch
    ) -> None:
        from luxar.cli import demo_commands

        killed: list[int] = []
        monkeypatch.setattr(
            demo_commands.demo_runs, "discover_runs", lambda **kw: self._fake_runs()
        )
        monkeypatch.setattr(
            demo_commands.demo_runs, "stop_run", lambda r: killed.append(r.pgid) or True
        )
        result = runner.invoke(app, ["demo", "stop", "--dry-run"])
        assert result.exit_code == 0
        assert "lorenz" in result.stdout
        assert "fractals_4d" in result.stdout  # module suffix translated to key
        assert killed == []

    def test_confirmation_abort_kills_nothing(self, runner, monkeypatch) -> None:
        from luxar.cli import demo_commands

        killed: list[int] = []
        monkeypatch.setattr(
            demo_commands.demo_runs, "discover_runs", lambda **kw: self._fake_runs()
        )
        monkeypatch.setattr(
            demo_commands.demo_runs, "stop_run", lambda r: killed.append(r.pgid) or True
        )
        result = runner.invoke(app, ["demo", "stop"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.stdout
        assert killed == []

    def test_yes_stops_all_and_key_filters(self, runner, monkeypatch) -> None:
        from luxar.cli import demo_commands

        killed: list[int] = []
        monkeypatch.setattr(
            demo_commands.demo_runs, "discover_runs", lambda **kw: self._fake_runs()
        )
        monkeypatch.setattr(
            demo_commands.demo_runs, "stop_run", lambda r: killed.append(r.pgid) or True
        )
        result = runner.invoke(app, ["demo", "stop", "-y"])
        assert result.exit_code == 0
        assert killed == [111, 222]
        assert "All demos stopped" in result.stdout

        killed.clear()
        # Filter by the TRANSLATED key of a swept run (and by index-free key).
        result = runner.invoke(app, ["demo", "stop", "fractals_4d", "-y"])
        assert result.exit_code == 0
        assert killed == [222]

    def test_survivors_exit_one_with_their_pgids(self, runner, monkeypatch) -> None:
        from luxar.cli import demo_commands

        monkeypatch.setattr(
            demo_commands.demo_runs, "discover_runs", lambda **kw: self._fake_runs()
        )
        monkeypatch.setattr(
            demo_commands.demo_runs, "stop_run", lambda r: r.pgid == 111
        )
        result = runner.invoke(app, ["demo", "stop", "-y"])
        assert result.exit_code == 1
        # The manual-cleanup hint names the SURVIVOR's pgid, not the first run's.
        assert "kill -9 -222" in result.stdout
        assert "kill -9 -111" not in result.stdout

    def test_manual_hint_matches_the_platform(self, runner, monkeypatch) -> None:
        """Without process groups, `kill -9 -<pgid>` is not a runnable command.

        That is the whole listing off POSIX: `stop_run` refuses to signal a pid
        it cannot verify, so every run surfaces here and the hint is the only
        way out.
        """
        from luxar.cli import demo_commands

        monkeypatch.setattr(demo_commands, "can_kill_process_groups", lambda: False)
        monkeypatch.setattr(
            demo_commands.demo_runs, "discover_runs", lambda **kw: self._fake_runs()
        )
        monkeypatch.setattr(demo_commands.demo_runs, "stop_run", lambda r: False)
        result = runner.invoke(app, ["demo", "stop", "-y"])
        assert result.exit_code == 1
        assert "taskkill /F /T /PID 111" in result.stdout
        assert "kill -9" not in result.stdout
