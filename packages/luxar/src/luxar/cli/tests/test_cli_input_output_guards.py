"""CLI guards around inputs and outputs: overwrite refusal, remote URLs,
piped ``--help`` width, the ``cal --pdf`` dependency preflight, the tiled
``Done:`` timing, and ``merge`` flattening a partition input.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.fitting.fit import _done_seconds, _refuse_existing_output
from luxar.cli.tests._testing import normalized_cli_output
from luxar.cli.utils import local_input_path, reject_remote_input

_FAST_FIT = ["--preset", "draft", "--seeds", "40", "--iters", "60", "--device", "cpu"]
_TILED = ["--tiling", "uniform", "--tile-size", "12", "--overlap", "2"]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(scope="module")
def volume_npy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small 20^3 volume with one bright blob."""
    rng = np.random.default_rng(0)
    volume = rng.random((20, 20, 20)).astype(np.float32) * 0.05
    volume[7:13, 6:12, 8:14] = 1.0
    path = tmp_path_factory.mktemp("guards") / "volume.npy"
    np.save(path, volume)
    return path


@pytest.fixture(scope="module")
def flat_fit(volume_npy: Path) -> Path:
    out = volume_npy.parent / "flat.gsplats.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "fit", str(volume_npy), str(out), *_FAST_FIT]
    )
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="module")
def tiled_fit_output(volume_npy: Path) -> tuple[Path, str]:
    out = volume_npy.parent / "tiled.gsplats.zarr"
    result = CliRunner().invoke(
        app, ["gsplat", "fit", str(volume_npy), str(out), *_FAST_FIT, *_TILED]
    )
    assert result.exit_code == 0, result.output
    return out, result.output


class TestFitOverwrite:
    def test_existing_output_is_refused_without_overwrite(
        self, runner: CliRunner, volume_npy: Path, flat_fit: Path
    ) -> None:
        marker = flat_fit / "zarr.json"
        before = marker.stat().st_mtime_ns if marker.exists() else None
        result = runner.invoke(
            app, ["gsplat", "fit", str(volume_npy), str(flat_fit), *_FAST_FIT]
        )
        assert result.exit_code == 2
        text = normalized_cli_output(result)
        assert "exists" in text and "--overwrite" in text
        if before is not None:
            assert marker.stat().st_mtime_ns == before

    def test_overwrite_replaces_the_output(
        self, runner: CliRunner, volume_npy: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "again.gsplats.zarr"
        args = ["gsplat", "fit", str(volume_npy), str(out), *_FAST_FIT]
        assert runner.invoke(app, args).exit_code == 0
        result = runner.invoke(app, [*args, "--overwrite"])
        assert result.exit_code == 0, result.output

    def test_workers_are_exempt(self, tmp_path: Path) -> None:
        existing = tmp_path / "tile.gsplats.zarr"
        existing.mkdir()
        _refuse_existing_output(existing, overwrite=False, worker=True, plan_only=False)

    def test_plan_only_is_exempt(self, tmp_path: Path) -> None:
        # It writes a plan JSON (replacing an old one with a notice), never a
        # store; outside --tiling content it is refused instead (next test).
        store = tmp_path / "out.gsplats.zarr"
        store.mkdir()
        _refuse_existing_output(store, overwrite=False, worker=False, plan_only=True)

    def test_plan_only_outside_content_tiling_is_refused(
        self, runner: CliRunner, volume_npy: Path, flat_fit: Path
    ) -> None:
        marker = flat_fit / "zarr.json"
        before = marker.stat().st_mtime_ns if marker.exists() else None
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(volume_npy),
                str(flat_fit),
                *_FAST_FIT,
                "--plan-only",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "--plan-only needs --tiling content" in normalized_cli_output(result)
        if before is not None:
            assert marker.stat().st_mtime_ns == before

    def test_missing_output_is_never_refused(self, tmp_path: Path) -> None:
        _refuse_existing_output(
            tmp_path / "new.gsplats.zarr",
            overwrite=False,
            worker=False,
            plan_only=False,
        )


class TestDoneTiming:
    def test_tiled_fit_reports_wall_clock(
        self, tiled_fit_output: tuple[Path, str]
    ) -> None:
        _, output = tiled_fit_output
        done = [line for line in output.splitlines() if "Done:" in line]
        assert done, output
        assert "in 0.0s" not in done[-1]

    def test_leaf_reports_its_fit_time_and_a_tree_the_wall_clock(self) -> None:
        result = SimpleNamespace(stats={"time_seconds": 12.5})
        assert _done_seconds(result, True, started=0.0) == 12.5
        started = time.perf_counter() - 3.0
        assert _done_seconds(result, False, started=started) >= 3.0


class TestRemoteInputs:
    @pytest.mark.parametrize(
        "url",
        ["https://example.org/data/x.zarr", "s3://bucket/x.zarr", "gs://b/v.zarr"],
    )
    def test_reject_remote_input_names_the_url(self, url: str) -> None:
        # Typer hands the command a Path, which collapses `//` to `/`.
        with pytest.raises(typer.BadParameter) as excinfo:
            reject_remote_input(Path(url))
        message = str(excinfo.value)
        assert url in message
        assert "not supported" in message

    def test_local_paths_pass(self, tmp_path: Path) -> None:
        reject_remote_input(tmp_path / "volume.npy")
        reject_remote_input(Path("C:/data/volume.npy"))
        assert local_input_path(tmp_path) == tmp_path
        with pytest.raises(typer.BadParameter, match="does not exist"):
            local_input_path(tmp_path / "missing.npy")

    @pytest.mark.parametrize(
        "argv",
        [
            ["gsplat", "fit", "https://example.org/x.zarr", "out.gsplats.zarr"],
            ["gsplat", "cal", "https://example.org/x.zarr", "cal.json"],
            ["gsplat", "info", "https://example.org/x.gsplats.zarr"],
            ["gsplat", "compare", "FIT", "https://example.org/x.zarr"],
        ],
        ids=["fit", "cal", "info", "compare"],
    )
    def test_commands_refuse_urls(
        self, runner: CliRunner, flat_fit: Path, argv: list[str]
    ) -> None:
        argv = [str(flat_fit) if arg == "FIT" else arg for arg in argv]
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, result.output
        text = normalized_cli_output(result)
        assert "https://example.org/x" in text
        assert "remote stores are not supported" in text


class TestPipedHelpWidth:
    @staticmethod
    def _help_width(**env_overrides: str) -> int:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"COLUMNS", "TERMINAL_WIDTH"}
        }
        env.update(env_overrides)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")]
        )
        out = subprocess.run(
            [sys.executable, "-m", "luxar", "gsplat", "fit", "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        ).stdout
        return max(len(line) for line in out.splitlines())

    def test_piped_help_is_not_clipped_to_80_columns(self) -> None:
        assert 80 < self._help_width() <= 120

    def test_columns_is_honoured(self) -> None:
        assert self._help_width(COLUMNS="100") == 100

    def test_importing_the_cli_leaves_the_environment_alone(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"COLUMNS", "TERMINAL_WIDTH"}
        }
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")]
        )
        code = (
            "import os, sys, luxar.cli; "
            "print(os.environ.get('TERMINAL_WIDTH'), 'typer.rich_utils' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        ).stdout.strip()
        assert out.splitlines()[-1] == "None False"


class TestCalPdfPreflight:
    def test_missing_matplotlib_fails_before_the_sweep(
        self,
        runner: CliRunner,
        volume_npy: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import luxar.gsplats.calibration as calibration

        def _no_sweep(*args: object, **kwargs: object) -> None:
            raise AssertionError("the sweep must not start")

        monkeypatch.setattr(calibration, "calibrate", _no_sweep)
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(volume_npy),
                str(tmp_path / "cal.json"),
                "--k-grid",
                "10,20",
                "--pdf",
                str(tmp_path / "report.pdf"),
                "--keep-fits",
                str(tmp_path / "fits"),
            ],
        )
        assert result.exit_code == 1, result.output
        text = normalized_cli_output(result)
        assert "--pdf needs matplotlib" in text
        assert "pip install matplotlib" in text
        assert not (tmp_path / "cal.json").exists()


class TestMergeFlattensPartitions:
    def test_flatness_is_read_from_attrs(
        self, tiled_fit_output: tuple[Path, str], flat_fit: Path
    ) -> None:
        from luxar.cli.gsplat_ops.transforms.merge import _is_matrix_shaped_store

        assert _is_matrix_shaped_store(flat_fit)
        assert not _is_matrix_shaped_store(tiled_fit_output[0])

    def test_channel_colors_accepts_a_tiled_fit(
        self,
        runner: CliRunner,
        tiled_fit_output: tuple[Path, str],
        flat_fit: Path,
        tmp_path: Path,
    ) -> None:
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import is_matrix_shaped, total_splats

        tiled, _ = tiled_fit_output
        tiled_node, _ = load_gsplat_node(tiled)
        flat_node, _ = load_gsplat_node(flat_fit)
        assert not is_matrix_shaped(tiled_node)

        out = tmp_path / "multi.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "merge",
                str(tiled),
                str(flat_fit),
                "-o",
                str(out),
                "--channel-colors",
                "#ff0080,#00ff00",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "flattening it to one leaf first" in result.output

        merged, _ = load_gsplat_node(out)
        assert total_splats(merged) == total_splats(tiled_node) + total_splats(
            flat_node
        )
        # The staging store lives beside the output and is removed afterwards.
        assert not any(
            p.name.startswith(".luxar-merge-flatten-") for p in tmp_path.iterdir()
        )
