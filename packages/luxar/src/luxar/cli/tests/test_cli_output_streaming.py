"""CLI output must stream line by line when stdout is redirected.

Python block-buffers a non-TTY stdout, so ``luxar … > log`` / ``nohup`` used to
show nothing until the process exited (a ``gsplat cal`` log stayed empty for
half an hour). ``main_callback`` reconfigures the standard streams to line
buffering; these pin that a real ``luxar`` process, its stdout a FILE, has
written its first lines while it is still running.
"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from luxar.cli.main import _line_buffer_std_streams


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_redirected_stdout_reaches_the_file_before_the_process_exits(
    tmp_path: Path,
) -> None:
    """`luxar serve` prints its banner, then blocks forever serving.

    Under block buffering the banner sits in the process's buffer until exit,
    so the file stays empty for as long as the server runs.
    """
    served = tmp_path / "served"
    served.mkdir()
    log = tmp_path / "stdout.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(  # nosec B603 - fixed argv, test-owned paths
            [
                sys.executable,
                "-m",
                "luxar",
                "serve",
                str(served),
                "--port",
                str(_free_port()),
            ],
            stdout=out,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 60.0
            text = ""
            while time.monotonic() < deadline:
                text = log.read_text(encoding="utf-8", errors="replace")
                if "Serving directory" in text:
                    break
                assert proc.poll() is None, f"server exited early:\n{text}"
                time.sleep(0.2)
            assert "Serving directory" in text, (
                f"no line reached the redirected stdout within 60 s:\n{text!r}"
            )
            # The point of the test: the line arrived while the process lives.
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait(timeout=30)


def test_the_helper_enables_line_buffering_on_real_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    err = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    assert not out.line_buffering
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    _line_buffer_std_streams()
    assert out.line_buffering and err.line_buffering


def test_the_helper_tolerates_streams_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typer's CliRunner (and pytest capture) swap in plain streams."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _line_buffer_std_streams()  # must not raise


# ── what `--quiet` / `-q` keeps and drops ─────────────────────────────────────


def _blob_volume(tmp_path: Path) -> Path:
    import numpy as np

    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[3:7, 3:7, 3:7] = 1.0
    path = tmp_path / "vol.npy"
    np.save(path, volume)
    return path


def test_quiet_fit_still_prints_the_backend_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one line saying which renderer runs survives `--quiet`."""
    from typer.testing import CliRunner

    from luxar.cli.main import app
    from luxar.gsplats.fitting import initialization

    monkeypatch.setattr(initialization, "_ANNOUNCED_BACKENDS", set())
    out = tmp_path / "fit.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(_blob_volume(tmp_path)),
            str(out),
            "--iters",
            "2",
            "--seeds",
            "4",
            "--tiling",
            "none",
            "--device",
            "cpu",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Fitting backend: GaussianSplatModel (PyTorch renderer" in result.stdout


def _compare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData

    ref = _blob_volume(tmp_path)
    rng = np.random.default_rng(0)
    n = 8
    data = GSplatData(
        centers=rng.uniform(3.0, 7.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 0.9, n).astype(np.float32),
        cholesky_factors=np.tile([1.2, 0, 1.2, 0, 0, 1.2], (n, 1)).astype(np.float32),
        stats={"image_min": 0.0, "floor": 0.0},
    )
    store = tmp_path / "fit.gsplats.zarr"
    data.save(store, include_fitting_info=True)
    return store, ref


def test_compare_quiet_prints_nothing_and_still_writes_json(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from luxar.cli.main import app

    store, ref = _compare_inputs(tmp_path)
    metrics = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app,
        ["gsplat", "compare", str(store), str(ref), "-j", str(metrics), "-q"],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert metrics.exists()


def test_compare_quiet_failure_still_reaches_stderr(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from luxar.cli.main import app

    store, ref = _compare_inputs(tmp_path)
    result = CliRunner().invoke(
        app,
        ["gsplat", "compare", str(store), str(ref), "--shape", "5,5,5", "-q"],
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "does not match reference shape" in result.stderr
