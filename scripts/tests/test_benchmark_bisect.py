"""Regression tests for the CUDA benchmark bisection launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmarks/benchmark_bisect.sh"


def _command_path(tmp_path: Path) -> Path:
    command_path = tmp_path / "bin"
    command_path.mkdir()
    for command in ("dirname", "mkdir"):
        target = shutil.which(command)
        assert target is not None
        (command_path / command).symlink_to(target)
    return command_path


def _run_script(
    cwd: Path, command_path: Path, bench_python: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if bench_python is None:
        env.pop("LUXAR_BENCH_PYTHON", None)
    else:
        env["LUXAR_BENCH_PYTHON"] = bench_python
    env["PATH"] = str(command_path)
    return subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_hatch_reports_setup_hint(tmp_path: Path) -> None:
    result = _run_script(tmp_path, _command_path(tmp_path))

    assert result.returncode == 1
    assert "Hatch Python not found at /nonexistent/bin/python" in result.stdout
    assert "Run 'hatch env create' or set LUXAR_BENCH_PYTHON" in result.stdout


def test_hatch_environment_is_resolved_from_repo_root(tmp_path: Path) -> None:
    command_path = _command_path(tmp_path)
    hatch_cwd = tmp_path / "hatch-cwd"
    missing_env = tmp_path / "missing-env"
    hatch = command_path / "hatch"
    hatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s' \"$PWD\" > {hatch_cwd!s}\n"
        f"printf '%s\\n' {missing_env!s}\n"
    )
    hatch.chmod(0o755)

    result = _run_script(tmp_path, command_path)

    assert result.returncode == 1
    assert hatch_cwd.read_text() == str(REPO_ROOT)
    assert f"Hatch Python not found at {missing_env}/bin/python" in result.stdout


def test_explicit_python_override_skips_hatch_resolution(tmp_path: Path) -> None:
    command_path = _command_path(tmp_path)
    hatch_called = tmp_path / "hatch-called"
    hatch = command_path / "hatch"
    hatch.write_text(f"#!/bin/bash\nprintf called > {hatch_called!s}\n")
    hatch.chmod(0o755)
    bench_python = tmp_path / "bench-python"
    bench_python.write_text("#!/bin/bash\nexit 1\n")
    bench_python.chmod(0o755)

    result = _run_script(tmp_path, command_path, str(bench_python))

    assert result.returncode == 1
    assert f"Python: {bench_python}" in result.stdout
    assert "PyTorch CUDA not available" in result.stdout
    assert not hatch_called.exists()


def test_python_override_supports_spaces_in_path(tmp_path: Path) -> None:
    command_path = _command_path(tmp_path)
    invoked = tmp_path / "python-invoked"
    bench_python = tmp_path / "Application Support" / "hatch" / "bin" / "python"
    bench_python.parent.mkdir(parents=True)
    bench_python.write_text(f"#!/bin/bash\nprintf invoked > {invoked!s}\nexit 1\n")
    bench_python.chmod(0o755)

    result = _run_script(tmp_path, command_path, str(bench_python))

    assert result.returncode == 1
    assert f"Python: {bench_python}" in result.stdout
    assert "PyTorch CUDA not available" in result.stdout
    assert invoked.read_text() == "invoked"
