#!/usr/bin/env python3
"""
Smoke tests for HPC environment setup (no sudo / no pipx).

Validates that the Makefile install-hatch and install-pnpm targets work correctly
on systems where global installs require root (e.g. HPC login nodes).

Run with:
    python scripts/check_hpc_setup.py

Or via hatch (after initial bootstrap):
    hatch run python scripts/check_hpc_setup.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

LOCAL_BIN = Path.home() / ".local" / "bin"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_python_310_plus() -> str | None:
    """Replicate the for-loop in install-hatch: find first Python >= 3.10."""
    candidates = ["python3.13", "python3.12", "python3.11", "python3.10", "python3"]
    for py in candidates:
        exe = shutil.which(py)
        if exe is None:
            continue
        result = run(
            [
                exe,
                "-c",
                "import sys; print(sys.version_info.major, sys.version_info.minor)",
            ]
        )
        if result.returncode != 0:
            continue
        parts = result.stdout.strip().split()
        if len(parts) == 2:
            major, minor = int(parts[0]), int(parts[1])
            if major == 3 and minor >= 10:
                return exe
    return None


def hatch_cmd() -> str | None:
    """Return path to hatch binary (PATH or ~/.local/bin)."""
    h = shutil.which("hatch")
    if h:
        return h
    local_hatch = LOCAL_BIN / "hatch"
    if local_hatch.is_file() and os.access(local_hatch, os.X_OK):
        return str(local_hatch)
    return None


def pnpm_cmd() -> str | None:
    """Return path to pnpm binary (PATH or ~/.local/bin)."""
    p = shutil.which("pnpm")
    if p:
        return p
    local_pnpm = LOCAL_BIN / "pnpm"
    if local_pnpm.is_file() and os.access(local_pnpm, os.X_OK):
        return str(local_pnpm)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_python_310_available():
    """A Python 3.10+ interpreter must be reachable (needed by install-hatch fallback)."""
    py = find_python_310_plus()
    assert py is not None, (
        "No Python 3.10+ found in PATH. "
        "install-hatch's pip/venv fallback will silently skip."
    )
    result = run([py, "--version"])
    print(f"PASS: Python 3.10+ found: {py} → {result.stdout.strip()}")


def test_hatch_installed_and_functional():
    """Hatch must be installed (PATH or ~/.local/bin) and return a version."""
    h = hatch_cmd()
    assert h is not None, (
        "Hatch not found in PATH or ~/.local/bin. "
        "Run 'make install-hatch' (ensure PATH includes ~/.local/bin)."
    )
    result = run([h, "--version"])
    assert result.returncode == 0, f"hatch --version failed:\n{result.stderr}"
    print(f"PASS: hatch found at {h} → {result.stdout.strip()}")


def test_hatch_can_list_envs():
    """Hatch must be able to list environments (checks pyproject.toml is parseable)."""
    h = hatch_cmd()
    if h is None:
        print("SKIP: hatch not installed")
        return
    result = run([h, "env", "show"], cwd=str(Path(__file__).parent.parent))
    assert result.returncode == 0, (
        f"hatch env show failed (pyproject.toml issue?):\n{result.stderr}"
    )
    print("PASS: hatch env show succeeded")


def test_hatch_uses_python_310_plus():
    """The hatch default env must use Python 3.10+ (project requires-python = >=3.10)."""
    h = hatch_cmd()
    if h is None:
        print("SKIP: hatch not installed")
        return
    result = run(
        [h, "run", "python", "-c", "import sys; print(sys.version_info[:2])"],
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, f"hatch run python failed:\n{result.stderr}"
    version_str = result.stdout.strip()
    # Output like "(3, 12)"
    import ast

    major, minor = ast.literal_eval(version_str)
    assert major == 3 and minor >= 10, (
        f"hatch env Python is {major}.{minor}, need >=3.10. "
        "hatch may be using the system Python 3.6."
    )
    print(f"PASS: hatch env uses Python {major}.{minor}")


def test_pnpm_installed_and_functional():
    """pnpm must be installed (PATH or ~/.local/bin) and return a version."""
    p = pnpm_cmd()
    assert p is not None, (
        "pnpm not found in PATH or ~/.local/bin. "
        "Run 'make install-pnpm' (ensure PATH includes ~/.local/bin)."
    )
    result = run([p, "--version"])
    assert result.returncode == 0, f"pnpm --version failed:\n{result.stderr}"
    print(f"PASS: pnpm found at {p} → {result.stdout.strip()}")


def test_local_bin_in_path():
    """~/.local/bin should be in PATH so make targets resolve hatch/pnpm without full path."""
    path_dirs = os.environ.get("PATH", "").split(":")
    local_bin_str = str(LOCAL_BIN)
    if local_bin_str not in path_dirs:
        print(
            f"WARN: ~/.local/bin ({local_bin_str}) is NOT in PATH. "
            "Add 'export PATH=\"$HOME/.local/bin:$PATH\"' to your ~/.bashrc"
        )
    else:
        print("PASS: ~/.local/bin is in PATH")


def test_npm_prefix_fallback_works():
    """
    Confirm that 'npm install --prefix ~/.local -g <pkg>' produces a binary in ~/.local/bin.
    This is the mechanism used by install-pnpm when global npm install fails (no sudo).
    We verify by checking pnpm is at ~/.local/bin/pnpm (already installed).
    """
    local_pnpm = LOCAL_BIN / "pnpm"
    if not local_pnpm.exists():
        print("SKIP: pnpm not installed at ~/.local/bin/pnpm (may be in system PATH)")
        return
    assert os.access(local_pnpm, os.X_OK), f"{local_pnpm} exists but is not executable"
    result = run([str(local_pnpm), "--version"])
    assert result.returncode == 0
    print(
        f"PASS: npm --prefix ~/.local fallback verified: {local_pnpm} → {result.stdout.strip()}"
    )


def test_hatch_env_venv_uses_correct_python():
    """
    When hatch was installed via venv (the HPC fallback), its internal env
    should still create project envs using a Python >= 3.10.
    Guards against hatch accidentally using the system Python 3.6.8.
    """
    h = hatch_cmd()
    if h is None:
        print("SKIP: hatch not installed")
        return
    result = run(
        [
            h,
            "run",
            "python",
            "-c",
            "import sys; v=sys.version_info; assert (v.major,v.minor)>=(3,10), 'Python '+str(v[:2])+' < 3.10'",
        ],
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, (
        f"hatch env Python version check failed:\n{result.stderr}\n{result.stdout}"
    )
    print("PASS: hatch venv env uses Python >= 3.10")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_python_310_available,
        test_hatch_installed_and_functional,
        test_hatch_can_list_envs,
        test_hatch_uses_python_310_plus,
        test_pnpm_installed_and_functional,
        test_local_bin_in_path,
        test_npm_prefix_fallback_works,
        test_hatch_env_venv_uses_correct_python,
    ]

    passed = failed = skipped = 0
    print(f"\n{'=' * 60}")
    print("HPC Setup Smoke Tests")
    print(f"{'=' * 60}\n")

    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}\n  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {name}\n  {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)
