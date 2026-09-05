"""The native compile gate must be able to fail, and for the right reasons.

Audit A15-03. `scripts/check_native_compiles.py` is the first check of any kind
over ~3,900 lines of shipped CUDA/Metal. A gate over code nobody was checking is
worth exactly what it catches, so these tests are the gate's own control arm:
each one breaks something and asserts it goes red.

Two of them exist because the first version of the gate was WRONG in that way:

- A listed source that no longer exists used to print a note and continue, so
  the run passed with a smaller denominator — a renamed file read as a checked
  one. `test_a_missing_source_fails_rather_than_shrinking_the_run`.
- The Metal `-Wno-sometimes-uninitialized` suppression is load-bearing in the
  other direction: without it the shader's CORRECT threadgroup-barrier pattern
  reds the gate, and the "fix" would be adding an initializer that MSL says is
  undefined behaviour. `test_the_metal_suppression_is_load_bearing`.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "check_native_compiles.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_native_compiles", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__].__dict__`, so a module absent from the table
    # makes the decorator raise on Python 3.14.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


def _run(source: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a mutated gate against the real packages tree, outside the repo."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        scripts = root / "scripts"
        scripts.mkdir()
        (root / "packages").symlink_to(
            GATE.parent.parent / "packages", target_is_directory=True
        )
        mutant = scripts / "check_native_compiles_mutant.py"
        mutant.write_text(source, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(mutant), *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )


@pytest.fixture(scope="module")
def gate_source() -> str:
    return GATE.read_text(encoding="utf-8")


def test_the_gate_passes_on_the_real_tree() -> None:
    """The baseline. Without this the mutation tests below prove nothing."""
    result = subprocess.run(
        [sys.executable, str(GATE), "--only", "cxx"],
        capture_output=True,
        text=True,
        check=False,
        cwd=GATE.parent.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "translation unit(s) compile-checked" in result.stdout


def test_it_actually_checked_something(gate) -> None:
    """A gate that compiled nothing must not be able to report success.

    The unit lists are the gate's inputs; an empty one would make every
    assertion vacuous, which is the failure mode of most "passing" gates.
    """
    assert gate.CXX_UNITS, "no C++ units listed"
    assert gate.METAL_UNITS, "no Metal units listed"
    assert gate.CUDA_UNITS, "no CUDA units listed"
    for unit in gate.CXX_UNITS + gate.METAL_UNITS + gate.CUDA_UNITS + gate.OBJCXX_UNITS:
        assert unit.path.exists(), f"{unit.label} -> {unit.path} does not exist"


def test_the_required_standard_is_read_from_torch_not_hardcoded(gate) -> None:
    """The defect was a hardcoded standard disagreeing with torch's real one.

    So the gate derives it. `torch/all.h` carries the `#error` guard, and the
    202002L in it is what makes this C++20 rather than a literal in our code.
    """
    torch_info = gate.torch_includes()
    if torch_info is None:
        pytest.skip("torch not importable")
    _, std = torch_info
    assert std in {"c++17", "c++20"}
    import torch

    guard = Path(torch.__file__).parent / "include/torch/csrc/api/include/torch/all.h"
    text = guard.read_text(encoding="utf-8", errors="replace")
    assert std == ("c++20" if "202002L" in text else "c++17")


def test_the_shipped_build_flags_agree_with_what_torch_demands(gate) -> None:
    """The bug itself, gated at its source rather than only at the compiler.

    Both native build paths append their `-std=` AFTER torch's, so the last one
    wins. If either pins a standard below torch's minimum, the extension cannot
    compile — the Metal one silently, at a user's first use.
    """
    torch_info = gate.torch_includes()
    if torch_info is None:
        pytest.skip("torch not importable")
    _, required = torch_info
    pinned = gate.shipped_cxx_std()
    assert int(pinned.removeprefix("c++")) >= int(required.removeprefix("c++")), (
        f"native builds pin {pinned} but torch requires at least {required}. "
        "torch appends its own -std FIRST, so the shipped pin wins."
    )
    assert gate.shipped_metal_std() == "metal3.0"


def test_main_compiles_at_the_shipped_standard(gate, monkeypatch) -> None:
    used: list[str] = []
    monkeypatch.setattr(gate, "torch_includes", lambda: ([], "c++17"))
    monkeypatch.setattr(gate, "shipped_cxx_std", lambda: "c++20")
    monkeypatch.setattr(gate, "shipped_metal_std", lambda: "metal3.0")
    monkeypatch.setattr(gate, "have", lambda tool: tool == "g++")
    monkeypatch.setattr(
        gate,
        "check_host_units",
        lambda units, std, includes: (used.append(std) or 1, 1),
    )
    monkeypatch.setattr(sys, "argv", [str(GATE), "--only", "cxx"])

    assert gate.main() == 0
    assert used == ["c++20"]


def test_metal_uses_the_shipped_sdk_and_language_standard(
    gate, monkeypatch, tmp_path
) -> None:
    source = tmp_path / "kernel.metal"
    source.write_text("kernel void noop() {}", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        gate, "run", lambda command, label: commands.append(command) or True
    )

    assert gate.check_metal([gate.Unit(source, "metal", "kernel")], "metal3.0") == (
        1,
        1,
    )
    assert commands == [
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-std=metal3.0",
            "-Werror",
            *gate.METAL_SUPPRESSED_WARNINGS,
            str(source),
            "-o",
            "/dev/null",
        ]
    ]


def test_a_missing_source_fails_rather_than_shrinking_the_run(gate_source) -> None:
    """Found by mutation: the first version PASSED when a source was renamed.

    It printed a note, skipped the unit, and reported `2/2`. An unchecked file
    must never read as a checked one.
    """
    mutant = gate_source.replace(
        "models/gsplats/cuda/src/bindings.cpp",
        "models/gsplats/cuda/src/RENAMED.cpp",
    )
    assert mutant != gate_source, "mutation did not apply"
    result = _run(mutant, "--only", "cxx")
    assert result.returncode == 1, (
        "a listed-but-absent source did not fail the gate:\n" + result.stdout
    )
    assert "RENAMED.cpp" in result.stdout
    assert "✓ nlm/bindings.cpp" in result.stdout


def test_the_metal_suppression_is_load_bearing(gate_source) -> None:
    """Dropping it reds the gate on CORRECT code — which is why it is narrow.

    `kernels.metal` writes `threadgroup uint s_total_voxels` from thread 0 and
    reads it everywhere after a `threadgroup_barrier`. The compiler cannot see
    the barrier and warns; MSL forbids initializing threadgroup storage, so the
    only "fix" at the source would be undefined behaviour. If this test starts
    failing, the shader changed and the suppression may no longer be needed —
    check before widening it.
    """
    if not shutil.which("xcrun"):
        pytest.skip("no xcrun")
    if not _load().have("metal"):
        pytest.skip("Metal toolchain not installed")
    mutant = gate_source.replace(
        'METAL_SUPPRESSED_WARNINGS = ["-Wno-sometimes-uninitialized"]',
        "METAL_SUPPRESSED_WARNINGS = []",
    )
    assert mutant != gate_source, "mutation did not apply"
    result = _run(mutant, "--only", "metal")
    assert result.returncode == 1, (
        "removing the suppression did not red the gate, so either the shader "
        "changed or -Werror is not in effect:\n" + result.stdout
    )


def test_require_turns_a_missing_toolchain_into_a_failure() -> None:
    """A silently-skipped arm must not be able to masquerade as a passing one.

    CI names the toolchains its runner is supposed to have; if one is absent the
    job fails rather than reporting green over nothing.
    """
    if shutil.which("nvcc"):
        pytest.skip("nvcc IS installed here, so this cannot be provoked")
    result = subprocess.run(
        [sys.executable, str(GATE), "--only", "nvcc", "--require", "nvcc"],
        capture_output=True,
        text=True,
        check=False,
        cwd=GATE.parent.parent,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout and "nvcc" in result.stdout


@pytest.mark.parametrize("required", ["cxx", "metal", "nvcc"])
def test_require_fails_when_torch_is_missing(gate, monkeypatch, required) -> None:
    monkeypatch.setattr(gate, "torch_includes", lambda: None)
    monkeypatch.setattr(sys, "argv", [str(GATE), "--require", required])
    assert gate.main() == 1


def test_require_also_selects_the_required_arm(gate, monkeypatch) -> None:
    monkeypatch.setattr(gate, "torch_includes", lambda: ([], "c++20"))
    monkeypatch.setattr(gate, "shipped_cxx_std", lambda: "c++20")
    monkeypatch.setattr(gate, "shipped_metal_std", lambda: "metal3.0")
    monkeypatch.setattr(gate, "have", lambda tool: tool == "g++")
    monkeypatch.setattr(gate, "check_host_units", lambda units, std, includes: (1, 1))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(GATE), "--only", "cxx", "--require", "nvcc"],
    )

    assert gate.main() == 1


def test_failed_commands_always_print_a_diagnostic(gate, monkeypatch, capsys) -> None:
    failure = subprocess.CompletedProcess(
        args=["nvcc"],
        returncode=1,
        stdout="",
        stderr="nvcc fatal   : Unknown option '-bad-flag'\n",
    )
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: failure)

    assert gate.run(["nvcc", "-bad-flag"], "cuda") is False
    assert "nvcc fatal" in capsys.readouterr().out
