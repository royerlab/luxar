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
import subprocess
import sys
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


def _run(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a mutated copy of the gate, from the repo root so its paths resolve."""
    mutant = tmp_path / "check_native_compiles.py"
    mutant.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(mutant)],
        capture_output=True,
        text=True,
        check=False,
        cwd=GATE.parent.parent,
    )


@pytest.fixture(scope="module")
def gate_source() -> str:
    return GATE.read_text(encoding="utf-8")


def test_the_gate_passes_on_the_real_tree() -> None:
    """The baseline. Without this the mutation tests below prove nothing."""
    result = subprocess.run(
        [sys.executable, str(GATE)],
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
    root = GATE.parent.parent / "packages/luxar/src/luxar/gsplats"
    for rel in ("models/gsplats/metal/setup.py", "_cuda_build.py"):
        text = (root / rel).read_text(encoding="utf-8")
        pinned = [
            line
            for line in text.splitlines()
            if "-std=c++" in line and "#" not in line.split("-std=")[0]
        ]
        assert pinned, f"{rel} pins no C++ standard — did the flag move?"
        for line in pinned:
            assert f"-std={required}" in line, (
                f"{rel} pins {line.strip()!r} but torch requires {required}. "
                f"torch appends its own -std FIRST, so this one wins and the "
                f"extension will not compile."
            )


def test_a_missing_source_fails_rather_than_shrinking_the_run(
    gate_source, tmp_path
) -> None:
    """Found by mutation: the first version PASSED when a source was renamed.

    It printed a note, skipped the unit, and reported `2/2`. An unchecked file
    must never read as a checked one.
    """
    mutant = gate_source.replace(
        "preprocessing/cuda/src/bindings.cpp",
        "preprocessing/cuda/src/RENAMED.cpp",
    )
    assert mutant != gate_source, "mutation did not apply"
    result = _run(mutant, tmp_path)
    assert result.returncode == 1, (
        "a listed-but-absent source did not fail the gate:\n" + result.stdout
    )
    assert "does not exist" in result.stdout


def test_the_metal_suppression_is_load_bearing(gate_source, tmp_path) -> None:
    """Dropping it reds the gate on CORRECT code — which is why it is narrow.

    `kernels.metal` writes `threadgroup uint s_total_voxels` from thread 0 and
    reads it everywhere after a `threadgroup_barrier`. The compiler cannot see
    the barrier and warns; MSL forbids initializing threadgroup storage, so the
    only "fix" at the source would be undefined behaviour. If this test starts
    failing, the shader changed and the suppression may no longer be needed —
    check before widening it.
    """
    import shutil

    if not (shutil.which("xcrun") and gate_source):
        pytest.skip("no xcrun")
    if not _load().have("metal"):
        pytest.skip("Metal toolchain not installed")
    mutant = gate_source.replace(
        'METAL_SUPPRESSED_WARNINGS = ["-Wno-sometimes-uninitialized"]',
        "METAL_SUPPRESSED_WARNINGS = []",
    )
    assert mutant != gate_source, "mutation did not apply"
    result = _run(mutant, tmp_path)
    assert result.returncode == 1, (
        "removing the suppression did not red the gate, so either the shader "
        "changed or -Werror is not in effect:\n" + result.stdout
    )


def test_require_turns_a_missing_toolchain_into_a_failure(
    gate_source, tmp_path
) -> None:
    """A silently-skipped arm must not be able to masquerade as a passing one.

    CI names the toolchains its runner is supposed to have; if one is absent the
    job fails rather than reporting green over nothing.
    """
    result = subprocess.run(
        [sys.executable, str(GATE), "--require", "nvcc"],
        capture_output=True,
        text=True,
        check=False,
        cwd=GATE.parent.parent,
    )
    import shutil

    if shutil.which("nvcc"):
        pytest.skip("nvcc IS installed here, so this cannot be provoked")
    assert result.returncode == 1
    assert "FAIL" in result.stdout and "nvcc" in result.stdout
