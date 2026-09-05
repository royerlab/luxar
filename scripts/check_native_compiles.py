#!/usr/bin/env python3
"""Compile-check the shipped CUDA and Metal sources. No GPU required.

Audit A15-03. ~3,900 lines of native code ride into the wheel — a second and
third independent implementation of the same splat-rasterization maths as the
torch reference — and until this script nothing compile-checked them without a
GPU. That is the same 1:1-sync hazard the project already gates for the Rust/TS
pair, with none of the protection, and a silent divergence there produces
slightly wrong splats rather than a crash.

**It was already broken.** Both native build paths pinned ``-std=c++17`` in the
flags torch appends LAST, so the last ``-std=`` won and ``torch/all.h``'s
``#error C++20 or later compatible compiler is required`` fired: the Metal
extension — which compiles itself on FIRST USE — could not build at all against
the pinned torch, and neither could the CUDA bindings. Nobody noticed because
the failure lands on a user at runtime, not in a build.

What this checks, and what it deliberately does not:

- **Syntax/semantics only.** ``-fsyntax-only`` for the C++/ObjC++ translation
  units, ``metal -c`` for the shader. No linking, no GPU, no device code
  execution, so it runs on any CI runner with the relevant toolchain.
- **NOT numeric parity** with the torch reference. That is the remaining half of
  A15-03 and needs a GPU; see the module docstring note in
  ``gsplats/models/gsplats/cuda/README.md``. This script makes divergence
  *compile*-visible, not *numerically* visible.

Every toolchain is optional and reported as SKIP when absent — a Linux runner
has no ``metal``, a macOS runner has no ``nvcc`` — but a skip is never silent
and ``--require`` turns a named toolchain's absence into a failure, which is how
CI pins that the arm it thinks it is running actually ran.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "packages" / "luxar" / "src" / "luxar" / "gsplats"

#: The C++ standard torch requires. Read from torch when importable so this
#: cannot drift from the real requirement; the literal is only the fallback.
DEFAULT_CXX_STD = "c++20"

#: `kernels.metal` writes `threadgroup uint s_total_voxels` from thread 0 and
#: reads it in every thread AFTER a `threadgroup_barrier`, which is correct and
#: which the compiler cannot see: it reports `-Wsometimes-uninitialized` twice.
#: The warning cannot be silenced at the source either — MSL forbids
#: threadgroup-storage initializers, so adding `= 0` would trade a false
#: positive for real undefined behaviour. Suppressed HERE, narrowly, so that
#: `-Werror` still holds for every other warning the shader might grow.
METAL_SUPPRESSED_WARNINGS = ["-Wno-sometimes-uninitialized"]

CXX_BUILD_FILES = [
    SRC / "models/gsplats/metal/setup.py",
    SRC / "_cuda_build.py",
]
METAL_BUILD_FILE = SRC / "models/gsplats/metal/setup.py"


@dataclass
class Unit:
    """One translation unit and how to check it."""

    path: Path
    toolchain: str
    label: str
    #: True when the unit pulls in CUDA runtime headers, so the host compiler
    #: needs the toolkit's include dir even though it is a plain `.cpp`.
    #: `models/gsplats` does (via `cuda_splatting.h` -> `cuda_runtime.h`);
    #: `preprocessing` does not, which is why only one of the two can be
    #: checked on a machine with no CUDA installed.
    needs_cuda: bool = False


CXX_UNITS = [
    Unit(
        SRC / "models/gsplats/cuda/src/bindings.cpp",
        "cxx",
        "cuda/bindings.cpp",
        needs_cuda=True,
    ),
    Unit(SRC / "preprocessing/cuda/src/bindings.cpp", "cxx", "nlm/bindings.cpp"),
]
OBJCXX_UNITS = [
    Unit(SRC / "models/gsplats/metal/src/bindings.mm", "objcxx", "metal/bindings.mm"),
]
METAL_UNITS = [
    Unit(
        SRC / "models/gsplats/metal/src/kernels.metal", "metal", "metal/kernels.metal"
    ),
]
CUDA_UNITS = [
    Unit(
        SRC / "models/gsplats/cuda/src/cuda_splatting.cu", "nvcc", "cuda_splatting.cu"
    ),
    Unit(SRC / "preprocessing/cuda/src/nlm_cuda.cu", "nvcc", "nlm_cuda.cu"),
]


def torch_includes() -> tuple[list[str], str] | None:
    """`(-I flags, required std)` for torch's headers, or None if unimportable."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    root = Path(torch.__file__).parent
    flags = [
        f"-I{root / 'include'}",
        f"-I{root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{sysconfig.get_paths()['include']}",
    ]
    return flags, _required_std(root)


def _required_std(torch_root: Path) -> str:
    """Read torch's own minimum standard out of its headers.

    Derived rather than restated: the whole defect this script exists for was a
    hardcoded standard that disagreed with what torch actually demanded.
    """
    guard = (
        torch_root
        / "include"
        / "torch"
        / "csrc"
        / "api"
        / "include"
        / "torch"
        / "all.h"
    )
    try:
        text = guard.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return DEFAULT_CXX_STD
    if "202002L" in text:
        return "c++20"
    if "201703L" in text:
        return "c++17"
    return DEFAULT_CXX_STD


def _pinned_standards(path: Path, pattern: str) -> set[str]:
    """Return compiler standards pinned as string literals in a build file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        match.group(1)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        if (match := re.fullmatch(pattern, node.value))
    }


def shipped_cxx_std() -> str:
    """Return the single C++ standard used by both shipped extension builds."""
    standards: set[str] = set()
    for path in CXX_BUILD_FILES:
        pinned = _pinned_standards(path, r"-std=(c\+\+\d+)")
        if len(pinned) != 1:
            raise ValueError(
                f"{path} must pin exactly one C++ standard, found {pinned}"
            )
        standards.update(pinned)
    if len(standards) != 1:
        raise ValueError(f"native build C++ standards disagree: {sorted(standards)}")
    return standards.pop()


def shipped_metal_std() -> str:
    """Return the Metal language standard used by the shipped shader build."""
    standards = _pinned_standards(METAL_BUILD_FILE, r"-std=(metal\d+\.\d+)")
    if len(standards) != 1:
        raise ValueError(
            f"{METAL_BUILD_FILE} must pin exactly one Metal standard, found {standards}"
        )
    return standards.pop()


def have(tool: str) -> bool:
    if tool == "metal":
        return (
            shutil.which("xcrun") is not None
            and subprocess.run(  # noqa: S603
                ["xcrun", "--find", "metal"],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
    return shutil.which(tool) is not None


def run(cmd: list[str], label: str) -> bool:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode == 0:
        print(f"  ✓ {label}")
        return True
    print(f"  ✗ {label}")
    output = (proc.stdout + proc.stderr).splitlines()
    diagnostics = [
        line for line in output if "error" in line.lower() or "warning" in line.lower()
    ]
    for line in (
        diagnostics
        or output[-20:]
        or [f"compiler exited with status {proc.returncode}"]
    ):
        print(f"      {line[:200]}")
    return False


def cuda_include() -> list[str]:
    """`-I` for the CUDA toolkit's headers, when it is installed."""
    for root in (Path("/usr/local/cuda"), Path("/opt/cuda")):
        if (root / "include" / "cuda_runtime.h").exists():
            return [f"-I{root / 'include'}"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidate = Path(nvcc).resolve().parent.parent / "include"
        if (candidate / "cuda_runtime.h").exists():
            return [f"-I{candidate}"]
    return []


def check_host_units(
    units: list[Unit], std: str, includes: list[str]
) -> tuple[int, int]:
    """`-fsyntax-only` the C++ / ObjC++ units through the host compiler."""
    compiler = "clang++" if shutil.which("clang++") else "g++"
    cuda_inc = cuda_include()
    ok = 0
    checked = 0
    for unit in units:
        if not unit.path.exists():
            # FAIL CLOSED. Printing a note and moving on made the gate pass
            # with a smaller denominator when a source was renamed — an
            # unchecked file read as a checked one. Caught by mutating this
            # script's own path list, which is why that mutation is in
            # `scripts/tests/test_check_native_compiles.py`.
            print(f"  ✗ {unit.label} — listed source does not exist: {unit.path}")
            checked += 1
            continue
        if unit.needs_cuda and not cuda_inc:
            # Not a pass and not a failure: this unit includes cuda_runtime.h,
            # so it is only checkable where the toolkit is. Counted in neither
            # numerator nor denominator, and said out loud.
            print(f"  - {unit.label} skipped (needs the CUDA toolkit's headers)")
            continue
        checked += 1
        cmd = [compiler, "-fsyntax-only", f"-std={std}", *includes]
        if unit.needs_cuda:
            cmd += cuda_inc
        if unit.toolchain == "objcxx":
            cmd += ["-ObjC++", "-fno-objc-arc", "-Wno-deprecated-declarations"]
        cmd.append(str(unit.path))
        ok += run(cmd, unit.label)
    return ok, checked


def check_metal(units: list[Unit], std: str) -> tuple[int, int]:
    ok = 0
    checked = 0
    for unit in units:
        if not unit.path.exists():
            print(f"  ✗ {unit.label} — listed source does not exist: {unit.path}")
            checked += 1
            continue
        checked += 1
        cmd = [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            f"-std={std}",
            "-Werror",
            *METAL_SUPPRESSED_WARNINGS,
            str(unit.path),
            "-o",
            "/dev/null",
        ]
        ok += run(cmd, unit.label)
    return ok, checked


def check_cuda(units: list[Unit], std: str, includes: list[str]) -> tuple[int, int]:
    ok = 0
    checked = 0
    for unit in units:
        if not unit.path.exists():
            print(f"  ✗ {unit.label} — listed source does not exist: {unit.path}")
            checked += 1
            continue
        checked += 1
        # `-fsyntax-only` is not an nvcc flag; `-cuda` stops after the CUDA
        # front end, which is the closest no-GPU, no-link equivalent.
        cmd = [
            "nvcc",
            f"-std={std}",
            "-cuda",
            "-o",
            "/dev/null",
            *includes,
            str(unit.path),
        ]
        ok += run(cmd, unit.label)
    return ok, checked


def check_arm(
    selected: set[str],
    name: str,
    available: bool,
    heading: str,
    checker: Callable[[], tuple[int, int]],
) -> tuple[int, int, str | None]:
    """Run one selected toolchain arm, or report it as missing."""
    if name not in selected:
        return 0, 0, None
    if not available:
        return 0, 0, name
    print(heading)
    passed, total = checker()
    return passed, total, None


def host_units() -> list[Unit]:
    """Host units available on this platform."""
    return [*CXX_UNITS, *(OBJCXX_UNITS if sys.platform == "darwin" else [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=["cxx", "metal", "nvcc"],
        help="Fail instead of skipping when this toolchain is missing. CI passes "
        "the ones its runner is supposed to have, so a silently-skipped arm "
        "cannot masquerade as a passing one.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=["cxx", "metal", "nvcc"],
        help="Run only the selected toolchain arm. May be repeated.",
    )
    args = parser.parse_args()
    selected = set(args.only or ("cxx", "metal", "nvcc")) | set(args.require)

    torch_info = torch_includes()
    if torch_info is None:
        print("torch is not importable — native compile checks need its headers.")
        if args.require:
            return 1
        print("SKIP: nothing to check.")
        return 0
    includes, required_std = torch_info
    try:
        cxx_std = shipped_cxx_std()
        metal_std = shipped_metal_std()
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"FAIL: cannot determine shipped compiler standards: {exc}")
        return 1
    print(
        f"torch requires {required_std}; shipped builds use {cxx_std} "
        "(read from their sources)\n"
    )

    results = [
        check_arm(
            selected,
            "cxx",
            have("clang++") or have("g++"),
            "Host C++ / ObjC++ (-fsyntax-only):",
            lambda: check_host_units(host_units(), cxx_std, includes),
        ),
        check_arm(
            selected,
            "metal",
            have("metal"),
            "\nMetal shader (metal -c -Werror):",
            lambda: check_metal(METAL_UNITS, metal_std),
        ),
        check_arm(
            selected,
            "nvcc",
            have("nvcc"),
            "\nCUDA device code (nvcc -cuda):",
            lambda: check_cuda(CUDA_UNITS, cxx_std, includes),
        ),
    ]
    passed = sum(result[0] for result in results)
    total = sum(result[1] for result in results)
    missing = [result[2] for result in results if result[2] is not None]

    print()
    for tool in missing:
        required = tool in args.require
        print(f"{'FAIL' if required else 'SKIP'}: {tool} toolchain not found")
    if any(tool in args.require for tool in missing):
        return 1

    # Fail closed: a run that checked nothing must not report success. Every
    # toolchain being absent is a SKIP above, but reaching here with a
    # toolchain present and zero units checked means the source list is stale.
    if total == 0 and len(missing) < len(selected):
        print("FAIL: a toolchain was available but no unit was checked.")
        return 1

    print(f"{passed}/{total} translation unit(s) compile-checked")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
