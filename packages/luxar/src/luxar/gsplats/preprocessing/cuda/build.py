#!/usr/bin/env python3
"""
Build script for NLM CUDA extension.

Uses torch.utils.cpp_extension directly (same pattern as splatting backend).
Run from project root: hatch run python packages/luxar/.../preprocessing/cuda/build.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
SRC_DIR = SCRIPT_DIR / "src"


def _get_nvcc_supported_archs() -> set[int] | None:
    """Query nvcc --list-gpu-arch to get supported compute capabilities.

    Returns a set of integer arch codes (e.g. {75, 80, 86, ...}), or None
    if the query fails (older nvcc without --list-gpu-arch).
    """
    try:
        result = subprocess.run(
            ["nvcc", "--list-gpu-arch"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        archs = set()
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            # Lines look like "compute_86" or "sm_86"
            for prefix in ("compute_", "sm_"):
                if line.startswith(prefix):
                    try:
                        archs.add(int(line[len(prefix) :]))
                    except ValueError:
                        pass
        return archs if archs else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def build() -> None:
    """Build the NLM CUDA extension in-place."""
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError:
        print("ERROR: PyTorch not found. Install with:")
        print(
            "  hatch run pip install torch --index-url"
            " https://download.pytorch.org/whl/cu128"
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in PyTorch.")
        print("Install PyTorch with CUDA support:")
        print(
            "  hatch run pip install torch --index-url"
            " https://download.pytorch.org/whl/cu128"
        )
        sys.exit(1)

    # Check source files exist
    sources = [
        SRC_DIR / "bindings.cpp",
        SRC_DIR / "nlm_cuda.cu",
    ]
    for src in sources:
        if not src.exists():
            print(f"ERROR: Source file not found: {src}")
            sys.exit(1)

    # Determine CUDA architectures (same logic as splatting build.py)
    archs_env = os.environ.get("CUDA_ARCHS", "")
    if archs_env:
        target_archs = [int(a.strip()) for a in archs_env.split(";") if a.strip()]
    else:
        major, minor = torch.cuda.get_device_capability()
        current = major * 10 + minor
        # Turing(75), Ampere(80,86), Ada(89), Hopper(90), Blackwell(100,120)
        # nvcc --list-gpu-arch filtering below drops archs the current toolkit
        # can't target (e.g. sm_100/sm_120 require CUDA 12.8+).
        default_archs = {75, 80, 86, 89, 90, 100, 120}
        default_archs.add(current)

        # Query nvcc for supported architectures and filter out unsupported ones
        supported = _get_nvcc_supported_archs()
        if supported:
            default_archs = {a for a in default_archs if a in supported}
            # Always keep the current GPU's arch — if nvcc can't target it the
            # build will fail with a clear compiler error instead of an empty list.
            default_archs.add(current)

        target_archs = sorted(a for a in default_archs if a >= 75)
        if not target_archs:
            print("ERROR: No supported CUDA architectures found.")
            print(f"  Detected GPU compute capability: {current}")
            print("  Minimum supported: sm_75 (Turing)")
            print("  Override with: CUDA_ARCHS='75' make build-cuda")
            sys.exit(1)

    gencode_flags = []
    for arch in target_archs:
        gencode_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    highest = target_archs[-1]
    gencode_flags.append(f"-gencode=arch=compute_{highest},code=compute_{highest}")

    arch_str = ", ".join(f"sm_{a}" for a in target_archs)
    print(f"Building NLM CUDA extension for: {arch_str} (+ PTX for sm_{highest})")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        *gencode_flags,
        "-Xcudafe",
        "--diag_suppress=esa_on_defaulted_function_ignored",
    ]

    print("Compiling NLM CUDA extension...")
    print()

    build_dir = SCRIPT_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    # cpp_extension.load JIT-compiles our own CUDA source here; it is not
    # torch.load deserialization of untrusted data (bandit B614 false positive).
    load(  # nosec B614
        name="nlm_cuda_backend",
        sources=[str(s) for s in sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(SRC_DIR)],
        verbose=True,
        build_directory=str(build_dir),
    )

    # Copy .so to the cuda/ directory
    so_files = list(build_dir.glob("nlm_cuda_backend*.so"))
    if so_files:
        import shutil

        src_so = so_files[0]
        dest_so = SCRIPT_DIR / src_so.name
        if dest_so.exists():
            dest_so.unlink()
        shutil.copy2(src_so, dest_so)
        print()
        print(f"SUCCESS: Built {dest_so.name}")
        print(f"  Location: {dest_so}")
        _write_build_info(dest_so)
    else:
        print()
        print("ERROR: Build completed but .so file not found")
        sys.exit(1)


def _write_build_info(so_path: Path) -> None:
    """Write a JSON metadata file alongside the .so."""

    def _loaded_modules() -> list[str]:
        try:
            result = subprocess.run(
                ["bash", "-c", "module list 2>&1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.strip()
            modules = []
            for line in output.split("\n"):
                line = line.strip()
                if not line or line.startswith("Currently") or line.startswith("No "):
                    continue
                for prefix_end in (") ", ". "):
                    idx = line.find(prefix_end)
                    if idx != -1 and idx < 5:
                        line = line[idx + len(prefix_end) :].strip()
                        break
                for part in line.split():
                    if part and "/" in part:
                        modules.append(part)
            return modules
        except Exception:
            return []

    import torch

    info: dict = {
        "so_file": so_path.name,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        ),
        "loaded_modules": _loaded_modules(),
        "cxx_compiler": os.environ.get("CXX", ""),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
    }

    info_path = so_path.parent / "nlm_build_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Build info: {info_path}")


if __name__ == "__main__":
    build()
