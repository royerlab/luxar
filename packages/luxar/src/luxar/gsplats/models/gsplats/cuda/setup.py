"""
Build script for CUDA splatting backend.

This script compiles the CUDA kernels into a PyTorch extension.

Usage:
    pip install -e .
    # or
    python setup.py develop

Environment variables:
    TORCH_CUDA_ARCH_LIST: Semicolon-separated list of architectures (e.g., "7.5;8.6")
    LUXAR_CUDA_ALL_ARCHS: Set to "1" to build for all modern architectures (slower)

Architecture options:
    - Default: Auto-detect from installed GPU(s)
    - LUXAR_CUDA_ALL_ARCHS=1: Build for all modern architectures (7.5 through 12.0)
      Good for distribution, but takes longer to compile
    - TORCH_CUDA_ARCH_LIST="7.5;8.6": Explicit list of architectures

Requirements:
    - CUDA 11.8+ (12.x recommended)
    - PyTorch 2.2+ with CUDA support
    - C++17 compatible compiler
"""

import os
import sys
from pathlib import Path

from setuptools import setup

# Check for CUDA availability before importing torch
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("PyTorch not found. Please install PyTorch with CUDA support.")
    sys.exit(1)

if not CUDA_AVAILABLE:
    print(
        "CUDA not available. Please install CUDA toolkit and PyTorch with CUDA support."
    )
    sys.exit(1)

# All modern CUDA architectures from Turing (7.5) to Blackwell (12.0)
# 7.5 = Turing (RTX 20 series, TITAN RTX, Quadro RTX)
# 8.0 = Ampere (A100, A30)
# 8.6 = Ampere (RTX 30 series, A40, A10)
# 8.9 = Ada Lovelace (RTX 40 series, L40)
# 9.0 = Hopper (H100, H200)
# 10.0 = Blackwell (B100, B200) - requires CUDA 12.8+
# 12.0 = Blackwell (future) - requires very recent CUDA
ALL_MODERN_ARCHS = "7.5;8.0;8.6;8.9;9.0;10.0;12.0"

# Blackwell arches (sm_100 / sm_120) only exist in CUDA Toolkit 12.8+. Targeting
# them with an older nvcc fails the whole build, so when we fall back to building
# "all modern" arches we drop them unless the toolkit is new enough.
_BLACKWELL_ARCHS = ("10.0", "12.0")


def _filter_archs_for_toolkit(arch_str: str) -> str:
    """Drop Blackwell arches when the active CUDA toolkit is < 12.8."""
    cuda_ver = getattr(torch.version, "cuda", None)  # e.g. "12.8"
    try:
        major, minor = (int(x) for x in cuda_ver.split(".")[:2])  # type: ignore[union-attr]
        supports_blackwell = (major, minor) >= (12, 8)
    except (AttributeError, ValueError):
        supports_blackwell = False
    archs = arch_str.replace(";", " ").split()
    if not supports_blackwell:
        dropped = [a for a in archs if a in _BLACKWELL_ARCHS]
        if dropped:
            print(
                f"⚠️  Toolkit CUDA {cuda_ver} < 12.8; dropping Blackwell arches "
                f"{dropped} (set TORCH_CUDA_ARCH_LIST to override)."
            )
        archs = [a for a in archs if a not in _BLACKWELL_ARCHS]
    return ";".join(archs)


# Get CUDA compute capabilities
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
build_all_archs = os.environ.get("LUXAR_CUDA_ALL_ARCHS", "0") == "1"

if cuda_arch_list is not None:
    # Explicit list provided by user
    print(f"Using TORCH_CUDA_ARCH_LIST from environment: {cuda_arch_list}")
elif build_all_archs:
    # Build for all modern architectures (good for distribution)
    cuda_arch_list = _filter_archs_for_toolkit(ALL_MODERN_ARCHS)
    print(f"Building for ALL modern architectures: {cuda_arch_list}")
    print("  (This may take a while, but ensures compatibility with most GPUs)")
else:
    # Auto-detect from all available GPUs
    if torch.cuda.is_available():
        detected_archs = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            detected_archs.add(f"{major}.{minor}")
        cuda_arch_list = ";".join(sorted(detected_archs))
        print(f"Auto-detected GPU architectures: {cuda_arch_list}")
    else:
        # Fallback to all modern architectures
        cuda_arch_list = _filter_archs_for_toolkit(ALL_MODERN_ARCHS)
        print(
            f"No GPU detected, building for all modern architectures: {cuda_arch_list}"
        )

print(f"Building for CUDA architectures: {cuda_arch_list}")

# Source directory
src_dir = Path(__file__).parent / "src"

# Source files
sources = [
    str(src_dir / "bindings.cpp"),
    str(src_dir / "cuda_splatting.cu"),
]

# Verify source files exist
for source in sources:
    if not Path(source).exists():
        raise FileNotFoundError(f"Source file not found: {source}")

# Compiler flags
extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
    ],
    "nvcc": [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-DNDEBUG",
        # Suppress specific warnings
        "-Xcudafe",
        "--diag_suppress=esa_on_defaulted_function_ignored",
    ],
}

# Add architecture flags
for arch in cuda_arch_list.replace(";", " ").split():
    arch_num = arch.replace(".", "")
    extra_compile_args["nvcc"].extend(
        [
            f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}",
        ]
    )

# Include directories
include_dirs = [
    str(src_dir),
]

# CUB is bundled with CUDA Toolkit 11+, no extra include needed

ext_modules = [
    CUDAExtension(  # type: ignore[no-untyped-call]
        name="cuda_splatting_backend",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        # Note: cuBLAS linkage removed - not currently used
    ),
]

setup(
    name="cuda_splatting_backend",
    version="0.1.0",
    description="CUDA backend for Gaussian splatting",
    author="Luxar Team",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
    ],
)
