"""
Build script for CUDA splatting backend.

This script compiles the CUDA kernels into a PyTorch extension.

Usage:
    pip install -e .
    # or
    python setup.py develop

Requirements:
    - CUDA 11.8+ (12.x recommended)
    - PyTorch 2.0+ with CUDA support
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

# Get CUDA compute capabilities
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if cuda_arch_list is None:
    # Auto-detect from current GPU
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        cuda_arch_list = f"{major}.{minor}"
    else:
        # Default to common architectures
        cuda_arch_list = "7.0;7.5;8.0;8.6;8.9;9.0"

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
    CUDAExtension(
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
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
)
