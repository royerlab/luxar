#!/usr/bin/env python3
"""
Build script for CUDA splatting extension.

Uses torch.utils.cpp_extension directly without setuptools complexity.
Run from project root: hatch run python packages/luxar/.../cuda/build.py
"""

import sys
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()
SRC_DIR = SCRIPT_DIR / "src"


def build():
    """Build the CUDA extension in-place."""
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError:
        print("ERROR: PyTorch not found. Install with:")
        print(
            "  hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in PyTorch.")
        print("Install PyTorch with CUDA support:")
        print(
            "  hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        sys.exit(1)

    # Check source files exist
    sources = [
        SRC_DIR / "bindings.cpp",
        SRC_DIR / "cuda_splatting.cu",
    ]
    for src in sources:
        if not src.exists():
            print(f"ERROR: Source file not found: {src}")
            sys.exit(1)

    # Get CUDA compute capability
    major, minor = torch.cuda.get_device_capability()
    cuda_arch = f"{major}.{minor}"
    print(f"Building for CUDA architecture: {cuda_arch}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    # Compiler flags
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
        "-Xcudafe",
        "--diag_suppress=esa_on_defaulted_function_ignored",
    ]

    print("Compiling CUDA extension (this may take a minute)...")
    print()

    # Ensure build directory exists (PyTorch's cpp_extension needs it for lock file)
    build_dir = SCRIPT_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    # Build using torch's JIT compilation
    # This compiles and loads the module, placing .so in a cache dir
    # We'll then copy it to the cuda/ directory
    load(
        name="cuda_splatting_backend",
        sources=[str(s) for s in sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(SRC_DIR)],
        verbose=True,
        build_directory=str(build_dir),
    )

    # Find the built .so file and copy/link to cuda/ directory

    so_files = list(build_dir.glob("cuda_splatting_backend*.so"))

    if so_files:
        src_so = so_files[0]
        # Create symlink or copy to cuda/ directory
        dest_so = SCRIPT_DIR / src_so.name
        if dest_so.exists():
            dest_so.unlink()

        import shutil

        shutil.copy2(src_so, dest_so)
        print()
        print(f"SUCCESS: Built {dest_so.name}")
        print(f"  Location: {dest_so}")
    else:
        print()
        print("ERROR: Build completed but .so file not found")
        sys.exit(1)


if __name__ == "__main__":
    build()
