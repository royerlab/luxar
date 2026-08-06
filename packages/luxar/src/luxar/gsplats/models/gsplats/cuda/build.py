#!/usr/bin/env python3
"""
Build script for CUDA splatting extension.

Uses torch.utils.cpp_extension directly without setuptools complexity.
Run from project root: hatch run python packages/luxar/.../cuda/build.py
"""

from pathlib import Path

from luxar.gsplats._cuda_build import build_cuda_extension

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()


def build() -> None:
    """Build the CUDA extension in-place."""
    build_cuda_extension(
        extension_name="cuda_splatting_backend",
        source_names=["bindings.cpp", "cuda_splatting.cu"],
        script_dir=SCRIPT_DIR,
        info_filename="cuda_build_info.json",
        banner="Building for CUDA architectures",
    )


if __name__ == "__main__":
    build()
