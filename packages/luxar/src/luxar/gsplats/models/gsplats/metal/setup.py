"""
Build script for Metal-accelerated Gaussian splatting extension.

This script compiles Metal shaders and builds the C++ extension for Apple Silicon.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import setuptools
from arbol import aprint
from torch.utils.cpp_extension import BuildExtension, CppExtension


def compile_metal_shaders() -> None:
    """Compile Metal shaders to .metallib format."""
    src_dir = Path(__file__).parent / "src"
    metal_file = src_dir / "kernels.metal"
    air_file = src_dir / "kernels.air"
    lib_file = src_dir / "default.metallib"

    if not metal_file.exists():
        raise FileNotFoundError(f"Metal source not found: {metal_file}")

    # Skip if already compiled and source unchanged
    if lib_file.exists():
        if lib_file.stat().st_mtime > metal_file.stat().st_mtime:
            aprint("Metal library up to date, skipping compilation")
            return

    aprint(f"Compiling Metal shaders: {metal_file}")

    # Compile .metal -> .air (intermediate representation)
    subprocess.check_call(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(metal_file),
            "-o",
            str(air_file),
            "-std=metal3.0",  # Use Metal 3.0 for SIMD intrinsics
            "-O2",  # Optimization level
            "-ffast-math",  # Match the kernels' fast math assumptions
        ]
    )

    # Link .air -> .metallib
    subprocess.check_call(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            str(air_file),
            "-o",
            str(lib_file),
        ]
    )

    # Clean up intermediate file
    air_file.unlink()
    aprint(f"Metal library created: {lib_file}")


class CustomBuildExtension(BuildExtension):
    """Build extension with Metal shader compilation."""

    def run(self) -> None:
        # Check platform
        if sys.platform != "darwin":
            aprint("WARNING: Metal extension only supported on macOS")
            return

        # Compile Metal shaders first
        compile_metal_shaders()

        # Then build C++ extension
        super().run()


# C++ extension configuration
ext_modules = [
    CppExtension(  # type: ignore[no-untyped-call]
        name="metal_splatting_backend",
        sources=["src/bindings.mm"],
        extra_compile_args={
            "cxx": [
                "-std=c++17",
                "-fno-objc-arc",  # Manual memory management for Metal objects
                "-Wno-deprecated-declarations",
            ],
        },
        extra_link_args=[
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
        ],
    ),
]


setuptools.setup(
    name="metal_splatting",
    version="1.0.0",
    description="Metal-accelerated Gaussian splatting for Luxar",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExtension},
    python_requires=">=3.10",
)
