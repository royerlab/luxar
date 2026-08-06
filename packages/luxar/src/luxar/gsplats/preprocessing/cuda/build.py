#!/usr/bin/env python3
"""
Build script for NLM CUDA extension.

Uses torch.utils.cpp_extension directly (same pattern as splatting backend).
Run from project root: hatch run python packages/luxar/.../preprocessing/cuda/build.py
"""

from pathlib import Path

from luxar.gsplats._cuda_build import build_cuda_extension

SCRIPT_DIR = Path(__file__).parent.absolute()


def build() -> None:
    """Build the NLM CUDA extension in-place."""
    build_cuda_extension(
        extension_name="nlm_cuda_backend",
        source_names=["bindings.cpp", "nlm_cuda.cu"],
        script_dir=SCRIPT_DIR,
        info_filename="nlm_build_info.json",
        banner="Building NLM CUDA extension for",
    )


if __name__ == "__main__":
    build()
