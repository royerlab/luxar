"""Compute-device helpers shared by Luxar demos."""

from __future__ import annotations

from arbol import aprint


def detect_device(verbose: bool = True) -> str:
    """Auto-detect the best available compute device (cuda > mps > cpu).

    Args:
        verbose: If True, print detected device via arbol.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    from luxar.gsplats.utils.device import resolve_torch_device

    device = str(resolve_torch_device())
    if verbose:
        aprint(
            {
                "cuda": "Using CUDA device",
                "mps": "Using MPS device (Metal acceleration)",
                "cpu": "Using CPU device",
            }.get(device, f"Using device: {device}")
        )
    return device


def warn_if_no_cuda_gpu() -> None:
    """Print a warning if no CUDA GPU is available.

    GSplat demos require significant GPU compute for fitting.  Running on
    CPU is orders of magnitude slower and generally impractical for
    production runs.  This function prints a prominent warning so users
    understand the hardware requirements before waiting hours for a CPU run.
    """
    try:
        import torch  # noqa: F401  # check PyTorch is importable

        from luxar.gsplats.utils.device import is_mps_available

        if torch.cuda.is_available():
            return  # All good
        device = "MPS" if is_mps_available() else "CPU"
    except ImportError:
        device = "CPU (PyTorch not installed)"

    aprint("")
    aprint("=" * 70)
    aprint("WARNING: No CUDA GPU detected — running on " + device)
    aprint("=" * 70)
    aprint("GSplat demos require a CUDA GPU for practical performance.")
    aprint("Without one, fitting can take hours instead of minutes.")
    if device.startswith("MPS"):
        aprint("MPS (Apple Metal) provides some acceleration but is much")
        aprint("slower than CUDA for Gaussian splatting workloads.")
    aprint("")
    aprint("Options:")
    aprint("  - Use a machine with an NVIDIA GPU (CUDA)")
    aprint("  - Run the default path (shipped precomputed data) instead of --recompute")
    aprint("  - Use --serve-only if a scene was already generated")
    aprint("=" * 70)
    aprint("")
