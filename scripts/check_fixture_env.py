"""Verify the viewer-fixture Hatch environment stays lean and CPU-only."""

from __future__ import annotations

import importlib.util
import sys

import torch

UNWANTED_PACKAGES = ("napari", "PyQt6", "ruff", "mypy")


def main() -> None:
    """Fail with actionable diagnostics when the fixture environment drifts."""
    if sys.platform == "linux" and not torch.__version__.endswith("+cpu"):
        raise RuntimeError(
            f"Expected the Linux CPU-only torch wheel, got {torch.__version__}. "
            "The PyTorch extra index is a preference, so check whether PyPI now "
            "offers a newer torch release than download.pytorch.org/whl/cpu."
        )
    if torch.version.cuda is not None:
        raise RuntimeError(
            f"Expected torch without CUDA support, got CUDA {torch.version.cuda}."
        )

    installed = [
        name for name in UNWANTED_PACKAGES if importlib.util.find_spec(name) is not None
    ]
    if installed:
        raise RuntimeError(
            "Default-only packages leaked into the fixture environment: "
            + ", ".join(installed)
        )


if __name__ == "__main__":
    main()
