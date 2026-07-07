"""``luxar gsplat inspect`` command registration + compatibility wrappers."""

from __future__ import annotations

import time as time  # noqa: F401 - kept for test monkeypatch path stability
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from .inspect_commands import _ascii_histogram as _ascii_histogram_impl
from .inspect_commands import (
    _compute_splat_volumes as _compute_splat_volumes_impl,
)
from .inspect_commands import (
    _print_gsplat_tree_summary as _print_gsplat_tree_summary_impl,
)
from .inspect_commands import (
    _print_statistics_table as _print_statistics_table_impl,
)
from .inspect_commands import annotate_quality as _annotate_quality_cmd
from .inspect_commands import compare_quality as _compare_quality_cmd
from .inspect_commands import info_dataset as _info_dataset_cmd
from .inspect_commands import napari_viewer as _napari_viewer_cmd
from .inspect_commands import quick_view as _quick_view_cmd

if TYPE_CHECKING:
    import numpy as np


def _ascii_histogram(
    data: "np.ndarray", bins: int = 40, width: int = 60, title: str = "Distribution"
) -> str:
    """Back-compat wrapper around histogram formatting helper."""
    return _ascii_histogram_impl(data, bins=bins, width=width, title=title)


def _compute_splat_volumes(cholesky_factors: "np.ndarray", ndim: int) -> "np.ndarray":
    """Back-compat wrapper around splat-volume computation helper."""
    return _compute_splat_volumes_impl(cholesky_factors, ndim)


def _print_statistics_table(data: "np.ndarray", label: str) -> None:
    """Back-compat wrapper around statistics printing helper."""
    return _print_statistics_table_impl(data, label)


def info_dataset(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the info command function."""
    return _info_dataset_cmd(*args, **kwargs)


def napari_viewer(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the napari command function."""
    return _napari_viewer_cmd(*args, **kwargs)


def quick_view(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the view command function."""
    return _quick_view_cmd(*args, **kwargs)


def compare_quality(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the compare command function."""
    return _compare_quality_cmd(*args, **kwargs)


def _print_gsplat_tree_summary(path: Path) -> None:
    """Back-compat wrapper around tree-summary helper."""
    return _print_gsplat_tree_summary_impl(path)


def annotate_quality(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around annotate-quality command function."""
    return _annotate_quality_cmd(*args, **kwargs)


def register_inspect_commands(app: typer.Typer) -> None:
    """Register the inspect commands onto ``app_gsplat``."""
    app.command("info")(_info_dataset_cmd)
    app.command("view")(_quick_view_cmd)
    app.command("napari")(_napari_viewer_cmd)
    app.command("compare")(_compare_quality_cmd)
    app.command("annotate-quality")(_annotate_quality_cmd)
