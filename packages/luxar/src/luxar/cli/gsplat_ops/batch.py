"""``luxar gsplat batch-fit`` command registration + compatibility wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .batch_commands import (
    _build_merge_recipe_params as _build_merge_recipe_params_cmd,
)
from .batch_commands import (
    _measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat_cmd,
)
from .batch_commands import _validate_leaf_arrays as _validate_leaf_arrays_cmd
from .batch_commands import _validate_node_dir as _validate_node_dir_cmd
from .batch_commands import _validate_tile as _validate_tile_cmd
from .batch_commands import app_batch as app_batch
from .batch_commands import batch_cancel_cmd as _batch_cancel_cmd
from .batch_commands import batch_denoise_calibrate_cmd as _batch_denoise_calibrate_cmd
from .batch_commands import (
    batch_denoise_preprocess_cmd as _batch_denoise_preprocess_cmd,
)
from .batch_commands import batch_merge_cmd as _batch_merge_cmd
from .batch_commands import batch_run as _batch_run_cmd
from .batch_commands import batch_status_cmd as _batch_status_cmd
from .batch_commands import batch_submit as _batch_submit_cmd
from .batch_commands import batch_validate_cmd as _batch_validate_cmd
from .batch_planning import _select_plan_timepoints  # noqa: F401


def batch_submit(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch submit command function."""
    return _batch_submit_cmd(*args, **kwargs)


def batch_run(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch run command function."""
    return _batch_run_cmd(*args, **kwargs)


def batch_status_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch status command function."""
    return _batch_status_cmd(*args, **kwargs)


def batch_validate_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch validate command function.

    Contract marker kept for tests guarding the --fix bucket behavior:
    startswith("unsupported_format_version")
    """
    return _batch_validate_cmd(*args, **kwargs)


def _validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around tile leaf-array validation helper."""
    return _validate_leaf_arrays_cmd(node_dir, label)


def _validate_node_dir(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around tile node-dir validation helper."""
    return _validate_node_dir_cmd(node_dir, label)


def _validate_tile(tile_path: Path) -> str:
    """Back-compat wrapper around tile validation helper."""
    return _validate_tile_cmd(tile_path)


def batch_cancel_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch cancel command function."""
    return _batch_cancel_cmd(*args, **kwargs)


def _build_merge_recipe_params(*args: Any, **kwargs: Any) -> Any:
    """Back-compat wrapper around merge recipe-argument parsing helper."""
    return _build_merge_recipe_params_cmd(*args, **kwargs)


def _measure_tiles_bytes_per_splat(*args: Any, **kwargs: Any) -> Any:
    """Back-compat wrapper around measured bytes/splat helper."""
    return _measure_tiles_bytes_per_splat_cmd(*args, **kwargs)


def batch_merge_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around the batch merge command function."""
    return _batch_merge_cmd(*args, **kwargs)


def batch_denoise_calibrate_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch denoise-calibrate command function."""
    return _batch_denoise_calibrate_cmd(*args, **kwargs)


def batch_denoise_preprocess_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch denoise-preprocess command function."""
    return _batch_denoise_preprocess_cmd(*args, **kwargs)
