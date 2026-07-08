"""``luxar gsplat batch-fit`` — cluster-scale fitting via Slurm.

Owns the ``app_batch`` Typer sub-app and its commands; the aggregator
(``cli/gsplat_commands.py``) mounts it via ``add_typer``. Extracted from the
former monolithic ``gsplat_commands.py`` (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import typer

if TYPE_CHECKING:
    from luxar.gsplats.lod.recipes import RecipeParams

# Re-exported from helpers (single source); kept importable here for
# back-compat with callers/tests that import these names from this module.
from luxar.cli.gsplat_ops.batch_denoise_workers import (
    run_batch_denoise_calibrate_cmd,
    run_batch_denoise_preprocess_cmd,
)
from luxar.cli.gsplat_ops.batch_measurement import (
    measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat_impl,
)
from luxar.cli.gsplat_ops.batch_merge_command import run_batch_merge_cmd
from luxar.cli.gsplat_ops.batch_planning import _select_plan_timepoints  # noqa: F401
from luxar.cli.gsplat_ops.batch_recipe_args import (
    _MERGE_ALLOWED_TOKENS as _MERGE_ALLOWED_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch_recipe_args import (
    _MERGE_OPTION_TOKENS as _MERGE_OPTION_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch_recipe_args import (
    build_merge_recipe_params as _build_merge_recipe_params_impl,
)
from luxar.cli.gsplat_ops.batch_run import run_batch_run
from luxar.cli.gsplat_ops.batch_status_validate_cancel import (
    run_batch_cancel_cmd,
    run_batch_status_cmd,
    run_batch_validate_cmd,
)
from luxar.cli.gsplat_ops.batch_submit import run_batch_submit
from luxar.cli.gsplat_ops.batch_validation import (
    validate_leaf_arrays as _validate_leaf_arrays_impl,
)
from luxar.cli.gsplat_ops.batch_validation import (
    validate_node_dir as _validate_node_dir_impl,
)
from luxar.cli.gsplat_ops.batch_validation import validate_tile as _validate_tile_impl

app_batch = typer.Typer(
    help="Fit a whole nD dataset across its axes — locally across GPUs "
    "(`batch-fit run`) or on a Slurm cluster (`batch-fit submit`). The "
    "scheduler-agnostic, scaled-up sibling of `gsplat fit`."
)


app_batch.command("submit")(run_batch_submit)


def batch_submit(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch submit command function."""
    return run_batch_submit(*args, **kwargs)


app_batch.command("run")(run_batch_run)


def batch_run(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch run command function."""
    return run_batch_run(*args, **kwargs)


@app_batch.command("status")
def batch_status_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check status of a batch fitting job.

    Reads the manifest, checks for output files, and queries sacct/squeue
    for job states.

    Examples:
        luxar gsplat batch-fit status output_dir/
    """
    return run_batch_status_cmd(output_dir=output_dir, verbose=verbose)


@app_batch.command("validate")
def batch_validate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    fix: bool = typer.Option(
        False, "--fix", help="Delete corrupt/incomplete tiles so they get re-fitted"
    ),
) -> None:
    """Validate integrity of all tiles in a batch output directory.

    Checks each tile for completeness (metadata, arrays, shapes).
    Reports OK, MISSING, CORRUPT, and STALE_TMP counts.

    Use --fix to delete corrupt tiles and leftover .tmp directories,
    so they get re-fitted on the next submit.

    Examples:
        luxar gsplat batch-fit validate output_dir/

        luxar gsplat batch-fit validate output_dir/ --fix
    """
    return run_batch_validate_cmd(output_dir=output_dir, fix=fix)


def _validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around leaf-array validation helper."""
    return _validate_leaf_arrays_impl(node_dir, label)


def _validate_node_dir(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around node-tree validation helper."""
    return _validate_node_dir_impl(node_dir, label)


def _validate_tile(tile_path: Path) -> str:
    """Back-compat wrapper around full tile validation helper."""
    return _validate_tile_impl(tile_path)


@app_batch.command("cancel")
def batch_cancel_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """Cancel all Slurm jobs for a batch fitting run.

    Reads the manifest to find job IDs (calibrate, denoise, fit array,
    merge) and cancels them via scancel.

    Examples:
        luxar gsplat batch-fit cancel output_dir/
    """
    return run_batch_cancel_cmd(output_dir=output_dir)


# Per-part merge recipe parsing lives in ``batch_recipe_args.py``; keep these
# names at module scope for back-compat with tests/importers.
_MERGE_OPTION_TOKENS = _MERGE_OPTION_TOKENS_IMPL
_MERGE_ALLOWED_TOKENS = _MERGE_ALLOWED_TOKENS_IMPL


def _build_merge_recipe_params(
    stored: dict,
    *,
    n_lods: Optional[int] = None,
    additive_method: Optional[str] = None,
    breakpoints: Optional[str] = None,
    compression_factor: Optional[int] = None,
    levels: Optional[int] = None,
    substitutive_method: Optional[str] = None,
    coarsen_dims: Optional[str] = None,
) -> "RecipeParams":
    return _build_merge_recipe_params_impl(
        stored,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
    )


def _measure_tiles_bytes_per_splat(
    tiles_dir: Path, tile_names: List[str]
) -> Tuple[Optional[float], int]:
    """Back-compat wrapper around tile-bytes measurement helper."""
    return _measure_tiles_bytes_per_splat_impl(tiles_dir, tile_names)


app_batch.command("merge")(run_batch_merge_cmd)


def batch_merge_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch merge command function."""
    return run_batch_merge_cmd(*args, **kwargs)


# ── Hidden batch worker commands for denoise pipeline ────────────


app_batch.command("denoise-calibrate", hidden=True)(run_batch_denoise_calibrate_cmd)


def batch_denoise_calibrate_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around hidden denoise-calibrate command."""
    return run_batch_denoise_calibrate_cmd(*args, **kwargs)


app_batch.command("denoise-preprocess", hidden=True)(run_batch_denoise_preprocess_cmd)


def batch_denoise_preprocess_cmd(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around hidden denoise-preprocess command."""
    return run_batch_denoise_preprocess_cmd(*args, **kwargs)
