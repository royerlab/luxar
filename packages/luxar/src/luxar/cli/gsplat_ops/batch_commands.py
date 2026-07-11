"""``luxar gsplat batch-fit`` — cluster-scale fitting via Slurm.

Owns the ``app_batch`` Typer sub-app and its commands; the aggregator
(``cli/gsplat_commands.py``) mounts it via ``add_typer``. Extracted from the
former monolithic ``gsplat_commands.py`` (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

from luxar.cli.gsplat_ops.batch_denoise_workers import (
    run_batch_denoise_calibrate_cmd,
    run_batch_denoise_preprocess_cmd,
)
from luxar.cli.gsplat_ops.batch_merge_command import run_batch_merge_cmd
from luxar.cli.gsplat_ops.batch_run import run_batch_run
from luxar.cli.gsplat_ops.batch_status_validate_cancel import (
    run_batch_cancel_cmd,
    run_batch_status_cmd,
    run_batch_validate_cmd,
)
from luxar.cli.gsplat_ops.batch_submit import run_batch_submit

app_batch = typer.Typer(
    help="Fit a whole nD dataset across its axes — locally across GPUs "
    "(`batch-fit run`) or on a Slurm cluster (`batch-fit submit`). The "
    "scheduler-agnostic, scaled-up sibling of `gsplat fit`."
)


app_batch.command("submit")(run_batch_submit)


app_batch.command("run")(run_batch_run)


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


app_batch.command("merge")(run_batch_merge_cmd)


# ── Hidden batch worker commands for denoise pipeline ────────────


app_batch.command("denoise-calibrate", hidden=True)(run_batch_denoise_calibrate_cmd)


app_batch.command("denoise-preprocess", hidden=True)(run_batch_denoise_preprocess_cmd)
