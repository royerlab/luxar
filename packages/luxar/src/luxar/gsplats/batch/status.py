"""Batch job status checking via output files and Slurm sacct."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from luxar.gsplats.batch.manifest import BatchManifest, load_manifest


@dataclass
class BatchStatus:
    """Aggregated status of a batch fitting job."""

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0
    unknown: int = 0
    merge_status: str = "not_started"


def check_batch_status(output_dir: Path) -> BatchStatus:
    """Check the status of a batch fitting job.

    1. Load manifest.
    2. Check which output files exist (→ completed).
    3. Query sacct for remaining tasks (→ running/pending/failed).
    4. Check merge status.

    Args:
        output_dir: Batch output directory containing ``manifest.json``.

    Returns:
        Aggregated :class:`BatchStatus`.
    """
    manifest = load_manifest(output_dir)
    status = BatchStatus(total_tasks=manifest.total_tasks)

    tiles_dir = output_dir / "tiles"

    # 1. Check output files. A slot is "completed" if its store exists OR it
    # produced a legitimately-empty result (a `<output>.empty` marker — content
    # boxes / sparse tiles that fit 0 splats), so those aren't miscounted as
    # failed/unknown below.
    completed_ids: set = set()
    for job in manifest.jobs:
        tile_path = tiles_dir / job.output_filename
        if tile_path.exists() or Path(f"{tile_path}.empty").exists():
            status.completed += 1
            completed_ids.add(job.task_id)

    # 2. Query sacct for non-completed tasks
    remaining = manifest.total_tasks - status.completed
    if remaining > 0 and manifest.array_job_id is not None:
        sacct_states = _query_sacct(manifest.array_job_id)
        for task_id in range(manifest.total_tasks):
            if task_id in completed_ids:
                continue
            state = sacct_states.get(task_id)
            if state is None:
                status.unknown += 1
            elif state in ("RUNNING",):
                status.running += 1
            elif state in ("PENDING", "REQUEUED"):
                status.pending += 1
            elif state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                status.failed += 1
            elif state in ("COMPLETED",):
                # sacct says completed but file missing — treat as failed
                status.failed += 1
            else:
                status.unknown += 1
    else:
        status.unknown = remaining

    # 3. Check merge status
    merged_dir = output_dir / "merged"
    # The final output depends on dataset shape:
    #   C>1: merged/final.gsplats.zarr
    #   C==1, T>1: merged/c00_4d.gsplats.zarr
    #   C==1, T==1: merged/t00_c00.gsplats.zarr
    if manifest.n_channels > 1:
        final_path = merged_dir / "final.gsplats.zarr"
    elif manifest.n_timepoints > 1:
        final_path = merged_dir / "c00_4d.gsplats.zarr"
    else:
        final_path = merged_dir / "t00_c00.gsplats.zarr"

    if final_path.exists():
        status.merge_status = "completed"
    elif manifest.merge_job_id is not None:
        merge_state = _query_single_job_state(manifest.merge_job_id)
        if merge_state == "RUNNING":
            status.merge_status = "running"
        elif merge_state in ("FAILED", "CANCELLED", "TIMEOUT"):
            status.merge_status = "failed"
        elif merge_state == "PENDING":
            status.merge_status = "pending"
        elif merge_state == "COMPLETED":
            status.merge_status = "completed"

    return status


def format_status_report(
    status: BatchStatus,
    manifest: BatchManifest,
    verbose: bool = False,
) -> str:
    """Format a human-readable status report.

    Args:
        status: Aggregated batch status.
        manifest: Batch manifest for context.
        verbose: Include per-task details.

    Returns:
        Multi-line status string.
    """
    pct = (status.completed / status.total_tasks * 100) if status.total_tasks > 0 else 0

    lines = [
        f"Batch Status: {manifest.output_dir}",
        f"  {status.completed}/{status.total_tasks} completed ({pct:.1f}%)",
    ]

    details = []
    if status.running:
        details.append(f"{status.running} running")
    if status.pending:
        details.append(f"{status.pending} pending")
    if status.failed:
        details.append(f"{status.failed} failed")
    if status.unknown:
        details.append(f"{status.unknown} unknown")
    if details:
        lines.append(f"    {', '.join(details)}")

    lines.append(f"  Merge: {status.merge_status}")

    if manifest.array_job_id:
        lines.append(f"  Slurm array job: {manifest.array_job_id}")
    if manifest.merge_job_id:
        lines.append(f"  Slurm merge job: {manifest.merge_job_id}")

    return "\n".join(lines)


def _query_sacct(array_job_id: int) -> Dict[int, str]:
    """Query sacct for task states of an array job.

    Returns:
        Dict mapping task_id → state string (e.g. "RUNNING", "COMPLETED").
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(array_job_id),
                "--format=JobID,State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}

        states: Dict[int, str] = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_id_str, state = parts[0], parts[1]
            # Parse array task ID: "12345_42" → task_id=42
            if "_" in job_id_str:
                try:
                    task_id = int(job_id_str.split("_")[1])
                    states[task_id] = state.strip()
                except (ValueError, IndexError):
                    pass
        return states

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


def _query_single_job_state(job_id: int) -> Optional[str]:
    """Query sacct for the state of a single job."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "--format=State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
        return lines[0] if lines else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
