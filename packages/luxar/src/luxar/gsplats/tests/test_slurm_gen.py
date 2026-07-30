"""Behavioral tests for generated Slurm fitting scripts."""

from __future__ import annotations

import subprocess

import pytest

from luxar.gsplats.batch.manifest import BatchManifest
from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch


def _packed_manifest(total_tasks: int) -> BatchManifest:
    return BatchManifest(
        input_path="/data/test.zarr",
        output_dir="/output",
        n_channels=1,
        n_tiles=total_tasks,
        total_tasks=total_tasks,
        tasks_per_job=3,
        tile_size=256,
        slurm_partition="gpu",
        slurm_time_limit="01:00:00",
    )


def _run_sequential_driver(
    script: str,
    *,
    task_statuses: dict[int, int],
    total_tasks: int,
) -> subprocess.CompletedProcess[str]:
    """Execute the generated sequential control flow with a stub task body."""
    marker = "# --- Sequential mode: run tasks one by one ---"
    driver = script[script.index(marker) :]
    status_cases = "\n".join(
        f"        {task_id}) return {status} ;;"
        for task_id, status in sorted(task_statuses.items())
    )
    harness = f"""
TASKS_PER_JOB=3
TOTAL_TASKS={total_tasks}
BASE_TASK=0
run_task() {{
    echo "ran:$1"
    case "$1" in
{status_cases}
        *) return 99 ;;
    esac
}}
{driver}
"""
    return subprocess.run(
        ["bash"],
        input=harness,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("total_tasks", "task_statuses", "expected_runs", "expected_warning"),
    [
        (
            3,
            {0: 0, 1: 1, 2: 0},
            ["ran:0", "ran:1", "ran:2"],
            "WARNING: 1 sequential task failed",
        ),
        (
            2,
            {0: 0, 1: 1},
            ["ran:0", "ran:1"],
            "WARNING: 1 sequential task failed",
        ),
        (
            3,
            {0: 1, 1: 0, 2: 1},
            ["ran:0", "ran:1", "ran:2"],
            "WARNING: 2 sequential tasks failed",
        ),
    ],
)
def test_sequential_packing_retains_any_task_failure(
    total_tasks: int,
    task_statuses: dict[int, int],
    expected_runs: list[str],
    expected_warning: str,
) -> None:
    """A later success or padded-tail break must not hide a failure."""
    script = generate_fit_sbatch(_packed_manifest(total_tasks), "")
    result = _run_sequential_driver(
        script,
        task_statuses=task_statuses,
        total_tasks=total_tasks,
    )

    assert result.returncode == 1
    assert result.stdout.splitlines()[: len(expected_runs)] == expected_runs
    assert expected_warning in result.stdout


def test_sequential_packing_succeeds_when_every_task_succeeds() -> None:
    script = generate_fit_sbatch(_packed_manifest(3), "")
    result = _run_sequential_driver(
        script,
        task_statuses={0: 0, 1: 0, 2: 0},
        total_tasks=3,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ran:0", "ran:1", "ran:2"]
