"""Regression tests for allocation-aware Slurm task packing."""

from luxar.cli.gsplat_ops.batch.submit_packing import resolve_tasks_per_job
from luxar.cli.gsplat_ops.batch.submit_pipeline import resolve_packing


def test_parallel_packing_is_bounded_by_node_cpus() -> None:
    resolution = resolve_tasks_per_job(
        None,
        parallel=True,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(512, 512, 512),
        tile_voxels=64**3,
        est_seconds=60,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=[(8, 256 * 1024)],
    )

    assert resolution.count == 2
    assert resolution.limit == "node CPU count"


def test_parallel_packing_uses_conservative_fallback_without_node_probe() -> None:
    resolution = resolve_tasks_per_job(
        None,
        parallel=True,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(1024, 1024, 1024),
        tile_voxels=32**3,
        est_seconds=60,
        gpus_per_task=2,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=None,
    )

    assert resolution.count == 4
    assert resolution.limit == "unknown node capacity"


def test_sequential_packing_preserves_existing_heuristic() -> None:
    resolution = resolve_tasks_per_job(
        None,
        parallel=False,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(100,),
        tile_voxels=10,
        est_seconds=60,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=None,
    )

    assert resolution.count == 10
    assert resolution.limit == "packing cap"


def test_parallel_packing_scales_requested_per_task_resources(monkeypatch) -> None:
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_slurm_scheduler_info",
        lambda: {"uses_backfill": False, "max_jobs_per_user": 10},
    )
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_partition_node_resources",
        lambda partition: [(16, 256 * 1024)],
    )

    plan = resolve_packing(
        None,
        parallel=True,
        max_shape=(1024, 1024, 1024),
        tile_voxels=64**3,
        est_seconds=60,
        total_tasks=8,
        time_limit="01:00:00",
        partition="gpu",
        gpus_per_task=1,
        cpus=4,
        mem=32,
    )

    assert plan.tasks_per_job == 4
    assert plan.limiting_resource == "node CPU count"
    assert plan.slurm_cpus == 16
    assert plan.slurm_mem_gb == 128
