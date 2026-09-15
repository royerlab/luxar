"""Regression tests for allocation-aware Slurm task packing."""

import subprocess
import sys

from luxar.cli.gsplat_ops.batch.submit_packing import (
    largest_node_class,
    resolve_tasks_per_job,
)
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
    assert resolution.node_class == (8, 256 * 1024)


def test_parallel_packing_does_not_require_torch() -> None:
    script = """
import builtins

original_import = builtins.__import__

def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError(
            "torch blocked for core-install regression test", name="torch"
        )
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_torch

from luxar.cli.gsplat_ops.batch.submit_packing import resolve_tasks_per_job

result = resolve_tasks_per_job(
    None,
    parallel=True,
    uses_backfill=False,
    no_job_limit=False,
    max_shape=(512, 512, 512),
    tile_voxels=64**3,
    est_seconds=60,
    gpus_per_task=2,
    cpus_per_task=4,
    mem_gb_per_task=32,
    node_resources=[(64, 512 * 1024)],
)
assert (result.count, result.limit) == (4, "GPU memory")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )

    assert completed.returncode == 0, completed.stderr


def test_parallel_packing_keeps_every_distinct_node_class_eligible() -> None:
    resolution = resolve_tasks_per_job(
        None,
        parallel=True,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(1024, 1024, 1024),
        tile_voxels=64**3,
        est_seconds=60,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=[
            (16, 128 * 1024),
            (64, 512 * 1024),
            (16, 128 * 1024),
        ],
    )

    assert resolution.count == 4
    assert resolution.limit == "node CPU count"
    assert resolution.node_class == (16, 128 * 1024)


def test_parallel_packing_is_bounded_by_gpu_memory() -> None:
    single_gpu = resolve_tasks_per_job(
        None,
        parallel=True,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(256, 256, 256),
        tile_voxels=200**3,
        est_seconds=60,
        gpus_per_task=1,
        cpus_per_task=1,
        mem_gb_per_task=1,
        node_resources=[(128, 1024 * 1024)],
    )
    two_gpus = resolve_tasks_per_job(
        None,
        parallel=True,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(256, 256, 256),
        tile_voxels=200**3,
        est_seconds=60,
        gpus_per_task=2,
        cpus_per_task=1,
        mem_gb_per_task=1,
        node_resources=[(128, 1024 * 1024)],
    )

    assert (single_gpu.count, single_gpu.limit) == (1, "GPU memory")
    assert (two_gpus.count, two_gpus.limit) == (2, "GPU memory")


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
    assert resolution.limit == "volume ratio"


def test_sequential_packing_reports_volume_ratio_when_it_binds() -> None:
    resolution = resolve_tasks_per_job(
        None,
        parallel=False,
        uses_backfill=False,
        no_job_limit=False,
        max_shape=(50,),
        tile_voxels=10,
        est_seconds=60,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=None,
    )

    assert (resolution.count, resolution.limit) == (5, "volume ratio")


def test_sequential_packing_reports_backfill_and_cap_limiters() -> None:
    backfill = resolve_tasks_per_job(
        None,
        parallel=False,
        uses_backfill=True,
        no_job_limit=True,
        max_shape=(1000,),
        tile_voxels=10,
        est_seconds=200,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=None,
    )
    capped = resolve_tasks_per_job(
        None,
        parallel=False,
        uses_backfill=True,
        no_job_limit=True,
        max_shape=(1000,),
        tile_voxels=10,
        est_seconds=10,
        gpus_per_task=1,
        cpus_per_task=4,
        mem_gb_per_task=32,
        node_resources=None,
    )

    assert (backfill.count, backfill.limit) == (1, "backfill window")
    assert (capped.count, capped.limit) == (3, "packing cap")


def test_largest_node_class_breaks_zero_capacity_ties_by_resources() -> None:
    node_class = largest_node_class(
        [(16, 128 * 1024), (64, 512 * 1024)],
        cpus_per_task=128,
        mem_gb_per_task=32,
    )

    assert node_class == (64, 512 * 1024)


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
    assert plan.limiting_resource == "node CPU count on 16-CPU/256G node class"
    assert plan.slurm_cpus == 4
    assert plan.slurm_mem_gb == 32
    assert plan.slurm_cpus_total == 16
    assert plan.slurm_mem_gb_total == 128


def test_sequential_packing_does_not_probe_node_resources(monkeypatch) -> None:
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_slurm_scheduler_info",
        lambda: {"uses_backfill": False, "max_jobs_per_user": 10},
    )
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_partition_node_resources",
        lambda _partition: (_ for _ in ()).throw(AssertionError("unexpected probe")),
    )

    plan = resolve_packing(
        None,
        parallel=False,
        max_shape=(100,),
        tile_voxels=10,
        est_seconds=60,
        total_tasks=10,
        time_limit="01:00:00",
        partition="gpu",
        gpus_per_task=1,
        cpus=4,
        mem=32,
    )

    assert plan.tasks_per_job == 10


def test_explicit_unschedulable_parallel_packing_warns(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_slurm_scheduler_info",
        lambda: {"uses_backfill": False, "max_jobs_per_user": 10},
    )
    monkeypatch.setattr(
        "luxar.gsplats.batch.env_capture.get_partition_node_resources",
        lambda _partition: [(16, 128 * 1024), (64, 512 * 1024)],
    )
    monkeypatch.setattr(
        "luxar.cli.gsplat_ops.batch.submit_pipeline.aprint", messages.append
    )

    plan = resolve_packing(
        64,
        parallel=True,
        max_shape=(256, 256, 256),
        tile_voxels=64**3,
        est_seconds=60,
        total_tasks=64,
        time_limit="01:00:00",
        partition="gpu",
        gpus_per_task=1,
        cpus=4,
        mem=32,
    )

    assert plan.tasks_per_job == 64
    assert messages == [
        "Warning: requested parallel packing needs 256 CPUs and 2048G RAM, "
        "but the largest gpu node class has 64 CPUs and 512G RAM; "
        "the job may remain pending."
    ]
