# test_fit_command.py
"""Tests for the shared per-task fit-command builder (local + Slurm parity)."""

from __future__ import annotations

import pytest

from luxar.gsplats.batch.fit_command import (
    build_task_fit_argv,
    fit_args_to_tokens,
    iter_fit_arg_flags,
)
from luxar.gsplats.batch.manifest import BatchJob, BatchManifest

_ARGV0 = ["luxar"]


def _job(t: int = 0, c: int = 0, k: int = 0) -> BatchJob:
    return BatchJob(
        task_id=0,
        timepoint=t,
        channel=c,
        tile_index=k,
        output_filename="x.gsplats.zarr",
        estimated_wall_seconds=0.0,
    )


def test_fit_args_to_tokens_expands_flags_and_bools() -> None:
    toks = fit_args_to_tokens({"seeds": "8000", "progressive": "", "iters": "5"})
    assert toks == ["--seeds", "8000", "--progressive", "--iters", "5"]


def test_fit_args_to_tokens_skips_none() -> None:
    assert fit_args_to_tokens({"seeds": None, "iters": "3"}) == ["--iters", "3"]


def test_iter_fit_arg_flags_maps_underscore_to_dash() -> None:
    pairs = list(iter_fit_arg_flags({"cull_retention": "0.9", "denoise": ""}))
    assert pairs == [("--cull-retention", "0.9"), ("--denoise", None)]


def test_fit_args_to_tokens_expands_floor() -> None:
    assert fit_args_to_tokens({"floor": "auto"}) == ["--floor", "auto"]
    assert fit_args_to_tokens({"floor": "0.03"}) == ["--floor", "0.03"]


def test_uniform_argv() -> None:
    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=4,
        tile_size=256,
        tile_overlap=32,
        n_timepoints=1,
        n_channels=1,
        preset="standard",
    )
    argv = build_task_fit_argv(m, _job(k=2), "out.tmp", argv0=_ARGV0)
    assert argv[:5] == ["luxar", "gsplat", "fit", "in.zarr", "out.tmp"]
    assert "--tile" in argv and argv[argv.index("--tile") + 1] == "2/4"
    assert argv[argv.index("--tile-size") + 1] == "256"
    assert argv[argv.index("--overlap") + 1] == "32"
    assert "--channel" not in argv  # single channel, no slicing
    assert "--timepoint" not in argv
    assert "--axes" not in argv
    assert argv[argv.index("--preset") + 1] == "standard"


def test_uniform_argv_carries_tile_local_plan_metadata() -> None:
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        spatial_shape=(8, 8),
        n_tiles=4,
        tile_size=6,
        tile_overlap=2,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
        fold_tile_slivers=True,
        tile_nonempty_counts=[2],
        tile_occupancy_weights=[[1.0, 3.0, 0.0, 0.0]],
        fit_args={"seeds": "100"},
    )
    argv = build_task_fit_argv(manifest, _job(k=0), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--tile-region") + 1] == "0:6,0:6"
    assert argv[argv.index("--tile-volume-shape") + 1] == "8,8"
    assert argv[argv.index("--tile-nonempty-count") + 1] == "2"
    assert argv[argv.index("--tile-seed-count") + 1] == "25"
    assert "--fold-tile-slivers" in argv

    script = generate_fit_sbatch(manifest, "# preamble\n")
    assert '--tile-region "$TILE_REGION"' in script
    assert "TILE_REGIONS=(0:6,0:6 0:6,4:8 4:8,0:6 4:8,4:8)" in script
    assert "NONEMPTY_COUNTS=(2)" in script
    assert "TILE_SEED_COUNTS=(25 75 0 0)" in script
    assert '--tile-seed-count "$TILE_SEED_COUNT"' in script
    assert "--fold-tile-slivers" in script


def test_slurm_large_seed_table_falls_back_to_equal_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.batch.slurm_gen as slurm_gen

    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        spatial_shape=(8, 8),
        n_tiles=4,
        tile_size=6,
        tile_overlap=2,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
        fold_tile_slivers=True,
        tile_nonempty_counts=[2],
        tile_occupancy_weights=[[1.0, 3.0, 0.0, 0.0]],
        fit_args={"seeds": "100"},
    )
    monkeypatch.setattr(slurm_gen, "_MAX_INLINE_TILE_SEED_BYTES", 1)

    script = slurm_gen.generate_fit_sbatch(manifest, "# preamble\n")

    assert "TILE_SEED_COUNTS" not in script
    assert "--tile-seed-count" not in script
    assert '--tile-nonempty-count "$NONEMPTY_COUNT"' in script


def test_old_manifest_keeps_historical_sliver_grid() -> None:
    manifest = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        spatial_shape=(532,),
        n_tiles=2,
        tile_size=512,
        tile_overlap=32,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
    )

    argv = build_task_fit_argv(manifest, _job(k=1), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--tile-region") + 1] == "480:532"
    assert "--fold-tile-slivers" not in argv


def test_tile_local_argv_without_integer_seed_count() -> None:
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        spatial_shape=(8, 8),
        n_tiles=4,
        tile_size=6,
        tile_overlap=2,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
    )
    argv = build_task_fit_argv(manifest, _job(k=2), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--tile-region") + 1] == "4:8,0:6"
    assert "--tile-nonempty-count" not in argv

    script = generate_fit_sbatch(manifest, "# preamble\n")
    assert '--tile-region "$TILE_REGION"' in script
    assert "NONEMPTY_COUNTS=" not in script
    assert "--tile-nonempty-count" not in script


def test_tile_local_grid_mismatch_is_loud_for_local_and_slurm() -> None:
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        spatial_shape=(8, 8, 8),
        n_tiles=7,
        tile_size=4,
        tile_overlap=0,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
    )
    with pytest.raises(ValueError, match="grid mismatch"):
        build_task_fit_argv(manifest, _job(), "out.tmp", argv0=_ARGV0)
    with pytest.raises(ValueError, match="grid mismatch"):
        generate_fit_sbatch(manifest, "# preamble\n")

    manifest.n_tiles = 8
    manifest.tile_nonempty_counts = [1, 1]
    with pytest.raises(ValueError, match="count mismatch"):
        build_task_fit_argv(manifest, _job(), "out.tmp", argv0=_ARGV0)
    with pytest.raises(ValueError, match="count mismatch"):
        generate_fit_sbatch(manifest, "# preamble\n")

    for invalid_count in (0, 9):
        manifest.tile_nonempty_counts = [invalid_count]
        with pytest.raises(ValueError, match="between 1 and n_tiles"):
            build_task_fit_argv(manifest, _job(), "out.tmp", argv0=_ARGV0)
        with pytest.raises(ValueError, match="between 1 and n_tiles"):
            generate_fit_sbatch(manifest, "# preamble\n")

    manifest.tile_nonempty_counts = [1]
    with pytest.raises(ValueError, match="tile index 8 is outside"):
        build_task_fit_argv(manifest, _job(k=8), "out.tmp", argv0=_ARGV0)

    missing_row_job = _job()
    missing_row_job.task_id = 8
    with pytest.raises(ValueError, match="missing tile-local count row 1"):
        build_task_fit_argv(manifest, missing_row_job, "out.tmp", argv0=_ARGV0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([[1.0] * 7], "expected 8"),
        ([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "non-negative"),
        ([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "disagrees"),
    ],
)
def test_tile_local_occupancy_weights_are_validated(
    weights: list[list[float]], message: str
) -> None:
    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        spatial_shape=(8, 8, 8),
        n_tiles=8,
        tile_size=4,
        tile_overlap=0,
        n_timepoints=1,
        n_channels=1,
        tile_local_reads=True,
        tile_nonempty_counts=[1],
        tile_occupancy_weights=weights,
        fit_args={"seeds": "100"},
    )

    with pytest.raises(ValueError, match=message):
        build_task_fit_argv(manifest, _job(), "out.tmp", argv0=_ARGV0)


def test_content_argv() -> None:
    m = BatchManifest(
        input_path="in.zarr",
        mode="content",
        plan_path="/p/plan.json",
        n_tiles=7,
        n_timepoints=1,
        n_channels=1,
    )
    argv = build_task_fit_argv(m, _job(k=5), "out.tmp", argv0=_ARGV0)
    assert "--tiling" in argv and argv[argv.index("--tiling") + 1] == "content"
    assert argv[argv.index("--plan") + 1] == "/p/plan.json"
    assert argv[argv.index("--plan-box") + 1] == "5"
    assert "--tile" not in argv


def test_channel_timepoint_emitted_when_multiple() -> None:
    m = BatchManifest(
        input_path="in.zarr", mode="uniform", n_tiles=1, n_timepoints=3, n_channels=2
    )
    argv = build_task_fit_argv(m, _job(t=2, c=1), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--channel") + 1] == "1"
    assert argv[argv.index("--timepoint") + 1] == "2"


def test_channel_emitted_when_sliced_even_if_single() -> None:
    # One channel selected via slicing -> real index must still be passed.
    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=1,
        n_timepoints=1,
        n_channels=1,
        channel_indices=[5],
    )
    argv = build_task_fit_argv(m, _job(c=5), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--channel") + 1] == "5"


def test_array_key_and_fit_args_forwarded() -> None:
    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=1,
        n_timepoints=1,
        n_channels=1,
        array_key="h2afva/fused",
        fit_args={"seeds": "8000", "progressive": ""},
    )
    argv = build_task_fit_argv(m, _job(), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--array-key") + 1] == "h2afva/fused"
    assert argv[argv.index("--seeds") + 1] == "8000"
    assert "--progressive" in argv


def test_axes_forwarded_to_argv_and_sbatch_parity() -> None:
    """--axes must reach BOTH the local argv and the Slurm bash template."""
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=2,
        tile_size=64,
        tile_overlap=8,
        n_timepoints=1,
        n_channels=1,
        axes="z,y,x",
        output_dir="/o",
    )
    argv = build_task_fit_argv(m, _job(), "out.tmp", argv0=_ARGV0)
    assert argv[argv.index("--axes") + 1] == "z,y,x"
    # Slurm side emits it too (regression for the latent --axes drop).
    script = generate_fit_sbatch(m, "# preamble\n")
    assert "--axes z,y,x" in script


def test_uniform_argv_allows_genuinely_empty_tile_without_floor() -> None:
    """A floor-free empty region remains a legitimate successful task."""
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=4,
        tile_size=256,
        tile_overlap=32,
        n_timepoints=1,
        n_channels=1,
        output_dir="/o",
    )
    argv = build_task_fit_argv(m, _job(k=2), "out.tmp", argv0=_ARGV0)
    assert "--allow-empty-tile" in argv
    # Slurm parity: the sbatch template carries the flag AND finalizes the
    # empty marker for uniform tiles too (not just content boxes). Staging is
    # per-attempt now (${STAGING}=${OUTPUT}.tmp.<job>.<task>.<restart>), so the
    # 0-splat worker writes a sibling ${STAGING}.empty and finalize promotes it
    # to the terminal ${OUTPUT}.empty marker.
    script = generate_fit_sbatch(m, "# preamble\n")
    assert "--allow-empty-tile" in script
    assert '"${STAGING}.empty"' in script
    assert 'touch "${OUTPUT}.empty"' in script
    # The pre-fit cleanup must also remove a STALE ${STAGING}.empty marker (from
    # a crashed run of this same attempt), or the finalize branch would see
    # FIT_RC=0 plus the stale marker, delete the freshly written real staging
    # store, and mark the task empty — a silent spatial hole the merge skips
    # without error. Two occurrences: the pre-fit cleanup and the finalize branch.
    assert script.count('rm -f "${STAGING}.empty"') == 2
    assert script.index('rm -f "${STAGING}.empty"') < script.index("local FIT_RC")


def test_uniform_argv_keeps_sparse_tiles_empty_when_floor_is_applied() -> None:
    """Individual background tiles stay valid; only an all-empty slice is fatal."""
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir="/o",
        mode="uniform",
        n_tiles=1,
        tile_size=64,
        tile_overlap=8,
        fit_args={"floor": "12.5"},
    )

    argv = build_task_fit_argv(manifest, _job(), "out.tmp", argv0=_ARGV0)
    script = generate_fit_sbatch(manifest, "")

    assert "--floor" in argv
    assert "--allow-empty-tile" in argv
    assert "--floor 12.5" in script
    assert "--allow-empty-tile" in script


def test_denoise_h_appended_for_on_the_fly_auto() -> None:
    m = BatchManifest(
        input_path="in.zarr",
        mode="uniform",
        n_tiles=1,
        n_timepoints=1,
        n_channels=1,
        denoise=True,
        denoise_mode="on-the-fly",
        denoise_h=None,
        fit_args={"denoise": ""},
    )
    argv = build_task_fit_argv(m, _job(), "out.tmp", argv0=_ARGV0, denoise_h=0.04)
    assert argv[argv.index("--denoise-h") + 1] == "0.04"
    # No denoise_h passed -> not appended.
    argv2 = build_task_fit_argv(m, _job(), "out.tmp", argv0=_ARGV0)
    assert "--denoise-h" not in argv2
