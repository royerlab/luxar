# test_fit_command.py
"""Tests for the shared per-task fit-command builder (local + Slurm parity)."""

from __future__ import annotations

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


def test_uniform_argv_allows_empty_tile() -> None:
    """Uniform tasks must tolerate a tile wholly below the background floor:
    the worker writes an ``.empty`` marker and exits 0 instead of the task
    failing (and re-failing on every re-fit) at the empty-store save."""
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
