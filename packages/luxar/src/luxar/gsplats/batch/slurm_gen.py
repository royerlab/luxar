"""Slurm sbatch script generation for batch fitting and merge jobs."""

from __future__ import annotations

import math
import shlex

from luxar.gsplats.batch.manifest import BatchManifest


def generate_fit_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the sbatch array job script for fitting.

    When ``manifest.tasks_per_job == 1`` (default), each Slurm array
    element processes exactly one ``(timepoint, channel, tile)`` combination.

    When ``tasks_per_job > 1``, each Slurm array element loops over
    *tasks_per_job* consecutive fitting tasks sequentially.  This packs
    multiple small volumes onto a single GPU allocation, reducing Slurm
    scheduling overhead for datasets where a single volume doesn't
    saturate the GPU.

    Args:
        manifest: Fully populated batch manifest.
        env_preamble: Shell preamble from :func:`generate_env_preamble`.

    Returns:
        Complete sbatch script as a string.
    """
    tpj = manifest.tasks_per_job
    n_slurm_jobs = math.ceil(manifest.total_tasks / tpj)

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-fit",
        f"#SBATCH --array=0-{n_slurm_jobs - 1}",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-task={manifest.slurm_gpus}",
        f"#SBATCH --cpus-per-task={manifest.slurm_cpus}",
        f"#SBATCH --mem={manifest.slurm_mem_gb}G",
        f"#SBATCH --time={manifest.slurm_time_limit}",
        f"#SBATCH --output={manifest.output_dir}/logs/fit_%a.out",
        f"#SBATCH --error={manifest.output_dir}/logs/fit_%a.err",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")
    for arg in manifest.slurm_extra_args:
        lines.append(f"#SBATCH {arg}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")

    # Compute printf format widths so filenames sort lexicographically.
    # For 1434 timepoints: %04d; for 8 channels: %02d; for 252 tiles: %03d.
    t_width = max(2, len(str(manifest.n_timepoints - 1))) if manifest.n_timepoints > 1 else 2
    c_width = max(2, len(str(manifest.n_channels - 1))) if manifest.n_channels > 1 else 2
    k_width = max(3, len(str(manifest.n_tiles - 1))) if manifest.n_tiles > 1 else 3

    # Build the fit command template (used in the loop body).
    # Only pass --channel / --timepoint when there are multiple values,
    # otherwise _load_zarr_volume's ndim heuristic may interpret them
    # incorrectly (e.g. for 4D TZYX, passing --channel 0 would override
    # --timepoint and always slice dim 0 = channel instead of timepoint).
    fit_cmd_parts = [
        f'luxar gsplat fit {shlex.quote(manifest.input_path)} "$OUTPUT"',
        f"    --tile $K/{manifest.n_tiles}",
        f"    --tile-size {manifest.tile_size}",
        f"    --overlap {manifest.tile_overlap}",
    ]
    if manifest.n_channels > 1:
        fit_cmd_parts.append("    --channel $C")
    if manifest.n_timepoints > 1:
        fit_cmd_parts.append("    --timepoint $T")
    if manifest.preset:
        fit_cmd_parts.append(f"    --preset {manifest.preset}")
    for key, value in manifest.fit_args.items():
        if value is not None:
            fit_cmd_parts.append(
                f"    --{key.replace('_', '-')} {shlex.quote(str(value))}"
            )
    fit_cmd = " \\\n    ".join(fit_cmd_parts)

    # Common variables
    lines.extend(
        [
            f"TASKS_PER_JOB={tpj}",
            f"TOTAL_TASKS={manifest.total_tasks}",
            f"N_CHANNELS={manifest.n_channels}",
            f"N_TILES={manifest.n_tiles}",
            "BASE_TASK=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))",
            "",
        ]
    )

    # Helper function: decode task ID and run fit
    lines.extend(
        [
            "run_task() {",
            "    local TASK_ID=$1",
            '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then return; fi',
            "",
            "    local T=$((TASK_ID / (N_CHANNELS * N_TILES)))",
            "    local R=$((TASK_ID % (N_CHANNELS * N_TILES)))",
            "    local C=$((R / N_TILES))",
            "    local K=$((R % N_TILES))",
            "",
            f'    local OUTPUT="{manifest.output_dir}/tiles/'
            f"t$(printf '%0{t_width}d' $T)_c$(printf '%0{c_width}d' $C)_tile$(printf '%0{k_width}d' $K)"
            '.gsplats.zarr"',
            "",
            '    if [ -d "$OUTPUT" ]; then',
            '        echo "Already exists, skipping: $OUTPUT"',
            "        return",
            "    fi",
            "",
            f'    echo "=== Task $TASK_ID / $TOTAL_TASKS (T=$T C=$C K=$K) ==="',
            f"    {fit_cmd}",
            "}",
            "",
        ]
    )

    if manifest.parallel_tasks_per_job:
        # Parallel mode: launch all tasks as background processes, then wait.
        # Each process gets its own CUDA stream; GPU memory is shared.
        lines.extend(
            [
                "# --- Parallel mode: launch tasks concurrently on the same GPU ---",
                "PIDS=()",
                "for OFFSET in $(seq 0 $((TASKS_PER_JOB - 1))); do",
                "    TASK_ID=$((BASE_TASK + OFFSET))",
                '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then break; fi',
                "    run_task $TASK_ID &",
                "    PIDS+=($!)",
                "done",
                "",
                "# Wait for all parallel tasks and collect exit codes",
                "FAILED=0",
                "for PID in ${PIDS[@]}; do",
                "    if ! wait $PID; then FAILED=$((FAILED + 1)); fi",
                "done",
                'if [ "$FAILED" -gt 0 ]; then',
                '    echo "WARNING: $FAILED of ${#PIDS[@]} parallel tasks failed"',
                "    exit 1",
                "fi",
                "",
            ]
        )
    else:
        # Sequential mode: run tasks one by one
        lines.extend(
            [
                "# --- Sequential mode: run tasks one by one ---",
                "for OFFSET in $(seq 0 $((TASKS_PER_JOB - 1))); do",
                "    TASK_ID=$((BASE_TASK + OFFSET))",
                '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then break; fi',
                "    run_task $TASK_ID",
                "done",
                "",
            ]
        )

    return "\n".join(lines)


def generate_merge_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the merge sbatch script (dependent job).

    Calls ``luxar gsplat batch merge`` to run the 3-level fan-in merge.

    Args:
        manifest: Fully populated batch manifest.
        env_preamble: Shell preamble from :func:`generate_env_preamble`.

    Returns:
        Complete sbatch script as a string.
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-merge",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-task=0",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --mem={max(manifest.slurm_mem_gb, 64)}G",
        "#SBATCH --time=04:00:00",
        f"#SBATCH --output={manifest.output_dir}/logs/merge.out",
        f"#SBATCH --error={manifest.output_dir}/logs/merge.err",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")

    # Merge command
    merge_cmd = f"luxar gsplat batch merge {shlex.quote(manifest.output_dir)}"
    if manifest.channel_colors:
        colors_str = ",".join(manifest.channel_colors)
        merge_cmd += f" --channel-colors {shlex.quote(colors_str)}"

    lines.append(merge_cmd)
    lines.append("")

    return "\n".join(lines)
