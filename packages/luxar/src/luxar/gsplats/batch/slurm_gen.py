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

    # Build the fit command template (used in the loop body)
    fit_cmd_parts = [
        f'luxar gsplat fit {shlex.quote(manifest.input_path)} "$OUTPUT"',
        f"    --tile $K/{manifest.n_tiles}",
        f"    --tile-size {manifest.tile_size}",
        f"    --overlap {manifest.tile_overlap}",
        "    --channel $C",
        "    --timepoint $T",
    ]
    if manifest.preset:
        fit_cmd_parts.append(f"    --preset {manifest.preset}")
    for key, value in manifest.fit_args.items():
        if value is not None:
            fit_cmd_parts.append(
                f"    --{key.replace('_', '-')} {shlex.quote(str(value))}"
            )
    fit_cmd = " \\\n    ".join(fit_cmd_parts)

    # Loop body — processes tasks_per_job consecutive tasks
    lines.extend(
        [
            f"TASKS_PER_JOB={tpj}",
            f"TOTAL_TASKS={manifest.total_tasks}",
            f"N_CHANNELS={manifest.n_channels}",
            f"N_TILES={manifest.n_tiles}",
            "BASE_TASK=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))",
            "",
            "for OFFSET in $(seq 0 $((TASKS_PER_JOB - 1))); do",
            "    TASK_ID=$((BASE_TASK + OFFSET))",
            "",
            "    # Stop if we've gone past the last task",
            '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then break; fi',
            "",
            "    # --- Decode task ID -> (timepoint, channel, tile) ---",
            "    T=$((TASK_ID / (N_CHANNELS * N_TILES)))",
            "    R=$((TASK_ID % (N_CHANNELS * N_TILES)))",
            "    C=$((R / N_TILES))",
            "    K=$((R % N_TILES))",
            "",
            "    # Output path",
            f'    OUTPUT="{manifest.output_dir}/tiles/'
            "t$(printf '%02d' $T)_c$(printf '%02d' $C)_tile$(printf '%03d' $K)"
            '.gsplats.zarr"',
            "",
            "    # Skip if already completed (for restarts)",
            '    if [ -d "$OUTPUT" ]; then',
            '        echo "Already exists, skipping: $OUTPUT"',
            "        continue",
            "    fi",
            "",
            f'    echo "=== Task $TASK_ID / $TOTAL_TASKS (T=$T C=$C K=$K) ==="',
            f"    {fit_cmd}",
            "",
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
