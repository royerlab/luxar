"""Slurm sbatch script generation for batch fitting and merge jobs."""

from __future__ import annotations

import shlex

from luxar.gsplats.batch.manifest import BatchManifest


def generate_fit_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the sbatch array job script for fitting.

    Each array task decodes its ``SLURM_ARRAY_TASK_ID`` into
    ``(timepoint, channel, tile_index)`` and runs
    ``luxar gsplat fit`` with the appropriate flags.

    Args:
        manifest: Fully populated batch manifest.
        env_preamble: Shell preamble from :func:`generate_env_preamble`.

    Returns:
        Complete sbatch script as a string.
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-fit",
        f"#SBATCH --array=0-{manifest.total_tasks - 1}",
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

    # Task ID decoding
    lines.extend(
        [
            "# --- Decode task ID -> (timepoint, channel, tile) ---",
            "TASK_ID=$SLURM_ARRAY_TASK_ID",
            f"N_CHANNELS={manifest.n_channels}",
            f"N_TILES={manifest.n_tiles}",
            "T=$((TASK_ID / (N_CHANNELS * N_TILES)))",
            "R=$((TASK_ID % (N_CHANNELS * N_TILES)))",
            "C=$((R / N_TILES))",
            "K=$((R % N_TILES))",
            "",
            "# Output path",
            f'OUTPUT="{manifest.output_dir}/tiles/'
            "t$(printf '%02d' $T)_c$(printf '%02d' $C)_tile$(printf '%03d' $K)"
            '.gsplats.zarr"',
            "",
            "# Skip if already completed (for restarts)",
            'if [ -d "$OUTPUT" ]; then',
            '    echo "Output already exists, skipping: $OUTPUT"',
            "    exit 0",
            "fi",
            "",
        ]
    )

    # Build the fit command
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

    # Add extra fit args from config (shell-escaped)
    for key, value in manifest.fit_args.items():
        if value is not None:
            fit_cmd_parts.append(
                f"    --{key.replace('_', '-')} {shlex.quote(str(value))}"
            )

    lines.append(" \\\n".join(fit_cmd_parts))
    lines.append("")

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
