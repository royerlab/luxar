"""Slurm sbatch script generation for batch fitting and merge jobs."""

from __future__ import annotations

import math
import shlex
from typing import Optional

from luxar.gsplats.batch.manifest import BatchManifest


def generate_fit_sbatch(
    manifest: BatchManifest,
    env_preamble: str,
    *,
    partition_override: Optional[str] = None,
    max_concurrent_override: Optional[int] = None,
    requeue: bool = False,
    job_name: str = "luxar-fit",
) -> str:
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

    effective_partition = partition_override or manifest.slurm_partition
    effective_concurrent = max_concurrent_override or manifest.max_concurrent

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{n_slurm_jobs - 1}"
        + (f"%{effective_concurrent}" if effective_concurrent else ""),
        f"#SBATCH --partition={effective_partition}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-task={manifest.slurm_gpus}",
        f"#SBATCH --cpus-per-task={manifest.slurm_cpus}",
        f"#SBATCH --mem={manifest.slurm_mem_gb}G",
        f"#SBATCH --time={manifest.slurm_time_limit}",
        f"#SBATCH --output={manifest.output_dir}/logs/{job_name.removeprefix('luxar-')}_%a.out",
        f"#SBATCH --error={manifest.output_dir}/logs/{job_name.removeprefix('luxar-')}_%a.err",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")
    if requeue:
        lines.append("#SBATCH --requeue")
    for arg in manifest.slurm_extra_args:
        lines.append(f"#SBATCH {arg}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")
    lines.append("# Unbuffered Python output for real-time Slurm logging")
    lines.append("export PYTHONUNBUFFERED=1")

    # Requeue logging (for preemptible jobs)
    if requeue:
        lines.extend(
            [
                "",
                "# Log requeue attempts (SLURM_RESTART_COUNT is undefined on first run)",
                'if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then',
                '    echo "Requeued (attempt $((SLURM_RESTART_COUNT + 1)))"',
                "fi",
                'if [ "${SLURM_RESTART_COUNT:-0}" -ge 5 ]; then',
                '    echo "ERROR: preempted 5+ times, giving up on this task"',
                "    exit 1",
                "fi",
            ]
        )
    lines.append("")

    # Compute printf format widths so filenames sort lexicographically.
    # When --timepoints slicing is used, the REAL indices (e.g. 1430) are
    # larger than n_timepoints (20), so width must be based on the max value.
    t_max = (
        max(manifest.timepoint_indices)
        if manifest.timepoint_indices
        else max(0, manifest.n_timepoints - 1)
    )
    c_max = (
        max(manifest.channel_indices)
        if manifest.channel_indices
        else max(0, manifest.n_channels - 1)
    )
    k_max = max(0, manifest.n_tiles - 1)
    t_width = max(2, len(str(t_max)))
    c_width = max(2, len(str(c_max)))
    k_width = max(3, len(str(k_max)))

    # Build the fit command template (used in the loop body).
    # Pass --channel / --timepoint when:
    #   - there are multiple values, OR
    #   - slicing selected specific indices (even a single non-default one)
    # Without slicing, omitting them lets _load_zarr_volume use its ndim
    # heuristic, which avoids the 4D TZYX ambiguity (--channel 0 would
    # override --timepoint).  With slicing, we must pass them to select
    # the correct index even when only one is selected.
    has_explicit_timepoints = manifest.timepoint_indices is not None
    has_explicit_channels = manifest.channel_indices is not None

    fit_cmd_parts = [
        f'luxar gsplat fit {shlex.quote(manifest.input_path)} "${{OUTPUT}}.tmp"',
        f"    --tile $K/{manifest.n_tiles}",
        f"    --tile-size {manifest.tile_size}",
        f"    --overlap {manifest.tile_overlap}",
    ]
    if manifest.array_key is not None:
        fit_cmd_parts.append(f"    --array-key {shlex.quote(manifest.array_key)}")
    if manifest.n_channels > 1 or has_explicit_channels:
        fit_cmd_parts.append("    --channel $C")
    if manifest.n_timepoints > 1 or has_explicit_timepoints:
        fit_cmd_parts.append("    --timepoint $T")
    if manifest.preset:
        fit_cmd_parts.append(f"    --preset {manifest.preset}")
    for key, value in manifest.fit_args.items():
        if value is not None:
            flag = f"--{key.replace('_', '-')}"
            if value == "":
                # Boolean flag (no value)
                fit_cmd_parts.append(f"    {flag}")
            else:
                fit_cmd_parts.append(f"    {flag} {shlex.quote(str(value))}")
    # For on-the-fly denoise with auto-calibration, read h from JSON at runtime
    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
    ):
        # The calibration job writes denoise_h_values.json.
        # Read per-channel h at runtime and inject --denoise-h.
        fit_cmd_parts.append("    --denoise-h $DENOISE_H")

    # For preprocess mode, override input path to denoised zarr.
    # The denoised.zarr stores volumes under "data" with shape
    # (n_selected_t, n_selected_c, *spatial) using sequential indices,
    # so we must use $T_IDX/$C_IDX (not $T/$C which are real dataset indices)
    # and explicitly point at the "data" array key.
    if (
        manifest.denoise
        and manifest.denoise_mode == "preprocess"
        and manifest.denoised_zarr_path
    ):
        # Replace the input path in the command
        fit_cmd_parts[0] = (
            f'luxar gsplat fit {shlex.quote(manifest.denoised_zarr_path)} "${{OUTPUT}}.tmp"'
        )
        # Replace or add --array-key data to point at the denoised dataset
        array_key_replaced = False
        for i, part in enumerate(fit_cmd_parts):
            if part.strip().startswith("--array-key "):
                fit_cmd_parts[i] = "    --array-key data"
                array_key_replaced = True
            elif part.strip() == "--channel $C":
                fit_cmd_parts[i] = "    --channel $C_IDX"
            elif part.strip() == "--timepoint $T":
                fit_cmd_parts[i] = "    --timepoint $T_IDX"
        if not array_key_replaced:
            fit_cmd_parts.append("    --array-key data")

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

    # Index mapping arrays (for --timepoints/--channels slicing)
    if manifest.timepoint_indices is not None:
        t_arr = " ".join(str(i) for i in manifest.timepoint_indices)
        lines.append(f"T_INDICES=({t_arr})")
    if manifest.channel_indices is not None:
        c_arr = " ".join(str(i) for i in manifest.channel_indices)
        lines.append(f"C_INDICES=({c_arr})")
    lines.append("")

    # Helper function: decode task ID and run fit
    has_t_map = manifest.timepoint_indices is not None
    has_c_map = manifest.channel_indices is not None
    lines.extend(
        [
            "run_task() {",
            "    local TASK_ID=$1",
            '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then return; fi',
            "",
            "    local T_IDX=$((TASK_ID / (N_CHANNELS * N_TILES)))",
            "    local R=$((TASK_ID % (N_CHANNELS * N_TILES)))",
            "    local C_IDX=$((R / N_TILES))",
            "    local K=$((R % N_TILES))",
            # Map sequential indices to actual dataset indices
            "    local T=${T_INDICES[$T_IDX]}" if has_t_map else "    local T=$T_IDX",
            "    local C=${C_INDICES[$C_IDX]}" if has_c_map else "    local C=$C_IDX",
            "",
            f'    local OUTPUT="{manifest.output_dir}/tiles/'
            f"t$(printf '%0{t_width}d' $T)_c$(printf '%0{c_width}d' $C)_tile$(printf '%0{k_width}d' $K)"
            '.gsplats.zarr"',
            "",
            "    # Clean up leftover .tmp from a previous crashed run",
            '    if [ -d "${OUTPUT}.tmp" ]; then',
            '        echo "Cleaning up incomplete tile: ${OUTPUT}.tmp"',
            '        rm -rf "${OUTPUT}.tmp"',
            "    fi",
            "",
            '    if [ -d "$OUTPUT" ]; then',
            '        echo "Already exists, skipping: $OUTPUT"',
            "        return",
            "    fi",
            "",
            '    echo "=== Task $TASK_ID / $TOTAL_TASKS (T=$T C=$C K=$K) ==="',
        ]
    )

    # For on-the-fly denoise, read per-channel h at runtime
    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
    ):
        h_json_path = shlex.quote(f"{manifest.output_dir}/denoise_h_values.json")
        lines.extend(
            [
                f"    local H_JSON={h_json_path}",
                '    local DENOISE_H=$(python3 -c "import json,sys; '
                "d=json.load(open(sys.argv[1])); "
                'print(d.get(str(int(sys.argv[2])), 0.04))" "$H_JSON" "$C")',
            ]
        )

    lines.extend(
        [
            f"    {fit_cmd}",
            "    local FIT_RC=$?",
            '    if [ "$FIT_RC" -ne 0 ] || [ ! -d "${OUTPUT}.tmp" ]; then',
            '        echo "ERROR: fit failed (rc=$FIT_RC), cleaning up"',
            '        rm -rf "${OUTPUT}.tmp"',
            "        return 1",
            "    fi",
            "    # Atomic rename — handles race with parallel preemptible job",
            '    if ! mv -T "${OUTPUT}.tmp" "$OUTPUT" 2>/dev/null; then',
            '        if [ -d "$OUTPUT" ]; then',
            '            echo "Tile completed by another task, cleaning up duplicate"',
            '            rm -rf "${OUTPUT}.tmp"',
            "        else",
            '            echo "ERROR: mv failed and output missing"',
            "            return 1",
            "        fi",
            "    else",
            '        echo "Tile saved: $OUTPUT"',
            "    fi",
            "}",
            "",
        ]
    )

    if manifest.parallel_tasks_per_job:
        # Parallel mode: launch all tasks as background processes, then wait.
        # Each process owns its own CUDA context (process-level isolation);
        # kernels share the same GPU and memory pool but execute on independent
        # default streams across processes.
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


def generate_calibrate_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate sbatch script for NLM calibration job.

    Single GPU, ~10 min. Runs ``luxar gsplat batch denoise-calibrate``
    which calibrates h per channel and writes results to manifest + JSON.
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-calibrate",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-task=1",
        "#SBATCH --cpus-per-task=4",
        f"#SBATCH --mem={manifest.slurm_mem_gb}G",
        "#SBATCH --time=01:00:00",  # Large zarr.zip archives need I/O time
        f"#SBATCH --output={manifest.output_dir}/logs/calibrate.out",
        f"#SBATCH --error={manifest.output_dir}/logs/calibrate.err",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")
    lines.append(
        f"luxar gsplat batch denoise-calibrate {shlex.quote(manifest.output_dir)}"
    )
    lines.append("")

    return "\n".join(lines)


def generate_denoise_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate sbatch script for denoise preprocessing array job.

    Array job: one task per (timepoint, channel). Each task denoises one
    volume and writes to ``denoised.zarr``.
    """
    # Use actual selected counts (not original dataset counts) for task array
    n_t = (
        len(manifest.timepoint_indices)
        if manifest.timepoint_indices
        else manifest.n_timepoints
    )
    n_c = (
        len(manifest.channel_indices)
        if manifest.channel_indices
        else manifest.n_channels
    )
    total_tasks = n_t * n_c
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-denoise",
        f"#SBATCH --array=0-{total_tasks - 1}"
        + (f"%{manifest.max_concurrent}" if manifest.max_concurrent else ""),
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-task=1",
        "#SBATCH --cpus-per-task=4",
        f"#SBATCH --mem={max(manifest.slurm_mem_gb, 64)}G",  # NLM + volume loading headroom
        "#SBATCH --time=01:00:00",
        f"#SBATCH --output={manifest.output_dir}/logs/denoise_%a.out",
        f"#SBATCH --error={manifest.output_dir}/logs/denoise_%a.err",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")
    lines.append(
        f"luxar gsplat batch denoise-preprocess "
        f"{shlex.quote(manifest.output_dir)} $SLURM_ARRAY_TASK_ID"
    )
    lines.append("")

    return "\n".join(lines)


def generate_merge_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the merge sbatch script (dependent job).

    Calls ``luxar gsplat batch merge`` (no ``--flat``), which streams the tiles
    into a ``kind=partition`` file — one part per spatial tile — by default.

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
