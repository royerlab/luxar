"""Slurm sbatch script generation for batch fitting and merge jobs."""

from __future__ import annotations

import math
import shlex
from typing import Optional

from luxar.gsplats.batch.fit_command import iter_fit_arg_flags
from luxar.gsplats.batch.manifest import BatchManifest
from luxar.io.ome_zarr import classify_axis_labels


def _preprocessed_axes(manifest: BatchManifest) -> Optional[str]:
    """Return explicit axes for the canonical preprocessed store."""
    if manifest.axes is None:
        spatial_rank = len(manifest.spatial_shape)
        if spatial_rank < 2:
            return None
        spatial_axes = ["z"] * (spatial_rank - 2) + ["y", "x"]
        return ",".join(["t", "c", *spatial_axes])
    source_axes = [label.strip() for label in manifest.axes.split(",")]
    _, _, spatial_indices = classify_axis_labels(source_axes)
    spatial_axes = [source_axes[index] for index in spatial_indices]
    return ",".join(["t", "c", *spatial_axes])


def _retarget_preprocessed_fit_command(
    fit_cmd_parts: list[str], manifest: BatchManifest, denoised_zarr_path: str
) -> None:
    """Retarget a fit command from the source to the canonical denoised store."""
    fit_cmd_parts[0] = (
        f'luxar gsplat fit {shlex.quote(denoised_zarr_path)} "${{STAGING}}"'
    )
    canonical_axes = _preprocessed_axes(manifest)
    axes_replaced = False
    array_key_replaced = False
    for index, part in enumerate(fit_cmd_parts):
        if canonical_axes is not None and part.strip().startswith("--axes "):
            fit_cmd_parts[index] = f"    --axes {shlex.quote(canonical_axes)}"
            axes_replaced = True
        elif part.strip().startswith("--array-key "):
            fit_cmd_parts[index] = "    --array-key data"
            array_key_replaced = True
        elif part.strip() == "--channel $C":
            fit_cmd_parts[index] = "    --channel $C_IDX"
        elif part.strip() == "--timepoint $T":
            fit_cmd_parts[index] = "    --timepoint $T_IDX"
    if canonical_axes is not None and not axes_replaced:
        fit_cmd_parts.append(f"    --axes {shlex.quote(canonical_axes)}")
    if not array_key_replaced:
        fit_cmd_parts.append("    --array-key data")


def _validated_output_dir(output_dir: str) -> str:
    """Return an output directory safe to embed in a generated sbatch file.

    Slurm's directive parser and Bash both honor shell-style quoting for
    ordinary metacharacters, but an sbatch directive is physically line-oriented.
    A line break could therefore terminate an
    ``--output``/``--error`` directive before either Slurm or Bash sees the
    quoted value.  NUL likewise truncates C-string parsing.
    """
    if any(char in output_dir for char in ("\0", "\r", "\n")):
        raise ValueError(
            "Batch output directory must not contain NUL, carriage returns, or newlines"
        )
    return output_dir


def _slurm_log_path(output_dir: str, log_name: str) -> str:
    """Quote one ``--output``/``--error`` directive value.

    Quoting alone is not enough here: Slurm expands filename-pattern tokens
    (``%j``, ``%A``, ``%a``, ...) in these directives after tokenization, so a
    percent sign in the user's directory must be doubled to stay literal.
    ``log_name`` is appended verbatim, keeping its intentional ``%a``.
    """
    return shlex.quote(f"{output_dir.replace('%', '%%')}/logs/{log_name}")


def _runtime_denoise_floor_lines(manifest: BatchManifest, output_dir: str) -> list[str]:
    """Bash lines that load deferred denoise/floor values without hiding errors."""
    lines: list[str] = []
    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
    ):
        h_json_path = shlex.quote(f"{output_dir}/denoise_h_values.json")
        lines.extend(
            [
                f"    local H_JSON={h_json_path}",
                "    local DENOISE_H",
                '    if ! DENOISE_H=$(python3 -c "import json,sys; '
                "d=json.load(open(sys.argv[1])); "
                'print(d.get(str(int(sys.argv[2])), 0.04))" "$H_JSON" "$C"); then',
                '        echo "Failed to read denoise h from $H_JSON" >&2',
                "        return 1",
                "    fi",
                '    if [ -z "$DENOISE_H" ]; then',
                '        echo "Empty denoise h in $H_JSON" >&2',
                "        return 1",
                "    fi",
            ]
        )
    if manifest.floor_deferred:
        floor_json_path = shlex.quote(f"{output_dir}/floor_level.json")
        lines.extend(
            [
                f"    local FLOOR_JSON={floor_json_path}",
                "    local FLOOR_LEVEL",
                '    if ! FLOOR_LEVEL=$(python3 -c "import json,sys; '
                'print(json.load(open(sys.argv[1]))[\'forward\'])" "$FLOOR_JSON"); then',
                '        echo "Failed to read floor level from $FLOOR_JSON" >&2',
                "        return 1",
                "    fi",
                '    if [ -z "$FLOOR_LEVEL" ]; then',
                '        echo "Empty floor level in $FLOOR_JSON" >&2',
                "        return 1",
                "    fi",
            ]
        )
    return lines


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
    output_dir = _validated_output_dir(manifest.output_dir)
    log_stem = job_name.removeprefix("luxar-")

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
        f"#SBATCH --output={_slurm_log_path(output_dir, f'{log_stem}_%a.out')}",
        f"#SBATCH --error={_slurm_log_path(output_dir, f'{log_stem}_%a.err')}",
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

    # Spatial slot = a uniform tile (`--tile K/M`) or a content box of the shared
    # plan (`--tiling content --plan … --plan-box K`). $K is the spatial index.
    is_content = manifest.mode == "content"
    slot_label = "box" if is_content else "tile"
    if is_content:
        fit_cmd_parts = [
            f'luxar gsplat fit {shlex.quote(manifest.input_path)} "${{STAGING}}"',
            "    --tiling content",
            f"    --plan {shlex.quote(manifest.plan_path or '')}",
            "    --plan-box $K",
        ]
    else:
        fit_cmd_parts = [
            f'luxar gsplat fit {shlex.quote(manifest.input_path)} "${{STAGING}}"',
            f"    --tile $K/{manifest.n_tiles}",
            f"    --tile-size {manifest.tile_size}",
            f"    --overlap {manifest.tile_overlap}",
            # A tile wholly below the run's background floor legitimately fits
            # 0 splats; the worker then writes an `.empty` marker and exits 0
            # (finalized below) instead of failing the task forever.
            "    --allow-empty-tile",
        ]
    if manifest.array_key is not None:
        fit_cmd_parts.append(f"    --array-key {shlex.quote(manifest.array_key)}")
    if manifest.n_channels > 1 or has_explicit_channels:
        fit_cmd_parts.append("    --channel $C")
    if manifest.n_timepoints > 1 or has_explicit_timepoints:
        fit_cmd_parts.append("    --timepoint $T")
    if manifest.preset:
        fit_cmd_parts.append(f"    --preset {shlex.quote(manifest.preset)}")
    # Shared fit_args -> flag mapping (single source: fit_command.iter_fit_arg_flags),
    # formatted here as quoted bash lines.
    for flag, value in iter_fit_arg_flags(manifest.fit_args):
        if value is None:
            fit_cmd_parts.append(f"    {flag}")  # boolean flag
        else:
            fit_cmd_parts.append(f"    {flag} {shlex.quote(value)}")
    if manifest.floor_deferred:
        fit_cmd_parts.append('    --floor "$FLOOR_LEVEL"')
    if manifest.axes:
        # Forward the explicit axis order so each task loads the same shape the
        # planner discovered (else the positional heuristic can mis-order axes).
        fit_cmd_parts.append(f"    --axes {shlex.quote(manifest.axes)}")
    # For on-the-fly denoise with auto-calibration, read h from JSON at runtime
    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
    ):
        # The calibration job writes denoise_h_values.json.
        # Read per-channel h at runtime and inject --denoise-h.
        fit_cmd_parts.append('    --denoise-h "$DENOISE_H"')

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
        _retarget_preprocessed_fit_command(
            fit_cmd_parts, manifest, manifest.denoised_zarr_path
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
    output_tiles_dir = shlex.quote(f"{output_dir}/tiles")
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
            f"    local OUTPUT={output_tiles_dir}/"
            f"t$(printf '%0{t_width}d' $T)_c$(printf '%0{c_width}d' $C)_{slot_label}$(printf '%0{k_width}d' $K)"
            ".gsplats.zarr",
            "",
            "    # Per-attempt staging dir: SLURM_JOB_ID/ARRAY_TASK_ID differ",
            "    # between the guaranteed and preemptible arrays, and",
            "    # SLURM_RESTART_COUNT increments on requeue, so every concurrent",
            "    # attempt owns a unique staging store and can never interleave",
            "    # chunk/metadata writes with another attempt.",
            '    local STAGING="${OUTPUT}.tmp.${SLURM_JOB_ID:-0}'
            '.${SLURM_ARRAY_TASK_ID:-0}.${SLURM_RESTART_COUNT:-0}"',
            "    # Clean up leftovers from a previous crashed run of THIS attempt.",
            "    # The staging dir is now per-attempt, so this only ever removes",
            "    # our own store — the old cross-attempt interleaving hazard is",
            "    # gone. A stale .empty marker must go too (the local runner does",
            "    # the same): with FIT_RC=0 it would make the finalize step",
            "    # delete a freshly written real staging store and mark the task",
            "    # empty — a silent spatial hole the merge skips without error.",
            '    if [ -d "${STAGING}" ]; then',
            '        echo "Cleaning up incomplete tile: ${STAGING}"',
            '        rm -rf "${STAGING}"',
            "    fi",
            '    rm -f "${STAGING}.empty"',
            "",
            '    if [ -d "$OUTPUT" ]; then',
            '        echo "Already exists, skipping: $OUTPUT"',
            "        return",
            "    fi",
            "",
            '    echo "=== Task $TASK_ID / $TOTAL_TASKS (T=$T C=$C K=$K) ==="',
        ]
    )

    if manifest.parallel_tasks_per_job:
        threads_per_worker = max(1, manifest.slurm_cpus // tpj)
        workers_per_device = math.ceil(tpj / max(1, manifest.slurm_gpus))
        lines.extend(
            [
                "    local WORKER_OFFSET=$((TASK_ID - BASE_TASK))",
                f"    export OMP_NUM_THREADS={threads_per_worker}",
                f"    export MKL_NUM_THREADS={threads_per_worker}",
                f"    export LUXAR_QUALITY_WORKERS_PER_HOST={tpj}",
                f"    export LUXAR_QUALITY_WORKERS_PER_DEVICE={workers_per_device}",
            ]
        )
        if manifest.slurm_gpus > 1:
            lines.append(
                "    export CUDA_VISIBLE_DEVICES="
                f"$((WORKER_OFFSET % {manifest.slurm_gpus}))"
            )
        lines.append("")

    lines.extend(_runtime_denoise_floor_lines(manifest, output_dir))

    lines.extend(
        [
            f"    {fit_cmd}",
            "    local FIT_RC=$?",
        ]
    )
    # A task that fits 0 splats (a content box, or a uniform tile wholly below
    # the run's background floor) writes a sibling `${STAGING}.empty` marker
    # (the writer rejects empty stores) instead of the staging store. Treat that
    # as a clean, legitimately-empty result: leave a `${OUTPUT}.empty` marker
    # the merge skips, and exit 0 (NOT a failed task).
    lines.extend(
        [
            '    if [ "$FIT_RC" -eq 0 ] && [ -f "${STAGING}.empty" ]; then',
            '        rm -f "${STAGING}.empty"',
            '        rm -rf "${STAGING}"',
            "        # Claim the empty marker FIRST, then recheck: a racing real",
            "        # attempt removes the marker after its mv -T claim, so in",
            "        # every interleaving a real store and the marker never both",
            "        # survive — a real result always wins. (Checking before",
            "        # touching leaves a window — the check passes, the real",
            "        # attempt promotes and clears, then the touch lands — that",
            "        # would strand both terminal representations on disk.)",
            '        touch "${OUTPUT}.empty"',
            '        if [ -d "$OUTPUT" ]; then',
            '            echo "Completed by another task, dropping empty result"',
            '            rm -f "${OUTPUT}.empty"',
            "        else",
            f'            echo "Empty {slot_label} (0 splats): ${{OUTPUT}}.empty"',
            "        fi",
            "        return 0",
            "    fi",
        ]
    )
    lines.extend(
        [
            '    if [ "$FIT_RC" -ne 0 ] || [ ! -d "${STAGING}" ]; then',
            '        echo "ERROR: fit failed (rc=$FIT_RC), cleaning up"',
            '        rm -rf "${STAGING}"',
            "        return 1",
            "    fi",
            "    # Atomic claim — mv -T of a directory onto an existing non-empty",
            "    # OUTPUT fails, so a concurrent attempt that already promoted its",
            "    # own (isolated) staging makes this attempt fall into the loser",
            "    # branch below instead of clobbering the winner.",
            '    if ! mv -T "${STAGING}" "$OUTPUT" 2>/dev/null; then',
            '        if [ -d "$OUTPUT" ]; then',
            '            echo "Tile completed by another task, cleaning up duplicate"',
            '            rm -rf "${STAGING}"',
            "        else",
            '            echo "ERROR: mv failed and output missing"',
            "            return 1",
            "        fi",
            "    else",
            '        echo "Tile saved: $OUTPUT"',
            "    fi",
            "    # A real store now stands at OUTPUT (ours or the winner's) —",
            "    # drop any stale empty marker from an earlier 0-splat attempt so",
            "    # resume/status/merge never mistake this slot for legitimately",
            "    # empty after the store is later removed.",
            '    rm -f "${OUTPUT}.empty"',
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
                "FAILED=0",
                "for OFFSET in $(seq 0 $((TASKS_PER_JOB - 1))); do",
                "    TASK_ID=$((BASE_TASK + OFFSET))",
                '    if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then break; fi',
                "    if ! run_task $TASK_ID; then FAILED=$((FAILED + 1)); fi",
                "done",
                'if [ "$FAILED" -gt 0 ]; then',
                '    if [ "$FAILED" -eq 1 ]; then',
                '        echo "WARNING: 1 sequential task failed"',
                "    else",
                '        echo "WARNING: $FAILED sequential tasks failed"',
                "    fi",
                "    exit 1",
                "fi",
                "",
            ]
        )

    return "\n".join(lines)


def generate_calibrate_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate sbatch script for NLM calibration job.

    Single GPU, ~10 min. Runs ``luxar gsplat batch-fit denoise-calibrate``
    which calibrates h per channel and writes results to manifest + JSON.
    """
    output_dir = _validated_output_dir(manifest.output_dir)
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-calibrate",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-task=1",
        "#SBATCH --cpus-per-task=4",
        f"#SBATCH --mem={manifest.slurm_mem_gb}G",
        "#SBATCH --time=01:00:00",  # Large zarr.zip archives need I/O time
        f"#SBATCH --output={_slurm_log_path(output_dir, 'calibrate.out')}",
        f"#SBATCH --error={_slurm_log_path(output_dir, 'calibrate.err')}",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")
    lines.append(f"luxar gsplat batch-fit denoise-calibrate {shlex.quote(output_dir)}")
    lines.append("")

    return "\n".join(lines)


def generate_denoise_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate sbatch script for denoise preprocessing array job.

    Array job: one task per (timepoint, channel). Each task denoises one
    volume and writes to ``denoised.zarr``.
    """
    output_dir = _validated_output_dir(manifest.output_dir)
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
        f"#SBATCH --output={_slurm_log_path(output_dir, 'denoise_%a.out')}",
        f"#SBATCH --error={_slurm_log_path(output_dir, 'denoise_%a.err')}",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")
    lines.append(
        f"luxar gsplat batch-fit denoise-preprocess "
        f"{shlex.quote(output_dir)} $SLURM_ARRAY_TASK_ID"
    )
    lines.append("")

    return "\n".join(lines)


def generate_floor_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the single dependent job that resolves a denoised floor."""
    output_dir = _validated_output_dir(manifest.output_dir)
    sampled_pairs = min(manifest.n_timepoints, 4) * min(manifest.n_channels, 4)
    walltime_hours = (
        max(1, sampled_pairs) if manifest.denoise_mode == "on-the-fly" else 1
    )
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-floor",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        f"#SBATCH --mem={max(manifest.slurm_mem_gb, 32)}G",
        f"#SBATCH --time={walltime_hours:02d}:00:00",
        f"#SBATCH --output={_slurm_log_path(output_dir, 'floor.out')}",
        f"#SBATCH --error={_slurm_log_path(output_dir, 'floor.err')}",
    ]
    if manifest.denoise_mode == "on-the-fly":
        lines.append("#SBATCH --gpus-per-task=1")
    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")
    lines.extend(
        [
            "",
            env_preamble,
            "",
            f"luxar gsplat batch-fit resolve-floor {shlex.quote(output_dir)}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_merge_sbatch(manifest: BatchManifest, env_preamble: str) -> str:
    """Generate the merge sbatch script (dependent job).

    Calls ``luxar gsplat batch-fit merge`` (no ``--flat``), which streams the tiles
    into a ``kind=partition`` file — one part per spatial tile — by default.

    Args:
        manifest: Fully populated batch manifest.
        env_preamble: Shell preamble from :func:`generate_env_preamble`.

    Returns:
        Complete sbatch script as a string.
    """
    output_dir = _validated_output_dir(manifest.output_dir)
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=luxar-merge",
        f"#SBATCH --partition={manifest.slurm_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-task=0",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --mem={max(manifest.slurm_mem_gb, 64)}G",
        "#SBATCH --time=04:00:00",
        f"#SBATCH --output={_slurm_log_path(output_dir, 'merge.out')}",
        f"#SBATCH --error={_slurm_log_path(output_dir, 'merge.err')}",
    ]

    if manifest.slurm_account:
        lines.append(f"#SBATCH --account={manifest.slurm_account}")
    if manifest.slurm_qos:
        lines.append(f"#SBATCH --qos={manifest.slurm_qos}")

    lines.append("")
    lines.append(env_preamble)
    lines.append("")

    # Merge command
    merge_cmd = f"luxar gsplat batch-fit merge {shlex.quote(output_dir)}"
    if manifest.channel_colors:
        colors_str = ",".join(manifest.channel_colors)
        merge_cmd += f" --channel-colors {shlex.quote(colors_str)}"
    # Per-part LOD recipe (if planned): emit `--recipe <r>` + its knobs so the
    # merge job streams a partition of LOD'd parts rather than bare leaves.
    # Manifests may carry legacy recipe spellings (pre-rename runs); the merge
    # CLI rejects those, so emit the canonical name.
    if manifest.merge_recipe:
        from luxar.gsplats.lod.recipes import canonical_recipe_name
        from luxar.utils.lod_methods import canonical_method_token

        merge_cmd += (
            f" --recipe {shlex.quote(canonical_recipe_name(manifest.merge_recipe))}"
        )
        # Same legacy-spelling problem as the recipe name above: a manifest
        # from before the 2026-08 method-flag rename stores `substitutive-method`,
        # which would emit `--substitutive-method` and die on an unknown option.
        for flag, value in manifest.merge_recipe_args.items():
            merge_cmd += f" --{canonical_method_token(flag)} {shlex.quote(str(value))}"

    lines.append(merge_cmd)
    lines.append("")

    return "\n".join(lines)
