"""Behavioral tests for generated Slurm fitting scripts."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from luxar.gsplats.batch.manifest import BatchManifest
from luxar.gsplats.batch.slurm_gen import (
    _parallel_worker_env_lines,
    generate_calibrate_sbatch,
    generate_denoise_sbatch,
    generate_fit_sbatch,
    generate_merge_sbatch,
)


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


def test_parallel_fit_scales_allocation_and_isolates_workers() -> None:
    manifest = _packed_manifest(3)
    manifest.parallel_tasks_per_job = True
    manifest.slurm_gpus = 2
    manifest.slurm_cpus = 4
    manifest.slurm_mem_gb = 32
    manifest.slurm_cpus_total = 12
    manifest.slurm_mem_gb_total = 96

    script = generate_fit_sbatch(manifest, "")

    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=96G" in script
    assert "export OMP_NUM_THREADS=4" in script
    assert "export MKL_NUM_THREADS=4" in script
    assert "export LUXAR_QUALITY_WORKERS_PER_HOST=3" in script
    assert "export LUXAR_QUALITY_WORKERS_PER_DEVICE=2" in script
    assert "local WORKER_GPU_INDEX=$((WORKER_OFFSET % 2))" in script
    assert (
        'export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$WORKER_GPU_INDEX]}"' in script
    )
    assert 'export CUDA_VISIBLE_DEVICES="$WORKER_GPU_INDEX"' not in script

    merge_script = generate_merge_sbatch(manifest, "")
    calibrate_script = generate_calibrate_sbatch(manifest, "")
    assert "#SBATCH --mem=64G" in merge_script
    assert "#SBATCH --mem=32G" in calibrate_script


def test_parallel_worker_selects_from_slurm_allocated_gpu_ids() -> None:
    manifest = _packed_manifest(3)
    manifest.parallel_tasks_per_job = True
    manifest.slurm_gpus = 2
    manifest.slurm_cpus = 4
    lines = _parallel_worker_env_lines(manifest, 3)
    harness = "\n".join(
        [
            "CUDA_VISIBLE_DEVICES=3,5",
            "IFS=',' read -r -a ALLOCATED_GPUS <<< \"$CUDA_VISIBLE_DEVICES\"",
            "BASE_TASK=0",
            "worker() {",
            "    local TASK_ID=1",
            *lines,
            '    printf "%s %s %s %s\\n" "$CUDA_VISIBLE_DEVICES" '
            '"$OMP_NUM_THREADS" "$LUXAR_QUALITY_WORKERS_PER_HOST" '
            '"$LUXAR_QUALITY_WORKERS_PER_DEVICE"',
            "}",
            "worker",
        ]
    )

    result = subprocess.run(
        ["bash"], input=harness, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "5 4 3 2"


def test_parallel_worker_keeps_parent_gpu_list_when_token_is_missing() -> None:
    manifest = _packed_manifest(3)
    manifest.parallel_tasks_per_job = True
    manifest.slurm_gpus = 2
    lines = _parallel_worker_env_lines(manifest, 3)
    harness = "\n".join(
        [
            "CUDA_VISIBLE_DEVICES=0",
            "IFS=',' read -r -a ALLOCATED_GPUS <<< \"$CUDA_VISIBLE_DEVICES\"",
            "BASE_TASK=0",
            "worker() {",
            "    local TASK_ID=1",
            *lines,
            '    printf "%s\\n" "$CUDA_VISIBLE_DEVICES"',
            "}",
            "worker",
        ]
    )

    result = subprocess.run(
        ["bash"], input=harness, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def _directive_value(script: str, option: str) -> str:
    """Parse one generated ``#SBATCH --option=value`` directive."""
    prefix = f"#SBATCH --{option}="
    line = next(line for line in script.splitlines() if line.startswith(prefix))
    arguments = shlex.split(line.removeprefix("#SBATCH "), comments=True)
    assert len(arguments) == 1
    return arguments[0].split("=", maxsplit=1)[1]


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


def test_all_sbatch_log_paths_quote_the_output_directory() -> None:
    output_dir = (
        "/scratch/output dir 'single' \"double\" "
        "$(touch injected) `touch injected-backtick` run%j 50% # hash"
    )
    manifest = _packed_manifest(1)
    manifest.output_dir = output_dir
    # Slurm expands %-patterns in --output/--error after tokenization, so the
    # directive must carry the directory's percent signs doubled; the trailing
    # %a array token stays single so per-task logs still split.
    directive_dir = output_dir.replace("%", "%%")

    scripts = [
        (generate_fit_sbatch(manifest, ""), "fit_%a.out", "fit_%a.err"),
        (generate_calibrate_sbatch(manifest, ""), "calibrate.out", "calibrate.err"),
        (generate_denoise_sbatch(manifest, ""), "denoise_%a.out", "denoise_%a.err"),
        (generate_merge_sbatch(manifest, ""), "merge.out", "merge.err"),
    ]

    for script, stdout_name, stderr_name in scripts:
        assert (
            _directive_value(script, "output") == f"{directive_dir}/logs/{stdout_name}"
        )
        assert (
            _directive_value(script, "error") == f"{directive_dir}/logs/{stderr_name}"
        )


def test_fit_script_keeps_output_dir_and_preset_literal(tmp_path: Path) -> None:
    output_dir = tmp_path / (
        "output $(touch output-injected) `touch output-backtick` "
        "\"double\" 'single' 50% # hash"
    )
    preset = (
        "standard $(touch preset-injected) `touch preset-backtick` \"quoted\" 'single'"
    )
    capture_path = tmp_path / "fit-args.txt"
    manifest = _packed_manifest(1)
    manifest.output_dir = str(output_dir)
    manifest.preset = preset

    preamble = f"""
SLURM_ARRAY_TASK_ID=0
CAPTURE_ARGS={shlex.quote(str(capture_path))}
luxar() {{
    printf '%s\\n' "$@" > "$CAPTURE_ARGS"
    mkdir -p "$4"
}}
mv() {{
    if [ "$1" = "-T" ]; then shift; fi
    command mv "$@"
}}
"""
    script = generate_fit_sbatch(manifest, preamble)
    result = subprocess.run(
        ["bash"],
        input=script,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture_path.read_text().splitlines()
    expected_output = output_dir / "tiles/t00_c00_tile000.gsplats.zarr"
    # Fit now writes into the per-attempt STAGING dir; the harness sets only
    # SLURM_ARRAY_TASK_ID=0, so JOB_ID/RESTART_COUNT default to 0 -> `.tmp.0.0.0`.
    assert arguments[:4] == [
        "gsplat",
        "fit",
        "/data/test.zarr",
        f"{expected_output}.tmp.0.0.0",
    ]
    preset_index = arguments.index("--preset")
    assert arguments[preset_index + 1] == preset
    assert expected_output.is_dir()
    assert not (tmp_path / "output-injected").exists()
    assert not (tmp_path / "output-backtick").exists()
    assert not (tmp_path / "preset-injected").exists()
    assert not (tmp_path / "preset-backtick").exists()


def test_fit_script_uses_per_attempt_staging_dir() -> None:
    """Issue #679: each fit attempt owns a unique staging dir, not one .tmp.

    The generated ``run_task()`` must derive ``STAGING`` from the Slurm attempt
    identifiers (JOB_ID / ARRAY_TASK_ID / RESTART_COUNT) so guaranteed and
    preemptible arrays — and requeues — never write into the same store. Every
    former ``${OUTPUT}.tmp`` reference must now go through ``${STAGING}``, and the
    atomic ``mv -T`` claim must promote ``${STAGING}`` (not ``.tmp``) to OUTPUT.
    """
    script = generate_fit_sbatch(_packed_manifest(1), "")

    staging_def = (
        'local STAGING="${OUTPUT}.tmp.${SLURM_JOB_ID:-0}'
        '.${SLURM_ARRAY_TASK_ID:-0}.${SLURM_RESTART_COUNT:-0}"'
    )
    assert staging_def in script

    # The ONLY line allowed to reference ${OUTPUT}.tmp is the STAGING definition;
    # everything else (fit output, cleanup, empty check, mv -T) uses ${STAGING}.
    tmp_lines = [ln for ln in script.splitlines() if "${OUTPUT}.tmp" in ln]
    assert tmp_lines == [f"    {staging_def}"], tmp_lines

    # Atomic claim promotes the isolated staging store.
    assert 'mv -T "${STAGING}" "$OUTPUT"' in script
    # Fit writes into the staging store.
    assert '"${STAGING}"' in script
    # Empty-marker handling is per-attempt too.
    assert '[ -f "${STAGING}.empty" ]' in script


def test_fit_script_empty_marker_defers_to_real_output() -> None:
    """A real store always wins over an empty result — never both on disk.

    The 0-splat branch claims the marker FIRST and then rechecks OUTPUT,
    dropping the marker if a real store already stands; a successful/lost
    `mv -T` claim also drops any stale marker. Touch-before-recheck (plus the
    real path's remove-after-mv) closes every interleaving — checking before
    touching leaves a window where a racing real promotion lands between the
    check and the touch and both terminal representations survive, so a later
    store removal would silently turn the slot 'legitimately empty'.
    """
    script = generate_fit_sbatch(_packed_manifest(1), "")
    lines = script.splitlines()

    # The empty branch touches the marker, THEN rechecks OUTPUT and drops the
    # marker when a real store already stands.
    empty_if = next(
        i for i, ln in enumerate(lines) if '[ -f "${STAGING}.empty" ]' in ln
    )
    touch = next(i for i, ln in enumerate(lines) if 'touch "${OUTPUT}.empty"' in ln)
    guard = next(
        i
        for i, ln in enumerate(lines)
        if i > empty_if and 'if [ -d "$OUTPUT" ]; then' in ln
    )
    drop = next(
        i for i, ln in enumerate(lines) if i > guard and 'rm -f "${OUTPUT}.empty"' in ln
    )
    assert empty_if < touch < guard < drop

    # After the atomic claim, a real store stands at OUTPUT: the stale marker
    # is removed (winner and loser paths both flow through this line).
    mv = next(i for i, ln in enumerate(lines) if 'mv -T "${STAGING}"' in ln)
    marker_rm = next(
        i for i, ln in enumerate(lines) if i > mv and 'rm -f "${OUTPUT}.empty"' in ln
    )
    assert marker_rm > mv


def test_content_fit_script_uses_staging_dir() -> None:
    """Content mode (plan-box) fit also writes into ${STAGING}, not .tmp."""
    manifest = _packed_manifest(1)
    manifest.mode = "content"
    manifest.plan_path = "/output/plan.json"
    script = generate_fit_sbatch(manifest, "")
    assert "--plan-box" in script
    tmp_lines = [ln for ln in script.splitlines() if "${OUTPUT}.tmp" in ln]
    assert all("local STAGING=" in ln for ln in tmp_lines), tmp_lines
    assert 'mv -T "${STAGING}" "$OUTPUT"' in script


@pytest.mark.parametrize("separator", ["\0", "\r", "\n"])
def test_sbatch_output_dir_rejects_line_terminators(separator: str) -> None:
    manifest = _packed_manifest(1)
    manifest.output_dir = f"/output{separator}#SBATCH --partition=other"

    with pytest.raises(ValueError, match="must not contain"):
        generate_fit_sbatch(manifest, "")
