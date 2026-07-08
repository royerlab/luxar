"""Submission helper for ``batch-fit submit`` Slurm job orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
from arbol import aprint


def submit_batch_jobs(
    *,
    output_dir: Path,
    manifest: Any,
    fit_script: str,
    merge_script: str,
    preamble: str,
    calibrate_script: Optional[str],
    denoise_script: Optional[str],
    preempt_fit_script: Optional[str],
    total_tasks: int,
    preempt_partition: Optional[str],
) -> None:
    """Write sbatch scripts, submit dependent jobs, and persist manifest updates."""
    import subprocess  # nosec B404 - controlled Slurm CLI invocations

    from luxar.gsplats.batch.manifest import save_manifest

    out = output_dir.resolve()
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "merged").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    fit_path = out / "fit_array.sbatch"
    merge_path = out / "merge.sbatch"
    env_path = out / "env_snapshot.sh"

    fit_path.write_text(fit_script)
    merge_path.write_text(merge_script)
    env_path.write_text(preamble)
    if calibrate_script:
        (out / "calibrate.sbatch").write_text(calibrate_script)
    if denoise_script:
        (out / "denoise_array.sbatch").write_text(denoise_script)
    if preempt_fit_script:
        (out / "fit_array_preempt.sbatch").write_text(preempt_fit_script)
    save_manifest(manifest, out)

    def _parse_job_id(stdout: str) -> Optional[int]:
        for word in stdout.strip().split():
            if word.isdigit():
                return int(word)
        return None

    # Submit calibration job (if needed)
    calibrate_job_id = None
    if calibrate_script:
        aprint("Submitting calibration job...")
        result = subprocess.run(  # nosec - trusted local sbatch invocation
            ["sbatch", str(out / "calibrate.sbatch")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            aprint(f"Error submitting calibration job: {result.stderr}")
            raise typer.Exit(1)
        calibrate_job_id = _parse_job_id(result.stdout)
        manifest.calibrate_job_id = calibrate_job_id
        aprint(f"  Calibration job: {calibrate_job_id}")

    # Submit denoise preprocessing array (if preprocess mode)
    denoise_job_id = None
    if denoise_script:
        aprint("Submitting denoise preprocessing array...")
        dep_cmd = ["sbatch"]
        if calibrate_job_id:
            dep_cmd.append(f"--dependency=afterok:{calibrate_job_id}")
        dep_cmd.append(str(out / "denoise_array.sbatch"))
        result = subprocess.run(  # nosec B603 - trusted local sbatch invocation
            dep_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            aprint(f"Error submitting denoise job: {result.stderr}")
            raise typer.Exit(1)
        denoise_job_id = _parse_job_id(result.stdout)
        manifest.denoise_job_id = denoise_job_id
        denoise_n_t = (
            len(manifest.timepoint_indices)
            if manifest.timepoint_indices
            else manifest.n_timepoints
        )
        denoise_n_c = (
            len(manifest.channel_indices)
            if manifest.channel_indices
            else manifest.n_channels
        )
        denoise_total = denoise_n_t * denoise_n_c
        aprint(f"  Denoise array job: {denoise_job_id} ({denoise_total} tasks)")

    # Submit fitting array (depends on denoise or calibrate)
    fit_dep_id = denoise_job_id or calibrate_job_id
    aprint("Submitting fitting array job...")
    fit_cmd = ["sbatch"]
    if fit_dep_id:
        fit_cmd.append(f"--dependency=afterok:{fit_dep_id}")
    fit_cmd.append(str(fit_path))
    result = subprocess.run(  # nosec B603 - trusted local sbatch invocation
        fit_cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        aprint(f"Error submitting fit job: {result.stderr}")
        raise typer.Exit(1)

    fit_job_id = _parse_job_id(result.stdout)
    aprint(f"  Fitting array job: {fit_job_id} ({total_tasks} tasks)")

    # Submit preemptible fit array (if enabled)
    preemptible_job_id = None
    if preempt_fit_script:
        aprint("Submitting preemptible fitting array...")
        preempt_cmd = ["sbatch"]
        if fit_dep_id:
            preempt_cmd.append(f"--dependency=afterok:{fit_dep_id}")
        preempt_cmd.append(str(out / "fit_array_preempt.sbatch"))
        result = subprocess.run(  # nosec B603 - trusted local sbatch invocation
            preempt_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            preemptible_job_id = _parse_job_id(result.stdout)
            manifest.preemptible_job_id = preemptible_job_id
            aprint(
                f"  Preemptible array job: {preemptible_job_id} "
                f"({total_tasks} tasks on {preempt_partition}, requeue)"
            )
        else:
            aprint(
                f"  Warning: preemptible submission failed: {result.stderr}\n"
                "  Continuing with guaranteed partition only."
            )

    # Merge depends on ALL fit arrays
    merge_deps = [jid for jid in [fit_job_id, preemptible_job_id] if jid]
    merge_cmd = ["sbatch"]
    if merge_deps:
        dep_str = ":".join(str(jid) for jid in merge_deps)
        merge_cmd.append(f"--dependency=afterok:{dep_str}")
    merge_cmd.append(str(merge_path))

    result = subprocess.run(  # nosec B603 - trusted local sbatch invocation
        merge_cmd,
        capture_output=True,
        text=True,
    )
    merge_job_id = None
    if result.returncode == 0:
        merge_job_id = _parse_job_id(result.stdout)
        dep_info = " + ".join(str(j) for j in merge_deps)
        aprint(f"  Merge job: {merge_job_id} (depends on {dep_info})")
    else:
        aprint(f"  Warning: merge job submission failed: {result.stderr}")

    manifest.array_job_id = fit_job_id
    manifest.merge_job_id = merge_job_id
    save_manifest(manifest, out)

    aprint(f"\nManifest: {out / 'manifest.json'}")
    aprint(f"Check status: luxar gsplat batch-fit status {out}")
