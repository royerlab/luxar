"""Implementation helpers for batch status/validate/cancel commands."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess  # nosec B404
from pathlib import Path

import typer
from arbol import aprint

from ..._traceback import exit_with_error
from .validation import validate_tile as _validate_tile_impl

# Per-attempt staging tokens (the part after `{tile}.tmp.`):
# Slurm attempts are `<jobid>.<taskid>.<restart>`, local attempts `<host>-<pid>`
# (see slurm_gen.generate_fit_sbatch / local_runner._staging_path).
_SLURM_TOKEN_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_LOCAL_TOKEN_RE = re.compile(r"^(.+)-(\d+)$")


def _staging_attempt_live(token: str) -> bool | None:
    """Best-effort liveness of the attempt that owns a staging ``token``.

    Returns ``True`` (the attempt is still running — its staging must NOT be
    deleted: a partial delete under a live zarr writer can end with a corrupt
    store being promoted), ``False`` (the attempt is gone — safe to reclaim), or
    ``None`` (cannot verify from this machine — keep it, to be safe). A legacy
    shared ``.tmp`` (empty token, pre-attempt-isolation) carries no owner and is
    always reclaimable.

    For a local token the probed pid is the runner process — the only process
    that ever promotes that staging to the final output — so a dead pid means
    the staging can never be promoted and is safe to reclaim, even if an
    orphaned fit worker is still writing into it. For a Slurm token the probed
    job spans both the fit and the ``mv -T`` promotion, so job liveness covers
    the writer and the promoter alike.
    """
    if not token:
        return False
    m = _SLURM_TOKEN_RE.match(token)
    if m:
        try:
            # Trusted squeue probe — fixed argv, no shell.
            res = subprocess.run(  # nosec B603, B607
                ["squeue", "-h", "-j", m.group(1), "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None  # no squeue here — cannot verify
        if res.returncode != 0:
            # squeue exits non-zero both when the job id is unknown (the job
            # left the queue — safe to reclaim) and on operational failures
            # (controller unreachable, auth/config error). Only the former
            # proves the attempt is gone; anything else is unverifiable.
            if "invalid job id" in (res.stderr or "").lower():
                return False  # Slurm no longer knows the job
            return None  # squeue itself failed — cannot verify
        return bool(res.stdout.strip())
    m = _LOCAL_TOKEN_RE.match(token)
    if m:
        host, pid = m.group(1), int(m.group(2))
        if host != socket.gethostname():
            return None  # another machine's pid — cannot probe over NFS
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # pid exists, owned by another user
        return True
    return None  # unrecognized token — keep, to be safe


def run_batch_status_cmd(*, output_dir: Path) -> None:
    """Run ``batch-fit status`` command implementation."""
    try:
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.status import (
            check_batch_status,
            format_status_report,
        )

        manifest = load_manifest(output_dir)
        status = check_batch_status(output_dir)
        aprint(format_status_report(status, manifest))

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)


def run_batch_validate_cmd(*, output_dir: Path, fix: bool) -> None:
    """Run ``batch-fit validate`` command implementation."""
    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)
        tiles_dir = output_dir / "tiles"

        if not tiles_dir.exists():
            aprint("No tiles directory found.")
            raise typer.Exit(1)

        # Build expected tile list from manifest
        expected_tiles = [job.output_filename for job in manifest.jobs]
        aprint(f"Checking {len(expected_tiles)} expected tiles...")

        ok = 0
        missing = 0
        empty = 0
        corrupt = 0
        unmigrated = 0
        stale_tmp = 0
        active_tmp = 0
        corrupt_reasons: list[str] = []
        unmigrated_reasons: list[str] = []

        for tile_name in expected_tiles:
            tile_path = tiles_dir / tile_name

            # Check for stale per-attempt staging leftovers. Staging dirs are now
            # named `{tile}.tmp.<token>` — `{tile}.tmp.<host>-<pid>` for a local
            # run, `{tile}.tmp.<jobid>.<taskid>.<restart>` for Slurm — plus a
            # possible `{tile}.tmp.<token>.empty` marker file and a possible
            # `{tile}.tmp.<token>.old` set-aside prior tile (a --no-resume refit
            # mid-promotion). The glob matches every such leftover while NEVER
            # matching the legitimate
            # `{tile}.empty` marker (it has no `.tmp` in its name). A leftover
            # whose owning attempt is still running (or can't be verified) is
            # counted as ACTIVE and never deleted — these are exactly the paths
            # live fits write into, and deleting one mid-write can corrupt the
            # store that attempt is about to promote.
            for leftover in sorted(tiles_dir.glob(f"{tile_name}.tmp*")):
                is_marker = not leftover.is_dir() and leftover.name.endswith(".empty")
                if not leftover.is_dir() and not is_marker:
                    continue
                suffix = leftover.name[len(tile_name) + len(".tmp") :]
                if is_marker:
                    suffix = suffix[: -len(".empty")]
                elif suffix.endswith(".old"):
                    # A --no-resume refit's set-aside prior tile (see
                    # local_runner._finalize_output): owned by the same attempt
                    # as its staging, reclaimed under the same liveness rule.
                    suffix = suffix[: -len(".old")]
                live = _staging_attempt_live(suffix.lstrip("."))
                if live is not False:
                    active_tmp += 1
                    if fix:
                        why = (
                            "attempt still running"
                            if live
                            else "cannot verify the attempt finished"
                        )
                        aprint(f"  Kept: {leftover.name} ({why})")
                    continue
                # Orphaned by a crash/preemption. Count it regardless of --fix
                # so report/dry-run mode is honest.
                stale_tmp += 1
                if fix:
                    if leftover.is_dir():
                        shutil.rmtree(leftover)
                    else:
                        leftover.unlink(missing_ok=True)
                    aprint(f"  Deleted: {leftover.name}")

            if not tile_path.is_dir():
                # A `<tile>.empty` marker = the task ran and legitimately produced
                # 0 splats (content boxes / sparse tiles); that is NOT missing.
                if (tiles_dir / f"{tile_name}.empty").exists():
                    empty += 1
                else:
                    missing += 1
                continue

            # Validate tile integrity
            reason = _validate_tile_impl(tile_path)
            if reason == "ok":
                ok += 1
            elif reason.startswith("unsupported_format_version"):
                # Recoverable, NOT corrupt: an unmigrated legacy tile. Never
                # delete it under --fix — it converts via `gsplat migrate-format`.
                unmigrated += 1
                unmigrated_reasons.append(f"  {tile_name}: {reason}")
            else:
                corrupt += 1
                corrupt_reasons.append(f"  {tile_name}: {reason}")
                if fix:
                    shutil.rmtree(tile_path)
                    aprint(f"  Deleted corrupt: {tile_name} ({reason})")

        # Summary
        aprint("")
        aprint(f"  OK:         {ok}")
        aprint(f"  EMPTY:      {empty}")
        aprint(f"  MISSING:    {missing}")
        aprint(f"  CORRUPT:    {corrupt}")
        aprint(f"  UNMIGRATED: {unmigrated}")
        aprint(f"  STALE_TMP:  {stale_tmp}")
        if active_tmp:
            aprint(f"  ACTIVE_TMP: {active_tmp} (in use by a running attempt — kept)")

        if corrupt_reasons and not fix:
            aprint("")
            aprint("Corrupt tiles:")
            for r in corrupt_reasons:
                aprint(r)
            aprint("")
            aprint("Run with --fix to delete corrupt tiles.")

        if unmigrated_reasons:
            aprint("")
            aprint("Unmigrated (legacy-format) tiles — NOT deleted:")
            for r in unmigrated_reasons:
                aprint(r)
            aprint("")
            aprint("Convert each with `luxar gsplat migrate-format <tile> <out>`.")

        if fix and (corrupt > 0 or stale_tmp > 0):
            aprint(f"\nFixed: deleted {corrupt} corrupt + {stale_tmp} stale .tmp")
            aprint("Resubmit to re-fit deleted tiles.")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)


def run_batch_cancel_cmd(*, output_dir: Path) -> None:
    """Run ``batch-fit cancel`` command implementation."""
    import subprocess  # nosec B404

    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)

        job_ids = []
        for attr in (
            "calibrate_job_id",
            "denoise_job_id",
            "floor_job_id",
            "array_job_id",
            "merge_job_id",
        ):
            jid = getattr(manifest, attr, None)
            if jid is not None:
                job_ids.append(str(jid))

        if not job_ids:
            aprint("No job IDs found in manifest — nothing to cancel.")
            raise typer.Exit(0)

        aprint(f"Cancelling {len(job_ids)} job(s): {', '.join(job_ids)}")
        # Trusted local scancel invocation — fixed argv, no shell.
        result = subprocess.run(  # nosec B603
            ["scancel"] + job_ids,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            aprint("All jobs cancelled.")
        else:
            # scancel may warn about already-completed jobs — that's fine
            aprint(f"scancel output: {result.stderr.strip()}")
            aprint("Cancel command sent (some jobs may have already completed).")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)
