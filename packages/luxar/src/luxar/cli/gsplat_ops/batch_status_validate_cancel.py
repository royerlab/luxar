"""Implementation helpers for batch status/validate/cancel commands."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from arbol import aprint

from .batch_validation import validate_tile as _validate_tile_impl


def run_batch_status_cmd(*, output_dir: Path, verbose: bool) -> None:
    """Run ``batch-fit status`` command implementation."""
    try:
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.status import (
            check_batch_status,
            format_status_report,
        )

        manifest = load_manifest(output_dir)
        status = check_batch_status(output_dir)
        aprint(format_status_report(status, manifest, verbose=verbose))

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1)


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
        corrupt_reasons: list[str] = []
        unmigrated_reasons: list[str] = []

        for tile_name in expected_tiles:
            tile_path = tiles_dir / tile_name
            tmp_path = tiles_dir / f"{tile_name}.tmp"

            # Check for stale .tmp
            if tmp_path.is_dir():
                stale_tmp += 1
                if fix:
                    shutil.rmtree(tmp_path)
                    aprint(f"  Deleted: {tile_name}.tmp")

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
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


def run_batch_cancel_cmd(*, output_dir: Path) -> None:
    """Run ``batch-fit cancel`` command implementation."""
    import subprocess  # nosec B404 - controlled Slurm CLI invocation

    try:
        from luxar.gsplats.batch.manifest import load_manifest

        manifest = load_manifest(output_dir)

        job_ids = []
        for attr in (
            "calibrate_job_id",
            "denoise_job_id",
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
        result = subprocess.run(  # nosec B603 - trusted local scancel invocation
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
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e
