"""The ``luxar optimise`` command — re-chunk an existing store for streaming.

A thin Typer layer over :mod:`luxar.io.optimise`; every decision about what may
change and what must not lives there. Registered onto the root app by
:func:`register_optimise_command`, mirroring ``info_command.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from .._zarr_compat import close, open_group
from ..io.optimise import (
    CHUNK_PROFILES,
    OptimisePlan,
    ensure_luxar_store,
    optimise_store,
    plan_optimisation,
    resolve_target_bytes,
)
from ._traceback import exit_with_error


def _report_dry_run(plan: OptimisePlan, path: Path) -> None:
    """Print what would change, most-improved array first."""
    aprint(f"\n🔎 Dry run: {path}")
    aprint(f"  Target chunk: {plan.target_bytes / 1024:.0f} KB")
    changed = sorted(
        (a for a in plan.arrays if a.rechunked),
        key=lambda a: a.source_n_chunks - a.target_n_chunks,
        reverse=True,
    )
    if changed:
        aprint("  Would re-chunk:")
        for a in changed:
            aprint(
                f"    {a.path}: {a.source_chunks} → {a.target_chunks} "
                f"({a.source_n_chunks} → {a.target_n_chunks} chunks, "
                f"{a.target_chunk_bytes / 1024:.1f} KB each"
                + (f", atom {a.atom}" if a.atom else "")
                + ")"
            )
    else:
        aprint("  Nothing to re-chunk — every array is already at or above target.")
    skipped = [a for a in plan.arrays if not a.rechunked]
    if skipped:
        reasons: dict[str, int] = {}
        for a in skipped:
            reasons[a.skip_reason] = reasons.get(a.skip_reason, 0) + 1
        summary = ", ".join(f"{n}× {r}" for r, n in sorted(reasons.items()))
        aprint(f"  Left alone: {len(skipped)} arrays ({summary})")
    aprint(
        f"  Totals: {plan.source_n_chunks} → {plan.target_n_chunks} chunk files "
        f"({plan.n_rechunked}/{len(plan.arrays)} arrays changed)"
    )
    aprint("  Nothing was written.")


def _plan_and_report(
    source: Path, budget: int, profile: Optional[str], generic: bool
) -> None:
    """The whole ``--dry-run`` path: same validation as a real run, no writing.

    Takes the ``--generic`` gate too. Without it ``luxar optimise foreign.zarr
    --dry-run`` printed a full plan for a store the identical non-dry-run
    command refuses — a plan the user cannot act on. And the group is CLOSED:
    a ``.zarr.zip`` source otherwise keeps its ``ZipStore`` handle open, which
    is a ``ResourceWarning`` (fatal under the ``-W error`` suites).
    """
    try:
        root = open_group(source, mode="r")
    except Exception as e:
        exit_with_error(f"❌ Error reading {source}: {e}", e)
    try:
        ensure_luxar_store(source, root, generic=generic)
        plan = plan_optimisation(root, target_bytes=budget, profile=profile)
    except ValueError as e:
        aprint(f"❌ {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        exit_with_error(f"❌ Error reading {source}: {e}", e)
    finally:
        close(root)
    _report_dry_run(plan, source)


def register_optimise_command(app: typer.Typer) -> None:
    """Attach the ``optimise`` command to ``app``."""

    @app.command()
    def optimise(
        source: Path = typer.Argument(..., help="Existing .zarr store to re-chunk"),
        output: Optional[Path] = typer.Argument(
            None, help="Destination store (omit only with --dry-run)"
        ),
        target_kb: Optional[int] = typer.Option(
            None, "--target-kb", help="Target chunk payload in KB"
        ),
        target_bytes: Optional[int] = typer.Option(
            None, "--target-bytes", help="Target chunk payload in bytes"
        ),
        profile: Optional[str] = typer.Option(
            None,
            "--profile",
            help=(
                "Preset target: "
                + ", ".join(f"{k} ({v // 1024} KB)" for k, v in CHUNK_PROFILES.items())
            ),
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Report the plan; write nothing"
        ),
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Re-read the output and compare every array and payload file",
        ),
        overwrite: bool = typer.Option(
            False, "--overwrite", help="Replace an existing output store"
        ),
        generic: bool = typer.Option(
            False, "--generic", help="Allow a plain (non-Luxar) zarr store"
        ),
    ) -> None:
        """Re-chunk a zarr store for streaming — values stay bit-identical.

        Only zarr chunk shapes change. dtype, codecs, ``fill_value``, every
        attribute, the zarr format version and the spatial-index grid are all
        preserved; each new chunk is a whole multiple of its node's
        ``chunk_size`` atom, and no array is ever chunked SMALLER than it
        already is.

        The output carries a fresh ``content_hash`` and a ``chunk_layout``
        summary attr, because a re-chunked store with an unchanged hash would
        let a warm viewer cache serve chunks whose keys have moved.
        """
        if not source.exists():
            aprint(f"❌ Error: path does not exist: {source}")
            raise typer.Exit(1)

        try:
            budget = resolve_target_bytes(
                target_bytes=target_bytes, target_kb=target_kb, profile=profile
            )
        except ValueError as e:
            aprint(f"❌ {e}")
            raise typer.Exit(1)

        if dry_run:
            if output is not None:
                aprint("❌ --dry-run writes nothing; drop the output argument")
                raise typer.Exit(1)
            _plan_and_report(source, budget, profile, generic)
            return

        if output is None:
            aprint("❌ An output path is required (or pass --dry-run)")
            raise typer.Exit(1)

        try:
            optimise_store(
                source,
                output,
                target_bytes=budget,
                profile=profile,
                overwrite=overwrite,
                verify=verify,
                generic=generic,
            )
        except typer.Exit:
            raise
        except Exception as e:
            exit_with_error(f"❌ Error optimising {source}: {e}", e)
