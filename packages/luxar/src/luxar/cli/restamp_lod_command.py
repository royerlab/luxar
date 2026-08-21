"""The ``luxar restamp-lod`` command — re-derive stored LOD switch thresholds.

A thin Typer layer over :mod:`luxar.io.lod_restamp`; every decision about what
may change and what must not lives there. Registered onto the root app by
:func:`register_restamp_lod_command`, mirroring ``optimise_command.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from arbol import aprint

from ..io.lod_restamp import restamp_lod_store


def register_restamp_lod_command(app: typer.Typer) -> None:
    """Attach the ``restamp-lod`` command to ``app``."""

    @app.command(name="restamp-lod")
    def restamp_lod(
        store: Path = typer.Argument(
            ..., help="Existing .luxar.zarr / .gsplats.zarr directory to upgrade"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Report the old→new ladders; write nothing"
        ),
        group: Optional[List[str]] = typer.Option(
            None,
            "--group",
            help=(
                "Restrict the pass to this kind=lod group path (repeatable); "
                "an unmatched path is an error"
            ),
        ),
    ) -> None:
        """Re-derive legacy LOD thresholds under the screen-area selector.

        An attrs-only pass, in place: no chunk data moves and no array is
        opened. Every ``kind=lod`` group still on the legacy ``coverage``
        diagonal metric (or carrying no ``selector`` at all, which means the
        same) gets its per-child ``coverage_fraction`` thresholds re-derived by
        screen-occupancy halving and its group stamped ``screen-area``. A group
        already on ``screen-area`` is skipped, so a second run changes nothing —
        not even the ``content_hash``.

        **This is an explicit opt-in, and it may override a deliberate choice.**
        An authored ``coverage_fractions=[...]`` list and a legacy derived ladder
        are indistinguishable on disk, which is why nothing does this
        automatically. The per-group old→new ladder is printed for exactly that
        reason — use ``--dry-run`` first, and ``--group`` to restrict the pass.

        When anything changes, the store's ``content_hash`` is restamped and the
        metadata re-consolidated, so a warm viewer cache invalidates on an
        attrs-only edit it would otherwise never notice. The result is then read
        back — from both the per-node documents and the consolidated index — and
        verified.

        Exits 1 if any group was left alone for a reason worth acting on: a
        selector outside the vocabulary (migrate with ``luxar gsplat
        migrate-format`` first), a finest level whose element count the store
        does not record, or a re-verification residual. Groups that WERE
        restamped are still written in that case; nothing is silently ignored.
        """
        if not store.exists():
            aprint(f"❌ Error: path does not exist: {store}")
            raise typer.Exit(1)

        try:
            report = restamp_lod_store(
                store, dry_run=dry_run, groups=group if group else None
            )
        except ValueError as e:
            aprint(f"❌ {e}")
            raise typer.Exit(1) from e
        except Exception as e:
            aprint(f"❌ Error restamping {store}: {e}")
            raise typer.Exit(1) from e

        if not report.clean:
            raise typer.Exit(1)
