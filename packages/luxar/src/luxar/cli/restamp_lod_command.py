"""The ``luxar restamp-lod`` command — re-derive stored LOD switch thresholds.

A thin Typer layer over :mod:`luxar.io.lod_restamp`; every decision about what
may change and what must not lives there. Registered onto the root app by
:func:`register_restamp_lod_command`, mirroring ``optimize_command.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from arbol import aprint

from ..io.lod_restamp import restamp_lod_store
from ._traceback import exit_with_error


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
                "an unmatched path is an error. Spell the store ROOT '/' — the "
                "only way to name the ladder of a .gsplats.zarr whose root IS "
                "the kind=lod group"
            ),
        ),
        anchor: Optional[float] = typer.Option(
            None,
            "--anchor",
            help=(
                "Re-derive whole-object ladders at this finest-level screen-area "
                "fraction (0 < anchor <= 1), both legacy ladders being migrated "
                "and ladders already stamped screen-area; partition-bound ladders "
                "stay anchored at 1"
            ),
        ),
    ) -> None:
        """Re-derive LOD thresholds under the screen-area selector.

        An attrs-only pass, in place: the ladder rewrite moves no chunk data and
        opens no array. Every ``kind=lod`` group still on the legacy ``coverage``
        diagonal metric (or carrying no ``selector`` at all, which means the
        same) gets its per-child ``coverage_fraction`` thresholds re-derived by
        screen-occupancy halving and its group stamped ``screen-area``. A group
        already on ``screen-area`` is skipped by default, so a second run changes
        nothing — not even the ``content_hash``. ``--anchor`` explicitly
        sets the requested finest-level area fraction for every whole-object
        ladder it processes, both legacy and already ``screen-area``;
        partition-bound ladders remain at ``1.0``.

        **This is an explicit opt-in, and it may override a deliberate choice.**
        An authored ``coverage_fractions=[...]`` list is indistinguishable from a
        derived ladder on disk, including one already stamped ``screen-area``,
        which is why nothing does this automatically. The per-group old→new
        ladder is printed for exactly that reason — use ``--dry-run`` first, and
        ``--group`` to restrict the pass.

        When anything changes, the store's ``content_hash`` is restamped and the
        metadata re-consolidated, so a warm viewer cache invalidates on an
        attrs-only edit it would otherwise never notice. **That restamp is the
        one expensive step:** a SCENE's digest covers array values, so it reads
        every array in the store once (minutes on a very large scene, and linear
        in its size); a standalone ``.gsplats.zarr`` gets a metadata-only stamp
        and stays instant. Nothing is read at all under ``--dry-run`` or when no
        ladder needed changing. The result is then read back — from both the
        per-node documents and the consolidated index — and verified. A store
        that carried NO index is not given one: ``is_consolidated`` is how
        ``batch-fit`` tells a finished tile from an interrupted one.

        A failed write is undone: every attr already rewritten is restored,
        ``content_hash`` included — put back as the store had it, never
        recomputed — and the error reported, rather than leaving a ladder whose
        thresholds and ``selector`` disagree about their units. The index is
        rebuilt only if the failed run had rewritten the store ROOT, which is
        the write that invalidates it.

        Exits 1 if any group was left alone for a reason worth acting on: a
        selector outside the vocabulary (migrate with ``luxar gsplat
        migrate-format`` first), a finest level whose element count the store
        does not record, a ladder child that carries a threshold but cannot be
        classified as a level, or a re-verification residual. Groups that WERE
        restamped are still written in that case; nothing is silently ignored.
        It also exits 1 when ladders WERE rewritten but the store carries no
        ``content_hash`` to restamp — the rewrite landed, but no warm viewer
        cache will notice it until the store is republished under a new URL
        prefix.
        """
        if not store.exists():
            aprint(f"❌ Error: path does not exist: {store}")
            raise typer.Exit(1)

        try:
            report = restamp_lod_store(
                store,
                dry_run=dry_run,
                groups=group if group else None,
                finest_anchor=anchor,
            )
        except ValueError as e:
            aprint(f"❌ {e}")
            raise typer.Exit(1) from e
        except Exception as e:
            exit_with_error(f"❌ Error restamping {store}: {e}", e)

        if not report.clean:
            raise typer.Exit(1)
