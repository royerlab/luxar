"""Periodic per-box progress lines for content-tiled (planned) fits.

A content-tiled fit runs every box's inner ``fit_gaussian_splats`` quietly —
the full per-fit narration repeated for hundreds of boxes would bury the run —
which left a long fit printing nothing between "Fitting N boxes" and the end.
:class:`BoxProgress` is the one progress stream that replaces it: a start line
per box, a few iteration lines (through the fitter's ``iter_callback`` hook),
and a completion line carrying the box's PSNR, elapsed time and a
budget-weighted ETA for the whole run. Boxes reloaded from a checkpoint are
recorded as resumed: they count towards the done total but not towards the
timing or the remaining budget.

Every line goes through :func:`~luxar.utils.verbosity.aprint_heartbeat`, so it
still prints while the driver silences the inner fit with
:func:`~luxar.utils.verbosity.quiet_narration`.
"""

from __future__ import annotations

import contextlib
import math
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from luxar.utils.verbosity import aprint_heartbeat

#: Iteration lines per box: one at each of these fractions of ``n_iters``.
ITER_REPORT_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)


def format_duration(seconds: float) -> str:
    """Compact ``1h02m`` / ``3m07s`` / ``42s`` rendering of a duration."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class OncePerRunWarnings:
    """Re-emit each distinct warning of a run's box fits once, not once per box.

    Every box fit repeats the same per-fit warnings (the CPU-fallback ones); a
    hundred boxes would print them a hundred times.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[type, str]] = set()

    @contextmanager
    def once(self) -> Iterator[None]:
        """Collect the block's warnings, then re-issue the ones not seen yet.

        Re-issued through the normal filters (so ``-W error`` and
        ``pytest.warns`` still see them), keyed by ``(category, message)``. A
        filter that turns a warning into an error still raises inside the
        block, as before. Warnings recorded before the block RAISES are still
        re-issued — but then a re-issue that itself raises is suppressed, so it
        never masks the original exception.
        """
        pending: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                try:
                    yield
                finally:
                    pending = list(caught)
        except BaseException:
            # Re-issued OUTSIDE the recording scope, or they would be recorded
            # again instead of displayed.
            with contextlib.suppress(Exception):
                self._reissue(pending)
            raise
        self._reissue(pending)

    def _reissue(self, pending: list[warnings.WarningMessage]) -> None:
        """Re-issue each not-yet-seen warning of ``pending`` through the filters."""
        for w in pending:
            key = (w.category, str(w.message))
            if key in self._seen:
                continue
            self._seen.add(key)
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


class BoxProgress:
    """Track a planned fit's boxes and print their progress.

    Args:
        budgets: Seed budget (the padded-crop ``_scaled_budget``) of every box
            this run handles, keyed by plan-box index — including boxes already
            known to be resumed.
        resumed: Boxes known up front to come from a checkpoint (the parallel
            driver); the sequential driver reports them one by one through
            :meth:`box_resumed` instead.

    Lines name a box by its plan index (``box 7``) — the number a ``--plan-box``
    worker, a checkpoint and the plan JSON use — and count progress against the
    boxes this run handles (``3/12``). The ETA weighs boxes by budget (fit cost
    grows with the splat count), so it is a throughput estimate that also holds
    for concurrent workers.
    """

    def __init__(self, budgets: Mapping[int, int], resumed: Iterable[int] = ()) -> None:
        self._budgets = {int(i): max(1, int(b)) for i, b in budgets.items()}
        self._remaining_budget = sum(self._budgets.values())
        self._fitted_budget = 0
        self._n_done = 0
        self._n_fitted = 0
        self._t0 = time.perf_counter()
        for i in resumed:
            self._record_resumed(int(i))

    @property
    def total(self) -> int:
        """Boxes this run handles (fitted + resumed)."""
        return len(self._budgets)

    def elapsed(self) -> float:
        """Seconds of fitting so far (resume loads before the first fit excluded)."""
        return time.perf_counter() - self._t0

    def eta(self) -> float:
        """Budget-weighted estimate of the seconds left (``nan`` before any fit)."""
        if self._fitted_budget <= 0:
            return float("nan")
        return self.elapsed() * self._remaining_budget / self._fitted_budget

    def _record_resumed(self, box: int) -> None:
        self._n_done += 1
        self._remaining_budget -= self._budgets.get(box, 0)
        if self._n_fitted == 0:
            self._t0 = time.perf_counter()  # loading is not fitting throughput

    def box_resumed(self, box: int) -> None:
        """Record box ``box`` as reloaded from a checkpoint, and say so."""
        self._record_resumed(int(box))
        aprint_heartbeat(
            f"  box {box} resumed from checkpoint ({self._n_done}/{self.total})"
        )

    def box_started(self, box: int) -> None:
        """Print box ``box``'s start line (budget and its position in the run)."""
        aprint_heartbeat(
            f"  box {box} ({self._n_done + 1}/{self.total}): fitting, "
            f"budget {self._budgets.get(int(box), 0):,} splats"
        )

    def iter_callback(
        self, box: int, n_iters: Optional[int]
    ) -> Callable[[int, Any, dict[str, Any]], None]:
        """An ``iter_callback`` printing box ``box``'s iteration at fixed fractions.

        The fitter calls it on its eval cadence; a line is printed the first time
        each of :data:`ITER_REPORT_FRACTIONS` of ``n_iters`` is reached.
        """
        total = int(n_iters) if n_iters else 0
        marks = sorted({max(1, int(total * f)) for f in ITER_REPORT_FRACTIONS})
        started = time.perf_counter()
        label = f"box {box}"

        def _cb(it: int, _pred: Any, info: dict[str, Any]) -> None:
            if total <= 0 or not marks or it < marks[0]:
                return
            while marks and it >= marks[0]:
                marks.pop(0)
            box_elapsed = time.perf_counter() - started
            box_eta = box_elapsed * (total - it) / max(it, 1)
            loss = info.get("loss", float("nan"))
            aprint_heartbeat(
                f"  {label}: iter {it}/{total}, loss {loss:.3g}, "
                f"{format_duration(box_elapsed)} elapsed, "
                f"box ETA {format_duration(box_eta)}"
            )

        return _cb

    def box_done(
        self,
        box: int,
        *,
        n_kept: Optional[int] = None,
        psnr_db: Optional[float] = None,
        failed: bool = False,
        detail: Optional[str] = None,
    ) -> None:
        """Record box ``box`` as finished (or failed) and print its line."""
        budget = self._budgets.get(int(box), 0)
        self._n_done += 1
        self._n_fitted += 1
        self._fitted_budget += budget
        self._remaining_budget -= budget
        state = "FAILED" if failed else "done"
        details: list[str] = [detail] if detail else []
        if n_kept is not None:
            details.append(f"kept {n_kept:,} splats")
        if psnr_db is not None and math.isfinite(psnr_db):
            details.append(f"PSNR {psnr_db:.2f} dB (padded crop)")
        details.append(f"{format_duration(self.elapsed())} elapsed")
        if self._n_done < self.total:
            details.append(f"ETA {format_duration(self.eta())}")
        aprint_heartbeat(
            f"  box {box} {state} ({self._n_done}/{self.total}): " + ", ".join(details)
        )
