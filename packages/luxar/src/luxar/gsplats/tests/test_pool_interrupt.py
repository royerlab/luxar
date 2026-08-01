# test_pool_interrupt.py
"""Ctrl-C handling for the three subprocess fan-out pools (issue #736).

Before the fix, a ``KeyboardInterrupt`` during the ``as_completed`` loop could
not escape the bare ``with ThreadPoolExecutor(...)`` block until ``__exit__``
ran ``shutdown(wait=True)`` with the default ``cancel_futures=False`` — so every
still-queued task ran to completion first, each freed worker thread spawning a
fresh ``luxar gsplat fit`` subprocess.  On a large run Ctrl-C did nothing for
hours.

These tests assert the observable contract without delivering a real OS SIGINT
(flaky): a subprocess launcher that raises ``KeyboardInterrupt`` on its first
call must (a) propagate the interrupt out of the pool function and (b) NOT keep
launching subprocesses for the still-queued tasks.  All three sites use
``max_workers/jobs=1`` so the assertion is deterministic — a single worker plus
the shared stop flag + ``cancel_futures=True`` means at most a couple of tasks
can slip through before the queue is drained, never all of them.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from luxar.gsplats.batch.task_pool import cancel_pool_on_interrupt, run_task_pool


class _FakeProc:
    """Stand-in for a ``subprocess.CompletedProcess`` (success)."""

    returncode = 0
    stdout = ""
    stderr = ""


def _ki_on_first_spy() -> tuple:
    """Return ``(fake_run, calls)`` — a ``subprocess.run`` spy that raises
    ``KeyboardInterrupt`` on its first call and records every invocation.

    Later (racy) calls return a success ``_FakeProc`` so the count reflects how
    many subprocesses were actually launched, not how many failed.
    """
    calls: list = []
    lock = threading.Lock()

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            calls.append(cmd)
            first = len(calls) == 1
        if first:
            raise KeyboardInterrupt
        return _FakeProc()

    return fake_run, calls


# ── shared helper ────────────────────────────────────────────────────────────


def test_cancel_pool_on_interrupt_sets_flag_and_shuts_down() -> None:
    """The helper sets the stop flag and shuts the executor down so no further
    tasks can be submitted (``cancel_futures=True`` is exercised by the loop)."""
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=2)
    try:
        cancel_pool_on_interrupt(ex, stop)
        assert stop.is_set()
        # Executor is shut down → submitting a new task raises.
        with pytest.raises(RuntimeError):
            ex.submit(lambda: None)
    finally:
        ex.shutdown(wait=False)


# ── run_task_pool (batch local fan-out) ──────────────────────────────────────


def test_task_pool_interrupt_stops_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run, calls = _ki_on_first_spy()
    monkeypatch.setattr("luxar.gsplats.batch.task_pool.subprocess.run", fake_run)

    n = 8
    with pytest.raises(KeyboardInterrupt):
        run_task_pool(
            list(range(n)),
            max_workers=1,
            argv_builder=lambda k: [sys.executable, "-c", "pass"],
        )
    # Before the fix all 8 queued tasks would have launched.
    assert 1 <= len(calls) < n


# ── fit_tiled_parallel (uniform tiled --jobs N) ──────────────────────────────


def test_fit_tiled_parallel_interrupt_stops_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from luxar.gsplats import fit_tiled_parallel as ftp

    fake_run, calls = _ki_on_first_spy()
    monkeypatch.setattr(ftp.subprocess, "run", fake_run)

    n = 8
    with pytest.raises(KeyboardInterrupt):
        ftp.fit_tiled_parallel(
            num_tiles=n,
            jobs=1,
            tmp_dir=tmp_path / "tiles",
            worker_cmd_builder=lambda i, m, out: [sys.executable, "-c", "pass"],
            volume_shape=(16, 16, 16),
            tile_size=16,
            overlap=0,
            progressive=False,
            cull_retention=None,
            verbose=False,
        )
    assert 1 <= len(calls) < n


# ── fit_planned_parallel (content plan --jobs N) ─────────────────────────────


def _toy_plan(n_boxes: int = 8, budget: int = 100, width: int = 16):
    """A FitPlan tiling x into ``n_boxes`` budgeted columns (no scan needed)."""
    from luxar.gsplats.planner import FitPlan, PlanBox

    boxes = [
        PlanBox(
            box=[0, width, 0, width, j * width, (j + 1) * width],
            n_features=10,
            budget=budget,
        )
        for j in range(n_boxes)
    ]
    return FitPlan(
        volume_shape=[width, width, n_boxes * width],
        boxes=boxes,
        overlap=0,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=16,
        density={"saturation_cap": 10_000},
    )


def test_fit_planned_parallel_interrupt_stops_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``planner/__init__`` re-exports the ``fit_planned_parallel`` *function*
    # under this name, shadowing the submodule attribute — fetch the real module
    # object from sys.modules to patch its ``subprocess``.
    import luxar.gsplats.planner.fit_planned_parallel  # noqa: F401  (ensure imported)

    fpp = sys.modules["luxar.gsplats.planner.fit_planned_parallel"]

    fake_run, calls = _ki_on_first_spy()
    monkeypatch.setattr(fpp.subprocess, "run", fake_run)

    n = 8
    plan = _toy_plan(n_boxes=n)
    with pytest.raises(KeyboardInterrupt):
        fpp.fit_planned_parallel(
            plan,
            jobs=1,
            tmp_dir=tmp_path / "boxes",
            worker_cmd_builder=lambda i, out: [sys.executable, "-c", "pass"],
            verbose=False,
        )
    assert 1 <= len(calls) < n
