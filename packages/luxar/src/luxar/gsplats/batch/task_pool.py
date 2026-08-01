# task_pool.py
"""Generic subprocess task pool — the shared spawn loop for local fan-out.

Runs a list of arbitrary task keys as concurrent subprocesses, ``max_workers`` at
a time, via a :class:`~concurrent.futures.ThreadPoolExecutor` (each thread just
waits on its child).  Two hooks beyond a plain ``subprocess.run`` loop make it
reusable for the local batch-fit runner:

- ``env_builder`` injects per-task environment (e.g. ``CUDA_VISIBLE_DEVICES`` to
  pin a worker to a specific GPU) — the seam the single-GPU
  :func:`luxar.gsplats.fit_tiled_parallel.fit_tiled_parallel` loop lacks.
- ``skip_if`` lets a task be skipped without launching a subprocess (resume:
  an output already on disk).

The pool itself never raises on a worker failure — it returns every
:class:`TaskResult` and lets the caller decide the failure policy (the
tile-merge callers and the file-output local runner want different handling).
"""

from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Sequence, TypeVar

# Task keys are any hashable; the generic ``T`` lets callers keep their concrete
# key type (e.g. int task ids) through the callbacks without casts.
T = TypeVar("T", bound=Hashable)


def cancel_pool_on_interrupt(ex: ThreadPoolExecutor, stop: threading.Event) -> None:
    """Abort a running subprocess pool on Ctrl-C (shared by the three fan-out sites).

    Sets the shared ``stop`` flag — so a worker that has *already dequeued* its
    task returns before spawning a subprocess — and shuts the executor down with
    ``cancel_futures=True`` so still-queued tasks are dropped instead of run to
    completion.  Call this from the ``except KeyboardInterrupt`` handler wrapping
    the ``as_completed`` loop, **then re-raise** so the CLI still exits on the
    interrupt.

    Without this, ``ThreadPoolExecutor.__exit__`` runs ``shutdown(wait=True)``
    with the default ``cancel_futures=False`` first, so every queued task — each
    spawning a fresh ``luxar gsplat fit`` subprocess — runs to completion before
    the ``KeyboardInterrupt`` can propagate (issue #736).
    """
    stop.set()
    ex.shutdown(wait=False, cancel_futures=True)


@dataclass
class TaskResult:
    """Outcome of one task.

    ``returncode`` is the subprocess exit code, or ``-1`` when the argv could not
    be built/launched (the error text is in ``output``).  ``skipped`` is True for
    a resume-skip (no subprocess was launched; ``returncode`` is 0).  ``key`` is
    the task key the caller supplied (typed ``Any`` so it indexes the caller's own
    maps without a cast).
    """

    key: Any
    returncode: int
    output: str
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.skipped or self.returncode == 0


def run_task_pool(
    tasks: Sequence[T],
    *,
    max_workers: int,
    argv_builder: Callable[[T], "list[str]"],
    env_builder: Optional[Callable[[T], dict[str, str]]] = None,
    skip_if: Optional[Callable[[T], bool]] = None,
    on_done: Optional[Callable[[TaskResult, int, int], None]] = None,
    verbose: bool = True,
) -> list[TaskResult]:
    """Run ``tasks`` as concurrent subprocesses, at most ``max_workers`` at once.

    Parameters
    ----------
    tasks
        Task keys (any hashable); ``argv_builder``/``env_builder``/``skip_if``
        are called with each key.
    max_workers
        Maximum concurrent subprocesses (clamped to ``>= 1``).
    argv_builder
        ``key -> argv`` for the worker command.
    env_builder
        ``key -> extra env`` merged over ``os.environ`` for that task's
        subprocess.  ``None`` (or returning ``{}``) inherits the parent
        environment unchanged — identical to a plain ``subprocess.run`` call.
    skip_if
        ``key -> bool``; when True the task is skipped (no subprocess) and
        reported as ``TaskResult(skipped=True)`` (used for resume).
    on_done
        ``(result, n_done, n_total)`` progress callback, invoked as each task
        finishes (including skips).
    verbose
        Reserved for symmetry with the callers; progress is delegated to
        ``on_done``.

    Returns
    -------
    list[TaskResult]
        One per task, in completion order.  Never raises on a worker failure.
    """
    total = len(tasks)
    stop = threading.Event()

    def _run(key: T) -> TaskResult:
        # A worker that dequeued this task after a Ctrl-C must not spawn a new
        # subprocess (issue #736): bail before launching anything.
        if stop.is_set():
            return TaskResult(key=key, returncode=-1, output="cancelled before launch")
        if skip_if is not None and skip_if(key):
            return TaskResult(key=key, returncode=0, output="", skipped=True)
        try:
            cmd = [str(c) for c in argv_builder(key)]
            env = None
            if env_builder is not None:
                extra = env_builder(key)
                if extra:
                    # Merge OVER os.environ — never replace it, or the worker
                    # loses PATH / PYTHONPATH / the active venv and `luxar`
                    # won't resolve.
                    env = {**os.environ, **extra}
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        except Exception as exc:  # bad argv → FileNotFoundError/OSError, builder bug
            return TaskResult(
                key=key,
                returncode=-1,
                output=f"failed to build/launch worker: {exc!r}",
            )
        stream = proc.stderr or proc.stdout or ""
        return TaskResult(key=key, returncode=proc.returncode, output=stream)

    results: list[TaskResult] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        done = 0
        try:
            futures = [ex.submit(_run, key) for key in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                done += 1
                results.append(res)
                if on_done is not None:
                    on_done(res, done, total)
        except KeyboardInterrupt:
            # Cancel queued tasks (and signal in-flight workers) BEFORE the
            # ``with`` block's __exit__ would otherwise drain them, then re-raise.
            cancel_pool_on_interrupt(ex, stop)
            raise
    return results
