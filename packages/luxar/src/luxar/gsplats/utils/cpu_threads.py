"""Automatic torch intra-op thread count for CPU fits and calibrations.

The hatch environments pin ``OMP_NUM_THREADS=1`` / ``MKL_NUM_THREADS=1`` (the
test suite is thousands of tiny tensor ops and loses several-fold to fork-join
overhead otherwise — see the note in ``pyproject.toml``). That pin also reaches
``hatch run luxar gsplat fit --device cpu``, where one thread is the wrong
default: a 128x256x256 / 4k-splat CPU fit takes 4.6 s/iter on one thread and
2.6 s/iter on 4-8 (scaling stops there; 16 threads is SLOWER, 3.4 s/iter, with
heavy system time).

The CLI ``fit`` and ``cal`` commands therefore call :func:`cpu_threads_for`,
which widens torch to ``min(physical cores, 8)`` for the duration of the
command — but only when torch still has one thread:

* ``LUXAR_CPU_THREADS=N`` (N >= 1) always wins: torch runs N threads.
* ``LUXAR_CPU_THREADS=0`` disables the adjustment (torch is left alone). The
  test suite sets this so its single-thread pin survives CLI-level tests.
* If torch already runs more than one thread (including when an OMP/MKL export
  took effect), it is left alone. An OMP-only export may leave torch at one
  thread under the hatch MKL pin; in that case the CLI still widens it. To
  explicitly run one thread, set ``LUXAR_CPU_THREADS=1``.
* A tile/box WORKER (``fit --tile`` / ``--plan-box``) is left alone: it is one
  of N concurrent processes whose count the parent sized against the CPU count.

Scope is the CLI only, and the previous count is restored on exit, so a
library caller (``fit_gaussian_splats`` from Python) never has process-global
torch state changed under it, and an in-process CLI invocation (Typer's
``CliRunner``) leaves the interpreter as it found it.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional

#: Upper bound on the automatic width. Measured on an M4 Max: 4 and 8 threads
#: tie, 16 is slower than 4 (system time dominates), so more never helps.
AUTO_CPU_THREADS_CAP = 8

#: Environment variable that overrides the automatic choice (``0`` = disable).
CPU_THREADS_ENV = "LUXAR_CPU_THREADS"

def physical_cpu_count() -> int:
    """Physical cores available to this process (logical count as fallback)."""
    from .device import _effective_cpu_count

    physical: Optional[int] = None
    try:
        # Looked up dynamically: psutil is optional and ships no type stubs.
        psutil = importlib.import_module("psutil")
        physical = psutil.cpu_count(logical=False)
    except ImportError:  # pragma: no cover - psutil is not a declared dependency
        # Logical count instead; the AUTO_CPU_THREADS_CAP bound does most of
        # the work anyway (8 is below the logical count of any machine with
        # more than 8 physical cores).
        physical = None
    available = _effective_cpu_count()
    return max(1, min(physical or available, available))


def _parse_override(value: Optional[str]) -> Optional[int]:
    """``LUXAR_CPU_THREADS`` as a non-negative int, or None if unset/invalid."""
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def resolve_cpu_threads(
    env: Mapping[str, str],
    *,
    current: int,
    physical_cores: int,
    worker: bool = False,
) -> Optional[int]:
    """The torch thread count to apply for a CPU run, or None to leave it.

    Pure decision function (see the module docstring for the rules).
    """
    override = _parse_override(env.get(CPU_THREADS_ENV))
    if override is not None:
        return override if override > 0 else None
    if worker or current != 1:
        return None
    return max(1, min(physical_cores, AUTO_CPU_THREADS_CAP))


def _runs_on_cpu(device: Optional[str]) -> bool:
    from .device import resolve_torch_device

    try:
        return resolve_torch_device(device).type == "cpu"
    except (RuntimeError, ValueError):
        return False


@contextmanager
def cpu_threads_for(device: Optional[str], *, worker: bool = False) -> Iterator[None]:
    """Widen torch's intra-op threads for a CPU run; restore them on exit.

    ``device`` is the CLI ``--device`` value (``None``/``"auto"`` auto-resolve).
    A no-op for GPU runs and whenever :func:`resolve_cpu_threads` says so.
    """
    import torch

    previous = torch.get_num_threads()
    threads = None
    if _runs_on_cpu(device):
        threads = resolve_cpu_threads(
            os.environ,
            current=previous,
            physical_cores=physical_cpu_count(),
            worker=worker,
        )
    if threads is None or threads == previous:
        yield
        return
    from arbol import aprint

    torch.set_num_threads(threads)
    aprint(f"CPU: using {threads} threads (set {CPU_THREADS_ENV} to change)")
    try:
        yield
    finally:
        torch.set_num_threads(previous)
