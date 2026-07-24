"""Deterministic teardown for long-lived child processes (stdlib-only).

The ``luxar demo run`` path spawns a tree of subprocesses — the demo script,
which in turn spawns ``luxar serve`` (uvicorn). On Ctrl-C, uvicorn's graceful
shutdown can hang on the browser's keep-alive sockets, and a plain
``subprocess.run`` parent exits without reaping its child, orphaning the server
on its port. :func:`run_child_process` fixes that: it *owns* the child's
lifecycle and, on any exit path, escalates SIGINT → SIGTERM → SIGKILL until the
child (and its whole process group, when isolated) is gone.

Kept in ``luxar.utils`` — not ``luxar.cli`` — so importing it does not drag in
uvicorn/fastapi (``luxar.cli.__init__`` → ``main``). Both the CLI (L0
``demo run``) and the demo helper (L1 ``launch_viewer``) import it cleanly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Callable, Optional, Sequence, TypeVar

from arbol import aprint

# POSIX process-group control. Absent on Windows, where we degrade to a
# single-PID teardown (no ``start_new_session`` isolation).
_CAN_KILLPG = hasattr(os, "killpg") and hasattr(os, "getpgid")


def run_child_process(
    cmd: Sequence[str],
    *,
    label: str = "child process",
    isolate_group: bool = True,
    interrupt_timeout: float = 5.0,
    term_timeout: float = 3.0,
) -> int:
    """Spawn ``cmd``, wait for it, and own its teardown on every exit path.

    Args:
        cmd: Command + args to run (passed to :class:`subprocess.Popen`).
        label: Human name used in the "Stopping …" message on interrupt.
        isolate_group: When True (POSIX), the child becomes its own session /
            process-group leader (``start_new_session``), so a terminal Ctrl-C
            does *not* reach it — this parent catches ``KeyboardInterrupt`` and
            tears down the child's entire group. Use this at the outermost
            owner (``demo run`` / ``run-all``). When False, the child stays in
            this process's group (so an ancestor's group-kill still cascades to
            it) and teardown targets the single child PID.
        interrupt_timeout: Seconds to wait after SIGINT before escalating to
            SIGTERM.
        term_timeout: Seconds to wait after SIGTERM before escalating to
            SIGKILL (also the reap timeout for the direct child).

    Returns:
        The child's exit code (signal death mapped to ``128 + n``), or ``130``
        if this call was interrupted with Ctrl-C.
    """
    isolate = isolate_group and _CAN_KILLPG
    proc = subprocess.Popen(list(cmd), start_new_session=isolate)
    # A session leader's process-group id equals its pid; capture it so the
    # group can still be signalled after the direct child has been reaped
    # (grandchildren may outlive it). The whole body below the spawn is inside
    # the try/finally so a signal can never skip teardown and orphan the child.
    pgid = proc.pid if isolate else None
    prev_handlers: dict[int, Any] = {}
    try:
        # SIGINT (Ctrl-C) already raises KeyboardInterrupt via Python's default
        # handler. Route SIGTERM (`kill <pid>`) and SIGHUP (terminal closed)
        # into the same path so those tear the child down too instead of
        # killing us outright and orphaning it.
        prev_handlers = _install_stop_handlers()
        try:
            proc.wait()
            return _exit_code(proc.returncode)
        except KeyboardInterrupt:
            aprint(f"\n🛑 Stopping {label}…")
            return 130
    finally:
        # Tear down BEFORE restoring handlers, so a second Ctrl-C / SIGTERM
        # arriving mid-teardown still routes into the escalation (which then
        # jumps straight to SIGKILL) rather than killing us and orphaning the
        # child. Restore always runs even if teardown raises.
        try:
            _teardown(proc, pgid, interrupt_timeout, term_timeout)
        finally:
            _restore_handlers(prev_handlers)


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    """Signal handler that funnels SIGTERM/SIGHUP into the KeyboardInterrupt path."""
    raise KeyboardInterrupt


def _install_stop_handlers() -> dict[int, Any]:
    """Install SIGTERM/SIGHUP handlers that raise KeyboardInterrupt.

    Returns the previous handlers for restoration. A no-op (empty dict) off the
    main thread or where a signal is already ignored (respecting an explicit
    ``SIG_IGN``, e.g. a job-control-backgrounded process).
    """
    previous: dict[int, Any] = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old = signal.getsignal(sig)
            # SIG_IGN: respect an explicit ignore (e.g. job-control background).
            # None: handler was installed from C and is not restorable via
            # signal.signal(sig, None) — leave it untouched rather than leak
            # our handler permanently.
            if old is signal.SIG_IGN or old is None:
                continue
            signal.signal(sig, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            continue  # not the main thread / unsupported platform
        previous[sig] = old
    return previous


def _restore_handlers(previous: dict[int, Any]) -> None:
    for sig, old in previous.items():
        _safe(signal.signal, sig, old)


def _exit_code(returncode: Optional[int]) -> int:
    """Normalize a Popen returncode; map signal death (-n) to ``128 + n``."""
    if returncode is None:
        return 0
    return returncode if returncode >= 0 else 128 + (-returncode)


# SIGKILL is absent on Windows; fall back to SIGTERM there so the escalation
# ladder never raises AttributeError on the single-PID degrade path.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _teardown(
    proc: "subprocess.Popen[bytes]",
    pgid: Optional[int],
    interrupt_timeout: float,
    term_timeout: float,
) -> None:
    """Escalate SIGINT → SIGTERM → SIGKILL until the target is gone.

    On the clean-exit path the direct child is already reaped, so ``alive()``
    only fires for lingering grandchildren; because a process group cannot be
    reused while any member is alive, the pgid we hold cannot have been
    recycled by then, so the kill targets our own subtree only.
    """
    if pgid is not None:

        def send(sig: int) -> None:
            _killpg(pgid, sig)

        def alive() -> bool:
            return _killpg(pgid, 0)

    else:

        def send(sig: int) -> None:
            _safe(proc.send_signal, sig)

        def alive() -> bool:
            return proc.poll() is None

    try:
        for sig, grace in (
            (signal.SIGINT, interrupt_timeout),
            (signal.SIGTERM, term_timeout),
            (_SIGKILL, None),
        ):
            if not alive():
                break
            send(sig)
            if grace is None:
                break
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and alive():
                time.sleep(0.05)
    except KeyboardInterrupt:
        # Impatient second Ctrl-C (or a SIGTERM) mid-escalation: skip the
        # graceful ladder and hard-kill now, so we never orphan the child by
        # bailing out of teardown early.
        if alive():
            send(_SIGKILL)

    # Reap our direct child so it does not linger as a zombie.
    _safe(proc.wait, timeout=term_timeout)


def _killpg(pgid: int, sig: int) -> bool:
    """Signal a process group; return False if the group is already gone."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


_T = TypeVar("_T")


def _safe(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> Optional[_T]:
    """Call ``fn`` swallowing any exception (teardown must never raise)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None
