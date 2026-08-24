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


def can_kill_process_groups() -> bool:
    """True where process-group signalling (``os.killpg``) is available."""
    return _CAN_KILLPG


def run_child_process(
    cmd: Sequence[str],
    *,
    label: str = "child process",
    isolate_group: bool = True,
    interrupt_timeout: float = 5.0,
    term_timeout: float = 3.0,
    on_spawn: Optional[Callable[[int], None]] = None,
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
        on_spawn: Called with the child's PID right after the spawn (when
            isolated, that PID is also the new process-group id). Used by
            ``demo run`` to register the run for ``luxar demo stop``; failures
            are swallowed so bookkeeping can never break the launch.

    Returns:
        The child's exit code (signal death mapped to ``128 + n``), or ``130``
        if this call was interrupted with Ctrl-C.
    """
    isolate = isolate_group and _CAN_KILLPG
    proc = subprocess.Popen(list(cmd), start_new_session=isolate)
    if on_spawn is not None:
        _safe(on_spawn, proc.pid)
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
    """Reinstall the signal handlers captured by :func:`_install_stop_handlers`."""
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


def _ps_proc_table() -> list[tuple[int, int, str, str]]:
    """Best-effort process-table rows from ``ps`` on POSIX systems."""
    rows: list[tuple[int, int, str, str]] = []
    if os.name != "posix":
        return rows
    try:
        out = subprocess.run(  # nosec B603, B607  # fixed argv, no user input
            ["ps", "-axww", "-o", "pid=,pgid=,state=,command="],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return rows
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not parts[2]:
            continue
        try:
            command = parts[3] if len(parts) == 4 else ""
            rows.append((int(parts[0]), int(parts[1]), parts[2][0], command))
        except ValueError:
            continue
    return rows


def proc_table() -> list[tuple[int, int, str, str]]:
    """Best-effort ``(pid, pgid, state, command)`` process-table rows.

    Read ``/proc`` directly where available, avoiding a subprocess and still
    working in slim containers without ``procps``. Fall back to ``ps`` on
    POSIX systems such as macOS. Returns ``[]`` when neither source is
    available — callers must read that as "process table unknown", never as
    "nothing is running".
    """
    rows: list[tuple[int, int, str, str]] = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return _ps_proc_table()
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            # /proc/<pid>/stat is "pid (comm) state ppid pgrp …"; comm can
            # itself contain spaces and parentheses, so split after the LAST
            # ')' — everything before it is untrustworthy for field indexing.
            with open(f"/proc/{pid}/stat", "rb") as handle:
                fields = handle.read().rsplit(b")", 1)[1].split()
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                argv = handle.read()
        except (OSError, IndexError):
            continue  # exited mid-scan, or not ours to read
        try:
            state, pgid = fields[0].decode(), int(fields[2])
        except (IndexError, ValueError, UnicodeDecodeError):
            continue
        command = " ".join(argv.decode("utf-8", "replace").split("\0")).strip()
        rows.append((pid, pgid, state, command))
    return rows


def _group_has_live_member(pgid: int) -> bool:
    """True unless every member of ``pgid`` is an unreaped zombie.

    ``killpg(pgid, 0)`` still succeeds for a group whose processes have all
    exited but whose corpses their parent has not collected yet. A zombie
    holds no port, no memory and no file descriptor, so counting one as "still
    running" would make teardown burn the whole signal ladder and then report
    failure for a group it did kill. Where the process table (or this group)
    cannot be seen, assume the group is alive.
    """
    states = [state for _pid, p, state, _cmd in proc_table() if p == pgid]
    return not states or any(state != "Z" for state in states)


def _group_gone_probe(pgid: int, recheck: float = 0.5) -> Callable[[], bool]:
    """Build a "is this group finished?" probe cheap enough to poll in a loop.

    ``killpg(pgid, 0)`` (microseconds) answers most calls; the process-table
    read that looks past unreaped zombies costs milliseconds, so it is re-run
    at most every ``recheck`` seconds and its answer cached in between.

    Signalling permission is part of the answer: ``EPERM`` means the group is
    alive and owned by someone else, which must never be reported as gone.
    """
    last = 0.0
    zombies_only = False

    def gone() -> bool:
        """True once nothing in the group can still run."""
        nonlocal last, zombies_only
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False  # EPERM &co: alive, and not ours to signal
        now = time.monotonic()
        if now - last >= recheck:
            last, zombies_only = now, not _group_has_live_member(pgid)
        return zombies_only

    return gone


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

    On the Ctrl-C path the direct child is NOT yet reaped (it is reaped below,
    after the escalation), so its corpse sits in the group as a zombie — hence
    the zombie-aware probe, without which every interrupted ``demo run`` would
    sit through the full SIGINT → SIGTERM → SIGKILL ladder before returning.
    """
    if pgid is not None:
        gone = _group_gone_probe(pgid)

        def send(sig: int) -> None:
            """Deliver ``sig`` to the child's whole process group."""
            _killpg(pgid, sig)

        def alive() -> bool:
            """True while a member of the child process group is still running."""
            return not gone()

    else:

        def send(sig: int) -> None:
            """Deliver ``sig`` to the single direct child PID."""
            _safe(proc.send_signal, sig)

        def alive() -> bool:
            """True while the direct child PID is still running."""
            return proc.poll() is None

    _escalate(send, alive, interrupt_timeout, term_timeout)

    # Reap our direct child so it does not linger as a zombie.
    _safe(proc.wait, timeout=term_timeout)


def _escalate(
    send: Callable[[int], None],
    alive: Callable[[], bool],
    interrupt_timeout: float,
    term_timeout: float,
) -> None:
    """Drive the SIGINT → SIGTERM → SIGKILL ladder until ``alive()`` is False."""
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


def terminate_process_group(
    pgid: int,
    *,
    interrupt_timeout: float = 5.0,
    term_timeout: float = 3.0,
) -> bool:
    """Kill a whole process group we did not spawn; True once it is gone.

    The ``luxar demo stop`` counterpart of :func:`run_child_process`'s owned
    teardown: the same graceful escalation, but targeting a group discovered
    after the fact (a forgotten or orphaned demo). There is no direct child to
    reap here — the group's own parent (or init, once orphaned) does that, and
    until it does the corpses linger as zombies the probe must see past — so
    after SIGKILL we only wait briefly for the group to drain before
    reporting. Returns False off POSIX or when the group survives SIGKILL
    (e.g. a process owned by another user, which we cannot signal at all).
    """
    if not _CAN_KILLPG:
        return False
    # killpg(0, sig) signals the CALLER'S own group and negative/1 values are
    # kill(2) wildcards or undefined — never forward them, whatever a corrupt
    # registry entry claims.
    if pgid <= 1:
        return False
    gone = _group_gone_probe(pgid)
    if gone():
        return True  # already gone

    def send(sig: int) -> None:
        """Deliver ``sig`` to the target group."""
        _killpg(pgid, sig)

    def alive() -> bool:
        """True while a member of the target group is still running."""
        return not gone()

    _escalate(send, alive, interrupt_timeout, term_timeout)
    # Give SIGKILL a moment to land before declaring the group stuck.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not gone():
        time.sleep(0.05)
    return gone()


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
