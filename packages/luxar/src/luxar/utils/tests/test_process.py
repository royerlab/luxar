"""Tests for :mod:`luxar.utils.process` — deterministic child teardown."""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from luxar.utils import process
from luxar.utils.process import _exit_code, _teardown, run_child_process

_POSIX = hasattr(os, "killpg")
posix_only = pytest.mark.skipif(not _POSIX, reason="requires os.killpg (POSIX)")


class _FakeProc:
    """A stand-in for Popen that records signals and reports liveness."""

    def __init__(self, pid: int = 4321, alive: bool = True) -> None:
        self.pid = pid
        self.returncode = 0
        self._alive = alive
        self.sent: list[int] = []

    def send_signal(self, sig: int) -> None:
        self.sent.append(sig)

    def poll(self):  # type: ignore[no-untyped-def]
        return None if self._alive else 0

    def wait(self, timeout=None):  # type: ignore[no-untyped-def]
        return self.returncode


def test_exit_code_maps_signal_death() -> None:
    assert _exit_code(0) == 0
    assert _exit_code(3) == 3
    assert _exit_code(-signal.SIGKILL) == 128 + signal.SIGKILL
    assert _exit_code(None) == 0


def test_normal_exit_returns_child_code() -> None:
    code = run_child_process(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        interrupt_timeout=0.1,
        term_timeout=0.1,
    )
    assert code == 3


@posix_only
def test_killpg_escalation_stops_when_group_gone(monkeypatch) -> None:
    """SIGINT then SIGTERM; no SIGKILL once the group reports gone."""
    calls: list[int] = []
    state = {"term_sent": False}

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append(sig)
        if sig == 0:  # liveness probe
            if state["term_sent"]:
                raise ProcessLookupError
            return
        if sig == signal.SIGTERM:
            state["term_sent"] = True

    monkeypatch.setattr(process.os, "killpg", fake_killpg)
    proc = _FakeProc()
    _teardown(proc, pgid=proc.pid, interrupt_timeout=0.05, term_timeout=0.05)

    non_probe = [s for s in calls if s != 0]
    assert non_probe == [signal.SIGINT, signal.SIGTERM]
    assert signal.SIGKILL not in calls


def test_finally_teardown_runs_on_keyboard_interrupt(monkeypatch) -> None:
    """A Ctrl-C during wait() returns 130 and still tears the child down."""
    killed: list[int] = []
    state = {"int_sent": False}

    def fake_killpg(pgid: int, sig: int) -> None:
        killed.append(sig)
        if sig == 0:
            if state["int_sent"]:
                raise ProcessLookupError
            return
        if sig == signal.SIGINT:
            state["int_sent"] = True

    class _InterruptingProc(_FakeProc):
        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            if timeout is None:  # the in-try wait()
                raise KeyboardInterrupt
            return 0

    monkeypatch.setattr(process.os, "killpg", fake_killpg)
    monkeypatch.setattr(
        process.subprocess, "Popen", lambda *a, **k: _InterruptingProc()
    )
    monkeypatch.setattr(process, "_CAN_KILLPG", True)

    code = run_child_process(["dummy"], interrupt_timeout=0.05, term_timeout=0.05)
    assert code == 130
    assert signal.SIGINT in killed  # teardown escalation ran


def test_non_isolate_uses_pid_signals_not_killpg(monkeypatch) -> None:
    recorded: list[int] = []
    monkeypatch.setattr(
        process.os, "killpg", lambda *a, **k: recorded.append(-1)
    )
    proc = _FakeProc(alive=True)
    _teardown(proc, pgid=None, interrupt_timeout=0.02, term_timeout=0.02)

    assert signal.SIGINT in proc.sent
    assert signal.SIGKILL in proc.sent  # stubborn (poll never dead) → escalates
    assert recorded == []  # os.killpg never touched in single-PID mode


@posix_only
def test_second_interrupt_jumps_to_sigkill(monkeypatch) -> None:
    """A Ctrl-C landing mid-escalation hard-kills instead of orphaning."""
    calls: list[int] = []
    state = {"sigint_raised": False, "dead": False}

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append(sig)
        if sig == 0:  # liveness probe
            if state["dead"]:
                raise ProcessLookupError
            return
        if sig == signal.SIGINT and not state["sigint_raised"]:
            state["sigint_raised"] = True
            raise KeyboardInterrupt  # impatient second Ctrl-C during the SIGINT step
        if sig == process._SIGKILL:
            state["dead"] = True

    monkeypatch.setattr(process.os, "killpg", fake_killpg)
    proc = _FakeProc()
    _teardown(proc, pgid=proc.pid, interrupt_timeout=0.05, term_timeout=0.05)

    assert signal.SIGINT in calls  # graceful step was attempted
    assert process._SIGKILL in calls  # ...then jumped straight to hard-kill
    assert signal.SIGTERM not in calls  # skipped the middle rung


def test_degrades_without_killpg(monkeypatch) -> None:
    captured: dict = {}

    class _Proc(_FakeProc):
        def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
            super().__init__()

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(process, "_CAN_KILLPG", False)
    monkeypatch.setattr(process.subprocess, "Popen", fake_popen)

    run_child_process(["dummy"], interrupt_timeout=0.02, term_timeout=0.02)
    assert captured.get("start_new_session") is False


@posix_only
def test_stop_handlers_install_and_restore() -> None:
    """SIGTERM/SIGHUP get a KeyboardInterrupt-raising handler, then restored."""
    before_term = signal.getsignal(signal.SIGTERM)
    prev = process._install_stop_handlers()
    try:
        assert signal.getsignal(signal.SIGTERM) is process._raise_keyboard_interrupt
        with pytest.raises(KeyboardInterrupt):
            process._raise_keyboard_interrupt(signal.SIGTERM, None)
    finally:
        process._restore_handlers(prev)
    assert signal.getsignal(signal.SIGTERM) == before_term


@posix_only
@pytest.mark.slow
def test_external_sigterm_reaps_grandchild() -> None:
    """External SIGTERM to a run_child_process owner tears the whole tree down.

    A helper process runs run_child_process on a sleeper (spawning a grandchild
    in its group), then receives an external SIGTERM on its main thread — the
    installed handler funnels it into teardown, killing the grandchild.
    """
    import subprocess

    helper = (
        "import sys;"
        "from luxar.utils.process import run_child_process;"
        "run_child_process([sys.executable,'-c','import time;time.sleep(30)'],"
        " isolate_group=False)"
    )
    proc = subprocess.Popen([sys.executable, "-c", helper])
    # Wait for the grandchild sleeper to exist.
    grandchild = None
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and grandchild is None:
        kids = subprocess.run(
            ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
        ).stdout.split()
        grandchild = kids[0] if kids else None
        if grandchild is None:
            time.sleep(0.1)
    assert grandchild is not None, "grandchild never spawned"

    proc.send_signal(signal.SIGTERM)  # external SIGTERM to the owner
    proc.wait(timeout=10)

    # Grandchild must be gone (teardown escalated to it).
    dead_deadline = time.monotonic() + 3.0
    while time.monotonic() < dead_deadline:
        alive = subprocess.run(
            ["kill", "-0", grandchild], capture_output=True
        ).returncode == 0
        if not alive:
            break
        time.sleep(0.05)
    still = subprocess.run(["kill", "-0", grandchild], capture_output=True).returncode
    assert still != 0, f"grandchild {grandchild} survived external SIGTERM"


@posix_only
@pytest.mark.slow
def test_sigkill_reaches_signal_ignoring_child() -> None:
    """A child that ignores SIGINT/SIGTERM is still killed via SIGKILL."""
    import subprocess

    stubborn = (
        "import signal, time;"
        "signal.signal(signal.SIGINT, signal.SIG_IGN);"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", stubborn], start_new_session=True
    )
    try:
        _teardown(proc, pgid=proc.pid, interrupt_timeout=0.2, term_timeout=0.2)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None, "SIGKILL escalation failed to kill child"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ───────────────────────── process table / zombies ───────────────────────────
_HAS_PROC = os.path.isdir("/proc")
linux_only = pytest.mark.skipif(not _HAS_PROC, reason="requires /proc (Linux)")


@linux_only
def test_proc_table_lists_this_process() -> None:
    rows = process.proc_table()
    mine = [row for row in rows if row[0] == os.getpid()]
    assert len(mine) == 1
    _pid, pgid, state, command = mine[0]
    assert pgid == os.getpgrp()
    assert state != "Z"
    assert "python" in command.lower()


def test_proc_table_falls_back_to_ps_without_proc(monkeypatch) -> None:
    class Result:
        stdout = """\
  101   101 S+   python -m luxar
  202   101 Z+   [python]
  303   303 R
bad row
"""

    def no_proc(_path: str) -> list[str]:
        raise OSError

    def ps_run(args, **kwargs):  # type: ignore[no-untyped-def]
        assert args == ["ps", "-axww", "-o", "pid=,pgid=,state=,command="]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 5.0,
            "check": True,
        }
        return Result()

    monkeypatch.setattr(process.os, "listdir", no_proc)
    monkeypatch.setattr(process.subprocess, "run", ps_run)

    assert process.proc_table() == [
        (101, 101, "S", "python -m luxar"),
        (202, 101, "Z", "[python]"),
        (303, 303, "R", ""),
    ]


def test_proc_table_is_unknown_when_proc_and_ps_are_unavailable(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError

    monkeypatch.setattr(process.os, "listdir", unavailable)
    monkeypatch.setattr(process.subprocess, "run", unavailable)

    assert process.proc_table() == []


@posix_only
@linux_only
def test_teardown_does_not_wait_out_an_unreaped_zombie() -> None:
    """A dead-but-unreaped child must not hold the escalation ladder open.

    `killpg(pgid, 0)` still succeeds for a zombie, so without looking at the
    process state every interrupted `demo run` sat through the full
    SIGINT (5s) + SIGTERM (3s) ladder before returning.
    """
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        # Wait for the corpse to appear — Popen has not reaped it, so the
        # group still answers killpg(0) with a zombie in it.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            states = [s for _p, g, s, _c in process.proc_table() if g == proc.pid]
            if states == ["Z"]:
                break
            time.sleep(0.05)
        assert states == ["Z"], f"expected a zombie group, saw {states}"

        start = time.monotonic()
        _teardown(proc, pgid=proc.pid, interrupt_timeout=5.0, term_timeout=3.0)
        assert time.monotonic() - start < 2.0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@posix_only
def test_terminate_process_group_accepts_ps_reported_zombie(monkeypatch) -> None:
    """A killed direct child is success before its parent reaps the zombie."""
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        os.killpg(proc.pid, signal.SIGKILL)

        def no_proc(_path: str) -> list[str]:
            raise OSError

        monkeypatch.setattr(process.os, "listdir", no_proc)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            states = [s for _p, g, s, _c in process.proc_table() if g == proc.pid]
            if states == ["Z"]:
                break
            time.sleep(0.05)
        assert states == ["Z"], f"expected a ps-reported zombie group, saw {states}"

        start = time.monotonic()
        assert process.terminate_process_group(
            proc.pid, interrupt_timeout=5.0, term_timeout=5.0
        )
        assert time.monotonic() - start < 2.0
        assert proc.wait(timeout=5) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@posix_only
def test_group_owned_by_another_user_is_never_reported_stopped(monkeypatch) -> None:
    """EPERM means "alive and not ours" — reporting it as stopped would lie."""

    def denied(pgid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(process.os, "killpg", denied)
    assert process._group_gone_probe(4321)() is False
    stopped = process.terminate_process_group(
        4321, interrupt_timeout=0.05, term_timeout=0.05
    )
    assert stopped is False
