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
