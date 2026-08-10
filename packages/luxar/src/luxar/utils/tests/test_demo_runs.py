"""Tests for :mod:`luxar.utils.demo_runs` — the ``luxar demo stop`` engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from luxar.utils import demo_runs
from luxar.utils.demo_runs import (
    DemoRun,
    _registry_runs,
    _sweep_runs,
    discover_runs,
    register_run,
    stop_run,
    unregister_run,
)
from luxar.utils.process import terminate_process_group

_POSIX = hasattr(os, "killpg")
posix_only = pytest.mark.skipif(not _POSIX, reason="requires os.killpg (POSIX)")


# ────────────────────────────── registry ─────────────────────────────────────
def test_register_unregister_roundtrip(tmp_path: Path) -> None:
    path = register_run("lorenz", 12345, runs_dir=tmp_path)
    assert path is not None and path.exists()
    entry = json.loads(path.read_text())
    assert entry["key"] == "lorenz"
    assert entry["pgid"] == 12345
    assert entry["pid"] == os.getpid()
    unregister_run(path)
    assert not path.exists()
    # Idempotent, and never raises on None.
    unregister_run(path)
    unregister_run(None)


def test_registry_runs_prunes_corrupt_entries(tmp_path: Path) -> None:
    register_run("good", 111, runs_dir=tmp_path)
    (tmp_path / "999.json").write_text("{not json")
    (tmp_path / "998.json").write_text('{"pgid": "not-an-int-key-missing"}')
    runs = _registry_runs(tmp_path)
    assert [r.key for r in runs] == ["good"]
    # Corrupt files were deleted on sight.
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["111.json"]


@posix_only
def test_discover_prunes_dead_entries(tmp_path: Path) -> None:
    """An entry whose process group no longer exists is deleted, not returned."""
    # A spawned-and-reaped child's pid is a group id that no longer exists.
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()
    path = register_run("ghost", proc.pid, runs_dir=tmp_path)
    assert path is not None
    # A REAL demo may be running on this machine and appear via the sweep, so
    # only assert about the ghost group — never about the whole result.
    assert all(r.pgid != proc.pid for r in discover_runs(runs_dir=tmp_path))
    assert not path.exists()


@posix_only
def test_discover_excludes_own_group(tmp_path: Path) -> None:
    """`demo stop` must never target the process group it runs inside."""
    register_run("self", os.getpgrp(), runs_dir=tmp_path)
    assert all(r.pgid != os.getpgrp() for r in discover_runs(runs_dir=tmp_path))


# ─────────────────────────────── ps sweep ────────────────────────────────────
_SNAPSHOT = [
    (100, 100, "/usr/bin/python3 -m luxar.demos.demo_lorenz"),
    (101, 100, "/usr/bin/python3 -m luxar serve out.zarr --viewer --open"),
    (200, 90, "/usr/local/bin/luxar demo run lorenz"),  # L0: terminal job group
    (300, 300, "/opt/py/python -m luxar.demos.demo_4d_fractals --no-serve"),
    (400, 400, "vim luxar.demos.demo_notes.txt"),  # no `-m`: not a demo
]


def test_sweep_matches_demo_modules_only() -> None:
    runs = _sweep_runs(_SNAPSHOT)
    assert {(r.key, r.pgid) for r in runs} == {("lorenz", 100), ("4d_fractals", 300)}
    assert all(r.source == "sweep" for r in runs)


def test_sweep_dedupes_by_group() -> None:
    doubled = _SNAPSHOT + [(102, 100, "python -m luxar.demos.demo_lorenz worker")]
    assert sum(r.pgid == 100 for r in _sweep_runs(doubled)) == 1


def test_discover_prefers_registry_over_sweep(tmp_path: Path, monkeypatch) -> None:
    """The same group discovered both ways yields ONE run, with the exact key."""
    monkeypatch.setattr(demo_runs, "_ps_snapshot", lambda: list(_SNAPSHOT))
    register_run("lorenz_exact_key", 100, runs_dir=tmp_path)
    runs = [r for r in discover_runs(runs_dir=tmp_path) if r.pgid == 100]
    assert len(runs) == 1
    assert runs[0].key == "lorenz_exact_key"
    assert runs[0].source == "registry"


def test_discover_prunes_hijacked_group_ids(tmp_path: Path, monkeypatch) -> None:
    """A recycled pgid now owned by an unrelated process is pruned, not killed."""
    monkeypatch.setattr(
        demo_runs, "_ps_snapshot", lambda: [(500, 500, "/usr/bin/some-daemon")]
    )
    path = register_run("stale", 500, runs_dir=tmp_path)
    assert path is not None
    assert discover_runs(runs_dir=tmp_path) == []
    assert not path.exists()


# ─────────────────────────────── stopping ────────────────────────────────────
def _spawn_marked_sleeper() -> "subprocess.Popen[bytes]":
    """An isolated-group child whose ps command line contains 'luxar'."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "luxar-demo-marker"],
        start_new_session=True,
    )


@posix_only
def test_stop_run_kills_live_group(tmp_path: Path) -> None:
    proc = _spawn_marked_sleeper()
    try:
        path = register_run("sleeper", proc.pid, runs_dir=tmp_path)
        # Filter to our own group: a real demo running on this machine would
        # otherwise show up via the sweep (and must NOT be stopped by a test).
        runs = [r for r in discover_runs(runs_dir=tmp_path) if r.pgid == proc.pid]
        assert [r.key for r in runs] == ["sleeper"]
        assert stop_run(runs[0])
        # The direct child dies with the group (SIGINT lands first).
        assert proc.wait(timeout=10) != 0
        assert path is not None and not path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@posix_only
def test_terminate_process_group_gone_is_true() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()
    assert terminate_process_group(proc.pid, interrupt_timeout=0.1, term_timeout=0.1)


@posix_only
def test_terminate_process_group_sigint_first() -> None:
    """A well-behaved child exits on the ladder's first rung (SIGINT)."""
    proc = _spawn_marked_sleeper()
    start = time.monotonic()
    try:
        assert terminate_process_group(
            proc.pid, interrupt_timeout=5.0, term_timeout=5.0
        )
        # SIGINT killed it well before the SIGTERM rung's timeout stacked up.
        assert time.monotonic() - start < 8.0
        assert proc.wait(timeout=10) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ───────────────────────── busy-port diagnosis ───────────────────────────────
def test_describe_port_holder_unused_port_is_none() -> None:
    # Port 1 requires root to bind, so nothing of ours can be listening there;
    # whether lsof exists or not, the answer must be a clean None.
    assert demo_runs.describe_port_holder(1) is None


def test_describe_port_holder_names_registered_demo(monkeypatch) -> None:
    class _Out:
        stdout = "100\n"

    monkeypatch.setattr(demo_runs.subprocess, "run", lambda *a, **k: _Out())
    monkeypatch.setattr(demo_runs, "_ps_snapshot", lambda: list(_SNAPSHOT))
    monkeypatch.setattr(
        demo_runs,
        "discover_runs",
        lambda: [DemoRun(key="lorenz", pgid=100, pid=0, started=0.0, source="sweep")],
    )
    hint = demo_runs.describe_port_holder(8000)
    assert hint is not None
    assert "'lorenz'" in hint and "luxar demo stop" in hint


def test_describe_port_holder_names_bare_serve(monkeypatch) -> None:
    class _Out:
        stdout = "101\n"

    monkeypatch.setattr(demo_runs.subprocess, "run", lambda *a, **k: _Out())
    monkeypatch.setattr(demo_runs, "_ps_snapshot", lambda: list(_SNAPSHOT))
    monkeypatch.setattr(demo_runs, "discover_runs", lambda: [])
    hint = demo_runs.describe_port_holder(8000)
    assert hint is not None and "luxar serve" in hint
