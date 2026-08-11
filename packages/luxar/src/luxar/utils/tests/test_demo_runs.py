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


def test_luxar_path_mention_is_not_a_live_demo(tmp_path: Path, monkeypatch) -> None:
    """A recycled pgid whose command merely MENTIONS a luxar path is pruned.

    The guard requires the `-m luxar` module fingerprint — an editor open on a
    repo file or the viewer dev server must never be sentenced as a demo.
    """
    monkeypatch.setattr(
        demo_runs,
        "_ps_snapshot",
        lambda: [
            (600, 600, "vim /Users/x/workspace/python/luxar/notes.txt"),
            (601, 601, "node /x/luxar/packages/luxar-viewer/node_modules/vite dev"),
        ],
    )
    for pgid in (600, 601):
        path = register_run("stale", pgid, runs_dir=tmp_path)
        assert path is not None
    assert discover_runs(runs_dir=tmp_path) == []
    assert sorted(tmp_path.glob("*.json")) == []


# ─────────────────────────────── stopping ────────────────────────────────────
def _spawn_marked_sleeper() -> "subprocess.Popen[bytes]":
    """An isolated-group child whose ps command line carries a `-m luxar` mark.

    The registry's pid-reuse guard requires the `-m luxar` module fingerprint
    (not a bare "luxar" substring), so the marker argv mimics it. Deliberately
    NOT `-m luxar.demos.demo_*`: that would make a concurrently running REAL
    `luxar demo stop` sweep this test process up as a demo.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "-m", "luxar"],
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
        lambda **kw: [
            DemoRun(key="lorenz", pgid=100, pid=0, started=0.0, source="sweep")
        ],
    )
    hint = demo_runs.describe_port_holder(8000)
    assert hint is not None
    assert "'lorenz'" in hint and "luxar demo stop" in hint


def test_describe_port_holder_names_bare_serve(monkeypatch) -> None:
    class _Out:
        stdout = "101\n"

    monkeypatch.setattr(demo_runs.subprocess, "run", lambda *a, **k: _Out())
    monkeypatch.setattr(demo_runs, "_ps_snapshot", lambda: list(_SNAPSHOT))
    monkeypatch.setattr(demo_runs, "discover_runs", lambda **kw: [])
    hint = demo_runs.describe_port_holder(8000)
    assert hint is not None and "luxar serve" in hint


@posix_only
def test_pgid_zero_never_signals_own_group(tmp_path: Path) -> None:
    """killpg(0, sig) hits the CALLER's group — corrupt entries must not reach it.

    A parseable registry entry claiming pgid 0 (or 1) must be treated as dead
    and pruned, and terminate_process_group must refuse the id outright.
    """
    path = register_run("corrupt", 0, runs_dir=tmp_path)
    assert path is not None
    # Even with no ps snapshot (signal-0 fallback), pgid 0 reads as dead.
    assert demo_runs._group_alive(0) is False
    assert demo_runs._group_alive(1) is False
    assert terminate_process_group(0) is False
    assert terminate_process_group(-1) is False


def test_sweep_requires_python_executable() -> None:
    """Argument MENTIONS of the fingerprint (grep/vim/less) are never demos."""
    rows = [
        (5, 5, "grep -- '-m luxar.demos.demo_fake' notes.txt"),
        (6, 6, "vim -m luxar.demos.demo_fake.txt"),
        (7, 7, "/usr/bin/python3 -m luxar.demos.demo_real"),
        (8, 8, "/opt/py/Python -mluxar.demos.demo_attached"),  # attached -m form
    ]
    assert {(r.key, r.pgid) for r in _sweep_runs(rows)} == {
        ("real", 7),
        ("attached", 8),
    }


def test_registry_guard_requires_python_invocation(tmp_path: Path, monkeypatch) -> None:
    """A recycled pgid running a grep that mentions luxar text is pruned."""
    monkeypatch.setattr(
        demo_runs,
        "_ps_snapshot",
        lambda: [(700, 700, "grep -r '-m luxar serve' /Users/x/luxar")],
    )
    path = register_run("stale", 700, runs_dir=tmp_path)
    assert path is not None
    assert discover_runs(runs_dir=tmp_path) == []
    assert not path.exists()


def test_registry_rejects_bool_and_nonpositive_pgids(tmp_path: Path) -> None:
    (tmp_path / "1.json").write_text('{"key": "x", "pgid": true}')
    (tmp_path / "0.json").write_text('{"key": "y", "pgid": 0}')
    (tmp_path / "n.json").write_text('{"key": "z", "pgid": -5}')
    assert _registry_runs(tmp_path) == []
    assert sorted(tmp_path.glob("*.json")) == []


@posix_only
def test_stop_run_revalidates_at_kill_time(tmp_path: Path, monkeypatch) -> None:
    """A group that stopped being a demo between listing and kill is spared.

    The confirmation prompt can sit for minutes; a recycled pgid must not
    inherit the death sentence. The innocent group here is a live sleeper
    WITHOUT the luxar fingerprint — stop_run must leave it running.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        path = register_run("recycled", proc.pid, runs_dir=tmp_path)
        run = DemoRun(
            key="recycled",
            pgid=proc.pid,
            pid=0,
            started=0.0,
            source="registry",
            path=path,
        )
        assert stop_run(run) is True  # "already gone" as a demo
        assert proc.poll() is None  # the innocent process SURVIVES
        assert path is not None and not path.exists()  # entry pruned
    finally:
        proc.kill()
        proc.wait()


# ───────────────────── degraded platforms / missing `ps` ─────────────────────
@pytest.mark.skipif(not os.path.isdir("/proc"), reason="requires /proc (Linux)")
def test_ps_snapshot_falls_back_to_proc_when_ps_is_missing(monkeypatch) -> None:
    """No `ps` must not silently strip discovery of its identity check.

    An empty snapshot would let a recycled pgid through on signal-0 liveness
    alone; /proc answers the same question without a subprocess.
    """

    def no_ps(*args, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(demo_runs.subprocess, "run", no_ps)
    rows = demo_runs._ps_snapshot()
    mine = [row for row in rows if row[0] == os.getpid()]
    assert len(mine) == 1
    assert mine[0][1] == os.getpgrp()


def test_non_posix_keeps_registry_and_stops_by_owner_pid(
    tmp_path: Path, monkeypatch
) -> None:
    """Off-POSIX there are no process groups — discovery must still work.

    Pruning every entry there would both hide running demos and delete the
    registry that `stop_run`'s owner-pid fallback needs.
    """
    monkeypatch.setattr(demo_runs, "can_kill_process_groups", lambda: False)
    monkeypatch.setattr(demo_runs, "_ps_snapshot", list)
    path = register_run("winrun", 4242, runs_dir=tmp_path)
    assert path is not None

    runs = discover_runs(runs_dir=tmp_path)
    assert [(r.key, r.pgid, r.pid) for r in runs] == [("winrun", 4242, os.getpid())]
    assert path.exists()

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(
        demo_runs.os, "kill", lambda pid, sig: signalled.append((pid, sig))
    )
    assert stop_run(runs[0]) is True
    assert signalled == [(os.getpid(), demo_runs.signal.SIGTERM)]
    assert not path.exists()


def test_non_posix_prunes_an_entry_with_no_owner_pid(
    tmp_path: Path, monkeypatch
) -> None:
    """…but an entry naming no owner pid is unusable off-POSIX: drop it."""
    monkeypatch.setattr(demo_runs, "can_kill_process_groups", lambda: False)
    monkeypatch.setattr(demo_runs, "_ps_snapshot", list)
    (tmp_path / "4243.json").write_text('{"key": "old", "pgid": 4243, "pid": 0}')
    assert discover_runs(runs_dir=tmp_path) == []
    assert sorted(tmp_path.glob("*.json")) == []
