"""Tests for ``luxar stop``, the server registry behind it, and export --open.

Covers the standalone-server half of :mod:`luxar.cli.demo_runs` (``kind`` =
``"server"`` entries, PID-targeted stops, dedup against a demo's own server)
and the ``luxar export --open`` server lifecycle (owned child, no orphan, no
lost output, no browser without a display).
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from luxar.cli import app, demo_runs
from luxar.cli.demo_runs import (
    DemoRun,
    _is_luxar_command,
    discover_runs,
    register_server,
    registered_server,
    stop_run,
)

_POSIX = hasattr(os, "killpg")
posix_only = pytest.mark.skipif(not _POSIX, reason="requires POSIX signals")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def marked_python_pkg(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway `luxar` package whose `python -m luxar` just sleeps.

    Its command line passes the server identity check without being anything a
    real `luxar stop` on this machine could mistake for a demo.
    """
    root = tmp_path_factory.mktemp("marked-server-pkg")
    pkg = root / "luxar"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text("import time\n\ntime.sleep(120)\n")
    return root


def _spawn(pkg_root: Path, **kwargs: object) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(  # type: ignore[call-overload]
        [sys.executable, "-m", "luxar", "serve"],
        cwd=pkg_root,
        env={**os.environ, "PYTHONPATH": str(pkg_root)},
        **kwargs,
    )


def _reap(*procs: "subprocess.Popen[bytes]") -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# ─────────────────────────── identity fingerprint ────────────────────────────
@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/python3 -m luxar serve x.luxar.zarr",
        "/venv/bin/python3 /venv/bin/luxar serve x.luxar.zarr --viewer",
        # A macOS hatch env: the console script path contains a space.
        "/opt/py/bin/python3.12 /Users/me/Library/Application Support/hatch/"
        "env/virtual/luxar/X/luxar/bin/luxar viewer --data s.zarr",
        "/usr/local/bin/luxar gsplat view f.gsplats.zarr",
    ],
)
def test_is_luxar_command_accepts_module_and_console_script(command: str) -> None:
    assert _is_luxar_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "vim /repo/bin/luxar",
        "/usr/bin/python3 -c print('luxar') /venv/bin/luxar",
        "/usr/bin/python3 /tmp/notluxar.py serve",
        "",
    ],
)
def test_is_luxar_command_rejects_lookalikes(command: str) -> None:
    assert not _is_luxar_command(command)


# ──────────────────────────────── registry ───────────────────────────────────
def test_registered_server_writes_and_removes_its_entry(tmp_path: Path) -> None:
    with registered_server("serve x.luxar.zarr", "serve", runs_dir=tmp_path):
        entries = list(tmp_path.glob("*.json"))
        assert [p.name for p in entries] == [f"server-{os.getpid()}.json"]
    assert list(tmp_path.glob("*.json")) == []


def test_demo_discovery_ignores_server_entries(tmp_path: Path, monkeypatch) -> None:
    """`luxar demo stop` lists demos only, but keeps a LIVE server's entry."""
    monkeypatch.setattr(demo_runs, "_process_started_at", lambda pid: 0.0)
    register_server("serve x", "serve", pid=4321, runs_dir=tmp_path)
    snapshot = [(4321, 4321, "/usr/bin/python3 -m luxar serve x.luxar.zarr")]
    assert discover_runs(runs_dir=tmp_path, snapshot=snapshot) == []
    assert len(list(tmp_path.glob("server-*.json"))) == 1


def test_demo_discovery_prunes_stale_server_entries(tmp_path: Path) -> None:
    """`luxar demo stop` tidies a dead server's entry from the shared registry."""
    path = register_server("serve x", "serve", pid=999_999, runs_dir=tmp_path)
    assert discover_runs(runs_dir=tmp_path, snapshot=[(1, 1, "init")]) == []
    assert path is not None and not path.exists()


def test_dead_server_entry_is_pruned(tmp_path: Path) -> None:
    path = register_server("serve x", "serve", pid=999_999, runs_dir=tmp_path)
    runs = discover_runs(
        runs_dir=tmp_path, snapshot=[(1, 1, "init")], include_servers=True
    )
    assert runs == []
    assert path is not None and not path.exists()


def test_server_inside_a_listed_demo_group_is_hidden(
    tmp_path: Path, monkeypatch
) -> None:
    """A demo's own `luxar serve` child dies with the demo; list it once."""
    monkeypatch.setattr(demo_runs, "_process_started_at", lambda pid: 0.0)
    demo_runs.register_run("lorenz", 4242, runs_dir=tmp_path)
    register_server("serve lorenz.luxar.zarr", "serve", pid=4243, runs_dir=tmp_path)
    register_server("serve other.luxar.zarr", "serve", pid=5000, runs_dir=tmp_path)
    snapshot = [
        (4242, 4242, "/usr/bin/python3 -m luxar.demos.demo_lorenz"),
        (4243, 4242, "/usr/bin/python3 -m luxar serve lorenz.luxar.zarr"),
        (5000, 5000, "/venv/bin/python /venv/bin/luxar serve other.luxar.zarr"),
    ]
    runs = discover_runs(runs_dir=tmp_path, snapshot=snapshot, include_servers=True)
    assert sorted((r.kind, r.key) for r in runs) == [
        ("demo", "lorenz"),
        ("server", "serve other.luxar.zarr"),
    ]


@posix_only
# ───────────────────────── PID-reuse identity guard ──────────────────────────
@pytest.mark.parametrize(
    ("text", "seconds"),
    [("05:07", 307.0), ("1:02:03", 3723.0), ("2-01:00:00", 176400.0), ("x", None)],
)
def test_parse_etime(text: str, seconds) -> None:
    assert demo_runs._parse_etime(text) == seconds


def _server_run(pid: int, started: float, subcommand: str = "serve") -> DemoRun:
    return DemoRun(
        key="serve x",
        pgid=pid,
        pid=pid,
        started=started,
        source="registry",
        kind="server",
        subcommand=subcommand,
    )


def test_identity_rejects_a_pid_recycled_after_registration(monkeypatch) -> None:
    snapshot = [(77, 77, "/usr/bin/python3 -m luxar serve other.luxar.zarr")]
    monkeypatch.setattr(demo_runs, "_process_started_at", lambda pid: 1000.0)
    # Started after the entry was written: someone else's process now.
    assert demo_runs._server_identity(_server_run(77, 500.0), snapshot) is False
    # Started before it (the server registers itself after starting): ours.
    assert demo_runs._server_identity(_server_run(77, 1001.0), snapshot) is True


def test_identity_requires_the_recorded_subcommand(monkeypatch) -> None:
    monkeypatch.setattr(demo_runs, "_process_started_at", lambda pid: 0.0)
    snapshot = [(77, 77, "/venv/bin/python /venv/bin/luxar viewer --data s.zarr")]
    assert demo_runs._server_identity(_server_run(77, 10.0), snapshot) is False
    run = _server_run(77, 10.0, subcommand="viewer")
    assert demo_runs._server_identity(run, snapshot) is True


def test_identity_is_unknown_without_a_process_table_or_start_time(
    monkeypatch,
) -> None:
    assert demo_runs._server_identity(_server_run(77, 10.0), []) is None
    monkeypatch.setattr(demo_runs, "_process_started_at", lambda pid: None)
    snapshot = [(77, 77, "/usr/bin/python3 -m luxar serve x")]
    assert demo_runs._server_identity(_server_run(77, 10.0), snapshot) is None


@posix_only
def test_stop_refuses_to_signal_an_unverifiable_server(monkeypatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(demo_runs, "_ps_snapshot", lambda: [])
    monkeypatch.setattr(demo_runs, "terminate_process", lambda pid: killed.append(pid))
    assert stop_run(_server_run(77, 10.0)) is False
    assert killed == []


@posix_only
def test_stop_does_not_kill_a_recycled_pid(tmp_path: Path, monkeypatch) -> None:
    killed: list[int] = []
    path = register_server("serve x", "serve", pid=77, runs_dir=tmp_path)
    run = demo_runs._registry_runs(tmp_path)[0]
    monkeypatch.setattr(
        demo_runs,
        "_ps_snapshot",
        lambda: [(77, 77, "/usr/bin/python3 -m luxar serve y")],
    )
    monkeypatch.setattr(
        demo_runs, "_process_started_at", lambda pid: run.started + 60.0
    )
    monkeypatch.setattr(demo_runs, "terminate_process", lambda pid: killed.append(pid))
    assert stop_run(run) is True  # nothing of ours left to stop...
    assert run.pruned  # ...reported as such, not as "stopped"
    assert killed == []  # ...and the stranger holding the PID is untouched
    assert path is not None and not path.exists()


@posix_only
def test_registered_server_turns_sighup_into_a_clean_exit(tmp_path: Path) -> None:
    import signal as _signal

    previous = _signal.signal(_signal.SIGHUP, _signal.SIG_DFL)
    try:
        with pytest.raises(SystemExit):
            with registered_server("serve x", "serve", runs_dir=tmp_path):
                assert _signal.getsignal(_signal.SIGHUP) is demo_runs._exit_on_hangup
                os.kill(os.getpid(), _signal.SIGHUP)
                time.sleep(1)  # the handler raises before this returns
        assert list(tmp_path.glob("*.json")) == []  # the finally still ran
        assert _signal.getsignal(_signal.SIGHUP) is _signal.SIG_DFL
    finally:
        _signal.signal(_signal.SIGHUP, previous)


@posix_only
def test_stop_run_kills_the_server_pid_but_not_its_group(
    tmp_path: Path, marked_python_pkg: Path
) -> None:
    """A server shares its group with whatever started it: spare the siblings."""
    # A new group in OUR session: setpgid across sessions is EPERM.
    leader = _spawn(marked_python_pkg, process_group=0)
    server = _spawn(marked_python_pkg, process_group=leader.pid)
    try:
        time.sleep(0.5)  # let `ps` see the children before the entry is written
        register_server("serve x", "serve", pid=server.pid, runs_dir=tmp_path)
        runs = [
            r
            for r in discover_runs(runs_dir=tmp_path, include_servers=True)
            if r.kind == "server"
        ]
        assert [r.pgid for r in runs] == [server.pid]
        assert stop_run(runs[0])
        assert server.wait(timeout=10) != 0
        assert leader.poll() is None, "the group sibling must survive"
        assert list(tmp_path.glob("server-*.json")) == []
    finally:
        _reap(server, leader)


# ─────────────────────────────── luxar stop ──────────────────────────────────
def _fake_runs() -> list[DemoRun]:
    return [
        DemoRun(key="lorenz", pgid=111, pid=0, started=0.0, source="registry"),
        DemoRun(
            key="serve tiny.luxar.zarr",
            pgid=333,
            pid=333,
            started=0.0,
            source="registry",
            kind="server",
        ),
    ]


def test_stop_lists_servers_and_demos_on_dry_run(runner, monkeypatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(demo_runs, "discover_runs", lambda **kw: _fake_runs())
    monkeypatch.setattr(demo_runs, "stop_run", lambda r: killed.append(r.pgid))
    result = runner.invoke(app, ["stop", "--dry-run"])
    assert result.exit_code == 0
    out = _strip_ansi(result.stdout)
    assert "lorenz" in out and "pgid 111" in out
    assert "serve tiny.luxar.zarr" in out and "pid 333" in out
    assert killed == []


def test_stop_without_a_tty_refuses_unless_yes(runner, monkeypatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(demo_runs, "discover_runs", lambda **kw: _fake_runs())
    monkeypatch.setattr(demo_runs, "stop_run", lambda r: killed.append(r.pgid) or True)
    monkeypatch.setattr("luxar.cli.utils.stdin_is_interactive", lambda: False)
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 1
    assert "--yes" in result.stdout
    assert killed == []

    result = runner.invoke(app, ["stop", "--yes"])
    assert result.exit_code == 0
    assert sorted(killed) == [111, 333]


def test_stop_reports_a_survivor_with_a_pid_hint(runner, monkeypatch) -> None:
    monkeypatch.setattr(demo_runs, "discover_runs", lambda **kw: _fake_runs()[1:])
    monkeypatch.setattr(demo_runs, "stop_run", lambda r: False)
    result = runner.invoke(app, ["stop", "--yes"])
    assert result.exit_code == 1
    # A server is killed by PID — never a `kill -9 -<pgid>` group hint.
    if _POSIX:
        assert "kill -9 333" in result.stdout


def test_stop_with_nothing_running(runner, monkeypatch) -> None:
    monkeypatch.setattr(demo_runs, "discover_runs", lambda **kw: [])
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    assert "No running" in result.stdout


def test_serve_registers_itself_while_running(tmp_path: Path, monkeypatch) -> None:
    """`luxar serve` is visible to `luxar stop` exactly while it serves."""
    runs_dir = tmp_path / "running"
    monkeypatch.setattr(demo_runs, "DEMO_RUNS_DIR", runs_dir)
    seen: list[list[str]] = []

    def _fake_run(*_a: object, **_k: object) -> None:
        seen.append([p.name for p in runs_dir.glob("*.json")])

    store = tmp_path / "s.luxar.zarr"
    store.mkdir()
    with patch("luxar.cli.main.uvicorn.run", _fake_run):
        result = CliRunner().invoke(
            app, ["serve", str(store), "--port", str(_free_port())]
        )
    assert result.exit_code == 0, result.stdout
    assert seen == [[f"server-{os.getpid()}.json"]]
    assert list(runs_dir.glob("*.json")) == []


# ───────────────────────────── export --open ─────────────────────────────────
def _fake_export(root: Path) -> Path:
    """An export folder with the REAL generated serve.py and stub content."""
    from luxar.cli.export import _generate_serve_script

    root.mkdir()
    (root / "viewer").mkdir()
    (root / "viewer" / "index.html").write_text("<html></html>")
    (root / "data").mkdir()
    _generate_serve_script(root, "data", title="t")
    return root


def test_export_open_passes_no_open_without_a_display(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli import main as cli_main

    result_dir = _fake_export(tmp_path / "x")
    monkeypatch.setattr(demo_runs, "DEMO_RUNS_DIR", tmp_path / "running")
    monkeypatch.setattr(
        "luxar.cli.utils.browser_unavailable_reason",
        lambda explicit=False: "LUXAR_OPEN_BROWSER=0",
    )
    with patch("luxar._process.run_child_process", return_value=0) as run:
        cli_main._serve_export(result_dir, 18999)
    cmd = run.call_args.args[0]
    assert cmd[1:] == [str(result_dir / "serve.py"), "--port", "18999", "--no-open"]
    assert run.call_args.kwargs["cwd"] == str(result_dir)


def test_export_open_reports_a_failing_server(tmp_path: Path, monkeypatch) -> None:
    """A serve.py that dies used to end `export --open` with status 0, silently."""
    import typer

    from luxar.cli import main as cli_main

    result_dir = _fake_export(tmp_path / "x")
    monkeypatch.setattr(demo_runs, "DEMO_RUNS_DIR", tmp_path / "running")
    with patch("luxar._process.run_child_process", return_value=1):
        with pytest.raises(typer.Exit) as exc:
            cli_main._serve_export(result_dir, 18999)
    assert exc.value.exit_code == 1


@posix_only
def test_export_open_server_survives_detached_and_dies_with_its_parent(
    tmp_path: Path,
) -> None:
    """Real processes: no TTY on any stream, output to a file, then SIGTERM.

    The server must come up and stay up without a terminal, the parent's own
    lines must reach the log (they used to sit in an unflushed buffer while
    the child wrote straight to the shared descriptor), and terminating the
    parent must take the server with it instead of orphaning it on its port.
    """
    result_dir = _fake_export(tmp_path / "x")
    port = _free_port()
    log = tmp_path / "export.log"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LUXAR_OPEN_BROWSER", "DISPLAY", "WAYLAND_DISPLAY")
    }
    env["HOME"] = str(tmp_path)  # keep the run registry out of the real ~/.cache
    env["LUXAR_OPEN_BROWSER"] = "0"  # `--open` is explicit; never pop a browser
    script = (
        "import sys; from pathlib import Path; "
        "from luxar.cli.main import _serve_export; "
        "_serve_export(Path(sys.argv[1]), int(sys.argv[2]))"
    )
    with log.open("wb") as out:
        parent = subprocess.Popen(
            [sys.executable, "-c", script, str(result_dir), str(port)],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    try:
        # serve.py shifts to the next port if another test grabbed ours in the
        # meantime, so read the port it actually bound from its printed URL.
        url_re = re.compile(r"http://127\.0\.0\.1:(\d+)/viewer/")
        deadline = time.monotonic() + 90
        status = None
        while time.monotonic() < deadline and parent.poll() is None:
            match = url_re.search(log.read_text(errors="replace"))
            if match is None:
                time.sleep(0.2)
                continue
            port = int(match.group(1))
            try:
                with urllib.request.urlopen(  # nosec B310 - loopback test URL
                    f"http://127.0.0.1:{port}/viewer/", timeout=1
                ) as resp:
                    status = resp.status
                break
            except OSError:
                time.sleep(0.2)
        assert status == 200, log.read_text()
        time.sleep(1.0)
        assert parent.poll() is None, log.read_text()

        text = log.read_text()
        assert "Starting local server" in text, text
        assert text.index("Starting local server") < text.index("Press Ctrl+C to stop")
        assert "Not opening a browser" in text, text

        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=30) == 0, log.read_text()
        # The child is gone with it: the port is free again.
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait()


# ─────────────────────────── explicit vs implicit open ───────────────────────
def test_serve_open_is_explicit_and_auto_open_is_not(tmp_path: Path) -> None:
    """`serve --open` always opens; demos' `--auto-open` defers to the display."""
    store = tmp_path / "s.luxar.zarr"
    store.mkdir()
    for flag, explicit in (("--open", True), ("--auto-open", False)):
        calls: list[dict] = []
        with (
            patch("luxar.cli.main.uvicorn.run"),
            patch("luxar.cli.main.ensure_viewer_built", return_value=True),
            patch("luxar.cli.main.wait_for_server", return_value=True),
            patch("luxar.cli.main.threading.Thread"),
            patch(
                "luxar.cli.main.open_browser_func",
                side_effect=lambda url, calls=calls, **kw: calls.append(kw),
            ),
            patch.object(demo_runs, "DEMO_RUNS_DIR", tmp_path / "running"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "serve",
                    str(store),
                    "--viewer",
                    flag,
                    "--port",
                    str(_free_port()),
                    "--viewer-port",
                    str(_free_port()),
                ],
            )
        assert result.exit_code == 0, result.stdout
        assert calls == [{"explicit": explicit}], flag


def test_viewer_open_default_is_implicit(tmp_path: Path) -> None:
    for argv, flag, explicit in (
        ([], True, False),
        (["--open"], True, True),
        (["--no-open"], False, False),
    ):
        seen: list[tuple] = []
        with (
            patch("luxar.cli.main.ensure_viewer_built", return_value=True),
            patch(
                "luxar.cli.main._serve_viewer",
                side_effect=lambda *a, seen=seen, **kw: seen.append(
                    (a[3], kw["open_explicit"])
                ),
            ),
            patch.object(demo_runs, "DEMO_RUNS_DIR", tmp_path / "running"),
        ):
            result = CliRunner().invoke(
                app, ["viewer", "--port", str(_free_port()), *argv]
            )
        assert result.exit_code == 0, result.stdout
        assert seen == [(flag, explicit)], argv
