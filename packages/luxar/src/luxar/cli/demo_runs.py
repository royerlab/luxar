"""Track and stop running ``luxar demo`` processes (stdlib-only).

``luxar demo run`` spawns its demo script as an isolated process-group leader
(see :mod:`luxar._process`), and that script in turn spawns
``luxar serve``. Ctrl-C teardown is deterministic — but a demo the user simply
*forgot* in another terminal never receives a signal, and keeps its ports, its
memory and its GPU. Re-running that demo then shifts to a neighbouring port
(a demo's derived port pair is stable, see
:func:`luxar.demos._support.runtime.viewer.demo_ports`)
while the old tab keeps serving the stale scene. ``luxar demo stop`` fixes
that; this module is its discovery + kill engine, with two complementary
sources:

- **Registry** (primary): ``demo run`` / ``run-all`` drop a JSON pidfile per
  launch under ``~/.cache/luxar/running/`` and remove it on exit. Precise —
  knows the demo key and the process group to kill.
- **Process-table sweep** (fallback): a ``ps`` scan for
  ``python -m luxar.demos.demo_*`` command lines catches runs that predate the
  registry or whose pidfile was lost.

The same registry also records standalone SERVERS — ``luxar serve``,
``luxar viewer``, ``luxar gsplat view``, ``luxar export --open`` — as
``kind="server"`` entries (:func:`registered_server`), which ``luxar stop``
lists and stops alongside the demos. A server is stopped by PID, never by
process group: unlike a demo it is not the leader of a group of its own, so a
group kill could reach whatever script or pipeline started it.

The module body imports only the standard library and :mod:`luxar._process`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess  # nosec B404  # fixed argv (ps/lsof), never shell, parsed
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .._process import (
    can_kill_process_groups,
    proc_table,
    terminate_process,
    terminate_process_group,
)

# A unit test pins this path to luxar.demos.registry.DEMO_CACHE_ROOT so the
# duplicated location cannot drift.
DEMO_RUNS_DIR = Path.home() / ".cache" / "luxar" / "running"

# The L1 demo script every `demo run` spawns: `<python> -m luxar.demos.demo_X`.
# This is the sweep's definitive fingerprint — it exists for the whole life of
# a demo, is an isolated group leader, and its module name encodes the key.
# The strict form anchors a full module token (see `_demo_module_suffix`).
_DEMO_MODULE_RE_STRICT = re.compile(r"luxar\.demos\.demo_([\w.]+)$")


@dataclass
class DemoRun:
    """One running demo process tree, however it was discovered."""

    key: str  # demo key ("?" when only swept and unparseable)
    pgid: int  # isolated process group of the demo script — the kill target
    pid: int  # pid of the `luxar demo run` owner (0 when unknown)
    started: float  # epoch seconds (0.0 when unknown)
    source: str  # "registry" | "sweep"
    path: Optional[Path] = None  # registry file, pruned after a kill
    # "demo": `pgid` is an isolated process group, killed whole.
    # "server": a standalone luxar server; `pgid` holds its PID, killed alone.
    kind: str = "demo"
    # Servers only: the luxar subcommand its argv must still carry ("serve",
    # "gsplat view", …) — part of the PID-reuse identity check.
    subcommand: str = ""
    # Set by stop_run when nothing was signalled because the run was already
    # gone (or its id now belongs to someone else) and only its entry was
    # pruned — so the caller does not report it as "stopped".
    pruned: bool = False


# ────────────────────────────── registry ─────────────────────────────────────
def register_run(
    key: str, pgid: int, runs_dir: Optional[Path] = None
) -> Optional[Path]:
    """Record a just-spawned demo group; returns the entry path (None on failure).

    Called from ``run_child_process``'s ``on_spawn`` hook, so it must never
    raise into the launch path — a failed registration only degrades
    ``demo stop`` to its sweep fallback.
    """
    runs_dir = runs_dir or DEMO_RUNS_DIR
    path = runs_dir / f"{pgid}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "pgid": pgid,
            "pid": os.getpid(),
            "started": time.time(),
            "python": sys.executable,
        }
        # Write-then-rename. `_registry_runs` prunes anything it cannot parse,
        # so a concurrent reader catching a half-written file would DELETE the
        # entry of a demo that is only just starting. The staging name is not
        # `*.json`, so it is invisible to that glob.
        tmp.write_text(json.dumps(entry))
        tmp.replace(path)
        return path
    except OSError:
        unregister_run(tmp)
        return None


def register_server(
    label: str,
    subcommand: str,
    pid: Optional[int] = None,
    runs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Record a standalone luxar server process; returns the entry path.

    ``subcommand`` is the luxar command the process runs (``"serve"``,
    ``"gsplat view"``); with the registration time it lets ``luxar stop`` tell
    this server from an unrelated luxar process that later inherits its PID.
    ``pid`` defaults to the calling process. Like :func:`register_run` this
    never raises: an unregistered server only means ``luxar stop`` cannot see it.
    """
    runs_dir = runs_dir or DEMO_RUNS_DIR
    pid = os.getpid() if pid is None else pid
    path = runs_dir / f"server-{pid}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "kind": "server",
            "key": label,
            "pgid": pid,  # the kill target; for a server that is its PID
            "pid": pid,
            "started": time.time(),
            "python": sys.executable,
            "subcommand": subcommand,
        }
        tmp.write_text(json.dumps(entry))  # write-then-rename, see register_run
        tmp.replace(path)
        return path
    except OSError:
        unregister_run(tmp)
        return None


def _exit_on_hangup(signum: int, frame: object) -> None:
    """SIGHUP/SIGTERM handler: unwind (running ``finally`` blocks), not die."""
    raise SystemExit(128 + signum)


def _install_hangup_exit() -> Optional[object]:
    """Route SIGHUP to SystemExit; returns the handler to restore (or None).

    A server whose terminal closes gets SIGHUP, whose default action kills the
    process outright — skipping the ``finally`` that removes its registry
    entry. Only the default disposition is replaced: an explicit ignore
    (``nohup``) or a handler someone else installed is left alone.
    """
    sig = getattr(signal, "SIGHUP", None)
    if sig is None or threading.current_thread() is not threading.main_thread():
        return None
    try:
        if signal.getsignal(sig) is not signal.SIG_DFL:
            return None
        signal.signal(sig, _exit_on_hangup)
    except (ValueError, OSError):
        return None
    return signal.SIG_DFL


@contextmanager
def exit_on_termination_signals() -> Iterator[None]:
    """Route SIGTERM and SIGHUP to ``SystemExit`` for the duration of the block.

    Their default action kills the process outright, skipping every ``finally``
    — so a command that owns temporary state (``gsplat view``'s temp scene, an
    extracted archive) leaks it on a harness timeout, ``luxar stop`` or a closed
    terminal. Unwinding instead runs the cleanup. As with
    :func:`_install_hangup_exit`, only a DEFAULT disposition is replaced (an
    explicit ignore such as ``nohup``, or someone else's handler, is left), only
    on the main thread, and the previous dispositions are restored on exit.
    """
    restore: list[tuple[int, object]] = []
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                if signal.getsignal(sig) is signal.SIG_DFL:
                    signal.signal(sig, _exit_on_hangup)
                    restore.append((sig, signal.SIG_DFL))
            except (ValueError, OSError):
                continue
    try:
        yield
    finally:
        for sig, previous in restore:
            try:
                signal.signal(sig, previous)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass


@contextmanager
def registered_server(
    label: str, subcommand: str, runs_dir: Optional[Path] = None
) -> Iterator[None]:
    """Keep this process registered as a server for the duration of the block."""
    path = register_server(label, subcommand, runs_dir=runs_dir)
    previous = _install_hangup_exit()
    try:
        yield
    finally:
        unregister_run(path)
        if previous is not None:
            try:
                signal.signal(signal.SIGHUP, previous)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass


def unregister_run(path: Optional[Path]) -> None:
    """Remove a registry entry written by :func:`register_run` (never raises)."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _registry_runs(runs_dir: Path) -> list[DemoRun]:
    """Parse every registry entry, deleting corrupt files on sight."""
    runs: list[DemoRun] = []
    if not runs_dir.is_dir():
        return runs
    for path in sorted(runs_dir.glob("*.json")):
        try:
            entry = json.loads(path.read_text())
            pgid = int(entry["pgid"])
            if pgid <= 1:
                # 0 / negative are kill(2) wildcards or the caller's own
                # group; a JSON `true` also coerces to 1. Corrupt, prune.
                raise ValueError(f"invalid pgid {pgid}")
            kind = str(entry.get("kind", "demo"))
            if kind not in ("demo", "server"):
                raise ValueError(f"unknown kind {kind!r}")
            runs.append(
                DemoRun(
                    key=str(entry["key"]),
                    pgid=pgid,
                    pid=int(entry.get("pid", 0)),
                    started=float(entry.get("started", 0.0)),
                    source="registry",
                    path=path,
                    kind=kind,
                    subcommand=str(entry.get("subcommand", "")),
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            unregister_run(path)
    return runs


# ─────────────────────────────── ps sweep ────────────────────────────────────
def _ps_snapshot() -> list[tuple[int, int, str]]:
    """``(pid, pgid, command)`` for every visible process; ``[]`` off POSIX.

    ``ps -axww`` so command lines are never truncated — the ``-m luxar.demos.``
    fingerprint sits past column 80 behind a long interpreter path.

    On Linux, falls back to ``/proc`` when ``ps`` is missing (a slim container),
    times out, or prints something we cannot parse (BusyBox takes different
    flags). An empty snapshot does not merely disable the sweep: it also strips
    discovery of the identity check that keeps a recycled pgid from being
    killed, so it is worth a second source where available before giving up.
    """
    try:
        out = subprocess.run(  # nosec B603, B607  # fixed argv, no user input
            ["ps", "-axww", "-o", "pid=,pgid=,command="],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows or [(pid, pgid, cmd) for pid, pgid, _state, cmd in proc_table()]


def _visible_pids() -> frozenset[int]:
    """Every pid the OS admits to, where no process *table* is reachable.

    Off POSIX ``_ps_snapshot`` is empty and ``os.kill(pid, 0)`` is a hard
    terminate rather than a probe, which used to leave a registry entry
    unfalsifiable: one written before a reboot (or by a hard-killed owner) was
    reported as a running demo forever, on every future ``demo stop``, with no
    command that could clear it. Windows' own ``tasklist`` answers the one
    question that *is* safe to ask — does this pid still exist? — so such a
    record prunes itself on the next listing instead.

    Deliberately NOT folded into ``_ps_snapshot``: ``tasklist`` reports no
    command line, and a snapshot without one fails the demo-identity check in
    :func:`_group_has_luxar_process`, which would prune LIVE demos. It also
    cannot make anything killable — identity is still unverifiable, so
    :func:`stop_run` keeps refusing to signal.

    An empty result means "unknown", never "nothing is running".
    """
    try:
        out = subprocess.run(  # nosec B603, B607  # fixed argv, no user input
            ["tasklist", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    pids: set[int] = set()
    for line in out.splitlines():
        # `"Image Name","PID","Session Name",…` — split on the quote-comma-quote
        # separator, so an image name containing a comma stays one field.
        fields = [field.strip('"') for field in line.split('","')]
        if len(fields) > 1 and fields[1].isdigit():
            pids.add(int(fields[1]))
    return frozenset(pids)


# Interpreter options whose argument is a SEPARATE token; stepping over it
# keeps a `python -W ignore -m luxar.demos.demo_x` recognisable as a demo.
_PY_OPTS_WITH_ARG = frozenset({"-W", "-X"})

# Tokens after which Python has chosen what to run, so everything left on the
# line is the program's own argv: a code string, end-of-options, or stdin.
_PY_RUN_SELECTORS = frozenset({"-c", "--", "-"})

# How many extra whitespace-split tokens an interpreter path may be rejoined
# from (see `_interpreter_arg_start`) — one or two spaces is what real install
# prefixes have, and the bound keeps the scan cheap on every other process.
_MAX_EXE_TOKENS = 5


def _interpreter_arg_start(tokens: list[str]) -> Optional[int]:
    """Index of the first interpreter *argument*, or None if this isn't python.

    ``ps`` renders argv space-joined and unquoted, so an interpreter path that
    itself CONTAINS a space arrives split across tokens — which is not exotic:
    a macOS hatch environment lives under ``~/Library/Application Support/``,
    so ``hatch run luxar demo run …`` spawns exactly such a command line.
    Reading only ``tokens[0]`` there sees ``…/Library/Application``, decides
    the process is not python, and the demo becomes invisible to ``demo stop``
    — worse, its registry entry is pruned as a dead group.

    So rejoin tokens until the candidate both names a python AND is a file on
    this machine. The existence check is what keeps this from loosening the
    guard: ``less /opt/x/python -m luxar.demos.demo_x`` only ever produces the
    candidate ``"less /opt/x/python"``, which is not a path that exists.
    """
    if "python" in Path(tokens[0]).name.lower():
        return 1
    for end in range(2, min(len(tokens), _MAX_EXE_TOKENS) + 1):
        if tokens[end - 1].startswith("-"):
            break  # an option: the executable ended before this token
        candidate = " ".join(tokens[:end])
        if "python" in Path(candidate).name.lower() and os.path.isfile(candidate):
            return end
    return None


def _python_module_token(command: str) -> Optional[str]:
    """The module a PYTHON invocation runs via ``-m``, or None.

    Two things have to hold before a ``-m`` fingerprint is trusted. The
    executable must look like a Python interpreter — a ``grep``/``vim``/
    ``less`` whose *arguments* merely mention ``-m luxar.demos.demo_*`` must
    never be swept up as a demo (or validated as one) and killed. And the
    ``-m`` must still be an *interpreter* option: once Python has picked ``-c``
    code, a script path or stdin, the rest of the line is that program's argv,
    so ``python -c '…' -m luxar.demos.demo_lorenz`` is not running the demo.
    Both ``-m module`` and the attached ``-mmodule`` spelling are handled.
    """
    tokens = command.split()
    if not tokens:
        return None
    i = _interpreter_arg_start(tokens)
    if i is None:
        return None
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-m":
            return tokens[i + 1] if i + 1 < len(tokens) else None
        if tok.startswith("-m") and not tok.startswith("--"):
            return tok[2:]
        if tok in _PY_RUN_SELECTORS or not tok.startswith("-"):
            return None  # code / stdin / script path: no module is being run
        i += 2 if tok in _PY_OPTS_WITH_ARG else 1
    return None


def _is_luxar_module(module: Optional[str]) -> bool:
    """True for ``luxar`` itself or any ``luxar.*`` submodule."""
    return module is not None and (module == "luxar" or module.startswith("luxar."))


# Basenames of the console-script entry point (`pip install luxar` → `luxar`).
_LUXAR_SCRIPT_NAMES = frozenset({"luxar", "luxar.exe", "luxar-script.py"})


def _is_luxar_command(command: str) -> bool:
    """True for ``python -m luxar…`` AND for the ``luxar`` console script.

    A server's identity check cannot use the ``-m`` fingerprint alone: a user
    starts ``luxar serve`` through the console script, which ``ps`` renders as
    ``<python> <venv>/bin/luxar serve …``. The script path may contain spaces
    (a macOS hatch env lives under ``Application Support``), so — as for the
    interpreter — the next few tokens are rejoined before taking the basename.
    """
    if _is_luxar_module(_python_module_token(command)):
        return True
    tokens = command.split()
    if not tokens:
        return False
    if Path(tokens[0]).name in _LUXAR_SCRIPT_NAMES:
        return True
    i = _interpreter_arg_start(tokens)
    if i is None:
        return False
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] in _PY_RUN_SELECTORS:
            return False
        i += 2 if tokens[i] in _PY_OPTS_WITH_ARG else 1
    for end in range(i + 1, min(len(tokens), i + _MAX_EXE_TOKENS) + 1):
        if Path(" ".join(tokens[i:end])).name in _LUXAR_SCRIPT_NAMES:
            return True
    return False


def _demo_module_suffix(command: str) -> Optional[str]:
    """The ``demo_<suffix>`` a python ``-m luxar.demos.demo_*`` command runs."""
    module = _python_module_token(command)
    if module is None:
        return None
    match = _DEMO_MODULE_RE_STRICT.match(module)
    return match.group(1) if match else None


def _sweep_runs(snapshot: list[tuple[int, int, str]]) -> list[DemoRun]:
    """Demo groups found by fingerprinting command lines in ``snapshot``.

    Only the L1 demo script (its own group leader, per ``demo run``'s
    ``isolate_group=True``) is targeted — never the L0 ``luxar demo run``
    owner, whose group is the user's terminal job and must not be signalled.
    Killing the L1 group makes L0's ``run_child_process`` return on its own.

    Group leadership (``pid == pgid``) is *required*, not assumed: the kill is
    aimed at a whole process group, and a demo that does not lead its own
    group — one started by hand from a shell without job control, or under a
    supervisor — shares that group with unrelated siblings which must not die
    with it. Such a run is left for the user to stop; the registry covers
    every demo we launched ourselves.
    """
    runs: list[DemoRun] = []
    seen: set[int] = set()
    for pid, pgid, command in snapshot:
        suffix = _demo_module_suffix(command)
        if suffix is None or pid != pgid or pgid in seen:
            continue
        seen.add(pgid)
        runs.append(DemoRun(key=suffix, pgid=pgid, pid=0, started=0.0, source="sweep"))
    return runs


# ────────────────────────────── discovery ────────────────────────────────────
def _group_has_luxar_process(pgid: int, snapshot: list[tuple[int, int, str]]) -> bool:
    """True when ``pgid`` still contains a ``python -m luxar…`` process.

    This is the registry's pid-reuse guard: an entry whose group id has been
    recycled by an unrelated process must be pruned, not killed. A live demo
    group always contains at least one luxar module invocation (the demo
    script, and usually its ``luxar serve`` child), so requiring the exact
    python-invocation fingerprint loses nothing, while a substring match
    could sentence an innocent recycled group whose command merely mentions
    luxar text (an editor on a repo file, a grep, the viewer dev server).
    """
    return any(
        p == pgid and _is_luxar_module(_python_module_token(cmd))
        for _pid, p, cmd in snapshot
    )


def _entry_still_live(
    run: DemoRun,
    snapshot: list[tuple[int, int, str]],
    visible_pids: frozenset[int] = frozenset(),
) -> bool:
    """Whether a registry entry still names a live demo, in four regimes.

    With a process table in hand the answer is exact (and safe against a
    recycled pgid). Without one, POSIX can at least ask whether the group
    survives. Off POSIX there is no signal-based probe at all —
    ``os.kill(pid, 0)`` TERMINATES its target on Windows rather than testing it
    — but a pid *listing* (``visible_pids``, from :func:`_visible_pids`) still
    settles existence, so a record left behind by a reboot or a hard-killed
    owner is dropped rather than reported as a demo forever. What it cannot
    settle is identity, so a pid that IS listed is kept unverified and
    :func:`stop_run` still refuses to signal it. With no listing either, only
    an entry naming nobody is dropped.
    """
    if run.kind == "server":
        return _server_still_live(run, snapshot, visible_pids)
    if snapshot:
        return _group_has_luxar_process(run.pgid, snapshot)
    if can_kill_process_groups():
        return _group_alive(run.pgid)
    if visible_pids:
        # `pgid` is the demo process's own pid off POSIX (no group isolation);
        # `pid` is its `demo run` owner, which is EXPECTED to be gone — that is
        # what makes the entry survive — so it is the demo that gets probed.
        return run.pgid in visible_pids
    return bool(run.pid)


# Slack for comparing a process's start time (``ps`` reports whole seconds)
# with the registration time the server itself recorded after starting.
_START_TIME_SLACK_S = 2.0


def _parse_etime(text: str) -> Optional[float]:
    """Seconds from a ``ps -o etime=`` value (``[[DD-]HH:]MM:SS``), else None."""
    text = text.strip()
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return float(days * 86400 + seconds)


def _process_started_at(pid: int) -> Optional[float]:
    """Epoch start time of ``pid`` via ``ps -o etime=``, or None if unknown."""
    try:
        out = subprocess.run(  # nosec B603, B607  # fixed argv, no user input
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    elapsed = _parse_etime(out)
    return None if elapsed is None else time.time() - elapsed


def _server_identity(
    run: DemoRun, snapshot: list[tuple[int, int, str]]
) -> Optional[bool]:
    """Is ``run``'s PID still THE server that registered? None = can't tell.

    Three things must hold, because a PID is recycled freely and "some luxar
    process" is not proof: the process is a luxar invocation, its argv still
    carries the recorded subcommand, and it started no later than the entry
    was written (a recycled PID started after). Without a process table, or
    without a start time, the answer is unknown — never a licence to signal.
    """
    if not snapshot:
        return None
    row = next((cmd for pid, _g, cmd in snapshot if pid == run.pgid), None)
    if row is None or not _is_luxar_command(row):
        return False
    tokens = row.split()
    if not run.subcommand or not all(t in tokens for t in run.subcommand.split()):
        return False
    started_at = _process_started_at(run.pgid)
    if started_at is None:
        return None
    return started_at <= run.started + _START_TIME_SLACK_S


def _server_still_live(
    run: DemoRun,
    snapshot: list[tuple[int, int, str]],
    visible_pids: frozenset[int],
) -> bool:
    """Whether a server entry may still name its server (see above)."""
    identity = _server_identity(run, snapshot)
    if identity is not None:
        return identity
    if snapshot:
        return True  # matches but start time unknown: list it, never signal it
    if can_kill_process_groups():
        try:
            os.kill(run.pgid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False
    if visible_pids:
        return run.pgid in visible_pids
    return bool(run.pid)


def _drop_servers_inside_demos(
    runs: list[DemoRun], snapshot: list[tuple[int, int, str]]
) -> list[DemoRun]:
    """Hide a demo's own ``luxar serve`` child: stopping the demo stops it.

    Listing it too would show one demo as two things to stop, and stopping
    the server alone would leave the demo script printing its cleanup banner
    while the user expected the demo itself to go.
    """
    demo_groups = {r.pgid for r in runs if r.kind == "demo"}
    pgid_of = {pid: pgid for pid, pgid, _cmd in snapshot}
    return [
        r
        for r in runs
        if r.kind != "server" or pgid_of.get(r.pgid, -1) not in demo_groups
    ]


def _skip_server_entry(
    run: DemoRun, snapshot: list[tuple[int, int, str]], include_servers: bool
) -> bool:
    """True when ``discover_runs`` must not consider this server entry further.

    This process's own entry is never a target. Without ``include_servers``
    the entry is not listed, but a provably stale one is still pruned, so
    ``luxar demo stop`` keeps the shared registry tidy too.
    """
    if run.pgid == os.getpid():
        return True
    if include_servers:
        return False
    if _server_identity(run, snapshot) is False:
        unregister_run(run.path)
    return True


def discover_runs(
    runs_dir: Optional[Path] = None,
    snapshot: Optional[list[tuple[int, int, str]]] = None,
    include_servers: bool = False,
) -> list[DemoRun]:
    """All live demo runs (registry first, sweep for the rest), oldest first.

    Dead or hijacked registry entries are pruned as a side effect. The calling
    process's own group is excluded, so ``demo stop`` (or a ``pick_port``
    diagnosis running inside a demo being launched) never targets itself.
    ``snapshot`` lets a caller that already paid for a ps snapshot reuse it.

    ``include_servers`` adds the standalone servers (``kind="server"``) that
    ``luxar stop`` also manages, minus those that belong to a listed demo.
    Without it, server entries are neither listed nor pruned here.
    """
    runs_dir = runs_dir or DEMO_RUNS_DIR
    if snapshot is None:
        snapshot = _ps_snapshot()
    own_pgid = os.getpgrp() if hasattr(os, "getpgrp") else -1

    live: list[DemoRun] = []
    covered: set[int] = set()
    entries = _registry_runs(runs_dir)
    # Off POSIX neither a process table nor a group probe exists; a pid listing
    # is the only liveness question available (see `_visible_pids`). Pay for it
    # once per scan rather than once per entry, and never on the POSIX path.
    visible_pids = (
        _visible_pids()
        if entries and not snapshot and not can_kill_process_groups()
        else frozenset()
    )
    for run in entries:
        if run.kind == "server" and _skip_server_entry(run, snapshot, include_servers):
            continue
        if run.kind == "demo" and run.pgid == own_pgid:
            covered.add(run.pgid)
            continue
        if not _entry_still_live(run, snapshot, visible_pids):
            unregister_run(run.path)
            continue
        covered.add(run.pgid)
        live.append(run)

    for run in _sweep_runs(snapshot):
        if run.pgid != own_pgid and run.pgid not in covered:
            live.append(run)

    if include_servers:
        live = _drop_servers_inside_demos(live, snapshot)
    return sorted(live, key=lambda r: r.started)


def _group_alive(pgid: int) -> bool:
    """True while any member of ``pgid`` survives (POSIX; False elsewhere)."""
    # killpg(0, 0) probes the CALLER'S own group (always "alive"), which would
    # keep a corrupt pgid<=1 registry entry forever; treat such ids as dead.
    if pgid <= 1 or not can_kill_process_groups():
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ─────────────────────────────── stopping ────────────────────────────────────
def stop_run(run: DemoRun) -> bool:
    """Kill one demo's process group; returns True when it is gone.

    Uses the same SIGINT → SIGTERM → SIGKILL escalation as Ctrl-C teardown, so
    uvicorn gets its graceful shutdown first. On success the registry entry is
    pruned; the L0 ``demo run`` owner (if any) exits by itself once its child
    group dies.

    Re-validates the group at KILL time: `demo stop`'s confirmation prompt can
    sit for minutes between discovery and this call, long enough for the demo
    to exit and (in principle) its group id to be recycled by an innocent
    process — which must not inherit the death sentence.

    Off POSIX nothing is signalled at all. There are no process groups to
    fingerprint there, and no way to even ASK whether a recorded pid is still
    the demo: on Windows ``os.kill(pid, 0)`` is a hard ``TerminateProcess``,
    not a probe. Meanwhile a registry entry outlives a reboot or a hard-killed
    owner, and pids get recycled freely — so signalling one blind would
    eventually terminate whatever innocent process inherited the number. The
    run is reported as not stopped instead (``demo stop`` then prints the
    manual command), and its entry is kept so the listing keeps naming it.
    """
    if run.kind == "server":
        return _stop_server(run)
    if can_kill_process_groups():
        snapshot = _ps_snapshot()
        if snapshot and not _group_has_luxar_process(run.pgid, snapshot):
            # No longer a demo group (exited, possibly recycled): nothing to
            # stop. Prune the entry and report success.
            unregister_run(run.path)
            run.pruned = True
            return True
        gone = terminate_process_group(run.pgid)
    else:
        gone = False
    if gone:
        unregister_run(run.path)
    return gone


def _stop_server(run: DemoRun) -> bool:
    """Stop a standalone server by PID, re-validating its identity first."""
    if not can_kill_process_groups():
        return False  # same no-blind-signal rule as for demos, see stop_run
    identity = _server_identity(run, _ps_snapshot())
    if identity is False:
        unregister_run(run.path)  # gone, or the PID now belongs to someone else
        run.pruned = True
        return True
    if identity is None:
        return False  # cannot verify who holds the PID: refuse to signal it
    gone = terminate_process(run.pgid)
    if gone:
        unregister_run(run.path)
    return gone


# ───────────────────────── busy-port diagnosis ───────────────────────────────
def describe_port_holder(port: int) -> Optional[str]:
    """Name what holds ``port`` when it looks luxar-owned; None otherwise.

    Best-effort (lsof + ps, both optional): feeds the ``pick_port`` "port busy"
    warning so a shifted port explains *why* — the difference between "the
    wrong demo came up" confusion and an actionable one-liner.
    """
    try:
        out = subprocess.run(  # nosec B603, B607  # fixed argv, no user input
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    holder_pids = {int(p) for p in out.split() if p.isdigit()}
    if not holder_pids:
        return None

    snapshot = _ps_snapshot()
    by_pid = {pid: (pgid, cmd) for pid, pgid, cmd in snapshot}
    runs = discover_runs(snapshot=snapshot, include_servers=True)
    runs_by_pgid = {run.pgid: run for run in runs if run.kind == "demo"}
    servers_by_pid = {run.pgid: run for run in runs if run.kind == "server"}
    for pid in sorted(holder_pids):
        pgid, cmd = by_pid.get(pid, (-1, ""))
        run = runs_by_pgid.get(pgid)
        if run is not None:
            return (
                f"Port {port} is held by demo '{run.key}' (PID {pid}) — "
                "run `luxar demo stop` to clear it."
            )
        server = servers_by_pid.get(pid)
        if server is not None:
            return (
                f"Port {port} is held by `luxar {server.key}` (PID {pid}) — "
                "run `luxar stop` to clear it."
            )
        if _is_luxar_module(_python_module_token(cmd)) and "serve" in cmd.split():
            return f"Port {port} is held by another `luxar serve` (PID {pid})."
    return None
