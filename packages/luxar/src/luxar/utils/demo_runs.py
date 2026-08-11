"""Track and stop running ``luxar demo`` processes (stdlib-only).

``luxar demo run`` spawns its demo script as an isolated process-group leader
(see :mod:`luxar.utils.process`), and that script in turn spawns
``luxar serve``. Ctrl-C teardown is deterministic — but a demo the user simply
*forgot* in another terminal never receives a signal, and keeps its ports, its
memory and its GPU. Re-running that demo then shifts to a neighbouring port
(a demo's derived port pair is stable, see :func:`luxar.utils.demos.demo_ports`)
while the old tab keeps serving the stale scene. ``luxar demo stop`` fixes
that; this module is its discovery + kill engine, with two complementary
sources:

- **Registry** (primary): ``demo run`` / ``run-all`` drop a JSON pidfile per
  launch under ``~/.cache/luxar/running/`` and remove it on exit. Precise —
  knows the demo key and the process group to kill.
- **Process-table sweep** (fallback): a ``ps`` scan for
  ``python -m luxar.demos.demo_*`` command lines catches runs that predate the
  registry or whose pidfile was lost.

Kept in ``luxar.utils`` — not ``luxar.cli`` — and stdlib-only for the same
reason as :mod:`luxar.utils.process`: importing it must not drag in
uvicorn/fastapi or the demo registry.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess  # nosec B404  # fixed argv (ps/lsof), never shell, parsed
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .process import can_kill_process_groups, proc_table, terminate_process_group

# Matches luxar.demos.registry.DEMO_CACHE_ROOT (not imported: that module pulls
# in the whole demo table, and cli/utils.py imports us on the serve hot path).
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
            runs.append(
                DemoRun(
                    key=str(entry["key"]),
                    pgid=pgid,
                    pid=int(entry.get("pid", 0)),
                    started=float(entry.get("started", 0.0)),
                    source="registry",
                    path=path,
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

    Falls back to ``/proc`` when ``ps`` is missing (a slim container), times
    out, or prints something we cannot parse (BusyBox takes different flags).
    An empty snapshot does not merely disable the sweep: it also strips
    discovery of the identity check that keeps a recycled pgid from being
    killed, so it is worth a second source before giving up.
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


# Interpreter options whose argument is a SEPARATE token; stepping over it
# keeps a `python -W ignore -m luxar.demos.demo_x` recognisable as a demo.
_PY_OPTS_WITH_ARG = frozenset({"-W", "-X"})

# Tokens after which Python has chosen what to run, so everything left on the
# line is the program's own argv: a code string, end-of-options, or stdin.
_PY_RUN_SELECTORS = frozenset({"-c", "--", "-"})


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
    if not tokens or "python" not in Path(tokens[0]).name.lower():
        return None
    i = 1
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


def _entry_still_live(run: DemoRun, snapshot: list[tuple[int, int, str]]) -> bool:
    """Whether a registry entry still names a live demo, in three regimes.

    With a process table in hand the answer is exact (and safe against a
    recycled pgid). Without one, POSIX can at least ask whether the group
    survives. Off POSIX there is no probe at all — ``os.kill(pid, 0)``
    TERMINATES its target on Windows rather than testing it — so an entry
    naming an owner pid is kept for ``stop_run`` to act on, and only one that
    names nobody is dropped.
    """
    if snapshot:
        return _group_has_luxar_process(run.pgid, snapshot)
    if can_kill_process_groups():
        return _group_alive(run.pgid)
    return bool(run.pid)


def discover_runs(
    runs_dir: Optional[Path] = None,
    snapshot: Optional[list[tuple[int, int, str]]] = None,
) -> list[DemoRun]:
    """All live demo runs (registry first, sweep for the rest), oldest first.

    Dead or hijacked registry entries are pruned as a side effect. The calling
    process's own group is excluded, so ``demo stop`` (or a ``pick_port``
    diagnosis running inside a demo being launched) never targets itself.
    ``snapshot`` lets a caller that already paid for a ps snapshot reuse it.
    """
    runs_dir = runs_dir or DEMO_RUNS_DIR
    if snapshot is None:
        snapshot = _ps_snapshot()
    own_pgid = os.getpgrp() if hasattr(os, "getpgrp") else -1

    live: list[DemoRun] = []
    covered: set[int] = set()
    for run in _registry_runs(runs_dir):
        if run.pgid == own_pgid:
            covered.add(run.pgid)
            continue
        if not _entry_still_live(run, snapshot):
            unregister_run(run.path)
            continue
        covered.add(run.pgid)
        live.append(run)

    for run in _sweep_runs(snapshot):
        if run.pgid != own_pgid and run.pgid not in covered:
            live.append(run)

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
    """
    if can_kill_process_groups():
        snapshot = _ps_snapshot()
        if snapshot and not _group_has_luxar_process(run.pgid, snapshot):
            # No longer a demo group (exited, possibly recycled): nothing to
            # stop. Prune the entry and report success.
            unregister_run(run.path)
            return True
        gone = terminate_process_group(run.pgid)
    elif run.pid:
        # Windows degrade: no process groups — terminate the registered owner.
        try:
            os.kill(run.pid, signal.SIGTERM)
            gone = True
        except ProcessLookupError:
            gone = True  # already dead: prune the stale entry
        except OSError:
            gone = False
    else:
        gone = False
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
    runs_by_pgid = {run.pgid: run for run in discover_runs(snapshot=snapshot)}
    for pid in sorted(holder_pids):
        pgid, cmd = by_pid.get(pid, (-1, ""))
        run = runs_by_pgid.get(pgid)
        if run is not None:
            return (
                f"Port {port} is held by demo '{run.key}' (PID {pid}) — "
                "run `luxar demo stop` to clear it."
            )
        if _is_luxar_module(_python_module_token(cmd)) and "serve" in cmd.split():
            return f"Port {port} is held by another `luxar serve` (PID {pid})."
    return None
