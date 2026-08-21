"""``luxar demo`` — list, inspect, run, and manage Luxar's bundled demos.

Bare ``luxar demo`` prints the demo table (= ``luxar demo list``). Demos are
executed as subprocesses (``python -m luxar.demos.demo_<name>``), never
imported: several parse ``sys.argv`` or create cache directories at import
time. Extra CLI args after the key are forwarded verbatim to the demo script.
"""

from __future__ import annotations

import importlib
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from ..demos import registry
from ..demos.registry import DemoInfo
from ..utils import demo_runs
from ..utils.data_fetch import LOCAL_FIT_DIRNAME
from ..utils.process import can_kill_process_groups, run_child_process
from .demo_render import (
    STATUS_BUILT,
    STATUS_CACHED,
    demo_console,
    render_caches,
    render_catalogue,
    render_dependencies,
    render_detail,
)
from .utils import format_memory_size

app_demo = typer.Typer(
    help=(
        "Run and manage Luxar's bundled demos "
        "(list / info / run / stop / deps / cache)."
    ),
    no_args_is_help=False,
)
cache_app = typer.Typer(help="Inspect and clear demo download/compute caches.")
app_demo.add_typer(cache_app, name="cache")


# ────────────────────────────── rendering ────────────────────────────────────
def _safe_output_paths(info: DemoInfo) -> list[Path]:
    """Resolve a demo's output paths, or ``[]`` when the root can't be found.

    ``registry.demo_output_paths`` → ``get_demos_output_dir`` → ``get_project_root``
    raises ``RuntimeError`` from an installed wheel (no ``pyproject.toml``). An
    empty list is the right answer there: "no known outputs to skip/clear".
    """
    try:
        return registry.demo_output_paths(info)
    except RuntimeError:
        return []


def _empty_dirs(cache_dir: Path, deleted: Optional[set[Path]] = None) -> list[Path]:
    """Directories a sweep of ``cache_dir`` would remove, deepest first.

    A download killed between ``tempfile.mkdtemp`` and its cleanup (SIGKILL, OOM,
    power loss) leaks the private staging directory that
    ``luxar.demos._graph_common.download_file`` streams into. Once its staged
    file is cleared the directory holds no bytes, so it is never a deletion
    *target* — but it keeps the cache dir non-empty, and a cache dir that never
    becomes empty is never removed, so :func:`_status` reports the demo as
    ``cached`` forever. ``cache_dir`` itself is included once nothing is left
    under it, for the same reason.

    ``deleted`` names files that are *about to* be unlinked, and so counts them
    as already gone. The real sweep runs after the deletions and needs none of
    it; both modes print their summary *before* deleting anything, and without
    it that preview would miss every directory that the deletions are what
    empties — above all the cache dir itself.
    """
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        return []
    gone = deleted if deleted is not None else set()
    removable: set[Path] = set()
    # Bottom-up: a directory is removable when it holds no files (that are not
    # already on their way out) and every subdirectory is itself removable.
    # ``os.walk`` does not follow symlinks, so a symlinked subdirectory is never
    # walked and never counted as removable.
    for parent, subdirs, files in os.walk(cache_dir, topdown=False):
        path = Path(parent)
        if all(path / f in gone for f in files) and all(
            path / s in removable for s in subdirs
        ):
            removable.add(path)
    return sorted(removable, reverse=True)


def _status(info: DemoInfo) -> str:
    """Whether this demo already has cached inputs or a generated output."""
    if any(p.exists() for p in _safe_output_paths(info)):
        return STATUS_BUILT
    if any(d.exists() for d in registry.demo_cache_dirs(info)):
        return STATUS_CACHED
    return ""


def _demos_or_exit() -> list[DemoInfo]:
    """All demos, or a clean error+exit if any ``DEMO_META`` is malformed.

    Without this, a single broken demo file makes ``registry.iter_demos`` raise
    ``DemoMetaError`` — an uncaught traceback for *every* ``demo`` subcommand.
    """
    try:
        return registry.iter_demos()
    except registry.DemoMetaError as e:
        aprint(f"❌ Broken demo metadata: {e}")
        raise typer.Exit(1) from e


def _starter_key(demos: Sequence[DemoInfo]) -> Optional[str]:
    """The cheapest demo to suggest by name in the footer's ``run`` hint.

    "Cheapest" = nothing to download, no GPU, no data to place by hand — the one
    class of demo that is guaranteed to work on a fresh checkout with no
    network.

    Sorted here rather than trusting the caller's order, so "the suggestion does
    not move around" is a property of this function instead of an accident of
    every call site passing key-sorted registry output: shuffling the input
    otherwise yields a dozen different suggestions, while the catalogue beside
    it stays byte-identical because `render_catalogue` sorts internally.
    """
    for demo in sorted(demos, key=lambda d: d.key):
        if not demo.download_mb and demo.gpu == "none" and not demo.local_data:
            return demo.key
    return None


def _print_table(demos: list[DemoInfo]) -> None:
    """Print the demo catalogue, grouped into category sections."""
    statuses = {d.key: _status(d) for d in demos}
    render_catalogue(demo_console(), demos, statuses, example_key=_starter_key(demos))


def _resolve_or_exit(key_or_index: str) -> DemoInfo:
    try:
        return registry.get_demo(key_or_index)
    except KeyError as e:
        aprint(f"❌ {e}")
        raise typer.Exit(1) from e
    except registry.DemoMetaError as e:
        aprint(f"❌ Broken demo metadata: {e}")
        raise typer.Exit(1) from e


def _run_registered_demo(key: str, cmd: list[str], label: str) -> int:
    """Run a demo subprocess with a `luxar demo stop` registry entry around it.

    The entry is written the instant the group is spawned (via ``on_spawn``,
    where the child PID *is* the new pgid) and removed on every exit path, so
    the registry only ever names groups that outlived their owner — exactly
    the forgotten/orphaned runs ``demo stop`` exists to clear.
    """
    entry: list[Optional[Path]] = [None]

    def _register(pid: int) -> None:
        """``on_spawn`` hook: record the just-created demo process group."""
        entry[0] = demo_runs.register_run(key, pid)

    try:
        return run_child_process(cmd, label=label, on_spawn=_register)
    finally:
        demo_runs.unregister_run(entry[0])


# ─────────────────────────────── callback ────────────────────────────────────
@app_demo.callback(invoke_without_command=True)
def demo_callback(ctx: typer.Context) -> None:
    """Show the demo table when invoked with no subcommand."""
    if ctx.invoked_subcommand is None:
        _print_table(_demos_or_exit())
        raise typer.Exit(0)


# ──────────────────────────────── list ───────────────────────────────────────
@app_demo.command("list")
def list_demos(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    geometry: Optional[str] = typer.Option(
        None, "--geometry", "-g", help="Filter by geometry"
    ),
) -> None:
    """List all bundled demos as a table."""
    demos = _demos_or_exit()
    if category:
        demos = [d for d in demos if d.category == category]
    if geometry:
        demos = [d for d in demos if d.geometry == geometry]
    if not demos:
        aprint("No demos match that filter.")
        raise typer.Exit(0)
    _print_table(demos)


# ──────────────────────────────── info ───────────────────────────────────────
@app_demo.command("info")
def demo_info(
    key: str = typer.Argument(..., help="Demo key or index (see `luxar demo list`)"),
) -> None:
    """Show full details for one demo."""
    info = _resolve_or_exit(key)
    render_detail(demo_console(), info, _status(info))


# ──────────────────────────────── run ────────────────────────────────────────
@app_demo.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def demo_run(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Demo key or index (see `luxar demo list`)"),
) -> None:
    """Run a demo by key or index; extra args are forwarded to the demo.

    Example: luxar demo run gsplats_3d_tribolium_embryo -- --recompute --no-serve
    """
    info = _resolve_or_exit(key)
    extra = list(ctx.args)
    aprint(f"🎬 {info.title}  [{info.key}]")
    aprint(f"   {info.description}")
    if extra:
        aprint(f"➡️  Forwarding args to the demo: {' '.join(extra)}")
    cmd = [sys.executable, "-m", info.module, *extra]
    # isolate_group=True: the demo (and the `luxar serve` it spawns) run in
    # their own process group, so a terminal Ctrl-C reaches only this command,
    # which then tears the whole subtree down deterministically (SIGINT →
    # SIGTERM → SIGKILL). Without this, a hung uvicorn is orphaned on its port.
    # run_child_process also maps signal death to 128+N (raw -N truncates).
    code = _run_registered_demo(info.key, cmd, label=f"demo '{info.key}'")
    if code != 0:
        raise typer.Exit(code)


# ─────────────────────────────── run-all ─────────────────────────────────────
@app_demo.command("run-all")
def demo_run_all(
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--force",
        help="Skip demos whose output scenes already exist.",
    ),
    keep_going: bool = typer.Option(
        True,
        "--keep-going/--fail-fast",
        help="Continue after a failing demo (default) or stop at the first.",
    ),
    include_gpu: bool = typer.Option(
        False,
        "--include-gpu/--skip-gpu",
        help="Also run demos that require a GPU (skipped by default — they "
        "fail unattended on a CPU-only machine).",
    ),
    max_download_mb: int = typer.Option(
        200,
        "--max-download-mb",
        help="Skip demos whose download exceeds this many MB (0 = no limit).",
    ),
) -> None:
    """Generate every demo's dataset (``--no-serve``), for batch/gallery builds.

    Skips demos that can't run unattended: manual/Kaggle data, GPU-required
    demos (unless ``--include-gpu``), and large downloads (over
    ``--max-download-mb``). By default also skips demos whose outputs exist.
    """
    demos = _demos_or_exit()
    ran, skipped, failed = 0, 0, []
    for d in demos:
        if d.local_data in ("manual-file", "kaggle-auth"):
            aprint(f"⏭️  {d.key}: needs {d.local_data}; skipping")
            skipped += 1
            continue
        if d.gpu == "required" and not include_gpu:
            aprint(f"⏭️  {d.key}: needs GPU (pass --include-gpu); skipping")
            skipped += 1
            continue
        if max_download_mb and d.download_mb > max_download_mb:
            aprint(
                f"⏭️  {d.key}: download {d.download_mb}MB > "
                f"{max_download_mb}MB limit; skipping"
            )
            skipped += 1
            continue
        outs = _safe_output_paths(d)
        if skip_existing and outs and all(p.exists() for p in outs):
            aprint(f"⏭️  {d.key}: output exists; skipping")
            skipped += 1
            continue
        aprint(f"▶️  {d.key}: {d.title}")
        # Default isolate_group=True: --no-serve demos spawn no server, but
        # group isolation still gives a clean Ctrl-C (only this batch runner
        # gets SIGINT) and a deterministic per-demo teardown.
        code = _run_registered_demo(
            d.key, [sys.executable, "-m", d.module, "--no-serve"], label=d.key
        )
        if code == 130:
            # 130 is our Ctrl-C convention (run_child_process maps SIGINT and a
            # KeyboardInterrupt to it): stop the whole batch, matching `demo
            # run`. A demo that deliberately exits 130 for another reason would
            # also stop the batch — acceptable, as bundled demos never do.
            aprint(f"\n🛑 Interrupted during {d.key}; stopping run-all.")
            raise typer.Exit(130)
        if code != 0:
            failed.append(d.key)
            aprint(f"❌ {d.key}: exited {code}")
            if not keep_going:
                break
        else:
            ran += 1
    aprint(f"\n✅ ran {ran}, skipped {skipped}, failed {len(failed)}")
    if failed:
        aprint(f"   failed: {', '.join(failed)}")
        raise typer.Exit(1)


# ──────────────────────────────── stop ───────────────────────────────────────
def _run_age(started: float) -> str:
    """Compact "how long has this been running" label for the stop listing."""
    if not started:
        return "unknown age"
    seconds = max(0.0, time.time() - started)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _translate_sweep_keys(runs: list["demo_runs.DemoRun"]) -> list["demo_runs.DemoRun"]:
    """Rewrite swept runs' module suffixes into real demo keys, in place.

    Swept runs carry the demo MODULE suffix, which is not always the demo key
    (demo_4d_fractals.py declares key "fractals_4d"). Translate through the
    demo table so display and ``stop <key>`` filtering use real keys; a broken
    table must not stop ``demo stop`` from working, so fall back to the suffix.
    """
    try:
        stem_to_key = {
            d.module.rsplit(".", 1)[-1]: d.key for d in registry.iter_demos()
        }
    except registry.DemoMetaError:
        stem_to_key = {}
    for r in runs:
        if r.source == "sweep":
            r.key = stem_to_key.get(f"demo_{r.key}", r.key)
    return runs


def _manual_stop_hint(run: "demo_runs.DemoRun") -> str:
    """Command the user can run by hand for a demo we could not stop.

    Off POSIX ``demo stop`` never signals anything (there is no way to check a
    recorded pid still belongs to the demo before a hard terminate, see
    ``demo_runs.stop_run``), so the hint has to be the local one — a
    `kill -9 -<pgid>` there is advice that cannot even be typed.
    """
    if can_kill_process_groups():
        return f"kill -9 -{run.pgid}"
    return f"taskkill /F /T /PID {run.pgid}"


def _stop_all(runs: list["demo_runs.DemoRun"]) -> list["demo_runs.DemoRun"]:
    """Kill every run's process group; returns the runs that survived."""
    survivors: list[demo_runs.DemoRun] = []
    for r in runs:
        if demo_runs.stop_run(r):
            aprint(f"   ✅ stopped {r.key}")
        else:
            survivors.append(r)
            aprint(f"   ❌ could not stop {r.key} (pgid {r.pgid})")
    return survivors


@app_demo.command("stop")
def demo_stop(
    key: Optional[str] = typer.Argument(
        None, help="Stop only this demo key/index (default: every running demo)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List running demos; stop nothing."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Stop running demos and free their ports (and memory, and GPU).

    Finds every live demo — via the registry ``demo run`` maintains, plus a
    process-table sweep for strays with no registry entry — and tears each
    one's process group down with the same SIGINT → SIGTERM → SIGKILL
    escalation Ctrl-C uses. The go-to fix when a new demo warns "port busy"
    because an earlier one is still running in a forgotten terminal.
    """
    only = _resolve_or_exit(key).key if key else None
    runs = _translate_sweep_keys(demo_runs.discover_runs())
    if only is not None:
        runs = [r for r in runs if r.key == only]
    if not runs:
        target = f"demo '{only}'" if only else "demos"
        aprint(f"✅ No running {target} found.")
        raise typer.Exit(0)

    noun = "demo" if len(runs) == 1 else "demos"
    aprint(f"🛑 [Luxar] {len(runs)} running {noun}:")
    for r in runs:
        origin = "" if r.source == "registry" else "  (found by process sweep)"
        aprint(f"   {r.key:<32} pgid {r.pgid:<7} {_run_age(r.started):>11}{origin}")
    if dry_run:
        aprint("\n(--dry-run: nothing stopped)")
        raise typer.Exit(0)
    # Several agents/people may run demos on this machine concurrently, and
    # "stop everything" would take a colleague's live server down with yours —
    # so the listing above always gets a confirmation, like `cache clear`.
    if not yes and not typer.confirm("\nStop these?"):
        aprint("Aborted.")
        raise typer.Exit(0)

    survivors = _stop_all(runs)
    if survivors:
        names = ", ".join(r.key for r in survivors)
        hints = "; ".join(_manual_stop_hint(r) for r in survivors)
        aprint(f"⚠️  {len(survivors)} still running: {names} — try `{hints}`.")
        raise typer.Exit(1)
    aprint("✅ All demos stopped; their ports are free again.")


# ──────────────────────────────── deps ──────────────────────────────────────
def _luxar_checkout_root() -> Path | None:
    """The Luxar dev-checkout root, but only if it owns the imported ``luxar``.

    ``get_project_root`` walks up to the first ancestor with a ``pyproject.toml``,
    which for a NON-editable install inside a project-local virtualenv
    (``<proj>/.venv/.../site-packages/luxar`` — the default uv/poetry layout)
    is the *user's own* project, not Luxar. Trust the root only when it actually
    owns the imported package: a real checkout is ``<root>/packages/luxar/src/
    luxar``. Otherwise (mismatch, or no root at all) return ``None``.
    """
    import luxar

    try:
        from ..utils.paths import get_project_root

        root = get_project_root()
    except RuntimeError:
        return None
    pkg_dir = Path(luxar.__file__).resolve().parent
    if (root / "packages" / "luxar" / "src" / "luxar").resolve() == pkg_dir:
        return root
    return None


def _pip_install_cmd(extras: list[str]) -> list[str]:
    """The pip command that installs ``extras``, editable only in a checkout.

    A genuine dev checkout must install ``-e <root>[…]``: a plain ``luxar[demos]``
    would fetch the *published* wheel from PyPI and shadow the tree the user is
    editing. The checkout is trusted only when the discovered root owns the
    imported ``luxar`` (see ``_luxar_checkout_root``); otherwise name the
    distribution so we never editable-install an unrelated project.
    """
    joined = ",".join(extras)
    base = [sys.executable, "-m", "pip", "install"]
    root = _luxar_checkout_root()
    if root is not None:
        return [*base, "-e", f"{root}[{joined}]"]
    return [*base, f"luxar[{joined}]"]


def _pip_install_requirement_cmd(requirement: str) -> list[str]:
    """The pip command that installs one exact tabled requirement."""
    return [sys.executable, "-m", "pip", "install", requirement]


@app_demo.command("deps")
def demo_deps(
    extra: Optional[str] = typer.Option(
        None,
        "--extra",
        # Long form only, deliberately: `-e` already means --encoding on the
        # gsplat commands, and means "editable" to the pip this command drives.
        help="Only consider one extra (demos / io / gsplats). Default: all.",
    ),
    only: Optional[str] = typer.Option(
        None,
        "--only",
        metavar="MODULE",
        help=(
            "Only consider one import module (case-insensitive) and install its "
            "exact constrained requirement. Cannot be combined with --extra."
        ),
    ),
    install: bool = typer.Option(
        False,
        "--install",
        help="Install the extras, or the exact --only requirement, for unmet rows.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="With --install, print the pip command only."
    ),
) -> None:
    """Report (and optionally install) the demos' optional dependencies.

    Demos deliberately keep heavyweight packages out of the core install, so a
    fresh checkout can run `luxar demo list` but not every demo. This reports
    what is missing or out of date and, with ``--install``, installs the Luxar
    extras that provide it. ``--only MODULE`` instead installs that module's
    exact constrained requirement, avoiding a whole-extra install for one gap.
    """
    from ..demos import DependencyStatus, extras_for, survey

    if dry_run and not install:
        # Silently ignoring a flag the user typed is worse than saying so.
        aprint("ℹ️  --dry-run only applies with --install; reporting only.")

    # Extra names are lowercase by PEP 685, so accept any casing the user types.
    # Normalize blank filters to None: neither `--extra ""` nor `--only ""`
    # should become an active filter with a surprising empty/orphan-only result.
    extra = (extra.strip().lower() or None) if extra else None
    only = (only.strip() or None) if only else None
    if extra is not None and only is not None:
        aprint("❌ --extra and --only cannot be combined.")
        raise typer.Exit(2)

    def selected_rows() -> list[DependencyStatus]:
        current = survey(extra)
        if only is None:
            return current
        folded = only.casefold()
        return [row for row in current if row.module.casefold() == folded]

    rows = selected_rows()
    if not rows:
        all_rows = survey()
        if only is not None:
            aprint(
                f"❌ Unknown optional dependency module {only!r}. Use the import "
                "name (for example PIL, umap, or sklearn), not the distribution "
                "name. Valid modules: "
                + ", ".join(row.module for row in all_rows)
                + "."
            )
        else:
            known = extras_for(all_rows) or ["(none)"]
            aprint(
                f"❌ No known dependencies for extra {extra!r}. "
                f"Valid extras: {', '.join(known)}."
            )
        raise typer.Exit(1)

    # "Unmet" = missing OR installed-but-below-its-pin (OUTDATED). Both need the
    # extra (re)installed, so both drive the same action set and exit code.
    unmet = [r for r in rows if not r.satisfied]
    plural = "dependency" if len(rows) == 1 else "dependencies"
    render_dependencies(demo_console(), rows)
    aprint("")

    if not unmet:
        # Phrased to avoid subject-verb agreement on the count ("1 dependency
        # are installed"), which a plural-noun-only fix leaves behind.
        aprint(f"✅ Nothing missing — all {len(rows)} optional {plural} installed.")
        raise typer.Exit(0)

    aprint(f"⚠️  {len(unmet)} missing or outdated: {', '.join(r.module for r in unmet)}")

    # Specs outside every extra can't be installed via luxar[…]; name them.
    orphans = [r for r in unmet if not r.spec.extra]
    extras = extras_for(unmet)
    if orphans:
        aprint(
            "   Not in any extra (install individually): "
            + ", ".join(f"'{r.spec.spec}'" for r in orphans)
        )

    targeted = only is not None
    if targeted:
        # `--only` selects exactly one table row and installs its constrained
        # requirement directly — including an orphan such as gdown.
        cmd = _pip_install_requirement_cmd(unmet[0].spec.spec)
        install_hint = f"luxar demo deps --only {unmet[0].module} --install"
    elif extras:
        cmd = _pip_install_cmd(extras)
        # Keep an active --extra filter in the hint: the bare form would install
        # every unmet extra, more than the pip command shown right beside it.
        install_hint = (
            "luxar demo deps --install"
            if extra is None
            else f"luxar demo deps --extra {extra} --install"
        )
    else:
        # Generic --install manages Luxar extras only. An orphan-only report has
        # no command to run, so it is a successful no-op rather than the old
        # exit-1 special case (the same orphan alongside an installed extra
        # already exited 0). --only provides the explicit individual path.
        noun = "requirement" if len(unmet) == 1 else "requirements"
        aprint(
            f"\nℹ️  No Luxar extra provides the unmet {noun}; "
            + ("nothing installed." if install else "install individually:")
        )
        for row in unmet:
            hint = f"luxar demo deps --only {row.module} --install"
            aprint(f"   {hint}")
            if not install:
                aprint(
                    f"   Or directly: {shlex.join(_pip_install_requirement_cmd(row.spec.spec))}"
                )
        raise typer.Exit(0 if install else 1)

    # shlex.join, not " ".join: the interpreter path and the checkout root both
    # routinely contain spaces (e.g. "Application Support"), and the extras
    # brackets are shell globs — an unquoted line would not paste back in.
    shown = shlex.join(cmd)
    if not install:
        aprint(f"\n   Install with:  {install_hint}")
        aprint(f"   Or directly:   {shown}")
        raise typer.Exit(1)

    aprint(f"\n▶️  {shown}")
    if dry_run:
        aprint("(--dry-run: nothing installed)")
        raise typer.Exit(0)
    code = run_child_process(cmd, label="pip install")
    if code != 0:
        aprint(f"❌ pip exited {code}")
        raise typer.Exit(code)

    # pip wrote into site-packages after our finders cached its contents, so a
    # re-survey without this reports everything still missing.
    importlib.invalidate_caches()
    left = [r for r in selected_rows() if not r.satisfied]
    # Judge an extras install ONLY on what it was asked to provide. An orphan
    # spec was never in that pip command, while a targeted install did attempt
    # its exact row and therefore must verify it like any other requested item.
    still = left if targeted else [r for r in left if r.spec.extra in extras]
    if still:
        aprint(
            "⚠️  Still missing or outdated after install: "
            f"{', '.join(r.module for r in still)}"
        )
        raise typer.Exit(1)
    if targeted:
        aprint(f"✅ Installed: '{unmet[0].spec.spec}'.")
    else:
        aprint(f"✅ Installed: {', '.join(f'luxar[{e}]' for e in extras)}.")
    # Don't claim completeness while an orphan is still absent — the extras
    # install genuinely could not cover it.
    if left:
        aprint(
            "ℹ️  Still to install by hand: "
            + ", ".join(f"'{r.spec.spec}'" for r in left)
        )


# ─────────────────────────────── cache list ──────────────────────────────────
@cache_app.command("list")
def cache_list() -> None:
    """Inventory the demo caches under ~/.cache/luxar/."""
    try:
        entries = registry.inventory_caches()
    except registry.DemoMetaError as exc:
        aprint(f"❌ Broken demo metadata: {exc}")
        raise typer.Exit(1) from exc
    if not entries:
        aprint(f"No demo cache directories under {registry.DEMO_CACHE_ROOT}")
        raise typer.Exit(0)
    render_caches(demo_console(), entries, registry.DEMO_CACHE_ROOT, format_memory_size)


# ─────────────────────────────── cache clear ─────────────────────────────────
def _clearable_cache_dirs(info: DemoInfo, kept: list[str]) -> list[Path]:
    """A demo's cache dirs minus its hand-placed inputs, recording what was kept.

    A :data:`registry.PROTECTED_INPUT_DIRS` directory holds bytes the user put
    there by hand with no download to get them back, so clearing it is not a
    cache eviction but data loss. It is dropped here — before the caller can
    collect its files OR add it to the sweep list, since the empty-directory
    sweep would otherwise rmdir an input directory that is (or has just become)
    empty. Kept names accumulate in ``kept``, in encounter order and deduplicated,
    for one notice each; only a directory that actually exists is worth a notice,
    as a demo can declare the name on a machine that never received the file.
    """
    clearable: list[Path] = []
    for cache_dir in registry.demo_cache_dirs(info):
        if cache_dir.name in registry.PROTECTED_INPUT_DIRS:
            if cache_dir.exists() and cache_dir.name not in kept:
                kept.append(cache_dir.name)
            continue
        clearable.append(cache_dir)
    return clearable


def _orphan_targets(
    entries: list["registry.CacheEntry"], kept: list[str]
) -> list[tuple[Path, int, str]]:
    """Deletion targets for ``--orphans``, sparing hand-placed inputs.

    A protected entry is claimed by definition, so it is not an orphan however
    empty its ``demo_keys`` is — that is exactly the state that used to make
    ``clear --orphans`` rmtree the Gaia catalog. Only that state earns a notice
    (see :func:`_clearable_cache_dirs`): a protected dir some demo *does* claim
    was never a candidate here, and announcing it would mean `clear <other-key>
    --orphans` volunteering advice about a directory the user never selected.
    """
    targets: list[tuple[Path, int, str]] = []
    for e in entries:
        if e.protected:
            if not e.demo_keys and e.path.name not in kept:
                kept.append(e.path.name)
        elif not e.demo_keys:
            targets.append((e.path, e.size_bytes, f"ORPHAN {e.path.name}"))
    return targets


@cache_app.command("clear")
def cache_clear(
    keys: Optional[list[str]] = typer.Argument(
        None, help="Demo keys/indices to clear (default: all when --all)."
    ),
    all_demos: bool = typer.Option(False, "--all", help="Clear caches for all demos."),
    downloads: bool = typer.Option(
        True, "--downloads/--no-downloads", help="Clear downloaded artifacts."
    ),
    computed: bool = typer.Option(
        True,
        "--computed/--no-computed",
        help="Clear computed artifacts (*.pkl and local/ fits).",
    ),
    outputs: bool = typer.Option(
        False, "--outputs", help="Also delete generated datasets/demos/*.luxar.zarr."
    ),
    orphans: bool = typer.Option(
        False,
        "--orphans",
        help="Also remove cache dirs claimed by no demo (hand-placed inputs are kept).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted; delete nothing."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Clear demo caches (and optionally generated outputs), with fine-grained scope.

    By default clears the input caches (downloads + computed pickles) under
    ~/.cache/luxar/<name>/ for the selected demos. Nothing is deleted without
    either --dry-run (preview) or a confirmation (or --yes).
    """
    import shutil

    if not keys and not all_demos and not orphans:
        aprint("❌ Specify demo key(s), --all, or --orphans.")
        raise typer.Exit(1)

    selected: list[DemoInfo] = (
        _demos_or_exit() if all_demos else [_resolve_or_exit(k) for k in (keys or [])]
    )

    # (path, size, label) tuples to delete.
    targets: list[tuple[Path, int, str]] = []

    def _add_dir_files(cache_dir: Path, demo_key: str) -> None:
        if not cache_dir.exists():
            return
        for f in sorted(cache_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(cache_dir)
            # A computed artifact is a pickle (or a corrupt pickle), or anything
            # inside the local-fit namespace — a demo's own GPU refit, which can
            # cost tens of minutes and which `--no-computed` must therefore
            # spare. Every other file — including a corrupt *download* like
            # ``foo.zip.corrupt`` — is a download. ``Path("x.pkl.corrupt").suffix
            # == ".corrupt"``, so the ``endswith`` check is what actually
            # classifies corrupt pickles.
            is_computed = (
                f.suffix == ".pkl"
                or f.name.endswith(".pkl.corrupt")
                or LOCAL_FIT_DIRNAME in rel.parts[:-1]
            )
            if (is_computed and computed) or (not is_computed and downloads):
                # Labelled by the path RELATIVE to the cache dir, so
                # ``local/x.zip`` is distinguishable from the manifest's own
                # ``x.zip`` sitting next to it.
                targets.append((f, f.stat().st_size, f"{demo_key}/{rel.as_posix()}"))

    # Hand-placed demo inputs (registry.PROTECTED_INPUT_DIRS) are refused on
    # every route — by key, --all and --orphans — and the names of the ones we
    # spared earn one notice each, printed in --dry-run and in a real run alike
    # so the preview matches the run. See :func:`_clearable_cache_dirs`.
    protected_kept: list[str] = []

    cache_dirs: list[Path] = []
    for d in selected:
        for cache_dir in _clearable_cache_dirs(d, protected_kept):
            cache_dirs.append(cache_dir)
            _add_dir_files(cache_dir, d.key)
        if outputs:
            for p in _safe_output_paths(d):
                if p.exists():
                    size = (
                        registry.dir_size_bytes(p) if p.is_dir() else p.stat().st_size
                    )
                    targets.append((p, size, f"{d.key} output {p.name}"))

    if orphans:
        try:
            entries = registry.inventory_caches()
        except registry.DemoMetaError as exc:
            aprint(f"❌ Broken demo metadata: {exc}")
            raise typer.Exit(1) from exc
        targets.extend(_orphan_targets(entries, protected_kept))

    # One cache name can be claimed by several demos (four share
    # ``gsplats_tribolium``), so a selection covering more than one claimant
    # walks that directory once per demo: the same file lands in ``targets``
    # once per claimant and the same directory in ``cache_dirs``. Delete, sweep
    # and count each path once — otherwise the summary promises (and the final
    # tally claims to have freed) several times the bytes actually there.
    cache_dirs = list(dict.fromkeys(cache_dirs))
    seen: set[Path] = set()
    unique_targets: list[tuple[Path, int, str]] = []
    for target in targets:
        if target[0] in seen:
            continue
        seen.add(target[0])
        unique_targets.append(target)
    targets = unique_targets

    # Empty directories hold no bytes, so they are swept rather than listed as
    # targets (the count/size summary stays a summary of actual data), but they
    # still have to go — see :func:`_empty_dirs`. The sweep happens after the
    # deletions while the summary is printed before them, in both modes, so the
    # summary has to look ahead at the files it is about to delete — otherwise
    # a real run silently removes a cache dir it never mentioned.
    doomed = {p for p, _, _ in targets}
    empties = [p for c in cache_dirs for p in _empty_dirs(c, deleted=doomed)]

    # Ahead of the "nothing to clear" exit: a selection that was *only* a
    # protected input still has to say why it cleared nothing.
    for name in protected_kept:
        aprint(
            f"🔒 Keeping {name}/ — a hand-placed demo input, not a download; "
            "delete it by hand if you really mean to."
        )

    if not targets and not empties:
        aprint("Nothing to clear for that selection.")
        raise typer.Exit(0)

    total = sum(size for _, size, _ in targets)
    aprint(f"🗑️  [Luxar] {len(targets)} item(s), {format_memory_size(total)}:")
    for _path, size, label in targets:
        aprint(f"   {format_memory_size(size):>10}  {label}")
    if empties:
        aprint(f"   {format_memory_size(0):>10}  {len(empties)} dir(s) removed")

    if dry_run:
        aprint("\n(--dry-run: nothing deleted)")
        raise typer.Exit(0)
    if not yes and not typer.confirm("\nDelete these?"):
        aprint("Aborted.")
        raise typer.Exit(0)

    freed, not_removed = 0, []
    for path, size, label in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        if path.exists():
            not_removed.append(label)
        else:
            freed += size
    # Sweep empty directories — those left behind by the file deletions above,
    # and any leaked staging dir that was already empty — deepest first, so a
    # cache dir emptied of everything is itself removed. A directory can refuse
    # to go (read-only parent, a Windows process holding it as its cwd, a
    # concurrent demo run staging into it between this listing and the rmdir);
    # that is a report line like any other failed removal, never a traceback
    # thrown over a summary the user has already earned.
    for cache_dir in cache_dirs:
        for empty in _empty_dirs(cache_dir):
            try:
                empty.rmdir()
            except OSError:
                not_removed.append(str(empty.relative_to(cache_dir.parent)))
    aprint(f"✅ Cleared {format_memory_size(freed)}.")
    if not_removed:
        aprint(
            f"⚠️  Could not remove {len(not_removed)} item(s): {', '.join(not_removed)}"
        )
