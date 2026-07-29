"""``luxar demo`` — list, inspect, run, and manage Luxar's bundled demos.

Bare ``luxar demo`` prints the demo table (= ``luxar demo list``). Demos are
executed as subprocesses (``python -m luxar.demos.demo_<name>``), never
imported: several parse ``sys.argv`` or create cache directories at import
time. Extra CLI args after the key are forwarded verbatim to the demo script.
"""

from __future__ import annotations

import importlib
import shlex
import sys
from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

from ..demos import registry
from ..demos.registry import DemoInfo
from ..utils.process import run_child_process
from .utils import format_memory_size

app_demo = typer.Typer(
    help="Run and manage Luxar's bundled demos (list / info / run / deps / cache).",
    no_args_is_help=False,
)
cache_app = typer.Typer(help="Inspect and clear demo download/compute caches.")
app_demo.add_typer(cache_app, name="cache")


# ────────────────────────────── rendering ────────────────────────────────────
def _needs_glyphs(info: DemoInfo) -> str:
    """Compact 'what does this demo need' summary for the table/info views."""
    parts: list[str] = []
    if info.download_mb:
        parts.append(f"⬇{info.download_mb}MB")
    if info.gpu == "required":
        parts.append("GPU")
    elif info.gpu == "optional":
        parts.append("GPU*")
    local = {"git-lfs": "LFS", "kaggle-auth": "🔑kaggle", "manual-file": "📁manual"}
    if info.local_data in local:
        parts.append(local[info.local_data])
    return " ".join(parts)


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


def _status(info: DemoInfo) -> str:
    """Whether this demo already has cached inputs or a generated output."""
    if any(p.exists() for p in _safe_output_paths(info)):
        return "output ✓"
    if any(d.exists() for d in registry.demo_cache_dirs(info)):
        return "cached"
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


def _print_table(demos: list[DemoInfo]) -> None:
    # Size the KEY column to the longest key so it is never truncated — the
    # key is what the user types into `demo run`, so it must be copy-pasteable.
    kw = max((len(d.key) for d in demos), default=3)
    kw = max(kw, len("KEY"))
    header = (
        f"{'#':>3}  {'KEY':<{kw}} {'GEOM':<12} {'CATEGORY':<14} {'NEEDS':<20} STATUS"
    )
    aprint(f"🎬 [Luxar] {len(demos)} demos\n")
    aprint(header)
    aprint("─" * len(header))
    for d in demos:
        aprint(
            f"{d.index:>3}  {d.key:<{kw}} {d.geometry:<12} {d.category:<14} "
            f"{_needs_glyphs(d):<20} {_status(d)}"
        )
    aprint("")
    aprint("Run one:  luxar demo run <key|#>       Details:  luxar demo info <key|#>")
    aprint("Caches:   luxar demo cache list        Clear:    luxar demo cache clear …")
    aprint("Deps:     luxar demo deps              Install:  luxar demo deps --install")


def _resolve_or_exit(key_or_index: str) -> DemoInfo:
    try:
        return registry.get_demo(key_or_index)
    except KeyError as e:
        aprint(f"❌ {e}")
        raise typer.Exit(1) from e
    except registry.DemoMetaError as e:
        aprint(f"❌ Broken demo metadata: {e}")
        raise typer.Exit(1) from e


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
    req = info.requirements
    aprint(f"🎬 {info.title}  [{info.key}]  #{info.index}")
    aprint(f"   {info.description}")
    aprint("")
    aprint(f"   Category:   {info.category}")
    aprint(f"   Geometry:   {info.geometry}")
    aprint(f"   Compute:    {req['compute']}")
    aprint(
        f"   Download:   {req['download_mb']} MB"
        if req["download_mb"]
        else "   Download:   none (offline)"
    )
    aprint(f"   GPU:        {req['gpu']}")
    if req["local_data"]:
        aprint(f"   Local data: {req['local_data']}")
    if info.caches:
        aprint(f"   Caches:     {', '.join(info.caches)}")
    if info.outputs:
        outs = ", ".join(f"{o}.luxar.zarr" for o in info.outputs)
        aprint(f"   Outputs:    {outs}")
    aprint("")
    aprint(f"   Run:        luxar demo run {info.key}")
    aprint(f"   Module:     python -m {info.module}")
    aprint(
        "   Network sim: run the generated scene through "
        "`luxar serve <scene> --viewer --profile 3g`"
    )


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
    code = run_child_process(cmd, label=f"demo '{info.key}'")
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
        code = run_child_process(
            [sys.executable, "-m", d.module, "--no-serve"], label=d.key
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


# ──────────────────────────────── deps ──────────────────────────────────────
def _pip_install_cmd(extras: list[str]) -> list[str]:
    """The pip command that installs ``extras``, editable when in a checkout.

    A dev checkout must install ``-e <root>[…]``: a plain ``luxar[demos]`` would
    fetch the *published* wheel from PyPI and shadow the tree the user is
    editing. From an installed wheel there is no root, so name the distribution.
    """
    joined = ",".join(extras)
    try:
        from ..utils.paths import get_project_root

        root = get_project_root()
    except RuntimeError:
        root = None
    base = [sys.executable, "-m", "pip", "install"]
    if root is not None:
        return [*base, "-e", f"{root}[{joined}]"]
    return [*base, f"luxar[{joined}]"]


@app_demo.command("deps")
def demo_deps(
    extra: Optional[str] = typer.Option(
        None,
        "--extra",
        # Long form only, deliberately: `-e` already means --encoding on the
        # gsplat commands, and means "editable" to the pip this command drives.
        help="Only consider one extra (demos / io / gsplats). Default: all.",
    ),
    install: bool = typer.Option(
        False, "--install", help="Install the extras that have missing packages."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="With --install, print the pip command only."
    ),
) -> None:
    """Report (and optionally install) the demos' optional dependencies.

    Demos deliberately keep heavyweight packages out of the core install, so a
    fresh checkout can run `luxar demo list` but not every demo. This reports
    exactly what is missing and, with ``--install``, installs the Luxar extras
    that provide it.
    """
    from ..demos import extras_for, survey

    if dry_run and not install:
        # Silently ignoring a flag the user typed is worse than saying so.
        aprint("ℹ️  --dry-run only applies with --install; reporting only.")

    # Extra names are lowercase by PEP 685, so accept any casing the user types.
    extra = extra.strip().lower() if extra else extra
    rows = survey(extra)
    if not rows:
        known = extras_for(survey()) or ["(none)"]
        aprint(
            f"❌ No known dependencies for extra {extra!r}. "
            f"Valid extras: {', '.join(known)}."
        )
        raise typer.Exit(1)

    missing = [r for r in rows if not r.installed]
    # Never let a column be narrower than its own header — a one-row report
    # (e.g. `--extra gsplats`) would otherwise print a ragged table.
    mw = max(max(len(r.module) for r in rows), len("MODULE"))
    sw = max(max(len(r.spec.spec) for r in rows), len("REQUIREMENT"))

    plural = "dependency" if len(rows) == 1 else "dependencies"
    aprint(f"📦 [Luxar] {len(rows)} optional demo {plural}\n")
    header = f"  {'MODULE':<{mw}}  {'REQUIREMENT':<{sw}}  {'EXTRA':<8} STATUS"
    aprint(header)
    # Rule the exact width of the header rather than a hand-counted constant
    # (the old `mw + sw + 20` overshot by one and left a dangling glyph).
    aprint("  " + "─" * (len(header) - 2))
    for r in rows:
        aprint(
            f"  {r.module:<{mw}}  {r.spec.spec:<{sw}}  "
            f"{(r.spec.extra or '—'):<8} {'ok' if r.installed else 'MISSING'}"
        )
    aprint("")

    if not missing:
        # Phrased to avoid subject-verb agreement on the count ("1 dependency
        # are installed"), which a plural-noun-only fix leaves behind.
        aprint(f"✅ Nothing missing — all {len(rows)} optional {plural} installed.")
        raise typer.Exit(0)

    aprint(f"⚠️  {len(missing)} missing: {', '.join(r.module for r in missing)}")

    # Specs outside every extra can't be installed via luxar[…]; name them.
    orphans = [r for r in missing if not r.spec.extra]
    extras = extras_for(missing)
    if orphans:
        aprint(
            "   Not in any extra (install individually): "
            + ", ".join(f"'{r.spec.spec}'" for r in orphans)
        )
    if not extras:
        raise typer.Exit(1)

    cmd = _pip_install_cmd(extras)

    # shlex.join, not " ".join: the interpreter path and the checkout root both
    # routinely contain spaces (e.g. "Application Support"), and the extras
    # brackets are shell globs — an unquoted line would not paste back in.
    shown = shlex.join(cmd)
    if not install:
        aprint("\n   Install with:  luxar demo deps --install")
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
    left = [r for r in survey(extra) if not r.installed]
    # Judge the install ONLY on what it was asked to provide. An orphan spec
    # (no extra) was never in the pip command, so listing it as "still missing
    # after install" blames the install for something it never attempted.
    still = [r for r in left if r.spec.extra in extras]
    if still:
        aprint(f"⚠️  Still missing after install: {', '.join(r.module for r in still)}")
        raise typer.Exit(1)
    aprint(f"✅ Installed: {', '.join(f'luxar[{e}]' for e in extras)}.")
    # Don't claim completeness while an orphan is still absent — the install
    # genuinely could not cover it.
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
    aprint(f"💾 [Luxar] Demo caches under {registry.DEMO_CACHE_ROOT}\n")
    total = 0
    for e in entries:
        total += e.size_bytes
        owner = ", ".join(e.demo_keys) if e.demo_keys else "⚠️  ORPHAN"
        aprint(f"  {format_memory_size(e.size_bytes):>10}  {e.path.name:<32} {owner}")
    aprint("")
    aprint(f"  {format_memory_size(total):>10}  TOTAL ({len(entries)} dirs)")


# ─────────────────────────────── cache clear ─────────────────────────────────
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
        True, "--computed/--no-computed", help="Clear pickled computations (*.pkl)."
    ),
    outputs: bool = typer.Option(
        False, "--outputs", help="Also delete generated datasets/demos/*.luxar.zarr."
    ),
    orphans: bool = typer.Option(
        False, "--orphans", help="Also remove cache dirs claimed by no demo."
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
            # A computed artifact is a pickle (or a corrupt pickle); every other
            # file — including a corrupt *download* like ``foo.zip.corrupt`` — is
            # a download. ``Path("x.pkl.corrupt").suffix == ".corrupt"``, so the
            # ``endswith`` check is what actually classifies corrupt pickles.
            is_computed = f.suffix == ".pkl" or f.name.endswith(".pkl.corrupt")
            if (is_computed and computed) or (not is_computed and downloads):
                targets.append((f, f.stat().st_size, f"{demo_key}/{f.name}"))

    for d in selected:
        for cache_dir in registry.demo_cache_dirs(d):
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
        for e in entries:
            if not e.demo_keys:
                targets.append((e.path, e.size_bytes, f"ORPHAN {e.path.name}"))

    if not targets:
        aprint("Nothing to clear for that selection.")
        raise typer.Exit(0)

    total = sum(size for _, size, _ in targets)
    aprint(f"🗑️  [Luxar] {len(targets)} item(s), {format_memory_size(total)}:")
    for _path, size, label in targets:
        aprint(f"   {format_memory_size(size):>10}  {label}")

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
    # Remove now-empty cache dirs left behind by file deletions.
    for d in selected:
        for cache_dir in registry.demo_cache_dirs(d):
            if cache_dir.exists() and not any(cache_dir.iterdir()):
                cache_dir.rmdir()
    aprint(f"✅ Cleared {format_memory_size(freed)}.")
    if not_removed:
        aprint(
            f"⚠️  Could not remove {len(not_removed)} item(s): {', '.join(not_removed)}"
        )
