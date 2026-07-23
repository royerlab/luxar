"""``luxar demo`` — list, inspect, run, and manage Luxar's bundled demos.

Bare ``luxar demo`` prints the demo table (= ``luxar demo list``). Demos are
executed as subprocesses (``python -m luxar.demos.demo_<name>``), never
imported: several parse ``sys.argv`` or create cache directories at import
time. Extra CLI args after the key are forwarded verbatim to the demo script.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import typer
from arbol import aprint

from ..demos import registry
from ..demos.registry import DemoInfo
from .utils import exit_code_from, format_memory_size

app_demo = typer.Typer(
    help="Run and manage Luxar's bundled demos (list / info / run / cache).",
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


def _status(info: DemoInfo) -> str:
    """Whether this demo already has cached inputs or a generated output."""
    try:
        if any(p.exists() for p in registry.demo_output_paths(info)):
            return "output ✓"
    except Exception:
        pass
    if any(d.exists() for d in registry.demo_cache_dirs(info)):
        return "cached"
    return ""


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


def _resolve_or_exit(key_or_index: str) -> DemoInfo:
    try:
        return registry.get_demo(key_or_index)
    except KeyError as e:
        aprint(f"❌ {e}")
        raise typer.Exit(1) from e


# ─────────────────────────────── callback ────────────────────────────────────
@app_demo.callback(invoke_without_command=True)
def demo_callback(ctx: typer.Context) -> None:
    """Show the demo table when invoked with no subcommand."""
    if ctx.invoked_subcommand is None:
        _print_table(registry.iter_demos())
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
    demos = registry.iter_demos()
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

    Example: luxar demo run tribolium -- --recompute --no-serve
    """
    info = _resolve_or_exit(key)
    extra = list(ctx.args)
    aprint(f"🎬 {info.title}  [{info.key}]")
    aprint(f"   {info.description}")
    if extra:
        aprint(f"➡️  Forwarding args to the demo: {' '.join(extra)}")
    cmd = [sys.executable, "-m", info.module, *extra]
    try:
        result = subprocess.run(cmd)
    except KeyboardInterrupt:
        aprint("\n🛑 Demo interrupted.")
        raise typer.Exit(130) from None
    if result.returncode != 0:
        # 128+N for signal-killed children (raw -N truncates to 256-N).
        raise typer.Exit(exit_code_from(result.returncode))


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
) -> None:
    """Generate every demo's dataset (``--no-serve``), for batch/gallery builds.

    Skips demos that need manual/Kaggle data (they can't run unattended) and,
    by default, demos whose output scenes already exist.
    """
    demos = registry.iter_demos()
    ran, skipped, failed = 0, 0, []
    for d in demos:
        if d.local_data in ("manual-file", "kaggle-auth"):
            aprint(f"⏭️  {d.key}: needs {d.local_data}; skipping")
            skipped += 1
            continue
        outs = registry.demo_output_paths(d)
        if skip_existing and outs and all(p.exists() for p in outs):
            aprint(f"⏭️  {d.key}: output exists; skipping")
            skipped += 1
            continue
        aprint(f"▶️  {d.key}: {d.title}")
        try:
            result = subprocess.run([sys.executable, "-m", d.module, "--no-serve"])
        except KeyboardInterrupt:
            # Same contract as `demo run`: Ctrl-C exits 130, not Click's
            # generic "Aborted!" exit 1.
            aprint(f"\n🛑 Interrupted during {d.key}.")
            raise typer.Exit(130) from None
        if result.returncode != 0:
            failed.append(d.key)
            aprint(f"❌ {d.key}: exited {exit_code_from(result.returncode)}")
            if not keep_going:
                break
        else:
            ran += 1
    aprint(f"\n✅ ran {ran}, skipped {skipped}, failed {len(failed)}")
    if failed:
        aprint(f"   failed: {', '.join(failed)}")
        raise typer.Exit(1)


# ─────────────────────────────── cache list ──────────────────────────────────
@cache_app.command("list")
def cache_list() -> None:
    """Inventory the demo caches under ~/.cache/luxar/."""
    entries = registry.inventory_caches()
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
    from pathlib import Path

    if not keys and not all_demos and not orphans:
        aprint("❌ Specify demo key(s), --all, or --orphans.")
        raise typer.Exit(1)

    selected: list[DemoInfo] = (
        registry.iter_demos()
        if all_demos
        else [_resolve_or_exit(k) for k in (keys or [])]
    )

    # (path, size, label) tuples to delete.
    targets: list[tuple[Path, int, str]] = []

    def _add_dir_files(cache_dir: Path, demo_key: str) -> None:
        if not cache_dir.exists():
            return
        for f in sorted(cache_dir.rglob("*")):
            if not f.is_file():
                continue
            is_pkl = f.suffix in (".pkl", ".corrupt") or f.name.endswith(".pkl.corrupt")
            if (is_pkl and computed) or (not is_pkl and downloads):
                targets.append((f, f.stat().st_size, f"{demo_key}/{f.name}"))

    for d in selected:
        for cache_dir in registry.demo_cache_dirs(d):
            _add_dir_files(cache_dir, d.key)
        if outputs:
            for p in registry.demo_output_paths(d):
                if p.exists():
                    size = (
                        registry.dir_size_bytes(p) if p.is_dir() else p.stat().st_size
                    )
                    targets.append((p, size, f"{d.key} output {p.name}"))

    if orphans:
        for e in registry.inventory_caches():
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

    for path, _size, _label in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    # Remove now-empty cache dirs left behind by file deletions.
    for d in selected:
        for cache_dir in registry.demo_cache_dirs(d):
            if cache_dir.exists() and not any(cache_dir.iterdir()):
                cache_dir.rmdir()
    aprint(f"✅ Cleared {format_memory_size(total)}.")
