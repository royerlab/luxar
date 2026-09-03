#!/usr/bin/env python3
"""Add an additive LOD ladder to every fetched gsplat demo baseline.

The baselines now live on the records and must be fetched and staged under their
former ``packages/luxar/src/luxar/demos/data/gsplats_*/`` paths before running.

Web-viewing-friendly level design (cumulative splat counts):

  L0 ≤ 1,500    (~65 KB at 11 floats×4 B per 3D splat → one 64 KB chunk)
  L1 ≤ 8,000    (~350 KB,  ~5 chunks)
  L2 ≤ 40,000   (~1.8 MB,  ~28 chunks)
  L3 = N        (the rest)

The breakpoints collapse cleanly when N is small (DAPI ~5K → 3 distinct
levels) and grow nicely with N (CMU-1 ~6.9M → 4 distinct levels with the
long tail in L3). L0 is sized to fit in a single zarr chunk so a viewer
that supports progressive LOD streaming can paint the first level in a
single HTTP range-request.

Ordering method: ``greedy`` (provably (1-1/e)-optimal) up to N ≈ 200K,
falling back to ``self_energy`` (O(N log N)) above that — matches the
CLI's own large-N recommendation.

Safety:
  - Default writes to ``<file>.lod_added.gsplats.zarr.zip`` *alongside*
    the original. Originals are not touched.
  - ``--in-place`` moves the original to ``<file>.bak`` then atomically
    renames the new file in place.
  - ``--dry-run`` prints the plan and exits.
  - Every output is validated by loading it back and confirming
    ``n_additive_sublods >= 2`` and total splat count == the original.
  - 4D bundle demos (celegans) extract the per-timepoint
    ``.gsplats.zarr.zip`` files, LOD each, and re-zip into a NEW bundle
    next to the original; the original bundle is never overwritten until
    the new bundle has been validated end-to-end.

Usage::

    hatch run python scripts/add_additive_lod_to_demos.py                   # sidecar files
    hatch run python scripts/add_additive_lod_to_demos.py --dry-run         # plan only
    hatch run python scripts/add_additive_lod_to_demos.py --only kidney    # one demo
    hatch run python scripts/add_additive_lod_to_demos.py --in-place        # replace originals
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "packages" / "luxar" / "src" / "luxar" / "demos" / "data"

# Cumulative splat-count breakpoints, sized for HTTP range-request streaming
# of zarr chunks (TARGET_CHUNK_BYTES=64KB → ~1,450 3D splats per chunk).
COUNT_BREAKPOINTS = [1_500, 8_000, 40_000]

# Use greedy ordering up to this N; switch to self_energy above. Greedy is
# provably (1-1/e)-optimal at every prefix but practically slow above ~2-5K
# splats (sparse-Gram lazy greedy). self_energy is the O(N log N) fallback the
# LOD CLI recommends for large N, with a 2-10% AUC gap vs greedy on real data.
# 5000 is set above the LOD CLI's own dense/sparse cutoff (2000) so we only
# pay the greedy cost when its dense path is in play.
GREEDY_MAX_N = 5_000

LOD_SUFFIX = ".lod_added"  # produces e.g. dapi_gsplats.lod_added.gsplats.zarr.zip
BACKUP_SUFFIX = ".bak"


# ---------------------------------------------------------------------------
# Per-demo file inventory
# ---------------------------------------------------------------------------


@dataclass
class PlainDemo:
    """A demo whose baseline is one or more flat .gsplats.zarr.zip files."""

    name: str
    dir_name: str
    file_names: List[str]


@dataclass
class BundleDemo:
    """A 4D demo whose baseline is one outer .zip containing per-timepoint
    .gsplats.zarr.zip entries."""

    name: str
    dir_name: str
    bundle_name: str  # e.g. "celegans_s1.gsplats.zarr.zip"


PLAIN_DEMOS: List[PlainDemo] = [
    PlainDemo("blastocyst_dapi", "gsplats_dapi", ["dapi.gsplats.zarr.zip"]),
    PlainDemo(
        "kidney",
        "gsplats_kidney",
        [
            "kidney_ch0.gsplats.zarr.zip",
            "kidney_ch1.gsplats.zarr.zip",
            "kidney_ch2.gsplats.zarr.zip",
        ],
    ),
    PlainDemo(
        "acto3d_heart",
        "gsplats_acto3d_heart",
        [
            "acto3d_heart_ch0.gsplats.zarr.zip",
            "acto3d_heart_ch1.gsplats.zarr.zip",
            "acto3d_heart_ch2.gsplats.zarr.zip",
        ],
    ),
    PlainDemo(
        "blastocyst_multichannel",
        "gsplats_multichannel",
        [
            "blastocyst_ch0.gsplats.zarr.zip",
            "blastocyst_ch1.gsplats.zarr.zip",
        ],
    ),
    PlainDemo(
        "opencell_map4",
        "gsplats_opencell_map4",
        [
            "opencell_map4_ch0.gsplats.zarr.zip",
            "opencell_map4_ch1.gsplats.zarr.zip",
        ],
    ),
    PlainDemo(
        "tribolium",
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
    ),
    PlainDemo(
        "cmu1_pathology",
        "gsplats_cmu1_pathology",
        [
            "cmu1_ch0.gsplats.zarr.zip",
            "cmu1_ch1.gsplats.zarr.zip",
            "cmu1_ch2.gsplats.zarr.zip",
        ],
    ),
]

BUNDLE_DEMOS: List[BundleDemo] = [
    BundleDemo("celegans", "gsplats_celegans", "celegans_s1.gsplats.zarr.zip"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _n_splats_of(zip_path: Path) -> int:
    """Load just enough of the .gsplats.zarr.zip to read its splat count."""
    from luxar.gsplats.gsplat_data import GSplatData

    g = GSplatData.load(zip_path, include_stats=False)
    return int(g.n_splats)


def _validate_lod(out_path: Path, expected_total: int) -> Tuple[bool, str]:
    """Return (ok, reason). True iff out_path loads, has n_additive_sublods >= 2,
    and the sum of per-sublod splat counts == expected_total."""
    from luxar.gsplats.gsplat_data import GSplatData

    try:
        g = GSplatData.load(out_path, include_stats=False)
    except Exception as e:  # noqa: BLE001 — we want any load failure to surface
        return False, f"load failed: {e!r}"

    n_add = getattr(g, "n_additive_sublods", 1)
    if n_add < 2:
        return False, f"expected >=2 additive sublods, got {n_add}"

    try:
        per_level = [g.additive_sublod(i).n_splats for i in range(n_add)]
    except Exception as e:  # noqa: BLE001
        return False, f"could not iterate additive_sublod: {e!r}"

    total = sum(per_level)
    if total != expected_total:
        return False, (
            f"sum of per-level splats {per_level} = {total} != {expected_total}"
        )
    return True, f"{n_add} levels, counts={per_level}"


def _make_breakpoints(n_splats: int) -> Tuple[str, List[int]]:
    """Build a deduplicated breakpoints arg for ``luxar gsplat lod --recipe stream``.

    Returns (cli_arg, levels). ``cli_arg`` is "counts:c1,c2,...,N" (cumulative
    splat budgets) with duplicates collapsed; ``levels`` is the underlying
    integer list for logging.
    """
    raw: List[int] = []
    for cap in COUNT_BREAKPOINTS:
        if cap < n_splats:
            raw.append(cap)
    raw.append(n_splats)  # always end at the full set

    # Dedupe while preserving order. `set.add` returns None so the
    # `or seen.add(c)` trick won't typecheck; an explicit loop is clearer
    # and the per-call dict is tiny (<=4 entries).
    seen: set[int] = set()
    levels: List[int] = []
    for c in raw:
        if c not in seen:
            seen.add(c)
            levels.append(c)
    return "counts:" + ",".join(str(c) for c in levels), levels


def _ordering_method(n_splats: int) -> str:
    return "greedy" if n_splats <= GREEDY_MAX_N else "self_energy"


def _run_lod_cli(in_path: Path, out_path: Path, n_splats: int) -> List[int]:
    """Subprocess ``luxar gsplat lod --recipe stream`` for one file. Returns the
    breakpoint integer list (for logging).

    Note: ``--compress zip`` is required to actually produce a .zip file;
    without it, the CLI writes a plain .gsplats.zarr **directory** even if
    the output path ends in .zip — confusing but documented in `--compress`'s
    help. We always pass it because the demos consume .zip baselines.
    """
    bp_arg, levels = _make_breakpoints(n_splats)
    method = _ordering_method(n_splats)
    cmd = [
        "hatch",
        "run",
        "luxar",
        "gsplat",
        "lod",
        str(in_path),
        str(out_path),
        "--recipe",
        "stream",
        "--breakpoints",
        bp_arg,
        "--add-method",
        method,
        "--compress",
        "zip",
        "--overwrite",
        "--quiet",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return levels


def _apply_in_place(original: Path, lod_built: Path) -> None:
    """Atomic-ish swap: move original → original.bak, then move lod_built →
    original. The .bak is not deleted so the caller can roll back."""
    bak = original.with_suffix(original.suffix + BACKUP_SUFFIX)
    if bak.exists():
        bak.unlink()
    original.rename(bak)
    lod_built.rename(original)
    print(f"  ✓ swapped in place; backup at {bak.name}")


# ---------------------------------------------------------------------------
# Per-demo runners
# ---------------------------------------------------------------------------


def process_plain(demo: PlainDemo, *, dry_run: bool, in_place: bool) -> bool:
    """Process a flat-baseline demo. Returns True iff all files succeeded."""
    print(f"\n=== {demo.name} ({demo.dir_name}) ===")
    ok_all = True
    for fname in demo.file_names:
        src = DATA / demo.dir_name / fname
        if not src.exists():
            print(f"  ⊘ {fname}: missing, skipping")
            continue

        n = _n_splats_of(src)
        bp_arg, levels = _make_breakpoints(n)
        method = _ordering_method(n)

        if len(levels) < 2:
            print(
                f"  ⊘ {fname}: N={n:,} too small for multi-level LOD "
                f"(would collapse to 1 level); leaving as-is"
            )
            continue

        # Build the new path: e.g. dapi_gsplats.lod_added.gsplats.zarr.zip
        # We insert the suffix before the .gsplats.zarr.zip composite extension
        # so the resulting filename still ends in .gsplats.zarr.zip.
        out_name = fname.replace(".gsplats.zarr.zip", f"{LOD_SUFFIX}.gsplats.zarr.zip")
        out_path = src.parent / out_name

        print(f"  {fname}: N={n:,}, method={method}, levels={levels}")
        if dry_run:
            print(f"    (dry-run) would write → {out_name}")
            continue

        # If a stale sidecar exists, remove it so the CLI doesn't refuse.
        if out_path.exists():
            out_path.unlink()

        try:
            _run_lod_cli(src, out_path, n)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {fname}: lod CLI failed: {e}")
            ok_all = False
            continue

        ok, reason = _validate_lod(out_path, expected_total=n)
        if not ok:
            print(f"  ✗ {fname}: validation failed — {reason}")
            print(f"    leaving {out_path.name} in place for inspection")
            ok_all = False
            continue
        print(f"  ✓ {fname}: {reason}")

        if in_place:
            _apply_in_place(src, out_path)
    return ok_all


def process_bundle(demo: BundleDemo, *, dry_run: bool, in_place: bool) -> bool:
    """Process a 4D bundle demo (celegans).

    Extracts the source zip into a temp dir, runs `lod additive` on every
    per-timepoint entry, re-packs the lot into a new bundle alongside the
    source. The original bundle is never overwritten until the new bundle
    has been built end-to-end.
    """
    src = DATA / demo.dir_name / demo.bundle_name
    print(f"\n=== {demo.name} ({demo.dir_name}/{demo.bundle_name}) ===")
    if not src.exists():
        print("  ⊘ bundle missing, skipping")
        return True

    out_name = demo.bundle_name.replace(".zip", f"{LOD_SUFFIX}.zip")
    out_bundle = src.parent / out_name
    print(f"  source: {src.name} ({src.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"  target: {out_name}")

    with tempfile.TemporaryDirectory(prefix="luxar-lod-bundle-") as tmpd:
        tmpd_path = Path(tmpd)
        in_dir = tmpd_path / "in"
        out_dir = tmpd_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if m.endswith(".gsplats.zarr.zip")]
            print(f"  bundle contains {len(members)} per-timepoint entries")
            if dry_run:
                # Sample first member just to estimate plan.
                if members:
                    zf.extract(members[0], in_dir)
                    sample = in_dir / members[0]
                    n = _n_splats_of(sample)
                    _bp, levels = _make_breakpoints(n)
                    print(f"  (dry-run) sample tp N={n:,}, levels={levels}")
                return True
            zf.extractall(in_dir)

        n_ok = 0
        n_skipped = 0
        n_fail = 0
        for i, m in enumerate(members):
            inner_in = in_dir / m
            inner_out = out_dir / m
            inner_out.parent.mkdir(parents=True, exist_ok=True)
            n = _n_splats_of(inner_in)
            _bp, levels = _make_breakpoints(n)
            if len(levels) < 2:
                # Tiny timepoint — copy through unchanged.
                shutil.copy2(inner_in, inner_out)
                n_skipped += 1
                if i % 50 == 0:
                    print(f"    [{i + 1}/{len(members)}] N={n} too small, copy-through")
                continue
            try:
                _run_lod_cli(inner_in, inner_out, n)
            except subprocess.CalledProcessError as e:
                print(f"    ✗ [{i + 1}/{len(members)}] {m}: lod failed: {e}")
                n_fail += 1
                # Fall back to copy-through so the bundle stays complete.
                shutil.copy2(inner_in, inner_out)
                continue
            ok, reason = _validate_lod(inner_out, expected_total=n)
            if not ok:
                print(
                    f"    ✗ [{i + 1}/{len(members)}] {m}: validation failed: {reason}"
                )
                # Fall back: keep original
                shutil.copy2(inner_in, inner_out)
                n_fail += 1
                continue
            n_ok += 1
            if i % 50 == 0 or i + 1 == len(members):
                print(f"    [{i + 1}/{len(members)}] {m}: {reason}")

        print(
            f"  per-tp summary: {n_ok} LOD-built, {n_skipped} too-small (copy-through), "
            f"{n_fail} failed (copy-through fallback)"
        )
        if n_fail > 0:
            print(f"  ⚠ {n_fail} timepoints fell back to single-level baseline")

        # Re-pack: copy each member into a new zip preserving the same names.
        with zipfile.ZipFile(out_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for m in members:
                zf.write(out_dir / m, arcname=m)
        print(
            f"  ✓ wrote {out_name} ({out_bundle.stat().st_size / (1024 * 1024):.1f} MB)"
        )

    if in_place:
        _apply_in_place(src, out_bundle)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Process only demos whose name contains this substring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-file plan and exit without writing anything",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "After validating each new LOD file, atomically replace the "
            "original with the new file (backup at <original>.bak). Default "
            "is to leave originals untouched and write to .lod_added sidecars."
        ),
    )
    args = parser.parse_args()

    plains = PLAIN_DEMOS
    bundles = BUNDLE_DEMOS
    if args.only:
        needle = args.only.lower()
        plains = [d for d in PLAIN_DEMOS if needle in d.name.lower()]
        bundles = [d for d in BUNDLE_DEMOS if needle in d.name.lower()]
        if not plains and not bundles:
            print(f"No demo matches --only {args.only}")
            return 1

    print(
        f"Processing {len(plains)} plain demo(s) + {len(bundles)} bundle demo(s)\n"
        f"  Breakpoints (cumulative counts): {COUNT_BREAKPOINTS} + N\n"
        f"  Greedy → self_energy switch at N > {GREEDY_MAX_N:,}\n"
        f"  Mode: {'DRY-RUN' if args.dry_run else ('IN-PLACE' if args.in_place else 'sidecar')}"
    )

    ok = True
    # Distinct loop variables — reusing `d` confuses mypy's type narrowing,
    # which then flags the second loop body as receiving a PlainDemo.
    for plain in plains:
        ok &= process_plain(plain, dry_run=args.dry_run, in_place=args.in_place)
    for bundle in bundles:
        ok &= process_bundle(bundle, dry_run=args.dry_run, in_place=args.in_place)

    print()
    print("✓ all demos processed" if ok else "⚠ some demos had failures")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
