#!/usr/bin/env python3
"""Upgrade the git-LFS gsplat demo baselines to format v3.2 + modern quantization.

The committed `.gsplats.zarr.zip` demo datasets under
``packages/luxar/src/luxar/demos/data/gsplats_*/`` are format **v3.0** and
effectively all-float32 (they predate every encoding PR that landed 2026-06-30
onward). This script re-encodes them to the current format at
``EncodingMode.AUTO`` (certified near-lossless: uint16 per-axis centers, split
Cholesky at certified-u8, rgb_uint8/geolog colors) and rebuilds their LOD
ladders — no re-fit, no source volume needed. The result is ~2-3x smaller with
visually-lossless fidelity.

Two dataset shapes are handled:
  * single-dataset ``*.gsplats.zarr.zip`` — one gsplat store per zip.
  * timelapse bundle ``*_gsplats.zip`` — an outer zip of many per-frame
    ``*.gsplats.zarr.zip`` stores (celegans, zebrafish).

Per store, the transform is chosen by structure:
  * flat leaf (with/without an additive ladder)  -> ``gsplat flatten`` then
    ``gsplat lod --recipe stream -e auto`` (rebuild a modern streaming ladder).
  * kind=partition / lod / nested                -> ``gsplat additive -e auto``
    (structure-preserving: re-quantize + re-ladder each leaf in place).

By default output is written to a STAGING dir and the committed LFS files are
NOT touched (verify first). Pass ``--apply`` to overwrite the committed files.

Usage:
    hatch run python scripts/reencode_gsplat_demos.py [--only NAME ...] \
        [--staging DIR] [--recipe stream|overview] [--apply] [--psnr] [--shape Z,Y,X]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "packages/luxar/src/luxar/demos/data"


def run_cli(*args: str) -> None:
    """Run a `luxar` CLI subcommand via hatch, raising on failure."""
    cmd = ["luxar", *args]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"CLI failed: {' '.join(cmd)}")


def find_store_dir(extract_root: Path) -> Path | None:
    """Return the single `*.gsplats.zarr` store dir at the top of an extract, or None."""
    candidates = [
        p
        for p in extract_root.iterdir()
        if p.is_dir() and (p / ".zgroup").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def read_root_attrs(store: Path) -> dict:
    zattrs = store / ".zattrs"
    if zattrs.exists():
        return json.loads(zattrs.read_text())
    return {}


def classify(store: Path) -> str:
    """Return 'partition' (structure-preserving) or 'leaf' (flatten+lod)."""
    attrs = read_root_attrs(store)
    kind = attrs.get("kind")
    if kind in ("partition", "lod"):
        return "partition"
    # A nested tree without a top-level splat leaf also counts as partition-like:
    # detect child groups that are themselves kind=partition/lod.
    for child in store.iterdir():
        if child.is_dir() and (child / ".zattrs").exists():
            ca = json.loads((child / ".zattrs").read_text())
            if ca.get("kind") in ("partition", "lod"):
                return "partition"
    return "leaf"


def reencode_store(src_store: Path, out_store: Path, recipe: str, tmp: Path) -> str:
    """Re-encode one gsplat store to AUTO/v3.2, rebuilding its ladder.

    Returns the classification used.
    """
    kind = classify(src_store)
    if kind == "partition":
        run_cli("gsplat", "additive", str(src_store), str(out_store), "-e", "auto")
    else:
        flat = tmp / (out_store.name + ".flat")
        if flat.exists():
            shutil.rmtree(flat)
        run_cli("gsplat", "flatten", str(src_store), str(flat))
        run_cli(
            "gsplat", "lod", str(flat), str(out_store),
            "--recipe", recipe, "-e", "auto",
        )
        shutil.rmtree(flat, ignore_errors=True)
    return kind


def zip_store(store_dir: Path, out_zip: Path, arcname_root: str) -> None:
    """Zip a store directory so the archive top-level is `arcname_root/...`."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(store_dir.rglob("*")):
            rel = path.relative_to(store_dir)
            zf.write(path, f"{arcname_root}/{rel}")


def process_single_zip(src_zip: Path, out_zip: Path, recipe: str, tmp: Path) -> dict:
    """Re-encode a single-dataset `.gsplats.zarr.zip`. Returns a report dict."""
    ex = tmp / (src_zip.stem + "_ex")
    if ex.exists():
        shutil.rmtree(ex)
    ex.mkdir(parents=True)
    with zipfile.ZipFile(src_zip) as zf:
        zf.extractall(ex)
    store = find_store_dir(ex)
    if store is None:
        raise RuntimeError(f"no single store found in {src_zip} (bundle?)")
    arc = store.name  # preserve the exact inner dir name
    out_store = tmp / (src_zip.stem + "_new")
    if out_store.exists():
        shutil.rmtree(out_store)
    kind = reencode_store(store, out_store, recipe, tmp)
    zip_store(out_store, out_zip, arc)
    return {
        "kind": kind,
        "old_bytes": src_zip.stat().st_size,
        "new_bytes": out_zip.stat().st_size,
    }


def process_bundle_zip(src_zip: Path, out_zip: Path, recipe: str, tmp: Path) -> dict:
    """Re-encode a timelapse bundle: outer zip of per-frame `.gsplats.zarr.zip`."""
    ex = tmp / (src_zip.stem + "_bundle")
    if ex.exists():
        shutil.rmtree(ex)
    ex.mkdir(parents=True)
    with zipfile.ZipFile(src_zip) as zf:
        names = zf.namelist()
        zf.extractall(ex)
    inner_zips = sorted(ex.rglob("*.gsplats.zarr.zip"))
    n = 0
    for iz in inner_zips:
        itmp = tmp / "frame_tmp"
        if itmp.exists():
            shutil.rmtree(itmp)
        itmp.mkdir()
        process_single_zip(iz, iz, recipe, itmp)  # overwrite the inner zip in place
        n += 1
    # Repackage the outer bundle, preserving the original member layout/order.
    if out_zip.exists():
        out_zip.unlink()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            member = ex / name
            if member.is_file():
                zf.write(member, name)
    return {
        "kind": f"bundle({n} frames)",
        "old_bytes": src_zip.stat().st_size,
        "new_bytes": out_zip.stat().st_size,
    }


def is_bundle(src_zip: Path, tmp: Path) -> bool:
    """A bundle contains inner `.gsplats.zarr.zip` members (zips within zip)."""
    with zipfile.ZipFile(src_zip) as zf:
        for name in zf.namelist():
            if name.endswith(".gsplats.zarr.zip"):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Process only these dataset dir names (e.g. gsplats_dapi).")
    ap.add_argument("--staging", default=str(REPO / "delme/gsplat_upgrade/staged"),
                    help="Directory for re-encoded output (not the LFS path).")
    ap.add_argument("--recipe", default="stream", choices=["stream", "overview"],
                    help="LOD recipe for flat-leaf datasets.")
    ap.add_argument("--apply", action="store_true",
                    help="Overwrite the committed LFS files instead of staging.")
    args = ap.parse_args()

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    zips = sorted(DATA_DIR.glob("gsplats_*/*.zip"))
    if args.only:
        keep = set(args.only)
        zips = [z for z in zips if z.parent.name in keep]

    rows: list[tuple[str, dict]] = []
    with tempfile.TemporaryDirectory(dir=str(REPO / "delme/gsplat_upgrade")) as td:
        tmp = Path(td)
        for src in zips:
            rel = src.relative_to(DATA_DIR)
            # Skip unmaterialized LFS pointers.
            if src.stat().st_size < 1024:
                print(f"SKIP (LFS pointer, run git lfs pull): {rel}")
                continue
            out = (src if args.apply else staging / rel)
            try:
                if is_bundle(src, tmp):
                    rep = process_bundle_zip(src, out, args.recipe, tmp)
                else:
                    rep = process_single_zip(src, out, args.recipe, tmp)
                rows.append((str(rel), rep))
                ratio = rep["old_bytes"] / max(rep["new_bytes"], 1)
                print(f"OK  {rel}  [{rep['kind']}]  "
                      f"{rep['old_bytes']:,} -> {rep['new_bytes']:,}  ({ratio:.2f}x)")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {rel}: {e}")

    # Summary
    print("\n=== SUMMARY ===")
    total_old = sum(r["old_bytes"] for _, r in rows)
    total_new = sum(r["new_bytes"] for _, r in rows)
    for name, r in rows:
        ratio = r["old_bytes"] / max(r["new_bytes"], 1)
        print(f"  {name:<60} {r['old_bytes']:>12,} -> {r['new_bytes']:>12,}  ({ratio:.2f}x)")
    if total_new:
        print(f"  {'TOTAL':<60} {total_old:>12,} -> {total_new:>12,}  "
              f"({total_old / total_new:.2f}x)")
    print(f"\nOutput: {'COMMITTED LFS PATHS (--apply)' if args.apply else staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
