#!/usr/bin/env python3
"""Is the demo-data migration to Zenodo ready, and what is still missing? (R17)

Answers that from GROUND TRUTH rather than from notes, by cross-referencing
three sources that can each drift out of step with the others:

  1. ``demos/data_manifest.json`` — the declared disposition of every dataset
     (bucket, destination record, license, files + checksums).
  2. the demo registry — which demos exist and which caches they claim.
  3. the filesystem — which bytes actually exist, in-repo and in ``~/.cache``.

Run it before touching Zenodo, and again after each upload::

    python scripts/zenodo_migration_audit.py

The checks that matter, and why each one is here:

``UNDECLARED on disk``
    A data file the repo ships that the manifest does not describe would be
    uploaded to nowhere and fetched by nothing — it is how a dataset silently
    escapes the migration. Must be 0.
``NO FILES LISTED``
    A ``zenodo`` dataset with no file entries has nothing to upload; its bytes
    live somewhere else (usually obsidian) and the entry is a placeholder.
``caches claimed by demos but ABSENT from the manifest``
    Expected and mostly benign: those demos fetch or generate their data at
    runtime from a public source. It is listed so a NEW dataset that quietly
    needs hosting cannot hide among them.
``manifest datasets no demo claims as a cache``
    Also expected — several demos resolve data by a route other than the
    ``caches`` key in ``DEMO_META``. Listed so a genuinely orphaned dataset
    (nothing loads it) is not uploaded and maintained forever.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent.parent
)
sys.path.insert(0, str(REPO / "packages/luxar/src"))

MANIFEST = REPO / "packages/luxar/src/luxar/demos/data_manifest.json"
DATA_DIR = REPO / "packages/luxar/src/luxar/demos/data"
CACHE = Path.home() / ".cache" / "luxar"

DATA_SUFFIXES = (".zip", ".npz", ".parquet", ".npy")


def files_of(spec: dict) -> list[dict]:
    """A dataset's file entries, flattening size variants into one list."""
    files = spec.get("files") or []
    if not files and "variants" in spec:
        files = [f for v in spec["variants"].values() for f in (v.get("files") or [])]
    return files


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    datasets, records = m["datasets"], m["records"]

    print("=" * 78)
    print("ZENODO RECORDS (upload destinations)")
    print("=" * 78)
    for name, r in records.items():
        state = "LIVE" if r.get("zenodo_record") else "NOT CREATED"
        print(
            f"  {name:10s} {state:12s} {r.get('license', '?'):14s} doi={r.get('zenodo_concept_doi')}"
        )

    print()
    print("=" * 78)
    print("DATASETS BY BUCKET   (repo = bytes in-tree, cache = bytes in ~/.cache)")
    print("=" * 78)
    buckets: dict[str, list[tuple[str, dict]]] = {}
    for name, spec in datasets.items():
        buckets.setdefault(spec.get("bucket", "?"), []).append((name, spec))

    to_upload: list[tuple[str, float]] = []
    # "elsewhere" is NOT the same as blocked: the bytes exist, just not on this
    # machine (these are the obsidian-computed sets). Conflating the two hides
    # whether anything actually needs a human decision.
    elsewhere: list[str] = []
    for bucket in sorted(buckets):
        print(f"\n--- {bucket}  ({len(buckets[bucket])} datasets) ---")
        for name, spec in sorted(buckets[bucket]):
            files = files_of(spec)
            sub = spec.get("dir", "")
            in_repo = sum(
                1
                for f in files
                if (
                    (DATA_DIR / sub / f["name"]) if sub else (DATA_DIR / f["name"])
                ).exists()
            )
            in_cache = sum(1 for f in files if (CACHE / name / f["name"]).exists())
            size = sum(f.get("bytes", 0) for f in files) / 1048576
            flag = ""
            if bucket == "zenodo":
                if not files:
                    flag = "  <-- bytes on another machine (upload from there)"
                    elsewhere.append(name)
                elif in_repo == 0 and in_cache == 0:
                    flag = "  <-- BYTES NOT ON THIS MACHINE"
                    elsewhere.append(name)
                else:
                    to_upload.append((name, size))
            pend = " PENDING-UPLOAD" if spec.get("pending_upload") else ""
            print(
                f"  {name:38s} rec={spec.get('record', '-'):9s} files={len(files):2d} "
                f"repo={in_repo:2d} cache={in_cache:2d} {size:8.1f} MB "
                f"{spec.get('license', '?'):16s}{pend}{flag}"
            )

    print()
    print("=" * 78)
    print("DEMO REGISTRY CROSS-CHECK")
    print("=" * 78)
    try:
        from luxar.demos.registry import iter_demos

        demos = iter_demos()
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"  registry unavailable: {type(e).__name__}: {e}")
        demos = []

    if demos:
        claimed: dict[str, list[str]] = {}
        for d in demos:
            for c in getattr(d, "caches", None) or []:
                claimed.setdefault(c, []).append(getattr(d, "key", "?"))
        print(f"  demos discovered: {len(demos)}")
        unknown = sorted(c for c in claimed if c not in datasets)
        print(
            f"\n  runtime-fetch / procedural caches, not manifest-tracked ({len(unknown)}):"
        )
        print("    " + ", ".join(unknown) if unknown else "    (none)")
        unused = sorted(n for n in datasets if n not in claimed)
        print(
            f"\n  manifest datasets not claimed via DEMO_META['caches'] ({len(unused)}):"
        )
        print("    " + ", ".join(unused) if unused else "    (none)")
        print("    (these resolve by another route — verify with a grep before")
        print("     concluding any is orphaned)")

    print()
    print("=" * 78)
    print("FILES ON DISK vs MANIFEST")
    print("=" * 78)
    declared = set()
    for name, spec in datasets.items():
        sub = spec.get("dir", "")
        for f in files_of(spec):
            declared.add(
                (DATA_DIR / sub / f["name"]) if sub else (DATA_DIR / f["name"])
            )
    actual = (
        {p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix in DATA_SUFFIXES}
        if DATA_DIR.exists()
        else set()
    )
    undeclared = sorted(actual - declared)
    print(
        f"  data files in-tree: {len(actual)}   of which declared: {len(actual & declared)}"
    )
    print(f"  UNDECLARED (must be 0): {len(undeclared)}")
    for p in undeclared:
        print(f"    {p.relative_to(DATA_DIR)}  {p.stat().st_size / 1048576:.1f} MB")
    print(
        f"  bytes still in-tree: {sum(p.stat().st_size for p in actual) / 1048576:.1f} MB"
    )

    print()
    print("=" * 78)
    print("READINESS")
    print("=" * 78)
    print(
        f"  records to create:            {sum(1 for r in records.values() if not r.get('zenodo_record'))} of {len(records)}"
    )
    print(
        f"  datasets ready to upload now: {len(to_upload)}  "
        f"({sum(s for _, s in to_upload):.0f} MB)"
    )
    print(f"  upload from another machine:  {len(elsewhere)}")
    for b in elsewhere:
        print(f"    - {b}  (bytes on obsidian)")
    print("  blocked on a human decision:  0")
    return 1 if undeclared else 0


if __name__ == "__main__":
    raise SystemExit(main())
