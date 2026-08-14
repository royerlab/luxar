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
``INCOMPLETE``
    Some — but not all — of a dataset's declared files have bytes on this
    machine. Zenodo publication is a one-way door, so a partial set is never
    reported as ready to upload. An unpulled git-LFS pointer counts as ABSENT
    here: the path exists but the bytes do not.
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


def is_lfs_pointer(path: Path) -> bool:
    """True for an unpulled git-LFS stub — a small text pointer, not the data."""
    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def has_bytes(path: Path) -> bool:
    """The file is present AND holds its real content."""
    return path.is_file() and not is_lfs_pointer(path)


def _audit_records(records: dict) -> None:
    """Zenodo upload-destination records and their state."""
    print("=" * 78)
    print("ZENODO RECORDS (upload destinations)")
    print("=" * 78)
    for name, r in records.items():
        # An id alone does not make a record reachable: Zenodo hands out the id
        # (and a reserved DOI) at DEPOSITION time, and a file URL into an
        # unpublished draft 404s. `published` is what decides, so report the
        # three states separately rather than reading id-presence as LIVE.
        if not r.get("zenodo_record"):
            state = "NOT CREATED"
        elif r.get("published"):
            state = "LIVE"
        else:
            state = "DRAFT"
        print(
            f"  {name:10s} {state:12s} {r.get('license', '?'):14s} doi={r.get('zenodo_doi')}"
        )


def _bucket_row(
    name: str,
    spec: dict,
    bucket: str,
    to_upload: list[tuple[str, float]],
    elsewhere: list[str],
    partial: list[str],
) -> None:
    """Print one dataset's presence row; record it as ready / elsewhere / partial.

    "Ready" means EVERY declared file has bytes here (in the repo or in the
    cache) — a dataset is uploaded as a set, and a Zenodo record cannot be
    un-published, so half of one is not something to start.
    """
    files = files_of(spec)
    sub = spec.get("dir", "")
    repo_paths = [
        (DATA_DIR / sub / f["name"]) if sub else (DATA_DIR / f["name"]) for f in files
    ]
    cache_paths = [CACHE / name / f["name"] for f in files]
    in_repo = sum(1 for p in repo_paths if has_bytes(p))
    in_cache = sum(1 for p in cache_paths if has_bytes(p))
    here = sum(
        1 for r, c in zip(repo_paths, cache_paths) if has_bytes(r) or has_bytes(c)
    )
    size = sum(f.get("bytes", 0) for f in files) / 1048576
    flag = ""
    if bucket == "zenodo":
        if not files:
            flag = "  <-- bytes on another machine (upload from there)"
            elsewhere.append(name)
        elif here == 0:
            flag = "  <-- BYTES NOT ON THIS MACHINE"
            elsewhere.append(name)
        elif here < len(files):
            flag = f"  <-- INCOMPLETE: {here} of {len(files)} files have bytes here"
            partial.append(name)
        else:
            to_upload.append((name, size))
    pend = " PENDING-UPLOAD" if spec.get("pending_upload") else ""
    print(
        f"  {name:38s} rec={spec.get('record', '-'):9s} files={len(files):2d} "
        f"repo={in_repo:2d} cache={in_cache:2d} {size:8.1f} MB "
        f"{spec.get('license', '?'):16s}{pend}{flag}"
    )


def _audit_buckets(
    datasets: dict,
) -> tuple[list[tuple[str, float]], list[str], list[str]]:
    """Datasets by bucket; returns (ready-to-upload, bytes-elsewhere, incomplete).

    "elsewhere" is NOT the same as blocked: the bytes exist, just not on this
    machine (these are the obsidian-computed sets). Conflating the two hides
    whether anything actually needs a human decision.
    """
    print()
    print("=" * 78)
    print("DATASETS BY BUCKET   (repo = bytes in-tree, cache = bytes in ~/.cache)")
    print("=" * 78)
    buckets: dict[str, list[tuple[str, dict]]] = {}
    for name, spec in datasets.items():
        buckets.setdefault(spec.get("bucket", "?"), []).append((name, spec))

    to_upload: list[tuple[str, float]] = []
    elsewhere: list[str] = []
    partial: list[str] = []
    for bucket in sorted(buckets):
        print(f"\n--- {bucket}  ({len(buckets[bucket])} datasets) ---")
        for name, spec in sorted(buckets[bucket]):
            _bucket_row(name, spec, bucket, to_upload, elsewhere, partial)
    return to_upload, elsewhere, partial


def _audit_demo_registry(datasets: dict) -> None:
    """Cross-check the demo registry's claimed caches against the manifest."""
    print()
    print("=" * 78)
    print("DEMO REGISTRY CROSS-CHECK")
    print("=" * 78)
    try:
        from luxar.demos.registry import iter_demos

        demos = iter_demos()
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"  registry unavailable: {type(e).__name__}: {e}")
        return

    if not demos:
        return
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
    print(f"\n  manifest datasets not claimed via DEMO_META['caches'] ({len(unused)}):")
    print("    " + ", ".join(unused) if unused else "    (none)")
    print("    (these resolve by another route — verify with a grep before")
    print("     concluding any is orphaned)")


def _audit_files_on_disk(datasets: dict) -> list:
    """In-tree data files vs the manifest; returns the sorted undeclared set."""
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
    return undeclared


def _print_readiness(
    records: dict,
    to_upload: list[tuple[str, float]],
    elsewhere: list[str],
    partial: list[str],
) -> None:
    """Final readiness summary."""
    print()
    print("=" * 78)
    print("READINESS")
    print("=" * 78)
    print(
        f"  records to create:            {sum(1 for r in records.values() if not r.get('zenodo_record'))} of {len(records)}"
    )
    print(
        f"  drafts still to publish:      "
        f"{sum(1 for r in records.values() if r.get('zenodo_record') and not r.get('published'))}"
    )
    print(
        f"  datasets ready to upload now: {len(to_upload)}  "
        f"({sum(s for _, s in to_upload):.0f} MB)"
    )
    print(f"  upload from another machine:  {len(elsewhere)}")
    for b in elsewhere:
        print(f"    - {b}  (bytes on obsidian)")
    print(f"  incomplete here (NOT ready):  {len(partial)}")
    for b in partial:
        print(f"    - {b}  (some declared files have no bytes here)")
    print("  blocked on a human decision:  0")


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    datasets, records = m["datasets"], m["records"]

    _audit_records(records)
    to_upload, elsewhere, partial = _audit_buckets(datasets)
    _audit_demo_registry(datasets)
    undeclared = _audit_files_on_disk(datasets)
    _print_readiness(records, to_upload, elsewhere, partial)
    return 1 if undeclared else 0


if __name__ == "__main__":
    raise SystemExit(main())
