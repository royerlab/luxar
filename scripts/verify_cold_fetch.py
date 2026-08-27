#!/usr/bin/env python3
"""Verify a hosted dataset really is fetchable, from nothing, and is the right bytes.

This is the gate for retiring the in-repo git-LFS demo payloads. It answers one
question that no other check can answer once the payloads are gone:

    On a machine with no cache and no in-repo copy, does ``ensure_dataset``
    obtain each declared file, and is what it obtained byte-identical to what
    the record is pinned to?

Run it BEFORE ``git rm``-ing a dataset's payloads, never after.

Why it has to exist
-------------------
The obvious safety net does not work here. ``DatasetUnavailable`` subclasses
``FileNotFoundError`` and **thirteen demos catch it deliberately** and fall back
to refitting locally — see the class docstring in
``luxar.demos._support.datasets.data_fetch``. That is a good design, but it means
that after the payloads are removed a broken hosting configuration does not
surface as a failure. It surfaces as a multi-minute GPU refit on a user's
machine, from source data they do not have, and the process exit code stays 0.

So "did the demo run?" cannot be the check, and neither can any exit-code sweep
downstream of it. The check has to happen while the in-repo copy still exists,
by deliberately hiding it.

Three requirements, each load-bearing
-------------------------------------
1. **Assert on sha256, never on "a file appeared."**
   The demo-site origin — and Cloudflare Pages generally — answers a miss with
   **HTTP 200 and ``text/html``**, not 404. A status-code check therefore passes
   on a missing object. Worse, ``ensure_dataset`` would then quarantine the HTML
   error page as a corrupt archive, reporting *data corruption* where the truth
   is *missing file*. Only the digest tells those two apart. Do not "simplify"
   this to a status or size check.

2. **Hide the in-repo copy, do not merely use a fresh cache.**
   ``ensure_dataset`` resolves cache -> in-repo LFS -> hosted. A fresh
   ``cache_root`` alone leaves leg two intact, so the run passes by reading
   ``demos/data/`` and proves nothing about the record. This module redirects
   ``data_fetch._DEMOS_DATA_DIR`` at an empty directory instead of moving the
   real tree aside: same guarantee, but it never mutates a working tree that
   other people and jobs are using.

3. **Prefer ``hosted_sha256`` over ``sha256``.**
   They are two different contracts: ``sha256`` describes the repo copy,
   ``hosted_sha256`` the record copy, and for most datasets they differ. Only
   the hosted digest says anything about what a stranger will download.

   Honest scoping of this one: the download leg inside ``ensure_dataset``
   already validates strictly against the hosted pin, so on a cold cache it is
   what rejects wrong bytes first, and the re-check below is a backstop. It is
   kept because it makes this gate independent of that internal — a resolver
   that ever accepted either contract on the download path (as
   ``_accepted_contract`` already does for a CACHED file) would otherwise pass
   a mis-uploaded record silently. Requirement 2, not this one, is what makes
   the harness irreplaceable.

Usage
-----
    python scripts/verify_cold_fetch.py                    # every reachable dataset
    python scripts/verify_cold_fetch.py gsplats_kidney     # named datasets
    python scripts/verify_cold_fetch.py --list             # what would be attempted

Exit codes: 0 everything checked passed, 1 a verification failed, 2 a usage or
configuration error. Datasets whose hosting is still dormant are reported as
SKIP and do not fail the run — that is the expected state before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages/luxar/src"))

from luxar.demos._support.datasets import data_fetch  # noqa: E402

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def hosted_datasets(manifest: dict[str, Any]) -> list[str]:
    """Dataset names whose files are meant to come from a record."""
    return sorted(
        name
        for name, spec in manifest.get("datasets", {}).items()
        if spec.get("bucket") == "zenodo"
    )


def is_reachable(manifest: dict[str, Any], name: str) -> bool:
    """Whether the hosted leg would produce a URL at all.

    Mirrors ``zenodo_file_url``'s own gating rather than reimplementing it, so a
    dormant record is reported as SKIP instead of failing.
    """
    spec = manifest["datasets"][name]
    record = manifest.get("records", {}).get(spec.get("record"), {})
    files, _ = data_fetch.resolve_variant(name, spec, None)
    if not files:
        return False
    return data_fetch.zenodo_file_url(record, files[0]["name"]) is not None


def expected_digest(entry: dict[str, Any]) -> tuple[Optional[str], str]:
    """The digest to check against, and which contract it came from."""
    if entry.get("hosted_sha256"):
        return entry["hosted_sha256"], "hosted"
    if entry.get("sha256"):
        # Falling back is worth doing but worth saying: this pins the REPO copy,
        # so a record holding different bytes would be reported as a mismatch.
        return entry["sha256"], "repo (no hosted_sha256 declared)"
    return None, "none declared"


def verify(name: str, manifest: dict[str, Any], keep: bool) -> tuple[bool, str]:
    """Fetch *name* into a throwaway cache with no in-repo copy, and check it."""
    spec = manifest["datasets"][name]
    files, _ = data_fetch.resolve_variant(name, spec, None)
    if not files:
        return True, "SKIP  no files declared (pending upload)"
    if not is_reachable(manifest, name):
        return True, "SKIP  hosting dormant (no base_url, record unpublished)"

    # `data_fetch` imports this from `.lfs`, so under --no-implicit-reexport
    # mypy cannot see it as an attribute even though it is one at runtime.
    original_lfs_dir = data_fetch._DEMOS_DATA_DIR  # type: ignore[attr-defined]
    cache_dir = Path(tempfile.mkdtemp(prefix=f"luxar-coldfetch-{name}-"))
    empty_lfs = Path(tempfile.mkdtemp(prefix="luxar-no-inrepo-"))
    try:
        # Requirement 2: the in-repo leg must not be able to satisfy this.
        data_fetch._DEMOS_DATA_DIR = empty_lfs  # type: ignore[attr-defined]
        try:
            paths = data_fetch.ensure_dataset(
                name, cache_root=cache_dir, manifest=manifest, verbose=False
            )
        except data_fetch.DatasetUnavailable as exc:
            return False, f"FAIL  nothing obtainable without the in-repo copy: {exc}"
        except data_fetch.LocalComputeDataset as exc:
            return True, f"SKIP  not redistributable: {exc}"
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            return False, f"FAIL  {type(exc).__name__}: {exc}"

        by_name = {p.name: p for p in paths}
        for entry in files:
            path = by_name.get(entry["name"])
            if path is None or not path.exists():
                return False, f"FAIL  {entry['name']}: not produced"
            wanted, contract = expected_digest(entry)
            if wanted is None:
                return False, f"FAIL  {entry['name']}: manifest declares no digest"
            got = sha256_of(path)
            if got != wanted:
                return False, (
                    f"FAIL  {entry['name']}: sha256 {got[:16]}… != "
                    f"{contract} {wanted[:16]}… "
                    f"({path.stat().st_size} bytes fetched)"
                )
        digests = ", ".join(sorted({expected_digest(e)[1] for e in files}))
        return True, f"OK    {len(files)} file(s) verified against {digests} sha256"
    finally:
        data_fetch._DEMOS_DATA_DIR = original_lfs_dir  # type: ignore[attr-defined]
        shutil.rmtree(empty_lfs, ignore_errors=True)
        if keep:
            print(f"      kept: {cache_dir}")
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what would be attempted, fetch nothing",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the throwaway caches for inspection"
    )
    args = parser.parse_args(argv)

    try:
        manifest = data_fetch.load_manifest()
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot load the manifest: {exc}", file=sys.stderr)
        return 2

    known = hosted_datasets(manifest)
    names = args.datasets or known
    unknown = [n for n in names if n not in manifest.get("datasets", {})]
    if unknown:
        print(f"error: unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.list:
        for name in names:
            state = "reachable" if is_reachable(manifest, name) else "dormant"
            print(f"{name:<40} {state}")
        return 0

    failures = 0
    skipped = 0
    for name in names:
        ok, detail = verify(name, manifest, args.keep)
        print(f"{name:<40} {detail}")
        if not ok:
            failures += 1
        elif detail.startswith("SKIP"):
            skipped += 1

    checked = len(names) - skipped - failures
    print()
    print(
        f"cold fetch: {checked} verified, {skipped} skipped (dormant), {failures} failed"
    )
    if failures:
        print(
            "\nA failure here means a stranger cannot obtain this dataset. Do NOT\n"
            "remove its in-repo payload — 13 demos would silently start refitting\n"
            "instead of reporting an error.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
