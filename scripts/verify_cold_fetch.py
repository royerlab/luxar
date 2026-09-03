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
``FileNotFoundError`` and **sixteen demos catch it deliberately**. Most fall back
to refitting locally; the Census demo prints regeneration guidance and exits
non-zero instead. For the refitting fallbacks, that means a broken hosting
configuration does not surface as a failure after the payloads are removed. It
surfaces as a multi-minute GPU refit on a user's machine, from source data they
do not have, and the process exit code stays 0. See the class docstring in
``luxar.demos._support.datasets.data_fetch``.

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
    python scripts/verify_cold_fetch.py                     # every hosted target
    python scripts/verify_cold_fetch.py gsplats_kidney      # one named target
    python scripts/verify_cold_fetch.py gsplats_kidney --allow-skip
    python scripts/verify_cold_fetch.py --require-verified N # pre-removal teardown
    python scripts/verify_cold_fetch.py --cache-root /data  # large temp caches
    python scripts/verify_cold_fetch.py --list              # what would be attempted

Exit codes: 0 everything checked passed, 1 a verification failed, 2 a usage or
configuration error. A dormant target is reported as SKIP. A bare all-target
run tolerates those skips before publication, but a SKIP on an explicitly named
target fails unless ``--allow-skip`` is passed. Before removing payloads, use
``--require-verified N`` with the expected target count so no variant is missed.
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


def variants_for(spec: dict[str, Any]) -> list[Optional[str]]:
    """Every variant to verify, or one unqualified target for flat datasets."""
    variants = spec.get("variants")
    return list(variants) if variants else [None]


def target_label(name: str, variant: Optional[str]) -> str:
    return f"{name}:{variant}" if variant else name


def verification_targets(
    manifest: dict[str, Any], names: list[str]
) -> list[tuple[str, Optional[str]]]:
    return [
        (name, variant)
        for name in names
        for variant in variants_for(manifest["datasets"][name])
    ]


def is_reachable(
    manifest: dict[str, Any], name: str, variant: Optional[str] = None
) -> bool:
    """Whether the hosted leg would produce a URL at all.

    Mirrors ``zenodo_file_url``'s own gating rather than reimplementing it, so a
    dormant record is reported as SKIP instead of failing.
    """
    spec = manifest["datasets"][name]
    record = manifest.get("records", {}).get(spec.get("record"), {})
    files, _ = data_fetch.resolve_variant(name, spec, variant)
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


def _check_digests(files: list[dict[str, Any]], paths: list[Path]) -> Optional[str]:
    by_name = {path.name: path for path in paths}
    for entry in files:
        path = by_name.get(entry["name"])
        if path is None or not path.exists():
            return f"FAIL  {entry['name']}: not produced"
        wanted, contract = expected_digest(entry)
        if wanted is None:
            return f"FAIL  {entry['name']}: manifest declares no digest"
        got = sha256_of(path)
        if got != wanted:
            return (
                f"FAIL  {entry['name']}: sha256 {got[:16]}… != "
                f"{contract} {wanted[:16]}… "
                f"({path.stat().st_size} bytes fetched)"
            )
    return None


def _temporary_dir(prefix: str, cache_root: Optional[Path]) -> Path:
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=cache_root))


def verify(
    name: str,
    manifest: dict[str, Any],
    keep: bool,
    *,
    variant: Optional[str] = None,
    cache_root: Optional[Path] = None,
) -> tuple[bool, str, Optional[Path]]:
    """Fetch *name* into a throwaway cache with no in-repo copy, and check it."""
    spec = manifest["datasets"][name]
    bucket = spec.get("bucket")
    if bucket != "zenodo":
        return True, f"SKIP  not hosted ({bucket} dataset)", None
    files, _ = data_fetch.resolve_variant(name, spec, variant)
    if not files:
        return True, "SKIP  no files declared (pending upload)", None
    if not is_reachable(manifest, name, variant):
        return True, "SKIP  hosting dormant (no base_url, record unpublished)", None

    # `data_fetch` imports this from `.lfs`, so under --no-implicit-reexport
    # mypy cannot see it as an attribute even though it is one at runtime.
    original_lfs_dir = data_fetch._DEMOS_DATA_DIR  # type: ignore[attr-defined]
    label = target_label(name, variant)
    cache_dir = _temporary_dir(f"luxar-coldfetch-{label}-", cache_root)
    empty_lfs = _temporary_dir("luxar-no-inrepo-", cache_root)
    kept_path = cache_dir if keep else None
    try:
        # Requirement 2: the in-repo leg must not be able to satisfy this.
        data_fetch._DEMOS_DATA_DIR = empty_lfs  # type: ignore[attr-defined]
        try:
            paths = data_fetch.ensure_dataset(
                name,
                variant=variant,
                cache_root=cache_dir,
                manifest=manifest,
                verbose=False,
            )
        except data_fetch.DatasetUnavailable as exc:
            return (
                False,
                f"FAIL  nothing obtainable without the in-repo copy: {exc}",
                kept_path,
            )
        except data_fetch.LocalComputeDataset as exc:
            return True, f"SKIP  not redistributable: {exc}", kept_path
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            return False, f"FAIL  {type(exc).__name__}: {exc}", kept_path

        failure = _check_digests(files, paths)
        if failure is not None:
            return False, failure, kept_path
        digests = ", ".join(sorted({expected_digest(e)[1] for e in files}))
        return (
            True,
            f"OK    {len(files)} file(s) verified against {digests} sha256",
            kept_path,
        )
    finally:
        data_fetch._DEMOS_DATA_DIR = original_lfs_dir  # type: ignore[attr-defined]
        shutil.rmtree(empty_lfs, ignore_errors=True)
        if not keep:
            shutil.rmtree(cache_dir, ignore_errors=True)


def _listing_state(manifest: dict[str, Any], name: str, variant: Optional[str]) -> str:
    spec = manifest["datasets"][name]
    if spec.get("bucket") != "zenodo":
        return "not hosted"
    return "reachable" if is_reachable(manifest, name, variant) else "dormant"


def _result_exit_code(
    *,
    failures: int,
    skipped: int,
    checked: int,
    explicit: bool,
    allow_skip: bool,
    require_verified: Optional[int],
) -> int:
    if failures:
        print(
            "\nA failure here means a stranger cannot obtain this dataset. Do NOT\n"
            "remove its in-repo payload — 13 demos would silently start refitting\n"
            "instead of reporting an error.",
            file=sys.stderr,
        )
        return 1

    # A tolerant skip branch is the same fails-open shape as a bad guard: it turns
    # "we could not check" into a clean exit, and the caller reads exit 0 as
    # permission to delete the only copy. So a skip is a pass only when nobody
    # asked for that dataset by name.
    if skipped and explicit and not allow_skip:
        print(
            f"\nerror: {skipped} explicitly requested target(s) were NOT verified "
            "— you asked for them by name and got no answer. This is not a pass. "
            "Pass --allow-skip only when exploring, never when deciding whether a "
            "payload can be removed.",
            file=sys.stderr,
        )
        return 1

    if require_verified is not None and checked < require_verified:
        print(
            f"\nerror: {checked} target(s) verified, --require-verified "
            f"{require_verified} demanded.",
            file=sys.stderr,
        )
        return 1

    if checked == 0 and skipped:
        print(
            "NOTE: nothing was actually verified — every target was skipped. "
            "That is expected only for records that are not yet published, and "
            "is NOT evidence any payload is safe to remove."
        )
    return 0


def _run_targets(
    targets: list[tuple[str, Optional[str]]],
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    failures = 0
    skipped = 0
    not_hosted = 0
    for name, variant in targets:
        ok, detail, kept_path = verify(
            name,
            manifest,
            args.keep,
            variant=variant,
            cache_root=args.cache_root,
        )
        print(f"{target_label(name, variant):<40} {detail}")
        if kept_path is not None:
            print(f"      kept: {kept_path}")
        if not ok:
            failures += 1
        elif detail.startswith("SKIP  not hosted"):
            not_hosted += 1
        elif detail.startswith("SKIP"):
            skipped += 1

    checked = len(targets) - skipped - not_hosted - failures
    print()
    summary = f"cold fetch: {checked} verified, {skipped} skipped"
    if not_hosted:
        summary += f", {not_hosted} not hosted"
    print(f"{summary}, {failures} failed")
    return failures, skipped, not_hosted


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
    parser.add_argument(
        "--require-verified",
        type=int,
        metavar="N",
        help=(
            "fail unless at least N targets actually verified. Use this at "
            "teardown: a run that skips everything otherwise exits 0."
        ),
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help=(
            "tolerate a skip on an explicitly named target (exploration only; "
            "never when deciding whether a payload can be removed)"
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="directory for throwaway caches (default: system temporary directory)",
    )
    args = parser.parse_args(argv)
    if (
        args.cache_root is not None
        and args.cache_root.exists()
        and not args.cache_root.is_dir()
    ):
        print(
            f"error: --cache-root is not a directory: {args.cache_root}",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = data_fetch.load_manifest()
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot load the manifest: {exc}", file=sys.stderr)
        return 2

    known = hosted_datasets(manifest)
    explicit = bool(args.datasets)
    names = args.datasets or known
    unknown = [n for n in names if n not in manifest.get("datasets", {})]
    if unknown:
        print(f"error: unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    targets = verification_targets(manifest, names)
    if args.list:
        for name, variant in targets:
            print(
                f"{target_label(name, variant):<40} "
                f"{_listing_state(manifest, name, variant)}"
            )
        return 0

    failures, skipped, not_hosted = _run_targets(targets, manifest, args)
    checked = len(targets) - skipped - not_hosted - failures
    return _result_exit_code(
        failures=failures,
        skipped=skipped,
        checked=checked,
        explicit=explicit,
        allow_skip=args.allow_skip,
        require_verified=args.require_verified,
    )


if __name__ == "__main__":
    raise SystemExit(main())
