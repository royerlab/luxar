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

The default is entirely local — no network — so it stays fast and runs anywhere.
``--live`` adds a fourth source, the Zenodo depositions themselves, and is the
check to run immediately before publishing by hand::

    ZENODO_TOKEN=... python scripts/zenodo_migration_audit.py --live

It is READ-ONLY: the only verb it issues is GET, and it cannot publish. What it
adds is the one question local state cannot answer — whether the files ON the
records are the files the manifest promises, in both directions, with no scratch
objects left behind and no stale numbers in the description.

One caveat it enforces rather than assumes: if nearly every pin disagrees with
the records, the answer is almost never "the records are broken" but "this is not
the manifest that will ship" (an un-merged re-pin branch). It says so, because
correct-looking output about a superseded input is otherwise indistinguishable
from a real defect.

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

import argparse
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _repo_from_argv(argv: list[str]) -> Path:
    """The repo-root positional, ignoring flags.

    Read at import time because ``MANIFEST``/``DATA_DIR`` are module constants
    (and the tests patch ``sys.argv`` before importing). Flags must be skipped or
    ``audit.py --live`` would resolve a repo root literally named ``--live`` and
    then fail on a missing manifest.
    """
    positional = [a for a in argv if not a.startswith("-")]
    if positional:
        return Path(positional[0]).resolve()
    return Path(__file__).resolve().parent.parent


REPO = _repo_from_argv(sys.argv[1:])
sys.path.insert(0, str(REPO / "packages/luxar/src"))

MANIFEST = REPO / "packages/luxar/src/luxar/demos/data_manifest.json"
DATA_DIR = REPO / "packages/luxar/src/luxar/demos/data"
CACHE = Path.home() / ".cache" / "luxar"

DATA_SUFFIXES = (".zip", ".npz", ".parquet", ".npy")


def files_of(spec: dict) -> list[tuple[str, dict]]:
    """``(variant, entry)`` per declared file; variant ``""`` when there are none.

    The variant name has to travel with the entry: a variant's files sit one level
    deeper than the dataset's, ``<dir>/<variant>/`` in-repo and ``<name>/<variant>/``
    in the cache, exactly as ``ensure_dataset`` resolves them. Flattening the
    variants into a bare file list drops that segment, so every path built from it
    points at a file that is never there — h2afva's pinned 253tp being the first
    declared variant file this could get wrong.
    """
    files = spec.get("files") or []
    if files:
        return [("", f) for f in files]
    return [
        (vname, f)
        for vname, v in (spec.get("variants") or {}).items()
        for f in (v.get("files") or [])
    ]


def _under(root: Path, *parts: str) -> Path:
    """*root* joined with the non-empty *parts* (an empty dir/variant is a no-op)."""
    return root.joinpath(*[p for p in parts if p])


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
        # states separately rather than reading id-presence as LIVE.
        # Decide it in the SAME ORDER as `zenodo_file_url`, or this report would
        # call a record dormant that the fetch leg happily downloads from: an
        # explicit `base_url` outranks the flag (that is how the Sandbox
        # rehearsal fetches while the real record is still a draft), and only an
        # explicit `published: false` means draft.
        if r.get("base_url"):
            state = "LIVE (base_url)"
        elif not r.get("zenodo_record"):
            state = "NOT CREATED"
        elif r.get("published") is False:
            state = "DRAFT"
        else:
            state = "LIVE"
        print(
            f"  {name:10s} {state:15s} {r.get('license', '?'):14s} doi={r.get('zenodo_doi')}"
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
    repo_paths = [_under(DATA_DIR, sub, var, f["name"]) for var, f in files]
    cache_paths = [_under(CACHE, name, var, f["name"]) for var, f in files]
    in_repo = sum(1 for p in repo_paths if has_bytes(p))
    in_cache = sum(1 for p in cache_paths if has_bytes(p))
    here = sum(
        1 for r, c in zip(repo_paths, cache_paths) if has_bytes(r) or has_bytes(c)
    )
    size = sum(f.get("bytes", 0) for _, f in files) / 1048576
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
        for var, f in files_of(spec):
            declared.add(_under(DATA_DIR, sub, var, f["name"]))
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
        f"{sum(1 for r in records.values() if r.get('zenodo_record') and r.get('published') is False)}"
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


# ---------------------------------------------------------------------------
# Live deposition checks (opt-in, --live)
#
# Everything above answers "is the migration ready?" from local state alone.
# These answer the one question local state cannot: does what is ON the records
# match what the manifest promises? Every check below is a mistake actually made
# during the migration, so none of them is hypothetical.
#
# Kept as pure functions over (manifest, depositions) with a thin network shell,
# so the whole thing is testable offline — and so the default stays offline.
# ---------------------------------------------------------------------------

ZENODO_API = "https://zenodo.org/api/deposit/depositions"

#: Above this share of pinned files failing, the diagnosis is "wrong manifest",
#: not "broken records". Two thirds is well clear of the handful of genuine
#: mismatches a real pre-publish run should ever show.
_STALE_MANIFEST_SHARE = 0.66


def hosted_size(entry: dict) -> int:
    """The size of the copy the RECORD serves, not the one this repo ships.

    A manifest entry carries both once a refit makes them differ: `bytes` and
    `sha256` describe the in-repo copy, `hosted_bytes` and `hosted_sha256` the
    record's. Comparing the local size against a deposition reports a mismatch
    for every diverged file — correct arithmetic, wrong end of the contract — so
    every live comparison must resolve the hosted value first.
    """
    return int(entry.get("hosted_bytes") or entry.get("bytes") or 0)


def pins_of(datasets: dict) -> dict[str, tuple[str, int]]:
    """``filename -> (record, hosted bytes)`` per zenodo file.

    Deliberately the HOSTED side of each entry: this map exists to be compared
    against a live deposition, whose API exposes md5 rather than sha256. Only
    size can therefore be compared here; a same-size swap is out of scope. See
    :func:`hosted_size`.

    Uses :func:`files_of`, so a variant's files are included on the same footing
    as a dataset's own — h2afva's pinned 253tp is only reachable that way.
    """
    pins: dict[str, tuple[str, int]] = {}
    for spec in datasets.values():
        if spec.get("bucket") != "zenodo":
            continue
        for _var, f in files_of(spec):
            pins[f["name"]] = (spec.get("record", ""), hosted_size(f))
    return pins


def dataset_totals(datasets: dict) -> dict[str, int]:
    """``dataset -> summed declared bytes``, for description size claims.

    A record's contents table names DATASETS, whose size is the sum over their
    files, while a size claim can also name a single file. Both are resolved, and
    both from the HOSTED side — the description describes the record's files, so a
    claim checked against in-repo sizes would be checked against the wrong bytes.
    """
    return {
        name: sum(hosted_size(f) for _var, f in files_of(spec))
        for name, spec in datasets.items()
    }


def human_bytes(n: int) -> str:
    """Match the units the record descriptions are written in (MB, then GB)."""
    mb = n / 1e6
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.1f} MB"


def _description_size_claims(desc: str) -> list[re.Match[str]]:
    """Rendered ``<li><code>name</code> (N MB)</li>`` size claims."""
    return list(re.finditer(r"<li><code>([^<]+)</code>\s*\(([\d.]+)\s?([MG]B)", desc))


def _check_pins(
    tag: str, rec: str, hosted: dict, pins: dict[str, tuple[str, int]]
) -> list[str]:
    """Pins vs hosted files, in BOTH directions.

    One direction alone misses the two ways this actually went wrong: a
    superseded file left on the record (which keeps a right-looking name), and a
    pin for something never uploaded (which reads as success until the download
    404s after publication).
    """
    fails = []
    for name, f in sorted(hosted.items()):
        pin = pins.get(name)
        if pin is None:
            fails.append(f"{tag} {name} is on the record but NOT pinned")
        elif pin[0] != rec:
            fails.append(f"{tag} {name} is on this record but pinned to {pin[0]}")
        elif f.get("filesize") is None:
            fails.append(f"{tag} {name} has no filesize/size")
        elif pin[1] != f["filesize"]:
            fails.append(f"{tag} {name} pinned {pin[1]:,} but hosted {f['filesize']:,}")
    fails += [
        f"{tag} {name} is pinned but ABSENT from the record"
        for name, (r, _b) in sorted(pins.items())
        if r == rec and name not in hosted
    ]
    return fails


def _check_no_scratch(tag: str, hosted: dict) -> list[str]:
    """No probe object left by a failed multipart upload.

    A published record cannot be tidied afterwards.
    """
    return [
        f"{tag} scratch file on the record: {name}"
        for name in sorted(hosted)
        if name.startswith("_") or name.endswith((".corrupt", ".part", ".tmp"))
    ]


def _check_description(
    tag: str,
    desc: str,
    hosted: dict,
    pins: dict[str, tuple[str, int]],
    totals: dict[str, int],
) -> tuple[list[str], list[str]]:
    """The current drafts' rendered HTML claims; stale numbers get published.

    This checks the HTML shape returned by the Zenodo API today, not the Markdown
    source emitted by ``gen_zenodo_records.py``. If Zenodo changes its rendering,
    the zero-claims warning below makes that loss of coverage visible.
    """
    fails: list[str] = []
    warns: list[str] = []
    claims = _description_size_claims(desc)
    for mt in claims:
        entry = mt.group(1)
        claimed_value = float(mt.group(2))
        unit = mt.group(3)
        claimed = f"{mt.group(2)} {unit}"
        pin = pins.get(entry)
        size = totals.get(entry) or (pin[1] if pin else None)
        if size is None:
            warns.append(f"{tag} description names {entry}, not resolvable to a pin")
        else:
            actual_value_text, actual_unit = human_bytes(size).split()
            if unit == actual_unit and round(claimed_value, 1) == round(
                float(actual_value_text), 1
            ):
                continue
            fails.append(
                f"{tag} description says {entry} is {claimed}, "
                f"actually {human_bytes(size)}"
            )
    if hosted and not claims:
        warns.append(f"{tag} description has 0 size claims for {len(hosted)} files")
    if "<table>" not in desc:
        fails.append(f"{tag} description has no contents table")
    else:
        rows = desc.count("<tr>")
        want = len(hosted) + 1  # + the header row
        if rows != want:
            fails.append(
                f"{tag} table has {rows} rows for {len(hosted)} files (want {want})"
            )
    if "<thead" in desc:
        # Zenodo's HTML sanitiser drops <thead>, so the header row vanishes on
        # the rendered page while looking correct in the payload you sent.
        warns.append(f"{tag} description contains <thead>, which Zenodo strips")
    return fails, warns


def check_deposition(
    rec: str,
    dep: dict,
    pins: dict[str, tuple[str, int]],
    totals: dict[str, int],
    record_meta: dict,
) -> tuple[list[str], list[str]]:
    """Compare one live deposition against the manifest. Returns (fails, warns).

    Pure: no I/O. *dep* is the deposition dict as the API returns it.
    """
    meta = dep.get("metadata") or {}
    tag = f"[{rec}]"
    hosted: dict[str, dict] = {}
    malformed: list[str] = []
    for entry in dep.get("files") or []:
        name = entry.get("filename") or entry.get("key")
        if not name:
            malformed.append(f"{tag} file entry has no filename/key")
            continue
        size = entry.get("filesize")
        if size is None:
            size = entry.get("size")
        hosted[str(name)] = {**entry, "filesize": size}

    live_published = bool(dep.get("submitted"))
    manifest_published = bool(record_meta.get("published"))

    # Publishing is a one-way door. A submitted deposition is only an error
    # while the manifest still claims it is a draft; otherwise it is the
    # expected live state of a published record.
    fails = (
        [f"{tag} ALREADY SUBMITTED — stop"]
        if live_published and not manifest_published
        else []
    )
    fails += malformed
    fails += _check_pins(tag, rec, hosted, pins)
    fails += _check_no_scratch(tag, hosted)

    desc_fails, warns = _check_description(
        tag, meta.get("description") or "", hosted, pins, totals
    )
    fails += desc_fails

    # Metadata Zenodo requires to publish at all.
    fails += [
        f"{tag} metadata missing {key}"
        for key in ("title", "creators", "license", "upload_type")
        if not meta.get(key)
    ]

    if manifest_published != live_published:
        state = (
            "live deposition is published but manifest is not"
            if live_published
            else "manifest marks published but live deposition is not"
        )
        warns.append(f"{tag} {state}")
    return fails, warns


def diagnose_manifest_staleness(fails: list[str], pins: dict) -> str | None:
    """Name the likely cause when nearly every pin fails: the wrong manifest.

    Exists because of a real incident. An audit run against ``dev`` reported the
    records stale "essentially everywhere" and it was believed, when in fact the
    records were ahead and the manifest was behind — the re-pin was sitting in an
    unmerged PR. Correct-looking output about a superseded input is the failure
    mode this whole tool is otherwise blind to, so say it out loud instead of
    letting a reader interpret 30 mismatches as 30 broken files.
    """
    if not pins:
        return None
    mismatched = sum(
        1 for f in fails if ("but hosted" in f or "ABSENT from the record" in f)
    )
    if mismatched < _STALE_MANIFEST_SHARE * len(pins):
        return None
    return (
        f"{mismatched} of {len(pins)} pins disagree with the live records.\n"
        "  That is too many to be a data problem: the likely cause is that THIS\n"
        "  manifest is not the one that will ship. Check for an unmerged re-pin\n"
        "  branch before believing any failure above:\n"
        "      git log --oneline --all -- packages/luxar/src/luxar/demos/data_manifest.json\n"
        "  Re-run the audit from that branch's worktree."
    )


def fetch_deposition(dep_id: str, token: str) -> dict:
    """GET one deposition. The only network call in this file, and read-only.

    ``scripts/zenodo_upload_draft.py`` has a near-identical helper, and this is a
    deliberate duplication rather than an oversight: importing it would pull the
    repo's ONE mutating Zenodo tool into this process, and "read-only" here is
    asserted by grepping THIS file for mutating verbs. A ten-line GET is a cheap
    price for keeping that guarantee local and checkable.

    The Authorization header is UNREDIRECTED on purpose, matching
    ``scripts/zenodo_upload_draft.py``: urllib's redirect handler copies
    ``Request.headers`` onto the follow-up request, to whatever host a
    ``Location`` names. It also keeps the token out of the URL.
    """
    req = urllib.request.Request(  # noqa: S310 - literal https URL, built here
        f"{ZENODO_API}/{dep_id}", method="GET"
    )
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
            payload: dict = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Zenodo returned HTTP {exc.code} {exc.reason} for deposition {dep_id}. "
            "Check the id and that ZENODO_TOKEN has deposit:write scope."
        ) from exc
    except (
        OSError,
        http.client.HTTPException,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise SystemExit(
            f"Zenodo request failed for deposition {dep_id}: {exc}"
        ) from exc
    return payload


def _audit_live_depositions(
    manifest: dict, depositions: dict[str, dict], unchecked: list[str]
) -> int:
    """Compare the manifest against fetched depositions; returns the fail count."""
    print()
    print("=" * 78)
    print("LIVE DEPOSITIONS vs MANIFEST")
    print("=" * 78)
    datasets, records = manifest["datasets"], manifest["records"]
    pins = pins_of(datasets)
    totals = dataset_totals(datasets)
    checked_records = set(depositions)
    unchecked_records = set(unchecked)
    dangling_records = sorted(
        {record for record, _bytes in pins.values()}
        - checked_records
        - unchecked_records
    )

    all_fails = [
        f"[{name}] manifest has no zenodo_record; live check did not run"
        for name in unchecked
    ]
    all_fails += [
        f"[{record}] no fetched deposition for pinned files: "
        + ", ".join(
            sorted(name for name, (rec, _bytes) in pins.items() if rec == record)
        )
        for record in dangling_records
    ]
    all_warns: list[str] = []
    claim_counts: dict[str, int] = {}
    for rec, dep in sorted(depositions.items()):
        fails, warns = check_deposition(rec, dep, pins, totals, records.get(rec) or {})
        all_fails += fails
        all_warns += warns
        description = (dep.get("metadata") or {}).get("description") or ""
        claim_counts[rec] = len(_description_size_claims(description))

    compared_pins = sum(1 for record, _bytes in pins.values() if record in depositions)
    print(f"  records checked: {len(depositions)}   pins: {compared_pins}")
    for rec, count in sorted(claim_counts.items()):
        print(f"  [{rec}] description size claims parsed: {count}")
    for w in all_warns:
        print(f"  WARN {w}")
    for f in all_fails:
        print(f"  FAIL {f}")
    if not all_fails and not all_warns:
        print("  all pins match the live records; no scratch files; metadata complete")

    note = diagnose_manifest_staleness(all_fails, pins)
    if note:
        print()
        print("  !! PROBABLY THE WRONG MANIFEST, NOT BROKEN RECORDS")
        print(f"  {note}")
    return len(all_fails)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Flags only. The repo positional is consumed at import time for ``REPO``.

    It is accepted (and ignored) here so both spellings keep working:
    ``audit.py`` and ``audit.py /path/to/repo``.
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("repo", nargs="?", help="repo root (default: this checkout)")
    ap.add_argument(
        "--live",
        action="store_true",
        help="also compare the manifest against the live Zenodo depositions "
        "(needs ZENODO_TOKEN; read-only, never publishes)",
    )
    return ap.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])
    m = json.loads(MANIFEST.read_text())
    datasets, records = m["datasets"], m["records"]

    _audit_records(records)
    to_upload, elsewhere, partial = _audit_buckets(datasets)
    _audit_demo_registry(datasets)
    undeclared = _audit_files_on_disk(datasets)
    _print_readiness(records, to_upload, elsewhere, partial)

    live_fails = 0
    if args.live:
        token = os.environ.get("ZENODO_TOKEN")
        if not token:
            print()
            print("  refusing to run --live without ZENODO_TOKEN; no live check ran")
            return 1
        # Ids come from the manifest, not a constant: a hardcoded id is how a
        # gate ends up auditing a record nothing points at any more.
        wanted = {
            name: r["zenodo_record"]
            for name, r in records.items()
            if r.get("zenodo_record")
        }
        missing = sorted(set(records) - set(wanted))
        depositions = {n: fetch_deposition(i, token) for n, i in sorted(wanted.items())}
        live_fails = _audit_live_depositions(m, depositions, missing)

    return 1 if (undeclared or live_fails) else 0


if __name__ == "__main__":
    raise SystemExit(main())
