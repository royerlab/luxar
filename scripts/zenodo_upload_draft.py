#!/usr/bin/env python3
"""Add files to an UNPUBLISHED Zenodo draft deposition. Never publishes.

The companion to :mod:`scripts.zenodo_migration_audit`: the audit says what is
ready, this puts the bytes there. Deliberately the *only* mutating Zenodo tool in
the repo, and deliberately incapable of publishing.

Safety, by construction rather than by discipline:

* **There is no publish call in this file.** Publishing a record is a one-way
  door (a DOI is minted and the files become immutable), so it stays a manual,
  human act. Nothing here can perform it even by accident.
* **A submitted deposition is refused outright.** If ``submitted`` is true, or
  ``state`` is anything but ``unsubmitted``, the run aborts before touching a
  byte — that is what stops an "upload" from mutating a published record.
* **Dry run is the default.** Uploading requires ``--yes``. Without it you get
  the full plan and nothing happens.
* **Idempotent.** A file already on the deposition with a matching MD5 and size
  is skipped, so a re-run after an interruption resumes instead of duplicating.
* Every upload is **verified** against the MD5 Zenodo computes server-side.

Usage::

    export ZENODO_TOKEN=...                     # never passed on the command line
    python scripts/zenodo_upload_draft.py --deposition 21912280 \\
        --files delme/cell_tracking_bundle/*.gsplats.zarr.zip \\
                delme/cell_tracking_bundle/*_tracks.npz          # dry run
    python scripts/zenodo_upload_draft.py --deposition 21912280 --files ... --yes

Record ids (all unpublished as of 2026-08-14):
    21912280  Luxar demo datasets: permissively licensed (CC-BY, CC0, public domain)
    21912282  Luxar demo datasets: ShareAlike (CC BY-SA 4.0)
    21912284  Zebrafish h2afva 253-timepoint timelapse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://zenodo.org/api/deposit/depositions"


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - Zenodo's file checksum algorithm, not a security use
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch_deposition(dep_id: str, token: str) -> dict:
    """Fetch the deposition, refusing anything that is not an open draft."""
    try:
        dep = _get(f"{API}/{dep_id}?access_token={token}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Zenodo returned HTTP {exc.code} {exc.reason} for deposition {dep_id}. "
            "Check the id and that ZENODO_TOKEN has deposit:write scope."
        ) from exc

    submitted = bool(dep.get("submitted"))
    state = dep.get("state")
    title = str(dep.get("metadata", {}).get("title", ""))[:80]
    print(f"deposition {dep_id}: state={state!r} submitted={submitted}")
    print(f"  title: {title}")

    if submitted or state != "unsubmitted":
        raise SystemExit(
            f"REFUSING: deposition {dep_id} is submitted={submitted} state={state!r}. "
            "This tool only adds files to an OPEN DRAFT — a published record's files "
            "are immutable, and publishing is a manual act by a human."
        )
    if not dep.get("links", {}).get("bucket"):
        raise SystemExit(
            f"Deposition {dep_id} exposes no bucket link; cannot upload files."
        )
    return dep


def existing_files(dep: dict) -> dict[str, tuple[str, int]]:
    """``{filename: (md5, size)}`` already on the deposition."""
    out: dict[str, tuple[str, int]] = {}
    for entry in dep.get("files", []):
        name = entry.get("filename") or entry.get("key")
        checksum = str(entry.get("checksum", ""))
        if checksum.startswith("md5:"):
            checksum = checksum[4:]
        size = int(entry.get("filesize") or entry.get("size") or 0)
        if name:
            out[name] = (checksum, size)
    return out


def upload(bucket: str, path: Path, token: str) -> dict:
    """PUT one file into the deposition's bucket and return Zenodo's response."""
    url = f"{bucket}/{path.name}?access_token={token}"
    size = path.stat().st_size
    with path.open("rb") as fh:
        req = urllib.request.Request(url, data=fh, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(size))
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return json.load(resp)


def plan_uploads(
    paths: list[Path], have: dict[str, tuple[str, int]]
) -> list[tuple[Path, str, str]]:
    """``(path, md5, action)`` per file, in name order.

    An identical name+md5+size already on the deposition is a skip (that is what
    makes an interrupted run resumable); the same name with different bytes is
    called out as a REPLACE rather than quietly re-uploaded.
    """
    plan: list[tuple[Path, str, str]] = []
    for path in sorted(paths):
        digest = _md5(path)
        prior = have.get(path.name)
        if prior and prior == (digest, path.stat().st_size):
            action = "skip (identical)"
        elif prior:
            action = "REPLACE (same name, different bytes)"
        else:
            action = "upload"
        plan.append((path, digest, action))
    return plan


def transfer(plan: list[tuple[Path, str, str]], bucket: str, token: str) -> int:
    """Upload every non-skipped file, verifying Zenodo's md5 for each."""
    uploaded = 0
    for path, digest, action in plan:
        if action == "skip (identical)":
            continue
        print(
            f"uploading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...", flush=True
        )
        resp = upload(bucket, path, token)
        got = str(resp.get("checksum", ""))
        if got.startswith("md5:"):
            got = got[4:]
        if got != digest:
            raise SystemExit(
                f"CHECKSUM MISMATCH for {path.name}: local md5 {digest}, "
                f"Zenodo reported {got!r}. The deposition now holds a bad copy — "
                "delete that file on Zenodo before retrying."
            )
        uploaded += 1
        print(f"  ok, md5 verified ({digest})")
    return uploaded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposition", required=True, help="Draft deposition id.")
    ap.add_argument(
        "--files", nargs="+", required=True, type=Path, help="Files to add."
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Actually upload. Without this the run is a dry run.",
    )
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set.")

    paths = []
    for p in args.files:
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")
        paths.append(p)

    dep = fetch_deposition(args.deposition, token)
    bucket = dep["links"]["bucket"]
    have = existing_files(dep)
    print(f"  already on the deposition: {len(have)} files")

    plan = plan_uploads(paths, have)
    total_new = sum(p.stat().st_size for p, _, a in plan if a != "skip (identical)")
    print(f"\nplan ({len(plan)} files, {total_new / 1e9:.2f} GB to transfer):")
    for path, digest, action in plan:
        print(f"  {action:<38} {path.name}  {path.stat().st_size / 1e6:8.1f} MB")

    if not args.yes:
        print("\nDRY RUN — nothing uploaded. Re-run with --yes to transfer.")
        print("This tool cannot publish; do that by hand when you are ready.")
        return 0

    uploaded = transfer(plan, bucket, token)
    print(
        f"\n{uploaded} file(s) uploaded and verified; {len(plan) - uploaded} skipped."
    )
    print(
        "The deposition is STILL A DRAFT. Publishing is deliberately not "
        "something this tool can do."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
