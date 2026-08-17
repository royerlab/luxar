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
* **The token travels in an ``Authorization: Bearer`` header, never as an
  ``access_token`` query parameter.** A query string is recorded by server and proxy
  access logs, TLS-terminating middleboxes, and any tracing layer that formats
  ``HTTPError.url``; a header is not. The header is *unredirected*, so it is never
  replayed to a redirect target.

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
import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
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


def _get(url: str, token: str) -> Any:
    req = urllib.request.Request(url, method="GET")
    # UNREDIRECTED, deliberately: urllib's redirect handler copies everything in
    # ``Request.headers`` onto the follow-up request — including to whatever host
    # a ``Location`` names. An unredirected header is still sent on the initial
    # request but is not carried across a redirect, which then fails closed with a
    # 401 that the SystemExit handler below reports without a URL.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_deposition(dep_id: str, token: str) -> dict:
    """Fetch the deposition, refusing anything that is not an open draft."""
    try:
        dep = _get(f"{API}/{dep_id}", token)
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
    # The filename is one URL path segment, so it is percent-encoded: a space made
    # ``http.client`` reject the request line outright (``InvalidURL``), and a ``?``
    # or ``#`` truncated the key Zenodo stored the file under — which then never
    # matches ``existing_files`` again, so an interrupted run stopped being
    # resumable. Zenodo decodes the segment, so the key is still ``path.name``.
    url = f"{bucket}/{urllib.parse.quote(path.name, safe='')}"
    size = path.stat().st_size
    with path.open("rb") as fh:
        req = urllib.request.Request(url, data=fh, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(size))
        # Unredirected for the same reason as in ``_get``. urllib refuses to
        # redirect a PUT at all, so there is no follow-up request to leak to here
        # today; both call sites are written the same way so neither can drift.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return json.load(resp)
        except (OSError, http.client.HTTPException) as exc:
            # Deliberately no URL in either message: the whole point of sending
            # the credential as a header is that it never reaches a log or a
            # console. Only HTTPError carries ``code``, so that probe is what
            # splits the answered-with-a-status branch from the rest.
            #
            # The clause is wider than ``URLError`` on purpose. ``do_open``
            # wraps only the SEND phase in ``URLError``; ``h.getresponse()`` is
            # not wrapped, so a response-phase death arrives raw as
            # ``RemoteDisconnected``, ``TimeoutError``, ``BadStatusLine`` or —
            # from inside ``json.load`` — ``IncompleteRead``. That is precisely
            # the multi-GB case: Zenodo computes the server-side md5 only after
            # the whole body lands, so the long wait is on the RESPONSE, where
            # an idle load balancer or the 3600 s timeout hits. ``OSError``
            # covers URLError/TimeoutError/RemoteDisconnected and a local
            # read error on the body; ``HTTPException`` covers the rest.
            code = getattr(exc, "code", None)
            if code is not None:
                raise SystemExit(
                    f"Zenodo returned HTTP {code} {exc.reason} while uploading "
                    f"{path.name} to the deposition bucket. A 401/403 means the "
                    "credential was rejected on the bucket endpoint — check that "
                    "ZENODO_TOKEN has deposit:write scope."
                ) from exc
            raise SystemExit(
                f"Upload of {path.name} to the deposition bucket did not "
                f"complete: {getattr(exc, 'reason', None) or exc}. Nothing was "
                "verified, so re-run once the cause is cleared; files already "
                "uploaded intact are skipped."
            ) from exc


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

    # Stripped and then refused if a line break survives. A CRLF-terminated token
    # file (``ZENODO_TOKEN=$(cat token.txt)`` leaves the ``\r``) used to be
    # tolerated because ``urlsplit`` drops ASCII \t\r\n from a URL; as a header
    # value it instead makes ``http.client.putheader`` raise ``ValueError: Invalid
    # header value b'Bearer <token>'``, which no handler here catches — so the
    # credential would end up in a traceback. Neither message echoes the value.
    token = (os.environ.get("ZENODO_TOKEN") or "").strip()
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set.")
    if "\r" in token or "\n" in token:
        raise SystemExit(
            "ZENODO_TOKEN contains a line break inside the value. Check how it was "
            "captured (a wrapped paste or a concatenated file) and re-copy the token "
            "from Zenodo."
        )

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
