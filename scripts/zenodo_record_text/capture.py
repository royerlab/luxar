#!/usr/bin/env python3
"""Capture the live Zenodo record descriptions into this directory.

Read-only against Zenodo: capture mode GETs each deposition and writes what it
finds locally; check mode compares those snapshots with public records. There is
deliberately no code here that writes to Zenodo, and none that publishes.

Usage:
    ZENODO_TOKEN=... python3 scripts/zenodo_record_text/capture.py
    python3 scripts/zenodo_record_text/capture.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = (
    HERE.parents[1]
    / "packages"
    / "luxar"
    / "src"
    / "luxar"
    / "demos"
    / "data_manifest.json"
)
FIELDS = (
    "title",
    "license",
    "version",
    "language",
    "publication_date",
    "keywords",
    "creators",
    "contributors",
    "related_identifiers",
    "notes",
    "custom",
)


def records() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return manifest["records"]


def fetch(
    dep: int, token: str | None = None, *, published_record: bool = False
) -> dict[str, Any]:
    """GET one Zenodo object, retrying transient 5xx and rate limits."""
    endpoint = "records" if published_record else "deposit/depositions"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(
        f"https://zenodo.org/api/{endpoint}/{dep}",
        headers=headers,
    )
    last: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - transient network failures
            last = exc
        if attempt < 5:
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"zenodo unreachable for {dep}: {last}")


def check_snapshots() -> int:
    """Compare captured record state with Zenodo's public record endpoint."""
    index = json.loads((HERE / "records.json").read_text(encoding="utf-8"))
    drift = False
    for key, record in records().items():
        dep = int(record["zenodo_record"])
        published = fetch(dep, published_record=True)
        description = published["metadata"]["description"]
        expected = index.get(key, {})
        actual = {
            "description_sha256": hashlib.sha256(
                description.encode("utf-8")
            ).hexdigest(),
            "description_chars": len(description),
            "submitted": published.get("submitted"),
            "zenodo_modified": published.get("modified"),
        }
        mismatches = [
            field for field, value in actual.items() if expected.get(field) != value
        ]
        if not mismatches:
            print(f"{key:16s} OK")
            continue
        drift = True
        for field in mismatches:
            print(f"{key} {field}: {expected.get(field)!r} != {actual[field]!r}")
    return int(drift)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare snapshots with public Zenodo records without writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    args = _parse_args(argv)
    if args.check:
        return check_snapshots()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_TOKEN")
    captured: dict[str, tuple[bytes, dict[str, object]]] = {}
    index: dict[str, object] = {}
    for key, record in records().items():
        dep = int(record["zenodo_record"])
        deposition = fetch(dep, token)
        metadata = deposition["metadata"]
        description = metadata["description"]
        description_bytes = description.encode("utf-8")
        captured[key] = (
            description_bytes,
            {
                "deposition": str(dep),
                "doi": record["zenodo_doi"],
                **{field: metadata.get(field) for field in FIELDS},
                "submitted": bool(deposition.get("submitted")),
                "description_sha256": hashlib.sha256(description_bytes).hexdigest(),
                "description_chars": len(description),
                "zenodo_modified": deposition.get("modified"),
            },
        )
        print(
            f"{key:16s} {len(description):6,d} chars  submitted={deposition.get('submitted')}"
        )
    for key, (description_bytes, entry) in captured.items():
        (HERE / f"{key}.html").write_bytes(description_bytes)
        index[key] = entry
    index["_captured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "records.json").write_bytes(
        (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
