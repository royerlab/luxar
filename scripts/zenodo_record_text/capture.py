#!/usr/bin/env python3
"""Capture the live Zenodo record descriptions into this directory.

Read-only against Zenodo: GETs each deposition and writes what it finds. There
is deliberately no code here that writes to Zenodo, and none that publishes.

Usage:
    ZENODO_TOKEN=... python3 scripts/zenodo_record_text/capture.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
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


def fetch(dep: int, token: str) -> dict:
    """GET one deposition, retrying transient Zenodo 5xx and rate limits."""
    request = urllib.request.Request(
        f"https://zenodo.org/api/deposit/depositions/{dep}",
        headers={"Authorization": f"Bearer {token}"},
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


def main() -> int:
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
    raise SystemExit(main())
