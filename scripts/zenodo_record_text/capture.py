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
import urllib.request

RECORDS = {
    "cc-by": 21912280,
    "cc-by-sa": 21912282,
    "h2afva": 21912284,
    "droso-timelapse": 22118695,
}
HERE = pathlib.Path(__file__).resolve().parent
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
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"zenodo unreachable for {dep}: {last}")


def main() -> int:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_TOKEN")
    index: dict[str, object] = {}
    for key, dep in RECORDS.items():
        deposition = fetch(dep, token)
        metadata = deposition["metadata"]
        description = metadata["description"]
        (HERE / f"{key}.html").write_text(description)
        index[key] = {
            "deposition": str(dep),
            "doi": f"10.5281/zenodo.{dep}",
            **{field: metadata.get(field) for field in FIELDS},
            "submitted": bool(deposition.get("submitted")),
            "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
            "description_chars": len(description),
            "zenodo_modified": deposition.get("modified"),
        }
        print(
            f"{key:16s} {len(description):6,d} chars  submitted={deposition.get('submitted')}"
        )
    index["_captured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "records.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
