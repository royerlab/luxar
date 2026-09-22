#!/usr/bin/env python3
"""Publish README / gallery media to ``data.luxarviewer.dev/media`` by content hash.

Every object under ``media/`` is named ``<sha256 of the bytes>[:16].<ext>``, so
a file is uploaded once, never overwritten (the edge cache ignores origin
cache-control, so replacing bytes under an existing key would serve stale
content for hours), and a re-cut gets a new URL automatically. This script:

1. hashes each file and derives its key,
2. skips files whose key already exists in the bucket,
3. uploads the rest with rclone (remote ``r2:luxar-demos``, see the Demo Site
   Runbook for credentials),
4. fetches every URL back over HTTPS and checks content type, byte count and
   hash, then prints the URL to paste into the README.

    hatch run python scripts/gallery/publish_media.py out/*.png out/*.webp
    hatch run python scripts/gallery/publish_media.py --dry-run out/*.png   # keys and URLs only
    hatch run python scripts/gallery/publish_media.py --record social-preview out/banner.png

``--record NAME`` also writes the entry into ``scripts/gallery/media-manifest.json``
under ``assets/NAME`` (one file at a time), which ``verify_media.py`` requires:
every README media URL must be listed there, and vice versa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess  # nosec B404: shells out to rclone with a fixed argv
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REMOTE = "r2:luxar-demos/media"
MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/gallery/media-manifest.json"
)
BASE_URL = "https://data.luxarviewer.dev/media"
CONTENT_TYPES = {
    "png": "image/png",
    "webp": "image/webp",
    "webm": "video/webm",
    "jpg": "image/jpeg",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key_for(path: Path, digest: str) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext not in CONTENT_TYPES:
        raise SystemExit(f"{path}: unsupported extension .{ext}")
    return f"{digest[:16]}.{ext}"


def exists_remote(key: str) -> bool:
    result = subprocess.run(  # nosec B603, B607: fixed argv, tool from PATH
        ["rclone", "lsf", f"{REMOTE}/{key}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == key


def upload(path: Path, key: str) -> None:
    content_type = CONTENT_TYPES[key.rsplit(".", 1)[1]]
    subprocess.run(  # nosec B603, B607: fixed argv, tool from PATH
        [
            "rclone",
            "copyto",
            "--header-upload",
            f"Content-Type: {content_type}",
            str(path),
            f"{REMOTE}/{key}",
        ],
        check=True,
    )


def verify(key: str, digest: str, size: int) -> str:
    url = f"{BASE_URL}/{key}"
    request = Request(
        url,
        headers={"User-Agent": "LuxarPublishMedia/1.0", "Cache-Control": "no-cache"},
    )
    with urlopen(request, timeout=120) as response:  # nosec B310 - https URL built from BASE_URL
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    expected_type = CONTENT_TYPES[key.rsplit(".", 1)[1]]
    problems = []
    if not content_type.startswith(expected_type):
        problems.append(f"content-type {content_type!r} != {expected_type!r}")
    if len(body) != size:
        problems.append(f"{len(body)} bytes served, {size} local")
    if hashlib.sha256(body).hexdigest() != digest:
        problems.append("served bytes differ from the local file")
    if problems:
        raise SystemExit(f"{url}: " + "; ".join(problems))
    return url


def record(name: str, key: str, digest: str, size: int) -> None:
    """Upsert a README media entry into the media manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    variant = key.rsplit(".", 1)[1]
    manifest.setdefault("assets", {})[name] = {
        variant: {
            "bytes": size,
            "content_type": CONTENT_TYPES[variant],
            "key": key,
            "sha256": digest,
        }
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"recorded readme/{name} -> {key} in {MANIFEST_PATH.relative_to(MANIFEST_PATH.parents[2])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="print keys and URLs, upload nothing"
    )
    parser.add_argument(
        "--record",
        metavar="NAME",
        help="also write the entry to media-manifest.json under assets/NAME (single file)",
    )
    args = parser.parse_args(argv)
    if args.record and len(args.files) != 1:
        parser.error("--record takes exactly one file")

    if not args.dry_run and shutil.which("rclone") is None:
        raise SystemExit(
            "rclone is not on PATH (needed to upload; use --dry-run to only print keys)"
        )
    for path in args.files:
        digest = sha256_of(path)
        key = key_for(path, digest)
        size = path.stat().st_size
        if args.dry_run:
            print(f"{path.name:32s} -> {BASE_URL}/{key}  ({size:,} B)")
            continue
        if exists_remote(key):
            state = "already hosted"
        else:
            upload(path, key)
            state = "uploaded"
        url = verify(key, digest, size)
        print(f"{path.name:32s} {state:15s} {url}  ({size:,} B, verified)")
        if args.record:
            record(args.record, key, digest, size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
