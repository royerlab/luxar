#!/usr/bin/env python3
"""Validate root-README gallery media metadata and optionally fetch every object."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "scripts/gallery/media-manifest.json"
README_PATH = REPO_ROOT / "README.md"
TIMEOUT_SECONDS = 60
USER_AGENT = "LuxarGalleryMediaAudit/1.0"
CONTENT_TYPES = {"webm": "video/webm", "webp": "image/webp", "png": "image/png"}
TILE_VARIANTS = {"webm", "webp"}  # every gallery tile ships a still and an orbit video
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class VerificationError(RuntimeError):
    """Gallery media metadata or hosted bytes failed verification."""


@dataclass(frozen=True)
class MediaEntry:
    demo_id: str
    variant: str
    key: str
    size_bytes: int
    sha256: str
    content_type: str


def _validated_entry(demo_id: str, variant: str, raw_entry: object) -> MediaEntry:
    if not isinstance(raw_entry, dict):
        raise VerificationError(f"{demo_id}/{variant}: entry must be an object")
    key = raw_entry.get("key")
    digest = raw_entry.get("sha256")
    size_bytes = raw_entry.get("bytes")
    content_type = raw_entry.get("content_type")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise VerificationError(f"{demo_id}/{variant}: sha256 must be 64 lowercase hex")
    expected_key = f"{digest[:16]}.{variant}"
    if key != expected_key:
        raise VerificationError(
            f"{demo_id}/{variant}: key must be {expected_key}, got {key!r}"
        )
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
    ):
        raise VerificationError(
            f"{demo_id}/{variant}: bytes must be a positive integer"
        )
    if content_type != CONTENT_TYPES[variant]:
        raise VerificationError(
            f"{demo_id}/{variant}: content_type must be {CONTENT_TYPES[variant]}"
        )
    return MediaEntry(demo_id, variant, key, size_bytes, digest, content_type)


def _require_readme_bijection(
    base_url: str, entries: list[MediaEntry], readme: str
) -> None:
    manifest_keys = Counter(entry.key for entry in entries)
    url_pattern = re.compile(
        rf"{re.escape(base_url)}/([0-9a-f]{{16}}\.(?:webp|webm|png))\b"
    )
    readme_keys = Counter(url_pattern.findall(readme))
    if readme_keys != manifest_keys:
        missing = sorted((manifest_keys - readme_keys).elements())
        extra = sorted((readme_keys - manifest_keys).elements())
        raise VerificationError(
            f"README media URLs differ from manifest; missing={missing}, extra={extra}"
        )


def _tile_entries(tiles: object) -> list[MediaEntry]:
    if not isinstance(tiles, dict) or not tiles:
        raise VerificationError("media manifest tiles must be a non-empty object")
    entries: list[MediaEntry] = []
    for demo_id, variants in tiles.items():
        if not isinstance(demo_id, str) or not isinstance(variants, dict):
            raise VerificationError("media manifest tiles must map demo ids to objects")
        if set(variants) != TILE_VARIANTS:
            raise VerificationError(f"{demo_id}: expected webp and webm variants")
        for variant, raw_entry in variants.items():
            entries.append(_validated_entry(demo_id, variant, raw_entry))
    return entries


def _asset_entries(assets: object) -> list[MediaEntry]:
    """README media that is not a gallery tile (banner, diagrams, recordings)."""
    if not isinstance(assets, dict):
        raise VerificationError("media manifest assets must be an object")
    entries: list[MediaEntry] = []
    for name, variants in assets.items():
        if not isinstance(name, str) or not isinstance(variants, dict) or not variants:
            raise VerificationError(
                "media manifest assets must map names to non-empty objects"
            )
        if not set(variants) <= set(CONTENT_TYPES):
            raise VerificationError(
                f"{name}: unknown media variant in {sorted(variants)}"
            )
        for variant, raw_entry in variants.items():
            entries.append(_validated_entry(name, variant, raw_entry))
    return entries


def validated_entries(
    manifest: dict[str, Any], readme: str
) -> tuple[str, list[MediaEntry]]:
    """Return validated media entries and require an exact README URL bijection."""
    base_url = manifest.get("base_url")
    if not isinstance(base_url, str) or urlsplit(base_url).scheme != "https":
        raise VerificationError("media manifest base_url must be an HTTPS URL")
    base_url = base_url.rstrip("/")
    entries = _tile_entries(manifest.get("tiles"))
    entries += _asset_entries(manifest.get("assets", {}))
    _require_readme_bijection(base_url, entries, readme)
    return base_url, entries


def _validate_response_metadata(response: Any, entry: MediaEntry) -> None:
    if urlsplit(response.url).scheme != "https":
        raise VerificationError("redirected to a non-HTTPS URL")
    if response.status != 200:
        raise VerificationError(f"HTTP {response.status}")
    content_length = response.headers.get("Content-Length")
    if content_length != str(entry.size_bytes):
        raise VerificationError(
            f"Content-Length {content_length!r}, expected {entry.size_bytes}"
        )
    content_type = response.headers.get_content_type()
    if content_type != entry.content_type:
        raise VerificationError(
            f"Content-Type {content_type!r}, expected {entry.content_type}"
        )


def _response_digest(response: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    while chunk := response.read(1024 * 1024):
        digest.update(chunk)
        size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def verify_hosted_entry(base_url: str, entry: MediaEntry) -> None:
    """Fetch one hosted object and compare status, headers, length, and digest."""
    request = Request(f"{base_url}/{entry.key}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # nosec B310
            _validate_response_metadata(response, entry)
            size_bytes, digest = _response_digest(response)
    except HTTPError as exc:
        raise VerificationError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise VerificationError(str(exc.reason)) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise VerificationError(str(exc)) from exc
    if size_bytes != entry.size_bytes:
        raise VerificationError(f"read {size_bytes} bytes, expected {entry.size_bytes}")
    if digest != entry.sha256:
        raise VerificationError("SHA-256 mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("verify_media.py takes no arguments")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
        base_url, entries = validated_entries(manifest, README_PATH.read_text())
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for entry in entries:
        try:
            verify_hosted_entry(base_url, entry)
        except VerificationError as exc:
            failures += 1
            print(f"FAIL {entry.demo_id}/{entry.variant}: {exc}")
        else:
            print(f"PASS {entry.demo_id}/{entry.variant}: {entry.key}")
    print(f"Gallery media: {len(entries) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
