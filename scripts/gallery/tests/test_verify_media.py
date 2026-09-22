"""Tests for the root-README gallery media verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "verify_media.py"
SPEC = importlib.util.spec_from_file_location("verify_media", SCRIPT)
assert SPEC and SPEC.loader
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)
REPO_ROOT = SCRIPT.parents[2]


def _manifest() -> dict:
    return json.loads((REPO_ROOT / "scripts/gallery/media-manifest.json").read_text())


def test_committed_manifest_is_consistent_with_readme() -> None:
    base_url, entries = verify.validated_entries(
        _manifest(), (REPO_ROOT / "README.md").read_text()
    )

    assert base_url == "https://data.luxarviewer.dev/media"
    assert len(entries) == 63  # 29 tiles x 2 variants + 5 README assets
    assert len({entry.key for entry in entries}) == 63


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (None, None, "entry must be an object"),
        ("key", "deadbeefdeadbeef.webp", "key must be"),
        ("bytes", 0, "positive integer"),
        ("bytes", True, "positive integer"),
        ("content_type", "video/webm", "content_type must be image/webp"),
        ("sha256", "not-a-digest", "64 lowercase hex"),
        ("sha256", "0" * 65, "64 lowercase hex"),
    ],
)
def test_invalid_manifest_entry_is_rejected(
    field: str | None, value: object, message: str
) -> None:
    manifest = _manifest()
    if field is None:
        manifest["tiles"]["atp_synthase"]["webp"] = value
    else:
        manifest["tiles"]["atp_synthase"]["webp"][field] = value

    with pytest.raises(verify.VerificationError, match=message):
        verify.validated_entries(manifest, (REPO_ROOT / "README.md").read_text())


@pytest.mark.parametrize("mismatch", ["missing", "duplicate", "extra", "wrong-origin"])
def test_readme_and_manifest_must_be_an_exact_bijection(mismatch: str) -> None:
    manifest = _manifest()
    readme = (REPO_ROOT / "README.md").read_text()
    base_url = manifest["base_url"]
    key = manifest["tiles"]["atp_synthase"]["webp"]["key"]
    url = f"{base_url}/{key}"
    if mismatch == "missing":
        readme = readme.replace(key, "0" * 16 + ".webp")
    elif mismatch == "duplicate":
        readme += f"\n{url}\n"
    elif mismatch == "extra":
        readme += f"\n{base_url}/{'f' * 16}.webp\n"
    else:
        readme = readme.replace(url, f"https://example.com/media/{key}")

    with pytest.raises(verify.VerificationError, match="README media URLs differ"):
        verify.validated_entries(copy.deepcopy(manifest), readme)


def _entry(variant: str) -> verify.MediaEntry:
    content_type = verify.CONTENT_TYPES[variant]
    return verify.MediaEntry(
        demo_id="demo",
        variant=variant,
        key=f"{'0' * 16}.{variant}",
        size_bytes=123,
        sha256="0" * 64,
        content_type=content_type,
    )


def _response(
    *,
    url: str = "https://data.luxarviewer.dev/media/object.webp",
    status: int = 200,
    content_length: str = "123",
    content_type: str = "image/webp",
) -> SimpleNamespace:
    headers = Message()
    headers["Content-Length"] = content_length
    headers["Content-Type"] = content_type
    return SimpleNamespace(url=url, status=status, headers=headers)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            _response(url="http://example.com/object.webp"),
            "redirected to a non-HTTPS URL",
        ),
        (_response(status=404), "HTTP 404"),
        (_response(content_length="122"), "Content-Length '122', expected 123"),
        (
            _response(content_type="image/png"),
            "Content-Type 'image/png', expected image/webp",
        ),
    ],
)
def test_response_metadata_rejects_mismatches(
    response: SimpleNamespace, message: str
) -> None:
    with pytest.raises(verify.VerificationError, match=message):
        verify._validate_response_metadata(response, _entry("webp"))


@pytest.mark.parametrize(
    ("variant", "content_type"), [("webp", "image/webp"), ("webm", "video/webm")]
)
def test_response_metadata_accepts_exact_match(variant: str, content_type: str) -> None:
    verify._validate_response_metadata(
        _response(
            url=f"https://data.luxarviewer.dev/media/object.{variant}",
            content_type=content_type,
        ),
        _entry(variant),
    )


def test_assets_section_accepts_png_but_tiles_still_need_both_variants() -> None:
    manifest = _manifest()
    readme = (REPO_ROOT / "README.md").read_text()
    banner = manifest["assets"]["social-preview"]["png"]
    assert banner["content_type"] == "image/png"
    assert banner["key"].endswith(".png") and banner["key"] in readme

    broken = copy.deepcopy(manifest)
    del broken["tiles"]["atp_synthase"]["webm"]
    with pytest.raises(verify.VerificationError, match="expected webp and webm"):
        verify.validated_entries(broken, readme)

    unknown = copy.deepcopy(manifest)
    unknown["assets"]["social-preview"] = {"gif": banner}
    with pytest.raises(verify.VerificationError, match="unknown media variant"):
        verify.validated_entries(unknown, readme)
