"""Tests for the root-README gallery media verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

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
    assert len(entries) == 58
    assert len({entry.key for entry in entries}) == 58


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("key", "deadbeefdeadbeef.webp", "key must be"),
        ("bytes", 0, "positive integer"),
        ("content_type", "video/webm", "content_type must be image/webp"),
        ("sha256", "not-a-digest", "64 lowercase hex"),
    ],
)
def test_invalid_manifest_entry_is_rejected(
    field: str, value: object, message: str
) -> None:
    manifest = _manifest()
    manifest["tiles"]["atp_synthase"]["webp"][field] = value

    with pytest.raises(verify.VerificationError, match=message):
        verify.validated_entries(manifest, (REPO_ROOT / "README.md").read_text())


def test_readme_and_manifest_must_be_an_exact_bijection() -> None:
    manifest = _manifest()
    readme = (REPO_ROOT / "README.md").read_text()
    key = manifest["tiles"]["atp_synthase"]["webp"]["key"]

    with pytest.raises(verify.VerificationError, match="README media URLs differ"):
        verify.validated_entries(
            copy.deepcopy(manifest), readme.replace(key, "0" * 16 + ".webp")
        )
