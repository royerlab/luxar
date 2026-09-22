"""Offline tests for the content-addressed media publisher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "publish_media.py"
SPEC = importlib.util.spec_from_file_location("publish_media", SCRIPT)
assert SPEC and SPEC.loader
pub = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pub
SPEC.loader.exec_module(pub)


def test_key_is_the_sha256_prefix_plus_extension(tmp_path: Path) -> None:
    path = tmp_path / "banner.PNG"
    path.write_bytes(b"not really a png")
    digest = hashlib.sha256(b"not really a png").hexdigest()
    assert pub.sha256_of(path) == digest
    assert pub.key_for(path, digest) == f"{digest[:16]}.png"
    with pytest.raises(SystemExit, match="unsupported extension"):
        pub.key_for(tmp_path / "x.gif", digest)


def test_record_writes_an_assets_entry_the_verifier_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "media-manifest.json"
    manifest_path.write_text(
        json.dumps({"base_url": "https://data.luxarviewer.dev/media", "tiles": {}})
    )
    monkeypatch.setattr(pub, "MANIFEST_PATH", manifest_path)
    digest = "0" * 64
    pub.record("social-preview", f"{digest[:16]}.png", digest, 123)
    written = json.loads(manifest_path.read_text())
    assert written["assets"] == {
        "social-preview": {
            "png": {
                "bytes": 123,
                "content_type": "image/png",
                "key": f"{digest[:16]}.png",
                "sha256": digest,
            }
        }
    }
    # Re-recording the same name replaces rather than duplicates.
    pub.record("social-preview", f"{'1' * 16}.png", "1" * 64, 456)
    assert (
        list(
            json.loads(manifest_path.read_text())["assets"]["social-preview"][
                "png"
            ].values()
        )[0]
        == 456
    )


def test_committed_assets_are_hash_named_and_match_their_digest() -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "media-manifest.json").read_text()
    )
    for name, variants in manifest["assets"].items():
        for variant, entry in variants.items():
            assert entry["key"] == f"{entry['sha256'][:16]}.{variant}", name
            assert entry["content_type"] == pub.CONTENT_TYPES[variant], name
            assert entry["bytes"] > 0, name
