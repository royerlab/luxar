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


def test_record_merges_variants_under_one_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "media-manifest.json"
    manifest_path.write_text(
        json.dumps({"base_url": "https://data.luxarviewer.dev/media", "tiles": {}})
    )
    monkeypatch.setattr(pub, "MANIFEST_PATH", manifest_path)
    pub.record("demo-clip", f"{'a' * 16}.webp", "a" * 64, 1)
    pub.record("demo-clip", f"{'b' * 16}.webm", "b" * 64, 2)
    assert set(json.loads(manifest_path.read_text())["assets"]["demo-clip"]) == {
        "webp",
        "webm",
    }


def test_superseded_objects_are_recorded_outside_the_bijection() -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "media-manifest.json").read_text()
    )
    live = {e["key"] for v in manifest["assets"].values() for e in v.values()}
    for obj in manifest["superseded"]["objects"]:
        assert obj["key"] not in live
        assert obj["replaced_by"] in live


def test_cli_dry_run_and_record_by_stem_parse_and_derive_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    png = tmp_path / "hero-drosophila.png"
    png.write_bytes(b"\x89PNG fake")
    assert pub.main(["--dry-run", str(png)]) == 0
    out = capsys.readouterr().out
    assert "hero-drosophila.png" in out and ".png" in out
    # --record-by-stem must parse and name the entry after the file stem; stub
    # the network and rclone calls so the test stays offline.
    manifest_path = tmp_path / "media-manifest.json"
    manifest_path.write_text(json.dumps({"base_url": pub.BASE_URL, "tiles": {}}))
    monkeypatch.setattr(pub, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(pub, "exists_remote", lambda key: True)
    monkeypatch.setattr(
        pub, "verify", lambda key, digest, size: f"{pub.BASE_URL}/{key}"
    )
    assert pub.main(["--record-by-stem", str(png)]) == 0
    assert "hero-drosophila" in json.loads(manifest_path.read_text())["assets"]
    with pytest.raises(SystemExit):
        pub.main(["--record", "x", "--record-by-stem", str(png)])
