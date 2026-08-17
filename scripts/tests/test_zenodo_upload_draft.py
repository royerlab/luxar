"""Tests for the Zenodo draft uploader's safety properties.

No network: ``_get`` is stubbed. What matters here is the REFUSALS — this is the
only tool in the repo that mutates a Zenodo record, and the guarantees it claims
(draft only, never publish) have to be enforced rather than merely documented.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "zenodo_upload_draft.py"


def _load():
    spec = importlib.util.spec_from_file_location("_zen_upload", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_zen_upload"] = module
    spec.loader.exec_module(module)
    return module


_up = _load()


def _draft(**over) -> dict:
    base = {
        "submitted": False,
        "state": "unsubmitted",
        "metadata": {"title": "Luxar demo datasets"},
        "links": {"bucket": "https://zenodo.org/api/files/abc"},
        "files": [],
    }
    base.update(over)
    return base


class TestDraftOnlyGuard:
    def test_an_open_draft_is_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(_up, "_get", lambda url: _draft())
        dep = _up.fetch_deposition("1", "tok")
        assert dep["links"]["bucket"].endswith("abc")

    def test_a_submitted_deposition_is_refused(self, monkeypatch) -> None:
        """A published record's files are immutable — never try to mutate one."""
        monkeypatch.setattr(
            _up, "_get", lambda url: _draft(submitted=True, state="done")
        )
        with pytest.raises(SystemExit, match="REFUSING"):
            _up.fetch_deposition("1", "tok")

    @pytest.mark.parametrize("state", ["done", "inprogress", "error", "published"])
    def test_any_state_other_than_unsubmitted_is_refused(
        self, monkeypatch, state: str
    ) -> None:
        monkeypatch.setattr(_up, "_get", lambda url: _draft(state=state))
        with pytest.raises(SystemExit, match="REFUSING"):
            _up.fetch_deposition("1", "tok")

    def test_a_deposition_without_a_bucket_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(_up, "_get", lambda url: _draft(links={}))
        with pytest.raises(SystemExit, match="no bucket link"):
            _up.fetch_deposition("1", "tok")


class TestNoPublishPath:
    def test_the_source_contains_no_publish_call(self) -> None:
        """The strongest form of "cannot publish": the call is not in the file.

        Zenodo publishes via POST to `<deposition>/actions/publish`; if that path
        ever appears here, this tool stopped being safe by construction.
        """
        source = _SCRIPT.read_text()
        assert "actions/publish" not in source
        assert "/publish" not in source

    def test_only_get_and_put_are_used(self) -> None:
        """PUT adds a file to the bucket; POST would be an action like publish."""
        source = _SCRIPT.read_text()
        assert 'method="PUT"' in source
        assert 'method="POST"' not in source


class TestIdempotence:
    def test_identical_file_is_recognised_by_md5_and_size(self, tmp_path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        digest = _up._md5(f)

        dep = _draft(
            files=[
                {
                    "filename": "a.bin",
                    "checksum": f"md5:{digest}",
                    "filesize": f.stat().st_size,
                }
            ]
        )
        have = _up.existing_files(dep)
        assert have["a.bin"] == (digest, f.stat().st_size)

    def test_checksum_prefix_is_optional(self) -> None:
        """Zenodo has returned both `md5:<hex>` and bare `<hex>` over time."""
        dep = _draft(files=[{"filename": "a", "checksum": "deadbeef", "filesize": 3}])
        assert _up.existing_files(dep)["a"] == ("deadbeef", 3)

    def test_key_and_size_spellings_are_accepted(self) -> None:
        """The bucket API spells them `key`/`size`, the deposition API differently."""
        dep = _draft(files=[{"key": "b", "checksum": "md5:abc", "size": 7}])
        assert _up.existing_files(dep)["b"] == ("abc", 7)

    def test_the_plan_skips_identical_and_flags_changed_bytes(self, tmp_path) -> None:
        """The resume policy: same bytes = skip, same name + new bytes = REPLACE."""
        same = tmp_path / "same.bin"
        same.write_bytes(b"identical")
        changed = tmp_path / "changed.bin"
        changed.write_bytes(b"new bytes")
        fresh = tmp_path / "fresh.bin"
        fresh.write_bytes(b"never seen")

        have = {
            "same.bin": (_up._md5(same), same.stat().st_size),
            "changed.bin": ("0" * 32, 1),
        }
        actions = {
            path.name: action
            for path, _, action in _up.plan_uploads([fresh, changed, same], have)
        }
        assert actions == {
            "same.bin": "skip (identical)",
            "changed.bin": "REPLACE (same name, different bytes)",
            "fresh.bin": "upload",
        }

    def test_a_matching_md5_with_a_different_size_is_not_skipped(
        self, tmp_path
    ) -> None:
        """Both halves are checked — a truncated remote copy must be re-sent."""
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello luxar")
        have = {"a.bin": (_up._md5(f), f.stat().st_size + 1)}
        ((_, _, action),) = _up.plan_uploads([f], have)
        assert action == "REPLACE (same name, different bytes)"

    def test_changed_bytes_under_the_same_name_are_not_silently_skipped(
        self, tmp_path
    ) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"new content")
        dep = _draft(
            files=[{"filename": "a.bin", "checksum": "md5:0" * 1, "filesize": 999}]
        )
        have = _up.existing_files(dep)
        assert have["a.bin"] != (_up._md5(f), f.stat().st_size)
