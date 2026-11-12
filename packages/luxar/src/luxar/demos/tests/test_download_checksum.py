"""Tests for ``verify_file_checksum``, in particular its ``verbose`` gate.

``ensure_dataset(verbose=False)`` re-hashes every warm cache hit, so the
per-file "Verifying…/Computing…/verified" block must be suppressible without
changing the boolean return contract (mismatch → False, match/no-hash → True).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from luxar.demos._support.downloads.download import verify_file_checksum

_PAYLOAD = b"luxar checksum test payload"
_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()
_BAD_SHA256 = "0" * 64


def _payload_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.bin"
    path.write_bytes(_PAYLOAD)
    return path


class TestVerifyFileChecksum:
    def test_match_returns_true_and_prints_by_default(self, tmp_path, capsys) -> None:
        path = _payload_file(tmp_path)
        assert verify_file_checksum(path, None, _SHA256) is True
        out = capsys.readouterr().out
        assert "Verifying" in out
        assert "SHA256 verified" in out

    def test_mismatch_returns_false_and_prints_by_default(
        self, tmp_path, capsys
    ) -> None:
        path = _payload_file(tmp_path)
        assert verify_file_checksum(path, None, _BAD_SHA256) is False
        out = capsys.readouterr().out
        assert "mismatch" in out

    def test_verbose_false_is_silent_on_match(self, tmp_path, capsys) -> None:
        path = _payload_file(tmp_path)
        assert verify_file_checksum(path, None, _SHA256, verbose=False) is True
        assert capsys.readouterr().out == ""

    def test_verbose_false_is_silent_on_mismatch(self, tmp_path, capsys) -> None:
        path = _payload_file(tmp_path)
        assert verify_file_checksum(path, None, _BAD_SHA256, verbose=False) is False
        assert capsys.readouterr().out == ""

    def test_verbose_false_is_silent_on_md5(self, tmp_path, capsys) -> None:
        path = _payload_file(tmp_path)
        good_md5 = hashlib.md5(_PAYLOAD, usedforsecurity=False).hexdigest()
        assert verify_file_checksum(path, good_md5, None, verbose=False) is True
        assert verify_file_checksum(path, "0" * 32, None, verbose=False) is False
        assert capsys.readouterr().out == ""

    def test_missing_file_returns_false(self, tmp_path, capsys) -> None:
        assert (
            verify_file_checksum(tmp_path / "nope.bin", None, _SHA256, verbose=False)
            is False
        )
        assert capsys.readouterr().out == ""


def test_is_download_failure_separates_the_network_from_faults() -> None:
    """A caller may route around the network failing, never a checksum or fault."""
    import requests

    from luxar.demos import is_download_failure
    from luxar.demos._support.downloads.download import RETRIES_EXHAUSTED

    assert is_download_failure(requests.exceptions.ConnectionError("offline"))
    assert is_download_failure(requests.exceptions.Timeout("slow"))
    assert is_download_failure(requests.exceptions.HTTPError("503"))
    assert is_download_failure(RuntimeError(RETRIES_EXHAUSTED))
    assert not is_download_failure(ValueError("failed checksum verification"))
    assert not is_download_failure(RuntimeError("the cache write is damaged"))
    assert not is_download_failure(FileNotFoundError("bad in-repo copy"))
