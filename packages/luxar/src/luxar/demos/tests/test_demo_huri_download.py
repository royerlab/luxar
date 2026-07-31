"""Tests for the bounded CORUM download in demo_huri_interactome.

These tests exercise ``_download_corum_zip_member`` with a fake ``requests.get``
so nothing touches the network. They verify the happy path, the two hard size
ceilings (compressed archive cap and extracted-member / decompression-bomb cap),
the zero-byte-member guard, and the multi-URL fallback contract — and that no
partial ``dest`` / ``.part`` / ``.zip.part`` files are left behind.
"""

from __future__ import annotations

import io
import zipfile

import pytest

# The demo imports pandas at module scope; pandas ships in the ``demos`` extra,
# not core — skip cleanly when it is absent (matches sibling demo tests).
pytest.importorskip("pandas")

from luxar.demos import demo_huri_interactome as demo  # noqa: E402


def _make_zip_bytes(member_name: str, member_data: bytes) -> bytes:
    """Build an in-memory zip archive holding one member."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, member_data)
    return buf.getvalue()


class _FakeResponse:
    """Minimal context-manager stand-in for a ``requests`` streamed response."""

    def __init__(self, body: bytes, content_length: int | None = None) -> None:
        self._body = body
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    @property
    def content(self) -> bytes:
        # Present so the OLD unbounded implementation (``blob = r.content``)
        # would succeed against this fake — the ceiling tests below would then
        # genuinely fail. Without this, the old code raised AttributeError and
        # the guards looked correct for the wrong reason.
        return self._body

    def iter_content(self, chunk_size: int = 1 << 20):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    content_length: int | None = -1,
) -> None:
    """Patch ``demo.requests.get`` to return one fixed body for any URL.

    ``content_length`` defaults to the real body length; pass ``None`` to omit
    the Content-Length header entirely.
    """
    cl = len(body) if content_length == -1 else content_length

    def fake_get(url: str, *args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(body, content_length=cl)

    monkeypatch.setattr(demo.requests, "get", fake_get)


def _patch_get_per_url(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, bytes | Exception]
) -> None:
    """Patch ``demo.requests.get`` to behave differently per URL.

    Map each URL to either a body (bytes) or an ``Exception`` to raise.
    """

    def fake_get(url: str, *args: object, **kwargs: object) -> _FakeResponse:
        outcome = responses[url]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome, content_length=len(outcome))

    monkeypatch.setattr(demo.requests, "get", fake_get)


def _no_temp_files_left(dest) -> bool:
    archive_tmp = dest.with_name(dest.name + ".zip.part")
    part = dest.with_name(dest.name + ".part")
    return not archive_tmp.exists() and not part.exists()


def test_happy_path_extracts_member(tmp_path, monkeypatch) -> None:
    member_data = b"complex_id\tname\n1\tribosome\n"
    body = _make_zip_bytes("allComplexes.txt", member_data)
    _patch_get(monkeypatch, body)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is True
    assert dest.exists()
    assert dest.read_bytes() == member_data
    assert _no_temp_files_left(dest)


def test_archive_ceiling_rejects_large_declared_body(tmp_path, monkeypatch) -> None:
    body = _make_zip_bytes("allComplexes.txt", b"x" * 4096)
    _patch_get(monkeypatch, body)
    # Force the compressed archive over the cap (Content-Length present → the
    # up-front declared check trips).
    monkeypatch.setattr(demo, "_CORUM_MAX_ARCHIVE_BYTES", 8)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is False
    assert not dest.exists()
    assert _no_temp_files_left(dest)


def test_archive_ceiling_streaming_without_content_length(
    tmp_path, monkeypatch
) -> None:
    # No Content-Length header → the up-front declared check is a no-op, so the
    # in-loop running-total ``written > cap`` raise must bound the body.
    body = _make_zip_bytes("allComplexes.txt", b"x" * 4096)
    _patch_get(monkeypatch, body, content_length=None)
    monkeypatch.setattr(demo, "_CORUM_MAX_ARCHIVE_BYTES", 8)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is False
    assert not dest.exists()
    assert _no_temp_files_left(dest)


def test_member_ceiling_guards_decompression_bomb(tmp_path, monkeypatch) -> None:
    # Highly compressible member: tiny zip, large uncompressed size.
    member_data = b"\0" * (2 * 1024 * 1024)
    body = _make_zip_bytes("allComplexes.txt", member_data)
    _patch_get(monkeypatch, body)
    # Keep the archive under its cap but the extracted member over its cap.
    monkeypatch.setattr(demo, "_CORUM_MAX_MEMBER_BYTES", 1024)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is False
    assert not dest.exists()
    assert _no_temp_files_left(dest)


def test_zero_byte_member_is_rejected(tmp_path, monkeypatch) -> None:
    body = _make_zip_bytes("allComplexes.txt", b"")
    _patch_get(monkeypatch, body)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is False
    assert not dest.exists()
    assert _no_temp_files_left(dest)


def test_missing_member_falls_through_to_next_url(tmp_path, monkeypatch) -> None:
    # A well-formed archive that simply lacks the member takes the
    # ``target is None`` branch (a plain ``continue``, not an exception), so the
    # next mirror must still be tried and the archive temp file cleaned up.
    member_data = b"complex_id\tname\n1\tribosome\n"
    url1 = "http://example.invalid/first.zip"
    url2 = "http://example.invalid/second.zip"
    _patch_get_per_url(
        monkeypatch,
        {
            url1: _make_zip_bytes("readme.txt", b"nothing useful here"),
            url2: _make_zip_bytes("allComplexes.txt", member_data),
        },
    )

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        (url1, url2),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is True
    assert dest.read_bytes() == member_data
    assert _no_temp_files_left(dest)


def test_missing_member_everywhere_returns_false(tmp_path, monkeypatch) -> None:
    body = _make_zip_bytes("readme.txt", b"nothing useful here")
    _patch_get(monkeypatch, body)

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        ("http://example.invalid/corum.zip",),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is False
    assert not dest.exists()
    assert _no_temp_files_left(dest)


def test_multi_url_fallback_second_succeeds(tmp_path, monkeypatch) -> None:
    member_data = b"complex_id\tname\n1\tribosome\n"
    good = _make_zip_bytes("allComplexes.txt", member_data)
    url1 = "http://example.invalid/first.zip"
    url2 = "http://example.invalid/second.zip"
    _patch_get_per_url(
        monkeypatch,
        {url1: RuntimeError("mirror down"), url2: good},
    )

    dest = tmp_path / "allComplexes.txt"
    ok = demo._download_corum_zip_member(
        (url1, url2),
        "allComplexes.txt",
        dest,
        "CORUM (test)",
    )

    assert ok is True
    assert dest.exists()
    assert dest.read_bytes() == member_data
    assert _no_temp_files_left(dest)
