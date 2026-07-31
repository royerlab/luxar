"""Regression tests for ``robust_download`` resume / HTTP 416 handling (issue #731).

A stdlib ``http.server`` subclass serves a real file with RFC 9110/7233 Range
semantics: an open-ended ``Range: bytes=<start>-`` whose ``start`` is at or past
EOF is *unsatisfiable* and answered with **416** (as real servers do), otherwise
206 (partial) or 200 (full). HEAD and unranged GET both report a correct
``Content-Length`` and advertise ``Accept-Ranges: bytes``. No network access.

The core regression (``test_complete_file_survives_redownload``) must FAIL
against the pre-fix code, which resumes at EOF, gets 416, and *deletes* the
already-complete cached file.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import io
import threading
from pathlib import Path

import numpy as np
import pytest
import requests

from luxar.utils.download import robust_download


class _RangeHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with RFC-correct open-ended Range + 416 support."""

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not path.is_file():
            return super().send_head()

        size = path.stat().st_size

        if range_header is None:
            # Unranged GET / HEAD: full body, correct Content-Length.
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if self.command == "HEAD":
                f.close()
                return None
            return _LimitedFile(f, size)

        spec = range_header.replace("bytes=", "").strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)

        # A Range at/after EOF is unsatisfiable -> 416 (the exact bug trigger).
        if start >= size or start > end:
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _LimitedFile(f, end - start + 1)


class _ChunkedRangeHTTPHandler(_RangeHTTPHandler):
    """Range handler that OMITS ``Content-Length`` on HEAD and unranged GET
    (chunked/dynamic style) but still answers a Range at/after EOF with **416**.

    This forces ``robust_download`` to resolve the remote size as *unknown*
    (``None``), issue the Range anyway, and hit the in-loop 416 handler — the
    path the size-probing tests never reach. The server records every status it
    serves in ``self.server.served`` so a test can assert no 206/200 *body*
    download happened.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not path.is_file():
            return super().send_head()

        size = path.stat().st_size

        if range_header is None:
            # Chunked style: 200 with NO Content-Length header.
            self.server.served.append(200)  # type: ignore[attr-defined]
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if self.command == "HEAD":
                f.close()
                return None
            return _LimitedFile(f, size)

        spec = range_header.replace("bytes=", "").strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)

        if start >= size or start > end:
            self.server.served.append(416)  # type: ignore[attr-defined]
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.server.served.append(206)  # type: ignore[attr-defined]
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _LimitedFile(f, end - start + 1)


class _NoContentRangeChunkedHandler(_ChunkedRangeHTTPHandler):
    """Chunked handler whose 416 carries NO ``Content-Range`` header at all.

    With neither a Content-Length (chunked) nor a Content-Range on the 416,
    ``robust_download`` has NO size signal whatsoever — exercising the
    conservative "the file is at least complete, return it" fallback.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not path.is_file():
            return super().send_head()

        size = path.stat().st_size

        if range_header is None:
            self.server.served.append(200)  # type: ignore[attr-defined]
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if self.command == "HEAD":
                f.close()
                return None
            return _LimitedFile(f, size)

        spec = range_header.replace("bytes=", "").strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)

        if start >= size or start > end:
            # 416 with NO Content-Range (and no Content-Length total).
            self.server.served.append(416)  # type: ignore[attr-defined]
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.server.served.append(206)  # type: ignore[attr-defined]
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _LimitedFile(f, end - start + 1)


def _etag_of(path: Path) -> str:
    """The strong ETag the ETag-aware test server derives from file content."""
    return '"' + hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() + '"'


class _EtagRangeHTTPHandler(_RangeHTTPHandler):
    """Range handler with a strong content-derived ETag and ``If-Range`` semantics.

    Sends ``ETag`` on every response. A ranged request carrying an ``If-Range``
    that does NOT match the current ETag is answered with a 200 FULL body
    (RFC 9110 §13.1.5 — the representation changed, so the tail must not be
    served), exactly as validating servers (Zenodo, GitHub, S3) behave. Every
    served status is recorded in ``self.server.served``.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not path.is_file():
            return super().send_head()

        size = path.stat().st_size
        etag = _etag_of(path)

        if_range = self.headers.get("If-Range")
        if range_header is not None and if_range is not None and if_range != etag:
            range_header = None  # validator mismatch → full body, ignore Range

        if range_header is None:
            self.server.served.append(200)  # type: ignore[attr-defined]
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.end_headers()
            if self.command == "HEAD":
                f.close()
                return None
            return _LimitedFile(f, size)

        spec = range_header.replace("bytes=", "").strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)

        if start >= size or start > end:
            self.server.served.append(416)  # type: ignore[attr-defined]
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.server.served.append(206)  # type: ignore[attr-defined]
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.end_headers()
        return _LimitedFile(f, end - start + 1)


class _MisalignedResumeHandler(_RangeHTTPHandler):
    """Range handler that ACCEPTS a resume but answers from byte 0.

    A misbehaving server: it returns 206 for any Range request, yet the body
    (and ``Content-Range``) start at 0 rather than the requested offset. Blindly
    appending such a body would corrupt the file at the wrong offset while
    still passing size verification.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")

        if not path.is_file() or range_header is None:
            return super().send_head()

        size = path.stat().st_size
        f = open(path, "rb")
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Range", f"bytes 0-{size - 1}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if self.command == "HEAD":
            f.close()
            return None
        return _LimitedFile(f, size)


class _StarTotalContentRangeHandler(_RangeHTTPHandler):
    """Unranged GET carries a malformed ``Content-Range: .../*`` and NO
    ``Content-Length`` — so the in-loop size parse sees a star total.

    Before the parse was hardened, ``int("*")`` raised ValueError into the
    generic handler and aborted the whole download non-retryably; the parse now
    degrades to "unknown size" and the body still downloads to completion.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        if not path.is_file() or self.headers.get("Range") is not None:
            return super().send_head()
        size = path.stat().st_size
        f = open(path, "rb")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes 0-{size - 1}/*")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if self.command == "HEAD":
            f.close()
            return None
        return _LimitedFile(f, size)


class _GzipRangeHTTPHandler(_RangeHTTPHandler):
    """Host that gzip-encodes the body and reports the COMPRESSED Content-Length
    UNLESS the client asks for ``Accept-Encoding: identity``.

    Mimics a gzip-capable static host. ``robust_download``'s size probe sets
    ``Accept-Encoding: identity`` so the reported Content-Length is the true
    DECODED size that matches the on-disk cache; without that guard the probe
    would (via requests' default ``gzip``) see the smaller COMPRESSED length and
    mistake a complete cache for one that needs re-downloading. Records
    ``(command, used_gzip)`` per request in ``self.server.served``.
    """

    def send_head(self):  # noqa: ANN201 - stdlib override
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()

        accept = (self.headers.get("Accept-Encoding") or "").lower()
        use_gzip = "gzip" in accept  # requests' default; the identity guard opts out
        raw = path.read_bytes()
        self.server.served.append((self.command, use_gzip))  # type: ignore[attr-defined]

        body = gzip.compress(raw) if use_gzip else raw
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if self.command == "HEAD":
            return None
        return _LimitedFile(io.BytesIO(body), len(body))


class _LimitedFile:
    """File-like that stops after ``limit`` bytes (for copyfile)."""

    def __init__(self, f, limit: int) -> None:  # noqa: ANN001
        self._f = f
        self._remaining = limit

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = self._remaining if n < 0 else min(n, self._remaining)
        data = self._f.read(n)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._f.close()


def _serve(directory: Path, handler_cls=_RangeHTTPHandler):  # noqa: ANN001, ANN202
    handler = lambda *a, **kw: handler_cls(  # noqa: E731
        *a, directory=str(directory), **kw
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.served = []  # type: ignore[attr-defined]  # status codes served
    # The mid-body-drop tests deliberately reset the connection client-side, so
    # the server thread's ConnectionResetError traceback is expected noise —
    # silence it to keep test output clean.
    server.handle_error = lambda request, client_address: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def range_server(tmp_path: Path):
    """Serve tmp_path over HTTP with RFC-correct Range + 416; yields base URL."""
    server, thread = _serve(tmp_path)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture()
def chunked_range_server(tmp_path: Path):
    """Serve tmp_path chunked-style (no Content-Length) but still 416 at EOF.

    Yields ``(base_url, server)`` so a test can inspect ``server.served``.
    """
    server, thread = _serve(tmp_path, _ChunkedRangeHTTPHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture()
def headerless_416_server(tmp_path: Path):
    """Serve chunked-style with a 416 that carries NO Content-Range at all.

    Yields ``(base_url, server)`` so a test can inspect ``server.served``.
    """
    server, thread = _serve(tmp_path, _NoContentRangeChunkedHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture()
def etag_range_server(tmp_path: Path):
    """Serve tmp_path with a strong ETag + ``If-Range`` validation.

    Yields ``(base_url, server)`` so a test can inspect ``server.served``.
    """
    server, thread = _serve(tmp_path, _EtagRangeHTTPHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture()
def misaligned_resume_server(tmp_path: Path):
    """Serve tmp_path with a 206 whose body always starts at byte 0."""
    server, thread = _serve(tmp_path, _MisalignedResumeHandler)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _make_payload(nbytes: int = 300_000, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, nbytes).astype(np.uint8).tobytes()


class TestRobustDownloadResume416:
    def test_complete_file_survives_redownload(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """CORE REGRESSION (issue #731): a fully-cached file re-downloaded with
        ``expected_size=None`` must be returned and left byte-identical on disk.

        Pre-fix, ``robust_download`` resumes at EOF -> server 416 -> HTTPError
        -> the complete cached file is UNLINKED. This test fails against the
        unpatched code. The cache has the SAME LENGTH as the remote but
        DIFFERENT content, so a silent full re-download would be caught too.
        """
        remote = _make_payload(seed=7)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Same length as remote (so the size-probe detects "complete") but
        # different bytes — proves the file is returned untouched, not refetched.
        cache = _make_payload(nbytes=len(remote), seed=13)
        assert cache != remote
        out.write_bytes(cache)

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists(), "complete cached file must NOT be deleted"
        assert out.read_bytes() == cache, "cache must NOT be re-downloaded"

    def test_partial_file_resumes_and_completes(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A genuinely partial file resumes via 206 (NOT a silent full restart).

        Under the ``.part`` staging contract the in-progress bytes live in a
        sibling ``<out>.part`` file, so the partial prefix is pre-seeded there
        (never at ``out``, which by contract means "complete"). The prefix bytes
        DIFFER from the true remote prefix, so the only way the promoted ``out``
        can equal ``local_prefix + remote[N:]`` is if bytes ``[0, N)`` were kept
        (never re-fetched) — proving a real 206 resume onto the ``.part`` file.

        This pins the NO-VALIDATOR best-effort path: neither a recorded
        ``.part.validator`` sidecar nor a server ETag exists, so ``If-Range``
        cannot be sent and the staged bytes are trusted as-is (the best that can
        be done without a validator). A partial staged from a VALIDATING host
        records its ETag and is protected against a changed remote — see
        ``test_changed_remote_invalidates_recorded_validator``.
        """
        n = 120_000
        payload = _make_payload(nbytes=500_000, seed=1)
        (tmp_path / "data.bin").write_bytes(payload)

        # A DIFFERENT prefix of the same length as the true remote's [0, N).
        local_prefix = _make_payload(nbytes=n, seed=424242)
        assert local_prefix != payload[:n]

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        part.write_bytes(local_prefix)  # partial staged, distinguishable bytes

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert not part.exists(), "staged .part must be renamed onto out"
        # Bytes [0, N) are the LOCAL ones (never re-downloaded); tail is remote.
        assert out.read_bytes() == local_prefix + payload[n:]

    def test_local_larger_than_remote_refetches_clean(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A local file LARGER than the remote (re-uploaded, smaller asset)
        must re-fetch cleanly and end byte-identical to the current remote,
        rather than 416-and-fail or be deleted."""
        payload = _make_payload(nbytes=200_000, seed=2)  # current (smaller) remote
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Stale, larger local copy (e.g. an older, bigger version of the asset).
        out.write_bytes(_make_payload(nbytes=350_000, seed=99))

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == payload

    def test_non_416_error_preserves_preexisting_file(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A non-416 HTTP error (404) on a URL whose destination pre-existed
        must leave the pre-existing file intact.

        Under ``.part`` staging the pre-existing ``out`` is MIGRATED into the
        staging path before the probe; the 404 is raised at ``raise_for_status``
        before any body byte is written, so the migrated staging file is still
        byte-identical and the terminal-failure RESTORE moves it back onto
        ``out`` untouched (no leftover ``.part``)."""
        sentinel = b"do-not-delete-me" * 1000

        out = tmp_path / "cache" / "missing.bin"
        out.parent.mkdir(parents=True)
        out.write_bytes(sentinel)  # pre-existing cache under a 404 URL
        part = out.with_name(out.name + ".part")

        with pytest.raises(requests.exceptions.HTTPError):
            robust_download(
                f"{range_server}/does-not-exist.bin",
                out,
                expected_size=None,
                verify_size=False,
                max_retries=0,
            )

        assert out.exists(), "pre-existing cache must survive a non-416 error"
        assert out.read_bytes() == sentinel
        assert not part.exists(), "the migrated cache must be restored, not orphaned"

    def test_complete_cache_returned_via_in_loop_416(
        self, chunked_range_server, tmp_path: Path
    ) -> None:
        """Exercise the IN-LOOP 416 handler (not the size-probe fast path).

        The chunked server omits Content-Length on HEAD/GET, so the remote size
        resolves as UNKNOWN — ``robust_download`` cannot early-detect "complete"
        and issues a Range at EOF for the staged ``.part`` file, which the server
        answers with 416. The handler must promote the complete staged file onto
        ``out`` (atomic rename) WITHOUT truncating or re-downloading it. The
        staged bytes DIFFER from the remote (same length) so a spurious
        re-download is caught by content; the server also records that it served
        ZERO 206 (resume) responses. (A 200 is expected from the unranged size
        probe, so it is not asserted against.)

        Under the ``.part`` staging contract the complete-length staged file is
        seeded at ``<out>.part`` (not ``out``), since a 416 now proves the
        ``.part`` file — not a pre-existing ``out`` — is at least complete.
        """
        base_url, server = chunked_range_server
        remote = _make_payload(nbytes=250_000, seed=5)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Same length as remote (so a Range at EOF 416s) but different content.
        cache = _make_payload(nbytes=250_000, seed=777)
        assert cache != remote
        part = out.with_name(out.name + ".part")
        part.write_bytes(cache)

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists(), "complete staged file must be promoted via in-loop 416"
        assert not part.exists(), "staged .part must be renamed onto out"
        assert out.read_bytes() == cache, "staged bytes must NOT be re-downloaded"
        assert 416 in server.served, "the in-loop 416 path must have been hit"
        assert 206 not in server.served, "no resume body should be fetched"

    def test_complete_dest_on_sizeless_host_not_redownloaded(
        self, chunked_range_server, tmp_path: Path
    ) -> None:
        """A COMPLETE file already at ``out`` on a size-less/chunked host must be
        CONFIRMED complete via a Range-at-EOF 416 and returned untouched — never
        fully re-downloaded.

        The chunked server omits Content-Length, so the pre-loop size probe
        cannot early-detect "complete". ``robust_download`` must MIGRATE the
        pre-existing ``out`` into the staging path, issue a Range at EOF, get a
        416, and promote the staged bytes straight back — ZERO body download.

        The seeded bytes DIFFER from the remote (same length) so a spurious full
        re-download is caught by content; the server also records the 416 (and
        no 206). This pins the migrate-and-confirm fix: the pre-migration code
        fell through, found no staging file, re-fetched the whole body over a
        200 (served would be ``[200, 200, 200]`` with NO 416) and overwrote the
        seeded bytes with the remote's — so it FAILS against that version on
        both the content and the ``416 in served`` assertions.
        """
        base_url, server = chunked_range_server
        remote = _make_payload(nbytes=250_000, seed=5)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Same length as remote (so a Range at EOF 416s) but different content.
        cached = _make_payload(nbytes=250_000, seed=777)
        assert cached != remote
        out.write_bytes(cached)  # COMPLETE file seeded AT out (NOT .part)
        part = out.with_name(out.name + ".part")

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists(), "complete dest must survive on a size-less host"
        assert out.read_bytes() == cached, "dest must NOT be re-downloaded"
        assert not part.exists(), "staging .part must be renamed onto out"
        assert 416 in server.served, "completeness must be confirmed via a 416"
        assert 206 not in server.served, "no resume body should be fetched"

    def test_zero_byte_remote_truncates_stale_cache(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A remote replaced by a 0-byte file must truncate a non-empty stale
        cache, not return it.

        A 0-byte remote reports ``Content-Length: 0`` → ``_parse_len`` → None, so
        ``remote_size`` is unknown and the Range-at-EOF 416 is answered with
        ``Content-Range: bytes */0``. ``_parse_content_range_total`` now accepts
        the literal ``0`` as a real total, so the handler sees ``local != 0``,
        restarts cleanly, and the file ends empty.
        """
        (tmp_path / "data.bin").write_bytes(b"")  # empty remote

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        out.write_bytes(_make_payload(nbytes=50_000))  # non-empty stale cache

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists()
        assert out.read_bytes() == b"", "stale cache must truncate to empty remote"

    def test_stale_oversized_cache_refetches_via_in_loop_416(
        self, chunked_range_server, tmp_path: Path
    ) -> None:
        """CORE REGRESSION follow-up: a stale, OVERSIZED staged partial on a
        chunked host (no Content-Length) must self-heal via the 416
        ``Content-Range`` total.

        The chunked server omits Content-Length, so ``remote_size`` resolves as
        UNKNOWN — pre-fix, the in-loop 416 handler treated "unknown remote size"
        as "the staged file is complete" and handed back the stale, oversized
        file forever. The 416 response's ``Content-Range: bytes */<total>`` now
        supplies the authoritative (smaller) total, so the handler detects
        ``part > total``, restarts cleanly, and ends byte-identical to the
        current remote.

        Under the ``.part`` staging contract the oversized partial is seeded at
        ``<out>.part`` (the 416 handler now reasons about the ``.part`` file).
        """
        base_url, server = chunked_range_server
        remote = _make_payload(nbytes=200_000, seed=5)  # current (smaller) remote
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Stale, LARGER staged copy (an older, bigger version of the asset).
        part = out.with_name(out.name + ".part")
        part.write_bytes(_make_payload(nbytes=350_000, seed=99))

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == remote, "stale staged partial must refetch clean"
        assert not part.exists(), "staged .part must be renamed onto out"
        assert 416 in server.served, "the in-loop 416 path must have been hit"

    def test_headerless_416_returns_at_least_complete_cache(
        self, headerless_416_server, tmp_path: Path
    ) -> None:
        """With NO size signal at all (chunked + a 416 lacking Content-Range),
        an existing staged partial is conservatively promoted untouched.

        A 416 still proves ``part >= total``, so the staged file is AT LEAST
        complete; with neither a Content-Length nor a Content-Range total to
        disambiguate, the handler promotes the staged bytes rather than
        destroying them.

        Under the ``.part`` staging contract the complete-length staged file is
        seeded at ``<out>.part``.
        """
        base_url, server = headerless_416_server
        remote = _make_payload(nbytes=250_000, seed=5)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        cache = _make_payload(nbytes=250_000, seed=777)
        assert cache != remote
        part = out.with_name(out.name + ".part")
        part.write_bytes(cache)

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists(), "complete staged file must survive a header-less 416"
        assert not part.exists(), "staged .part must be renamed onto out"
        assert out.read_bytes() == cache, "staged bytes must NOT be re-downloaded"
        assert 416 in server.served, "the in-loop 416 path must have been hit"

    def test_interrupted_download_leaves_no_trusted_dest(
        self, range_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #732: an interrupted download must NEVER leave a truncated file
        at the destination that a size-less cache check would trust forever.

        The response body is dropped mid-transfer on every attempt (a simulated
        ``ConnectionError``), so all retries are exhausted. Afterwards ``out``
        must NOT exist — the truncated bytes stay in ``<out>.part`` — so a
        subsequent ``cached_download``-style ``dest.exists()`` check returns
        False and the poisoned partial can never be served as a valid cache.

        Pre-fix, ``robust_download`` wrote straight into ``out`` and the
        truncated file survived at the canonical name, poisoning the cache.
        """
        remote = _make_payload(nbytes=500_000, seed=3)
        (tmp_path / "data.bin").write_bytes(remote)

        real_iter = requests.models.Response.iter_content

        def dropping_iter(self, *args, **kwargs):  # noqa: ANN001, ANN202
            # Pass a little real data through, then simulate the connection
            # dropping mid-body — on EVERY attempt.
            for chunk in real_iter(self, *args, **kwargs):
                yield chunk
                raise requests.exceptions.ConnectionError("simulated mid-body drop")

        monkeypatch.setattr(requests.models.Response, "iter_content", dropping_iter)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")

        with pytest.raises(requests.exceptions.ConnectionError):
            robust_download(
                f"{range_server}/data.bin",
                out,
                expected_size=None,
                verify_size=True,
                max_retries=0,
                chunk_size=40_000,
            )

        assert not out.exists(), "interrupted download must not create a dest"
        # The bytes that landed live ONLY in the (truncated) staging file, never
        # at `out` — so a cached_download-style `dest.exists()` check sees nothing.
        assert part.exists(), "the truncated bytes must remain in the staging file"
        assert part.stat().st_size < len(remote)

    def test_resume_after_midbody_drop_does_not_duplicate(
        self, range_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #721: a mid-body drop on a resume must NOT duplicate a byte range.

        The staged ``.part`` file already holds the true ``remote[:N]``. The
        first resumed (206) attempt drops after writing a few more chunks;
        pre-fix the retry re-sent ``Range: bytes=<N>-`` (the STALE offset
        computed once before the loop) and re-opened in append mode, appending
        the already-written bytes AGAIN — a silently duplicated region. The fix
        re-stats the ``.part`` file before retrying, so the retry's Range matches
        what is really on disk and the promoted ``out`` is byte-identical to the
        remote payload (exact length, no duplication).
        """
        n = 120_000
        payload = _make_payload(nbytes=500_000, seed=1)
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        part.write_bytes(payload[:n])  # true remote prefix → valid 206 resume

        real_iter = requests.models.Response.iter_content
        state = {"attempt": 0}

        def flaky_iter(self, *args, **kwargs):  # noqa: ANN001, ANN202
            state["attempt"] += 1
            gen = real_iter(self, *args, **kwargs)
            if state["attempt"] == 1:
                # First resumed attempt: pass two real chunks, then drop.
                for i, chunk in enumerate(gen):
                    yield chunk
                    if i >= 1:
                        raise requests.exceptions.ChunkedEncodingError(
                            "simulated mid-body drop"
                        )
            else:
                yield from gen

        monkeypatch.setattr(requests.models.Response, "iter_content", flaky_iter)

        result = robust_download(
            f"{range_server}/data.bin",
            out,
            expected_size=None,
            verify_size=True,
            chunk_size=40_000,
        )

        assert result == out
        assert not part.exists(), "staged .part must be renamed onto out"
        assert out.read_bytes() == payload, "resume must not duplicate any bytes"
        assert len(out.read_bytes()) == len(payload)

    def test_stale_shorter_dest_on_sized_host_refetches_clean(
        self, range_server: str, tmp_path: Path
    ) -> None:
        """A stale COMPLETE dest SHORTER than the remote on a size-REPORTING host
        must be re-fetched cleanly — NEVER migrated-and-spliced.

        The migration probe exists only to CONFIRM a dest whose completeness the
        size probe can't judge (size-less host). On a host that reports
        Content-Length a size mismatch is authoritative, so the dest must NOT be
        fed to the resume path: doing so would splice ``stale[:local] +
        remote[local:]`` into a file that (by construction) passes size
        verification and is then trusted forever. Here the stale dest is
        150 KB with a DIFFERENT prefix from the 300 KB remote, so a splice would
        leave the first 150 KB non-remote; a clean fresh fetch yields the full
        remote payload byte-for-byte.

        This pins FIX 1: against the UNGATED migration it FAILS — the dest is
        migrated to ``.part`` (150 KB), the ``< remote`` branch resumes at
        150 KB, a 206 appends ``remote[150000:]``, and ``out`` ends as
        ``seed999[:150000] + remote[150000:]`` (first 150 KB differ from
        ``remote``). With the ``remote_size is None`` gate it does not migrate,
        fetches fresh, and ends byte-identical to ``remote``.
        """
        remote = _make_payload(nbytes=300_000, seed=7)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Stale, SHORTER, DIFFERENT-content complete cache (an older asset).
        out.write_bytes(_make_payload(nbytes=150_000, seed=999))
        part = out.with_name(out.name + ".part")

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == remote, "stale dest must refetch clean, not splice"
        assert not part.exists(), "staging .part must be renamed onto out"

    def test_offline_restore_preserves_migrated_cache(
        self, chunked_range_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A migrated complete cache must be RESTORED when the fetch fails offline
        (network-error terminal path).

        On a size-less/chunked host a complete dest is migrated into the staging
        path to be confirmed via a Range-at-EOF 416. If the connection is dead,
        the ranged GET raises ``ConnectionError`` before any body byte is
        written, so the staging file is still byte-identical to the migrated
        cache — the terminal-failure restore must move it back onto ``out`` so a
        transient/offline failure never strands (destroys) a usable cache.
        """
        base_url, _server = chunked_range_server
        remote = _make_payload(nbytes=200_000, seed=5)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        cached = _make_payload(nbytes=200_000, seed=321)  # the complete cache
        out.write_bytes(cached)
        part = out.with_name(out.name + ".part")

        real_get = requests.sessions.Session.get

        def offline_get(self, url, **kwargs):  # noqa: ANN001, ANN202
            # A ranged (resume) GET is the download body request — simulate the
            # network being down for it, while leaving the unranged size-probe
            # GET to behave normally (it resolves the size as unknown anyway).
            if "Range" in dict(kwargs.get("headers") or {}):
                raise requests.exceptions.ConnectionError("simulated offline")
            return real_get(self, url, **kwargs)

        monkeypatch.setattr(requests.sessions.Session, "get", offline_get)

        with pytest.raises(requests.exceptions.ConnectionError):
            robust_download(
                f"{base_url}/data.bin",
                out,
                expected_size=None,
                verify_size=False,
                max_retries=0,
            )

        assert out.exists(), "offline failure must restore the migrated cache"
        assert out.read_bytes() == cached, "restored cache must be byte-identical"
        assert not part.exists(), "the migrated cache must be restored, not orphaned"

    def test_stale_complete_dest_sizeless_206_refetches_clean(
        self, chunked_range_server, tmp_path: Path
    ) -> None:
        """A stale complete dest SHORTER than the remote on a size-less host whose
        Range-at-EOF is SATISFIABLE (206, not 416) must re-fetch clean, not splice.

        On a chunked/size-less host a complete dest is migrated to the staging
        path to be confirmed via a Range at EOF. When the dest is SHORTER than
        the (changed) remote, that Range is satisfiable, so the server answers
        206 with the tail ``remote[N:]``. A MIGRATED file is by invariant
        complete, so a 206 PROVES it is STALE — appending the tail would splice
        ``stale[:N] + remote[N:]`` into a file that passes size verification and
        is trusted forever. The fix treats the 206 as staleness and re-downloads
        from scratch, so ``out`` ends byte-identical to the FULL remote.

        This pins the fix: against the pre-fix code the migrated file is appended
        to (206 → ``mode="ab"``), so ``out`` becomes ``seed555[:150000] +
        remote[150000:]`` — differing from ``remote`` in the first 150 000 bytes
        — and the assertion FAILS. After the fix ``out == remote``.
        """
        base_url, server = chunked_range_server
        remote = _make_payload(nbytes=300_000, seed=7)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Stale, SHORTER, DIFFERENT-content complete cache → Range-at-EOF is
        # satisfiable (150000 < 300000) so the server answers 206, not 416.
        out.write_bytes(_make_payload(nbytes=150_000, seed=555))
        part = out.with_name(out.name + ".part")

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.read_bytes() == remote, "stale dest must refetch clean, not splice"
        assert not part.exists(), "staging .part must be renamed onto out"
        assert 206 in server.served, "the satisfiable Range must have yielded a 206"

    def test_stale_206_restart_failure_leaves_no_poisoned_part(
        self, chunked_range_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 206-proven-stale cache whose from-scratch re-fetch FAILS must not
        leave the full stale copy in ``.part`` for a later run to splice.

        Run 1: the stale complete dest (150 KB, shorter than the 300 KB remote)
        is migrated on the size-less host, the Range-at-EOF gets a 206 proving
        it stale, and the unranged from-scratch re-fetch then dies (network
        drop) with retries exhausted. Pre-fix the condemned bytes survived in
        ``.part`` — the ``"wb"`` truncation that was supposed to discard them
        was never reached — and run 2 then resumed at 150 KB, got a 206 tail,
        and spliced ``stale[:150k] + remote[150k:]`` into a corrupt file that
        passes size verification and is promoted as trusted. Post-fix the stale
        bytes are unlinked the moment the 206 proves them stale, so run 1
        leaves no ``.part`` and run 2 fetches the full remote clean.
        """
        base_url, _server = chunked_range_server
        remote = _make_payload(nbytes=300_000, seed=7)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        stale = _make_payload(nbytes=150_000, seed=555)
        out.write_bytes(stale)  # stale complete cache, shorter than the remote
        part = out.with_name(out.name + ".part")

        real_get = requests.sessions.Session.get
        state = {"ranged_seen": False, "offline": True}

        def flaky_get(self, url, **kwargs):  # noqa: ANN001, ANN202
            # Let the size probe and the ranged (stale-proving 206) GET through,
            # then fail the unranged from-scratch re-fetch while "offline".
            headers = dict(kwargs.get("headers") or {})
            if "Range" in headers:
                state["ranged_seen"] = True
            elif state["ranged_seen"] and state["offline"]:
                raise requests.exceptions.ConnectionError("simulated network drop")
            return real_get(self, url, **kwargs)

        monkeypatch.setattr(requests.sessions.Session, "get", flaky_get)

        with pytest.raises(requests.exceptions.ConnectionError):
            robust_download(
                f"{base_url}/data.bin",
                out,
                expected_size=None,
                verify_size=True,
                max_retries=0,
            )

        assert not part.exists(), "proven-stale bytes must not survive in .part"
        assert not out.exists(), "the proven-stale cache is legitimately discarded"

        # Run 2: network back — must fetch the FULL remote, never splice.
        state["offline"] = False
        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == remote, "re-fetch must be clean, never spliced"
        assert not part.exists(), "staging .part must be renamed onto out"

    def test_changed_remote_invalidates_recorded_validator(
        self, etag_range_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A partial staged against an asset that CHANGED before the retry must
        be re-fetched from scratch — never spliced ``old_prefix + new_tail``.

        Run 1 downloads fresh from a validating (ETag) host and drops mid-body,
        leaving a truncated ``.part`` plus the recorded validator of the OLD
        representation. The remote is then replaced (different content, larger
        size — the dangerous case, where a bare Range at the partial's length is
        perfectly satisfiable). Run 2 must send ``If-Range`` with the recorded
        validator, receive a 200 full body from the changed remote, and end
        byte-identical to the NEW payload.

        Pre-fix (no validator recorded, no ``If-Range`` sent) the server answers
        206 with the new asset's tail, splicing ``old[:N] + new[N:]`` into a
        corrupt file whose length passes size verification — so this test fails
        against that code on the content assertion.
        """
        base_url, server = etag_range_server
        old_remote = _make_payload(nbytes=300_000, seed=11)
        (tmp_path / "data.bin").write_bytes(old_remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        validator = out.with_name(out.name + ".part.validator")

        real_iter = requests.models.Response.iter_content

        def dropping_iter(self, *args, **kwargs):  # noqa: ANN001, ANN202
            for chunk in real_iter(self, *args, **kwargs):
                yield chunk
                raise requests.exceptions.ConnectionError("simulated mid-body drop")

        monkeypatch.setattr(requests.models.Response, "iter_content", dropping_iter)
        with pytest.raises(requests.exceptions.ConnectionError):
            robust_download(
                f"{base_url}/data.bin",
                out,
                expected_size=None,
                verify_size=True,
                max_retries=0,
                chunk_size=40_000,
            )
        monkeypatch.undo()

        assert part.exists(), "run 1 must leave a resumable partial"
        assert validator.exists(), "run 1 must record the representation validator"
        assert validator.read_text().strip() == _etag_of(tmp_path / "data.bin")

        # The asset is re-uploaded: different content, LARGER size, so a bare
        # (unvalidated) Range at the partial's length would be satisfiable.
        new_remote = _make_payload(nbytes=400_000, seed=22)
        (tmp_path / "data.bin").write_bytes(new_remote)
        server.served.clear()

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == new_remote, "changed remote must re-fetch clean"
        assert 206 not in server.served, "no tail of the NEW asset may be appended"
        assert not part.exists(), "staging .part must be renamed onto out"
        assert not validator.exists(), "the validator sidecar must not outlive .part"

    def test_matching_validator_resumes_via_206(
        self, etag_range_server, tmp_path: Path
    ) -> None:
        """A recorded validator that still matches the remote must RESUME (206),
        not force a full re-download, and complete byte-identical."""
        base_url, server = etag_range_server
        n = 120_000
        payload = _make_payload(nbytes=500_000, seed=1)
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        part.write_bytes(payload[:n])  # true remote prefix
        validator = out.with_name(out.name + ".part.validator")
        validator.write_text(_etag_of(tmp_path / "data.bin"))

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == payload
        assert 206 in server.served, "matching validator must resume via 206"
        assert not part.exists()
        assert not validator.exists(), "the validator sidecar must not outlive .part"

    def test_failed_restart_never_leaves_stale_part_under_new_validator(
        self, etag_range_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A restart that dies before the staging file is truncated must not leave
        the OLD bytes behind under the NEW representation's validator.

        Run 1 resumes a partial of the OLD asset against a CHANGED remote: the
        ``If-Range`` mismatch correctly yields a 200 full body, so the download
        restarts from scratch — and records the new representation's validator.
        The staging file is then opened ``"wb"``, and it is that open which
        discards the old bytes. If it fails (disk full, EMFILE — simulated here)
        after the validator was published, the old bytes survive paired with a
        validator that vouches for the NEW asset: run 2 sends it, the server
        happily answers 206, and ``old[:N] + new[N:]`` is spliced into a corrupt
        file whose length passes size verification.

        The condemned bytes are therefore discarded BEFORE the validator is
        replaced, so an interruption in that window leaves nothing resumable.
        """
        base_url, server = etag_range_server
        old_remote = _make_payload(nbytes=300_000, seed=11)
        (tmp_path / "data.bin").write_bytes(old_remote)

        n = 120_000
        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        validator = out.with_name(out.name + ".part.validator")
        part.write_bytes(old_remote[:n])  # a genuine partial of the OLD asset
        validator.write_text(_etag_of(tmp_path / "data.bin"))

        # The asset is re-uploaded LARGER, so a bare Range at the partial's
        # length stays satisfiable — the splice is only prevented by the
        # validator, which must not end up describing these old bytes.
        new_remote = _make_payload(nbytes=400_000, seed=22)
        (tmp_path / "data.bin").write_bytes(new_remote)

        real_open = open

        def failing_open(file, mode="r", *args, **kwargs):  # noqa: ANN001, ANN202
            if Path(file) == part and "w" in mode:
                raise OSError(28, "No space left on device")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", failing_open)
        with pytest.raises(OSError):
            robust_download(
                f"{base_url}/data.bin", out, expected_size=None, max_retries=0
            )
        monkeypatch.undo()

        assert not part.exists(), (
            "condemned staged bytes must not survive a failed restart"
        )

        # Run 2: whatever survived must not be spliced onto the new asset.
        server.served.clear()
        result = robust_download(f"{base_url}/data.bin", out, expected_size=None)

        assert result == out
        assert out.read_bytes() == new_remote, "old bytes must never be spliced in"
        assert 206 not in server.served, "nothing resumable should have been left"
        assert not part.exists()
        assert not validator.exists()

    def test_misaligned_206_restarts_clean(
        self, misaligned_resume_server: str, tmp_path: Path
    ) -> None:
        """A 206 whose ``Content-Range`` start differs from the requested offset
        must NOT be appended — restart clean and end byte-identical.

        The misbehaving server answers every Range request 206-from-byte-0.
        Pre-fix the body was appended at the resume offset, producing
        ``prefix + whole_remote`` — which even passed size verification, because
        the expected total was computed as ``content_length + resume_byte_pos``.
        Post-fix the misalignment is detected, the staged bytes are discarded,
        and the unranged re-fetch yields the exact remote payload.
        """
        n = 120_000
        payload = _make_payload(nbytes=300_000, seed=1)
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        part = out.with_name(out.name + ".part")
        part.write_bytes(payload[:n])  # true prefix — a genuine partial

        result = robust_download(
            f"{misaligned_resume_server}/data.bin",
            out,
            expected_size=None,
            verify_size=True,
        )

        assert result == out
        assert out.read_bytes() == payload, "misaligned 206 must never be appended"
        assert not part.exists(), "staging .part must be renamed onto out"

    def test_malformed_content_range_total_degrades_to_unknown(
        self, tmp_path: Path
    ) -> None:
        """A ``Content-Range: .../*`` (star total) with no Content-Length must
        not abort the download: the in-loop size parse degrades to "unknown"
        and the body still downloads to completion.

        Before the parse was hardened, ``int("*")`` raised ValueError into the
        generic handler, which re-raised non-retryably.
        """
        remote = _make_payload(seed=99)
        (tmp_path / "data.bin").write_bytes(remote)

        server, thread = _serve(tmp_path, _StarTotalContentRangeHandler)
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            out = tmp_path / "cache" / "data.bin"
            out.parent.mkdir(parents=True)
            result = robust_download(
                f"{base_url}/data.bin", out, expected_size=None, verify_size=True
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        assert result == out
        assert out.read_bytes() == remote, "body must download despite a star total"

    def test_gzip_probe_uses_identity_and_returns_complete_cache(
        self, tmp_path: Path
    ) -> None:
        """The size probe must send ``Accept-Encoding: identity`` so a gzip host
        reports the true DECODED size — not the smaller COMPRESSED one.

        On a gzip-capable host the probe's ``Accept-Encoding: identity`` guard
        makes the reported Content-Length equal the on-disk (decoded) cache size,
        so a complete cache is recognized and returned untouched from the HEAD
        alone — no body GET.

        Remove that guard and the probe (via requests' default ``gzip``) sees the
        smaller COMPRESSED length, decides the complete cache is "incomplete",
        and re-fetches; with ``verify_size=True`` the decoded body then fails the
        ``final_size == total_size`` check (total is the compressed length) and
        the download raises ``ValueError`` — so this test fails against the
        unguarded code (both on the raised error and the assertions below).
        """
        # Highly compressible payload so the COMPRESSED length is far smaller
        # than the DECODED length — the gap the identity guard must avoid seeing.
        remote = b"gzip-probe-guard-regression-payload\n" * 8000
        (tmp_path / "data.bin").write_bytes(remote)

        server, thread = _serve(tmp_path, _GzipRangeHTTPHandler)
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            out = tmp_path / "cache" / "data.bin"
            out.parent.mkdir(parents=True)
            out.write_bytes(remote)  # complete cache, byte-identical to remote

            result = robust_download(
                f"{base_url}/data.bin", out, expected_size=None, verify_size=True
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        assert result == out
        assert out.read_bytes() == remote, "complete cache must be returned untouched"
        # A HEAD probe that DID send identity must have actually run — otherwise
        # the two ledger checks below would pass vacuously on an empty log.
        assert ("HEAD", False) in server.served, "identity size probe must have run"
        # The identity guard means the probe never negotiated gzip, so the
        # reported size matched the decoded cache and NO body GET was needed.
        assert all(
            not used_gzip for _cmd, used_gzip in server.served
        ), "size probe must send Accept-Encoding: identity, not gzip"
        assert not any(
            cmd == "GET" for cmd, _used_gzip in server.served
        ), "a complete cache must be confirmed by the HEAD probe, not re-downloaded"
