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

import http.server
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

        The local prefix bytes DIFFER from the true remote prefix, so the only
        way the final file can equal ``local_prefix + remote[N:]`` is if bytes
        ``[0, N)`` were kept (never re-fetched) — proving a real 206 resume.
        """
        n = 120_000
        payload = _make_payload(nbytes=500_000, seed=1)
        (tmp_path / "data.bin").write_bytes(payload)

        # A DIFFERENT prefix of the same length as the true remote's [0, N).
        local_prefix = _make_payload(nbytes=n, seed=424242)
        assert local_prefix != payload[:n]

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        out.write_bytes(local_prefix)  # partial, with distinguishable bytes

        result = robust_download(
            f"{range_server}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
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
        must leave the pre-existing file intact (only files THIS call created
        are cleaned up)."""
        sentinel = b"do-not-delete-me" * 1000

        out = tmp_path / "cache" / "missing.bin"
        out.parent.mkdir(parents=True)
        out.write_bytes(sentinel)  # pre-existing cache under a 404 URL

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

    def test_complete_cache_returned_via_in_loop_416(
        self, chunked_range_server, tmp_path: Path
    ) -> None:
        """Exercise the IN-LOOP 416 handler (not the size-probe fast path).

        The chunked server omits Content-Length on HEAD/GET, so the remote size
        resolves as UNKNOWN — ``robust_download`` cannot early-detect "complete"
        and issues a Range at EOF, which the server answers with 416. The
        handler must return the existing complete file WITHOUT truncating or
        re-downloading it. The local cache bytes DIFFER from the remote (same
        length) so a spurious re-download is caught by content; the server also
        records that it served ZERO 206 (resume) responses. (A 200 is expected
        from the unranged size probe, so it is not asserted against.)
        """
        base_url, server = chunked_range_server
        remote = _make_payload(nbytes=250_000, seed=5)
        (tmp_path / "data.bin").write_bytes(remote)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        # Same length as remote (so a Range at EOF 416s) but different content.
        cache = _make_payload(nbytes=250_000, seed=777)
        assert cache != remote
        out.write_bytes(cache)

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=False
        )

        assert result == out
        assert out.exists(), "complete cache must survive an in-loop 416"
        assert out.read_bytes() == cache, "cache must NOT be re-downloaded"
        assert 416 in server.served, "the in-loop 416 path must have been hit"
        assert 206 not in server.served, "no resume body should be fetched"
