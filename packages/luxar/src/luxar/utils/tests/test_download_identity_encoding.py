"""Regression tests for ``robust_download`` identity-encoding handling (issue #996).

A stdlib ``http.server`` subclass inspects the request's ``Accept-Encoding``:

* If the client leaves requests' default ``Accept-Encoding: gzip, deflate`` in
  place (i.e. does NOT force ``identity``), the server responds with a
  **gzip-COMPRESSED** body, ``Content-Encoding: gzip``, and a ``Content-Length``
  equal to the COMPRESSED length. ``requests`` transparently decompresses, so
  ``iter_content`` yields the larger DECODED body — bytes-on-disk then differ
  from the advertised ``Content-Length`` and size verification fails.
* If the client forces ``Accept-Encoding: identity``, the server responds with
  the raw uncompressed body and a ``Content-Length`` equal to the true decoded
  length (the byte stream the ``.part`` resume machinery assumes).

The probe (HEAD/unranged GET) already forces identity, so HEAD reports the
decoded length. The bug is that the body GET in the retry loop did NOT force
identity, so a FRESH download with ``verify_size=True`` failed non-retryably.
No network access.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from luxar.utils.download import robust_download


def _wants_gzip(accept_encoding: str | None) -> bool:
    """True if the client would accept a gzip-encoded body (default requests)."""
    if not accept_encoding:
        return False
    return "gzip" in accept_encoding.lower()


class _GzipUnlessIdentityHandler(http.server.SimpleHTTPRequestHandler):
    """Serve a gzip-compressed body UNLESS the client forces identity.

    Demonstrates the bug: the pre-fix body GET inherits requests' default
    ``Accept-Encoding: gzip, deflate``, so the server gzip-encodes and reports
    the COMPRESSED ``Content-Length`` while requests writes the DECODED bytes.
    """

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _serve(self, include_body: bool) -> None:  # noqa: FBT001
        path = Path(self.translate_path(self.path))
        # Record what the client actually put on the wire so a test can assert
        # the request reached the server (method, Accept-Encoding, whether it
        # was a body GET). `include_body` distinguishes GET (True) from HEAD.
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "command": self.command,
                "accept_encoding": self.headers.get("Accept-Encoding"),
                "range": self.headers.get("Range"),
                "body": include_body,
            }
        )
        if not path.is_file():
            self.send_error(404)
            return

        raw = path.read_bytes()
        accept = self.headers.get("Accept-Encoding")

        if _wants_gzip(accept):
            body = gzip.compress(raw)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))  # COMPRESSED length
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        else:
            body = raw
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))  # DECODED length
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        if include_body:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib override
        self._serve(include_body=False)

    def do_GET(self) -> None:  # noqa: N802 - stdlib override
        self._serve(include_body=True)


def _serve(directory: Path):  # noqa: ANN202
    handler = lambda *a, **kw: _GzipUnlessIdentityHandler(  # noqa: E731
        *a, directory=str(directory), **kw
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests = []  # type: ignore[attr-defined]  # observed requests
    server.handle_error = lambda request, client_address: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def gzip_server(tmp_path: Path):
    """Serve tmp_path, gzip-encoding unless the client forces identity.

    Yields ``(base_url, server)`` so a test can inspect ``server.requests``.
    """
    server, thread = _serve(tmp_path)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _make_payload(nbytes: int = 400_000) -> bytes:
    """Highly compressible bytes so the gzip size differs clearly from raw."""
    return (b"luxar-identity-encoding-regression\n" * (nbytes // 35 + 1))[:nbytes]


class TestRobustDownloadIdentityEncoding:
    def test_fresh_download_forces_identity(self, gzip_server, tmp_path: Path) -> None:
        """CORE REGRESSION (issue #996): a fresh download from a gzip-encoding
        host with ``verify_size=True`` must SUCCEED and land the decoded bytes.

        Pre-fix, the body GET inherited ``Accept-Encoding: gzip, deflate``, so
        ``total_size`` came from the COMPRESSED ``Content-Length`` while
        ``iter_content`` wrote the larger DECODED body — the final
        ``final_size != total_size`` check raised ``ValueError``. The
        highly-compressible payload makes the two sizes clearly diverge.
        """
        base_url, _server = gzip_server
        payload = _make_payload()
        assert len(gzip.compress(payload)) < len(payload)  # sizes must diverge
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)  # empty dest dir, no `.part`

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.exists()
        assert out.read_bytes() == payload, "on-disk bytes must be the decoded body"
        assert (
            hashlib.md5(out.read_bytes(), usedforsecurity=False).hexdigest()
            == hashlib.md5(payload, usedforsecurity=False).hexdigest()
        )

    @pytest.mark.parametrize("header_key", ["Accept-Encoding", "accept-encoding"])
    def test_explicit_accept_encoding_is_honored(
        self, gzip_server, tmp_path: Path, header_key: str
    ) -> None:
        """An explicit ``extra_headers`` Accept-Encoding must WIN over the
        forced-identity default — case-INSENSITIVELY (fix #1).

        With ``gzip`` explicitly requested the server gzip-encodes and reports
        the compressed length, so the decoded-vs-advertised size mismatch
        returns: the download must fail (``verify_size=True``). We ALSO assert
        the server actually observed ``Accept-Encoding: gzip`` on the wire, so
        this can't pass merely because ``extra_headers`` was dropped (requests'
        own default gzip would also trigger the gzip branch). The lowercase-key
        variant proves ``_force_identity_encoding`` honors a caller header
        regardless of case.
        """
        base_url, server = gzip_server
        payload = _make_payload()
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)

        with pytest.raises(ValueError, match="size mismatch"):
            robust_download(
                f"{base_url}/data.bin",
                out,
                expected_size=None,
                verify_size=True,
                max_retries=0,
                extra_headers={header_key: "gzip"},
            )

        # The caller's gzip preference reached the wire on the body GET (not the
        # forced identity default) — proving setdefault-vs-clobber is correct.
        body_gets = [r for r in server.requests if r["body"] and r["range"] is None]
        assert body_gets, "a body GET must have been issued"
        assert all(r["accept_encoding"] == "gzip" for r in body_gets), (
            f"server did not observe the caller's gzip header: {server.requests}"
        )

    def test_complete_cache_on_gzip_host_not_redownloaded(
        self, gzip_server, tmp_path: Path
    ) -> None:
        """A COMPLETE, correct dest on a gzip-capable host must SHORT-CIRCUIT via
        the identity-forcing size probe — ZERO body GET (issue #832/#996 family).

        The probe (``_resolve_remote_size``) forces identity, so it reports the
        DECODED size, which matches the local file → the complete-cache fast
        path returns it untouched. If the probe were case-sensitive/broken it
        would report the COMPRESSED size, see a mismatch, and pointlessly
        re-download — so this guards the probe path too.
        """
        base_url, server = gzip_server
        payload = _make_payload()
        (tmp_path / "data.bin").write_bytes(payload)

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)
        out.write_bytes(payload)  # COMPLETE, correct cache seeded up front

        result = robust_download(
            f"{base_url}/data.bin", out, expected_size=None, verify_size=True
        )

        assert result == out
        assert out.read_bytes() == payload, "complete cache must be left untouched"
        # The server MUST have been probed (otherwise "no re-download" is
        # vacuously true), and that probe MUST have forced identity — that is
        # what makes the decoded size match the cache and take the fast path.
        probes = [r for r in server.requests if not r["body"]]
        assert probes, "complete-cache check must actually contact the server"
        assert all(r["accept_encoding"] == "identity" for r in probes), (
            f"size probe must force identity, got: {probes}"
        )
        # No body GET for the content (HEAD probe only is fine). A ranged or
        # unranged body GET would mean the fast path was missed.
        body_gets = [r for r in server.requests if r["body"]]
        assert not body_gets, f"complete cache must not be re-downloaded: {body_gets}"
