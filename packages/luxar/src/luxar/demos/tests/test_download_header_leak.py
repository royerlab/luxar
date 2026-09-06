"""Security regression tests for ``robust_download`` redirect header leakage.

``requests`` strips the ``Authorization`` header on a redirect that crosses to a
different host, but it does NOT strip arbitrary custom headers. A caller passing
a credential-bearing ``extra_headers`` entry (an ``api-key``, an ``x-api-key``, a
Zenodo/S3 bearer token, …) would therefore leak that credential to whatever host
a redirect points at (finding A13-02). ``robust_download`` now routes through a
origin-scoped session that drops those headers whenever requests would strip
``Authorization`` (host, port, or unsafe scheme changes).

The end-to-end tests use two loopback aliases (``127.0.0.1`` and ``localhost``)
as distinct "hosts" so a redirect between them exercises the real requests
redirect machinery with NO network access. ``test_plain_session_leaks_...``
documents the underlying requests behaviour the fix defends against, so the
control is demonstrably real and not vacuous.
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest
import requests

from luxar.demos._support.downloads.download import (
    _make_host_scoped_session,
    robust_download,
)

_PAYLOAD = b"luxar-cross-host-header-leak-regression\n" * 64
_SECRET = "super-secret-token"


class _RedirectingHandler(http.server.SimpleHTTPRequestHandler):
    """Redirect ``/start`` to a configured target and serve ``/payload``.

    Records every request's ``api-key`` / ``Authorization`` headers keyed by
    path on ``server.seen`` so a test can assert what reached the redirect
    target.
    """

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _record(self) -> None:
        self.server.seen.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "host": self.headers.get("Host"),
                "api_key": self.headers.get("api-key"),
                "authorization": self.headers.get("Authorization"),
            }
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib override
        self._record()
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", self.server.redirect_target)  # type: ignore[attr-defined]
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/payload":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(_PAYLOAD)))
            self.end_headers()
            self.wfile.write(_PAYLOAD)
            return
        self.send_error(404)


def _serve() -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectingHandler)
    server.seen = []  # type: ignore[attr-defined]  # observed requests
    server.handle_error = lambda request, client_address: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def redirect_server():  # noqa: ANN201
    """A loopback server that 302-redirects ``/start`` to ``server.redirect_target``.

    Yields ``(port, server)``; the test sets ``server.redirect_target`` and picks
    the original-request hostname to make the redirect same-host or cross-host.
    """
    server, thread = _serve()
    try:
        yield server.server_address[1], server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _payload_request(server: http.server.ThreadingHTTPServer) -> dict:
    """Return the recorded request that reached ``/payload`` (must be exactly one)."""
    hits = [r for r in server.seen if r["path"] == "/payload"]  # type: ignore[attr-defined]
    assert len(hits) == 1, f"expected one /payload request, got: {server.seen}"  # type: ignore[attr-defined]
    return hits[0]


class TestRobustDownloadCrossHostHeaderLeak:
    @pytest.mark.parametrize(
        ("original_url", "redirect_url", "should_strip"),
        [
            (
                "https://example.com/start",
                "http://example.com/payload",
                True,
            ),
            (
                "https://example.com/start",
                "https://example.com:8443/payload",
                True,
            ),
            (
                "http://example.com/start",
                "https://example.com/payload",
                False,
            ),
        ],
        ids=["https-downgrade", "port-change", "default-port-http-upgrade"],
    )
    def test_custom_headers_follow_requests_authorization_policy(
        self, original_url: str, redirect_url: str, should_strip: bool
    ) -> None:
        original = requests.Request("GET", original_url).prepare()
        response = requests.Response()
        response.request = original
        redirected = requests.Request(
            "GET", redirect_url, headers={"api-key": _SECRET}
        ).prepare()

        scoped = _make_host_scoped_session(original_url, {"api-key"})
        scoped.rebuild_auth(redirected, response)

        assert ("api-key" not in redirected.headers) is should_strip

    def test_cross_host_redirect_drops_custom_header(
        self, redirect_server, tmp_path: Path
    ) -> None:
        """A credential header must NOT reach a redirect target on a different host.

        The request starts at ``127.0.0.1`` and is redirected to ``localhost``
        (a distinct hostname), so the caller's ``api-key`` — and requests' own
        ``Authorization`` — must be stripped before the ``/payload`` fetch.
        """
        port, server = redirect_server
        server.redirect_target = f"http://localhost:{port}/payload"  # type: ignore[attr-defined]

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)

        result = robust_download(
            f"http://127.0.0.1:{port}/start",
            out,
            verify_size=True,
            max_retries=0,
            extra_headers={"api-key": _SECRET, "Authorization": f"Bearer {_SECRET}"},
        )

        assert result == out
        assert out.read_bytes() == _PAYLOAD  # redirect was followed and downloaded

        delivered = _payload_request(server)
        assert delivered["api_key"] is None, (
            f"custom credential header leaked cross-host: {delivered}"
        )
        assert delivered["authorization"] is None, (
            f"Authorization leaked cross-host: {delivered}"
        )

    def test_same_host_redirect_keeps_custom_header(
        self, redirect_server, tmp_path: Path
    ) -> None:
        """A same-host redirect must PRESERVE the caller's credential header.

        Both hops are on ``127.0.0.1``, so scoping must not over-strip: the
        ``api-key`` is legitimately needed to authenticate the ``/payload`` fetch.
        """
        port, server = redirect_server
        server.redirect_target = f"http://127.0.0.1:{port}/payload"  # type: ignore[attr-defined]

        out = tmp_path / "cache" / "data.bin"
        out.parent.mkdir(parents=True)

        result = robust_download(
            f"http://127.0.0.1:{port}/start",
            out,
            verify_size=True,
            max_retries=0,
            extra_headers={"api-key": _SECRET},
        )

        assert result == out
        assert out.read_bytes() == _PAYLOAD

        delivered = _payload_request(server)
        assert delivered["api_key"] == _SECRET, (
            f"same-host credential header was wrongly dropped: {delivered}"
        )

    def test_plain_session_leaks_custom_header_cross_host(self) -> None:
        """Documents the vulnerability the fix defends against.

        A stock ``requests.Session`` strips ``Authorization`` on a cross-host
        redirect but leaves an arbitrary custom header in place. This anchors the
        control: without the host-scoped session the ``api-key`` above would
        reach the redirect target.
        """
        original = requests.Request(
            "GET", "http://example.com/start", headers={"api-key": _SECRET}
        ).prepare()
        response = requests.Response()
        response.request = original

        redirected = requests.Request(
            "GET", "http://evil.example.net/payload", headers={"api-key": _SECRET}
        ).prepare()

        plain = requests.Session()
        plain.trust_env = False  # no netrc surprises
        plain.rebuild_auth(redirected, response)
        assert redirected.headers.get("api-key") == _SECRET, (
            "stock requests should leak the custom header (documents the bug)"
        )

        scoped = _make_host_scoped_session("http://example.com/start", {"api-key"})
        scoped.trust_env = False
        scoped_req = requests.Request(
            "GET", "http://evil.example.net/payload", headers={"api-key": _SECRET}
        ).prepare()
        scoped.rebuild_auth(scoped_req, response)
        assert "api-key" not in scoped_req.headers, (
            "host-scoped session must strip the custom header cross-host"
        )
